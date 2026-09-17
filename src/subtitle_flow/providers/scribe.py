"""ElevenLabs Scribe v2 speech-to-text adapter.

The adapter is synchronous and mirrors the Phase 2A ``SpeechToTextProvider``
contract exactly. It performs no import-time work: ``httpx`` is resolved lazily
and credentials are supplied only through an injected resolver, never read from
the environment or logged. Every attempt reserves its upper bound in the durable
job ledger before the request is sent and archives the exact response bytes
before any parsing, so a received-but-invalid response is never silently
re-billed and no raw body is ever invented.

Endpoint and request fields follow the official Create-transcript reference:
``POST {base_url}/v1/speech-to-text`` with ``xi-api-key``, ``model_id=scribe_v2``,
``timestamps_granularity=word``, ``diarize=true`` and ``tag_audio_events=false``.
Spacing/word token text is preserved literally by
:func:`subtitle_flow.segmentation.segment_scribe_tokens`; no spaces are
invented and no token is dropped.

Before any paid dispatch the pipeline calls :meth:`preflight_route` (local
configuration/argument validation with no credential resolution and no HTTP) and
then :meth:`resolve_credentials`. A missing project, resolver or blank token in
the whole selected route therefore fails before the first Scribe request. A 3xx
response is archived and stops without a resend, credentials are never
forwarded, and the language/confidence gates below prevent paid MT on an
unknown or low-confidence transcript.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from subtitle_flow.config import (
    SCRIBE_MAX_AUDIO_BYTES,
    SCRIBE_MAX_KEYTERMS,
    SCRIBE_MAX_KEYTERM_LENGTH,
    SCRIBE_MAX_KEYTERM_WORDS,
    SCRIBE_MIN_AUDIO_DURATION_MS,
    SCRIBE_OFFICIAL_BASE_URL,
    ApiSettings,
)
from subtitle_flow.languages import normalize_language
from subtitle_flow.providers.api_common import (
    ApiBindingError,
    ApiContext,
    AttemptOutcomeLabel,
    budget_preflight,
    build_client,
    canonical_endpoint,
    classify_predispatch_transport_error,
    monotonic_ms,
    parse_retry_after,
    record_outcome,
    refused_error,
    release_reservation,
    require_httpx,
    reserve_attempt,
    safe_response_headers,
)
from subtitle_flow.providers.errors import ProviderCallError
from subtitle_flow.providers.stt_base import SpeechToTextProvider
from subtitle_flow.schemas import RawArtifactRef, StageError, StageKind, Transcript
from subtitle_flow.segmentation import (
    SUPPORTED_SEGMENTER_VERSIONS,
    SegmentationError,
    SegmentationNeedsReviewError,
    SegmenterSettings,
    segment_scribe_tokens,
)
from subtitle_flow.storage import utc_now

__all__ = ["ScribeV2STTProvider"]

#: Characters the provider forbids in a keyterm (the backslash is escaped in the
#: documentation snippet ``< > { } [ ] \``).
_KEYTERM_FORBIDDEN: frozenset[str] = frozenset("<>{}[]`\\")

_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "flac": "audio/flac",
}

_API_RUN_BINDING_ERROR = (
    "STT adapter is already bound to a different job run; a single adapter "
    "instance must not be shared across concurrent jobs"
)


class ScribeV2STTProvider(SpeechToTextProvider):
    """Concrete ElevenLabs Scribe v2 provider over an injected HTTP client."""

    provider_name: ClassVar[str] = "elevenlabs"
    model_name: ClassVar[str] = "scribe_v2"

    def __init__(
        self,
        *,
        api: ApiSettings,
        credential_resolver: Callable[[], str] | None = None,
        client: Any | None = None,
        client_factory: Callable[[Any, Any], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        # Re-validate at the trust boundary: ``model_copy`` bypasses validators.
        api = ApiSettings.model_validate(api.model_dump()) if isinstance(api, ApiSettings) else api
        if not isinstance(api, ApiSettings):
            raise TypeError("api must be an ApiSettings instance")
        self._api = api
        self._credential_resolver = credential_resolver
        self._client = client
        self._client_factory = client_factory
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self._jitter = jitter
        self._raw_sink: Callable[..., RawArtifactRef] | None = None
        self._context: ApiContext | None = None
        self._run_key: tuple[str, str] | None = None
        #: Guards the check-and-set of run/sink/context ownership so two threads
        #: sharing one adapter cannot both pass the owner check and overwrite a
        #: binding. A single adapter is still not meant to serve two jobs at once.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Pipeline binding
    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> ApiSettings:
        return self._api

    def bind_run(self, run_key: tuple[str, str]) -> None:
        """Claim this adapter for one job run; refuse a different overlapping job."""

        if not (isinstance(run_key, tuple) and len(run_key) == 2):
            raise TypeError("run_key must be a (root, job_id) tuple")
        with self._lock:
            if self._run_key is not None and self._run_key != run_key:
                raise ApiBindingError(_API_RUN_BINDING_ERROR)
            self._run_key = run_key

    def bind_raw_sink(self, sink: Callable[..., RawArtifactRef] | None) -> None:
        with self._lock:
            self._raw_sink = sink

    def clear_run(self, run_key: tuple[str, str]) -> None:
        """Clear the sink/context only when this run owns the binding."""

        with self._lock:
            if self._run_key is None or self._run_key != run_key:
                return
            self._run_key = None
            self._raw_sink = None
            self._context = None

    def bind_context(self, context: ApiContext | None) -> None:
        """Bind or clear the per-dispatch context; refuse overlapping jobs."""

        if context is None:
            with self._lock:
                self._context = None
            return
        if not isinstance(context, ApiContext):
            raise TypeError("context must be an ApiContext or None")
        key = (str(context.store.root), context.store.job_id)
        with self._lock:
            if self._run_key is not None and self._run_key != key:
                raise ApiBindingError(_API_RUN_BINDING_ERROR)
            self._run_key = key
            self._context = context

    # ------------------------------------------------------------------ #
    # Route preflight
    # ------------------------------------------------------------------ #
    def preflight_route(
        self, context: ApiContext, *, require_credential: bool = True
    ) -> None:
        """Validate the whole Scribe route without resolving credentials or HTTP.

        Called by the pipeline for the selected route *before* the first paid
        dispatch, so a broken model/endpoint/limit/paid policy is a typed,
        non-dispatched failure and never costs a call.

        ``require_credential`` is ``False`` for an STT stage that is already
        verified complete and will not dispatch (for example an MT-only resume):
        the stored config/audio snapshot is still validated locally, but a
        genuinely absent credential is tolerated because it will not be used.
        """

        self._require_context(context)
        if context.config.api != self._api:
            raise ApiBindingError(
                "adapter settings do not match the bound job snapshot; refusing "
                "to call with a different configuration"
            )
        if context.stage is not StageKind.stt or context.audio is None:
            raise ApiBindingError("STT adapter requires an STT context with audio")
        settings = self._api.scribe
        if settings.base_url != SCRIBE_OFFICIAL_BASE_URL:
            raise refused_error(
                "STT_ENDPOINT_INVALID",
                "Scribe base_url is not the canonical official endpoint",
            )
        if require_credential and self._credential_resolver is None:
            raise refused_error(
                "API_CREDENTIAL_MISSING",
                "no Scribe credential resolver was supplied; refusing to send",
            )
        # Bind the snapshot's model/segmenter identity and its effective
        # hint/keyterms locally too, so an invalid or mismatched route is refused
        # before any credential resolver or HTTP request.
        self._validate_bound_config(context)
        self._validate_keyterms(
            context.config.keyterms
            if context.config.stage_options.stt_send_keyterms
            else None
        )
        self._preflight_audio(context)

    def resolve_credentials(self) -> None:
        """Resolve and validate the credential, discarding it immediately.

        The token never leaves this method; it is never stored on the instance,
        written to config, the ledger or logs.
        """

        self._resolve_credential()

    # ------------------------------------------------------------------ #
    # Public contract
    # ------------------------------------------------------------------ #
    def transcribe(
        self,
        audio_path: str | Path,
        language_hint: str | None = None,
        keyterms: Sequence[str] | None = None,
    ) -> Transcript:
        context = self._require_context()
        self._require_sink()
        settings = self._api.scribe
        if context.config.api != self._api:
            raise ApiBindingError(
                "adapter settings do not match the bound job snapshot; refusing "
                "to call with a different configuration"
            )
        if context.stage is not StageKind.stt or context.audio is None:
            raise ApiBindingError("STT adapter requires an STT context with audio")

        hint = normalize_language(language_hint) if language_hint is not None else None
        if hint is not None and hint.canonical_code is None:
            raise refused_error(
                "STT_LANGUAGE_HINT_INVALID",
                "the language hint is not a supported pilot language; refusing to "
                "send a hint the provider cannot honor",
            )
        validated_keyterms = self._validate_keyterms(keyterms)
        self._validate_bound_arguments(language_hint, keyterms, context)

        if str(audio_path) != context.audio.path:
            raise ApiBindingError(
                "transcribe() audio path does not match the validated job audio"
            )
        payload = self._read_verified_audio(context)
        self._preflight_audio(context)

        # Paid gate and credential resolution happen only after local validation.
        budget_preflight(self._api.paid)
        token = self._resolve_credential()

        httpx = require_httpx()
        client = build_client(
            self._api.limits, client=self._client, client_factory=self._client_factory
        )
        self._client = client
        timeout = self._timeout(httpx)

        url = canonical_endpoint(settings.base_url, "/v1/speech-to-text")
        files = {
            "file": (
                context.audio.original_filename,
                payload,
                _CONTENT_TYPES.get(context.audio.container, "application/octet-stream"),
            )
        }
        data: dict[str, object] = {
            "model_id": settings.model_id,
            "timestamps_granularity": settings.timestamps_granularity,
            "diarize": "true" if settings.diarize else "false",
            "tag_audio_events": "true" if settings.tag_audio_events else "false",
        }
        if hint is not None and hint.canonical_code is not None:
            data["language_code"] = hint.canonical_code
        if validated_keyterms:
            data["keyterms"] = list(validated_keyterms)
        params = {"enable_logging": "true" if settings.enable_logging else "false"}
        headers = {"xi-api-key": token, "Accept": "application/json"}

        return self._dispatch(
            client=client,
            httpx=httpx,
            timeout=timeout,
            context=context,
            hint=hint,
            url=url,
            params=params,
            data=data,
            files=files,
            headers=headers,
        )

    # ------------------------------------------------------------------ #
    # Dispatch with durable budget + bounded retries
    # ------------------------------------------------------------------ #
    def _dispatch(
        self,
        *,
        client: Any,
        httpx: Any,
        timeout: Any,
        context: ApiContext,
        hint: Any,
        url: str,
        params: Mapping[str, str],
        data: Mapping[str, object],
        files: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> Transcript:
        limits = self._api.limits
        max_attempts = limits.max_attempts
        sink = self._require_sink()
        attempt = 0
        while True:
            attempt += 1
            attempt_id, amount = reserve_attempt(
                context.store,
                self._api.paid,
                stage=StageKind.stt,
                group_id=None,
                pipeline_attempt=context.pipeline_attempt,
                http_attempt=attempt,
                provider=self.provider_name,
                model=self.model_name,
                now=self._now,
            )
            started = self._monotonic()
            try:
                response = client.post(
                    url,
                    params=params,
                    data=data,
                    files=files,
                    headers=headers,
                    follow_redirects=False,
                    timeout=timeout,
                )
            except Exception as exc:  # network / transport failure
                pre_dispatch = classify_predispatch_transport_error(exc, httpx)
                elapsed = monotonic_ms(started, self._monotonic)
                if pre_dispatch:
                    record_outcome(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.stt,
                        group_id=None,
                        pipeline_attempt=context.pipeline_attempt,
                        http_attempt=attempt,
                        provider=self.provider_name,
                        model=self.model_name,
                        outcome=AttemptOutcomeLabel.predispatch_error,
                        error_code="STT_CONNECT_ERROR",
                        error_message="connection failed before the request was sent",
                        elapsed_ms=elapsed,
                        now=self._now,
                    )
                    release_reservation(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.stt,
                        group_id=None,
                        pipeline_attempt=context.pipeline_attempt,
                        http_attempt=attempt,
                        provider=self.provider_name,
                        model=self.model_name,
                        amount=amount,
                        currency=self._api.paid.currency or "",
                        reason="proven pre-dispatch connect failure",
                        now=self._now,
                    )
                    if attempt < max_attempts:
                        self._sleep(self._backoff_seconds(attempt))
                        continue
                    raise ProviderCallError(
                        StageError(
                            code="STT_CONNECT_FAILED",
                            message="could not connect to Scribe before sending",
                            retryable=True,
                        ),
                        dispatched=False,
                        cause=exc,
                    )
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.stt,
                    group_id=None,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.remote_unknown,
                    remote_status_unknown=True,
                    error_code="STT_REMOTE_STATUS_UNKNOWN",
                    error_message=f"transport failure after dispatch: {type(exc).__name__}",
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                raise ProviderCallError(
                    StageError(
                        code="STT_REMOTE_STATUS_UNKNOWN",
                        message="Scribe response was not received; remote status "
                        "is unknown and replay is refused",
                        remote_status_unknown=True,
                    ),
                    dispatched=True,
                    cause=exc,
                ) from exc

            content = response.content
            elapsed = monotonic_ms(started, self._monotonic)
            status = int(response.status_code)
            header_request_id = self._header_request_id(response)
            response_headers = safe_response_headers(response)
            raw_reference = sink(
                StageKind.stt,
                content,
                request_id=header_request_id,
                content_subtype="json",
            )
            body_request_id = (
                self._body_request_id(content) if header_request_id is None else None
            )
            request_id = header_request_id or body_request_id

            if status == 200:
                try:
                    transcript = self._build_transcript(
                        content=content,
                        context=context,
                        hint=hint,
                        raw_reference=raw_reference,
                        request_id=request_id,
                    )
                except ProviderCallError as exc:
                    record_outcome(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.stt,
                        group_id=None,
                        pipeline_attempt=context.pipeline_attempt,
                        http_attempt=attempt,
                        provider=self.provider_name,
                        model=self.model_name,
                        outcome=AttemptOutcomeLabel.invalid_response,
                        http_status=status,
                        request_id=request_id,
                        raw_reference=raw_reference,
                        response_headers=response_headers,
                        error_code=exc.error.code,
                        error_message=self._fixed_error_message(exc.error.code),
                        elapsed_ms=elapsed,
                        now=self._now,
                    )
                    raise
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    error = StageError(
                        code="STT_RESPONSE_INVALID",
                        message="Scribe HTTP 200 body could not be normalized",
                    )
                    record_outcome(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.stt,
                        group_id=None,
                        pipeline_attempt=context.pipeline_attempt,
                        http_attempt=attempt,
                        provider=self.provider_name,
                        model=self.model_name,
                        outcome=AttemptOutcomeLabel.invalid_response,
                        http_status=status,
                        request_id=request_id,
                        raw_reference=raw_reference,
                        response_headers=response_headers,
                        error_code=error.code,
                        error_message=error.message,
                        elapsed_ms=elapsed,
                        now=self._now,
                    )
                    raise ProviderCallError(error, dispatched=True, cause=exc) from exc
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.stt,
                    group_id=None,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.complete,
                    http_status=status,
                    request_id=request_id,
                    raw_reference=raw_reference,
                    response_headers=response_headers,
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                return transcript

            if 300 <= status < 400:
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.stt,
                    group_id=None,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.http_error,
                    http_status=status,
                    request_id=request_id,
                    raw_reference=raw_reference,
                    response_headers=response_headers,
                    error_code="STT_REDIRECT_REFUSED",
                    error_message="Scribe returned a redirect; it was archived and "
                    "never followed",
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                raise ProviderCallError(
                    StageError(
                        code="STT_REDIRECT_REFUSED",
                        message="Scribe returned a redirect; refusing to follow it "
                        "or forward credentials",
                    ),
                    dispatched=True,
                )

            if status == 429:
                retry_after = parse_retry_after(
                    response.headers.get("retry-after"), now=self._now()
                )
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.stt,
                    group_id=None,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.http_error,
                    http_status=status,
                    request_id=request_id,
                    raw_reference=raw_reference,
                    response_headers=response_headers,
                    retry_after_seconds=retry_after,
                    error_code="STT_RATE_LIMITED",
                    error_message="Scribe rate/concurrency rejection",
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                if attempt >= max_attempts:
                    raise ProviderCallError(
                        StageError(
                            code="STT_RATE_LIMITED",
                            message="Scribe rejected the request after the maximum "
                            "number of attempts",
                            retryable=True,
                        ),
                        dispatched=True,
                    )
                if (
                    retry_after is not None
                    and retry_after > limits.retry_after_ceiling_seconds
                ):
                    raise ProviderCallError(
                        StageError(
                            code="STT_RETRY_AFTER_TOO_LONG",
                            message="Scribe Retry-After exceeds the configured wait "
                            "ceiling; stopping as retryable",
                            retryable=True,
                        ),
                        dispatched=True,
                    )
                self._sleep(
                    retry_after if retry_after is not None else self._backoff_seconds(attempt)
                )
                continue

            if status == 202 or status >= 500:
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.stt,
                    group_id=None,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.remote_unknown,
                    http_status=status,
                    request_id=request_id,
                    raw_reference=raw_reference,
                    response_headers=response_headers,
                    remote_status_unknown=True,
                    error_code="STT_REMOTE_STATUS_UNKNOWN",
                    error_message=f"Scribe returned HTTP {status} after the request "
                    "was sent",
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                raise ProviderCallError(
                    StageError(
                        code="STT_REMOTE_STATUS_UNKNOWN",
                        message=f"Scribe returned HTTP {status} after dispatch; not "
                        "resending automatically",
                        remote_status_unknown=True,
                    ),
                    dispatched=True,
                )

            record_outcome(
                context.store,
                attempt_id=attempt_id,
                stage=StageKind.stt,
                group_id=None,
                pipeline_attempt=context.pipeline_attempt,
                http_attempt=attempt,
                provider=self.provider_name,
                model=self.model_name,
                outcome=AttemptOutcomeLabel.http_error,
                http_status=status,
                request_id=request_id,
                raw_reference=raw_reference,
                response_headers=response_headers,
                error_code=f"STT_HTTP_{status}",
                error_message=self._fixed_error_message(f"STT_HTTP_{status}"),
                elapsed_ms=elapsed,
                now=self._now,
            )
            raise ProviderCallError(
                StageError(
                    code=f"STT_HTTP_{status}",
                    message=f"Scribe rejected the request with HTTP {status}",
                    retryable=False,
                ),
                dispatched=True,
            )

    # ------------------------------------------------------------------ #
    # Response normalization
    # ------------------------------------------------------------------ #
    def _build_transcript(
        self,
        *,
        content: bytes,
        context: ApiContext,
        hint: Any,
        raw_reference: RawArtifactRef,
        request_id: str | None,
    ) -> Transcript:
        try:
            payload = json.loads(content)
        except ValueError as exc:
            raise ProviderCallError(
                StageError(
                    code="STT_RESPONSE_INVALID",
                    message="Scribe returned HTTP 200 with a body that is not JSON",
                ),
                dispatched=True,
                cause=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderCallError(
                StageError(
                    code="STT_RESPONSE_INVALID",
                    message="Scribe returned an unexpected top-level JSON value",
                ),
                dispatched=True,
            )
        words = payload.get("words")
        text = payload.get("text")
        if not isinstance(words, list) or not isinstance(text, str):
            raise ProviderCallError(
                StageError(
                    code="STT_RESPONSE_INVALID",
                    message="Scribe response is missing the 'words' list or 'text' "
                    "string",
                ),
                dispatched=True,
            )
        if "transcripts" in payload:
            raise ProviderCallError(
                StageError(
                    code="STT_MULTICHANNEL_UNSUPPORTED",
                    message="multi-channel Scribe responses are outside this phase",
                ),
                dispatched=True,
            )

        raw_code = payload.get("language_code")
        provider_code = raw_code if isinstance(raw_code, str) else None
        provider_norm = normalize_language(provider_code)
        canonical = provider_norm.canonical_code
        has_probability = "language_probability" in payload
        probability = self._validated_probability(payload.get("language_probability"))

        has_words = self._has_spoken_words(words)

        if not has_words and text.strip() != "":
            raise ProviderCallError(
                StageError(
                    code="STT_SEGMENTATION_INVALID",
                    message="Scribe returned non-empty text without usable word "
                    "tokens; refusing to report a false success",
                ),
                dispatched=True,
            )

        if not has_words:
            if canonical is not None and not provider_norm.uncertain:
                language = canonical
                language_uncertain = False
            else:
                language = None
                language_uncertain = True
            return self._empty_transcript(
                context=context,
                language=language,
                language_uncertain=language_uncertain,
                provider_code=provider_code,
                probability=probability if has_probability else None,
                raw_reference=raw_reference,
                request_id=request_id,
                provider_full_text=text,
            )

        # Non-empty speech: the provider language must be known and supported.
        if canonical is None or provider_norm.uncertain:
            raise ProviderCallError(
                StageError(
                    code="STT_LANGUAGE_UNSUPPORTED",
                    message="Scribe returned an unknown or unsupported language for "
                    "non-empty speech; the raw response is preserved and the job "
                    "needs review",
                ),
                dispatched=True,
            )
        if hint is not None and hint.canonical_code is not None and hint.canonical_code != canonical:
            raise ProviderCallError(
                StageError(
                    code="STT_LANGUAGE_COLLISION",
                    message="Scribe language collides with the explicit language "
                    "hint; the raw response is preserved and no MT will run",
                ),
                dispatched=True,
            )
        settings: SegmenterSettings = context.config.segmenter
        # The confidence floor lives on ScribeSettings; read it through the job
        # snapshot so a tampered adapter instance cannot lower it.
        minimum = self._api.scribe.min_language_probability
        if not has_probability:
            raise ProviderCallError(
                StageError(
                    code="STT_PROBABILITY_MISSING",
                    message="Scribe returned non-empty speech without a language "
                    "confidence; treating it as uncertain and requiring review",
                ),
                dispatched=True,
            )
        assert probability is not None
        if probability < minimum:
            raise ProviderCallError(
                StageError(
                    code="STT_LANGUAGE_LOW_CONFIDENCE",
                    message="Scribe language confidence is below the configured "
                    "minimum; the raw response is preserved and no MT will run",
                ),
                dispatched=True,
            )

        assert context.audio is not None
        selected_version = context.config.segmenter_version
        try:
            result = segment_scribe_tokens(
                tokens=words,
                full_text=text,
                source_language=canonical,
                audio_duration_ms=context.audio.duration_ms,
                settings=settings,
                algorithm_version=selected_version,
            )
        except SegmentationNeedsReviewError as exc:
            # The raw response is already archived. A v3 zero-only point timing
            # with no adjacent same-speaker span has no honest duration, so the
            # job is routed to human review instead of inventing one.
            raise ProviderCallError(
                StageError(
                    code="STT_SEGMENTATION_NEEDS_REVIEW",
                    message=(
                        "Scribe returned zero-duration point timings that cannot "
                        "borrow a real span from adjacent same-speaker text; the "
                        "raw response is preserved and the job needs review"
                    ),
                ),
                dispatched=True,
                cause=exc,
            ) from exc
        except (SegmentationError, TypeError) as exc:
            raise ProviderCallError(
                StageError(
                    code="STT_SEGMENTATION_INVALID",
                    message=f"Scribe word tokens cannot be segmented: {exc}",
                ),
                dispatched=True,
                cause=exc,
            ) from exc

        metadata: dict[str, Any] = dict(result.metadata)
        metadata.update(
            {
                "provider_language_code": provider_code,
                "language_probability": probability,
                "request_id": request_id,
                "model_id": self._api.scribe.model_id,
                "segmenter_version": selected_version,
                "segmenter_settings": settings.model_dump(mode="json"),
                "diarize": self._api.scribe.diarize,
                "tag_audio_events": self._api.scribe.tag_audio_events,
                "enable_logging": self._api.scribe.enable_logging,
                "min_language_probability": minimum,
            }
        )
        for key in ("transcription_id", "audio_duration_secs"):
            value = payload.get(key)
            if value is None or isinstance(value, (str, int, float, bool)):
                metadata[key] = value

        return Transcript(
            provider=self.provider_name,
            model=self.model_name,
            source_language=canonical,
            provider_language_code=provider_code,
            language_uncertain=False,
            audio_duration_ms=context.audio.duration_ms,
            request_id=request_id,
            segments=result.segments,
            raw_reference=raw_reference,
            request_metadata=metadata,
        )

    def _empty_transcript(
        self,
        *,
        context: ApiContext,
        language: str | None,
        language_uncertain: bool,
        provider_code: str | None,
        probability: float | None,
        raw_reference: RawArtifactRef,
        request_id: str | None,
        provider_full_text: str,
    ) -> Transcript:
        assert context.audio is not None
        metadata: dict[str, Any] = {
            "provider_language_code": provider_code,
            "language_probability": probability,
            "request_id": request_id,
            "model_id": self._api.scribe.model_id,
            "segmenter_version": context.config.segmenter_version,
            "no_speech": True,
            "provider_full_text": provider_full_text,
        }
        return Transcript(
            provider=self.provider_name,
            model=self.model_name,
            source_language=language,
            provider_language_code=provider_code if isinstance(provider_code, str) else None,
            language_uncertain=language_uncertain or language is None,
            audio_duration_ms=context.audio.duration_ms,
            request_id=request_id,
            segments=(),
            raw_reference=raw_reference,
            request_metadata=metadata,
        )

    @staticmethod
    def _validated_probability(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderCallError(
                StageError(
                    code="STT_PROBABILITY_INVALID",
                    message="Scribe language_probability is not a finite number",
                ),
                dispatched=True,
            )
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")) or not (
            0.0 <= number <= 1.0
        ):
            raise ProviderCallError(
                StageError(
                    code="STT_PROBABILITY_INVALID",
                    message="Scribe language_probability is outside the valid 0..1 range",
                ),
                dispatched=True,
            )
        return number

    @staticmethod
    def _has_spoken_words(words: Sequence[Any]) -> bool:
        for token in words:
            if not isinstance(token, Mapping):
                continue
            if token.get("type") == "word":
                text = token.get("text")
                if isinstance(text, str) and text.strip() != "":
                    return True
        return False

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _require_context(self, context: ApiContext | None = None) -> ApiContext:
        with self._lock:
            bound = self._context if context is None else context
        if bound is None:
            raise ApiBindingError(
                "STT adapter has no bound pipeline context; direct calls are "
                "refused because validated audio, raw sink and budget are required"
            )
        return bound

    def _require_sink(self) -> Callable[..., RawArtifactRef]:
        with self._lock:
            sink = self._raw_sink
        if sink is None:
            raise ApiBindingError(
                "STT adapter has no raw sink bound; refusing to call without "
                "durable raw archiving"
            )
        return sink

    def _resolve_credential(self) -> str:
        if self._credential_resolver is None:
            raise refused_error(
                "API_CREDENTIAL_MISSING",
                "no Scribe credential resolver was supplied; refusing to send",
            )
        token = self._credential_resolver()
        if not isinstance(token, str) or not token.strip():
            raise refused_error(
                "API_CREDENTIAL_MISSING",
                "the Scribe credential resolver returned no usable key",
            )
        return token

    def _validate_bound_config(self, context: ApiContext) -> None:
        """Enforce the adapter's model and segmenter identity against the snapshot."""

        config = context.config
        if (
            config.stt.provider != self.provider_name
            or config.stt.model != self.model_name
        ):
            raise ApiBindingError(
                "STT adapter identity does not match the bound job snapshot; "
                "refusing to call with a different provider/model"
            )
        if config.segmenter_version not in SUPPORTED_SEGMENTER_VERSIONS:
            raise refused_error(
                "STT_SEGMENTER_VERSION_MISMATCH",
                "the bound segmenter version is not a version this build can "
                "reproduce; refusing to segment a transcript with an unknown "
                "algorithm",
            )

    def _validate_bound_arguments(
        self,
        language_hint: str | None,
        keyterms: Sequence[str] | None,
        context: ApiContext,
    ) -> None:
        """Refuse a direct call whose hint/keyterms disagree with the snapshot.

        Runs before any credential resolution, reservation or HTTP, so a caller
        holding a valid bound context cannot send a language hint or keyterm set
        that differs from the stored job snapshot/fingerprint. The *effective*
        keyterm set respects ``stage_options.stt_send_keyterms``: when keyterms
        are not sent, the only accepted argument is ``None``/empty.
        """

        self._validate_bound_config(context)
        if language_hint is None:
            passed_hint: str | None = None
        else:
            normalized = normalize_language(language_hint)
            passed_hint = normalized.canonical_code
        if passed_hint != context.config.source_language_hint:
            raise refused_error(
                "STT_ARGUMENT_MISMATCH",
                "the requested language hint does not match the bound job snapshot",
            )
        expected = (
            tuple(context.config.keyterms)
            if context.config.stage_options.stt_send_keyterms
            else ()
        )
        provided = tuple(keyterms) if keyterms is not None else ()
        if provided != expected:
            raise refused_error(
                "STT_ARGUMENT_MISMATCH",
                "the requested keyterms do not exactly match the bound job snapshot",
            )

    def _validate_keyterms(
        self, keyterms: Sequence[str] | None
    ) -> tuple[str, ...]:
        if keyterms is None:
            return ()
        settings = self._api.scribe
        terms = tuple(keyterms)
        if not terms:
            return ()
        # Official hard limits are enforced independently of the (possibly
        # tightened) configured limits.
        if len(terms) > min(settings.max_keyterms, SCRIBE_MAX_KEYTERMS):
            raise refused_error(
                "STT_KEYTERMS_INVALID",
                f"too many keyterms ({len(terms)} > "
                f"{min(settings.max_keyterms, SCRIBE_MAX_KEYTERMS)})",
            )
        for term in terms:
            if not isinstance(term, str) or term == "":
                raise refused_error(
                    "STT_KEYTERMS_INVALID", "keyterms must be non-empty strings"
                )
            if len(term) >= min(settings.max_keyterm_length, SCRIBE_MAX_KEYTERM_LENGTH):
                raise refused_error(
                    "STT_KEYTERMS_INVALID",
                    "a keyterm is too long for the provider limit",
                )
            if len(term.split()) > min(settings.max_keyterm_words, SCRIBE_MAX_KEYTERM_WORDS):
                raise refused_error(
                    "STT_KEYTERMS_INVALID", "a keyterm has too many words"
                )
            if any(character in _KEYTERM_FORBIDDEN for character in term):
                raise refused_error(
                    "STT_KEYTERMS_INVALID", "a keyterm contains a forbidden character"
                )
        return terms

    def _read_verified_audio(self, context: ApiContext) -> bytes:
        assert context.audio is not None
        try:
            data = Path(context.audio.path).read_bytes()
        except OSError as exc:
            raise refused_error(
                "STT_AUDIO_UNREADABLE", "validated audio is no longer readable"
            ) from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != context.audio.audio_sha256 or len(data) != context.audio.size_bytes:
            raise refused_error(
                "STT_AUDIO_CHANGED",
                "audio bytes changed since validation; refusing to send",
            )
        return data

    def _preflight_audio(self, context: ApiContext) -> None:
        assert context.audio is not None
        settings = self._api.scribe
        effective_min = max(
            settings.min_audio_duration_ms, SCRIBE_MIN_AUDIO_DURATION_MS + 1
        )
        if context.audio.duration_ms < effective_min:
            raise refused_error(
                "STT_AUDIO_TOO_SHORT",
                f"audio is shorter than the Scribe minimum of "
                f"{SCRIBE_MIN_AUDIO_DURATION_MS} ms",
            )
        effective_max = min(settings.max_audio_bytes, SCRIBE_MAX_AUDIO_BYTES)
        if context.audio.size_bytes >= effective_max:
            raise refused_error(
                "STT_AUDIO_TOO_LARGE",
                "audio reaches or exceeds the configured/offical Scribe upload "
                "byte limit",
            )

    def _timeout(self, httpx: Any) -> Any:
        limits = self._api.limits
        return httpx.Timeout(
            connect=limits.connect_timeout_seconds,
            read=limits.read_timeout_seconds,
            write=limits.write_timeout_seconds,
            pool=limits.pool_timeout_seconds,
        )

    def _backoff_seconds(self, attempt: int) -> float:
        limits = self._api.limits
        raw = limits.backoff_base_seconds * (2 ** (attempt - 1))
        capped = min(raw, limits.backoff_max_seconds)
        return capped * (0.5 + 0.5 * self._jitter())

    @staticmethod
    def _header_request_id(response: Any) -> str | None:
        for name in ("request-id", "x-trace-id", "x-request-id"):
            value = response.headers.get(name)
            if value:
                return str(value)
        return None

    @staticmethod
    def _body_request_id(content: bytes) -> str | None:
        try:
            payload = json.loads(content)
        except ValueError:
            return None
        if isinstance(payload, dict):
            direct = payload.get("request_id")
            if isinstance(direct, str) and direct:
                return direct
            detail = payload.get("detail")
            if isinstance(detail, dict):
                nested = detail.get("request_id")
                if isinstance(nested, str) and nested:
                    return nested
        return None

    @staticmethod
    def _fixed_error_message(code: str) -> str:
        """Return a fixed, non-echoing summary for a provider error code."""

        summaries = {
            "STT_LANGUAGE_UNSUPPORTED": "unknown provider language",
            "STT_LANGUAGE_COLLISION": "provider language collides with the hint",
            "STT_PROBABILITY_INVALID": "invalid language probability",
            "STT_PROBABILITY_MISSING": "missing language probability",
            "STT_LANGUAGE_LOW_CONFIDENCE": "low language confidence",
            "STT_SEGMENTATION_INVALID": "invalid provider word tokens",
            "STT_SEGMENTATION_NEEDS_REVIEW": "uncertain point timings need review",
            "STT_RESPONSE_INVALID": "invalid provider response",
        }
        return summaries.get(code, "provider error body archived")
