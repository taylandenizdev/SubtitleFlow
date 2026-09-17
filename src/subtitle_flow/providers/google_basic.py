"""Google Cloud Translation **Basic v2** (API key) adapter.

This is the product's single Turkish MT transport. Basic v2 serves the standard
Translation LLM through ``POST /language/translate/v2`` with API-key
authentication:

* the API key is presented only through the ``X-goog-api-key`` header -- never a
  URL query parameter, never logged, never persisted and never echoed in a
  message, repr or exception string;
* the ordered source texts are sent as ``q`` together with ``source``, ``target``
  and ``format=text``, and the request ``model`` is the full canonical
  ``projects/{PROJECT}/locations/global/models/general/translation-llm`` resource;
* the response is parsed from ``data.translations[].translatedText`` with an exact
  count/order/type check and byte-exact entity preservation; the Basic v2
  ``detectedSourceLanguage`` field is recorded.

Basic v2 may omit the returned model. When it does, the requested canonical
resource stays in the stored identity and the response honestly records
``model_reported=false`` with no returned model; nothing is fabricated. When a
model *is* reported it must name the requested project (or its documented numeric
normalisation), the ``global`` location and the ``general/translation-llm``
family, otherwise the response is refused as drift. Basic v2 may report only the
exact short model name ``translation-llm`` instead of the family form
``general/translation-llm``; both exact forms are accepted as equivalent for this
fixed route and the reported string is preserved verbatim alongside the
separately stored requested resource.

Like the other paid adapters it resolves ``httpx`` lazily, reserves every HTTP
attempt in the durable ledger before sending, archives the raw response body
before parsing, retries only a proven pre-dispatch failure or a documented rate
limit, and never auto-resends a post-dispatch timeout/5xx/unknown outcome.
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, ClassVar

from subtitle_flow.config import (
    GOOGLE_BASIC_OFFICIAL_BASE_URL,
    GOOGLE_BASIC_OFFICIAL_PATH,
    GOOGLE_BASIC_PROVIDER,
    ApiSettings,
    GoogleBasicSettings,
)
from subtitle_flow.languages import TARGET_LANGUAGE_CODE
from subtitle_flow.providers.api_common import (
    ApiBindingError,
    ApiContext,
    AttemptOutcomeLabel,
    ExpectedSegment,
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
from subtitle_flow.providers.mt_base import MachineTranslationProvider
from subtitle_flow.schemas import (
    RawArtifactRef,
    Segment,
    StageError,
    StageKind,
    TranslatedSegment,
    Translation,
    TranslationStatus,
)
from subtitle_flow.storage import utc_now

__all__ = ["GoogleTranslationBasicProvider"]

_MODEL_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/models/(?P<family>.+)$"
)


#: Basic v2 may report only the final model segment. This is derived from the
#: configured family by exact string equality, never by suffix/substring
#: matching, so only ``translation-llm`` is accepted for ``general/translation-llm``.
def _short_model_name(family: str) -> str:
    return family.rsplit("/", 1)[-1]


_API_RUN_BINDING_ERROR = (
    "MT adapter is already bound to a different job run; a single adapter "
    "instance must not be shared across concurrent jobs"
)


class GoogleTranslationBasicProvider(MachineTranslationProvider):
    """Concrete Google Translation Basic v2 provider over an injected client."""

    provider_name: ClassVar[str] = GOOGLE_BASIC_PROVIDER
    model_name: ClassVar[str] = "general/translation-llm"

    def __init__(
        self,
        *,
        api: ApiSettings,
        basic: GoogleBasicSettings,
        credential_resolver: Callable[[], str] | None = None,
        client: Any | None = None,
        client_factory: Callable[[Any, Any], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        api = ApiSettings.model_validate(api.model_dump()) if isinstance(api, ApiSettings) else api
        if not isinstance(api, ApiSettings):
            raise TypeError("api must be an ApiSettings instance")
        basic = (
            GoogleBasicSettings.model_validate(basic.model_dump())
            if isinstance(basic, GoogleBasicSettings)
            else basic
        )
        if not isinstance(basic, GoogleBasicSettings):
            raise TypeError("basic must be a GoogleBasicSettings instance")
        self._api = api
        self._basic = basic
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
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Pipeline binding
    # ------------------------------------------------------------------ #
    @property
    def settings(self) -> GoogleBasicSettings:
        return self._basic

    @property
    def api(self) -> ApiSettings:
        return self._api

    def bind_run(self, run_key: tuple[str, str]) -> None:
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
        with self._lock:
            if self._run_key is None or self._run_key != run_key:
                return
            self._run_key = None
            self._raw_sink = None
            self._context = None

    def bind_context(self, context: ApiContext | None) -> None:
        """Bind or clear the per-group context; refuse overlapping jobs."""

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
    def preflight_route(self, context: ApiContext) -> None:
        """Validate the whole Basic route without the key or any HTTP."""

        self._require_context(context)
        if context.config.api != self._api or context.config.google_basic != self._basic:
            raise ApiBindingError(
                "adapter settings do not match the bound job snapshot; refusing "
                "to call with a different configuration"
            )
        if context.stage is not StageKind.mt:
            raise ApiBindingError("MT adapter requires an MT context")
        if self._basic.project is None:
            raise refused_error(
                "MT_PROJECT_MISSING",
                "Google Translation Basic v2 requires an explicit project id",
            )
        if self._basic.base_url != GOOGLE_BASIC_OFFICIAL_BASE_URL:
            raise refused_error(
                "MT_ENDPOINT_INVALID",
                "Google Basic base_url is not the canonical official endpoint",
            )
        if self._credential_resolver is None:
            raise refused_error(
                "API_CREDENTIAL_MISSING",
                "no Google Translation API key resolver was supplied; refusing "
                "to send",
            )

    def resolve_credentials(self) -> None:
        self._resolve_credential()

    # ------------------------------------------------------------------ #
    # Provider-specific translation
    # ------------------------------------------------------------------ #
    def _translate(
        self,
        segments: Sequence[Segment],
        *,
        source_language: str,
        target_language: str,
    ) -> Translation:
        context = self._require_context()
        self._require_sink()
        if context.config.api != self._api or context.config.google_basic != self._basic:
            raise ApiBindingError(
                "adapter settings do not match the bound job snapshot; refusing "
                "to call with a different configuration"
            )
        if context.stage is not StageKind.mt or context.group_id is None:
            raise ApiBindingError("MT adapter requires an MT group context")

        ordered = tuple(segments)
        self._validate_bound_arguments(ordered, source_language, target_language, context)

        contents = [
            segment.translation_input
            if segment.translation_input is not None
            else segment.source_text
            for segment in ordered
        ]
        self._preflight_group(contents, context)

        budget_preflight(self._api.paid)
        token = self._resolve_credential()

        httpx = require_httpx()
        client = build_client(
            self._api.limits, client=self._client, client_factory=self._client_factory
        )
        self._client = client
        timeout = self._timeout(httpx)

        url = canonical_endpoint(
            self._basic.base_url, GOOGLE_BASIC_OFFICIAL_PATH
        )
        body = {
            "q": contents,
            "source": source_language,
            "target": TARGET_LANGUAGE_CODE,
            "format": "text",
            "model": self._basic.model_resource,
        }
        headers = {
            "X-goog-api-key": token,
            "Content-Type": "application/json; charset=utf-8",
        }

        return self._dispatch(
            client=client,
            httpx=httpx,
            timeout=timeout,
            context=context,
            segments=ordered,
            contents=contents,
            source_language=source_language,
            url=url,
            body=body,
            headers=headers,
        )

    def _validate_bound_arguments(
        self,
        segments: Sequence[Segment],
        source_language: str,
        target_language: str,
        context: ApiContext,
    ) -> None:
        if context.source_language is not None and source_language != context.source_language:
            raise refused_error(
                "MT_ARGUMENT_MISMATCH",
                "requested source language does not match the bound job context",
            )
        if context.target_language is not None and target_language != context.target_language:
            raise refused_error(
                "MT_ARGUMENT_MISMATCH",
                "requested target language does not match the bound job context",
            )
        expected = context.expected_segments
        if len(expected) != len(segments):
            raise refused_error(
                "MT_ARGUMENT_MISMATCH",
                "segment count does not match the bound job context",
            )
        for bound, segment in zip(expected, segments):
            if not isinstance(bound, ExpectedSegment):
                raise refused_error(
                    "MT_ARGUMENT_MISMATCH", "bound context segment is invalid"
                )
            if (
                segment.segment_id != bound.segment_id
                or segment.source_text != bound.source_text
                or segment.translation_input != bound.translation_input
            ):
                raise refused_error(
                    "MT_ARGUMENT_MISMATCH",
                    "segment identity/text does not match the bound job context",
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
        segments: Sequence[Segment],
        contents: Sequence[str],
        source_language: str,
        url: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Translation:
        limits = self._api.limits
        max_attempts = limits.max_attempts
        sink = self._require_sink()
        attempt = 0
        while True:
            attempt += 1
            attempt_id, amount = reserve_attempt(
                context.store,
                self._api.paid,
                stage=StageKind.mt,
                group_id=context.group_id,
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
                    json=body,
                    headers=headers,
                    follow_redirects=False,
                    timeout=timeout,
                )
            except Exception as exc:
                pre_dispatch = classify_predispatch_transport_error(exc, httpx)
                elapsed = monotonic_ms(started, self._monotonic)
                if pre_dispatch:
                    record_outcome(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.mt,
                        group_id=context.group_id,
                        pipeline_attempt=context.pipeline_attempt,
                        http_attempt=attempt,
                        provider=self.provider_name,
                        model=self.model_name,
                        outcome=AttemptOutcomeLabel.predispatch_error,
                        error_code="MT_CONNECT_ERROR",
                        error_message="connection failed before the request was sent",
                        elapsed_ms=elapsed,
                        now=self._now,
                    )
                    release_reservation(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.mt,
                        group_id=context.group_id,
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
                            code="MT_CONNECT_FAILED",
                            message="could not connect to Google before sending",
                            retryable=True,
                        ),
                        dispatched=False,
                        cause=exc,
                    )
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.mt,
                    group_id=context.group_id,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.remote_unknown,
                    remote_status_unknown=True,
                    error_code="MT_REMOTE_STATUS_UNKNOWN",
                    error_message=f"transport failure after dispatch: {type(exc).__name__}",
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                raise ProviderCallError(
                    StageError(
                        code="MT_REMOTE_STATUS_UNKNOWN",
                        message="Google response was not received; remote status is "
                        "unknown and replay is refused",
                        remote_status_unknown=True,
                    ),
                    dispatched=True,
                    cause=exc,
                ) from exc

            content = response.content
            elapsed = monotonic_ms(started, self._monotonic)
            status = int(response.status_code)
            request_id = self._header_request_id(response)
            response_headers = safe_response_headers(response)
            raw_reference = sink(
                StageKind.mt,
                content,
                request_id=request_id,
                group_id=context.group_id,
                attempt=attempt,
                content_subtype="json",
            )

            if status == 200:
                try:
                    translation = self._build_translation(
                        content=content,
                        context=context,
                        segments=segments,
                        contents=contents,
                        source_language=source_language,
                        raw_reference=raw_reference,
                        request_id=request_id,
                    )
                except ProviderCallError as exc:
                    record_outcome(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.mt,
                        group_id=context.group_id,
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
                        code="MT_RESPONSE_INVALID",
                        message="Google HTTP 200 body could not be normalized",
                    )
                    record_outcome(
                        context.store,
                        attempt_id=attempt_id,
                        stage=StageKind.mt,
                        group_id=context.group_id,
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
                    stage=StageKind.mt,
                    group_id=context.group_id,
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
                return translation

            if 300 <= status < 400:
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.mt,
                    group_id=context.group_id,
                    pipeline_attempt=context.pipeline_attempt,
                    http_attempt=attempt,
                    provider=self.provider_name,
                    model=self.model_name,
                    outcome=AttemptOutcomeLabel.http_error,
                    http_status=status,
                    request_id=request_id,
                    raw_reference=raw_reference,
                    response_headers=response_headers,
                    error_code="MT_REDIRECT_REFUSED",
                    error_message="Google returned a redirect; it was archived and "
                    "never followed",
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                raise ProviderCallError(
                    StageError(
                        code="MT_REDIRECT_REFUSED",
                        message="Google returned a redirect; refusing to follow it "
                        "or forward the API key",
                    ),
                    dispatched=True,
                )

            retry_after = parse_retry_after(
                response.headers.get("retry-after"), now=self._now()
            )
            error_code, error_message = self._error_summary(content)
            rate_limited = status == 429 or (
                status == 403 and self._is_minute_rate(content)
            )
            if rate_limited:
                record_outcome(
                    context.store,
                    attempt_id=attempt_id,
                    stage=StageKind.mt,
                    group_id=context.group_id,
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
                    error_code=error_code or "MT_RATE_LIMITED",
                    error_message=error_message,
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                if attempt >= max_attempts:
                    raise ProviderCallError(
                        StageError(
                            code="MT_RATE_LIMITED",
                            message="Google rate rejection persisted after the "
                            "maximum number of attempts",
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
                            code="MT_RETRY_AFTER_TOO_LONG",
                            message="Google Retry-After exceeds the configured wait "
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
                    stage=StageKind.mt,
                    group_id=context.group_id,
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
                    error_code="MT_REMOTE_STATUS_UNKNOWN",
                    error_message=error_message,
                    elapsed_ms=elapsed,
                    now=self._now,
                )
                raise ProviderCallError(
                    StageError(
                        code="MT_REMOTE_STATUS_UNKNOWN",
                        message=f"Google returned HTTP {status} after dispatch; not "
                        "resending automatically",
                        remote_status_unknown=True,
                    ),
                    dispatched=True,
                )

            record_outcome(
                context.store,
                attempt_id=attempt_id,
                stage=StageKind.mt,
                group_id=context.group_id,
                pipeline_attempt=context.pipeline_attempt,
                http_attempt=attempt,
                provider=self.provider_name,
                model=self.model_name,
                outcome=AttemptOutcomeLabel.http_error,
                http_status=status,
                request_id=request_id,
                raw_reference=raw_reference,
                response_headers=response_headers,
                error_code=error_code or f"MT_HTTP_{status}",
                error_message=error_message,
                elapsed_ms=elapsed,
                now=self._now,
            )
            raise ProviderCallError(
                StageError(
                    code=error_code or f"MT_HTTP_{status}",
                    message=error_message
                    or f"Google rejected the request with HTTP {status}",
                    retryable=False,
                ),
                dispatched=True,
            )

    # ------------------------------------------------------------------ #
    # Response normalization
    # ------------------------------------------------------------------ #
    def _build_translation(
        self,
        *,
        content: bytes,
        context: ApiContext,
        segments: Sequence[Segment],
        contents: Sequence[str],
        source_language: str,
        raw_reference: RawArtifactRef,
        request_id: str | None,
    ) -> Translation:
        try:
            payload = json.loads(content)
        except ValueError as exc:
            raise ProviderCallError(
                StageError(
                    code="MT_RESPONSE_INVALID",
                    message="Google returned HTTP 200 with a body that is not JSON",
                ),
                dispatched=True,
                cause=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderCallError(
                StageError(
                    code="MT_RESPONSE_INVALID",
                    message="Google returned an unexpected top-level JSON value",
                ),
                dispatched=True,
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ProviderCallError(
                StageError(
                    code="MT_RESPONSE_INVALID",
                    message="Google Basic response is missing the 'data' object",
                ),
                dispatched=True,
            )
        translations = data.get("translations")
        if not isinstance(translations, list):
            raise ProviderCallError(
                StageError(
                    code="MT_RESPONSE_INVALID",
                    message="Google Basic response is missing the 'translations' list",
                ),
                dispatched=True,
            )
        if len(translations) != len(contents):
            raise ProviderCallError(
                StageError(
                    code="MT_COUNT_MISMATCH",
                    message=f"Google returned {len(translations)} translations for "
                    f"{len(contents)} contents; refusing to pair them",
                ),
                dispatched=True,
            )

        requested = self._basic.model_resource
        returned_models: list[str] = []
        detected_languages: list[str | None] = []
        texts: list[str] = []
        for index, item in enumerate(translations):
            if not isinstance(item, dict):
                raise ProviderCallError(
                    StageError(
                        code="MT_RESPONSE_INVALID",
                        message=f"translation item {index} is not an object",
                    ),
                    dispatched=True,
                )
            text = item.get("translatedText")
            if not isinstance(text, str):
                raise ProviderCallError(
                    StageError(
                        code="MT_RESPONSE_INVALID",
                        message=f"translation item {index} has no translatedText string",
                    ),
                    dispatched=True,
                )
            texts.append(text)
            model = item.get("model")
            if model is not None:
                if not isinstance(model, str) or model == "":
                    raise ProviderCallError(
                        StageError(
                            code="MT_RESPONSE_INVALID",
                            message=f"translation item {index} has a non-string model",
                        ),
                        dispatched=True,
                    )
                returned_models.append(model)
                self._validate_returned_model(model, requested)
            # Basic v2 documents the detected source language as
            # ``detectedSourceLanguage``; only that field is read here.
            detected = item.get("detectedSourceLanguage")
            detected_languages.append(
                detected if isinstance(detected, str) and detected else None
            )
        if len(set(returned_models)) > 1:
            raise ProviderCallError(
                StageError(
                    code="MT_MODEL_INCONSISTENT",
                    message="Google returned different models within one request",
                ),
                dispatched=True,
            )

        reported = bool(returned_models)
        metadata: dict[str, Any] = {
            "requested_model": requested,
            "model_reported": reported,
            "returned_model": returned_models[0] if reported else None,
            "returned_models": returned_models,
            "detected_language_codes": detected_languages,
            "source_language_code": source_language,
            "target_language_code": TARGET_LANGUAGE_CODE,
            "format": "text",
            "transport": "basic-v2",
            "request_id": request_id,
            "group_id": context.group_id,
        }
        translated_segments = tuple(
            TranslatedSegment(
                segment_id=segment.segment_id,
                source_text=segment.source_text,
                translation_input=segment.translation_input,
                translated_text_tr=texts[index],
            )
            for index, segment in enumerate(segments)
        )
        return Translation(
            provider=self.provider_name,
            model=self.model_name,
            status=TranslationStatus.translated,
            source_language=source_language,
            target_language=TARGET_LANGUAGE_CODE,
            segments=translated_segments,
            raw_reference=raw_reference,
            request_metadata=metadata,
        )

    def _validate_returned_model(self, model: str, requested: str) -> None:
        if model == self._basic.model_family:
            # Basic v2 may report the family form ``general/translation-llm``.
            return
        if model == _short_model_name(self._basic.model_family):
            # Basic v2 may report only the exact short model name
            # ``translation-llm`` for the requested standard LLM family.
            return
        match = _MODEL_RE.match(model)
        if match is None:
            raise ProviderCallError(
                StageError(
                    code="MT_MODEL_DRIFT",
                    message="Google returned a model resource outside the expected "
                    "global Translation LLM family",
                ),
                dispatched=True,
            )
        if (
            match.group("location") != self._basic.location
            or match.group("family") != self._basic.model_family
        ):
            raise ProviderCallError(
                StageError(
                    code="MT_MODEL_DRIFT",
                    message="Google returned a different location or model family "
                    "than requested",
                ),
                dispatched=True,
            )
        requested_match = _MODEL_RE.match(requested)
        assert requested_match is not None
        requested_project = requested_match.group("project")
        returned_project = match.group("project")
        if returned_project == requested_project:
            return
        if requested_project.isdigit():
            raise ProviderCallError(
                StageError(
                    code="MT_MODEL_DRIFT",
                    message="Google returned a different project than requested",
                ),
                dispatched=True,
            )
        if returned_project.isdigit():
            # Documented normalisation of a named project to its number.
            return
        raise ProviderCallError(
            StageError(
                code="MT_MODEL_DRIFT",
                message="Google returned a different non-numeric project than requested",
            ),
            dispatched=True,
        )

    def _preflight_group(
        self, contents: Sequence[str], context: ApiContext
    ) -> None:
        if self._basic.project is None:
            raise refused_error(
                "MT_PROJECT_MISSING",
                "Google Translation Basic v2 requires an explicit project id",
            )
        if not contents:
            raise refused_error("MT_EMPTY_GROUP", "cannot translate an empty group")
        max_items = self._basic.max_items_per_request
        max_codepoints = self._basic.max_codepoints_per_request
        if len(contents) > max_items:
            raise refused_error(
                "MT_GROUP_TOO_LARGE",
                f"group has {len(contents)} items, exceeding the conservative "
                f"{max_items} item Basic v2 limit",
            )
        for item in contents:
            if len(item) > max_codepoints:
                raise refused_error(
                    "MT_SEGMENT_TOO_LONG",
                    "a single segment exceeds the Basic v2 codepoint limit and "
                    "cannot be split",
                )
        total = sum(len(item) for item in contents)
        if total > max_codepoints:
            raise refused_error(
                "MT_GROUP_TOO_LARGE",
                "group exceeds the conservative Basic v2 per-request codepoint "
                "limit; the indivisible segment group is refused before any HTTP",
            )

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    def _require_context(self, context: ApiContext | None = None) -> ApiContext:
        with self._lock:
            bound = self._context if context is None else context
        if bound is None:
            raise ApiBindingError(
                "MT adapter has no bound pipeline context; direct calls are "
                "refused because raw sink, group and budget are required"
            )
        return bound

    def _require_sink(self) -> Callable[..., RawArtifactRef]:
        with self._lock:
            sink = self._raw_sink
        if sink is None:
            raise ApiBindingError(
                "MT adapter has no raw sink bound; refusing to call without "
                "durable raw archiving"
            )
        return sink

    def _resolve_credential(self) -> str:
        if self._credential_resolver is None:
            raise refused_error(
                "API_CREDENTIAL_MISSING",
                "no Google Translation API key resolver was supplied; refusing "
                "to send",
            )
        token = self._credential_resolver()
        if not isinstance(token, str) or not token.strip():
            raise refused_error(
                "API_CREDENTIAL_MISSING",
                "the Google Translation API key resolver returned no usable key",
            )
        return token.strip()

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
        for name in ("x-request-id", "x-goog-request-id", "request-id"):
            value = response.headers.get(name)
            if value:
                return str(value)
        return None

    @staticmethod
    def _error_payload(content: bytes) -> dict[str, Any] | None:
        try:
            payload = json.loads(content)
        except ValueError:
            return None
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            return payload["error"]
        return None

    def _error_summary(self, content: bytes) -> tuple[str | None, str | None]:
        """Return only a fixed safe code/summary, never arbitrary provider text."""

        error = self._error_payload(content)
        if error is None:
            return None, None
        status = error.get("status")
        code = status if isinstance(status, str) and status else None
        return code, self._fixed_error_message(code)

    @staticmethod
    def _fixed_error_message(code: str | None) -> str:
        summaries = {
            "INVALID_ARGUMENT": "provider rejected the request as invalid",
            "UNAUTHENTICATED": "provider authentication failed",
            "PERMISSION_DENIED": "provider permission denied",
            "RESOURCE_EXHAUSTED": "provider quota exhausted",
            "NOT_FOUND": "provider resource not found",
            "FAILED_PRECONDITION": "provider precondition failed",
        }
        if code is None:
            return "provider error body archived"
        return summaries.get(code, "provider error body archived")

    def _is_minute_rate(self, content: bytes) -> bool:
        """Conservative, documented detection of a minute-rate rejection.

        Basic v2 can report a per-minute quota as an HTTP 403 with an explicit
        ``RESOURCE_EXHAUSTED`` status whose message mentions a per-minute limit.
        Every other 403 (permission, billing, daily quota) stays non-retryable.
        The provider message is inspected here only and never persisted.
        """

        error = self._error_payload(content)
        if error is None:
            return False
        message = error.get("message")
        status = error.get("status")
        if not isinstance(message, str):
            return False
        lowered = message.lower()
        return status == "RESOURCE_EXHAUSTED" and "per minute" in lowered
