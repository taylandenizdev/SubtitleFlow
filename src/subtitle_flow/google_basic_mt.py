"""Google Translation **Basic v2** full-text and timed MT orchestration.

The desktop/CLI route produces two human-facing Markdown documents from a single
full-text pass and a *separate* timed pass for burned-in subtitles. This module
gives the Basic v2 API-key route that shape, with its own provider-scoped
artifacts:

* the canonical full text is chunked deterministically and translated through
  :class:`~subtitle_flow.providers.google_basic.GoogleTranslationBasicProvider`
  in conservative groups, and persisted to
  ``artifacts/mt.fulltext.google-basic.json``;
* a Turkish source is handled as a documented no-op: the exact text is echoed
  into a terminal ``skipped_same_language`` artifact with zero HTTP and zero
  reservations;
* burned subtitles get a *second*, independent timed-segment pass persisted to
  ``artifacts/mt.timed.google-basic.json`` that keeps each cue's real STT times;
* reuse validates the source identity, project/model/route/settings and every
  archived raw body *including the registered request/group/attempt correlation*,
  so a repeated run performs zero HTTP. Provider-scoped artifacts never
  cross-reuse.

No timestamp is ever fabricated and the two passes are never derived from each
other. The API key never appears in an artifact, message or trace.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Final, Sequence

from subtitle_flow.config import (
    GOOGLE_BASIC_PROVIDER,
    ApiSettings,
    GoogleBasicSettings,
    PipelineConfig,
)
from subtitle_flow.fulltext_mt import (
    FullTextChunk,
    FullTextChunkRecord,
    FullTextTranslationArtifact,
    chunk_text_with_budget,
)
from subtitle_flow.providers.api_common import (
    ApiContext,
    AttemptOutcomeLabel,
    ExpectedSegment,
)
from subtitle_flow.providers.errors import ProviderCallError
from subtitle_flow.providers.google_basic import GoogleTranslationBasicProvider
from subtitle_flow.quality import QualityInput, QualitySettings, assess_quality
from subtitle_flow.schemas import (
    ProviderIdentity,
    ProviderKind,
    RawArtifactRef,
    Segment,
    StageKind,
    Transcript,
    TranslationStatus,
)
from subtitle_flow.storage import (
    ApiTraceKind,
    JobStore,
    StoredInput,
    outcome_proves_predispatch,
    utc_now,
)
from subtitle_flow.timed_subtitles import (
    TimedCueRecord,
    TimedSubtitleArtifact,
    TimedSubtitleError,
    require_timed_transcript,
    timed_transcript_sha256,
)

__all__ = [
    "GOOGLE_BASIC_FULLTEXT_LOCATOR",
    "GOOGLE_BASIC_TIMED_LOCATOR",
    "GoogleBasicTimedRoute",
    "GoogleBasicMTError",
    "fulltext_artifact_locator_basic",
    "load_google_basic_fulltext",
    "load_google_basic_timed",
    "reuse_google_basic_fulltext",
    "reuse_google_basic_timed",
    "translate_google_basic_fulltext",
    "translate_google_basic_timed",
]

_TARGET_LANGUAGE: Final[str] = "tr"

#: Provider-scoped artifact locators. They never collide with the local
#: ``mt.fulltext*`` / ``mt.timed*`` files, so the two routes cannot cross-reuse.
GOOGLE_BASIC_FULLTEXT_LOCATOR: Final[str] = "artifacts/mt.fulltext.google-basic.json"
GOOGLE_BASIC_TIMED_LOCATOR: Final[str] = "artifacts/mt.timed.google-basic.json"

#: ``pipeline_attempt`` discriminates the two independent Basic MT passes in the
#: durable API trace. The deterministic ``gb0000`` group ids are shared between
#: the full-text and the timed-subtitle pass, so without this discriminator a
#: completed full-text group could be mistaken for timed evidence (or the other
#: way around) and vice versa. The value never changes across an explicit
#: re-send: ``http_attempt`` carries the in-call retry number instead.
_FULLTEXT_PIPELINE_ATTEMPT: Final[int] = 1
_TIMED_PIPELINE_ATTEMPT: Final[int] = 2

#: Durable replay decisions for one Basic group, mirroring the classic pipeline.
_REPLAY_SAFE: Final[str] = "safe"  # fresh, or only proven pre-dispatch outcomes
_REPLAY_BLOCKED: Final[str] = "blocked"  # dispatched terminal, no reusable output
_REPLAY_UNKNOWN: Final[str] = "remote_unknown"  # ambiguity: never auto-resend


class GoogleBasicMTError(Exception):
    """A typed failure to produce, reuse or persist a Basic-route artifact.

    When the failure originated inside the Basic provider adapter, the
    provider's own classification (whether the request may have been dispatched
    and whether the remote outcome is unknown/retryable) is preserved here rather
    than collapsed into a generic error. The CLI maps that classification to the
    documented outcomes: a pre-dispatch refusal is invalid input, an ambiguous
    dispatched result is incomplete with no automatic resend, and an
    evidence/provenance corruption is a human review item.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        dispatched: bool = False,
        remote_status_unknown: bool = False,
        retryable: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.dispatched = bool(dispatched)
        self.remote_status_unknown = bool(remote_status_unknown)
        self.retryable = bool(retryable)
        self.cause = cause
        super().__init__(f"{code}: {message}")

    @classmethod
    def from_provider_error(cls, exc: ProviderCallError) -> "GoogleBasicMTError":
        """Wrap a :class:`ProviderCallError` without losing its classification."""

        return cls(
            exc.error.code,
            exc.error.message,
            dispatched=exc.dispatched,
            remote_status_unknown=exc.error.remote_status_unknown,
            retryable=exc.error.retryable,
            cause=exc,
        )


def fulltext_artifact_locator_basic() -> str:
    return GOOGLE_BASIC_FULLTEXT_LOCATOR


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _generation_settings(
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None,
    kind: str,
) -> dict[str, Any]:
    """Canonical output-affecting settings that bind a stored Basic output.

    Binds the project, requested model resource, route version and request
    ceilings plus the quality thresholds; excludes the API key entirely.
    """

    resolved_quality = (
        quality_settings if quality_settings is not None else QualitySettings()
    )
    block: dict[str, Any] = {
        "kind": kind,
        "route_version": basic.route_version,
        "transport": "basic-v2",
        "provider": GOOGLE_BASIC_PROVIDER,
        "model": basic.model_name,
        "project": basic.project,
        "location": basic.location,
        "max_items_per_request": basic.max_items_per_request,
        "max_codepoints_per_request": basic.max_codepoints_per_request,
        "quality": resolved_quality.model_dump(mode="json"),
    }
    if basic.project is not None:
        block["model_resource"] = basic.model_resource
    return block


def _model_identity(basic: GoogleBasicSettings) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "transport": "basic-v2",
        "provider": GOOGLE_BASIC_PROVIDER,
        "model": basic.model_name,
        "project": basic.project,
        "location": basic.location,
    }
    if basic.project is not None:
        identity["model_resource"] = basic.model_resource
    return identity


def _group_id(index: int) -> str:
    """The deterministic Basic group id used for dispatch and evidence reuse."""

    return f"gb{index:04d}"


def _group_replay_state(
    store: JobStore, group_id: str, *, pipeline_attempt: int
) -> str:
    """Classify one Basic group's durable prior outcomes before any new send.

    Mirrors the classic pipeline decision: only a proven pre-dispatch outcome
    (with its release) is safely retryable; a reserve with no outcome is an
    ambiguous hard kill; any other dispatched terminal outcome (complete,
    invalid response, HTTP error or remote-unknown) is not replayable without an
    explicit authorization. The validated, hash-chained ``api.trace.jsonl`` is
    the only source, and records are scoped to one pass by ``pipeline_attempt``
    so the full-text and timed passes never see each other's evidence.
    """

    reserves: dict[str, Any] = {}
    outcomes: list[Any] = []
    for record in store.read_api_trace(stage=StageKind.mt):
        if record.group_id != group_id or record.pipeline_attempt != pipeline_attempt:
            continue
        if record.kind is ApiTraceKind.reserve:
            reserves[record.attempt_id] = record
        elif record.kind is ApiTraceKind.outcome:
            outcomes.append(record)
    outcome_ids = {record.attempt_id for record in outcomes}
    if any(attempt_id not in outcome_ids for attempt_id in reserves):
        # A reservation with no recorded outcome is a hard-killed in-flight
        # attempt: the remote status is genuinely unknown, never "fresh".
        return _REPLAY_UNKNOWN
    if any(
        record.remote_status_unknown is True
        or record.outcome == AttemptOutcomeLabel.remote_unknown
        for record in outcomes
    ):
        return _REPLAY_UNKNOWN
    if any(not outcome_proves_predispatch(record) for record in outcomes):
        # Includes a prior *complete* group whose artifact was never committed
        # (partial operation) and any invalid/provider-error response.
        return _REPLAY_BLOCKED
    return _REPLAY_SAFE


def _require_group_dispatch_allowed(
    store: JobStore,
    group_id: str,
    *,
    pipeline_attempt: int,
    allow_remote_retry: bool,
    pass_label: str,
) -> None:
    """Refuse a default replay of a group with unreusable durable paid evidence.

    Raised before the credential resolver, any reservation or any HTTP, so a
    refused default replay spends nothing. An explicit ``allow_remote_retry``
    authorizes exactly one fresh paid attempt for the group; the old trace and
    raw bodies are never rewritten or deleted.
    """

    state = _group_replay_state(store, group_id, pipeline_attempt=pipeline_attempt)
    if state == _REPLAY_SAFE or allow_remote_retry:
        return
    if state == _REPLAY_UNKNOWN:
        raise GoogleBasicMTError(
            "GOOGLE_BASIC_REMOTE_STATUS_UNKNOWN",
            f"önceki ücretli Google {pass_label} isteği ({group_id}) sağlayıcıya "
            "ulaşmış olabilir ve durumu bilinmiyor; otomatik yeniden gönderim "
            "durduruldu. Açıkça yeniden denemek için --allow-remote-retry gerekir.",
            dispatched=True,
            remote_status_unknown=True,
        )
    raise GoogleBasicMTError(
        "GOOGLE_BASIC_REPLAY_REFUSED",
        f"önceki ücretli Google {pass_label} isteği ({group_id}) sağlayıcı "
        "yanıtı üretti ama yeniden kullanılabilir bir belge yok; otomatik "
        "yeniden gönderim durduruldu. Açıkça yeniden göndermek yeni bir ücret "
        "doğurabilir ve --allow-remote-retry gerektirir.",
    )


def _group_by_limits(
    items: Sequence[Any], *, max_items: int, max_codepoints: int
) -> tuple[tuple[Any, ...], ...]:
    groups: list[tuple[Any, ...]] = []
    current: list[Any] = []
    total = 0
    for item in items:
        length = len(item.text if hasattr(item, "text") else item.source_text)
        if current and (len(current) >= max_items or total + length > max_codepoints):
            groups.append(tuple(current))
            current = []
            total = 0
        current.append(item)
        total += length
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _sink_for(store: JobStore) -> Callable[..., RawArtifactRef]:
    def sink(
        stage: StageKind,
        payload: bytes,
        *,
        request_id: str | None = None,
        group_id: str | None = None,
        attempt: int | None = None,
        content_subtype: str = "json",
    ) -> RawArtifactRef:
        # Preserve the provider request id, group and attempt exactly as the
        # adapter reports them, so the archived registration and the returned
        # reference both carry real correlation and reuse can verify them.
        return store.archive_raw(
            stage,
            payload,
            request_id=request_id,
            group_id=group_id,
            attempt=attempt,
            content_subtype=content_subtype,
        )

    return sink


def _accepted_texts(body: bytes) -> list[str]:
    """Parse an archived Basic response body into its ordered text list."""

    import json

    try:
        payload = json.loads(body)
    except ValueError as exc:
        raise GoogleBasicMTError(
            "GOOGLE_BASIC_EVIDENCE_CORRUPT",
            "archived Basic evidence is not valid JSON",
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise GoogleBasicMTError(
            "GOOGLE_BASIC_EVIDENCE_CORRUPT",
            "archived Basic evidence is missing the data object",
        )
    translations = payload["data"].get("translations")
    if not isinstance(translations, list):
        raise GoogleBasicMTError(
            "GOOGLE_BASIC_EVIDENCE_CORRUPT",
            "archived Basic evidence is missing the translations list",
        )
    texts: list[str] = []
    for item in translations:
        if not isinstance(item, dict) or not isinstance(item.get("translatedText"), str):
            raise GoogleBasicMTError(
                "GOOGLE_BASIC_EVIDENCE_CORRUPT",
                "archived Basic evidence carries an invalid translation item",
            )
        texts.append(item["translatedText"])
    return texts


def _bind(
    provider: GoogleTranslationBasicProvider,
    store: JobStore,
) -> None:
    provider.bind_run((str(store.root), store.job_id))
    provider.bind_raw_sink(_sink_for(store))


def _context_config(
    stored: StoredInput, api: ApiSettings, basic: GoogleBasicSettings
) -> Any:
    """Return a context config the Basic adapter can bind against.

    An ephemeral, non-persisted context is derived that pins the exact Basic
    identity and the live adapter settings, so a stored snapshot created under a
    different paid policy can still be translated under the current invocation.
    Nothing about the job snapshot is rewritten and no secret is added.
    """

    config = stored.config
    identity = ProviderIdentity(
        kind=ProviderKind.mt,
        provider=GOOGLE_BASIC_PROVIDER,
        model=basic.model_name,
    )
    return PipelineConfig.model_construct(
        **{
            **config.model_dump(),
            "api": api,
            "google_basic": basic,
            "mt": identity,
        }
    )


# --------------------------------------------------------------------------- #
# Full text
# --------------------------------------------------------------------------- #
def _chunk_fulltext(text: str, basic: GoogleBasicSettings) -> tuple[FullTextChunk, ...]:
    max_codepoints = basic.max_codepoints_per_request
    return chunk_text_with_budget(
        text, char_budget=max_codepoints, byte_budget=max_codepoints * 4
    )


def _chunks_match(
    artifact: FullTextTranslationArtifact, chunks: Sequence[FullTextChunk]
) -> bool:
    if len(artifact.chunks) != len(chunks):
        return False
    for record, chunk in zip(artifact.chunks, chunks):
        if (
            record.chunk_id != chunk.chunk_id
            or record.start_char != chunk.start_char
            or record.end_char != chunk.end_char
            or record.source_text != chunk.text
        ):
            return False
    return True


def load_google_basic_fulltext(
    store: JobStore,
) -> FullTextTranslationArtifact | None:
    path = store.resolve(GOOGLE_BASIC_FULLTEXT_LOCATOR)
    if not path.is_file():
        return None
    try:
        return FullTextTranslationArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:
        raise GoogleBasicMTError(
            "GOOGLE_BASIC_FULLTEXT_CORRUPT",
            f"the stored google-basic full-text translation is invalid: {exc}",
        ) from exc


def _segments_for_chunks(
    chunks: Sequence[FullTextChunk], source_language: str
) -> tuple[Segment, ...]:
    return tuple(
        Segment(
            segment_id=chunk.chunk_id,
            start_ms=0,
            end_ms=1,
            source_language=source_language,
            source_text=chunk.text,
        )
        for chunk in chunks
    )


def translate_google_basic_fulltext(
    store: JobStore,
    stored: StoredInput,
    source: Any,
    *,
    provider: GoogleTranslationBasicProvider,
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None = None,
    allow_remote_retry: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], Any] = utc_now,
    progress: Callable[[str], None] | None = None,
) -> FullTextTranslationArtifact:
    """Translate the canonical full text through Basic v2 and persist it.

    ``allow_remote_retry`` authorizes a fresh paid attempt for a group whose
    durable trace already shows an unreusable dispatched outcome. It defaults
    to ``False``: a prior injected ``invalid_response``/``MT_MODEL_DRIFT``,
    ``remote_unknown``, nonretryable error or partial-operation complete group
    refuses the default replay before any credential/reservation/HTTP.
    """

    if not store.locked:
        raise GoogleBasicMTError(
            "GOOGLE_BASIC_UNLOCKED",
            "the job store must hold its writer lock before a Basic translation",
        )
    if source.source_language == _TARGET_LANGUAGE:
        # Mirror the documented no-op: a Turkish source is echoed back.
        # No provider is bound, no reservation is made and no HTTP is
        # sent; the exact text is echoed and persisted as terminal evidence.
        artifact = _skipped_basic_fulltext(
            source,
            basic=basic,
            quality_settings=quality_settings,
            now=now,
        )
        store.write_artifact(
            GOOGLE_BASIC_FULLTEXT_LOCATOR,
            artifact.model_dump_json(indent=2).encode("utf-8"),
        )
        return artifact
    chunks = _chunk_fulltext(source.full_text, basic)
    identity = _model_identity(basic)
    generation = _generation_settings(basic, quality_settings, "fulltext")
    context_config = _context_config(stored, provider.api, basic)
    _bind(provider, store)
    started = monotonic()
    try:
        records = _translate_chunks(
            provider,
            store,
            context_config,
            chunks,
            source_language=source.source_language,
            basic=basic,
            allow_remote_retry=allow_remote_retry,
            progress=progress,
        )
    finally:
        elapsed_ms = max(0, int((monotonic() - started) * 1000))
        provider.clear_run((str(store.root), store.job_id))

    report = assess_quality(
        [
            QualityInput(
                segment_id=record.chunk_id,
                source_text=record.source_text,
                translated_text_tr=record.translated_text_tr or "",
            )
            for record in records
        ],
        settings=quality_settings,
    )
    merged = tuple(
        FullTextChunkRecord.model_validate(
            {**record.model_dump(mode="json"), "flags": report.flags_for(record.chunk_id)}
        )
        for record in records
    )
    artifact = FullTextTranslationArtifact(
        provider=GOOGLE_BASIC_PROVIDER,
        model=basic.model_name,
        source_language=source.source_language,
        source_sha256=hashlib.sha256(source.full_text.encode("utf-8")).hexdigest(),
        source_char_count=len(source.full_text),
        status="translated",
        needs_review=report.review_required,
        model_identity=identity,
        generation_settings=generation,
        elapsed_ms=elapsed_ms,
        chunks=merged,
        created_at_utc=now(),
    )
    store.write_artifact(
        GOOGLE_BASIC_FULLTEXT_LOCATOR,
        artifact.model_dump_json(indent=2).encode("utf-8"),
    )
    return artifact


def _translate_chunks(
    provider: GoogleTranslationBasicProvider,
    store: JobStore,
    context_config: Any,
    chunks: Sequence[FullTextChunk],
    *,
    source_language: str,
    basic: GoogleBasicSettings,
    allow_remote_retry: bool,
    progress: Callable[[str], None] | None,
) -> list[FullTextChunkRecord]:
    groups = _group_by_limits(
        chunks,
        max_items=basic.max_items_per_request,
        max_codepoints=basic.max_codepoints_per_request,
    )
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    records: dict[str, FullTextChunkRecord] = {}
    done = 0
    for index, group in enumerate(groups):
        group_id = _group_id(index)
        _require_group_dispatch_allowed(
            store,
            group_id,
            pipeline_attempt=_FULLTEXT_PIPELINE_ATTEMPT,
            allow_remote_retry=allow_remote_retry,
            pass_label="tam metin",
        )
        segments = _segments_for_chunks(group, source_language)
        context = ApiContext(
            store=store,
            stage=StageKind.mt,
            config=context_config,
            group_id=group_id,
            pipeline_attempt=_FULLTEXT_PIPELINE_ATTEMPT,
            expected_segments=tuple(
                ExpectedSegment(
                    segment_id=segment.segment_id,
                    source_text=segment.source_text,
                    translation_input=None,
                )
                for segment in segments
            ),
            source_language=source_language,
            target_language=_TARGET_LANGUAGE,
        )
        provider.bind_context(context)
        try:
            translation = provider.translate(
                segments, source_language, _TARGET_LANGUAGE
            )
        except ProviderCallError as exc:
            # Preserve the adapter's own classification (dispatched / remote
            # unknown / retryable) so the CLI can report the honest outcome.
            raise GoogleBasicMTError.from_provider_error(exc) from exc
        if translation.status is not TranslationStatus.translated:
            raise GoogleBasicMTError(
                "GOOGLE_BASIC_MT_FAILED", "the Basic provider did not translate"
            )
        raw_reference = translation.raw_reference
        if raw_reference is None:
            raise GoogleBasicMTError(
                "GOOGLE_BASIC_INCOMPLETE",
                "the Basic provider returned no archived raw evidence",
            )
        translated_by_id = {
            segment.segment_id: segment for segment in translation.segments
        }
        for chunk in group:
            translated = translated_by_id.get(chunk.chunk_id)
            if translated is None or translated.source_text != chunk.text:
                raise GoogleBasicMTError(
                    "GOOGLE_BASIC_PROVENANCE",
                    f"the Basic provider altered chunk {chunk.chunk_id!r}",
                )
            records[chunk.chunk_id] = FullTextChunkRecord(
                chunk_id=chunk.chunk_id,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                source_text=chunk.text,
                translated_text_tr=translated.translated_text_tr,
                raw_reference=raw_reference,
            )
        done += len(group)
        if progress is not None:
            progress(f"Google Translation LLM çevirisi: parça {done}/{len(chunks)}")
    return [records[chunk.chunk_id] for chunk in chunks]


def reuse_google_basic_fulltext(
    store: JobStore,
    source: Any,
    *,
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None = None,
) -> FullTextTranslationArtifact | None:
    """Return a completed Basic full-text artifact that matches the source, or ``None``."""

    artifact = load_google_basic_fulltext(store)
    if artifact is None:
        return None
    if artifact.fulltext_mt_version != "1":
        return None
    digest = hashlib.sha256(source.full_text.encode("utf-8")).hexdigest()
    if (
        artifact.source_sha256 != digest
        or artifact.source_char_count != len(source.full_text)
        or artifact.source_language != source.source_language
    ):
        return None
    if artifact.provider != GOOGLE_BASIC_PROVIDER or artifact.model != basic.model_name:
        return None
    if artifact.generation_settings != _generation_settings(
        basic, quality_settings, "fulltext"
    ):
        return None
    chunks = _chunk_fulltext(source.full_text, basic)
    if not _chunks_match(artifact, chunks):
        return None
    if artifact.status == "translated":
        _verify_fulltext_evidence(store, artifact, chunks, basic)
    return artifact


def _verify_fulltext_evidence(
    store: JobStore,
    artifact: FullTextTranslationArtifact,
    chunks: Sequence[FullTextChunk],
    basic: GoogleBasicSettings,
) -> None:
    by_id = {record.chunk_id: record for record in artifact.chunks}
    groups = _group_by_limits(
        chunks,
        max_items=basic.max_items_per_request,
        max_codepoints=basic.max_codepoints_per_request,
    )
    for index, group in enumerate(groups):
        references = {by_id[chunk.chunk_id].raw_reference for chunk in group}
        if len(references) != 1 or None in references:
            raise GoogleBasicMTError(
                "GOOGLE_BASIC_EVIDENCE_CORRUPT",
                "a Basic full-text group does not bind one archived response",
            )
        reference = next(iter(references))
        assert reference is not None
        try:
            body = store.verify_raw(
                reference,
                stage=StageKind.mt,
                group_id=_group_id(index),
                attempt=reference.attempt,
            )
        except Exception as exc:  # noqa: BLE001 - normalized to a typed failure
            raise GoogleBasicMTError(
                "GOOGLE_BASIC_EVIDENCE_CORRUPT",
                f"archived Basic evidence is not usable: {exc}",
            ) from exc
        texts = _accepted_texts(body)
        if len(texts) != len(group):
            raise GoogleBasicMTError(
                "GOOGLE_BASIC_EVIDENCE_CORRUPT",
                "archived Basic evidence does not cover its whole group",
            )
        for chunk, text in zip(group, texts):
            if by_id[chunk.chunk_id].translated_text_tr != text:
                raise GoogleBasicMTError(
                    "GOOGLE_BASIC_EVIDENCE_CORRUPT",
                    f"chunk {chunk.chunk_id!r} does not match its archived body",
                )


# --------------------------------------------------------------------------- #
# Timed segments
# --------------------------------------------------------------------------- #
def load_google_basic_timed(store: JobStore) -> TimedSubtitleArtifact | None:
    path = store.resolve(GOOGLE_BASIC_TIMED_LOCATOR)
    if not path.is_file():
        return None
    try:
        return TimedSubtitleArtifact.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (ValueError, OSError) as exc:
        raise TimedSubtitleError(
            "TIMED_SUBTITLE_CORRUPT",
            f"the stored google-basic timed translation is invalid: {exc}",
        ) from exc


def _cues_match(artifact: TimedSubtitleArtifact, transcript: Transcript) -> bool:
    if len(artifact.cues) != len(transcript.segments):
        return False
    for record, segment in zip(artifact.cues, transcript.segments):
        if (
            record.segment_id != segment.segment_id
            or record.start_ms != segment.start_ms
            or record.end_ms != segment.end_ms
            or record.source_text != segment.source_text
            or record.translation_input != segment.translation_input
        ):
            return False
    return True


def _sent_text(segment: Segment) -> str:
    """The exact text the provider request sends for one segment.

    The Basic provider sends ``translation_input`` when it is present and falls
    back to ``source_text`` otherwise; grouping and the per-segment length guard
    must therefore measure the same value, never the source alone.
    """

    if segment.translation_input is not None:
        return segment.translation_input
    return segment.source_text


def _groups_for_segments(
    segments: Sequence[Segment],
    *,
    max_items: int,
    max_codepoints: int,
) -> tuple[tuple[Segment, ...], ...]:
    groups: list[tuple[Segment, ...]] = []
    current: list[Segment] = []
    total = 0
    for segment in segments:
        length = len(_sent_text(segment))
        if current and (len(current) >= max_items or total + length > max_codepoints):
            groups.append(tuple(current))
            current = []
            total = 0
        current.append(segment)
        total += length
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def translate_google_basic_timed(
    store: JobStore,
    stored: StoredInput,
    transcript: Transcript,
    *,
    provider: GoogleTranslationBasicProvider,
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None = None,
    allow_remote_retry: bool = False,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], Any] = utc_now,
    progress: Callable[[str], None] | None = None,
) -> TimedSubtitleArtifact:
    """Translate the canonical timed segments through Basic v2 and persist them.

    Like the full-text pass this is fail-closed by default: a group with a prior
    unreusable dispatched outcome (including an ambiguous/unknown result) is not
    resent without an explicit ``allow_remote_retry``. The timed pass keeps its
    own ``pipeline_attempt`` scope, so a completed full-text group is never
    mistaken for timed evidence.
    """

    if not store.locked:
        raise TimedSubtitleError(
            "TIMED_SUBTITLE_UNLOCKED",
            "the job store must hold its writer lock before a timed translation",
        )
    source_language = require_timed_transcript(transcript)
    if source_language == _TARGET_LANGUAGE:
        artifact = _skipped_basic_timed(
            transcript, basic=basic, quality_settings=quality_settings, now=now
        )
        store.write_artifact(
            GOOGLE_BASIC_TIMED_LOCATOR,
            artifact.model_dump_json(indent=2).encode("utf-8"),
        )
        return artifact

    generation = _generation_settings(basic, quality_settings, "timed")
    identity = _model_identity(basic)
    context_config = _context_config(stored, provider.api, basic)
    _bind(provider, store)
    started = monotonic()
    try:
        cues = _translate_cues(
            provider,
            store,
            context_config,
            transcript,
            source_language=source_language,
            basic=basic,
            allow_remote_retry=allow_remote_retry,
            progress=progress,
        )
    finally:
        elapsed_ms = max(0, int((monotonic() - started) * 1000))
        provider.clear_run((str(store.root), store.job_id))

    report = assess_quality(
        [
            QualityInput(
                segment_id=record.segment_id,
                source_text=record.source_text,
                translated_text_tr=record.translated_text_tr or "",
                translation_input=record.translation_input,
                existing_flags=record.flags,
            )
            for record in cues
        ],
        settings=quality_settings,
    )
    merged = tuple(
        TimedCueRecord.model_validate(
            {**record.model_dump(mode="json"), "flags": report.flags_for(record.segment_id)}
        )
        for record in cues
    )
    artifact = TimedSubtitleArtifact(
        provider=GOOGLE_BASIC_PROVIDER,
        model=basic.model_name,
        source_provider=transcript.provider,
        source_model=transcript.model,
        source_language=source_language,
        source_transcript_sha256=timed_transcript_sha256(transcript),
        source_segment_count=len(transcript.segments),
        status="translated",
        needs_review=report.review_required,
        model_identity=identity,
        generation_settings=generation,
        elapsed_ms=elapsed_ms,
        cues=merged,
        created_at_utc=now(),
    )
    store.write_artifact(
        GOOGLE_BASIC_TIMED_LOCATOR,
        artifact.model_dump_json(indent=2).encode("utf-8"),
    )
    return artifact


def _translate_cues(
    provider: GoogleTranslationBasicProvider,
    store: JobStore,
    context_config: Any,
    transcript: Transcript,
    *,
    source_language: str,
    basic: GoogleBasicSettings,
    allow_remote_retry: bool,
    progress: Callable[[str], None] | None,
) -> list[TimedCueRecord]:
    groups = _groups_for_segments(
        transcript.segments,
        max_items=basic.max_items_per_request,
        max_codepoints=basic.max_codepoints_per_request,
    )
    records: dict[str, TimedCueRecord] = {}
    done = 0
    for index, group in enumerate(groups):
        group_id = _group_id(index)
        for segment in group:
            if len(_sent_text(segment)) > basic.max_codepoints_per_request:
                raise TimedSubtitleError(
                    "TIMED_SUBTITLE_SEGMENT_TOO_LONG",
                    f"segment {segment.segment_id!r} exceeds the Basic v2 codepoint "
                    "limit; a subtitle cue is never split",
                )
        _require_group_dispatch_allowed(
            store,
            group_id,
            pipeline_attempt=_TIMED_PIPELINE_ATTEMPT,
            allow_remote_retry=allow_remote_retry,
            pass_label="zaman kodlu altyazı",
        )
        context = ApiContext(
            store=store,
            stage=StageKind.mt,
            config=context_config,
            group_id=group_id,
            pipeline_attempt=_TIMED_PIPELINE_ATTEMPT,
            expected_segments=tuple(
                ExpectedSegment(
                    segment_id=segment.segment_id,
                    source_text=segment.source_text,
                    translation_input=segment.translation_input,
                )
                for segment in group
            ),
            source_language=source_language,
            target_language=_TARGET_LANGUAGE,
        )
        provider.bind_context(context)
        try:
            translation = provider.translate(
                group, source_language, _TARGET_LANGUAGE
            )
        except ProviderCallError as exc:
            # Keep the adapter's dispatch/unknown/retryable classification instead
            # of collapsing every failure into one opaque code.
            raise GoogleBasicMTError.from_provider_error(exc) from exc
        if translation.status is not TranslationStatus.translated:
            raise TimedSubtitleError(
                "TIMED_SUBTITLE_FAILED", "the Basic provider did not translate"
            )
        raw_reference = translation.raw_reference
        if raw_reference is None:
            raise TimedSubtitleError(
                "TIMED_SUBTITLE_INCOMPLETE",
                "the Basic provider returned no archived raw evidence",
            )
        translated_by_id = {
            segment.segment_id: segment for segment in translation.segments
        }
        for segment in group:
            translated = translated_by_id.get(segment.segment_id)
            if translated is None or translated.source_text != segment.source_text:
                raise TimedSubtitleError(
                    "TIMED_SUBTITLE_PROVENANCE",
                    f"the Basic provider altered segment {segment.segment_id!r}",
                )
            records[segment.segment_id] = TimedCueRecord(
                segment_id=segment.segment_id,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                source_text=segment.source_text,
                translation_input=segment.translation_input,
                translated_text_tr=translated.translated_text_tr,
                raw_reference=raw_reference,
            )
        done += len(group)
        if progress is not None:
            progress(
                f"Altyazı için Google Translation LLM çevirisi: "
                f"{done}/{len(transcript.segments)}"
            )
    return [records[segment.segment_id] for segment in transcript.segments]


def _skipped_basic_timed(
    transcript: Transcript,
    *,
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None,
    now: Callable[[], Any],
) -> TimedSubtitleArtifact:
    records = tuple(
        TimedCueRecord(
            segment_id=segment.segment_id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            source_text=segment.source_text,
            translation_input=segment.translation_input,
            translated_text_tr=segment.source_text,
            flags=("same_language",),
            raw_reference=None,
        )
        for segment in transcript.segments
    )
    report = assess_quality(
        [
            QualityInput(
                segment_id=record.segment_id,
                source_text=record.source_text,
                translated_text_tr=record.translated_text_tr or "",
                translation_input=record.translation_input,
                existing_flags=record.flags,
            )
            for record in records
        ],
        settings=quality_settings,
        same_language=True,
    )
    return TimedSubtitleArtifact(
        provider=GOOGLE_BASIC_PROVIDER,
        model=basic.model_name,
        source_provider=transcript.provider,
        source_model=transcript.model,
        source_language=transcript.source_language or _TARGET_LANGUAGE,
        source_transcript_sha256=timed_transcript_sha256(transcript),
        source_segment_count=len(transcript.segments),
        status="skipped_same_language",
        needs_review=report.review_required,
        model_identity={},
        generation_settings=_generation_settings(basic, quality_settings, "timed"),
        elapsed_ms=0,
        cues=records,
        created_at_utc=now(),
    )


def _skipped_basic_fulltext(
    source: Any,
    *,
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None,
    now: Callable[[], Any],
) -> FullTextTranslationArtifact:
    """Persist a Turkish-source full-text no-op with zero HTTP and no model call."""

    chunks = _chunk_fulltext(source.full_text, basic)
    records = tuple(
        FullTextChunkRecord(
            chunk_id=chunk.chunk_id,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            source_text=chunk.text,
            translated_text_tr=chunk.text,
            flags=("same_language",),
            raw_reference=None,
        )
        for chunk in chunks
    )
    report = assess_quality(
        [
            QualityInput(
                segment_id=record.chunk_id,
                source_text=record.source_text,
                translated_text_tr=record.translated_text_tr or "",
                existing_flags=record.flags,
            )
            for record in records
        ],
        settings=quality_settings,
        same_language=True,
    )
    return FullTextTranslationArtifact(
        provider=GOOGLE_BASIC_PROVIDER,
        model=basic.model_name,
        source_language=source.source_language,
        source_sha256=hashlib.sha256(source.full_text.encode("utf-8")).hexdigest(),
        source_char_count=len(source.full_text),
        status="skipped_same_language",
        needs_review=report.review_required,
        model_identity={},
        generation_settings=_generation_settings(basic, quality_settings, "fulltext"),
        elapsed_ms=0,
        chunks=records,
        created_at_utc=now(),
    )


def reuse_google_basic_timed(
    store: JobStore,
    stored: StoredInput,
    transcript: Transcript,
    *,
    basic: GoogleBasicSettings,
    quality_settings: QualitySettings | None = None,
) -> TimedSubtitleArtifact | None:
    """Return a completed Basic timed artifact that matches the transcript, or ``None``."""

    require_timed_transcript(transcript)
    artifact = load_google_basic_timed(store)
    if artifact is None:
        return None
    if artifact.timed_subtitle_version != "2":
        return None
    if (
        artifact.source_transcript_sha256 != timed_transcript_sha256(transcript)
        or artifact.source_segment_count != len(transcript.segments)
        or artifact.source_language != transcript.source_language
        or artifact.source_provider != transcript.provider
        or artifact.source_model != transcript.model
    ):
        return None
    if artifact.provider != GOOGLE_BASIC_PROVIDER or artifact.model != basic.model_name:
        return None
    if artifact.generation_settings != _generation_settings(
        basic, quality_settings, "timed"
    ):
        return None
    if not _cues_match(artifact, transcript):
        return None
    if artifact.status == "translated":
        _verify_timed_evidence(store, artifact, transcript, basic)
    return artifact


def _verify_timed_evidence(
    store: JobStore,
    artifact: TimedSubtitleArtifact,
    transcript: Transcript,
    basic: GoogleBasicSettings,
) -> None:
    by_id = {record.segment_id: record for record in artifact.cues}
    groups = _groups_for_segments(
        transcript.segments,
        max_items=basic.max_items_per_request,
        max_codepoints=basic.max_codepoints_per_request,
    )
    for index, group in enumerate(groups):
        references = {by_id[segment.segment_id].raw_reference for segment in group}
        if len(references) != 1 or None in references:
            raise TimedSubtitleError(
                "TIMED_SUBTITLE_EVIDENCE_CORRUPT",
                "a Basic timed group does not bind one archived response",
            )
        reference = next(iter(references))
        assert reference is not None
        try:
            body = store.verify_raw(
                reference,
                stage=StageKind.mt,
                group_id=_group_id(index),
                attempt=reference.attempt,
            )
        except Exception as exc:  # noqa: BLE001 - normalized to a typed failure
            raise TimedSubtitleError(
                "TIMED_SUBTITLE_EVIDENCE_CORRUPT",
                f"archived Basic evidence is not usable: {exc}",
            ) from exc
        texts = _accepted_texts(body)
        if len(texts) != len(group):
            raise TimedSubtitleError(
                "TIMED_SUBTITLE_EVIDENCE_CORRUPT",
                "archived Basic evidence does not cover its whole group",
            )
        for segment, text in zip(group, texts):
            if by_id[segment.segment_id].translated_text_tr != text:
                raise TimedSubtitleError(
                    "TIMED_SUBTITLE_EVIDENCE_CORRUPT",
                    f"cue {segment.segment_id!r} does not match its archived body",
                )


class GoogleBasicTimedRoute:
    """A ``video_burn``-facing route for the Basic v2 timed pass."""

    def __init__(
        self,
        *,
        store: JobStore,
        provider: GoogleTranslationBasicProvider,
        basic: GoogleBasicSettings,
        allow_remote_retry: bool = False,
        before_dispatch: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._basic = basic
        self._allow_remote_retry = allow_remote_retry
        #: Optional paid/config gate invoked only when this route is about to
        #: make a genuinely required remote call. A reused timed artifact (and a
        #: Turkish-source no-op) never runs it, so an offline completion does not
        #: demand fresh consent, credentials or reservation.
        self._before_dispatch = before_dispatch

    def load_existing(self) -> TimedSubtitleArtifact | None:
        return load_google_basic_timed(self._store)

    def reuse(
        self,
        stored: StoredInput,
        transcript: Transcript,
        *,
        quality_settings: QualitySettings | None = None,
    ) -> TimedSubtitleArtifact | None:
        return reuse_google_basic_timed(
            self._store,
            stored,
            transcript,
            basic=self._basic,
            quality_settings=quality_settings,
        )

    def translate(
        self,
        stored: StoredInput,
        transcript: Transcript,
        *,
        quality_settings: QualitySettings | None = None,
        progress: Callable[[str], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], Any] = utc_now,
    ) -> TimedSubtitleArtifact:
        if (
            self._before_dispatch is not None
            and require_timed_transcript(transcript) != _TARGET_LANGUAGE
        ):
            self._before_dispatch()
        return translate_google_basic_timed(
            self._store,
            stored,
            transcript,
            provider=self._provider,
            basic=self._basic,
            quality_settings=quality_settings,
            allow_remote_retry=self._allow_remote_retry,
            progress=progress,
            monotonic=monotonic,
            now=now,
        )
