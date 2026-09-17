"""Common, JSON-serializable records shared by every STT/MT provider.

This module defines the Phase 1 contract only. File persistence, job-state
transitions, resume semantics, locks and cache keys are explicitly out of scope
and belong to Phase 2.

Frozen fields and tuple storage protect the identity, time and segment
invariants: a validated record cannot be mutated in place, so unique segment IDs,
source text and monotonic time ranges stay intact. ``request_metadata`` is the
deliberate exception. It is a JSON-validated, caller-owned *mutable* mapping; the
validated copy is detached from the caller's input, but mutating the stored copy
can still invalidate the JSON invariant, so the record must be re-validated
before it is trusted again. Only real JSON values are accepted, recursively:
dictionaries with string keys at every depth, lists, strings, finite numbers,
booleans and null. Tuples and other non-JSON types are rejected rather than
converted, so ``model_dump_json`` followed by a load always returns an equal
value. Mutating the stored mapping bypasses that guarantee; validate it again
with ``type(record).model_validate(record.model_dump())`` before trusting it.
"""

from __future__ import annotations

import copy
import math
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Annotated, Any, Final, Sequence

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from subtitle_flow.languages import (
    TARGET_LANGUAGE_CODE,
    normalize_language,
    require_supported_language,
)

__all__ = [
    "SCHEMA_VERSION",
    "CostAmount",
    "JobCost",
    "JobInput",
    "JobRecord",
    "JobStatus",
    "ProviderIdentity",
    "ProviderKind",
    "RawArtifactRef",
    "RetryRecord",
    "Segment",
    "SegmentContractError",
    "StageError",
    "StageKind",
    "StageResult",
    "Transcript",
    "TranslatedSegment",
    "Translation",
    "TranslationStatus",
    "segment_contract_issues",
    "validate_segment_contract",
]

SCHEMA_VERSION: Final[str] = "1"

# Strict integers: bool, str and float are rejected instead of being coerced.
Milliseconds = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=0)]
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

_FROZEN = ConfigDict(extra="forbid", frozen=True)


def _ensure_utc(value: datetime) -> datetime:
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be timezone-aware UTC (offset 0)")
    return value


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_ensure_utc)]


def _validate_json_value(value: Any, path: str, active: set[int]) -> None:
    """Reject any value that JSON cannot roundtrip faithfully.

    ``active`` holds the ids of containers on the current recursion path, so a
    reference cycle raises a normal ``ValueError`` instead of recursing without
    bound. Sharing the same container more than once is not a cycle.
    """

    if value is None or isinstance(value, str) or isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"metadata must be JSON-serializable: non-finite number at {path}"
            )
        return
    if isinstance(value, int):
        return
    if isinstance(value, dict):
        marker = id(value)
        if marker in active:
            raise ValueError(
                f"metadata must be JSON-serializable: circular reference at {path}"
            )
        active.add(marker)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        "metadata must be JSON-serializable: non-string mapping "
                        f"key at {path} ({type(key).__name__})"
                    )
                _validate_json_value(item, f"{path}.{key}", active)
        finally:
            active.discard(marker)
        return
    if isinstance(value, list):
        marker = id(value)
        if marker in active:
            raise ValueError(
                f"metadata must be JSON-serializable: circular reference at {path}"
            )
        active.add(marker)
        try:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active)
        finally:
            active.discard(marker)
        return
    raise ValueError(
        f"metadata must be JSON-serializable: {path} has non-JSON type "
        f"{type(value).__name__}"
    )


def _ensure_json_object(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("metadata must be a mapping with string keys")
    _validate_json_value(value, "metadata", set())
    # Detach the validated mapping from caller-owned nested containers so a later
    # mutation of the caller's input cannot silently alter this record. Every
    # value is a real JSON type, so the copy is exact and is never canonicalized.
    return copy.deepcopy(value)


JsonMetadata = Annotated[dict[str, Any], AfterValidator(_ensure_json_object)]


def _normalize_turkish_target(value: str) -> str:
    """Return canonical ``'tr'`` for any supported Turkish alias, else raise."""

    normalized = normalize_language(value)
    if normalized.canonical_code != TARGET_LANGUAGE_CODE:
        raise ValueError(
            f"target_language must be Turkish ({TARGET_LANGUAGE_CODE!r}); "
            f"got {value!r}"
        )
    return TARGET_LANGUAGE_CODE


class ProviderKind(StrEnum):
    stt = "stt"
    mt = "mt"


class ProviderIdentity(BaseModel):
    """Explicit provider and exact model identity. No silent fallback exists."""

    model_config = _FROZEN

    kind: ProviderKind
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class RawArtifactRef(BaseModel):
    """Reference to raw provider data without claiming file persistence exists.

    ``locator`` is an opaque caller-defined pointer (path, object-store key,
    in-memory handle id). This record only makes the raw payload representable so
    it can be persisted later without losing provenance.

    ``request_id``, ``group_id`` and ``attempt`` are the provider-side
    correlation values the durable archive registered for this body. They are
    carried on the returned reference (and through any artifact that stores it)
    so reuse can prove the exact registered provenance instead of re-checking
    with an unconstrained ``None``. They stay optional so a body archived with no
    group/attempt (for example a legacy full-text chunk) is still valid.
    """

    model_config = _FROZEN

    kind: str = Field(min_length=1)
    locator: str | None = None
    sha256: Sha256Hex | None = None
    request_id: str | None = None
    group_id: str | None = None
    attempt: int | None = Field(default=None, ge=1)


class StageError(BaseModel):
    """Representable failure or remote uncertainty. Never a fabricated success."""

    model_config = _FROZEN

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    remote_status_unknown: bool = False


class RetryRecord(BaseModel):
    model_config = _FROZEN

    attempt: PositiveInt = Field(ge=1)
    reason: str = Field(min_length=1)
    occurred_at_utc: UtcDatetime
    request_id: str | None = None


class StageKind(StrEnum):
    stt = "stt"
    mt = "mt"


class Segment(BaseModel):
    """One timed source segment produced by STT.

    ``source_text`` is stored exactly as received and is never rewritten.
    ``translation_input`` is a separate, optionally derived value used for MT.
    Time ranges may overlap across speakers and are never sorted or repaired;
    ``speaker=None`` is valid.
    """

    model_config = _FROZEN

    segment_id: str = Field(min_length=1)
    start_ms: Milliseconds
    end_ms: Milliseconds
    source_language: str
    source_text: str
    translation_input: str | None = None
    translated_text_tr: str | None = None
    speaker: str | None = None
    flags: tuple[str, ...] = ()

    @field_validator("source_language")
    @classmethod
    def _canonical_source_language(cls, value: str) -> str:
        canonical = require_supported_language(value, field="source_language")
        if canonical != value:
            raise ValueError(
                f"source_language must be a canonical code, got {value!r}; "
                f"use {canonical!r} and keep the provider code separately"
            )
        return value

    @model_validator(mode="after")
    def _ordered_times(self) -> "Segment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be strictly greater than start_ms")
        return self

    def with_translation_input(self, text: str) -> "Segment":
        return Segment.model_validate({**self.model_dump(), "translation_input": text})

    def with_translation(self, text: str) -> "Segment":
        return Segment.model_validate({**self.model_dump(), "translated_text_tr": text})


class Transcript(BaseModel):
    """STT result for one audio file, preserving incoming segment order."""

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_language: str | None = None
    provider_language_code: str | None = None
    language_uncertain: bool = False
    audio_path: str | None = None
    audio_duration_ms: Milliseconds
    request_id: str | None = None
    segments: tuple[Segment, ...] = ()
    raw_reference: RawArtifactRef | None = None
    request_metadata: JsonMetadata = Field(default_factory=dict)

    @field_validator("source_language")
    @classmethod
    def _canonical_source_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        canonical = require_supported_language(value, field="source_language")
        if canonical != value:
            raise ValueError(f"source_language must be canonical, got {value!r}")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "Transcript":
        ids = [segment.segment_id for segment in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("segment_id values must be unique within a transcript")
        for segment in self.segments:
            if segment.end_ms > self.audio_duration_ms:
                raise ValueError(
                    f"segment {segment.segment_id!r} ends after audio_duration_ms"
                )
        # A known transcript language with low confidence is legitimate and stays
        # valid. What is not accepted is a known contradiction: segments that
        # disagree with each other or with a known transcript language must be
        # recorded with language_uncertain=True instead of being silently
        # relabeled. Actual segment languages and content are always preserved.
        segment_languages = {segment.source_language for segment in self.segments}
        mixed_languages = len(segment_languages) > 1
        disagrees_with_transcript = (
            self.source_language is not None
            and any(
                language != self.source_language for language in segment_languages
            )
        )
        if (
            mixed_languages or disagrees_with_transcript
        ) and not self.language_uncertain:
            raise ValueError(
                "segment source languages disagree with each other or with the "
                "transcript; set language_uncertain=True instead of guessing"
            )
        if self.provider_language_code is not None:
            normalized = normalize_language(self.provider_language_code)
            if normalized.canonical_code is None and not self.language_uncertain:
                raise ValueError(
                    "unknown provider_language_code requires language_uncertain=True"
                )
            if (
                normalized.canonical_code is not None
                and self.source_language is not None
                and normalized.canonical_code != self.source_language
                and not self.language_uncertain
            ):
                raise ValueError(
                    "provider_language_code and source_language disagree"
                )
        return self


class TranslatedSegment(BaseModel):
    """A translation item retaining the source segment identity and text."""

    model_config = _FROZEN

    segment_id: str = Field(min_length=1)
    source_text: str
    translation_input: str | None = None
    translated_text_tr: str
    flags: tuple[str, ...] = ()


class TranslationStatus(StrEnum):
    translated = "translated"
    skipped_same_language = "skipped_same_language"
    failed = "failed"


class SegmentContractError(ValueError):
    """Raised when MT output does not match the expected source contract.

    Covers segment set/order mismatches as well as provider/model identity and
    source-text/translation-input provenance drift.
    """

    def __init__(self, issues: Sequence[str], *, context: str = "segment contract") -> None:
        self.issues: tuple[str, ...] = tuple(issues)
        detail = "; ".join(self.issues) if self.issues else "unknown mismatch"
        super().__init__(f"{context}: {detail}")


def segment_contract_issues(
    expected_segment_ids: Sequence[str],
    actual_segment_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return human-readable alignment issues without raising.

    Detects missing, duplicate, extra and reordered MT items so pairing never
    happens on a mismatched set. Order is significant and never repaired.
    """

    expected = list(expected_segment_ids)
    actual = list(actual_segment_ids)
    issues: list[str] = []
    if len(set(expected)) != len(expected):
        issues.append("expected source segment ids contain duplicates")
    expected_set = set(expected)
    if len(set(actual)) != len(actual):
        issues.append("translation segment ids contain duplicates")
    actual_set = set(actual)
    missing = [segment_id for segment_id in expected if segment_id not in actual_set]
    extra = [segment_id for segment_id in actual if segment_id not in expected_set]
    if missing:
        issues.append(f"missing translated segments: {missing}")
    if extra:
        issues.append(f"unexpected translated segments: {extra}")
    if not missing and not extra and expected != actual:
        issues.append("translation segment order differs from source order")
    return tuple(issues)


def validate_segment_contract(
    expected_segment_ids: Sequence[str],
    actual_segment_ids: Sequence[str],
    *,
    context: str = "segment contract",
) -> None:
    """Raise :class:`SegmentContractError` if the MT contract is not satisfied."""

    issues = segment_contract_issues(expected_segment_ids, actual_segment_ids)
    if issues:
        raise SegmentContractError(issues, context=context)


class Translation(BaseModel):
    """MT result. Retains segment IDs and never fabricates success on failure."""

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    status: TranslationStatus
    source_language: str | None = None
    target_language: str = "tr"
    segments: tuple[TranslatedSegment, ...] = ()
    raw_reference: RawArtifactRef | None = None
    request_metadata: JsonMetadata = Field(default_factory=dict)
    error: StageError | None = None

    @field_validator("source_language")
    @classmethod
    def _canonical_source_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        canonical = require_supported_language(value, field="source_language")
        if canonical != value:
            raise ValueError(f"source_language must be canonical, got {value!r}")
        return value

    @field_validator("target_language")
    @classmethod
    def _turkish_target_language(cls, value: str) -> str:
        return _normalize_turkish_target(value)

    @model_validator(mode="after")
    def _check_status(self) -> "Translation":
        ids = [segment.segment_id for segment in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("segment_id values must be unique within a translation")
        if self.status is TranslationStatus.failed:
            if self.error is None:
                raise ValueError("failed translation requires an error record")
            if self.segments:
                raise ValueError("failed translation must not carry translated segments")
        else:
            if self.error is not None:
                raise ValueError("only failed translations may carry an error record")
        if self.status is TranslationStatus.skipped_same_language:
            if self.source_language != self.target_language:
                raise ValueError("skipped_same_language requires source == target")
            for segment in self.segments:
                if segment.translated_text_tr != segment.source_text:
                    raise ValueError(
                        "skipped_same_language must preserve the source text"
                    )
        return self

    def validate_against(self, expected_segment_ids: Sequence[str]) -> None:
        """Validate retained IDs and order against expected source segments."""

        validate_segment_contract(
            expected_segment_ids,
            [segment.segment_id for segment in self.segments],
            context=f"translation {self.provider}/{self.model}",
        )


class CostAmount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    amount: float = Field(ge=0)
    currency: str = Field(pattern=r"^[A-Z]{3}$")


class JobCost(BaseModel):
    """Cost representation. Amounts are never invented.

    A cost with no amounts must be recorded as ``unknown=True`` with a non-blank
    ``unknown_reason``. A known cost carries at least one amount and must not
    simultaneously claim to be unknown. Bare ``JobCost()`` is therefore rejected
    as ambiguous; callers with nothing measured yet must say so explicitly (see
    :func:`_unmeasured_cost`).
    """

    model_config = _FROZEN

    estimated: CostAmount | None = None
    actual: CostAmount | None = None
    unknown: bool = False
    unknown_reason: str | None = None

    @model_validator(mode="after")
    def _check_unknown(self) -> "JobCost":
        if self.unknown:
            if self.unknown_reason is None or not self.unknown_reason.strip():
                raise ValueError("unknown cost requires a non-blank unknown_reason")
            if self.estimated is not None or self.actual is not None:
                raise ValueError("unknown cost must not carry amounts")
        else:
            if self.unknown_reason is not None:
                raise ValueError("unknown_reason is only valid when unknown=True")
            if self.estimated is None and self.actual is None:
                raise ValueError(
                    "a cost with no amounts must be recorded as unknown=True with "
                    "an unknown_reason"
                )
        return self


def _unmeasured_cost() -> JobCost:
    """Safe default for a job whose cost has not been measured yet."""

    return JobCost(unknown=True, unknown_reason="not measured yet")


class StageResult(BaseModel):
    """Per-stage metadata with UTC timestamps and nonnegative elapsed duration."""

    model_config = _FROZEN

    stage: StageKind
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    started_at_utc: UtcDatetime
    finished_at_utc: UtcDatetime | None = None
    elapsed_ms: PositiveInt | None = None
    source_char_count: PositiveInt | None = None
    target_char_count: PositiveInt | None = None
    retries: tuple[RetryRecord, ...] = ()
    error: StageError | None = None
    request_metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_timing(self) -> "StageResult":
        if self.finished_at_utc is not None and self.finished_at_utc < self.started_at_utc:
            raise ValueError("finished_at_utc must not precede started_at_utc")
        attempts = [retry.attempt for retry in self.retries]
        if len(set(attempts)) != len(attempts):
            raise ValueError("retry attempt numbers must be unique")
        return self


class JobStatus(StrEnum):
    created = "created"
    stt_running = "stt_running"
    stt_complete = "stt_complete"
    mt_running = "mt_running"
    complete = "complete"
    needs_review = "needs_review"
    failed = "failed"
    interrupted = "interrupted"
    remote_status_unknown = "remote_status_unknown"


class JobInput(BaseModel):
    """Stable job input: audio identity, configuration and versions.

    No video is required. An optional, separate video hash is representable.

    The ``audio_*`` media fields after ``audio_duration_ms`` are additive Phase 2
    foundation metadata describing the *verified* ready-audio file (actual
    container, codec, channel count, sample rate and byte size). They are all
    optional so Phase 1 callers and stored records remain valid; a value of
    ``None`` means only that the field was not recorded, never that validation
    passed.
    """

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    created_at_utc: UtcDatetime
    audio_path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    audio_sha256: Sha256Hex
    audio_duration_ms: Milliseconds
    audio_format: str | None = Field(default=None, min_length=1)
    audio_codec: str | None = Field(default=None, min_length=1)
    audio_channels: StrictPositiveInt | None = None
    audio_sample_rate: StrictPositiveInt | None = None
    audio_size_bytes: StrictPositiveInt | None = None
    video_path: str | None = None
    video_sha256: Sha256Hex | None = None
    source_language: str | None = None
    provider_language_code: str | None = None
    stt: ProviderIdentity
    mt: ProviderIdentity
    target_language: str = "tr"
    keyterms: tuple[str, ...] = ()
    keyterms_version: str | None = None
    segmenter_version: str = Field(min_length=1)

    @field_validator("source_language")
    @classmethod
    def _canonical_source_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        canonical = require_supported_language(value, field="source_language")
        if canonical != value:
            raise ValueError(f"source_language must be canonical, got {value!r}")
        return value

    @field_validator("target_language")
    @classmethod
    def _turkish_target_language(cls, value: str) -> str:
        return _normalize_turkish_target(value)

    @model_validator(mode="after")
    def _check_input(self) -> "JobInput":
        if self.stt.kind is not ProviderKind.stt:
            raise ValueError("stt identity must have kind='stt'")
        if self.mt.kind is not ProviderKind.mt:
            raise ValueError("mt identity must have kind='mt'")
        if (self.video_path is None) != (self.video_sha256 is None):
            raise ValueError("video_path and video_sha256 must be set together")
        return self


class JobRecord(BaseModel):
    """Status representation for one job. State transitions are Phase 2."""

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    status: JobStatus
    created_at_utc: UtcDatetime
    updated_at_utc: UtcDatetime
    finished_at_utc: UtcDatetime | None = None
    input: JobInput
    stages: tuple[StageResult, ...] = ()
    cost: JobCost = Field(default_factory=_unmeasured_cost)
    errors: tuple[StageError, ...] = ()
    request_metadata: JsonMetadata = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_record(self) -> "JobRecord":
        if self.input.job_id != self.job_id:
            raise ValueError("job_id must match input.job_id")
        if self.updated_at_utc < self.created_at_utc:
            raise ValueError("updated_at_utc must not precede created_at_utc")
        if (
            self.finished_at_utc is not None
            and self.finished_at_utc < self.created_at_utc
        ):
            raise ValueError("finished_at_utc must not precede created_at_utc")
        stage_kinds = [stage.stage for stage in self.stages]
        if len(set(stage_kinds)) != len(stage_kinds):
            raise ValueError("each stage may appear at most once")
        if self.status is JobStatus.failed and not self.errors:
            raise ValueError("failed job requires at least one error record")
        terminal = {
            JobStatus.complete,
            JobStatus.failed,
            JobStatus.interrupted,
            JobStatus.needs_review,
        }
        if self.status in terminal and self.finished_at_utc is None:
            raise ValueError(f"status {self.status.value!r} requires finished_at_utc")
        return self
