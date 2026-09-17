"""Durable per-job storage with confinement, locking and crash-safe writes.

Every job lives in its own directory under a configurable root. The store is the
only component that touches the filesystem, so the pipeline can stay a pure
orchestrator. Guarantees:

* **Confinement.** Job IDs and relative artifact locators are validated; resolved
  paths must stay inside the job directory, so traversal (``..``), absolute
  paths and symlink escape are rejected.
* **Single writer.** An advisory file lock is held for the store's lifetime and
  released automatically by the OS on exit or crash (``fcntl.flock`` on POSIX,
  a byte-range ``msvcrt`` lock on Windows). A second writer fails
  with :class:`JobLockedError` instead of corrupting state.
* **Atomic writes.** Every JSON body is written to a same-directory temporary
  file, ``fsync``-ed, atomically replaced and the directory is synced where the
  platform supports it. A failed write leaves the previous valid artifact intact.
* **Append-only raw bodies.** :meth:`JobStore.archive_raw` writes a uniquely
  named file, never overwrites an existing body and preserves the exact bytes,
  including non-JSON error bodies. Filenames never embed untrusted remote IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from subtitle_flow import platform_compat
from subtitle_flow.config import (
    CONFIG_VERSION,
    MONEY_MAX_ABSOLUTE_EXPONENT,
    MONEY_MAX_DECIMAL_PRECISION,
    PipelineConfig,
)
from subtitle_flow.media import AudioInfo
from subtitle_flow.schemas import (
    SCHEMA_VERSION,
    JobInput,
    JobRecord,
    RawArtifactRef,
    StageError,
    StageKind,
    Transcript,
    Translation,
    UtcDatetime,
)

__all__ = [
    "F_API_TRACE",
    "F_API_TRACE_HEAD",
    "F_ATTEMPTS",
    "F_EXTRACTION_MANIFEST",
    "F_INPUT",
    "F_INTENTS",
    "F_LOCK",
    "F_MT_MANIFEST",
    "F_RAW_INDEX",
    "F_STATUS",
    "F_STT_MANIFEST",
    "F_TRANSCRIPT",
    "F_TRANSLATION",
    "F_YOUTUBE_MANIFEST",
    "G_MT_GROUP_DIR",
    "ApiLedgerHead",
    "ApiTraceKind",
    "ApiTraceRecord",
    "AttemptIntent",
    "AttemptOutcome",
    "AttemptRecord",
    "MTGroupRecord",
    "MTGroupStatus",
    "MTManifest",
    "RELEASE_PROOF_OUTCOME",
    "RawArchiveRecord",
    "STTManifest",
    "JobExistsError",
    "JobLockedError",
    "JobMissingError",
    "JobStore",
    "PathSecurityError",
    "StorageCorruptionError",
    "StorageError",
    "StoredArtifact",
    "StoredInput",
    "exact_decimal_sum",
    "outcome_proves_predispatch",
]

STORAGE_VERSION: Final[str] = "1"

F_INPUT: Final[str] = "input.json"
F_STATUS: Final[str] = "status.json"
F_LOCK: Final[str] = ".lock"
F_STT_MANIFEST: Final[str] = "manifests/stt.manifest.json"
F_MT_MANIFEST: Final[str] = "manifests/mt.manifest.json"
F_TRANSCRIPT: Final[str] = "artifacts/transcript.source.json"
F_TRANSLATION: Final[str] = "artifacts/translation.tr.json"
F_ATTEMPTS: Final[str] = "manifests/attempts.jsonl"
F_INTENTS: Final[str] = "manifests/intents.jsonl"
F_RAW_INDEX: Final[str] = "raw/index.jsonl"
F_API_TRACE: Final[str] = "manifests/api.trace.jsonl"
F_API_TRACE_HEAD: Final[str] = "manifests/api.trace.head.json"
F_EXTRACTION_MANIFEST: Final[str] = "manifests/extraction.manifest.json"
F_YOUTUBE_MANIFEST: Final[str] = "manifests/youtube.manifest.json"
G_MT_GROUP_DIR: Final[str] = "artifacts/mt.groups"
G_ORPHAN_DIR: Final[str] = "artifacts/orphans"
D_RAW: Final[str] = "raw"

_JOB_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SAFE_FRAGMENT_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_-]+")

_FROZEN = ConfigDict(extra="forbid", frozen=True)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class StorageError(ValueError):
    """Base class for durable-store failures."""


class JobExistsError(StorageError):
    pass


class JobMissingError(StorageError):
    pass


class JobLockedError(StorageError):
    pass


class PathSecurityError(StorageError):
    pass


class StorageCorruptionError(StorageError):
    """A durable artifact or manifest does not match its recorded hash."""

    def __init__(self, message: str, *, code: str = "STORAGE_CORRUPTION") -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class StoredArtifact(BaseModel):
    """A normalized, persisted JSON artifact with integrity metadata."""

    model_config = _FROZEN

    locator: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, ge=0)


class StoredInput(BaseModel):
    """Contents of ``input.json``: the immutable job snapshot.

    Every duplicated fact is cross-checked against its source of truth when the
    snapshot is built or reloaded. The stored ``config`` body must reproduce the
    recorded ``config_fingerprint`` and ``job_fingerprint``; the ``input`` record
    must agree with the verified ``audio`` facts and with the config's provider,
    language, keyterm, schema and segmenter identities. A single-field edit of
    any duplicated value (duration, path, hash, format, codec, rate, channels,
    size, version, languages) therefore stops resume before any provider call.
    """

    model_config = _FROZEN

    storage_version: str = STORAGE_VERSION
    schema_version: str = SCHEMA_VERSION
    input: JobInput
    config: PipelineConfig
    audio: AudioInfo
    config_fingerprint: str = Field(min_length=1)
    job_fingerprint: str = Field(min_length=1)
    stored_at_utc: UtcDatetime

    @field_validator("storage_version")
    @classmethod
    def _supported_storage_version(cls, value: str) -> str:
        if value != STORAGE_VERSION:
            raise ValueError(
                f"unsupported storage_version {value!r}; this build reads and "
                f"writes {STORAGE_VERSION!r}"
            )
        return value

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value!r}; this build reads and "
                f"writes {SCHEMA_VERSION!r}"
            )
        return value

    @field_validator("input")
    @classmethod
    def _check_input(cls, value: JobInput) -> JobInput:
        if value.job_id == "":
            raise ValueError("input.job_id must not be empty")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "StoredInput":
        issues: list[str] = []
        config = self.config
        job_input = self.input
        audio = self.audio

        if job_input.schema_version != config.schema_version:
            issues.append(
                f"input.schema_version {job_input.schema_version!r} != "
                f"config.schema_version {config.schema_version!r}"
            )
        if job_input.stt != config.stt:
            issues.append("input.stt identity != config.stt")
        if job_input.mt != config.mt:
            issues.append("input.mt identity != config.mt")
        if job_input.target_language != config.target_language:
            issues.append("input.target_language != config.target_language")
        if job_input.source_language != config.source_language_hint:
            issues.append("input.source_language != config.source_language_hint")
        if job_input.keyterms != config.keyterms:
            issues.append("input.keyterms != config.keyterms")
        if job_input.keyterms_version != config.keyterms_version:
            issues.append("input.keyterms_version != config.keyterms_version")
        if job_input.segmenter_version != config.segmenter_version:
            issues.append("input.segmenter_version != config.segmenter_version")

        if job_input.audio_path != audio.path:
            issues.append("input.audio_path != audio.path")
        if job_input.audio_sha256 != audio.audio_sha256:
            issues.append("input.audio_sha256 != audio.audio_sha256")
        if job_input.audio_duration_ms != audio.duration_ms:
            issues.append("input.audio_duration_ms != audio.duration_ms")
        if job_input.audio_format != audio.container:
            issues.append("input.audio_format != audio.container")
        if job_input.audio_codec != audio.codec_name:
            issues.append("input.audio_codec != audio.codec_name")
        if job_input.audio_channels != audio.channels:
            issues.append("input.audio_channels != audio.channels")
        if job_input.audio_sample_rate != audio.sample_rate:
            issues.append("input.audio_sample_rate != audio.sample_rate")
        if job_input.audio_size_bytes != audio.size_bytes:
            issues.append("input.audio_size_bytes != audio.size_bytes")
        if job_input.original_filename != audio.original_filename:
            issues.append("input.original_filename != audio.original_filename")
        # Video provenance is additive: an audio-only job must not carry a video
        # identity, and a video job's recorded source must match its frozen
        # origin exactly.
        origin = config.video_origin
        if origin is not None:
            if job_input.video_path is None or job_input.video_sha256 is None:
                issues.append(
                    "config.video_origin requires input.video_path and "
                    "input.video_sha256"
                )
            else:
                if job_input.video_path != origin.source_path:
                    issues.append("input.video_path != config.video_origin.source_path")
                if job_input.video_sha256 != origin.source_sha256:
                    issues.append(
                        "input.video_sha256 != config.video_origin.source_sha256"
                    )
        elif job_input.video_path is not None or job_input.video_sha256 is not None:
            issues.append("input.video_* is set without a config.video_origin")
        # YouTube provenance is additive and independently typed: a YouTube job's
        # canonical artifact hash/size must match its frozen YouTube origin, and a
        # local-video origin can never coexist with it.
        youtube = config.youtube_origin
        if youtube is not None:
            if origin is not None:
                issues.append("config.youtube_origin and config.video_origin coexist")
            if youtube.canonical_audio_sha256 != audio.audio_sha256:
                issues.append(
                    "config.youtube_origin canonical hash != audio.audio_sha256"
                )
            if youtube.canonical_audio_size_bytes != audio.size_bytes:
                issues.append(
                    "config.youtube_origin canonical size != audio.size_bytes"
                )
        # ``media.AudioInfo.original_filename`` is documented as the file's base
        # name (``Path(path).name``), so a stored pair that disagrees is a
        # single-field tamper of a duplicated fact.
        if os.path.basename(audio.path) != audio.original_filename:
            issues.append("audio.original_filename != basename(audio.path)")
        if config.config_version != CONFIG_VERSION:
            issues.append(
                f"config.config_version {config.config_version!r} is not "
                f"{CONFIG_VERSION!r}"
            )
        if self.config_fingerprint != config.config_fingerprint():
            issues.append("config_fingerprint does not match the stored config body")
        if self.job_fingerprint != config.job_fingerprint(audio.audio_sha256):
            issues.append("job_fingerprint does not match config + audio hash")

        if issues:
            raise ValueError(
                "stored job snapshot is inconsistent: " + "; ".join(issues)
            )
        return self


class MTGroupStatus(StrEnum):
    running = "running"
    complete = "complete"
    failed = "failed"


class RawArchiveRecord(BaseModel):
    """Durable registration binding one append-only raw body to its origin.

    Written by :meth:`JobStore.archive_raw` itself, never by an adapter, so a
    reference fabricated by a provider (or pointing at ``input.json`` /
    ``status.json`` / another stage's body) can never satisfy
    :meth:`JobStore.verify_raw`. ``group_id``/``attempt`` are supplied by the
    pipeline's archive context and bind the body to the exact dispatch.
    """

    model_config = _FROZEN

    stage: StageKind
    kind: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, ge=0)
    request_id: str | None = None
    group_id: str | None = None
    attempt: int | None = Field(default=None, strict=True, ge=1)
    content_subtype: str = Field(min_length=1)
    archived_at_utc: UtcDatetime


class AttemptOutcome(StrEnum):
    complete = "complete"
    failed = "failed"
    remote_unknown = "remote_unknown"
    skipped = "skipped"


class AttemptRecord(BaseModel):
    """One append-only provider dispatch attempt for audit and Phase 2B budgeting.

    ``finished_at_utc`` and ``elapsed_ms`` are the measured values; ``error``
    and ``replay_reason`` preserve why a call was made or refused. Records are
    never rewritten, so earlier request ids, raw references and failures remain
    auditable even after the convenience manifest advances.
    """

    model_config = _FROZEN

    stage: StageKind
    group_id: str | None = None
    attempt: int = Field(strict=True, ge=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_id: str | None = None
    raw_reference: RawArtifactRef | None = None
    source_char_count: int = Field(strict=True, ge=0)
    target_char_count: int = Field(strict=True, ge=0)
    started_at_utc: UtcDatetime
    finished_at_utc: UtcDatetime
    elapsed_ms: int = Field(strict=True, ge=0)
    outcome: AttemptOutcome
    error: StageError | None = None
    replay_reason: str | None = None
    recorded_at_utc: UtcDatetime

    @model_validator(mode="after")
    def _check_attempt(self) -> "AttemptRecord":
        if self.finished_at_utc < self.started_at_utc:
            raise ValueError("finished_at_utc must not precede started_at_utc")
        if self.outcome in {AttemptOutcome.failed, AttemptOutcome.remote_unknown}:
            if self.error is None:
                raise ValueError(f"{self.outcome.value} attempt requires an error record")
        elif self.error is not None:
            raise ValueError("a successful or skipped attempt must not carry an error")
        return self


class AttemptIntent(BaseModel):
    """Append-only pre-dispatch intent for one provider attempt.

    Written **before** an adapter is called and independently of the mutable
    convenience status/manifest, so a hard kill can never erase the fact that an
    attempt number was allocated. There is deliberately no ``finished_at_utc`` or
    ``elapsed_ms`` here: a killed attempt's timing stays explicitly unknown until
    a matching :class:`AttemptRecord` outcome is appended. Explicit replay appends
    a new intent with the next number instead of rewriting history.
    """

    model_config = _FROZEN

    stage: StageKind
    group_id: str | None = None
    attempt: int = Field(strict=True, ge=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    request_id: str | None = None
    replay_reason: str | None = None
    started_at_utc: UtcDatetime
    declared_at_utc: UtcDatetime

    @model_validator(mode="after")
    def _check_intent(self) -> "AttemptIntent":
        if self.declared_at_utc < self.started_at_utc:
            raise ValueError("declared_at_utc must not precede started_at_utc")
        return self



class ApiTraceKind(StrEnum):
    """Append-only API attempt/budget trace record kinds."""

    reserve = "reserve"
    release = "release"
    outcome = "outcome"


#: Hash chained into the first ledger record as its predecessor.
LEDGER_GENESIS_HASH: Final[str] = "0" * 64

#: Significant digits an accepted monetary value may carry. More is rejected
#: explicitly instead of being silently rounded near a cap.
LEDGER_MAX_DECIMAL_PRECISION: Final[int] = MONEY_MAX_DECIMAL_PRECISION

#: An accepted monetary value's base-10 exponent must stay within this bound so
#: exact additions cannot require an unbounded amount of work. A value outside
#: the bound is rejected at validation time, never silently rounded.
LEDGER_MAX_ABSOLUTE_EXPONENT: Final[int] = MONEY_MAX_ABSOLUTE_EXPONENT

#: The only outcome label that proves a paid attempt never reached the provider.
#: A ``release`` is accepted solely when its anchoring outcome carries this label
#: with no HTTP status and no archived raw body.
RELEASE_PROOF_OUTCOME: Final[str] = "predispatch_error"


def _significant_digit_count(value: Decimal) -> int:
    return len(value.as_tuple().digits)


def _bounded_exponent(value: Decimal) -> bool:
    return abs(int(value.as_tuple().exponent)) <= LEDGER_MAX_ABSOLUTE_EXPONENT


def _canonical_decimal_string(value: str) -> str:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"amount must be a decimal string: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("amount must be a finite non-negative decimal")
    _sign, digits, _exponent = parsed.as_tuple()
    if len(digits) > LEDGER_MAX_DECIMAL_PRECISION:
        raise ValueError(
            f"amount carries more than {LEDGER_MAX_DECIMAL_PRECISION} significant "
            "digits; refusing an unrepresentable monetary value"
        )
    if not _bounded_exponent(parsed):
        raise ValueError(
            f"amount exponent exceeds the accepted +/-{LEDGER_MAX_ABSOLUTE_EXPONENT} "
            "bound; refusing an unrepresentable monetary value"
        )
    return str(parsed)


def exact_decimal_sum(values: Iterable[Decimal]) -> Decimal:
    """Add finite decimals exactly, independent of the global Decimal context.

    ``Decimal.__add__`` rounds to the active context precision, so a sequence of
    ostensibly valid 28-significant-digit caps can sum to a value whose rounding
    crosses a budget cap (for example three times
    ``0.3333333333333333333333333334`` sums to
    ``1.0000000000000000000000000002``, not ``1``). This helper scales every
    term to a common base-10 exponent and sums the exact integer coefficients, so
    accepted reservations and comparisons never silently round.
    """

    scaled: list[tuple[int, int]] = []
    minimum_exponent: int | None = None
    for value in values:
        if not value.is_finite():
            raise ValueError("monetary arithmetic requires finite decimals")
        sign, digits, exponent = value.as_tuple()
        coefficient = 0
        for digit in digits:
            coefficient = coefficient * 10 + digit
        if sign:
            coefficient = -coefficient
        scaled.append((coefficient, int(exponent)))
        minimum_exponent = (
            int(exponent)
            if minimum_exponent is None
            else min(minimum_exponent, int(exponent))
        )
    if minimum_exponent is None:
        return Decimal(0)
    total = 0
    for coefficient, exponent in scaled:
        total += coefficient * (10 ** (exponent - minimum_exponent))
    if total == 0:
        return Decimal((0, (0,), minimum_exponent))
    digits = tuple(int(character) for character in str(abs(total)))
    return Decimal((1 if total < 0 else 0, digits, minimum_exponent))


def outcome_proves_predispatch(record: "ApiTraceRecord") -> bool:
    """Whether an outcome record proves no remote work happened for an attempt."""

    return (
        record.kind is ApiTraceKind.outcome
        and record.outcome == RELEASE_PROOF_OUTCOME
        and record.remote_status_unknown is not True
        and record.http_status is None
        and record.raw_reference is None
    )


class ApiTraceRecord(BaseModel):
    """One append-only HTTP attempt / budget event for real adapters.

    ``reserve`` and ``release`` records carry the reserved amount as a canonical
    decimal string, so remaining budget is derived purely from this durable,
    append-only history and survives new adapter objects and process resumes. The
    ledger is a hash chain: every record carries its ``seq``, the previous
    record's ``prev_hash`` and its own ``record_hash``; :class:`JobStore` also
    keeps an independently written head anchor. A single line edit, deletion or
    truncation therefore fails the chain/anchor check and halts new sends.
    """

    model_config = _FROZEN

    #: Assigned by :meth:`JobStore.append_api_trace`; callers leave them unset.
    seq: int | None = Field(default=None, strict=True, ge=0)
    prev_hash: str | None = None
    record_hash: str | None = None

    record_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    kind: ApiTraceKind
    stage: StageKind
    group_id: str | None = None
    pipeline_attempt: int | None = Field(default=None, strict=True, ge=1)
    http_attempt: int = Field(strict=True, ge=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    amount: str | None = None
    currency: str | None = None
    outcome: str | None = None
    http_status: int | None = Field(default=None, strict=True, ge=0)
    remote_status_unknown: bool | None = None
    request_id: str | None = None
    raw_reference: RawArtifactRef | None = None
    response_headers: dict[str, str] = Field(default_factory=dict)
    retry_after_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    error_code: str | None = None
    error_message: str | None = None
    elapsed_ms: int | None = Field(default=None, strict=True, ge=0)
    recorded_at_utc: UtcDatetime

    @field_validator("amount")
    @classmethod
    def _check_amount(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_decimal_string(value)

    @model_validator(mode="after")
    def _check_kind(self) -> "ApiTraceRecord":
        if self.kind in {ApiTraceKind.reserve, ApiTraceKind.release}:
            if self.amount is None or self.currency is None:
                raise ValueError(
                    f"{self.kind.value} record requires amount and currency"
                )
        return self


class ApiLedgerHead(BaseModel):
    """Atomic anchor for the append-only API ledger: count + last record hash."""

    model_config = _FROZEN

    ledger_version: str = Field(default="1", min_length=1)
    count: int = Field(strict=True, ge=0)
    head_hash: str = Field(min_length=1)
    updated_at_utc: UtcDatetime


def _ledger_record_hash(
    record: ApiTraceRecord, *, seq: int, prev_hash: str
) -> str:
    payload = record.model_dump(mode="json", exclude={"record_hash"})
    payload["seq"] = seq
    payload["prev_hash"] = prev_hash
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MTGroupRecord(BaseModel):
    """Per-group MT state. ``running`` is durable dispatch evidence.

    ``remote_may_have_run`` is true once bytes may have reached the provider
    (in-flight, response received but invalid, or unknown outcome).
    ``auto_resumable`` is true only for a pre-dispatch retryable failure, which
    an ordinary resume may retry; every other failure stays pending review until
    an explicit ``allow_remote_retry`` is recorded.
    """

    model_config = _FROZEN

    group_id: str = Field(min_length=1)
    segment_ids: tuple[str, ...] = ()
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    status: MTGroupStatus
    attempt: int = Field(strict=True, ge=1)
    raw_reference: RawArtifactRef | None = None
    output: StoredArtifact | None = None
    error: StageError | None = None
    source_char_count: int = Field(strict=True, ge=0)
    target_char_count: int = Field(strict=True, ge=0)
    started_at_utc: UtcDatetime
    finished_at_utc: UtcDatetime | None = None
    remote_may_have_run: bool = False
    auto_resumable: bool = False
    replay_reason: str | None = None

    @field_validator("segment_ids")
    @classmethod
    def _unique_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("group segment_ids must be unique")
        return value


class STTManifest(BaseModel):
    """Durable proof that a normalized transcript and its raw body exist."""

    model_config = _FROZEN

    storage_version: str = STORAGE_VERSION
    schema_version: str = SCHEMA_VERSION
    stage: Literal["stt"] = "stt"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_fingerprint: str = Field(min_length=1)
    job_fingerprint: str = Field(min_length=1)
    transcript: StoredArtifact
    raw_reference: RawArtifactRef
    completed_at_utc: UtcDatetime
    #: Pipeline attempt number whose raw body the reference binds. Defaults to 1
    #: so pre-intent snapshots stay readable; every new manifest records it.
    attempt: int = Field(default=1, strict=True, ge=1)

    @field_validator("storage_version")
    @classmethod
    def _supported_storage_version(cls, value: str) -> str:
        if value != STORAGE_VERSION:
            raise ValueError(
                f"unsupported storage_version {value!r}; this build reads "
                f"{STORAGE_VERSION!r}"
            )
        return value

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value!r}; this build reads "
                f"{SCHEMA_VERSION!r}"
            )
        return value


class MTManifest(BaseModel):
    """Durable per-group MT state enabling resume without re-billing."""

    model_config = _FROZEN

    storage_version: str = STORAGE_VERSION
    schema_version: str = SCHEMA_VERSION
    stage: Literal["mt"] = "mt"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_fingerprint: str = Field(min_length=1)
    job_fingerprint: str = Field(min_length=1)
    status: Literal["running", "complete", "partial", "failed", "skipped"] = "running"
    groups: tuple[MTGroupRecord, ...] = ()
    translation: StoredArtifact | None = None
    updated_at_utc: UtcDatetime

    @field_validator("groups")
    @classmethod
    def _unique_groups(cls, value: tuple[MTGroupRecord, ...]) -> tuple[MTGroupRecord, ...]:
        ids = [group.group_id for group in value]
        if len(set(ids)) != len(ids):
            raise ValueError("group_id values must be unique within an MT manifest")
        return value

    @field_validator("storage_version")
    @classmethod
    def _supported_storage_version(cls, value: str) -> str:
        if value != STORAGE_VERSION:
            raise ValueError(
                f"unsupported storage_version {value!r}; this build reads "
                f"{STORAGE_VERSION!r}"
            )
        return value

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value!r}; this build reads "
                f"{SCHEMA_VERSION!r}"
            )
        return value


def _safe_fragment(value: str, *, limit: int = 24) -> str:
    cleaned = _SAFE_FRAGMENT_RE.sub("-", value).strip("-")
    return cleaned[:limit] or "x"


class JobStore:
    """Filesystem-backed store for exactly one job directory.

    Typical lifecycle::

        store = JobStore(root, "job_...")
        store.create(stored_input)          # first time only
        with store:                          # acquires the exclusive lock
            store.write_status(record)
            ref = store.archive_raw(StageKind.stt, payload, request_id="req-1")
    """

    def __init__(self, root: str | Path, job_id: str) -> None:
        if not isinstance(job_id, str) or not _JOB_ID_RE.match(job_id):
            raise StorageError(
                f"invalid job_id {job_id!r}; expected 1-64 chars of [A-Za-z0-9._-] "
                "starting with an alphanumeric"
            )
        if ".." in job_id or job_id in {".", ".."}:
            raise PathSecurityError(f"job_id must not contain traversal: {job_id!r}")
        # A user-supplied *root* symlink is resolved explicitly once; the job
        # directory itself must never be a symlink that could move the
        # confinement base outside the configured root.
        self.root = Path(root).resolve()
        self.job_id = job_id
        self._lock_fd: int | None = None

    # ------------------------------------------------------------------ #
    # Paths and confinement
    # ------------------------------------------------------------------ #
    @property
    def job_dir(self) -> Path:
        return self.root / self.job_id

    def _safe_relative(self, relative: str) -> str:
        if not isinstance(relative, str) or relative.strip() == "":
            raise PathSecurityError("artifact locator must be a non-empty string")
        if os.path.isabs(relative) or relative.startswith(("~", "/")):
            raise PathSecurityError(f"absolute locator is not allowed: {relative!r}")
        parts = PurePosixPath(relative).parts
        if any(part in {"..", ""} for part in parts) or "." in parts:
            raise PathSecurityError(f"locator contains traversal: {relative!r}")
        return str(PurePosixPath(*parts))

    def _reject_symlinked_job_dir(self) -> None:
        if os.path.islink(self.job_dir):
            raise PathSecurityError(
                f"job directory is a symlink and is not accepted as a confinement "
                f"base: {self.job_dir}"
            )

    def _confined_path(self, relative: str) -> Path:
        """Resolve a job-relative locator, rejecting symlinks and escapes.

        Every component between the job directory and the target is checked so a
        symlinked file or intermediate directory cannot redirect a read or write
        outside the root. The returned path is the canonical, confined path.
        """

        safe = self._safe_relative(relative)
        self._reject_symlinked_job_dir()
        unresolved = self.job_dir / safe
        current = self.job_dir
        for part in PurePosixPath(safe).parts:
            current = current / part
            if os.path.islink(current):
                raise PathSecurityError(
                    f"refusing symlinked path component: {relative!r}"
                )
        resolved = unresolved.resolve()
        base = self.job_dir.resolve()
        if resolved != base and base not in resolved.parents:
            raise PathSecurityError(
                f"resolved path escapes the job directory: {relative!r}"
            )
        return resolved

    def resolve(self, relative: str) -> Path:
        """Return a confined absolute path for a job-relative locator."""

        return self._confined_path(relative)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def exists(self) -> bool:
        return self._confined_path(F_INPUT).is_file()

    def create(self, stored_input: StoredInput) -> "JobStore":
        """Create the job directory and write ``input.json`` exactly once."""

        if stored_input.input.job_id != self.job_id:
            raise StorageError("stored input job_id does not match this store")
        self._reject_symlinked_job_dir()
        try:
            self.job_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise JobExistsError(f"job already exists: {self.job_id}") from exc
        # Re-check after creation so a symlink raced in place cannot become the
        # confinement base for the snapshot write.
        self._reject_symlinked_job_dir()
        self._atomic_write_bytes(
            F_INPUT,
            stored_input.model_dump_json(indent=2).encode("utf-8"),
            allow_missing_lock=True,
        )
        return self

    def acquire_lock(self) -> "JobStore":
        """Acquire the exclusive writer lock or fail with ``JobLockedError``."""

        if self._lock_fd is not None:
            return self
        lock_path = self._confined_path(F_LOCK)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise StorageError(f"cannot open lock file for {self.job_id}: {exc}") from exc
        try:
            platform_compat.lock_fd_exclusive_nonblocking(fd)
        except OSError as exc:
            os.close(fd)
            raise JobLockedError(
                f"job {self.job_id} is already locked by another writer"
            ) from exc
        self._lock_fd = fd
        return self

    def release_lock(self) -> None:
        if self._lock_fd is None:
            return
        try:
            platform_compat.unlock_fd(self._lock_fd)
        finally:
            os.close(self._lock_fd)
            self._lock_fd = None

    @property
    def locked(self) -> bool:
        return self._lock_fd is not None

    def __enter__(self) -> "JobStore":
        return self.acquire_lock()

    def __exit__(self, *_exc: object) -> None:
        self.release_lock()

    def close(self) -> None:
        self.release_lock()

    def _require_lock(self) -> None:
        if self._lock_fd is None:
            raise StorageError(
                "the job store must hold its writer lock before mutating files"
            )

    # ------------------------------------------------------------------ #
    # Atomic IO
    # ------------------------------------------------------------------ #
    def _fsync_dir(self, directory: Path) -> None:
        try:
            fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _atomic_write_bytes(
        self, relative: str, data: bytes, *, allow_missing_lock: bool = False
    ) -> None:
        if not allow_missing_lock:
            self._require_lock()
        target = self.resolve(relative)
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Re-check confinement after directory creation so a symlinked parent
        # cannot redirect the write outside the job directory.
        base = self.job_dir.resolve()
        if parent.resolve() != base and base not in parent.resolve().parents:
            raise PathSecurityError(f"artifact parent escapes the job directory: {relative!r}")
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            platform_compat.replace_file_atomically(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._fsync_dir(parent)

    def write_artifact(self, relative: str, data: bytes) -> StoredArtifact:
        """Atomically write a normalized artifact and return its integrity ref."""

        self._atomic_write_bytes(relative, data)
        digest = hashlib.sha256(data).hexdigest()
        return StoredArtifact(
            locator=self._safe_relative(relative),
            sha256=digest,
            size_bytes=len(data),
        )

    def adopt_file(self, relative: str, source: str | Path) -> StoredArtifact:
        """Atomically publish an already-validated file into a confined locator.

        Used to move a deterministic extraction temporary file (created in the
        same job directory) into its final artifact name without ever copying or
        rewriting the bytes. The source must be a regular, non-symlink file; the
        destination is confined and replaced atomically. The returned artifact
        carries the hash measured from the exact bytes moved.
        """

        self._require_lock()
        source_path = Path(source)
        if os.path.islink(source_path) or not os.path.isfile(source_path):
            raise PathSecurityError(
                "adopted file must be a regular non-symlink file"
            )
        target = self.resolve(relative)
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        base = self.job_dir.resolve()
        if parent.resolve() != base and base not in parent.resolve().parents:
            raise PathSecurityError(
                f"artifact parent escapes the job directory: {relative!r}"
            )
        try:
            if os.stat(source_path).st_dev != os.stat(parent).st_dev:
                raise StorageError(
                    "adopted file and destination are on different filesystems; "
                    "refusing a non-atomic publish"
                )
        except OSError as exc:
            raise StorageError(f"cannot stat the adopted file: {exc}") from exc
        digest = hashlib.sha256()
        size = 0
        with open(source_path, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        platform_compat.replace_file_atomically(source_path, target)
        self._fsync_dir(parent)
        return StoredArtifact(
            locator=self._safe_relative(relative),
            sha256=digest.hexdigest(),
            size_bytes=size,
        )

    def read_bytes(self, relative: str) -> bytes:
        return self.resolve(relative).read_bytes()

    def verify_artifact(self, artifact: StoredArtifact) -> bytes:
        """Read an artifact and fail if its bytes no longer match its hash."""

        data = self.read_bytes(artifact.locator)
        if len(data) != artifact.size_bytes:
            raise StorageCorruptionError(
                f"artifact size changed: {artifact.locator} "
                f"(expected {artifact.size_bytes}, found {len(data)})"
            )
        if hashlib.sha256(data).hexdigest() != artifact.sha256:
            raise StorageCorruptionError(
                f"artifact hash mismatch: {artifact.locator}"
            )
        return data

    # ------------------------------------------------------------------ #
    # Append-only line stores (raw index, attempt history)
    # ------------------------------------------------------------------ #
    def _append_jsonl(self, relative: str, model: BaseModel) -> None:
        """Append one JSON line durably. Never rewrites prior lines."""

        self._require_lock()
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        base = self.job_dir.resolve()
        parent = target.parent.resolve()
        if parent != base and base not in parent.parents:
            raise PathSecurityError(f"line store parent escapes the job directory: {relative!r}")
        line = model.model_dump_json().encode("utf-8") + b"\n"
        flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(target, flags, 0o600)
        try:
            written = 0
            while written < len(line):
                wrote = os.write(fd, line[written:])
                if wrote <= 0:
                    raise StorageError(
                        f"line store {relative!r} made no write progress; refusing "
                        "a partial durable registration"
                    )
                written += wrote
            os.fsync(fd)
        finally:
            os.close(fd)
        self._fsync_dir(target.parent)

    def _read_jsonl(self, relative: str, model: type[_ModelT]) -> list[_ModelT]:
        path = self.resolve(relative)
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageCorruptionError(
                f"line store is unreadable: {relative}: {exc}"
            ) from exc
        records: list[_ModelT] = []
        for number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(model.model_validate_json(stripped))
            except ValueError as exc:
                raise StorageCorruptionError(
                    f"line store {relative!r} line {number} is invalid: {exc}"
                ) from exc
        return records

    def append_attempt(self, record: AttemptRecord) -> None:
        """Append one provider attempt to the durable, append-only history."""

        self._append_jsonl(F_ATTEMPTS, record)

    def read_attempts(
        self, *, stage: StageKind | None = None
    ) -> tuple[AttemptRecord, ...]:
        """Return the append-only attempt history, newest last."""

        records = self._read_jsonl(F_ATTEMPTS, AttemptRecord)
        if stage is None:
            return tuple(records)
        return tuple(record for record in records if record.stage is stage)

    def append_attempt_intent(self, intent: AttemptIntent) -> None:
        """Append one pre-dispatch attempt intent (durable before any call)."""

        self._append_jsonl(F_INTENTS, intent)

    def read_attempt_intents(
        self, *, stage: StageKind | None = None
    ) -> tuple[AttemptIntent, ...]:
        """Return the append-only pre-dispatch intents, newest last.

        This is the durable source of truth for which attempt numbers were
        allocated before a provider call, even when the call produced no outcome
        (hard kill). ``read_attempts`` remains the reader for final outcomes.
        """

        records = self._read_jsonl(F_INTENTS, AttemptIntent)
        if stage is None:
            return tuple(records)
        return tuple(record for record in records if record.stage is stage)

    # ------------------------------------------------------------------ #
    # Durable paid-call ledger (hash chain + atomic head anchor)
    # ------------------------------------------------------------------ #
    def _read_api_head(self) -> ApiLedgerHead:
        path = self.resolve(F_API_TRACE_HEAD)
        try:
            return ApiLedgerHead.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise StorageCorruptionError(
                f"api ledger head is unreadable or invalid: {F_API_TRACE_HEAD}: {exc}"
            ) from exc

    def api_ledger_initialized(self) -> bool:
        """Whether the durable API ledger/head files exist at all."""

        return (
            self.resolve(F_API_TRACE).is_file()
            or self.resolve(F_API_TRACE_HEAD).is_file()
        )

    def initialize_api_ledger(self) -> None:
        """Create the head anchor for a genuinely fresh job exactly once.

        Raises when only one of the two files exists, so a partially deleted
        ledger is never silently reinitialized and a reset budget is refused.
        """

        ledger_exists = self.resolve(F_API_TRACE).is_file()
        head_exists = self.resolve(F_API_TRACE_HEAD).is_file()
        if ledger_exists or head_exists:
            if not (ledger_exists and head_exists):
                raise StorageCorruptionError(
                    "api ledger is partially present (ledger/head mismatch); "
                    "refusing to initialize a fresh budget"
                )
            return
        self._atomic_write_bytes(
            F_API_TRACE_HEAD,
            ApiLedgerHead(
                count=0,
                head_hash=LEDGER_GENESIS_HASH,
                updated_at_utc=utc_now(),
            ).model_dump_json().encode("utf-8"),
        )

    def _load_api_ledger(self) -> tuple[list[ApiTraceRecord], ApiLedgerHead]:
        ledger_exists = self.resolve(F_API_TRACE).is_file()
        head_exists = self.resolve(F_API_TRACE_HEAD).is_file()
        if not ledger_exists and not head_exists:
            return [], ApiLedgerHead(
                count=0,
                head_hash=LEDGER_GENESIS_HASH,
                updated_at_utc=utc_now(),
            )
        if head_exists and not ledger_exists:
            head = self._read_api_head()
            if head.count != 0:
                raise StorageCorruptionError(
                    "api ledger head has records but the ledger file is missing; "
                    "refusing to recreate a budget"
                )
            return [], head
        records = self._read_jsonl(F_API_TRACE, ApiTraceRecord)
        if not head_exists:
            if records:
                raise StorageCorruptionError(
                    "api ledger has records but no head anchor; refusing new sends"
                )
            return [], ApiLedgerHead(
                count=0,
                head_hash=LEDGER_GENESIS_HASH,
                updated_at_utc=utc_now(),
            )
        head = self._read_api_head()
        self._validate_api_ledger(records, head)
        return records, head

    def _validate_api_ledger(
        self, records: list[ApiTraceRecord], head: ApiLedgerHead
    ) -> None:
        if len(records) != head.count:
            raise StorageCorruptionError(
                f"api ledger has {len(records)} records but its head anchor counts "
                f"{head.count}; refusing new sends"
            )
        previous = LEDGER_GENESIS_HASH
        seen_records: set[str] = set()
        reserves: dict[str, ApiTraceRecord] = {}
        outcomes: dict[str, ApiTraceRecord] = {}
        releases: set[str] = set()
        reserve_amounts: list[Decimal] = []
        release_amounts: list[Decimal] = []
        for index, record in enumerate(records):
            if record.seq != index:
                raise StorageCorruptionError(
                    f"api ledger record {index} carries sequence {record.seq!r}; "
                    "refusing new sends"
                )
            if record.prev_hash != previous:
                raise StorageCorruptionError(
                    f"api ledger chain is broken at record {index}; refusing new sends"
                )
            if record.record_hash is None or record.record_hash != _ledger_record_hash(
                record, seq=index, prev_hash=previous
            ):
                raise StorageCorruptionError(
                    f"api ledger record {index} hash does not match its content; "
                    "refusing new sends"
                )
            previous = record.record_hash
            if record.record_id in seen_records:
                raise StorageCorruptionError("api ledger has a duplicate record id")
            seen_records.add(record.record_id)
            if record.kind is ApiTraceKind.reserve:
                if record.attempt_id in reserves:
                    raise StorageCorruptionError(
                        "api ledger has more than one reserve for one attempt"
                    )
                reserves[record.attempt_id] = record
                assert record.amount is not None
                reserve_amounts.append(Decimal(record.amount))
            elif record.kind is ApiTraceKind.outcome:
                base = reserves.get(record.attempt_id)
                if base is None:
                    raise StorageCorruptionError(
                        "api ledger outcome has no matching reservation"
                    )
                if record.attempt_id in outcomes:
                    raise StorageCorruptionError(
                        "api ledger has a duplicate outcome for one attempt"
                    )
                self._check_ledger_identity(base, record, check_currency=False)
                outcomes[record.attempt_id] = record
            else:
                base = reserves.get(record.attempt_id)
                if base is None:
                    raise StorageCorruptionError(
                        "api ledger release has no matching reservation"
                    )
                if record.attempt_id in releases:
                    raise StorageCorruptionError(
                        "api ledger has a duplicate release for one attempt"
                    )
                anchoring = outcomes.get(record.attempt_id)
                if anchoring is None:
                    raise StorageCorruptionError(
                        "api ledger release is not anchored by a recorded outcome"
                    )
                if not outcome_proves_predispatch(anchoring):
                    raise StorageCorruptionError(
                        "api ledger release requires a proven pre-dispatch outcome "
                        "with no HTTP status and no archived raw body; refusing to "
                        "erase a possibly-dispatched paid attempt from the budget"
                    )
                self._check_ledger_identity(base, record, check_currency=True)
                assert base.amount is not None and record.amount is not None
                if Decimal(record.amount) > Decimal(base.amount):
                    raise StorageCorruptionError(
                        "api ledger release exceeds its reservation"
                    )
                releases.add(record.attempt_id)
                release_amounts.append(Decimal(record.amount))
        if head.head_hash != previous:
            raise StorageCorruptionError(
                "api ledger head hash does not match the last record; refusing new sends"
            )
        # ``copy_negate`` is exact and context-free; unary minus would round the
        # release to the global Decimal precision before the exact sum, so a
        # valid release could look like a budget overrun (or leave a residual).
        total = exact_decimal_sum(
            reserve_amounts + [amount.copy_negate() for amount in release_amounts]
        )
        if total < 0:
            raise StorageCorruptionError(
                "api ledger releases exceed reservations; refusing new sends"
            )

    @staticmethod
    def _check_ledger_identity(
        reserve: ApiTraceRecord, later: ApiTraceRecord, *, check_currency: bool
    ) -> None:
        fields = [
            "stage",
            "group_id",
            "pipeline_attempt",
            "http_attempt",
            "provider",
            "model",
        ]
        if check_currency:
            fields.append("currency")
        for field in fields:
            if getattr(reserve, field) != getattr(later, field):
                raise StorageCorruptionError(
                    f"api ledger {later.kind.value} {field} does not match its "
                    "reservation; refusing new sends"
                )

    def append_api_trace(self, record: ApiTraceRecord) -> ApiTraceRecord:
        """Append one HTTP attempt / budget event and advance the head anchor."""

        self._require_lock()
        records, head = self._load_api_ledger()
        seq = len(records)
        record_hash = _ledger_record_hash(record, seq=seq, prev_hash=head.head_hash)
        final = record.model_copy(
            update={
                "seq": seq,
                "prev_hash": head.head_hash,
                "record_hash": record_hash,
            }
        )
        self._append_jsonl(F_API_TRACE, final)
        self._atomic_write_bytes(
            F_API_TRACE_HEAD,
            ApiLedgerHead(
                count=seq + 1,
                head_hash=record_hash,
                updated_at_utc=utc_now(),
            ).model_dump_json().encode("utf-8"),
        )
        return final

    def read_api_trace(
        self, *, stage: StageKind | None = None
    ) -> tuple[ApiTraceRecord, ...]:
        """Return the validated append-only API trace, newest last.

        A missing ledger/head for a fresh job returns ``()``. A single edited,
        deleted or truncated record, a partially present ledger, or a head/chain
        mismatch raises :class:`StorageCorruptionError`, so a corrupt budget or
        trace artifact halts new sends instead of resetting a counter.
        """

        records, _head = self._load_api_ledger()
        if stage is None:
            return tuple(records)
        return tuple(record for record in records if record.stage is stage)

    # ------------------------------------------------------------------ #
    # Raw append-only archive
    # ------------------------------------------------------------------ #
    def _next_raw_sequence(self, directory: Path) -> int:
        highest = 0
        if directory.is_dir():
            for entry in directory.iterdir():
                match = re.match(r"^(\d{6})-", entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    def archive_raw(
        self,
        stage: StageKind,
        payload: bytes,
        *,
        request_id: str | None = None,
        group_id: str | None = None,
        attempt: int | None = None,
        content_subtype: str = "json",
    ) -> RawArtifactRef:
        """Append one raw provider body and durably register its provenance.

        The body is written to a new, uniquely named file (never overwritten)
        and a :class:`RawArchiveRecord` binding stage, kind, locator, hash, size,
        request id, group and attempt is appended to ``raw/index.jsonl``. The
        registration is written by this method, never by an adapter, so a
        fabricated :class:`RawArtifactRef` cannot pass :meth:`verify_raw`.

        Args:
            stage: ``stt`` or ``mt``.
            payload: Exact bytes to preserve (may be non-JSON or an error body).
            request_id: Provider request id, recorded only.
            group_id: MT group id; bound into the registration.
            attempt: Attempt number; bound into the registration.
            content_subtype: Filename suffix; ``json`` by default.

        Returns:
            A :class:`RawArtifactRef` pointing at the new, immutable file.
        """

        if not isinstance(payload, bytes):
            raise StorageError("raw payload must be bytes to preserve exact content")
        if not isinstance(stage, StageKind):
            raise StorageError("stage must be a StageKind")
        if attempt is not None and (not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1):
            raise StorageError("attempt must be a positive integer when provided")
        self._require_lock()
        subtype = _safe_fragment(content_subtype, limit=12).lower()
        directory = self.resolve(f"{D_RAW}/{stage.value}")
        directory.mkdir(parents=True, exist_ok=True)
        if self.job_dir.resolve() not in directory.resolve().parents:
            raise PathSecurityError("raw directory escapes the job directory")

        fragment = _safe_fragment(group_id or "body")
        attempt_suffix = f"_a{attempt}" if attempt is not None else ""
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        while True:
            sequence = self._next_raw_sequence(directory)
            token = secrets.token_hex(3)
            filename = f"{sequence:06d}-{fragment}{attempt_suffix}-{token}.{subtype}"
            target = directory / filename
            try:
                fd = os.open(target, flags, 0o600)
            except FileExistsError:
                continue
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                try:
                    os.unlink(target)
                except OSError:
                    pass
                raise
            self._fsync_dir(directory)
            break

        relative = f"{D_RAW}/{stage.value}/{filename}"
        digest = hashlib.sha256(payload).hexdigest()
        registration = RawArchiveRecord(
            stage=stage,
            kind=f"{stage.value}_raw_{subtype}",
            locator=relative,
            sha256=digest,
            size_bytes=len(payload),
            request_id=request_id,
            group_id=group_id,
            attempt=attempt,
            content_subtype=subtype,
            archived_at_utc=utc_now(),
        )
        self._append_jsonl(F_RAW_INDEX, registration)
        return RawArtifactRef(
            kind=registration.kind,
            locator=relative,
            sha256=digest,
            request_id=request_id,
            group_id=group_id,
            attempt=attempt,
        )

    def raw_registrations(
        self, *, stage: StageKind | None = None
    ) -> tuple[RawArchiveRecord, ...]:
        """Return the durable archive registrations, optionally for one stage."""

        records = self._read_jsonl(F_RAW_INDEX, RawArchiveRecord)
        if stage is None:
            return tuple(records)
        return tuple(record for record in records if record.stage is stage)

    def has_raw(self, *, stage: StageKind) -> bool:
        return bool(self.raw_registrations(stage=stage))

    def verify_raw(
        self,
        reference: RawArtifactRef,
        *,
        stage: StageKind | None = None,
        group_id: str | None = None,
        attempt: int | None = None,
    ) -> bytes:
        """Read a raw body and prove its registered provenance.

        Fails when the reference has no hash, is not registered by
        :meth:`archive_raw`, points at another stage, disagrees with the
        registration's kind/request id/hash/size, or does not match the expected
        group/attempt for the current dispatch. A ``None`` ``attempt`` leaves the
        attempt number unconstrained; every pipeline call passes the exact number.
        """

        if not isinstance(reference, RawArtifactRef):
            raise StorageCorruptionError("raw reference is not a RawArtifactRef")
        if reference.locator is None:
            raise StorageCorruptionError("raw reference has no locator")
        if reference.sha256 is None:
            raise StorageCorruptionError(
                f"raw reference {reference.locator!r} has no SHA-256 and cannot be verified"
            )
        registration = next(
            (
                record
                for record in self._read_jsonl(F_RAW_INDEX, RawArchiveRecord)
                if record.locator == reference.locator
            ),
            None,
        )
        if registration is None:
            raise StorageCorruptionError(
                f"raw body {reference.locator!r} is not registered by archive_raw"
            )
        expected_stage = stage if stage is not None else registration.stage
        if registration.stage is not expected_stage:
            raise StorageCorruptionError(
                f"raw body {reference.locator!r} is registered for stage "
                f"{registration.stage.value!r}, not {expected_stage.value!r}"
            )
        if not reference.kind.startswith(f"{expected_stage.value}_raw_"):
            raise StorageCorruptionError(
                f"raw reference kind {reference.kind!r} does not match stage "
                f"{expected_stage.value!r}"
            )
        if not reference.locator.startswith(f"{D_RAW}/{expected_stage.value}/"):
            raise StorageCorruptionError(
                f"raw reference locator {reference.locator!r} is outside the "
                f"{expected_stage.value!r} raw subtree"
            )
        if reference.kind != registration.kind:
            raise StorageCorruptionError(
                f"raw reference kind {reference.kind!r} disagrees with its registration"
            )
        if reference.request_id != registration.request_id:
            raise StorageCorruptionError(
                f"raw reference request id {reference.request_id!r} disagrees with "
                f"its registration {registration.request_id!r}: {reference.locator!r}"
            )
        if reference.sha256 != registration.sha256:
            raise StorageCorruptionError(
                f"raw reference hash disagrees with its registration: {reference.locator!r}"
            )
        # Correlation values carried on the reference must match the durable
        # registration. A reference that still lacks them (a legacy body with no
        # group/attempt) is left to the explicit keyword checks below.
        if (
            reference.group_id is not None
            and reference.group_id != registration.group_id
        ):
            raise StorageCorruptionError(
                f"raw reference group {reference.group_id!r} disagrees with its "
                f"registration {registration.group_id!r}: {reference.locator!r}"
            )
        if reference.attempt is not None and reference.attempt != registration.attempt:
            raise StorageCorruptionError(
                f"raw reference attempt {reference.attempt!r} disagrees with its "
                f"registration {registration.attempt!r}: {reference.locator!r}"
            )
        if registration.group_id != group_id:
            raise StorageCorruptionError(
                f"raw body {reference.locator!r} is bound to group "
                f"{registration.group_id!r}, not {group_id!r}"
            )
        if attempt is not None and registration.attempt != attempt:
            raise StorageCorruptionError(
                f"raw body {reference.locator!r} is bound to attempt "
                f"{registration.attempt!r}, not {attempt!r}"
            )
        data = self.read_bytes(reference.locator)
        if len(data) != registration.size_bytes:
            raise StorageCorruptionError(
                f"raw body size changed: {reference.locator} "
                f"(expected {registration.size_bytes}, found {len(data)})"
            )
        if hashlib.sha256(data).hexdigest() != reference.sha256:
            raise StorageCorruptionError(
                f"raw body hash mismatch: {reference.locator}"
            )
        return data

    # ------------------------------------------------------------------ #
    # Snapshot / status / manifests
    # ------------------------------------------------------------------ #
    def read_input(self) -> StoredInput:
        path = self.resolve(F_INPUT)
        if not path.is_file():
            raise JobMissingError(f"job input not found: {self.job_id}")
        try:
            stored = StoredInput.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise StorageCorruptionError(
                f"job input is unreadable or invalid: {self.job_id}: {exc}"
            ) from exc
        if stored.input.job_id != self.job_id:
            raise StorageCorruptionError(
                f"stored input belongs to job {stored.input.job_id!r}, not "
                f"{self.job_id!r}; refusing a foreign snapshot"
            )
        return stored

    def read_status(self) -> JobRecord | None:
        path = self.resolve(F_STATUS)
        if not path.is_file():
            return None
        try:
            return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise StorageCorruptionError(
                f"job status is unreadable or invalid: {self.job_id}: {exc}"
            ) from exc

    def write_status(self, record: JobRecord) -> None:
        if record.job_id != self.job_id:
            raise StorageError("status job_id does not match this store")
        self._atomic_write_bytes(F_STATUS, record.model_dump_json(indent=2).encode("utf-8"))

    def read_stt_manifest(self) -> STTManifest | None:
        return self._read_manifest(F_STT_MANIFEST, STTManifest)

    def write_stt_manifest(self, manifest: STTManifest) -> None:
        self._atomic_write_bytes(
            F_STT_MANIFEST, manifest.model_dump_json(indent=2).encode("utf-8")
        )

    def read_mt_manifest(self) -> MTManifest | None:
        return self._read_manifest(F_MT_MANIFEST, MTManifest)

    def write_mt_manifest(self, manifest: MTManifest) -> None:
        self._atomic_write_bytes(
            F_MT_MANIFEST, manifest.model_dump_json(indent=2).encode("utf-8")
        )

    def write_extraction_manifest(self, manifest: BaseModel) -> StoredArtifact:
        """Atomically write the extraction manifest (always written last)."""

        return self.write_artifact(
            F_EXTRACTION_MANIFEST, manifest.model_dump_json(indent=2).encode("utf-8")
        )

    def read_extraction_manifest(self, model: type[_ModelT]) -> _ModelT | None:
        return self._read_manifest(F_EXTRACTION_MANIFEST, model)

    def write_youtube_manifest(self, manifest: BaseModel) -> StoredArtifact:
        """Atomically write the YouTube source manifest (always written last)."""

        return self.write_artifact(
            F_YOUTUBE_MANIFEST, manifest.model_dump_json(indent=2).encode("utf-8")
        )

    def read_youtube_manifest(self, model: type[_ModelT]) -> _ModelT | None:
        return self._read_manifest(F_YOUTUBE_MANIFEST, model)

    def _read_manifest(
        self, relative: str, model: type[_ModelT]
    ) -> _ModelT | None:
        path = self.resolve(relative)
        if not path.is_file():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise StorageCorruptionError(
                f"manifest is unreadable or invalid: {relative}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Typed normalized outputs
    # ------------------------------------------------------------------ #
    def write_transcript(self, transcript: Transcript) -> StoredArtifact:
        return self.write_artifact(
            F_TRANSCRIPT, transcript.model_dump_json(indent=2).encode("utf-8")
        )

    def read_transcript(self) -> Transcript | None:
        return self._read_model(F_TRANSCRIPT, Transcript)

    def write_translation(self, translation: Translation) -> StoredArtifact:
        return self.write_artifact(
            F_TRANSLATION, translation.model_dump_json(indent=2).encode("utf-8")
        )

    def read_translation(self) -> Translation | None:
        return self._read_model(F_TRANSLATION, Translation)

    def group_locator(self, group_id: str) -> str:
        return f"{G_MT_GROUP_DIR}/{_safe_fragment(group_id, limit=48)}.json"

    def write_group_translation(
        self, group_id: str, translation: Translation
    ) -> StoredArtifact:
        return self.write_artifact(
            self.group_locator(group_id),
            translation.model_dump_json(indent=2).encode("utf-8"),
        )

    def read_group_translation(self, group_id: str) -> Translation | None:
        return self._read_model(self.group_locator(group_id), Translation)

    def artifact_exists(self, relative: str) -> bool:
        """Return whether a confined artifact file currently exists."""

        return self.resolve(relative).is_file()

    def preserve_orphan(self, relative: str) -> StoredArtifact | None:
        """Copy an unreferenced artifact to a unique append-only name.

        Used before an explicit replay overwrites an unanchored normalized
        output, so the old evidence is never lost. Returns ``None`` when the
        source does not exist.
        """

        source = self.resolve(relative)
        if not source.is_file():
            return None
        data = self.read_bytes(relative)
        stem = _safe_fragment(PurePosixPath(relative).name, limit=40)
        stamp = utc_now().strftime("%Y%m%dT%H%M%S%f")
        token = secrets.token_hex(3)
        target = f"{G_ORPHAN_DIR}/{stamp}-{token}-{stem}"
        return self.write_artifact(target, data)

    def _read_model(self, relative: str, model: type[_ModelT]) -> _ModelT | None:
        path = self.resolve(relative)
        if not path.is_file():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise StorageCorruptionError(
                f"artifact is unreadable or invalid: {relative}: {exc}"
            ) from exc


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)
