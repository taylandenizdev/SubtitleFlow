"""Deterministic alignment, readable exports and honest job metrics.

This module owns the Phase 3 derived outputs. It never rewrites raw text, never
invents a translation and never repairs a source timing. Alignment is a pure
function of the verified transcript and, when present, the verified translation:
segment IDs must match exactly in the original order, and ``source_text`` and
``translation_input`` must be byte-identical, otherwise no success artifact is
written.

Crash safety
------------
All derived files are written atomically through
:class:`~subtitle_flow.storage.JobStore`. A manifest
(``manifests/exports.manifest.json``) is written **last** and records the input
artifact hashes, the quality settings and every derived artifact's hash/size. A
crash between two writes therefore leaves an incomplete set whose manifest does
not match; the next run rebuilds every derived file from the verified durable
transcript/translation without any STT/MT replay and without presenting a
partially updated set as complete.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from subtitle_flow.languages import TARGET_LANGUAGE_CODE
from subtitle_flow.evidence import collect_supplementary_refs
from subtitle_flow.quality import (
    QUALITY_LIMITATIONS,
    QualityInput,
    QualityReport,
    QualitySettings,
    assess_quality,
)
from subtitle_flow.schemas import (
    SCHEMA_VERSION,
    JobRecord,
    RawArtifactRef,
    Segment,
    StageKind,
    StageResult,
    TranslatedSegment,
    Translation,
    TranslationStatus,
    Transcript,
    UtcDatetime,
)
from subtitle_flow.storage import (
    JobStore,
    MTGroupStatus,
    MTManifest,
    PathSecurityError,
    StorageCorruptionError,
    StoredArtifact,
    StoredInput,
    exact_decimal_sum,
    utc_now,
)

__all__ = [
    "EXPORTER_VERSION",
    "F_ALIGNED",
    "F_EXPORTS_MANIFEST",
    "F_METRICS",
    "F_TRANSCRIPT_TXT",
    "F_TRANSLATION_TXT",
    "AlignedArtifact",
    "AlignedSegment",
    "AlignmentError",
    "AlignmentProvenance",
    "ExportError",
    "ExportResult",
    "ExportsManifest",
    "GroupProvenance",
    "JobMetrics",
    "build_alignment",
    "build_metrics",
    "export_job",
    "render_source_text",
    "render_translation_text",
]

EXPORTER_VERSION: Final[str] = "2"

F_TRANSCRIPT_TXT: Final[str] = "artifacts/transcript.source.txt"
F_TRANSLATION_TXT: Final[str] = "artifacts/translation.tr.txt"
F_ALIGNED: Final[str] = "artifacts/aligned.json"
F_METRICS: Final[str] = "artifacts/metrics.json"
F_EXPORTS_MANIFEST: Final[str] = "manifests/exports.manifest.json"

#: The exact derived files an export set must contain, keyed by the manifest
#: artifact key. The translation text file is required only when a translation
#: is included. A manifest that maps a key to another locator is not reused.
_REQUIRED_DERIVED: Final[dict[str, str]] = {
    F_TRANSCRIPT_TXT: F_TRANSCRIPT_TXT,
    F_ALIGNED: F_ALIGNED,
    F_METRICS: F_METRICS,
}

#: Google returns a project number where a named project was requested; the
#: canonical resource shape is ``projects/<id>/locations/<loc>/models/<family>``.
_MT_MODEL_RE: Final[re.Pattern[str]] = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)/models/(?P<family>.+)$"
)

_FROZEN = ConfigDict(extra="forbid", frozen=True)
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class AlignmentError(ValueError):
    """Alignment or provenance validation failed; no success artifact exists."""

    def __init__(self, issues: Sequence[str], *, context: str = "alignment") -> None:
        self.issues = tuple(issues)
        detail = "; ".join(self.issues) if self.issues else "unknown mismatch"
        super().__init__(f"{context}: {detail}")


class ExportError(RuntimeError):
    """A derived export could not be produced from verified durable artifacts."""


class AlignedSegment(BaseModel):
    """One aligned segment preserving every source/translation value verbatim.

    ``source_text`` is never rewritten. ``translation_input`` and
    ``translated_text_tr`` stay distinct fields, and ``corrections`` is an
    optional, caller-supplied note list that is never applied to the source.
    """

    model_config = _FROZEN

    index: NonNegativeInt
    segment_id: str = Field(min_length=1)
    start_ms: NonNegativeInt
    end_ms: NonNegativeInt
    speaker: str | None = None
    source_language: str = Field(min_length=1)
    source_text: str
    translation_input: str | None = None
    translated_text_tr: str | None = None
    flags: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_times(self) -> "AlignedSegment":
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be strictly greater than start_ms")
        return self


class GroupProvenance(BaseModel):
    """Durable per-group MT provenance carried into the aligned artifact."""

    model_config = _FROZEN

    group_id: str = Field(min_length=1)
    attempt: int | None = Field(default=None, strict=True, ge=1)
    status: str = Field(min_length=1)
    segment_ids: tuple[str, ...] = ()
    raw_reference: RawArtifactRef | None = None


class AlignmentProvenance(BaseModel):
    """Requested/returned model identity and raw references for one job."""

    model_config = _FROZEN

    job_id: str = Field(min_length=1)
    source_language: str | None = None
    target_language: str = TARGET_LANGUAGE_CODE
    config_fingerprint: str = Field(min_length=1)
    job_fingerprint: str = Field(min_length=1)
    #: Canonical (extracted) audio identity. For a video job this is the
    #: extracted artifact's SHA-256; the original video is recorded separately so
    #: the two origins are never conflated.
    audio_sha256: str = Field(min_length=1)
    #: Additive Phase 5 origin. ``None`` for every ordinary audio-only job, so
    #: the aligned artifact of an existing job is unchanged.
    source_video_sha256: str | None = None
    extraction_id: str | None = None
    #: Additive YouTube source origin. ``None`` for every non-YouTube job, so an
    #: inherited aligned artifact stays byte-identical. The canonical audio hash
    #: above remains the derived artifact's hash; these fields retain the
    #: separate YouTube provenance alongside it.
    source_youtube_video_id: str | None = None
    source_youtube_url: str | None = None
    source_youtube_intermediate_sha256: str | None = None
    source_youtube_ytdlp_version: str | None = None
    stt_provider: str = Field(min_length=1)
    stt_requested_model: str = Field(min_length=1)
    stt_reported_model: str | None = None
    stt_raw_reference: RawArtifactRef | None = None
    mt_provider: str = Field(min_length=1)
    mt_requested_model: str = Field(min_length=1)
    mt_returned_model: str | None = None
    mt_groups: tuple[GroupProvenance, ...] = ()


class AlignedArtifact(BaseModel):
    """Structured, deterministic alignment of one job's verified artifacts."""

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    exporter_version: str = EXPORTER_VERSION
    job_id: str = Field(min_length=1)
    source_language: str | None = None
    target_language: str = TARGET_LANGUAGE_CODE
    language_uncertain: bool = False
    translation_status: str = Field(min_length=1)
    translations_included: bool = False
    audio_duration_ms: NonNegativeInt
    segments: tuple[AlignedSegment, ...] = ()
    provenance: AlignmentProvenance

    @model_validator(mode="after")
    def _check_contract(self) -> "AlignedArtifact":
        ids = [segment.segment_id for segment in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("aligned segment ids must be unique")
        if [segment.index for segment in self.segments] != list(range(len(self.segments))):
            raise ValueError("aligned segment index must be contiguous from zero")
        for segment in self.segments:
            if segment.end_ms > self.audio_duration_ms:
                raise ValueError(
                    f"segment {segment.segment_id!r} ends after audio_duration_ms"
                )
        return self


class StageTiming(BaseModel):
    model_config = _FROZEN

    status: str = Field(min_length=1)
    elapsed_ms: NonNegativeInt | None = None
    reason: str | None = None


class ModelIdentity(BaseModel):
    model_config = _FROZEN

    provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    reported_model: str | None = None
    reported_model_reason: str | None = None


class CostSummary(BaseModel):
    """Evidence-labelled cost view.

    ``status`` is ``unknown`` when nothing is measured, ``estimated`` when only
    a caller-supplied estimate exists, and ``actual`` when a provider-reported
    amount exists. An estimate is never relabelled as a measured amount, and the
    currency of each amount is kept separately so a mixed-currency record cannot
    masquerade as one figure.
    """

    model_config = _FROZEN

    status: str = Field(min_length=1)
    estimated: float | None = None
    actual: float | None = None
    currency: str | None = None
    estimated_currency: str | None = None
    actual_currency: str | None = None
    unknown_reason: str | None = None
    reservation_total: str | None = None
    reservation_currency: str | None = None
    reservation_note: str = (
        "reservation is a local worst-case budget bound, not the provider invoice"
    )


class RetrySummary(BaseModel):
    model_config = _FROZEN

    attempts: NonNegativeInt = 0
    complete: NonNegativeInt = 0
    failed: NonNegativeInt = 0
    remote_unknown: NonNegativeInt = 0
    retry_decisions: NonNegativeInt = 0


class QualitySummary(BaseModel):
    model_config = _FROZEN

    checked_segments: NonNegativeInt = 0
    translated_segments: NonNegativeInt = 0
    flagged_segments: NonNegativeInt = 0
    flag_counts: dict[str, int] = Field(default_factory=dict)
    review_required: bool = False
    limitations: tuple[str, ...] = QUALITY_LIMITATIONS


class UnmeasuredMetric(BaseModel):
    model_config = _FROZEN

    status: str = Field(default="unmeasured", min_length=1)
    value: float | None = None
    reason: str = Field(min_length=1)


class JobMetrics(BaseModel):
    """Honest, evidence-derived metrics for one job.

    Timing is the measured sum of stage elapsed times; wall-clock time including
    resume gaps is deliberately *not* reported. Missing invoices stay
    ``unknown`` with a reason instead of a fabricated zero, and human WER/CER
    stays explicitly unmeasured without a reference transcript.
    """

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    exporter_version: str = EXPORTER_VERSION
    job_id: str = Field(min_length=1)
    generated_at_utc: UtcDatetime
    status: str = Field(min_length=1)
    source_language: str | None = None
    language_uncertain: bool = False
    audio_duration_ms: NonNegativeInt
    timing_scope: str = (
        "sum of measured stage elapsed times; wall-clock time including resume "
        "gaps is not reported"
    )
    stt: ModelIdentity
    mt: ModelIdentity
    stt_timing: StageTiming
    mt_timing: StageTiming
    total_stage_elapsed_ms: NonNegativeInt | None = None
    source_char_count: NonNegativeInt = 0
    target_char_count: NonNegativeInt | None = None
    quality: QualitySummary
    retries: RetrySummary
    cost: CostSummary
    wer_cer: UnmeasuredMetric
    translation_included: bool = False
    exports_complete: bool = True
    notes: tuple[str, ...] = ()


class ExportsManifest(BaseModel):
    """Completion marker binding every derived artifact to its verified inputs."""

    model_config = _FROZEN

    storage_version: str = Field(default="1", min_length=1)
    exporter_version: str = EXPORTER_VERSION
    job_id: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    transcript_sha256: str = Field(min_length=1)
    translation_sha256: str | None = None
    translation_included: bool = False
    review_required: bool = False
    flagged_segment_count: NonNegativeInt = 0
    #: Durable paid-call evidence bound into the export set. A later export that
    #: finds these recorded but the ledger gone fails closed instead of
    #: silently dropping a previously measured reservation.
    ledger_digest: str | None = None
    reservation_total: str | None = None
    reservation_currency: str | None = None
    artifacts: dict[str, StoredArtifact] = Field(default_factory=dict)
    completed_at_utc: UtcDatetime


class ExportResult(BaseModel):
    model_config = _FROZEN

    job_id: str
    status: str
    transcript_text_path: str
    translation_text_path: str | None = None
    aligned_path: str
    metrics_path: str
    manifest_path: str
    translation_included: bool
    review_required: bool
    flagged_segment_count: NonNegativeInt
    already_present: bool
    metrics: JobMetrics


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def _revalidate_transcript(transcript: Transcript) -> Transcript:
    if not isinstance(transcript, Transcript):
        raise AlignmentError(["transcript must be a Transcript"], context="transcript")
    try:
        return Transcript.model_validate(transcript.model_dump())
    except ValueError as exc:  # model_copy can bypass validators
        raise AlignmentError([f"transcript failed revalidation: {exc}"]) from exc


def _revalidate_translation(translation: Translation) -> Translation:
    if not isinstance(translation, Translation):
        raise AlignmentError(["translation must be a Translation"], context="translation")
    try:
        return Translation.model_validate(translation.model_dump())
    except ValueError as exc:
        raise AlignmentError([f"translation failed revalidation: {exc}"]) from exc


def _check_times(segments: Sequence[Segment], audio_duration_ms: int) -> list[str]:
    issues: list[str] = []
    ids = [segment.segment_id for segment in segments]
    if len(set(ids)) != len(ids):
        issues.append("source segment ids contain duplicates")
    if audio_duration_ms <= 0:
        issues.append("audio_duration_ms must be positive")
    for segment in segments:
        if segment.start_ms < 0:
            issues.append(f"segment {segment.segment_id!r} has a negative start_ms")
        if segment.end_ms <= segment.start_ms:
            issues.append(
                f"segment {segment.segment_id!r} does not have start_ms < end_ms"
            )
        elif segment.end_ms > audio_duration_ms:
            issues.append(
                f"segment {segment.segment_id!r} ends after audio_duration_ms"
            )
    return issues


def _returned_model_issues(requested_model: str, returned_model: str | None) -> list[str]:
    """Reject a claimed returned MT model outside the canonical Google shape.

    A returned model may echo the requested family, be the exact short model
    name Basic v2 reports (``translation-llm`` for ``general/translation-llm``),
    or be the documented normalisation of a named project to its numeric project
    id; any other family, location or malformed resource is rejected. The output
    artifact hash alone is never treated as proof of the claimed provider
    identity.
    """

    if returned_model is None or returned_model == requested_model:
        return []
    if "/" in requested_model and returned_model == requested_model.rsplit("/", 1)[-1]:
        # Basic v2 may report only the exact final model segment; equality is
        # exact, never a suffix/substring match.
        return []
    match = _MT_MODEL_RE.match(returned_model)
    if match is None:
        return ["returned MT model is not a canonical projects/.../models resource"]
    if match.group("family") != requested_model:
        return ["returned MT model family does not match the requested model"]
    if match.group("location") != "global":
        return ["returned MT model location is not global"]
    return []


def _claimed_identity_issues(
    transcript: Transcript,
    translation: Translation | None,
    provenance: AlignmentProvenance,
    resolved_language: str | None,
    same_language: bool,
) -> list[str]:
    """Check claimed provider/model/language provenance against the artifacts."""

    issues: list[str] = []
    if transcript.provider != provenance.stt_provider:
        issues.append("transcript provider does not match the claimed STT provider")
    if transcript.model != provenance.stt_requested_model:
        issues.append("transcript model does not match the claimed requested STT model")
    if (
        provenance.source_language is not None
        and transcript.source_language is not None
        and provenance.source_language != transcript.source_language
    ):
        issues.append("claimed source language does not match the transcript")

    if translation is None:
        return issues

    if translation.provider != provenance.mt_provider:
        issues.append("translation provider does not match the claimed MT provider")
    if translation.model != provenance.mt_requested_model:
        issues.append("translation model does not match the claimed requested MT model")
    if translation.target_language != TARGET_LANGUAGE_CODE:
        issues.append("translation target language is not Turkish")
    if same_language:
        if translation.source_language != TARGET_LANGUAGE_CODE:
            issues.append("a skipped same-language translation must be Turkish")
    elif translation.source_language is None:
        issues.append("translation is missing its source language")
    elif (
        resolved_language is not None
        and translation.source_language != resolved_language
    ):
        issues.append("translation source language does not match the transcript")

    issues.extend(
        _returned_model_issues(
            provenance.mt_requested_model, provenance.mt_returned_model
        )
    )
    return issues


#: Provider-flagged STT conditions that must surface as human review even though
#: they are not translation-content defects. A cloud STT route that never emits
#: them leaves this set unused; a provider that does emit them must be reviewed.
STT_REVIEW_FLAGS: Final[frozenset[str]] = frozenset(
    {
        "stt_compression_ratio_high",
        "stt_avg_logprob_low",
        "stt_no_speech_prob_high",
        "stt_repetition_suspected",
        "stt_review_rerun_disagreement",
    }
)


def _with_stt_review_flags(report: QualityReport) -> QualityReport:
    """Fold STT review flags into the deterministic review decision.

    ``assess_quality`` counts only translation-content defects, so a suspicious
    STT region or a re-run disagreement would otherwise not require review. The
    aligned segment flags already preserve those flags; this raises the review
    counter (and flag counts) for the affected segments without inventing any
    new content check or rewriting ``source_text``.
    """

    defect_flags = set(report.flag_counts)
    review_ids = {
        entry.segment_id
        for entry in report.segments
        if set(entry.flags) & (defect_flags | STT_REVIEW_FLAGS)
    }
    if not review_ids:
        return report
    counts = dict(report.flag_counts)
    for entry in report.segments:
        for flag in entry.flags:
            if flag in STT_REVIEW_FLAGS:
                counts[flag] = counts.get(flag, 0) + 1
    return QualityReport.model_validate(
        {
            **report.model_dump(),
            "flagged_segment_count": len(review_ids),
            "flag_counts": counts,
        }
    )


def build_alignment(
    transcript: Transcript,
    translation: Translation | None,
    *,
    provenance: AlignmentProvenance,
    audio_duration_ms: int,
    source_language: str | None = None,
    quality_settings: QualitySettings | None = None,
    corrections: Mapping[str, Sequence[str]] | None = None,
) -> tuple[AlignedArtifact, QualityReport]:
    """Align verified artifacts by exact segment identity and order.

    Missing, duplicate, extra or reordered translation segment IDs raise
    :class:`AlignmentError` before any success artifact can be produced. Source
    text, translation input and timings are never repaired.
    """

    transcript = _revalidate_transcript(transcript)
    translation = (
        _revalidate_translation(translation) if translation is not None else None
    )

    issues = _check_times(transcript.segments, audio_duration_ms)
    if issues:
        raise AlignmentError(issues, context="source transcript")

    expected_ids = [segment.segment_id for segment in transcript.segments]
    translated_by_id: dict[str, TranslatedSegment] = {}
    translation_status = "not_available"
    translations_included = False
    if translation is not None:
        if translation.status is TranslationStatus.failed:
            raise AlignmentError(
                ["a failed translation has no alignable segments"],
                context="translation",
            )
        try:
            translation.validate_against(expected_ids)
        except ValueError as exc:
            raise AlignmentError([str(exc)], context="translation contract") from exc
        actual_ids = [segment.segment_id for segment in translation.segments]
        if len(actual_ids) != len(expected_ids):
            raise AlignmentError(
                [
                    "translation segment count differs from the transcript: "
                    f"{len(actual_ids)} != {len(expected_ids)}"
                ],
                context="translation contract",
            )
        translated_by_id = {
            segment.segment_id: segment for segment in translation.segments
        }
        for source_segment in transcript.segments:
            translated = translated_by_id[source_segment.segment_id]
            if translated.source_text != source_segment.source_text:
                raise AlignmentError(
                    [f"source_text altered for {source_segment.segment_id!r}"],
                    context="translation provenance",
                )
            if translated.translation_input != source_segment.translation_input:
                raise AlignmentError(
                    [f"translation_input altered for {source_segment.segment_id!r}"],
                    context="translation provenance",
                )
        translation_status = translation.status.value
        translations_included = translation.status in {
            TranslationStatus.translated,
            TranslationStatus.skipped_same_language,
        }

    resolved_quality = (
        quality_settings if quality_settings is not None else QualitySettings()
    )
    resolved_language = source_language
    if resolved_language is None:
        resolved_language = transcript.source_language
    same_language = (
        translation is not None
        and translation.status is TranslationStatus.skipped_same_language
    )
    if not same_language and resolved_language == TARGET_LANGUAGE_CODE:
        same_language = True
    language_uncertain = bool(transcript.language_uncertain)

    identity_issues = _claimed_identity_issues(
        transcript, translation, provenance, resolved_language, same_language
    )
    if identity_issues:
        raise AlignmentError(identity_issues, context="provenance")

    correction_map: dict[str, tuple[str, ...]] = {}
    if corrections:
        unknown_ids = sorted(set(corrections) - set(expected_ids))
        if unknown_ids:
            raise AlignmentError(
                [f"corrections reference unknown segment ids: {unknown_ids}"],
                context="corrections",
            )
        for key, value in corrections.items():
            correction_map[key] = tuple(str(item) for item in value)

    quality_items: list[QualityInput] = []
    for source_segment in transcript.segments:
        translated = translated_by_id.get(source_segment.segment_id)
        existing = tuple(source_segment.flags)
        if translated is not None:
            for flag in translated.flags:
                if flag not in existing:
                    existing = (*existing, flag)
        quality_items.append(
            QualityInput(
                segment_id=source_segment.segment_id,
                source_text=source_segment.source_text,
                translated_text_tr=(
                    translated.translated_text_tr if translated is not None else None
                ),
                translation_input=source_segment.translation_input,
                existing_flags=existing,
            )
        )
    report = _with_stt_review_flags(
        assess_quality(
            quality_items,
            settings=resolved_quality,
            same_language=same_language,
            language_uncertain=language_uncertain,
        )
    )

    aligned_segments: list[AlignedSegment] = []
    for index, source_segment in enumerate(transcript.segments):
        translated = translated_by_id.get(source_segment.segment_id)
        aligned_segments.append(
            AlignedSegment(
                index=index,
                segment_id=source_segment.segment_id,
                start_ms=source_segment.start_ms,
                end_ms=source_segment.end_ms,
                speaker=source_segment.speaker,
                source_language=source_segment.source_language,
                source_text=source_segment.source_text,
                translation_input=source_segment.translation_input,
                translated_text_tr=(
                    translated.translated_text_tr if translated is not None else None
                ),
                flags=report.flags_for(source_segment.segment_id),
                corrections=correction_map.get(source_segment.segment_id, ()),
            )
        )

    artifact = AlignedArtifact(
        job_id=provenance.job_id,
        source_language=resolved_language,
        target_language=TARGET_LANGUAGE_CODE,
        language_uncertain=language_uncertain,
        translation_status=translation_status,
        translations_included=translations_included,
        audio_duration_ms=audio_duration_ms,
        segments=tuple(aligned_segments),
        provenance=provenance,
    )
    try:
        artifact = AlignedArtifact.model_validate(artifact.model_dump())
    except ValueError as exc:
        raise AlignmentError([f"aligned artifact failed revalidation: {exc}"]) from exc
    return artifact, report


def render_source_text(artifact: AlignedArtifact) -> str:
    """Render exactly the preserved source segment texts, one per line."""

    lines = [segment.source_text for segment in artifact.segments]
    return "\n".join(lines) + ("\n" if lines else "")


def render_translation_text(artifact: AlignedArtifact) -> str:
    """Render the Turkish segment texts, one per line, only when translated."""

    if not artifact.translations_included:
        raise ExportError(
            "translation was not included; refusing to write a text export"
        )
    lines = [
        segment.translated_text_tr if segment.translated_text_tr is not None else ""
        for segment in artifact.segments
    ]
    return "\n".join(lines) + ("\n" if lines else "")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _stage_timing(stages: Sequence[StageResult], stage: StageKind) -> StageTiming:
    for result in stages:
        if result.stage is stage:
            if result.elapsed_ms is None:
                return StageTiming(
                    status="not_recorded",
                    reason="the stage result carries no measured elapsed time",
                )
            return StageTiming(status="measured", elapsed_ms=result.elapsed_ms)
    return StageTiming(
        status="not_recorded",
        reason="no stage result is recorded for this job",
    )


def _outcome_value(record: object) -> str | None:
    outcome = getattr(record, "outcome", None)
    if outcome is None:
        return None
    return getattr(outcome, "value", str(outcome))


def _reported_stt_model(transcript: Transcript) -> tuple[str | None, str | None]:
    """Return the independently reported STT model, if any.

    ``transcript.request_metadata['model_id']`` is written by the Scribe adapter
    from the *requested* configuration (``scribe.py``), not from the provider
    response, so it is not reported evidence. Until a verified response field
    supplies it independently, the reported identity stays null with a reason;
    the requested identity is carried separately as ``requested_model``.
    """

    return (
        None,
        "the STT provider response does not independently report a model id",
    )


def build_metrics(
    *,
    stored: StoredInput,
    transcript: Transcript,
    translation: Translation | None,
    job_record: JobRecord | None,
    attempts: Sequence[object],
    quality: QualityReport,
    translation_included: bool,
    exports_complete: bool,
    mt_returned_model: str | None,
    mt_returned_model_reason: str | None,
    retry_decisions: int,
    reservation_total: str | None,
    reservation_currency: str | None,
    now: datetime | None = None,
) -> JobMetrics:
    """Assemble metrics from durable evidence only (no estimates)."""

    stages = job_record.stages if job_record is not None else ()
    stt_timing = _stage_timing(stages, StageKind.stt)
    mt_timing = _stage_timing(stages, StageKind.mt)
    total_stage: int | None = None
    if stt_timing.elapsed_ms is not None and mt_timing.elapsed_ms is not None:
        total_stage = stt_timing.elapsed_ms + mt_timing.elapsed_ms

    source_char_count = sum(len(segment.source_text) for segment in transcript.segments)
    target_count: int | None = None
    if translation is not None and translation_included:
        target_count = sum(
            len(segment.translated_text_tr) for segment in translation.segments
        )

    outcome_values = [_outcome_value(record) for record in attempts]
    reported_stt, reported_stt_reason = _reported_stt_model(transcript)

    cost = job_record.cost if job_record is not None else None
    if cost is None:
        from subtitle_flow.schemas import JobCost

        cost = JobCost(
            unknown=True,
            unknown_reason="no measured provider invoice is available",
        )
    if cost.unknown or (cost.estimated is None and cost.actual is None):
        cost_status = "unknown"
    elif cost.actual is not None:
        cost_status = "actual"
    else:
        cost_status = "estimated"
    estimated_currency = (
        cost.estimated.currency if cost.estimated is not None else None
    )
    actual_currency = cost.actual.currency if cost.actual is not None else None
    cost_summary = CostSummary(
        status=cost_status,
        estimated=float(cost.estimated.amount) if cost.estimated is not None else None,
        actual=float(cost.actual.amount) if cost.actual is not None else None,
        # The headline currency follows the measured amount when present; each
        # amount's own currency is also kept so they are never conflated.
        currency=actual_currency or estimated_currency,
        estimated_currency=estimated_currency,
        actual_currency=actual_currency,
        unknown_reason=cost.unknown_reason,
        reservation_total=reservation_total,
        reservation_currency=reservation_currency,
    )

    return JobMetrics(
        job_id=stored.input.job_id,
        generated_at_utc=now if now is not None else utc_now(),
        status=job_record.status.value if job_record is not None else "unknown",
        source_language=transcript.source_language or stored.input.source_language,
        language_uncertain=bool(transcript.language_uncertain),
        audio_duration_ms=stored.audio.duration_ms,
        stt=ModelIdentity(
            provider=stored.config.stt.provider,
            requested_model=stored.config.stt.model,
            reported_model=reported_stt,
            reported_model_reason=reported_stt_reason,
        ),
        mt=ModelIdentity(
            provider=stored.config.mt.provider,
            requested_model=stored.config.mt.model,
            reported_model=mt_returned_model,
            reported_model_reason=mt_returned_model_reason,
        ),
        stt_timing=stt_timing,
        mt_timing=mt_timing,
        total_stage_elapsed_ms=total_stage,
        source_char_count=source_char_count,
        target_char_count=target_count,
        quality=QualitySummary(
            checked_segments=quality.checked_segment_count,
            translated_segments=quality.translated_segment_count,
            flagged_segments=quality.flagged_segment_count,
            flag_counts=dict(quality.flag_counts),
            review_required=quality.review_required,
            limitations=quality.limitations,
        ),
        retries=RetrySummary(
            attempts=len(attempts),
            complete=sum(1 for value in outcome_values if value == "complete"),
            failed=sum(1 for value in outcome_values if value == "failed"),
            remote_unknown=sum(
                1 for value in outcome_values if value == "remote_unknown"
            ),
            retry_decisions=retry_decisions,
        ),
        cost=cost_summary,
        wer_cer=UnmeasuredMetric(
            reason=(
                "no human-verified reference transcript or adequacy score was "
                "provided; WER/CER and semantic quality are not measured"
            )
        ),
        translation_included=translation_included,
        exports_complete=exports_complete,
        notes=(
            "timing is measured stage elapsed time, not wall-clock time",
            "quality flags are provisional and do not prove semantic accuracy",
        ),
    )


# --------------------------------------------------------------------------- #
# Canonical evidence verification
# --------------------------------------------------------------------------- #
def _verified_bytes(store: JobStore, artifact: StoredArtifact) -> bytes:
    """Read a canonical artifact, converting store I/O failures to ExportError.

    ``PathSecurityError`` (a symlink/confinement violation) is deliberately not
    wrapped: the security boundary must stay visible to the caller.
    """

    try:
        return store.verify_artifact(artifact)
    except PathSecurityError:
        raise
    except OSError as exc:
        raise StorageCorruptionError(
            f"canonical artifact is missing on disk: {artifact.locator}"
        ) from exc
    # A ``StorageCorruptionError`` is already the correct typed failure and is
    # allowed to propagate unchanged.


def _verified_raw(
    store: JobStore,
    reference: RawArtifactRef,
    *,
    stage: StageKind,
    group_id: str | None,
    attempt: int | None,
) -> bytes:
    try:
        return store.verify_raw(
            reference, stage=stage, group_id=group_id, attempt=attempt
        )
    except PathSecurityError:
        raise
    except OSError as exc:
        raise StorageCorruptionError(
            f"canonical raw evidence is missing on disk: {reference.locator}"
        ) from exc
    # A ``StorageCorruptionError`` from ``verify_raw`` is already the correct
    # typed failure (unregistered body, stage/attempt mismatch, tampered bytes).


def _verify_manifest_identity(manifest: object, stored: StoredInput) -> None:
    if manifest.config_fingerprint != stored.config_fingerprint:  # type: ignore[attr-defined]
        raise ExportError("manifest configuration fingerprint does not match the job")
    if manifest.job_fingerprint != stored.job_fingerprint:  # type: ignore[attr-defined]
        raise ExportError("manifest job fingerprint does not match the job")
    if manifest.audio_sha256 != stored.audio.audio_sha256:  # type: ignore[attr-defined]
        raise ExportError("manifest audio hash does not match the job")


def _read_verified_transcript(
    store: JobStore, stt_manifest: object, stored: StoredInput
) -> Transcript:
    _verify_manifest_identity(stt_manifest, stored)
    if (
        stt_manifest.provider != stored.config.stt.provider  # type: ignore[attr-defined]
        or stt_manifest.model != stored.config.stt.model  # type: ignore[attr-defined]
    ):
        raise ExportError("STT manifest provider/model does not match the job snapshot")
    data = _verified_bytes(store, stt_manifest.transcript)  # type: ignore[attr-defined]
    try:
        transcript = Transcript.model_validate_json(data)
    except ValueError as exc:
        raise ExportError(f"stored transcript is invalid: {exc}") from exc
    if transcript.raw_reference is None:
        raise ExportError("stored transcript has no raw body reference")
    if transcript.raw_reference != stt_manifest.raw_reference:  # type: ignore[attr-defined]
        raise ExportError("stored transcript raw reference disagrees with the manifest")
    # The attempt is left unconstrained here: pre-intent snapshots may carry the
    # manifest-attempt default (1) while the raw registration predates attempts.
    # The body is still proven to be the registered STT raw with the exact hash,
    # size, kind, locator subtree and request id; the pipeline binds the exact
    # attempt on its own reuse path.
    _verified_raw(
        store,
        stt_manifest.raw_reference,  # type: ignore[attr-defined]
        stage=StageKind.stt,
        group_id=None,
        attempt=None,
    )
    # Every supplementary review re-run body must be independently proven too.
    for reference in collect_supplementary_refs(
        StageKind.stt, transcript.request_metadata
    ):
        _verified_raw(
            store, reference, stage=StageKind.stt, group_id=None, attempt=None
        )
    return transcript


def _read_verified_translation(
    store: JobStore, mt_manifest: MTManifest, stored: StoredInput
) -> tuple[Translation, StoredArtifact]:
    artifact = mt_manifest.translation
    assert artifact is not None
    _verify_manifest_identity(mt_manifest, stored)
    if (
        mt_manifest.provider != stored.config.mt.provider
        or mt_manifest.model != stored.config.mt.model
    ):
        raise ExportError("MT manifest provider/model does not match the job snapshot")
    data = _verified_bytes(store, artifact)
    try:
        translation = Translation.model_validate_json(data)
    except ValueError as exc:
        raise ExportError(f"stored translation is invalid: {exc}") from exc
    if (
        translation.provider != stored.config.mt.provider
        or translation.model != stored.config.mt.model
    ):
        raise ExportError(
            "stored translation provider/model does not match the job snapshot"
        )
    if translation.target_language != TARGET_LANGUAGE_CODE:
        raise ExportError("stored translation target language is not Turkish")
    return translation, artifact


def _verified_mt_groups(
    store: JobStore, mt_manifest: MTManifest
) -> dict[str, Translation]:
    """Verify every complete MT group output and raw body before any reuse.

    A complete/skipped manifest that still names a non-complete group is
    corruption and fails closed; a non-complete group under a partial/running
    manifest is skipped (never presented as success).
    """

    groups: dict[str, Translation] = {}
    terminal = mt_manifest.status in {"complete", "skipped"}
    for record in mt_manifest.groups:
        if record.status is not MTGroupStatus.complete:
            if terminal:
                raise ExportError(
                    f"MT manifest is {mt_manifest.status!r} but group "
                    f"{record.group_id!r} is {record.status.value!r}"
                )
            continue
        if record.output is None or record.raw_reference is None:
            raise ExportError(
                f"completed MT group {record.group_id!r} is missing its output "
                "or raw reference"
            )
        data = _verified_bytes(store, record.output)
        try:
            group = Translation.model_validate_json(data)
        except ValueError as exc:
            raise ExportError(
                f"stored MT group {record.group_id!r} is invalid: {exc}"
            ) from exc
        if [segment.segment_id for segment in group.segments] != list(
            record.segment_ids
        ):
            raise ExportError(
                f"MT group {record.group_id!r} segment ids differ from the manifest"
            )
        if group.provider != record.provider or group.model != record.model:
            raise ExportError(
                f"MT group {record.group_id!r} provider/model differs from the manifest"
            )
        _verified_raw(
            store,
            record.raw_reference,
            stage=StageKind.mt,
            group_id=record.group_id,
            attempt=record.attempt,
        )
        # Each individual grouped response (not just the last) must be verified.
        for reference in collect_supplementary_refs(
            StageKind.mt, group.request_metadata
        ):
            _verified_raw(
                store,
                reference,
                stage=StageKind.mt,
                group_id=record.group_id,
                attempt=record.attempt,
            )
        groups[record.group_id] = group
    return groups


def _returned_model_from_groups(
    mt_manifest: MTManifest, groups: Mapping[str, Translation]
) -> tuple[str | None, str | None]:
    if mt_manifest.status == "skipped":
        return None, "same-language skip performed no provider call"
    models: set[str] = set()
    for group in groups.values():
        value = group.request_metadata.get("returned_model")
        if isinstance(value, str) and value:
            models.add(value)
    if not models:
        return None, "the provider response did not report a returned model"
    if len(models) > 1:
        return None, "groups reported different returned models"
    return next(iter(models)), None


def _ledger_evidence(store: JobStore) -> tuple[str | None, str | None, str | None]:
    """Return ``(reservation_total, currency, ledger_digest)`` for the ledger.

    A ledger that was never created yields three ``None`` values. The digest
    binds the exact validated trace into the export fingerprint so a changed or
    deleted ledger can never silently reuse a stale summary.
    """

    if not store.api_ledger_initialized():
        return None, None, None
    records = store.read_api_trace()
    digest: str | None = None
    if records:
        payload = [record.model_dump(mode="json") for record in records]
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
    amounts: list[Decimal] = []
    currency: str | None = None
    for record in records:
        if record.currency is not None:
            if currency is None:
                currency = record.currency
            elif currency != record.currency:
                raise ExportError("budget ledger mixes currencies")
        if record.amount is None:
            continue
        amount = Decimal(record.amount)
        amounts.append(
            amount if record.kind.value == "reserve" else amount.copy_negate()
        )
    total = str(exact_decimal_sum(amounts)) if amounts else None
    return total, currency, digest


def _count_retry_decisions(record: JobRecord | None) -> int:
    if record is None:
        return 0
    pipeline = record.request_metadata.get("pipeline")
    if not isinstance(pipeline, dict):
        return 0
    decisions = pipeline.get("retry_decisions")
    if not isinstance(decisions, list):
        return 0
    return len(decisions)


def _status_evidence(record: JobRecord | None) -> object:
    """Return the durable, metrics-relevant fields of the current status.

    Volatile write timestamps (``updated_at_utc``) are excluded so an untouched
    job cannot churn the fingerprint; every field that feeds a metric is kept.
    """

    if record is None:
        return None
    return {
        "status": record.status.value,
        "stages": [stage.model_dump(mode="json") for stage in record.stages],
        "cost": record.cost.model_dump(mode="json"),
        "errors": [error.model_dump(mode="json") for error in record.errors],
        "request_metadata": record.request_metadata,
    }


def _export_fingerprint(
    *,
    transcript_sha256: str,
    translation_sha256: str | None,
    translation_included: bool,
    quality_settings: QualitySettings,
    corrections: Mapping[str, Sequence[str]] | None,
    status_evidence: object,
    attempts_evidence: Sequence[object],
    ledger_digest: str | None,
    reservation_total: str | None,
    reservation_currency: str | None,
    mt_returned_model: str | None,
) -> str:
    payload = {
        "exporter_version": EXPORTER_VERSION,
        "transcript_sha256": transcript_sha256,
        "translation_sha256": translation_sha256,
        "translation_included": translation_included,
        "quality_settings": quality_settings.model_dump(mode="json"),
        "corrections": {
            key: list(value) for key, value in sorted((corrections or {}).items())
        },
        "status": status_evidence,
        "attempts": list(attempts_evidence),
        "ledger_digest": ledger_digest,
        "reservation_total": reservation_total,
        "reservation_currency": reservation_currency,
        "mt_returned_model": mt_returned_model,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reusable_metrics(
    store: JobStore,
    existing: ExportsManifest,
    *,
    fingerprint: str,
    stored: StoredInput,
    transcript_artifact: StoredArtifact,
    translation_artifact: StoredArtifact | None,
    translation_included: bool,
) -> JobMetrics | None:
    """Return the cached metrics if the whole derived set is safely reusable.

    The manifest identity, current version, required file/path set, input hashes
    and the complete current durable evidence are all re-checked. Missing or
    corrupt *derived* files return ``None`` so the caller rebuilds; a previously
    recorded but now-missing paid ledger is a fail-closed error, never a silent
    downgrade. ``PathSecurityError`` always propagates.
    """

    if existing.job_id != stored.input.job_id:
        return None
    if existing.exporter_version != EXPORTER_VERSION:
        return None
    if existing.ledger_digest is not None and not store.api_ledger_initialized():
        raise ExportError(
            "the previous export recorded a paid-call ledger but it is now missing"
        )
    if existing.fingerprint != fingerprint:
        return None
    if existing.transcript_sha256 != transcript_artifact.sha256:
        return None
    expected_translation = (
        translation_artifact.sha256 if translation_artifact is not None else None
    )
    if existing.translation_sha256 != expected_translation:
        return None
    if existing.translation_included != translation_included:
        return None

    required = dict(_REQUIRED_DERIVED)
    if translation_included:
        required[F_TRANSLATION_TXT] = F_TRANSLATION_TXT
    if set(existing.artifacts) != set(required):
        return None
    for key, locator in required.items():
        artifact = existing.artifacts.get(key)
        if artifact is None or artifact.locator != locator:
            return None

    try:
        for artifact in existing.artifacts.values():
            store.verify_artifact(artifact)
        metrics_data = store.verify_artifact(existing.artifacts[F_METRICS])
        return JobMetrics.model_validate_json(metrics_data)
    except PathSecurityError:
        raise
    except (KeyError, ValueError, OSError):
        return None


def export_job(
    store: JobStore,
    *,
    quality_settings: QualitySettings | None = None,
    corrections: Mapping[str, Sequence[str]] | None = None,
) -> ExportResult:
    """Produce (or idempotently reuse) the derived exports for one locked job.

    Every canonical input (snapshot, STT/MT manifest identity, groups, raw
    bodies) is verified before any reuse; missing or corrupt canonical evidence
    fails closed with :class:`ExportError`. Derived files are rebuilt from the
    verified durable artifacts when absent or stale, without any provider call.
    """

    if not store.locked:
        raise ExportError("the job store must hold its writer lock before exporting")

    stored = store.read_input()
    stt_manifest = store.read_stt_manifest()
    if stt_manifest is None:
        raise ExportError("no completed STT manifest exists; nothing to export")
    transcript = _read_verified_transcript(store, stt_manifest, stored)

    mt_manifest = store.read_mt_manifest()
    translation: Translation | None = None
    translation_artifact: StoredArtifact | None = None
    groups: dict[str, Translation] = {}
    if mt_manifest is not None:
        groups = _verified_mt_groups(store, mt_manifest)
        if (
            mt_manifest.translation is not None
            and mt_manifest.status in {"complete", "skipped"}
        ):
            translation, translation_artifact = _read_verified_translation(
                store, mt_manifest, stored
            )
    translation_included = translation is not None and translation.status in {
        TranslationStatus.translated,
        TranslationStatus.skipped_same_language,
    }

    mt_returned_model: str | None = None
    mt_returned_reason: str | None = None
    if translation_included and mt_manifest is not None:
        mt_returned_model, mt_returned_reason = _returned_model_from_groups(
            mt_manifest, groups
        )
    elif mt_manifest is not None and mt_manifest.status == "skipped":
        mt_returned_reason = "same-language skip performed no provider call"

    resolved_source_language = (
        transcript.source_language or stored.input.source_language
    )
    video_origin = stored.config.video_origin
    youtube_origin = stored.config.youtube_origin
    provenance = AlignmentProvenance(
        job_id=stored.input.job_id,
        source_language=resolved_source_language,
        target_language=TARGET_LANGUAGE_CODE,
        config_fingerprint=stored.config_fingerprint,
        job_fingerprint=stored.job_fingerprint,
        audio_sha256=stored.audio.audio_sha256,
        source_video_sha256=(
            video_origin.source_sha256 if video_origin is not None else None
        ),
        extraction_id=(
            video_origin.extraction_id if video_origin is not None else None
        ),
        source_youtube_video_id=(
            youtube_origin.video_id if youtube_origin is not None else None
        ),
        source_youtube_url=(
            youtube_origin.canonical_url if youtube_origin is not None else None
        ),
        source_youtube_intermediate_sha256=(
            youtube_origin.intermediate_sha256
            if youtube_origin is not None
            else None
        ),
        source_youtube_ytdlp_version=(
            youtube_origin.ytdlp_version if youtube_origin is not None else None
        ),
        stt_provider=stored.config.stt.provider,
        stt_requested_model=stored.config.stt.model,
        # ``request_metadata['model_id']`` is the requested config, never
        # provider-reported, so the reported identity stays null.
        stt_reported_model=None,
        stt_raw_reference=stt_manifest.raw_reference,
        mt_provider=stored.config.mt.provider,
        mt_requested_model=stored.config.mt.model,
        mt_returned_model=mt_returned_model,
        mt_groups=tuple(
            GroupProvenance(
                group_id=record.group_id,
                attempt=record.attempt,
                status=record.status.value,
                segment_ids=record.segment_ids,
                raw_reference=record.raw_reference,
            )
            for record in (mt_manifest.groups if mt_manifest is not None else ())
        ),
    )

    resolved_quality = (
        quality_settings if quality_settings is not None else QualitySettings()
    )
    try:
        alignment, report = build_alignment(
            transcript,
            translation,
            provenance=provenance,
            audio_duration_ms=stored.audio.duration_ms,
            source_language=resolved_source_language,
            quality_settings=resolved_quality,
            corrections=corrections,
        )
    except AlignmentError as exc:
        raise ExportError(str(exc)) from exc

    job_record = store.read_status()
    attempts = store.read_attempts()
    reservation_total, reservation_currency, ledger_digest = _ledger_evidence(store)
    fingerprint = _export_fingerprint(
        transcript_sha256=stt_manifest.transcript.sha256,
        translation_sha256=(
            translation_artifact.sha256 if translation_artifact is not None else None
        ),
        translation_included=translation_included,
        quality_settings=resolved_quality,
        corrections=corrections,
        status_evidence=_status_evidence(job_record),
        attempts_evidence=[record.model_dump(mode="json") for record in attempts],
        ledger_digest=ledger_digest,
        reservation_total=reservation_total,
        reservation_currency=reservation_currency,
        mt_returned_model=mt_returned_model,
    )

    existing = _read_exports_manifest(store)
    if existing is not None:
        cached = _reusable_metrics(
            store,
            existing,
            fingerprint=fingerprint,
            stored=stored,
            transcript_artifact=stt_manifest.transcript,
            translation_artifact=translation_artifact,
            translation_included=translation_included,
        )
        if cached is not None:
            return ExportResult(
                job_id=stored.input.job_id,
                status=cached.status,
                transcript_text_path=F_TRANSCRIPT_TXT,
                translation_text_path=(
                    F_TRANSLATION_TXT if existing.translation_included else None
                ),
                aligned_path=F_ALIGNED,
                metrics_path=F_METRICS,
                manifest_path=F_EXPORTS_MANIFEST,
                translation_included=existing.translation_included,
                review_required=existing.review_required,
                flagged_segment_count=existing.flagged_segment_count,
                already_present=True,
                metrics=cached,
            )

    metrics = build_metrics(
        stored=stored,
        transcript=transcript,
        translation=translation,
        job_record=job_record,
        attempts=attempts,
        quality=report,
        translation_included=translation_included,
        exports_complete=True,
        mt_returned_model=mt_returned_model,
        mt_returned_model_reason=mt_returned_reason,
        retry_decisions=_count_retry_decisions(job_record),
        reservation_total=reservation_total,
        reservation_currency=reservation_currency,
    )
    metrics = JobMetrics.model_validate(metrics.model_dump())

    artifacts: dict[str, StoredArtifact] = {}
    artifacts[F_TRANSCRIPT_TXT] = store.write_artifact(
        F_TRANSCRIPT_TXT, render_source_text(alignment).encode("utf-8")
    )
    if translation_included:
        artifacts[F_TRANSLATION_TXT] = store.write_artifact(
            F_TRANSLATION_TXT, render_translation_text(alignment).encode("utf-8")
        )
    artifacts[F_ALIGNED] = store.write_artifact(
        F_ALIGNED, alignment.model_dump_json(indent=2).encode("utf-8")
    )
    artifacts[F_METRICS] = store.write_artifact(
        F_METRICS, metrics.model_dump_json(indent=2).encode("utf-8")
    )

    manifest = ExportsManifest(
        job_id=stored.input.job_id,
        fingerprint=fingerprint,
        transcript_sha256=stt_manifest.transcript.sha256,
        translation_sha256=(
            translation_artifact.sha256 if translation_artifact is not None else None
        ),
        translation_included=translation_included,
        review_required=report.review_required,
        flagged_segment_count=report.flagged_segment_count,
        ledger_digest=ledger_digest,
        reservation_total=reservation_total,
        reservation_currency=reservation_currency,
        artifacts=artifacts,
        completed_at_utc=utc_now(),
    )
    store.write_artifact(
        F_EXPORTS_MANIFEST, manifest.model_dump_json(indent=2).encode("utf-8")
    )
    return ExportResult(
        job_id=stored.input.job_id,
        status=metrics.status,
        transcript_text_path=F_TRANSCRIPT_TXT,
        translation_text_path=F_TRANSLATION_TXT if translation_included else None,
        aligned_path=F_ALIGNED,
        metrics_path=F_METRICS,
        manifest_path=F_EXPORTS_MANIFEST,
        translation_included=translation_included,
        review_required=report.review_required,
        flagged_segment_count=report.flagged_segment_count,
        already_present=False,
        metrics=metrics,
    )


def _read_exports_manifest(store: JobStore) -> ExportsManifest | None:
    path = store.resolve(F_EXPORTS_MANIFEST)
    if not path.is_file():
        return None
    try:
        return ExportsManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ExportError(f"exports manifest is unreadable: {exc}") from exc
