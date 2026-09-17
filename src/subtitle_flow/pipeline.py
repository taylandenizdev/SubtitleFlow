"""Synchronous, resumable STT -> batched MT orchestration.

The pipeline is a pure library orchestrator. It owns no provider SDK, opens no
network connection and never interprets media itself: audio validation, durable
storage and provider calls are separate, injectable collaborators.

Resume and crash-safety model
-----------------------------
* Every provider call is preceded by a durable *running intent* (``status.json``
  for STT, a ``running`` group record in the MT manifest) and a durable
  dispatch attempt is appended to ``manifests/attempts.jsonl``.
* Normalized output is written atomically **before** the stage is marked
  complete, and the raw provider body is archived by the adapter through the
  injected raw sink **before** the normalized result is committed.
* A raw body is only trusted when it is registered by
  :meth:`subtitle_flow.storage.JobStore.archive_raw` and equals the reference
  the pipeline issued for *this* dispatch (stage, group and attempt bound).
* A *known received but invalid / unverifiable* response is non-retryable
  pending review and is never re-billed automatically; a *pre-dispatch retryable*
  failure may resume. Any unknown remote outcome stays
  ``remote_status_unknown``. An explicit ``allow_remote_retry`` is the only way
  to deliberately replay an unresolved or blocked attempt; it is recorded in
  ``retry_decisions`` and the append-only attempt history.

Adapters that perform real provider calls archive their response through the raw
sink. The sink has the signature of
:meth:`subtitle_flow.storage.JobStore.archive_raw`; the pipeline supplies the
authoritative stage/group/attempt context, so adapters only pass the payload and
optional ``request_id``/``content_subtype``.
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from subtitle_flow.config import (
    BatchSettings,
    ExtractionSettings,
    PipelineConfig,
    VideoOriginSettings,
    YouTubeOriginSettings,
)
from subtitle_flow.languages import TARGET_LANGUAGE_CODE
from subtitle_flow.evidence import (
    collect_supplementary_refs,
    verify_supplementary_raw,
)
from subtitle_flow.media import (
    AudioInfo,
    MediaError,
    recheck_audio,
    sha256_file,
    validate_audio,
)
from subtitle_flow.providers.api_common import (
    ApiContext,
    ExpectedSegment,
    budget_preflight,
)
from subtitle_flow.providers.errors import ProviderCallError
from subtitle_flow.providers.mt_base import MachineTranslationProvider
from subtitle_flow.providers.stt_base import SpeechToTextProvider
from subtitle_flow.schemas import (
    JobInput,
    JobRecord,
    JobStatus,
    ProviderIdentity,
    ProviderKind,
    RawArtifactRef,
    Segment,
    SegmentContractError,
    StageError,
    StageKind,
    StageResult,
    Transcript,
    TranslatedSegment,
    Translation,
    TranslationStatus,
)
from subtitle_flow.storage import (
    AttemptIntent,
    AttemptOutcome,
    AttemptRecord,
    F_TRANSCRIPT,
    F_TRANSLATION,
    JobExistsError,
    JobStore,
    MTGroupRecord,
    MTGroupStatus,
    MTManifest,
    STTManifest,
    StorageCorruptionError,
    StoredArtifact,
    StoredInput,
    utc_now,
)
from subtitle_flow.video import (
    EXTRACTION_VERSION,
    ExtractionManifest,
    VideoInfo,
    VideoMediaError,
    build_extraction_manifest,
    compute_extraction_id,
    extract_audio,
    extraction_locator,
    ffmpeg_tool_version,
    recheck_video,
    validate_video,
)
from subtitle_flow.youtube_source import (
    YouTubeMediaError,
    acquire_youtube_source,
    canonicalize_youtube_url,
    youtube_locator,
)

__all__ = [
    "AudioPipeline",
    "PipelineError",
    "PipelineNeedsReviewError",
    "PipelineProviderError",
    "PipelineResult",
    "PreflightError",
    "ProviderBindingError",
    "ProviderResultError",
    "RawSink",
    "RemoteStatusUnknownError",
    "ReplayRefused",
    "ResumeRefused",
    "SegmentGroupingError",
    "group_segments",
]

#: Callable shape of :meth:`subtitle_flow.storage.JobStore.archive_raw`.
RawSink = Callable[..., RawArtifactRef]

_TERMINAL_STATUSES = frozenset(
    {
        JobStatus.complete,
        JobStatus.failed,
        JobStatus.interrupted,
        JobStatus.needs_review,
    }
)

#: ``stt_dispatch`` metadata values describing the last STT attempt.
_DISPATCH_NONE = "none"
_DISPATCH_UNKNOWN = "unknown"
_DISPATCH_BLOCKED = "blocked"
_DISPATCH_RETRYABLE_PREDISPATCH = "retryable_predispatch"

#: Stages a pipeline invocation may actually dispatch. The whole-route preflight
#: validates and resolves only the stages in the active scope, so a
#: transcribe-only run never demands the unrelated Google MT project, while a
#: process/resume/translate run still gate the full paid route up front.
_STT_STAGE = frozenset({StageKind.stt})
_ALL_STAGES = frozenset({StageKind.stt, StageKind.mt})


class PipelineError(Exception):
    """Base class for orchestration failures."""


class PreflightError(PipelineError):
    """Configuration, directory or media preconditions failed before any call."""


class ProviderBindingError(PipelineError):
    """An injected provider's identity does not match the configured identity."""


class ResumeRefused(PipelineError):
    """A stored job does not match the current configuration; do not reuse it."""


class ReplayRefused(PipelineError):
    """A previous attempt is known/received but unverifiable; automatic replay is refused."""

    def __init__(self, message: str, *, stage: StageKind, group_id: str | None = None) -> None:
        self.stage = stage
        self.group_id = group_id
        super().__init__(message)


class RemoteStatusUnknownError(PipelineError):
    """A provider call may have reached the remote service; replay is refused."""

    def __init__(self, message: str, *, stage: StageKind, group_id: str | None = None) -> None:
        self.stage = stage
        self.group_id = group_id
        super().__init__(message)


class PipelineNeedsReviewError(PipelineError):
    """The transcript language is uncertain/mixed; MT was not attempted."""


class SegmentGroupingError(PipelineError):
    """A segment cannot be grouped without splitting or altering the source."""


class PipelineProviderError(PipelineError):
    """A provider call failed with a known (non-ambiguous) outcome."""

    def __init__(self, error: StageError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class ProviderResultError(PipelineError):
    """A provider result violated identity, timing or provenance validation."""

    def __init__(self, error: StageError) -> None:
        self.error = error
        super().__init__(f"{error.code}: {error.message}")


class PipelineResult(BaseModel):
    """Validated outcome of a job run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    status: JobStatus
    record: JobRecord
    transcript: Transcript | None = None
    translation: Translation | None = None


def _segment_text(segment: Segment) -> str:
    return segment.translation_input if segment.translation_input is not None else segment.source_text


def _rebind_audio_path(audio: AudioInfo, path: str) -> AudioInfo:
    """Return an ``AudioInfo`` for the same bytes at a new (published) path."""

    return AudioInfo.model_validate(
        {
            **audio.model_dump(),
            "path": path,
            "original_filename": Path(path).name,
        }
    )


def group_segments(
    segments: Sequence[Segment],
    *,
    batch: BatchSettings,
) -> tuple[tuple[Segment, ...], ...]:
    """Split ordered segments into deterministic bounded groups.

    Grouping is by item count and by Unicode codepoint count of the text that
    will actually be sent to MT. Segments are never split or trimmed: a single
    segment whose text exceeds ``max_chars_per_group`` is an explicit, non-
    dispatching :class:`SegmentGroupingError` rather than a silently oversized
    request.
    """

    groups: list[tuple[Segment, ...]] = []
    current: list[Segment] = []
    current_chars = 0
    for segment in segments:
        text_length = len(_segment_text(segment))
        if text_length > batch.max_chars_per_group:
            raise SegmentGroupingError(
                f"segment {segment.segment_id!r} has {text_length} characters, "
                f"exceeding max_chars_per_group={batch.max_chars_per_group}; it "
                "cannot be split without altering the source"
            )
        exceeds = current and (
            len(current) >= batch.max_items_per_group
            or current_chars + text_length > batch.max_chars_per_group
        )
        if exceeds:
            groups.append(tuple(current))
            current = []
            current_chars = 0
        current.append(segment)
        current_chars += text_length
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _group_id(index: int) -> str:
    return f"g{index:04d}"


class _RawArchive:
    """Context-bound raw sink that binds every body to the current dispatch.

    Adapters may only archive through the sink the pipeline binds. The pipeline
    supplies the authoritative stage/group/attempt, so a fabricated or
    cross-group reference can never be attributed to the current call.
    """

    def __init__(self, store: JobStore) -> None:
        self.store = store
        self.stage: StageKind | None = None
        self.group_id: str | None = None
        self.attempt: int | None = None
        self.issued: RawArtifactRef | None = None

    def arm(self, *, stage: StageKind, group_id: str | None, attempt: int | None) -> None:
        self.stage = stage
        self.group_id = group_id
        self.attempt = attempt
        self.issued = None

    def disarm(self) -> None:
        self.stage = None
        self.group_id = None
        self.attempt = None

    def __call__(
        self,
        stage: StageKind,
        payload: bytes,
        *,
        request_id: str | None = None,
        group_id: str | None = None,
        attempt: int | None = None,
        content_subtype: str = "json",
    ) -> RawArtifactRef:
        if self.stage is None:
            raise StorageCorruptionError(
                "raw sink used outside an armed dispatch; the pipeline owns the "
                "archive context"
            )
        reference = self.store.archive_raw(
            self.stage,
            payload,
            request_id=request_id,
            group_id=self.group_id,
            attempt=self.attempt,
            content_subtype=content_subtype,
        )
        self.issued = reference
        return reference


class _RunState:
    """Accumulates durable status across a single run and writes it atomically."""

    def __init__(self, store: JobStore, stored: StoredInput, now: Callable[[], datetime]) -> None:
        self.store = store
        self.stored = stored
        self._now = now
        existing = store.read_status()
        self.initial_status = existing
        self._stages: dict[StageKind, StageResult] = (
            {result.stage: result for result in existing.stages} if existing else {}
        )
        self._errors: list[StageError] = list(existing.errors) if existing else []
        if existing is not None:
            meta = existing.request_metadata.get("pipeline")
            self.meta: dict[str, object] = dict(meta) if isinstance(meta, dict) else {}
        else:
            self.meta = {}
        self._cost = existing.cost if existing is not None else None

    @property
    def now(self) -> datetime:
        return self._now()

    def record_stage(self, result: StageResult) -> None:
        self._stages[result.stage] = result

    def has_stage(self, stage: StageKind) -> bool:
        return stage in self._stages

    def add_error(self, error: StageError) -> None:
        if error not in self._errors:
            self._errors.append(error)

    def set_meta(self, key: str, value: object) -> None:
        self.meta[key] = value

    def get_meta(self, key: str) -> object:
        return self.meta.get(key)

    def note_retry_decision(self, *, scope: str, group_id: str | None, reason: str) -> None:
        decisions = self.meta.get("retry_decisions")
        history: list[dict[str, object]] = list(decisions) if isinstance(decisions, list) else []
        history.append(
            {
                "at_utc": self.now.isoformat(),
                "scope": scope,
                "group_id": group_id,
                "reason": reason,
            }
        )
        self.meta["retry_decisions"] = history

    def write(self, status: JobStatus) -> JobRecord:
        now = self.now
        finished = now if status in _TERMINAL_STATUSES else None
        cost = self._cost
        if cost is None:
            from subtitle_flow.schemas import JobCost

            cost = JobCost(unknown=True, unknown_reason="stage cost not measured in Phase 2A")
        request_metadata: dict[str, object] = {}
        if self.meta:
            request_metadata["pipeline"] = {key: value for key, value in self.meta.items()}
        record = JobRecord(
            job_id=self.stored.input.job_id,
            status=status,
            created_at_utc=self.stored.input.created_at_utc,
            updated_at_utc=now,
            finished_at_utc=finished,
            input=self.stored.input,
            stages=tuple(
                sorted(self._stages.values(), key=lambda result: result.stage.value)
            ),
            cost=cost,
            errors=tuple(self._errors),
            request_metadata=request_metadata,
        )
        self.store.write_status(record)
        self._cost = record.cost
        return record


class AudioPipeline:
    """Per-configuration orchestrator over injected STT/MT providers."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        stt_provider: SpeechToTextProvider,
        mt_provider: MachineTranslationProvider,
        ffprobe_bin: str = "ffprobe",
        ffmpeg_bin: str = "ffmpeg",
        ytdlp_bin: str = "yt-dlp",
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        if config.stt.kind is not ProviderKind.stt or config.mt.kind is not ProviderKind.mt:
            raise PreflightError("config provider kinds are invalid")
        if stt_provider.identity != config.stt:
            raise ProviderBindingError(
                f"STT provider identity {stt_provider.identity!r} does not match "
                f"configured {config.stt!r}"
            )
        if mt_provider.identity != config.mt:
            raise ProviderBindingError(
                f"MT provider identity {mt_provider.identity!r} does not match "
                f"configured {config.mt!r}"
            )
        # Re-validate the snapshot from its dumped form: ``model_copy`` bypasses
        # field validators, so a caller-supplied object must be re-checked at this
        # trust boundary before it can influence a paid dispatch.
        config = PipelineConfig.model_validate(config.model_dump())
        self.config = config
        self._stt = stt_provider
        self._mt = mt_provider
        self._ffprobe_bin = ffprobe_bin
        self._ffmpeg_bin = ffmpeg_bin
        self._ytdlp_bin = ytdlp_bin
        self._monotonic = monotonic
        self._now = now
        self._archive: _RawArchive | None = None
        self._run_key: tuple[str, str] = ("", "")
        self._route_preflight_done = False
        # Fail closed: an invocation that has not declared its stage scope is
        # treated as a full process and must satisfy both paid routes.
        self._stage_scope: frozenset[StageKind] = _ALL_STAGES

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def derive_job_id(self, audio: AudioInfo) -> str:
        return f"job_{audio.audio_sha256[:12]}_{self.config.config_fingerprint()[:8]}"

    def preflight(self, audio_path: str) -> AudioInfo:
        """Validate config, job root writability and the audio file, without calls.

        The audio is always fully decoded; there is no configuration switch that
        can let a truncated or corrupt container reach a paid provider.
        """

        root = self.config.job_root_path
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PreflightError(f"job root is not writable: {root}: {exc}") from exc
        if not root.is_dir():
            raise PreflightError(f"job root is not a directory: {root}")
        return validate_audio(
            audio_path,
            limits=self.config.media,
            ffprobe_bin=self._ffprobe_bin,
            ffmpeg_bin=self._ffmpeg_bin,
        )

    def start(
        self,
        audio_path: str,
        *,
        job_id: str | None = None,
        input_kind: str = "audio",
    ) -> JobStore:
        """Validate and create a new job directory (or open a matching existing one).

        When an explicit ``job_id`` already exists, the caller-supplied audio is
        validated and compared to the stored job *before* it is returned. A
        different content (SHA-256/size) is refused with zero provider calls; an
        identical content at a different path is accepted under a documented
        same-content policy, and the stored path stays authoritative for resume.

        ``input_kind`` is explicit: the default ``"audio"`` path is byte-for-byte
        the Phase 2/3/4 behavior. ``"video"`` runs the additive video acceptance
        and deterministic extraction stage before the ready-audio path; the
        format is proven by ``ffprobe``, never by the file extension.
        """

        if input_kind not in {"audio", "video", "youtube"}:
            raise PreflightError(
                f"input_kind must be 'audio', 'video' or 'youtube', got "
                f"{input_kind!r}"
            )
        if input_kind == "youtube":
            raise PreflightError(
                "use start_youtube/process_youtube for a YouTube source; a "
                "positional path is not accepted as a URL"
            )
        if input_kind == "video":
            return self._start_video(audio_path, job_id=job_id)

        # A pipeline that already carries a video origin has mutated its snapshot
        # (``_start_video`` binds the origin into ``self.config``). Reusing that
        # instance for an unrelated ready-audio job cannot be reconciled without
        # silently keeping or dropping the origin, so it is refused early — with
        # no media read and no provider bound — instead of leaking a raw snapshot
        # validation error when the stored input is assembled. A fresh CLI
        # invocation always builds an origin-free config, so its behavior is
        # unchanged.
        if self.config.video_origin is not None or self.config.youtube_origin is not None:
            raise PreflightError(
                "this pipeline instance is bound to a source origin; start a new "
                "instance for a ready-audio job instead of reusing the source one"
            )

        audio = self.preflight(audio_path)
        resolved_id = job_id or self.derive_job_id(audio)
        store = JobStore(self.config.job_root, resolved_id)
        if store.exists():
            stored = store.read_input()
            self._verify_job_matches(stored)
            self._verify_caller_audio(stored, audio)
            return store
        stored_input = self._build_stored_input(audio, resolved_id)
        try:
            store.create(stored_input)
        except JobExistsError:
            # A concurrent creator won the race; verify the caller's audio too.
            existing = store.read_input()
            self._verify_job_matches(existing)
            self._verify_caller_audio(existing, audio)
        return store

    # ------------------------------------------------------------------ #
    # Video origin (Phase 5)
    # ------------------------------------------------------------------ #
    def derive_video_job_id(self, video: VideoInfo) -> str:
        """Derive a job id bound to the original video identity.

        The fingerprint already covers the extraction identity (source hash,
        selected stream, settings, extraction and tool versions), so two
        different source videos that happen to extract to the same audio bytes
        do not collapse into one job.
        """

        return f"job_{video.video_sha256[:12]}_{self.config.config_fingerprint()[:8]}"

    def _start_video(self, video_path: str, *, job_id: str | None) -> JobStore:
        """Accept a local video, extract deterministic audio and create the job."""

        root = self.config.job_root_path
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PreflightError(f"job root is not writable: {root}: {exc}") from exc
        if not root.is_dir():
            raise PreflightError(f"job root is not a directory: {root}")

        video = validate_video(
            video_path,
            limits=self.config.video,
            ffprobe_bin=self._ffprobe_bin,
            ffmpeg_bin=self._ffmpeg_bin,
        )
        settings = self.config.video_extraction
        tool_version = ffmpeg_tool_version(
            self._ffmpeg_bin, timeout=self.config.video.probe_timeout_seconds
        )
        origin = VideoOriginSettings(
            source_path=video.path,
            source_sha256=video.video_sha256,
            source_size_bytes=video.size_bytes,
            source_duration_ms=video.duration_ms,
            selected_audio_stream_index=video.selected_audio_stream_index,
            extraction_id=compute_extraction_id(video, settings, tool_version),
            extraction_version=EXTRACTION_VERSION,
            tool_version=tool_version,
        )
        # Bind the origin into the snapshot before any fingerprint or job id is
        # derived, so a video job can never share an audio-only job identity.
        self.config = self.config.with_video_origin(origin)
        resolved_id = job_id or self.derive_video_job_id(video)
        store = JobStore(self.config.job_root, resolved_id)
        if store.exists():
            stored = store.read_input()
            self._verify_job_matches(stored)
            self._verify_caller_video(stored, video)
            with store:
                self._ensure_extraction(store, stored, video=video)
            return store

        # Extract into a staging directory under the (outside-repo) job root so
        # the final atomic publish is a same-filesystem move. A crash before the
        # manifest leaves the input snapshot that resume can reconcile.
        staging = Path(tempfile.mkdtemp(prefix=".video-stage-", dir=str(root)))
        try:
            out_path = staging / f"extracted.{settings.container}"
            result = extract_audio(
                video,
                settings=settings,
                out_path=out_path,
                ffmpeg_bin=self._ffmpeg_bin,
                tool_version=tool_version,
                limits=self.config.video,
                media_limits=self.config.media,
                ffprobe_bin=self._ffprobe_bin,
            )
            locator = extraction_locator(settings)
            final_path = str(store.resolve(locator))
            audio_final = _rebind_audio_path(result.audio, final_path)
            stored_input = self._build_stored_input(
                audio_final, resolved_id, video=video
            )
            try:
                store.create(stored_input)
            except JobExistsError:
                existing = store.read_input()
                self._verify_job_matches(existing)
                self._verify_caller_video(existing, video)
                store = JobStore(self.config.job_root, existing.input.job_id)
                with store:
                    self._ensure_extraction(store, existing, video=video)
                return store
            with store:
                artifact = store.adopt_file(locator, out_path)
                manifest = build_extraction_manifest(origin, settings, artifact)
                store.write_extraction_manifest(manifest)
            return store
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _verify_caller_video(self, stored: StoredInput, video: VideoInfo) -> None:
        origin = stored.config.video_origin
        if origin is None:
            raise ResumeRefused(
                "the stored job was not created from a video; refusing to treat a "
                "video as an alternate audio route"
            )
        if (
            origin.source_sha256 != video.video_sha256
            or origin.source_size_bytes != video.size_bytes
        ):
            raise ResumeRefused(
                "the caller-supplied video differs from the video recorded for "
                "this job; refusing to ignore the new input"
            )
        if (
            stored.input.video_path is None
            or stored.input.video_sha256 is None
            or stored.input.video_sha256 != video.video_sha256
        ):
            raise StorageCorruptionError(
                "stored video identity is inconsistent with the recorded origin"
            )

    # ------------------------------------------------------------------ #
    # YouTube source (fork)
    # ------------------------------------------------------------------ #
    def derive_youtube_job_id(self, origin: YouTubeOriginSettings) -> str:
        """Derive a job id bound to the canonical video identity.

        The fingerprint already covers the timestamp-free source identity (video
        id, tool versions, intermediate and canonical hashes, settings), so two
        different videos that derive to byte-identical audio never collapse into
        one job.
        """

        return f"job_yt_{origin.video_id}_{self.config.config_fingerprint()[:8]}"

    def _require_permitted_paid_route(self, *, require_mt_project: bool) -> None:
        """Fail a source command before any download is attempted.

        The call policy is always required. The Google project is required only
        when the requested stage will actually run MT: a transcribe-only request
        honours the inherited Scribe-only behavior and must not demand an
        unrelated MT project. Credential resolution still happens later at the
        normal preflight, immediately before dispatch.
        """

        if not self.config.api.paid.allow_paid_api_calls:
            raise PipelineProviderError(
                StageError(
                    code="PAID_CALLS_DISABLED",
                    message=(
                        "paid API calls are disabled; refusing to download a "
                        "source before a paid provider could run"
                    ),
                )
            )
        if require_mt_project and self.config.google_basic.project is None:
            raise PipelineProviderError(
                StageError(
                    code="MT_PROJECT_MISSING",
                    message=(
                        "GOOGLE_TRANSLATION_PROJECT is not configured; the paid MT "
                        "route cannot run, so no source is downloaded"
                    ),
                )
            )

    def start_youtube(
        self,
        url: str,
        *,
        job_id: str | None = None,
        require_mt_project: bool = True,
    ) -> JobStore:
        """Acquire a single public YouTube audio source and create its job.

        The URL is validated offline, the canonical audio and its provenance are
        acquired through the durable source cache, and the artifact is adopted
        into the job's confined directory. A repeated call with the same URL and
        tool versions reuses the verified cache with no download; a call for a
        different video or a changed tool/settings produces a new job identity
        instead of relabelling an existing one.

        ``require_mt_project`` is ``True`` for a process/MT request and ``False``
        for a transcribe-only request, so a missing (unrelated) Google project
        does not block STT-only source work.
        """

        if self.config.video_origin is not None:
            raise PreflightError(
                "this pipeline instance is bound to a local video origin; start a "
                "new instance for a YouTube source"
            )
        root = self.config.job_root_path
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PreflightError(f"job root is not writable: {root}: {exc}") from exc
        if not root.is_dir():
            raise PreflightError(f"job root is not a directory: {root}")

        reference = canonicalize_youtube_url(url)
        # Fail the paid route before any download work.
        self._require_permitted_paid_route(require_mt_project=require_mt_project)
        result = acquire_youtube_source(
            reference,
            ytdlp_bin=self._ytdlp_bin,
            ffprobe_bin=self._ffprobe_bin,
            ffmpeg_bin=self._ffmpeg_bin,
            limits=self.config.youtube,
            settings=self.config.video_extraction,
            cache_root=self.config.youtube_cache_root,
            media_limits=self.config.media,
        )
        origin = result.origin
        self.config = self.config.with_youtube_origin(origin)
        settings = self.config.video_extraction
        resolved_id = job_id or self.derive_youtube_job_id(origin)
        store = JobStore(self.config.job_root, resolved_id)
        locator = youtube_locator(settings)
        final_path = store.resolve(locator)
        audio_final = _rebind_audio_path(result.audio, str(final_path))

        if store.exists():
            stored = store.read_input()
            self._verify_job_matches(stored)
            self._verify_caller_youtube(stored, origin)
            with store:
                self._ensure_youtube_audio(store, stored)
            return store

        staging = Path(tempfile.mkdtemp(prefix=".youtube-stage-", dir=str(root)))
        try:
            staged = staging / f"source.{settings.container}"
            shutil.copyfile(result.audio.path, staged)
            digest, size = sha256_file(staged)
            if digest != result.audio.audio_sha256 or size != result.audio.size_bytes:
                raise YouTubeMediaError(
                    "YOUTUBE_STAGE_MISMATCH",
                    "the staged canonical audio does not match the verified "
                    "source artifact",
                )
            stored_input = self._build_stored_input(audio_final, resolved_id)
            try:
                store.create(stored_input)
            except JobExistsError:
                existing = store.read_input()
                self._verify_job_matches(existing)
                self._verify_caller_youtube(existing, origin)
                store = JobStore(self.config.job_root, existing.input.job_id)
                with store:
                    self._ensure_youtube_audio(store, existing)
                return store
            with store:
                store.adopt_file(locator, staged)
                store.write_youtube_manifest(origin)
            return store
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def process_youtube(
        self,
        url: str,
        *,
        job_id: str | None = None,
        allow_remote_retry: bool = False,
    ) -> PipelineResult:
        """Acquire a YouTube source and run STT then MT, resuming if needed."""

        store = self.start_youtube(url, job_id=job_id)
        return self.resume(store.job_id, allow_remote_retry=allow_remote_retry)

    def _verify_caller_youtube(
        self, stored: StoredInput, origin: YouTubeOriginSettings
    ) -> None:
        stored_origin = stored.config.youtube_origin
        if stored_origin is None:
            raise ResumeRefused(
                "the stored job was not created from a YouTube source; refusing "
                "to treat a YouTube URL as an alternate input"
            )
        if stored_origin.identity_payload() != origin.identity_payload():
            raise ResumeRefused(
                "the caller-supplied YouTube source differs from the source "
                "recorded for this job; refusing to ignore the new input"
            )

    def _ensure_youtube_audio(self, store: JobStore, stored: StoredInput) -> AudioInfo:
        """Prove the canonical source artifact without ever re-downloading.

        A missing or corrupt artifact fails closed: this version does not
        implement an exact, proven regeneration of a fixed media hash, so it
        never silently re-downloads and rebinds an existing job.
        """

        origin = stored.config.youtube_origin
        if origin is None:
            raise StorageCorruptionError("job has no recorded YouTube origin")
        settings = stored.config.video_extraction
        locator = youtube_locator(settings)
        canonical_path = store.resolve(locator)
        if Path(stored.audio.path) != canonical_path:
            raise StorageCorruptionError(
                "the recorded source audio path does not match the job-owned "
                "canonical artifact locator"
            )
        try:
            recheck_audio(canonical_path, stored.audio)
        except MediaError as exc:
            raise YouTubeMediaError(
                "YOUTUBE_AUDIO_MISSING",
                "the canonical source audio is missing or corrupt and this "
                "version cannot regenerate a fixed media hash without a new "
                "download; refusing to silently re-download and rebind the job",
            ) from exc
        manifest = store.read_youtube_manifest(YouTubeOriginSettings)
        if manifest is not None:
            if manifest.identity_payload() != origin.identity_payload():
                raise StorageCorruptionError(
                    "the YouTube source manifest does not match the stored origin"
                )
        else:
            store.write_youtube_manifest(origin)
        return stored.audio

    def _verify_input_media(self, store: JobStore, stored: StoredInput) -> None:
        """Verify the job's canonical audio, including a video origin.

        Audio-only jobs keep the exact Phase 2/3/4 recheck. A video job verifies
        its extraction manifest and origin identity (and regenerates a lost or
        corrupt derived artifact only when the frozen identity still matches).
        """

        if stored.config.youtube_origin is not None:
            self._ensure_youtube_audio(store, stored)
            return
        if stored.config.video_origin is None:
            recheck_audio(stored.audio.path, stored.audio)
            return
        self._ensure_extraction(store, stored)

    def _ensure_extraction(
        self,
        store: JobStore,
        stored: StoredInput,
        *,
        video: VideoInfo | None = None,
    ) -> AudioInfo:
        """Prove the frozen video origin and the canonical extracted audio.

        A tampered manifest is a hard failure. A missing manifest with an intact
        artifact is a safe recovery (rewrite the marker). A missing or corrupt
        artifact is regenerated only when the source/settings/tool identity still
        matches the frozen job and the regenerated audio hashes exactly to the
        job's recorded audio; otherwise the job fails explicitly.
        """

        origin = stored.config.video_origin
        if origin is None:
            raise StorageCorruptionError("job has no recorded video origin")
        settings = stored.config.video_extraction
        locator = extraction_locator(settings)

        # The extracted audio is a job-owned canonical artifact: its leaf and
        # every parent must be real components inside the job directory, and the
        # recorded audio path must be exactly that frozen confined locator. A
        # symlink (same or different bytes) or an escaping/substituted path is
        # refused here, before any byte/hash reuse or regeneration, so a cached
        # artifact can never be a link to a file the job does not own.
        canonical_path = store.resolve(locator)
        if Path(stored.audio.path) != canonical_path:
            raise StorageCorruptionError(
                "the recorded extracted audio path does not match the job-owned "
                "canonical artifact locator"
            )

        artifact_ok = True
        try:
            recheck_audio(canonical_path, stored.audio)
        except MediaError:
            artifact_ok = False

        manifest = store.read_extraction_manifest(ExtractionManifest)
        if manifest is not None:
            self._verify_extraction_manifest(manifest, origin, settings, locator, stored)

        source_video = video
        if source_video is None:
            source_video = self._revalidate_video_source(origin, stored)

        if artifact_ok:
            if manifest is None:
                artifact = StoredArtifact(
                    locator=locator,
                    sha256=stored.audio.audio_sha256,
                    size_bytes=stored.audio.size_bytes,
                )
                store.write_extraction_manifest(
                    build_extraction_manifest(origin, settings, artifact)
                )
            return stored.audio

        if source_video is None:
            raise VideoMediaError(
                "VIDEO_SOURCE_MISSING",
                "the extracted audio is missing or corrupt and the original video "
                "is unavailable; the artifact cannot be regenerated",
            )
        return self._regenerate_extraction(
            store, stored, origin, settings, locator, source_video
        )

    @staticmethod
    def _verify_extraction_manifest(
        manifest: ExtractionManifest,
        origin: VideoOriginSettings,
        settings: ExtractionSettings,
        locator: str,
        stored: StoredInput,
    ) -> None:
        issues: list[str] = []
        if manifest.extraction_id != origin.extraction_id:
            issues.append("extraction id")
        if manifest.extraction_version != origin.extraction_version:
            issues.append("extraction version")
        if manifest.tool_version != origin.tool_version:
            issues.append("tool version")
        if manifest.source_sha256 != origin.source_sha256:
            issues.append("source hash")
        if manifest.source_size_bytes != origin.source_size_bytes:
            issues.append("source size")
        if (
            manifest.selected_audio_stream_index
            != origin.selected_audio_stream_index
        ):
            issues.append("selected stream")
        if manifest.settings != settings:
            issues.append("extraction settings")
        if manifest.audio.locator != locator:
            issues.append("artifact locator")
        if manifest.audio.sha256 != stored.audio.audio_sha256:
            issues.append("extracted audio hash")
        if manifest.audio.size_bytes != stored.audio.size_bytes:
            issues.append("extracted audio size")
        if issues:
            raise StorageCorruptionError(
                "the extraction manifest does not match the stored video origin: "
                + ", ".join(issues)
            )

    def _revalidate_video_source(
        self, origin: VideoOriginSettings, stored: StoredInput
    ) -> VideoInfo | None:
        source = Path(origin.source_path)
        if not source.exists():
            return None
        source_video = validate_video(
            source,
            limits=stored.config.video,
            ffprobe_bin=self._ffprobe_bin,
            ffmpeg_bin=self._ffmpeg_bin,
        )
        if (
            source_video.video_sha256 != origin.source_sha256
            or source_video.size_bytes != origin.source_size_bytes
        ):
            raise ResumeRefused(
                "the original video source changed after the job was created; "
                "start a new job instead of relabelling the existing one"
            )
        # The installed tool version is part of the extraction identity, so a
        # changed ffmpeg cannot silently re-extract under the old provenance.
        current_tool = ffmpeg_tool_version(
            self._ffmpeg_bin, timeout=stored.config.video.probe_timeout_seconds
        )
        if current_tool != origin.tool_version:
            raise ResumeRefused(
                "the ffmpeg tool version changed after the job was created; "
                "start a new job instead of reusing the old extraction"
            )
        current_id = compute_extraction_id(
            source_video, stored.config.video_extraction, current_tool
        )
        if current_id != origin.extraction_id:
            raise ResumeRefused(
                "the extraction identity changed; start a new job instead of "
                "reusing the old extraction"
            )
        return source_video

    def _regenerate_extraction(
        self,
        store: JobStore,
        stored: StoredInput,
        origin: VideoOriginSettings,
        settings: ExtractionSettings,
        locator: str,
        video: VideoInfo,
    ) -> AudioInfo:
        artifacts_dir = store.resolve("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        temp = artifacts_dir / f".tmp-extract-{secrets.token_hex(4)}.{settings.container}"
        try:
            result = extract_audio(
                video,
                settings=settings,
                out_path=temp,
                ffmpeg_bin=self._ffmpeg_bin,
                tool_version=origin.tool_version,
                limits=stored.config.video,
                media_limits=stored.config.media,
                ffprobe_bin=self._ffprobe_bin,
            )
            if (
                result.audio.audio_sha256 != stored.audio.audio_sha256
                or result.audio.size_bytes != stored.audio.size_bytes
            ):
                raise VideoMediaError(
                    "VIDEO_EXTRACTION_MISMATCH",
                    "the regenerated audio does not match the job's frozen "
                    "extracted audio hash; refusing to relabel the job",
                )
            artifact = store.adopt_file(locator, temp)
            store.write_extraction_manifest(
                build_extraction_manifest(origin, settings, artifact)
            )
        except BaseException:
            try:
                temp.unlink()
            except OSError:
                pass
            raise
        return stored.audio

    def process(
        self,
        audio_path: str,
        *,
        job_id: str | None = None,
        allow_remote_retry: bool = False,
        input_kind: str = "audio",
    ) -> PipelineResult:
        """Validate/create a job and run STT then MT, resuming if needed."""

        store = self.start(audio_path, job_id=job_id, input_kind=input_kind)
        return self.resume(store.job_id, allow_remote_retry=allow_remote_retry)

    def resume(
        self, job_id: str, *, allow_remote_retry: bool = False
    ) -> PipelineResult:
        """Run or resume a job, reusing every verified completed artifact."""

        store = JobStore(self.config.job_root, job_id)
        with store:
            self._run_key = (str(store.root), store.job_id)
            self._route_preflight_done = False
            self._stage_scope = _ALL_STAGES
            stored = store.read_input()
            self._verify_job_matches(stored)
            self._verify_input_media(store, stored)
            self._validate_durable_history(store)
            state = _RunState(store, stored, self._now)
            archive = _RawArchive(store)
            self._archive = archive
            try:
                self._bind_raw_sink(archive)
                transcript = self._ensure_stt(store, stored, state, allow_remote_retry)
                translation = self._ensure_mt(
                    store, stored, state, transcript, allow_remote_retry
                )
            finally:
                self._clear_contexts()
                archive.disarm()
                self._archive = None
            record = store.read_status()
            assert record is not None
            return PipelineResult(
                job_id=job_id,
                status=record.status,
                record=record,
                transcript=transcript,
                translation=translation,
            )

    def transcribe(
        self,
        job_id: str,
        *,
        allow_remote_retry: bool = False,
        before_dispatch: Callable[[], None] | None = None,
    ) -> Transcript:
        """Run only the STT stage for a job that already exists.

        ``before_dispatch`` is an optional caller gate invoked only once a
        genuinely required fresh STT dispatch is about to be prepared: a verified
        completed STT stage is reused, and a blocked/unknown prior attempt still
        raises its replay refusal, without ever running the gate. It runs before
        the route preflight can resolve a credential, reserve budget or send
        HTTP, so a caller that must also satisfy an unrelated paid route (for
        example the Google Basic MT project/key/consent) can fail closed before
        the paid Scribe request is billed.
        """

        store = JobStore(self.config.job_root, job_id)
        with store:
            self._run_key = (str(store.root), store.job_id)
            self._route_preflight_done = False
            self._stage_scope = _STT_STAGE
            stored = store.read_input()
            self._verify_job_matches(stored)
            self._verify_input_media(store, stored)
            self._validate_durable_history(store)
            state = _RunState(store, stored, self._now)
            archive = _RawArchive(store)
            self._archive = archive
            try:
                self._bind_raw_sink(archive)
                transcript = self._ensure_stt(
                    store,
                    stored,
                    state,
                    allow_remote_retry,
                    before_dispatch=before_dispatch,
                )
                self._reconcile_stt_complete(store, state)
                return transcript
            finally:
                self._clear_contexts()
                archive.disarm()
                self._archive = None

    def translate(self, job_id: str, *, allow_remote_retry: bool = False) -> Translation:
        """Run only the MT stage; requires a verified transcript."""

        store = JobStore(self.config.job_root, job_id)
        with store:
            self._run_key = (str(store.root), store.job_id)
            self._route_preflight_done = False
            self._stage_scope = _ALL_STAGES
            stored = store.read_input()
            self._verify_job_matches(stored)
            self._verify_input_media(store, stored)
            self._validate_durable_history(store)
            manifest = store.read_stt_manifest()
            if manifest is None:
                raise PreflightError("cannot translate before a completed STT stage")
            transcript = self._reuse_stt(store, stored, manifest)
            state = _RunState(store, stored, self._now)
            archive = _RawArchive(store)
            self._archive = archive
            try:
                self._bind_raw_sink(archive)
                return self._ensure_mt(store, stored, state, transcript, allow_remote_retry)
            finally:
                self._clear_contexts()
                archive.disarm()
                self._archive = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _release_provider(self, provider: object) -> None:
        """Release only this run's binding on one provider.

        A provider that supports ``clear_run`` releases only when this run owns
        the binding, so another job's acquired adapter is never disturbed. Plain
        fakes fall back to clearing the sink they were given.
        """

        clearer = getattr(provider, "clear_run", None)
        if callable(clearer):
            clearer(self._run_key)
            return
        binder = getattr(provider, "bind_raw_sink", None)
        if callable(binder):
            binder(None)

    def _bind_raw_sink(self, sink: RawSink | None) -> None:
        """Bind the raw sink and run ownership, unwinding a partial acquisition.

        If the first provider binds and a later ``bind_run`` is rejected (for
        example a shared MT adapter still owned by another job), the providers
        acquired by *this* failed attempt are released and the foreign owner is
        left untouched.
        """

        acquired: list[object] = []
        try:
            for provider in (self._stt, self._mt):
                acquired.append(provider)
                if sink is not None:
                    binder_run = getattr(provider, "bind_run", None)
                    if callable(binder_run):
                        binder_run(self._run_key)
                binder = getattr(provider, "bind_raw_sink", None)
                if callable(binder):
                    binder(sink)
        except BaseException:
            for provider in reversed(acquired):
                self._release_provider(provider)
            raise

    @staticmethod
    def _is_api_provider(provider: object) -> bool:
        return callable(getattr(provider, "preflight_route", None))

    @staticmethod
    def _paid_policy(provider: object):
        """The paid-call policy declared by a real adapter.

        The Scribe and Google Basic adapters expose their ``ApiSettings`` as
        ``api``; the Basic adapter's ``settings`` property is the route settings,
        so the policy is read from ``api`` when present.
        """

        api = getattr(provider, "api", None)
        if api is not None:
            return api.paid
        return provider.settings.paid  # type: ignore[attr-defined]

    def _bind_context(self, provider: object, context: ApiContext | None) -> None:
        """Bind the explicit call context on adapters that support it.

        The context carries the durable store, validated audio, immutable config
        and the authoritative stage/group/attempt/segment identity, so a real
        adapter can reject a direct call that lacks them.
        """

        binder = getattr(provider, "bind_context", None)
        if callable(binder):
            binder(context)

    def _clear_contexts(self) -> None:
        """Release only this run's binding; never clear another job's context."""

        for provider in (self._stt, self._mt):
            self._release_provider(provider)

    def _verify_api_ledger(self, store: JobStore) -> None:
        """Validate the durable API ledger and every historical raw ref.

        Called at every resume/reuse entrypoint. A fresh job with no prior API
        evidence initializes the ledger once; a missing ledger/head after prior
        API evidence, a broken hash chain, a single edited/deleted/truncated
        record or a tampered historical raw body all fail closed before any send.
        """

        api_providers = [
            provider
            for provider in (self._stt, self._mt)
            if self._is_api_provider(provider)
        ]
        if not api_providers:
            return
        if not store.api_ledger_initialized():
            prior_evidence = (
                store.has_raw(stage=StageKind.stt)
                or store.has_raw(stage=StageKind.mt)
                or bool(store.read_attempt_intents())
                or bool(store.read_attempts())
            )
            if prior_evidence:
                raise StorageCorruptionError(
                    "durable API history exists but the paid-call ledger is "
                    "missing; refusing to create a fresh budget"
                )
            store.initialize_api_ledger()
        for record in store.read_api_trace():
            if record.raw_reference is not None:
                store.verify_raw(
                    record.raw_reference,
                    stage=record.stage,
                    group_id=record.group_id,
                    attempt=record.pipeline_attempt,
                )

    def _preflight_api_route(
        self, store: JobStore, stored: StoredInput, *, need_stt: bool
    ) -> None:
        """Validate the entire selected paid route before the first dispatch.

        Every selected adapter's local config/argument/limit/paid gates are
        validated first, with **no** credential resolver call and no HTTP. Only
        then are credentials resolved, and only for stages that may actually
        dispatch: a resume whose STT is already verified complete resolves the MT
        credential alone, and an absent STT credential is tolerated because that
        stage will not dispatch (its stored config/audio snapshot is still
        validated locally). The Google route is validated before the first paid STT
        even when the configured hint is Turkish, because a hint is not the
        verified transcript; a genuinely Turkish transcript still skips Google
        HTTP. A disabled policy or an invalid route therefore performs no
        resolver call and no HTTP.

        The active ``_stage_scope`` decides *which* stages the whole-route
        preflight covers: a transcribe-only invocation validates and resolves the
        Scribe route alone and never touches the Google adapter, so a missing,
        unrelated Google project cannot block STT-only work. A process/resume/
        translate invocation covers both routes and still refuses the missing
        Google project before the first Scribe send.
        """

        stt_api = self._is_api_provider(self._stt) and StageKind.stt in self._stage_scope
        mt_api = self._is_api_provider(self._mt) and StageKind.mt in self._stage_scope
        if not stt_api and not mt_api:
            return
        if self._route_preflight_done:
            return
        try:
            # Phase 1: local validation of *every* selected route.
            if stt_api:
                # An already verified, reused STT still has its stored
                # config/audio snapshot validated, but its credential is not
                # required here because it will not dispatch.
                self._stt.preflight_route(  # type: ignore[attr-defined]
                    ApiContext(
                        store=store,
                        stage=StageKind.stt,
                        config=self.config,
                        audio=stored.audio,
                    ),
                    require_credential=need_stt,
                )
            if mt_api:
                self._mt.preflight_route(  # type: ignore[attr-defined]
                    ApiContext(
                        store=store,
                        stage=StageKind.mt,
                        config=self.config,
                        audio=stored.audio,
                    )
                )
            if stt_api:
                budget_preflight(self._paid_policy(self._stt))
            if mt_api:
                budget_preflight(self._paid_policy(self._mt))
            # Phase 2: credentials, only for the stages that may dispatch.
            if need_stt and stt_api:
                self._stt.resolve_credentials()  # type: ignore[attr-defined]
            if mt_api:
                self._mt.resolve_credentials()  # type: ignore[attr-defined]
        except ProviderCallError as exc:
            raise PipelineProviderError(exc.error) from exc
        self._route_preflight_done = True

    def _build_stored_input(
        self,
        audio: AudioInfo,
        job_id: str,
        *,
        video: VideoInfo | None = None,
    ) -> StoredInput:
        now = self._now()
        config = self.config
        job_input = JobInput(
            job_id=job_id,
            created_at_utc=now,
            audio_path=audio.path,
            original_filename=audio.original_filename,
            audio_sha256=audio.audio_sha256,
            audio_duration_ms=audio.duration_ms,
            audio_format=audio.container,
            audio_codec=audio.codec_name,
            audio_channels=audio.channels,
            audio_sample_rate=audio.sample_rate,
            audio_size_bytes=audio.size_bytes,
            video_path=video.path if video is not None else None,
            video_sha256=video.video_sha256 if video is not None else None,
            source_language=config.source_language_hint,
            stt=config.stt,
            mt=config.mt,
            target_language=config.target_language,
            keyterms=config.keyterms,
            keyterms_version=config.keyterms_version,
            segmenter_version=config.segmenter_version,
        )
        return StoredInput(
            input=job_input,
            config=config,
            audio=audio,
            config_fingerprint=config.config_fingerprint(),
            job_fingerprint=config.job_fingerprint(audio.audio_sha256),
            stored_at_utc=now,
        )

    def _verify_job_matches(self, stored: StoredInput) -> None:
        # Reading StoredInput already rejected unknown versions and internal
        # snapshot inconsistencies; here we only reconcile with the live config.
        audio_sha = stored.audio.audio_sha256
        if stored.config_fingerprint != self.config.config_fingerprint():
            raise ResumeRefused(
                "stored configuration fingerprint differs from the current config; "
                "start a new job instead of resuming"
            )
        if stored.job_fingerprint != self.config.job_fingerprint(audio_sha):
            raise ResumeRefused(
                "stored job fingerprint (config + audio) differs from the current "
                "config; start a new job instead of resuming"
            )
        if stored.input.stt != self.config.stt or stored.input.mt != self.config.mt:
            raise ResumeRefused("stored provider identities differ from the current config")

    def _verify_caller_audio(self, stored: StoredInput, audio: AudioInfo) -> None:
        if (
            stored.audio.audio_sha256 != audio.audio_sha256
            or stored.audio.size_bytes != audio.size_bytes
        ):
            raise ResumeRefused(
                f"caller-supplied audio differs from the audio recorded for job "
                f"{stored.input.job_id!r} (stored sha256={stored.audio.audio_sha256[:12]}.., "
                f"size={stored.audio.size_bytes}); refusing to ignore the new input"
            )
        # Content-identical audio at a different path is deliberately accepted:
        # the stored path remains authoritative and resume re-checks it.

    # ------------------------------------------------------------------ #
    # Durable history validation and attempt numbering
    # ------------------------------------------------------------------ #
    def _identity_for(self, stage: StageKind) -> ProviderIdentity:
        return self.config.stt if stage is StageKind.stt else self.config.mt

    def _validate_durable_history(self, store: JobStore) -> None:
        """Parse every append-only store and manifest before any provider call.

        Corrupt intent/outcome/raw histories or an invalid manifest fail closed
        here, so a reuse path can never silently ignore a damaged audit trail or
        a tampered manifest. Recorded provider/model identities must agree with
        the configured identity for their stage.
        """

        for intent in store.read_attempt_intents():
            expected = self._identity_for(intent.stage)
            if intent.provider != expected.provider or intent.model != expected.model:
                raise StorageCorruptionError(
                    f"attempt intent {intent.attempt} for {intent.stage.value} names "
                    f"{intent.provider!r}/{intent.model!r}, not the configured "
                    f"({expected.provider!r}/{expected.model!r})"
                )
        for record in store.read_attempts():
            expected = self._identity_for(record.stage)
            if record.provider != expected.provider or record.model != expected.model:
                raise StorageCorruptionError(
                    f"attempt outcome {record.attempt} for {record.stage.value} "
                    f"names {record.provider!r}/{record.model!r}, not the configured "
                    f"({expected.provider!r}/{expected.model!r})"
                )
        store.raw_registrations()
        store.read_stt_manifest()
        store.read_mt_manifest()
        self._verify_api_ledger(store)

    def _next_attempt(
        self,
        store: JobStore,
        *,
        stage: StageKind,
        group_id: str | None,
        floor: int = 0,
    ) -> int:
        """Allocate the next attempt number from durable evidence.

        The convenience status/manifest may lag or have been removed, so the
        maximum of every recorded intent, every recorded outcome and any supplied
        manifest floor is authoritative. A hard-killed attempt therefore never
        reuses its number.
        """

        highest = max(0, floor)
        for intent in store.read_attempt_intents(stage=stage):
            if intent.group_id == group_id:
                highest = max(highest, intent.attempt)
        for record in store.read_attempts(stage=stage):
            if record.group_id == group_id:
                highest = max(highest, record.attempt)
        return highest + 1

    def _reconcile_stt_complete(self, store: JobStore, state: _RunState) -> None:
        """Close a lagging convenience status after a verified STT reuse.

        Never downgrades a job that already reached MT running/complete, and
        never overrides a complete/skipped MT manifest.
        """

        current = store.read_status()
        if current is not None and current.status in {
            JobStatus.mt_running,
            JobStatus.complete,
        }:
            return
        # Any downstream MT manifest means the job moved past STT; never rewrite
        # its state from the STT-only endpoint.
        if store.read_mt_manifest() is not None:
            return
        if current is None or current.status is not JobStatus.stt_complete:
            state.write(JobStatus.stt_complete)

    def _reconcile_mt_complete(self, store: JobStore, state: _RunState) -> None:
        """Close a lagging convenience status from a verified complete manifest."""

        current = store.read_status()
        if current is not None and current.status is JobStatus.complete:
            return
        state.write(JobStatus.complete)

    def _mt_durable_evidence(self, store: JobStore, group_ids: Sequence[str]) -> bool:
        """Whether any durable artifact proves a prior MT stage actually started."""

        if store.has_raw(stage=StageKind.mt):
            return True
        if store.artifact_exists(F_TRANSLATION):
            return True
        if store.read_attempt_intents(stage=StageKind.mt):
            return True
        if store.read_attempts(stage=StageKind.mt):
            return True
        return any(store.artifact_exists(store.group_locator(gid)) for gid in group_ids)

    def _group_has_durable_evidence(self, store: JobStore, group_id: str) -> bool:
        """Whether one MT group has durable dispatch/raw/output evidence."""

        if any(
            record.group_id == group_id
            for record in store.raw_registrations(stage=StageKind.mt)
        ):
            return True
        if any(
            intent.group_id == group_id
            for intent in store.read_attempt_intents(stage=StageKind.mt)
        ):
            return True
        return any(
            record.group_id == group_id
            for record in store.read_attempts(stage=StageKind.mt)
        )

    # ------------------------------------------------------------------ #
    # STT stage
    # ------------------------------------------------------------------ #
    def _ensure_stt(
        self,
        store: JobStore,
        stored: StoredInput,
        state: _RunState,
        allow_remote_retry: bool,
        *,
        before_dispatch: Callable[[], None] | None = None,
    ) -> Transcript:
        manifest = store.read_stt_manifest()
        if manifest is not None:
            return self._reuse_stt(store, stored, manifest)

        intents = store.read_attempt_intents(stage=StageKind.stt)
        attempts = store.read_attempts(stage=StageKind.stt)
        outcome_attempts = {record.attempt for record in attempts}
        dangling_intents = [i for i in intents if i.attempt not in outcome_attempts]
        durable_unknown = bool(dangling_intents) or any(
            record.outcome is AttemptOutcome.remote_unknown
            or (
                record.outcome is AttemptOutcome.failed
                and record.error is not None
                and record.error.remote_status_unknown
            )
            for record in attempts
        )
        durable_activity = bool(intents) or bool(attempts)

        dispatch = state.get_meta("stt_dispatch")
        initial = state.initial_status
        status_now = initial.status if initial is not None else None
        running = status_now in {JobStatus.stt_running, JobStatus.remote_status_unknown}
        has_orphan = store.artifact_exists(F_TRANSCRIPT)
        has_raw = store.has_raw(stage=StageKind.stt)
        committed = status_now in {
            JobStatus.stt_complete,
            JobStatus.mt_running,
            JobStatus.complete,
        }
        if committed and not has_orphan and not has_raw:
            raise StorageCorruptionError(
                "status claims STT was completed but no verifiable STT manifest exists"
            )

        # A missing status.json for a job that already has durable dispatch
        # evidence is a deleted/rewritten convenience file, never a fresh job: the
        # append-only history and raw archive decide. A genuinely fresh created
        # job has no such evidence and may dispatch normally.
        deleted_status = status_now is None and (durable_activity or has_raw or has_orphan)
        unknown_evidence = (
            running
            or dispatch == _DISPATCH_UNKNOWN
            or durable_unknown
            or deleted_status
        )
        blocked_evidence = has_orphan or has_raw or dispatch == _DISPATCH_BLOCKED
        evidence = unknown_evidence or blocked_evidence

        if evidence:
            if not allow_remote_retry:
                if unknown_evidence:
                    error = StageError(
                        code="REMOTE_STATUS_UNKNOWN",
                        message="a previous STT call may have reached the provider; "
                        "explicit allow_remote_retry is required to replay it",
                        remote_status_unknown=True,
                    )
                    state.add_error(error)
                    state.set_meta("stt_dispatch", _DISPATCH_UNKNOWN)
                    state.write(JobStatus.remote_status_unknown)
                    raise RemoteStatusUnknownError(
                        "STT remote status unknown; refusing automatic replay",
                        stage=StageKind.stt,
                    )
                error = StageError(
                    code="STT_PENDING_REVIEW",
                    message="a previous STT response was received but is not verifiable "
                    "(orphan/unregistered output); refusing automatic replay without "
                    "allow_remote_retry",
                )
                state.add_error(error)
                state.set_meta("stt_dispatch", _DISPATCH_BLOCKED)
                state.write(JobStatus.needs_review)
                raise ReplayRefused(
                    "STT result is pending review; refusing automatic replay",
                    stage=StageKind.stt,
                )
            state.note_retry_decision(
                scope="stt", group_id=None, reason="explicit_allow_remote_retry"
            )
            if has_orphan:
                # Preserve the unanchored orphan before deliberately overwriting it.
                store.preserve_orphan(F_TRANSCRIPT)

        # A genuinely required fresh STT dispatch is about to be prepared. The
        # optional caller gate runs here -- after a verified completed stage has
        # been reused, and after every blocked/unknown replay check has already
        # decided or been explicitly authorized, but before the route preflight
        # can resolve a credential, reserve budget or send HTTP. An offline reuse
        # therefore never runs it, and a paid Scribe request cannot be billed
        # only for an unrelated paid route to be found unconfigured afterwards.
        if before_dispatch is not None:
            before_dispatch()

        try:
            self._preflight_api_route(store, stored, need_stt=True)
        except PipelineProviderError as exc:
            state.add_error(exc.error)
            state.write(
                JobStatus.failed if exc.error.retryable else JobStatus.needs_review
            )
            raise
        state.set_meta("stt_dispatch", _DISPATCH_UNKNOWN)
        state.write(JobStatus.stt_running)

        attempt = self._next_attempt(store, stage=StageKind.stt, group_id=None)
        state.set_meta("stt_attempt", attempt)
        started = self._now()
        mono0 = self._monotonic()
        hint = self.config.source_language_hint
        keyterms = self.config.keyterms if self.config.stage_options.stt_send_keyterms else None
        replay_reason = (
            "explicit_allow_remote_retry" if allow_remote_retry and evidence else None
        )
        assert self._archive is not None
        self._archive.arm(stage=StageKind.stt, group_id=None, attempt=attempt)
        self._bind_context(
            self._stt,
            ApiContext(
                store=store,
                stage=StageKind.stt,
                config=self.config,
                audio=stored.audio,
                pipeline_attempt=attempt,
            ),
        )
        store.append_attempt_intent(
            AttemptIntent(
                stage=StageKind.stt,
                group_id=None,
                attempt=attempt,
                provider=self.config.stt.provider,
                model=self.config.stt.model,
                replay_reason=replay_reason,
                started_at_utc=started,
                declared_at_utc=self._now(),
            )
        )
        try:
            transcript = self._stt.transcribe(
                stored.audio.path, language_hint=hint, keyterms=keyterms
            )
        except ProviderCallError as exc:
            self._record_stt_failure(
                state,
                store,
                exc.error,
                started,
                mono0,
                dispatched=exc.dispatched,
                attempt=attempt,
                replay_reason=replay_reason,
            )
            if exc.remote_status_unknown:
                raise RemoteStatusUnknownError(
                    "STT remote status unknown; refusing automatic replay",
                    stage=StageKind.stt,
                ) from exc
            raise PipelineProviderError(exc.error) from exc
        except KeyboardInterrupt:
            state.set_meta("stt_dispatch", _DISPATCH_UNKNOWN)
            state.write(JobStatus.interrupted)
            raise
        except BaseException as exc:  # ambiguous: conservatively unknown
            error = StageError(
                code="PROVIDER_CALL_AMBIGUOUS",
                message=f"ambiguous STT failure: {exc}",
                remote_status_unknown=True,
            )
            self._record_stt_failure(
                state, store, error, started, mono0, dispatched=True, attempt=attempt
            )
            raise RemoteStatusUnknownError(
                "STT failed ambiguously; remote status stays unknown",
                stage=StageKind.stt,
            ) from exc

        # Require a verified raw body BEFORE committing normalized success. The
        # response was received, so a validation failure here is a known outcome
        # and is *not* retried automatically.
        try:
            self._require_raw(
                store,
                transcript.raw_reference,
                stage=StageKind.stt,
                group_id=None,
                attempt=attempt,
            )
            self._validate_stt_result(transcript, stored)
        except ProviderResultError as exc:
            self._record_stt_failure(
                state, store, exc.error, started, mono0, dispatched=True, attempt=attempt
            )
            raise
        except StorageCorruptionError as exc:
            error = StageError(code="STT_RAW_INVALID", message=str(exc))
            self._record_stt_failure(
                state, store, error, started, mono0, dispatched=True, attempt=attempt
            )
            raise

        elapsed_ms = max(0, int((self._monotonic() - mono0) * 1000))
        self._commit_stt(
            store,
            stored,
            state,
            transcript,
            started=started,
            elapsed_ms=elapsed_ms,
            attempt=attempt,
        )
        self._record_stt_success(store, transcript, started, elapsed_ms, attempt)
        state.set_meta("stt_dispatch", _DISPATCH_NONE)
        state.write(JobStatus.stt_complete)
        return transcript

    def _record_stt_failure(
        self,
        state: _RunState,
        store: JobStore,
        error: StageError,
        started: datetime,
        mono0: float,
        *,
        dispatched: bool,
        attempt: int,
        replay_reason: str | None = None,
    ) -> None:
        finished = self._now()
        elapsed_ms = max(0, int((self._monotonic() - mono0) * 1000))
        state.record_stage(
            StageResult(
                stage=StageKind.stt,
                provider=self.config.stt.provider,
                model=self.config.stt.model,
                started_at_utc=started,
                finished_at_utc=finished,
                elapsed_ms=elapsed_ms,
                source_char_count=0,
                target_char_count=0,
                error=error,
            )
        )
        state.add_error(error)
        outcome = (
            AttemptOutcome.remote_unknown
            if error.remote_status_unknown
            else AttemptOutcome.failed
        )
        store.append_attempt(
            AttemptRecord(
                stage=StageKind.stt,
                group_id=None,
                attempt=attempt,
                provider=self.config.stt.provider,
                model=self.config.stt.model,
                request_id=None,
                raw_reference=None,
                source_char_count=0,
                target_char_count=0,
                started_at_utc=started,
                finished_at_utc=finished,
                elapsed_ms=elapsed_ms,
                outcome=outcome,
                error=error,
                replay_reason=replay_reason,
                recorded_at_utc=finished,
            )
        )
        if error.remote_status_unknown:
            state.set_meta("stt_dispatch", _DISPATCH_UNKNOWN)
            state.write(JobStatus.remote_status_unknown)
        elif not dispatched and error.retryable:
            state.set_meta("stt_dispatch", _DISPATCH_RETRYABLE_PREDISPATCH)
            state.write(JobStatus.failed)
        else:
            state.set_meta("stt_dispatch", _DISPATCH_BLOCKED)
            state.write(JobStatus.needs_review)

    def _record_stt_success(
        self,
        store: JobStore,
        transcript: Transcript,
        started: datetime,
        elapsed_ms: int,
        attempt: int,
    ) -> None:
        finished = self._now()
        store.append_attempt(
            AttemptRecord(
                stage=StageKind.stt,
                group_id=None,
                attempt=attempt,
                provider=transcript.provider,
                model=transcript.model,
                request_id=transcript.request_id,
                raw_reference=transcript.raw_reference,
                source_char_count=0,
                target_char_count=sum(len(segment.source_text) for segment in transcript.segments),
                started_at_utc=started,
                finished_at_utc=finished,
                elapsed_ms=elapsed_ms,
                outcome=AttemptOutcome.complete,
                recorded_at_utc=finished,
            )
        )

    def _commit_stt(
        self,
        store: JobStore,
        stored: StoredInput,
        state: _RunState,
        transcript: Transcript,
        *,
        started: datetime | None,
        elapsed_ms: int | None,
        attempt: int,
    ) -> None:
        artifact = store.write_transcript(transcript)
        raw_reference = transcript.raw_reference
        if raw_reference is None:
            raise StorageCorruptionError("cannot commit STT without a raw body reference")
        manifest = STTManifest(
            provider=transcript.provider,
            model=transcript.model,
            audio_sha256=stored.audio.audio_sha256,
            config_fingerprint=stored.config_fingerprint,
            job_fingerprint=stored.job_fingerprint,
            transcript=artifact,
            raw_reference=raw_reference,
            completed_at_utc=self._now(),
            attempt=attempt,
        )
        store.write_stt_manifest(manifest)
        if started is not None:
            finished = self._now()
            state.record_stage(
                StageResult(
                    stage=StageKind.stt,
                    provider=transcript.provider,
                    model=transcript.model,
                    started_at_utc=started,
                    finished_at_utc=finished,
                    elapsed_ms=elapsed_ms if elapsed_ms is not None else 0,
                    source_char_count=0,
                    target_char_count=sum(len(segment.source_text) for segment in transcript.segments),
                )
            )

    def _reuse_stt(
        self, store: JobStore, stored: StoredInput, manifest: STTManifest
    ) -> Transcript:
        self._verify_manifest_identity(
            manifest.job_fingerprint,
            manifest.config_fingerprint,
            manifest.audio_sha256,
            stored,
        )
        if manifest.provider != self.config.stt.provider or manifest.model != self.config.stt.model:
            raise ResumeRefused("STT manifest provider/model differs from the current config")
        data = store.verify_artifact(manifest.transcript)
        try:
            transcript = Transcript.model_validate_json(data)
        except ValueError as exc:
            raise StorageCorruptionError(f"stored transcript is invalid: {exc}") from exc
        self._validate_stt_result(transcript, stored)
        if transcript.raw_reference is None:
            raise StorageCorruptionError("stored transcript has no raw body reference")
        if transcript.raw_reference != manifest.raw_reference:
            raise StorageCorruptionError(
                "stored transcript raw reference does not exactly match the manifest"
            )
        store.verify_raw(
            manifest.raw_reference,
            stage=StageKind.stt,
            group_id=None,
            attempt=manifest.attempt,
        )
        self._verify_supplementary(
            store,
            StageKind.stt,
            transcript.request_metadata,
            group_id=None,
            attempt=manifest.attempt,
        )
        return transcript

    def _validate_stt_result(self, transcript: Transcript, stored: StoredInput) -> None:
        issues: list[str] = []
        if transcript.provider != self.config.stt.provider:
            issues.append(
                f"result provider {transcript.provider!r} != {self.config.stt.provider!r}"
            )
        if transcript.model != self.config.stt.model:
            issues.append(f"result model {transcript.model!r} != {self.config.stt.model!r}")
        if transcript.audio_duration_ms <= 0:
            issues.append("audio_duration_ms must be positive")
        tolerance = max(1000, stored.audio.duration_ms // 100)
        if transcript.audio_duration_ms > stored.audio.duration_ms + tolerance:
            issues.append(
                f"transcript duration {transcript.audio_duration_ms}ms exceeds the "
                f"validated media duration {stored.audio.duration_ms}ms"
            )
        hint = self.config.source_language_hint
        if hint is not None and transcript.source_language not in (None, hint):
            if not transcript.language_uncertain:
                issues.append(
                    f"transcript language {transcript.source_language!r} collides with "
                    f"the explicit hint {hint!r}"
                )
        if issues:
            raise ProviderResultError(
                StageError(code="STT_RESULT_INVALID", message="; ".join(issues))
            )

    def _require_raw(
        self,
        store: JobStore,
        reference: RawArtifactRef | None,
        *,
        stage: StageKind,
        group_id: str | None,
        attempt: int | None,
    ) -> None:
        if reference is None:
            raise ProviderResultError(
                StageError(
                    code="RAW_REQUIRED",
                    message=f"{stage.value} result has no archived raw body; refusing "
                    "to commit normalized output",
                )
            )
        issued = self._archive.issued if self._archive is not None else None
        if issued is None or reference != issued:
            raise ProviderResultError(
                StageError(
                    code="RAW_UNVERIFIED",
                    message=f"{stage.value} result raw reference was not produced by this "
                    "job's archive for the current dispatch; refusing to commit "
                    "normalized output",
                )
            )
        store.verify_raw(reference, stage=stage, group_id=group_id, attempt=attempt)

    def _verify_supplementary(
        self,
        store: JobStore,
        stage: StageKind,
        metadata: object,
        *,
        group_id: str | None,
        attempt: int | None,
    ) -> None:
        """Independently re-verify every grouped response / review re-run body.

        The primary raw reference only covers the *last* body. Each earlier
        grouped provider response and each STT review re-run must be registered
        and intact on every reuse path, or the job fails closed.
        """

        references = collect_supplementary_refs(stage, metadata)
        if not references:
            return
        verify_supplementary_raw(
            store,
            references,
            stage=stage,
            group_id=group_id,
            attempt=attempt,
        )

    # ------------------------------------------------------------------ #
    # MT stage
    # ------------------------------------------------------------------ #
    def _resolve_source_language(self, transcript: Transcript) -> str | None:
        hint = self.config.source_language_hint
        segment_languages = {segment.source_language for segment in transcript.segments}
        if len(segment_languages) > 1:
            return None
        segment_language = next(iter(segment_languages), None)
        declared = transcript.source_language
        if transcript.language_uncertain:
            if hint is not None and segment_language in (None, hint):
                return hint
            return None
        if declared is not None:
            if hint is not None and declared != hint:
                return None
            return declared
        if hint is not None:
            return hint if segment_language in (None, hint) else None
        return segment_language

    def _ensure_mt(
        self,
        store: JobStore,
        stored: StoredInput,
        state: _RunState,
        transcript: Transcript,
        allow_remote_retry: bool,
    ) -> Translation:
        mono0 = self._monotonic()
        started = self._now()
        source = self._resolve_source_language(transcript)
        if source is None:
            error = StageError(
                code="LANGUAGE_UNCERTAIN",
                message="source language is uncertain or mixed and no explicit compatible "
                "hint was configured; refusing to guess before MT",
            )
            state.add_error(error)
            state.write(JobStatus.needs_review)
            raise PipelineNeedsReviewError(error.message)

        segments = transcript.segments
        manifest = store.read_mt_manifest()
        if manifest is not None:
            self._verify_mt_manifest_identity(manifest, stored, source)
        if source == TARGET_LANGUAGE_CODE:
            return self._ensure_skipped_mt(
                store, stored, state, transcript, source, started, mono0, manifest
            )

        try:
            groups = group_segments(segments, batch=self.config.batch)
        except SegmentGroupingError as exc:
            error = StageError(code="MT_SEGMENT_TOO_LONG", message=str(exc))
            state.add_error(error)
            state.write(JobStatus.needs_review)
            raise
        group_segments_by_id = {
            _group_id(index): group for index, group in enumerate(groups)
        }

        if manifest is None:
            if (
                state.initial_status is not None
                and state.initial_status.status is JobStatus.complete
            ):
                raise StorageCorruptionError(
                    "status claims the job is complete but no MT manifest exists"
                )
            if self._mt_durable_evidence(store, tuple(group_segments_by_id)):
                raise StorageCorruptionError(
                    "MT manifest is missing but durable MT history/artifacts exist; "
                    "refusing to reconstruct paid work from an empty manifest"
                )
        records: dict[str, MTGroupRecord] = (
            {record.group_id: record for record in manifest.groups}
            if manifest is not None
            else {}
        )

        # A group that has durable dispatch/raw/output evidence but no manifest
        # record means the convenience manifest lost an entry (selective deletion
        # or truncation): never resubmit it blindly.
        for group_id in group_segments_by_id:
            if records.get(group_id) is None and self._group_has_durable_evidence(
                store, group_id
            ):
                raise StorageCorruptionError(
                    f"MT group {group_id!r} has durable history but no manifest "
                    "record; refusing to resubmit it"
                )

        # Verify every already-complete group before any new call (tamper => stop).
        for group_id, group in group_segments_by_id.items():
            record = records.get(group_id)
            if record is not None and record.status is MTGroupStatus.complete:
                self._verify_group_artifact(store, record, group)

        if manifest is not None and manifest.status == "complete":
            translation = self._load_combined_translation(store, stored, source, segments)
            self._reconcile_mt_complete(store, state)
            return translation

        def _is_unknown(record: MTGroupRecord) -> bool:
            return record.status is MTGroupStatus.running or (
                record.status is MTGroupStatus.failed
                and record.error is not None
                and record.error.remote_status_unknown
            )

        def _is_blocked(record: MTGroupRecord) -> bool:
            return (
                record.status is MTGroupStatus.failed
                and not record.auto_resumable
                and not (record.error is not None and record.error.remote_status_unknown)
            )

        unknown = [
            group_id
            for group_id, record in records.items()
            if group_id in group_segments_by_id and _is_unknown(record)
        ]
        blocked = [
            group_id
            for group_id, record in records.items()
            if group_id in group_segments_by_id and _is_blocked(record)
        ]
        if (unknown or blocked) and not allow_remote_retry:
            if unknown:
                error = StageError(
                    code="REMOTE_STATUS_UNKNOWN",
                    message="MT group(s) may have reached the provider: "
                    f"{sorted(unknown)}; explicit allow_remote_retry is required",
                    remote_status_unknown=True,
                )
                state.add_error(error)
                state.write(JobStatus.remote_status_unknown)
                raise RemoteStatusUnknownError(
                    "MT remote status unknown for in-flight groups; refusing replay",
                    stage=StageKind.mt,
                    group_id=unknown[0],
                )
            error = StageError(
                code="MT_PENDING_REVIEW",
                message="MT group(s) have a received but unverifiable or failed "
                f"response: {sorted(blocked)}; explicit allow_remote_retry is required",
            )
            state.add_error(error)
            state.write(JobStatus.needs_review)
            raise ReplayRefused(
                "MT result is pending review; refusing automatic replay",
                stage=StageKind.mt,
                group_id=blocked[0],
            )
        for group_id in unknown + blocked:
            state.note_retry_decision(
                scope="mt_group", group_id=group_id, reason="explicit_allow_remote_retry"
            )

        try:
            self._preflight_api_route(store, stored, need_stt=False)
        except PipelineProviderError as exc:
            state.add_error(exc.error)
            state.write(
                JobStatus.failed if exc.error.retryable else JobStatus.needs_review
            )
            raise
        state.write(JobStatus.mt_running)
        self._persist_mt_manifest(
            store, stored, source, tuple(records.values()), status="running"
        )

        completed: list[Translation] = []
        for group_id, group in group_segments_by_id.items():
            record = records.get(group_id)
            if record is not None and record.status is MTGroupStatus.complete:
                completed.append(self._load_group_translation(store, record))
                continue
            replay_reason = (
                "explicit_allow_remote_retry"
                if group_id in unknown or group_id in blocked
                else None
            )
            translation, record = self._translate_group(
                store,
                stored,
                source,
                group_id,
                group,
                records,
                state,
                replay_reason,
            )
            completed.append(translation)

        combined = self._combine_translations(source, completed)
        combined.validate_against([segment.segment_id for segment in segments])
        self._validate_translation_provenance(combined, segments)
        translation_artifact = store.write_translation(combined)
        self._persist_mt_manifest(
            store,
            stored,
            source,
            tuple(records.values()),
            status="complete",
            translation=translation_artifact,
        )
        state.record_stage(
            StageResult(
                stage=StageKind.mt,
                provider=self.config.mt.provider,
                model=self.config.mt.model,
                started_at_utc=started,
                finished_at_utc=self._now(),
                elapsed_ms=max(0, int((self._monotonic() - mono0) * 1000)),
                source_char_count=sum(len(_segment_text(segment)) for segment in segments),
                target_char_count=sum(
                    len(segment.translated_text_tr) for segment in combined.segments
                ),
            )
        )
        state.write(JobStatus.complete)
        return combined

    def _ensure_skipped_mt(
        self,
        store: JobStore,
        stored: StoredInput,
        state: _RunState,
        transcript: Transcript,
        source: str,
        started: datetime,
        mono0: float,
        manifest: MTManifest | None,
    ) -> Translation:
        """Reuse a verified same-language translation or produce it locally.

        A same-language result makes no provider call, but a stored one is still
        verified before reuse, so a tampered local translation is reported as
        corruption instead of being silently repaired by a rewrite.
        """

        if manifest is not None and manifest.translation is not None:
            translation = self._verify_skipped_translation(
                store, manifest, transcript, source
            )
            self._reconcile_mt_complete(store, state)
            return translation
        if manifest is None and self._mt_durable_evidence(store, ()):
            raise StorageCorruptionError(
                "MT manifest is missing but durable MT artifacts exist; refusing "
                "to rebuild a same-language translation"
            )
        translation = self._run_skipped_mt(
            store, stored, state, transcript, source, started, mono0
        )
        self._reconcile_mt_complete(store, state)
        return translation

    def _verify_skipped_translation(
        self,
        store: JobStore,
        manifest: MTManifest,
        transcript: Transcript,
        source: str,
    ) -> Translation:
        if manifest.translation is None:
            raise StorageCorruptionError(
                "same-language manifest has no translation artifact"
            )
        data = store.verify_artifact(manifest.translation)
        try:
            translation = Translation.model_validate_json(data)
        except ValueError as exc:
            raise StorageCorruptionError(
                f"stored same-language translation is invalid: {exc}"
            ) from exc
        if (
            translation.provider != self.config.mt.provider
            or translation.model != self.config.mt.model
            or translation.source_language != source
            or translation.status is not TranslationStatus.skipped_same_language
        ):
            raise StorageCorruptionError(
                "stored same-language translation fails identity checks"
            )
        translation.validate_against(
            [segment.segment_id for segment in transcript.segments]
        )
        return translation

    def _run_skipped_mt(
        self,
        store: JobStore,
        stored: StoredInput,
        state: _RunState,
        transcript: Transcript,
        source: str,
        started: datetime,
        mono0: float,
    ) -> Translation:
        translation = self._mt.translate(transcript.segments, source, TARGET_LANGUAGE_CODE)
        if translation.status is not TranslationStatus.skipped_same_language:
            raise ProviderResultError(
                StageError(
                    code="SKIP_STATE_INVALID",
                    message="same-language translation must be skipped_same_language",
                )
            )
        translation.validate_against([segment.segment_id for segment in transcript.segments])
        translation_artifact = store.write_translation(translation)
        manifest = MTManifest(
            provider=self.config.mt.provider,
            model=self.config.mt.model,
            source_language=source,
            target_language=TARGET_LANGUAGE_CODE,
            audio_sha256=stored.audio.audio_sha256,
            config_fingerprint=stored.config_fingerprint,
            job_fingerprint=stored.job_fingerprint,
            status="skipped",
            groups=(),
            translation=translation_artifact,
            updated_at_utc=self._now(),
        )
        store.write_mt_manifest(manifest)
        state.record_stage(
            StageResult(
                stage=StageKind.mt,
                provider=self.config.mt.provider,
                model=self.config.mt.model,
                started_at_utc=started,
                finished_at_utc=self._now(),
                elapsed_ms=max(0, int((self._monotonic() - mono0) * 1000)),
                source_char_count=sum(len(segment.source_text) for segment in transcript.segments),
                target_char_count=sum(
                    len(segment.translated_text_tr) for segment in translation.segments
                ),
            )
        )
        state.write(JobStatus.complete)
        return translation

    def _translate_group(
        self,
        store: JobStore,
        stored: StoredInput,
        source: str,
        group_id: str,
        group: Sequence[Segment],
        records: dict[str, MTGroupRecord],
        state: _RunState,
        replay_reason: str | None,
    ) -> tuple[Translation, MTGroupRecord]:
        previous = records.get(group_id)
        attempt = self._next_attempt(
            store,
            stage=StageKind.mt,
            group_id=group_id,
            floor=previous.attempt if previous is not None else 0,
        )
        started = self._now()
        mono0 = self._monotonic()
        running = MTGroupRecord(
            group_id=group_id,
            segment_ids=tuple(segment.segment_id for segment in group),
            provider=self.config.mt.provider,
            model=self.config.mt.model,
            status=MTGroupStatus.running,
            attempt=attempt,
            source_char_count=sum(len(_segment_text(segment)) for segment in group),
            target_char_count=0,
            started_at_utc=started,
            remote_may_have_run=True,
            auto_resumable=False,
            replay_reason=replay_reason,
        )
        records[group_id] = running
        self._persist_mt_manifest(store, stored, source, tuple(records.values()), status="running")
        store.append_attempt_intent(
            AttemptIntent(
                stage=StageKind.mt,
                group_id=group_id,
                attempt=attempt,
                provider=self.config.mt.provider,
                model=self.config.mt.model,
                replay_reason=replay_reason,
                started_at_utc=started,
                declared_at_utc=self._now(),
            )
        )

        assert self._archive is not None
        self._archive.arm(stage=StageKind.mt, group_id=group_id, attempt=attempt)
        self._bind_context(
            self._mt,
            ApiContext(
                store=store,
                stage=StageKind.mt,
                config=self.config,
                audio=stored.audio,
                group_id=group_id,
                pipeline_attempt=attempt,
                expected_segments=tuple(
                    ExpectedSegment(
                        segment_id=segment.segment_id,
                        source_text=segment.source_text,
                        translation_input=segment.translation_input,
                    )
                    for segment in group
                ),
                source_language=source,
                target_language=TARGET_LANGUAGE_CODE,
            ),
        )
        try:
            translation = self._mt.translate(list(group), source, TARGET_LANGUAGE_CODE)
        except ProviderCallError as exc:
            self._record_group_failure(
                records,
                store,
                stored,
                source,
                group_id,
                exc.error,
                started,
                mono0,
                dispatched=exc.dispatched,
                replay_reason=replay_reason,
            )
            state.add_error(exc.error)
            if exc.remote_status_unknown:
                state.write(JobStatus.remote_status_unknown)
                raise RemoteStatusUnknownError(
                    "MT remote status unknown; refusing automatic replay",
                    stage=StageKind.mt,
                    group_id=group_id,
                ) from exc
            if exc.safely_resumable:
                state.write(JobStatus.failed)
            else:
                state.write(JobStatus.needs_review)
            raise PipelineProviderError(exc.error) from exc
        except KeyboardInterrupt:
            state.write(JobStatus.interrupted)
            raise
        except BaseException as exc:  # ambiguous: conservatively unknown
            error = StageError(
                code="PROVIDER_CALL_AMBIGUOUS",
                message=f"ambiguous MT failure: {exc}",
                remote_status_unknown=True,
            )
            self._record_group_failure(
                records, store, stored, source, group_id, error, started, mono0, dispatched=True
            )
            state.add_error(error)
            state.write(JobStatus.remote_status_unknown)
            raise RemoteStatusUnknownError(
                "MT failed ambiguously; remote status stays unknown",
                stage=StageKind.mt,
                group_id=group_id,
            ) from exc

        if translation.status is TranslationStatus.failed:
            error = translation.error or StageError(
                code="MT_FAILED", message=f"group {group_id} reported failure"
            )
            self._record_group_failure(
                records,
                store,
                stored,
                source,
                group_id,
                error,
                started,
                mono0,
                dispatched=True,
                replay_reason=replay_reason,
            )
            state.add_error(error)
            if error.remote_status_unknown:
                state.write(JobStatus.remote_status_unknown)
                raise RemoteStatusUnknownError(
                    "MT result reports an unknown remote status",
                    stage=StageKind.mt,
                    group_id=group_id,
                )
            state.write(JobStatus.needs_review)
            raise PipelineProviderError(error)

        # The response was received: any validation failure is a known outcome and
        # is refused by an ordinary resume (no silent re-billing).
        try:
            self._require_raw(
                store,
                translation.raw_reference,
                stage=StageKind.mt,
                group_id=group_id,
                attempt=attempt,
            )
            if translation.status is not TranslationStatus.translated:
                raise ProviderResultError(
                    StageError(
                        code="MT_STATUS_INVALID",
                        message=f"group {group_id} translation status "
                        f"{translation.status.value!r}",
                    )
                )
            self._validate_group_translation(translation, group, group_id)
        except (ProviderResultError, StorageCorruptionError) as exc:
            error = exc.error if isinstance(exc, ProviderResultError) else StageError(
                code="MT_RAW_INVALID", message=str(exc)
            )
            self._record_group_failure(
                records,
                store,
                stored,
                source,
                group_id,
                error,
                started,
                mono0,
                dispatched=True,
                replay_reason=replay_reason,
            )
            state.add_error(error)
            state.write(JobStatus.needs_review)
            raise

        output = store.write_group_translation(group_id, translation)
        finished = self._now()
        elapsed_ms = max(0, int((self._monotonic() - mono0) * 1000))
        complete = running.model_copy(
            update={
                "status": MTGroupStatus.complete,
                "raw_reference": translation.raw_reference,
                "output": output,
                "target_char_count": sum(
                    len(segment.translated_text_tr) for segment in translation.segments
                ),
                "finished_at_utc": finished,
                "remote_may_have_run": False,
                "auto_resumable": False,
            }
        )
        records[group_id] = complete
        store.append_attempt(
            AttemptRecord(
                stage=StageKind.mt,
                group_id=group_id,
                attempt=attempt,
                provider=self.config.mt.provider,
                model=self.config.mt.model,
                request_id=None,
                raw_reference=translation.raw_reference,
                source_char_count=running.source_char_count,
                target_char_count=complete.target_char_count,
                started_at_utc=started,
                finished_at_utc=finished,
                elapsed_ms=elapsed_ms,
                outcome=AttemptOutcome.complete,
                replay_reason=replay_reason,
                recorded_at_utc=finished,
            )
        )
        self._persist_mt_manifest(store, stored, source, tuple(records.values()), status="running")
        return translation, complete

    def _record_group_failure(
        self,
        records: dict[str, MTGroupRecord],
        store: JobStore,
        stored: StoredInput,
        source: str,
        group_id: str,
        error: StageError,
        started: datetime,
        mono0: float,
        *,
        dispatched: bool,
        replay_reason: str | None = None,
    ) -> None:
        previous = records[group_id]
        finished = self._now()
        elapsed_ms = max(0, int((self._monotonic() - mono0) * 1000))
        auto_resumable = (
            not dispatched and not error.remote_status_unknown and error.retryable
        )
        failed = previous.model_copy(
            update={
                "status": MTGroupStatus.failed,
                "error": error,
                "finished_at_utc": finished,
                "remote_may_have_run": dispatched or error.remote_status_unknown,
                "auto_resumable": auto_resumable,
                "replay_reason": replay_reason,
            }
        )
        records[group_id] = failed
        outcome = (
            AttemptOutcome.remote_unknown
            if error.remote_status_unknown
            else AttemptOutcome.failed
        )
        store.append_attempt(
            AttemptRecord(
                stage=StageKind.mt,
                group_id=group_id,
                attempt=previous.attempt,
                provider=self.config.mt.provider,
                model=self.config.mt.model,
                request_id=None,
                raw_reference=previous.raw_reference,
                source_char_count=previous.source_char_count,
                target_char_count=previous.target_char_count,
                started_at_utc=previous.started_at_utc,
                finished_at_utc=finished,
                elapsed_ms=elapsed_ms,
                outcome=outcome,
                error=error,
                replay_reason=replay_reason,
                recorded_at_utc=finished,
            )
        )
        self._persist_mt_manifest(store, stored, source, tuple(records.values()), status="partial")

    def _persist_mt_manifest(
        self,
        store: JobStore,
        stored: StoredInput,
        source: str,
        groups: Sequence[MTGroupRecord],
        *,
        status: str,
        translation: StoredArtifact | None = None,
    ) -> MTManifest:
        manifest = MTManifest(
            provider=self.config.mt.provider,
            model=self.config.mt.model,
            source_language=source,
            target_language=TARGET_LANGUAGE_CODE,
            audio_sha256=stored.audio.audio_sha256,
            config_fingerprint=stored.config_fingerprint,
            job_fingerprint=stored.job_fingerprint,
            status=status,  # type: ignore[arg-type]
            groups=tuple(groups),
            translation=translation,
            updated_at_utc=self._now(),
        )
        store.write_mt_manifest(manifest)
        return manifest

    def _verify_manifest_identity(
        self,
        job_fingerprint: str,
        config_fingerprint: str,
        audio_sha256: str,
        stored: StoredInput,
    ) -> None:
        if job_fingerprint != stored.job_fingerprint:
            raise ResumeRefused("manifest job fingerprint differs from the stored job")
        if config_fingerprint != stored.config_fingerprint:
            raise ResumeRefused("manifest configuration fingerprint differs from the stored job")
        if audio_sha256 != stored.audio.audio_sha256:
            raise ResumeRefused("manifest audio hash differs from the stored audio")

    def _verify_mt_manifest_identity(
        self, manifest: MTManifest, stored: StoredInput, source: str
    ) -> None:
        self._verify_manifest_identity(
            manifest.job_fingerprint,
            manifest.config_fingerprint,
            manifest.audio_sha256,
            stored,
        )
        if manifest.provider != self.config.mt.provider or manifest.model != self.config.mt.model:
            raise ResumeRefused("MT manifest provider/model differs from the current config")
        if manifest.source_language != source or manifest.target_language != TARGET_LANGUAGE_CODE:
            raise ResumeRefused("MT manifest language differs from the current request")

    def _verify_group_artifact(
        self, store: JobStore, record: MTGroupRecord, group: Sequence[Segment]
    ) -> Translation:
        if record.output is None or record.raw_reference is None:
            raise StorageCorruptionError(
                f"completed MT group {record.group_id} is missing output or raw reference"
            )
        data = store.verify_artifact(record.output)
        try:
            translation = Translation.model_validate_json(data)
        except ValueError as exc:
            raise StorageCorruptionError(
                f"stored MT group {record.group_id} is invalid: {exc}"
            ) from exc
        if translation.raw_reference != record.raw_reference:
            raise StorageCorruptionError(
                f"MT group {record.group_id} raw reference disagrees with manifest"
            )
        store.verify_raw(
            record.raw_reference,
            stage=StageKind.mt,
            group_id=record.group_id,
            attempt=record.attempt,
        )
        self._verify_supplementary(
            store,
            StageKind.mt,
            translation.request_metadata,
            group_id=record.group_id,
            attempt=record.attempt,
        )
        if [segment.segment_id for segment in translation.segments] != list(record.segment_ids):
            raise StorageCorruptionError(
                f"MT group {record.group_id} segment ids differ from the manifest"
            )
        try:
            self._validate_group_translation(translation, group, record.group_id)
        except ProviderResultError as exc:
            raise StorageCorruptionError(
                f"stored MT group {record.group_id} fails provenance: {exc}"
            ) from exc
        return translation

    def _load_group_translation(self, store: JobStore, record: MTGroupRecord) -> Translation:
        if record.output is None:
            raise StorageCorruptionError(f"group {record.group_id} has no stored output")
        data = store.verify_artifact(record.output)
        try:
            return Translation.model_validate_json(data)
        except ValueError as exc:
            raise StorageCorruptionError(
                f"stored MT group {record.group_id} is invalid: {exc}"
            ) from exc

    def _load_combined_translation(
        self,
        store: JobStore,
        stored: StoredInput,
        source: str,
        segments: Sequence[Segment],
    ) -> Translation:
        manifest = store.read_mt_manifest()
        if manifest is None:
            raise StorageCorruptionError("MT manifest is complete but is missing")
        if manifest.translation is None:
            raise StorageCorruptionError(
                "MT manifest is complete but has no anchored translation artifact"
            )
        data = store.verify_artifact(manifest.translation)
        try:
            translation = Translation.model_validate_json(data)
        except ValueError as exc:
            raise StorageCorruptionError(
                f"stored combined translation is invalid: {exc}"
            ) from exc
        if (
            translation.provider != self.config.mt.provider
            or translation.model != self.config.mt.model
            or translation.source_language != source
            or translation.target_language != TARGET_LANGUAGE_CODE
            or translation.status is not TranslationStatus.translated
        ):
            raise StorageCorruptionError("stored combined translation fails identity checks")
        try:
            translation.validate_against([segment.segment_id for segment in segments])
            self._validate_translation_provenance(translation, segments)
        except (SegmentContractError, ProviderResultError) as exc:
            raise StorageCorruptionError(
                f"stored combined translation fails provenance: {exc}"
            ) from exc
        # Ensure the referenced raw bodies still exist and hash correctly.
        for record in manifest.groups:
            if record.status is MTGroupStatus.complete:
                if record.raw_reference is None:
                    raise StorageCorruptionError(
                        f"group {record.group_id} has no raw reference"
                    )
                store.verify_raw(
                    record.raw_reference,
                    stage=StageKind.mt,
                    group_id=record.group_id,
                    attempt=record.attempt,
                )
                self._verify_supplementary(
                    store,
                    StageKind.mt,
                    self._load_group_translation(store, record).request_metadata,
                    group_id=record.group_id,
                    attempt=record.attempt,
                )
        return translation

    def _validate_group_translation(
        self, translation: Translation, group: Sequence[Segment], group_id: str
    ) -> None:
        issues: list[str] = []
        if translation.provider != self.config.mt.provider:
            issues.append(f"group {group_id} provider drift")
        if translation.model != self.config.mt.model:
            issues.append(f"group {group_id} model drift")
        expected_ids = [segment.segment_id for segment in group]
        if [segment.segment_id for segment in translation.segments] != expected_ids:
            issues.append(f"group {group_id} segment id/order mismatch")
        else:
            for source, translated in zip(group, translation.segments):
                if translated.source_text != source.source_text:
                    issues.append(f"group {group_id} source_text altered for {source.segment_id}")
                if translated.translation_input != source.translation_input:
                    issues.append(
                        f"group {group_id} translation_input altered for {source.segment_id}"
                    )
        if issues:
            raise ProviderResultError(
                StageError(code="MT_RESULT_INVALID", message="; ".join(issues))
            )

    def _validate_translation_provenance(
        self, translation: Translation, segments: Sequence[Segment]
    ) -> None:
        issues: list[str] = []
        if len(translation.segments) != len(segments):
            issues.append("combined translation segment count differs from the transcript")
        for source, translated in zip(segments, translation.segments):
            if translated.segment_id != source.segment_id:
                issues.append(f"segment id mismatch at {source.segment_id}")
            if translated.source_text != source.source_text:
                issues.append(f"source_text altered for {source.segment_id}")
            if translated.translation_input != source.translation_input:
                issues.append(f"translation_input altered for {source.segment_id}")
        if issues:
            raise ProviderResultError(
                StageError(code="MT_PROVENANCE_INVALID", message="; ".join(issues))
            )

    def _combine_translations(
        self, source: str, translations: Sequence[Translation]
    ) -> Translation:
        segments: list[TranslatedSegment] = []
        for translation in translations:
            if translation.status is not TranslationStatus.translated:
                raise ProviderResultError(
                    StageError(
                        code="MT_STATUS_INVALID",
                        message="cannot combine a non-translated group result",
                    )
                )
            segments.extend(translation.segments)
        return Translation(
            provider=self.config.mt.provider,
            model=self.config.mt.model,
            status=TranslationStatus.translated,
            source_language=source,
            target_language=TARGET_LANGUAGE_CODE,
            segments=tuple(segments),
        )
