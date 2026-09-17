"""Burned-in Turkish subtitle video: full-video acquisition, render, publication.

This module adds the opt-in desktop/CLI feature that produces an MP4 with
permanently burned-in Turkish subtitles. It is deliberately additive: nothing
here runs unless the operator explicitly selects it, and the accepted audio-only
YouTube source path (its argv, extraction version and source cache) is untouched.

Pipeline
--------
1. **Timed translation.** The canonical, timed source transcript is translated a
   *second* time by the fixed Google Translation Basic v2 route through
   :mod:`subtitle_flow.timed_subtitles`. No timestamp is ever fabricated; when no
   valid timed transcript exists a typed review refusal is raised instead.
2. **Full video acquisition.** The same accepted public YouTube URL is re-used
   against a *separate* hardened ``yt-dlp`` path
   (:func:`subtitle_flow.yt_dlp_runner.download_video`) into a job-private
   intermediate. The full video is never written to the durable audio cache.
3. **Sanitized ASS + burn.** A resolution-aware, injection-safe ASS document
   (:mod:`subtitle_flow.ass_subtitles`) is rendered and burned with
   ``ffmpeg``/``libass`` using shell-free argv and a fixed-name working directory.
4. **Validation + atomic publication.** The output is probed for video+audio,
   duration, bounded size and a full decode; a deterministic manifest binding the
   render identity and output hash is published next to it. A differing existing
   publication is archived (never silently overwritten) and the new file is
   moved into place atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from subtitle_flow import platform_compat
from subtitle_flow.ass_subtitles import render_ass_document
from subtitle_flow.config import YouTubeLimits, YouTubeOriginSettings
from subtitle_flow.media import sha256_file
from subtitle_flow.output_paths import unsafe_symlink_ancestor
from subtitle_flow.quality import QualitySettings
from subtitle_flow.storage import JobStore, StoredInput, utc_now
from subtitle_flow.timed_subtitles import (
    TimedSubtitleArtifact,
    cues_from_artifact,
)
from subtitle_flow.transcript_markdown import (
    MarkdownExportError,
    recover_timed_transcript,
)
from subtitle_flow.video import (
    LOCAL_PROTOCOLS,
    VideoMediaError,
    _run_tool,
    ffmpeg_tool_version,
)
from subtitle_flow.yt_dlp_runner import YtDlpError, YtDlpLimits, download_video
from subtitle_flow.youtube_source import (
    YouTubeReference,
    parse_youtube_metadata,
)

__all__ = [
    "BURN_INTERMEDIATE_LOCATOR",
    "VIDEO_PUBLICATION_VERSION",
    "BurnOutcome",
    "FullVideoInfo",
    "VideoBurnError",
    "VideoPublicationManifest",
    "burn_turkish_subtitles",
    "load_video_manifest",
    "video_manifest_name",
    "video_output_name",
    "verify_published_video",
]

#: Version of the render/publication policy. A change makes a published video
#: non-reusable (it is re-rendered, never silently relabeled).
VIDEO_PUBLICATION_VERSION: Final[str] = "1"

#: Confined job-relative locator of the full-video intermediate + manifest.
BURN_INTERMEDIATE_LOCATOR: Final[str] = "intermediates/full_video.manifest.json"
_FULL_VIDEO_STEM: Final[str] = "full_video"
_RENDER_DIRNAME: Final[str] = "render"
_ASS_FILENAME: Final[str] = "subs.ass"
_ARCHIVE_DIRNAME: Final[str] = "_arsiv"

#: Duration tolerance between the recorded source duration and the downloaded
#: full video. A YouTube remux can differ by a small amount; a truncated download
#: is far outside this window.
_MIN_DURATION_TOLERANCE_MS: Final[int] = 5000

_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class VideoBurnError(ValueError):
    """A typed failure of the burned-in subtitle video feature."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class RenderSettings:
    """Exact, deterministic encoder settings for the burned-in output."""

    video_codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    preset: str = "medium"
    crf: int = 23
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    movflags: str = "+faststart"

    def payload(self) -> dict[str, Any]:
        return {
            "video_codec": self.video_codec,
            "pix_fmt": self.pix_fmt,
            "preset": self.preset,
            "crf": self.crf,
            "audio_codec": self.audio_codec,
            "audio_bitrate": self.audio_bitrate,
            "movflags": self.movflags,
        }


RENDER_SETTINGS: Final[RenderSettings] = RenderSettings()


@dataclass(frozen=True)
class BurnToolchain:
    """Provenance of the installed ffmpeg used for one render."""

    ffmpeg_version: str
    has_subtitles_filter: bool
    has_libx264: bool
    has_aac: bool


@dataclass(frozen=True)
class FullVideoInfo:
    """Verified media facts for one downloaded full-video intermediate."""

    path: Path
    sha256: str
    size_bytes: int
    duration_ms: int
    width: int
    height: int


class VideoPublicationManifest(BaseModel):
    """Deterministic evidence published next to the final MP4."""

    model_config = _FROZEN

    video_publication_version: str = Field(min_length=1)
    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    canonical_url: str = Field(min_length=1)
    output_name: str = Field(min_length=1)
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_size_bytes: int = Field(strict=True, gt=0)
    render_identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timed_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ass_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_duration_ms: int = Field(strict=True, gt=0)
    render_settings: dict[str, Any] = Field(default_factory=dict)
    ffmpeg_version: str = Field(min_length=1)
    #: Propagated from the timed translation so a completed render still carries
    #: the human-review signal (and the exact per-cue flags) into reuse and the
    #: API/UI surface.
    needs_review: bool = False
    cue_flags: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class BurnOutcome:
    """Result of one successful (or reused) subtitle-video publication."""

    video_path: Path
    manifest_path: Path
    timed_artifact: TimedSubtitleArtifact
    cue_count: int
    reused: bool
    source_video_reused: bool
    #: Whether the timed translation requires human review (quality flags).
    needs_review: bool = False
    #: Per-segment cue flags that were burned into this video (may be empty).
    cue_flags: tuple[tuple[str, tuple[str, ...]], ...] = ()


def video_output_name(video_id: str) -> str:
    if not isinstance(video_id, str) or _SAFE_NAME_RE.match(video_id) is None:
        raise VideoBurnError("VIDEO_NAME_INVALID", "the video id is not a safe identifier")
    return f"{video_id}-turkce-altyazili.mp4"


def video_manifest_name(video_id: str) -> str:
    if not isinstance(video_id, str) or _SAFE_NAME_RE.match(video_id) is None:
        raise VideoBurnError("VIDEO_NAME_INVALID", "the video id is not a safe identifier")
    return f"{video_id}-turkce-altyazili.manifest.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest, _size = sha256_file(path)
    return digest


def _cue_flags(
    artifact: TimedSubtitleArtifact,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the deterministic per-cue flag set of a timed artifact.

    A cue with no flags is omitted so an all-clean artifact serializes as an
    empty tuple; the order follows the artifact's own cue order.
    """

    return tuple(
        (record.segment_id, tuple(record.flags))
        for record in artifact.cues
        if record.flags
    )


# --------------------------------------------------------------------------- #
# ffprobe / ffmpeg
# --------------------------------------------------------------------------- #
def _probe_media(
    path: Path, *, ffprobe_bin: str, timeout: float
) -> dict[str, Any]:
    args = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-protocol_whitelist",
        LOCAL_PROTOCOLS,
        str(path),
    ]
    try:
        completed = _run_tool(
            args,
            timeout=timeout,
            missing_code="BURN_FFPROBE_MISSING",
            missing_label="ffprobe",
            timeout_code="BURN_FFPROBE_TIMEOUT",
            failed_code="BURN_FFPROBE_FAILED",
            overflow_code="BURN_OUTPUT_OVERFLOW",
        )
    except VideoMediaError as exc:
        raise VideoBurnError(exc.code, exc.message) from exc
    if completed.returncode != 0:
        raise VideoBurnError(
            "BURN_MEDIA_INVALID",
            "ffprobe could not read the file; the container is invalid, unsupported "
            "or uses a non-local protocol",
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoBurnError("BURN_MEDIA_INVALID", "ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise VideoBurnError("BURN_MEDIA_INVALID", "ffprobe returned an unexpected payload")
    return payload


def _positive_int(value: Any, *, label: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise VideoBurnError(
            "BURN_MEDIA_INVALID", f"{label} is not a positive integer"
        ) from exc
    if parsed <= 0:
        raise VideoBurnError("BURN_MEDIA_INVALID", f"{label} must be positive")
    return parsed


def _duration_ms(payload: dict[str, Any]) -> int:
    format_map = payload.get("format")
    format_map = format_map if isinstance(format_map, dict) else {}
    raw = format_map.get("duration")
    if raw in (None, "", "N/A"):
        streams = payload.get("streams")
        if isinstance(streams, list):
            for entry in streams:
                if isinstance(entry, dict) and entry.get("codec_type") == "video":
                    raw = entry.get("duration")
                    if raw not in (None, "", "N/A"):
                        break
    try:
        seconds = float(str(raw))
    except (TypeError, ValueError) as exc:
        raise VideoBurnError(
            "BURN_MEDIA_INVALID", "the media duration is unknown"
        ) from exc
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds <= 0:
        raise VideoBurnError(
            "BURN_MEDIA_INVALID", "the media duration must be a finite positive number"
        )
    return int(round(seconds * 1000))


def _streams(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise VideoBurnError("BURN_MEDIA_INVALID", "ffprobe reported no stream list")
    video: dict[str, Any] | None = None
    audio: dict[str, Any] | None = None
    for entry in raw_streams:
        if not isinstance(entry, dict):
            raise VideoBurnError("BURN_MEDIA_INVALID", "ffprobe reported a malformed stream")
        kind = entry.get("codec_type")
        if kind == "video":
            disposition = entry.get("disposition")
            if isinstance(disposition, dict) and disposition.get("attached_pic") == 1:
                continue
            if video is not None:
                raise VideoBurnError(
                    "BURN_MEDIA_INVALID",
                    "the file has more than one video stream; exactly one is required",
                )
            video = entry
        elif kind == "audio":
            if audio is not None:
                raise VideoBurnError(
                    "BURN_MEDIA_INVALID",
                    "the file has more than one audio stream; exactly one is required",
                )
            audio = entry
    return video, audio


def _decode_check(path: Path, *, ffmpeg_bin: str, timeout: float) -> None:
    args = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-protocol_whitelist",
        LOCAL_PROTOCOLS,
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = _run_tool(
            args,
            timeout=timeout,
            missing_code="BURN_FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="BURN_RENDER_TIMEOUT",
            failed_code="BURN_RENDER_FAILED",
            overflow_code="BURN_OUTPUT_OVERFLOW",
        )
    except VideoMediaError as exc:
        raise VideoBurnError(exc.code, exc.message) from exc
    if completed.returncode != 0:
        raise VideoBurnError(
            "BURN_OUTPUT_CORRUPT",
            "the produced video did not decode cleanly; refusing to publish it",
        )


def _listing_lines(text: str) -> list[list[str]]:
    lines: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        lines.append(stripped.split())
    return lines


def _has_filter(lines: list[list[str]], name: str) -> bool:
    # ffmpeg lists each filter as ``<flags> <name> <in->out> <description>``; the
    # description may legitimately contain the filter name, so only the first two
    # columns are inspected.
    return any(name in tokens[:2] for tokens in lines)


def _has_encoder(lines: list[list[str]], name: str) -> bool:
    return any(len(tokens) >= 2 and tokens[1] == name for tokens in lines)


def preflight_burn_toolchain(
    ffmpeg_bin: str = "ffmpeg", *, timeout: float = 30.0
) -> BurnToolchain:
    """Verify ffmpeg, the libass ``subtitles`` filter and the H.264/AAC encoders.

    This is a local, offline probe. A missing filter or encoder is a typed,
    actionable refusal instead of a render that fails halfway with a raw stderr.
    """

    try:
        version = ffmpeg_tool_version(ffmpeg_bin, timeout=timeout)
    except VideoMediaError as exc:
        raise VideoBurnError(exc.code, exc.message) from exc
    try:
        filters = _run_tool(
            [ffmpeg_bin, "-hide_banner", "-filters"],
            timeout=timeout,
            missing_code="BURN_FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="BURN_FFMPEG_TIMEOUT",
            failed_code="BURN_FFMPEG_FAILED",
            overflow_code="BURN_OUTPUT_OVERFLOW",
        )
        encoders = _run_tool(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            timeout=timeout,
            missing_code="BURN_FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="BURN_FFMPEG_TIMEOUT",
            failed_code="BURN_FFMPEG_FAILED",
            overflow_code="BURN_OUTPUT_OVERFLOW",
        )
    except VideoMediaError as exc:
        raise VideoBurnError(exc.code, exc.message) from exc
    if filters.returncode != 0 or encoders.returncode != 0:
        raise VideoBurnError(
            "BURN_FFMPEG_FAILED",
            "ffmpeg could not report its filters/encoders; the installation is "
            "incomplete",
        )
    filter_lines = _listing_lines(filters.stdout or "")
    encoder_lines = _listing_lines(encoders.stdout or "")
    has_subtitles = _has_filter(filter_lines, "subtitles")
    has_libx264 = _has_encoder(encoder_lines, "libx264")
    has_aac = _has_encoder(encoder_lines, "aac")
    if not has_subtitles:
        raise VideoBurnError(
            "BURN_LIBASS_MISSING",
            "the installed ffmpeg has no 'subtitles' filter (libass); a "
            "permanently burned-in subtitle video cannot be produced",
        )
    if not has_libx264 or not has_aac:
        missing = ", ".join(
            name
            for name, present in (("libx264", has_libx264), ("aac", has_aac))
            if not present
        )
        raise VideoBurnError(
            "BURN_ENCODER_MISSING",
            f"the installed ffmpeg is missing the required encoder(s): {missing}",
        )
    return BurnToolchain(
        ffmpeg_version=version,
        has_subtitles_filter=has_subtitles,
        has_libx264=has_libx264,
        has_aac=has_aac,
    )


# --------------------------------------------------------------------------- #
# Full-video acquisition (job-private intermediate)
# --------------------------------------------------------------------------- #
def _validate_full_video(
    path: Path,
    *,
    origin: YouTubeOriginSettings,
    limits: YouTubeLimits,
    ffprobe_bin: str,
) -> FullVideoInfo:
    try:
        info = os.stat(path)
    except OSError as exc:
        raise VideoBurnError("VIDEO_FULL_MISSING", "the downloaded full video is missing") from exc
    if info.st_size == 0:
        raise VideoBurnError("VIDEO_FULL_INVALID", "the downloaded full video is empty")
    if info.st_size > limits.max_video_bytes:
        raise VideoBurnError(
            "VIDEO_FULL_TOO_LARGE",
            f"the downloaded full video is {info.st_size} bytes, exceeding the "
            f"limit {limits.max_video_bytes}",
        )
    payload = _probe_media(
        path, ffprobe_bin=ffprobe_bin, timeout=limits.probe_timeout_seconds
    )
    video, audio = _streams(payload)
    if video is None or audio is None:
        raise VideoBurnError(
            "VIDEO_FULL_INVALID",
            "the downloaded full video must carry exactly one video and one audio "
            "stream",
        )
    width = _positive_int(video.get("width"), label="video width")
    height = _positive_int(video.get("height"), label="video height")
    duration_ms = _duration_ms(payload)
    tolerance = max(_MIN_DURATION_TOLERANCE_MS, origin.source_duration_ms // 200)
    if abs(duration_ms - origin.source_duration_ms) > tolerance:
        raise VideoBurnError(
            "VIDEO_FULL_INCOMPLETE",
            f"the downloaded full video is {duration_ms}ms but the accepted source "
            f"is {origin.source_duration_ms}ms (beyond {tolerance}ms); refusing a "
            "truncated or unrelated download",
        )
    digest = _sha256_path(path)
    return FullVideoInfo(
        path=path,
        sha256=digest,
        size_bytes=info.st_size,
        duration_ms=duration_ms,
        width=width,
        height=height,
    )


def _load_intermediate_manifest(store: JobStore) -> dict[str, Any] | None:
    path = store.resolve(BURN_INTERMEDIATE_LOCATOR)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _acquire_full_video(
    store: JobStore,
    *,
    origin: YouTubeOriginSettings,
    ytdlp_bin: str,
    ffprobe_bin: str,
    limits: YouTubeLimits,
    progress: Callable[[str], None] | None,
) -> tuple[FullVideoInfo, bool]:
    """Return the job-private full video, reusing a verified intermediate if present.

    The intermediate is confined to the job directory and never enters the durable
    audio source cache. A present intermediate whose recorded hash/size/duration
    still match is reused; otherwise the accepted canonical URL is downloaded again
    through the separate hardened video path.
    """

    intermediates = store.resolve("intermediates")
    intermediates.mkdir(parents=True, exist_ok=True)
    manifest = _load_intermediate_manifest(store)
    if manifest is not None:
        rel = manifest.get("path")
        if isinstance(rel, str) and rel.startswith(f"intermediates/{_FULL_VIDEO_STEM}"):
            candidate = store.resolve(rel)
            if candidate.is_file() and not candidate.is_symlink():
                try:
                    if (
                        _sha256_path(candidate) == manifest.get("sha256")
                        and os.stat(candidate).st_size == manifest.get("size_bytes")
                    ):
                        info = _validate_full_video(
                            candidate,
                            origin=origin,
                            limits=limits,
                            ffprobe_bin=ffprobe_bin,
                        )
                        return info, True
                except (VideoBurnError, OSError):
                    pass

    if progress is not None:
        progress("Tam video kaynağı indiriliyor…")
    work = Path(tempfile.mkdtemp(prefix=".full-video-", dir=str(intermediates)))
    ytdlp_limits = YtDlpLimits(
        max_directory_bytes=limits.max_video_bytes,
        download_timeout_seconds=limits.video_download_timeout_seconds,
        max_stdout_bytes=limits.max_stdout_bytes,
        max_stderr_bytes=limits.max_stderr_bytes,
        retries=limits.max_retries,
        fragment_retries=limits.max_fragment_retries,
        socket_timeout_seconds=limits.socket_timeout_seconds,
        max_duration_seconds=max(1, limits.max_duration_ms // 1000),
    )
    try:
        try:
            download = download_video(
                ytdlp_bin, origin.canonical_url, work_dir=work, limits=ytdlp_limits
            )
        except YtDlpError as exc:
            raise VideoBurnError(exc.code, exc.message) from exc
        # Re-validate the same allowlisted metadata contract the accepted audio
        # path uses: a non-public, live, changed-id or over-long source is
        # refused before any render. A metadata refusal stays a typed YouTube
        # input error (mapped by the CLI), never a silent success.
        parse_youtube_metadata(
            download.metadata,
            YouTubeReference(origin.video_id, origin.canonical_url),
            limits=limits,
        )
        ext = download.output_path.suffix.lstrip(".").lower() or "bin"
        if not re.match(r"^[A-Za-z0-9]{1,8}$", ext):
            raise VideoBurnError(
                "VIDEO_FULL_INVALID", "the downloaded container extension is not usable"
            )
        staged = work / f"{_FULL_VIDEO_STEM}.{ext}"
        if download.output_path != staged:
            if staged.exists():
                staged.unlink()
            shutil.move(str(download.output_path), str(staged))
        info = _validate_full_video(
            staged, origin=origin, limits=limits, ffprobe_bin=ffprobe_bin
        )
        locator = f"intermediates/{_FULL_VIDEO_STEM}.{ext}"
        artifact = store.adopt_file(locator, staged)
        store.write_artifact(
            BURN_INTERMEDIATE_LOCATOR,
            json.dumps(
                {
                    "path": locator,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                    "duration_ms": info.duration_ms,
                    "width": info.width,
                    "height": info.height,
                    "canonical_url": origin.canonical_url,
                    "created_at_utc": utc_now().isoformat(),
                },
                sort_keys=True,
                indent=2,
            ).encode("utf-8"),
        )
        published = store.resolve(locator)
        return (
            FullVideoInfo(
                path=published,
                sha256=artifact.sha256,
                size_bytes=artifact.size_bytes,
                duration_ms=info.duration_ms,
                width=info.width,
                height=info.height,
            ),
            False,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Render identity, reuse and publication
# --------------------------------------------------------------------------- #
def _render_identity(
    *,
    origin: YouTubeOriginSettings,
    artifact: TimedSubtitleArtifact,
    ass_sha256: str,
    width: int,
    height: int,
    toolchain: BurnToolchain,
    settings: RenderSettings = RENDER_SETTINGS,
) -> str:
    payload = {
        "video_publication_version": VIDEO_PUBLICATION_VERSION,
        "video_id": origin.video_id,
        "canonical_url": origin.canonical_url,
        "source_duration_ms": origin.source_duration_ms,
        "intermediate_sha256": origin.intermediate_sha256,
        "source_transcript_sha256": artifact.source_transcript_sha256,
        "timed_artifact_sha256": artifact.integrity_sha256,
        "ass_sha256": ass_sha256,
        "width": width,
        "height": height,
        "render_settings": settings.payload(),
        "ffmpeg_version": toolchain.ffmpeg_version,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_video_manifest(video_dir: str | os.PathLike[str], video_id: str) -> VideoPublicationManifest | None:
    """Read and validate the publication manifest, or ``None`` when unusable."""

    path = Path(video_dir).expanduser() / video_manifest_name(video_id)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return VideoPublicationManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def _published_video_is_complete(
    target: Path, manifest: VideoPublicationManifest, *, expected_dir: Path
) -> bool:
    if target.is_symlink() or not target.is_file():
        return False
    if os.path.realpath(target.parent) != os.path.realpath(expected_dir):
        return False
    try:
        size = target.stat().st_size
    except OSError:
        return False
    if size != manifest.output_size_bytes or size == 0:
        return False
    return _sha256_path(target) == manifest.output_sha256


def verify_published_video(
    path: str | os.PathLike[str] | None,
    *,
    expected_dir: str | os.PathLike[str],
    video_id: str | None,
) -> str | None:
    """Return ``None`` when ``path`` is a verified published video, else a reason.

    A publication only counts when it is a regular, non-symlinked file with the
    exact expected name inside the selected output directory and its bytes match
    the recorded manifest hash/size. A stale or edited file can never authorize
    the automatic cleanup of the job evidence.
    """

    if not isinstance(video_id, str) or _SAFE_NAME_RE.match(video_id) is None:
        return "video kimliği eksik veya güvenli değil"
    if not path:
        return "yayımlanmış video yolu yok"
    candidate = Path(path)
    expected_name = video_output_name(video_id)
    if candidate.name != expected_name:
        return "video dosya adı video kimliğiyle eşleşmiyor"
    if os.path.islink(candidate):
        return "video dosyası sembolik bağlantı"
    if not candidate.is_file():
        return "video dosyası bulunamadı"
    unsafe = unsafe_symlink_ancestor(candidate)
    if unsafe is not None:
        return f"video yolu güvenli olmayan sembolik bileşen içeriyor: {unsafe}"
    base = Path(expected_dir).expanduser()
    manifest = load_video_manifest(base, video_id)
    if manifest is None:
        return "video yayın kaydı (manifest) yok veya geçersiz"
    if manifest.video_id != video_id:
        return "video yayın kaydı başka bir kimliğe ait"
    if not _published_video_is_complete(candidate, manifest, expected_dir=base):
        return "video içeriği yayın kaydıyla eşleşmiyor"
    return None


def _confine_output_dir(video_dir: Path) -> Path:
    # Resolve a relative custom directory (``--video-dir`` or a derived sibling)
    # to a stable absolute path *before* any mkstemp / size guard / ffmpeg call.
    # ``ffmpeg`` runs with the job render directory as its cwd, so a still
    # relative output path would be re-interpreted against the wrong directory.
    # ``abspath`` normalizes ``.``/``..`` but never follows a symlink, so the
    # confinement checks below still see and refuse a symlinked component.
    video_dir = Path(os.path.abspath(os.fspath(video_dir)))
    unsafe = unsafe_symlink_ancestor(video_dir)
    if unsafe is not None:
        raise VideoBurnError(
            "VIDEO_DIR_UNSAFE",
            f"the video directory is below a symlinked path component: {unsafe}",
        )
    if video_dir.exists() and video_dir.is_symlink():
        raise VideoBurnError("VIDEO_DIR_UNSAFE", "the video directory is a symlink")
    try:
        video_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise VideoBurnError(
            "VIDEO_DIR_INVALID", f"cannot create the video directory: {exc}"
        ) from exc
    if video_dir.is_symlink() or not video_dir.is_dir():
        raise VideoBurnError("VIDEO_DIR_UNSAFE", "the video directory is not a real directory")
    return video_dir


def _link_no_clobber(source: Path, target: Path, *, digest: str) -> None:
    """Hard-link ``source`` to ``target`` without ever overwriting a different file."""

    if target.exists():
        if _sha256_path(target) == digest:
            return
        raise VideoBurnError(
            "VIDEO_ARCHIVE_COLLISION",
            "an archive with the same name but different content already exists; "
            "refusing to overwrite it",
        )
    try:
        os.link(source, target)
    except FileExistsError:
        if _sha256_path(target) != digest:
            raise VideoBurnError(
                "VIDEO_ARCHIVE_COLLISION",
                "an archive appeared concurrently with different content",
            )
    except OSError as exc:
        raise VideoBurnError(
            "VIDEO_ARCHIVE_FAILED", f"cannot archive the existing publication: {exc}"
        ) from exc


def _archive_existing(target: Path, video_dir: Path, *, video_id: str) -> None:
    """Preserve a differing existing publication and its manifest under ``_arsiv``."""

    archive_dir = _confine_output_dir(video_dir / _ARCHIVE_DIRNAME)
    digest = _sha256_path(target)
    archive_name = f"{digest[:16]}_{target.name}"
    if len(archive_name) > 120:
        archive_name = archive_name[:120]
    _link_no_clobber(target, archive_dir / archive_name, digest=digest)
    manifest = video_dir / video_manifest_name(video_id)
    if manifest.is_file() and not manifest.is_symlink():
        manifest_digest = _sha256_path(manifest)
        _link_no_clobber(
            manifest,
            archive_dir / f"{digest[:16]}_{manifest.name}",
            digest=manifest_digest,
        )


def _publish_video(
    temp: Path,
    *,
    video_dir: Path,
    video_id: str,
    expected_sha256: str,
    expected_size: int,
) -> Path:
    target = video_dir / video_output_name(video_id)
    if target.is_symlink():
        raise VideoBurnError("VIDEO_TARGET_UNSAFE", "the video target is a symlink")
    if target.exists():
        if not target.is_file():
            # A directory, device, FIFO or other non-regular target is refused with
            # a typed error instead of letting a raw IsADirectoryError/OSError
            # escape from the hashing or archival path.
            raise VideoBurnError(
                "VIDEO_TARGET_UNSAFE",
                "the video target exists but is not a regular file",
            )
        if (
            target.stat().st_size == expected_size
            and _sha256_path(target) == expected_sha256
        ):
            try:
                temp.unlink()
            except OSError:
                pass
            return target
        _archive_existing(target, video_dir, video_id=video_id)
    try:
        platform_compat.replace_file_atomically(str(temp), target)
    except OSError as exc:
        raise VideoBurnError(
            "VIDEO_WRITE_FAILED", f"cannot publish the video: {exc}"
        ) from exc
    try:
        directory_fd = os.open(video_dir, os.O_RDONLY)
    except OSError:
        return target
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)
    return target


def _burn_ffmpeg(
    *,
    video_path: Path,
    render_dir: Path,
    out_path: Path,
    ffmpeg_bin: str,
    limits: YouTubeLimits,
    settings: RenderSettings,
) -> None:
    """Run the shell-free libass burn with a fixed-name filter path."""

    args = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-protocol_whitelist",
        LOCAL_PROTOCOLS,
        "-i",
        str(video_path),
        "-vf",
        f"subtitles={_ASS_FILENAME}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        settings.video_codec,
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-pix_fmt",
        settings.pix_fmt,
        "-c:a",
        settings.audio_codec,
        "-b:a",
        settings.audio_bitrate,
        "-movflags",
        settings.movflags,
        "-y",
        "-f",
        "mp4",
        str(out_path),
    ]
    try:
        completed = _run_tool(
            args,
            timeout=limits.render_timeout_seconds,
            missing_code="BURN_FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="BURN_RENDER_TIMEOUT",
            failed_code="BURN_RENDER_FAILED",
            overflow_code="BURN_OUTPUT_OVERFLOW",
            size_guard=(out_path, limits.max_render_bytes),
            size_guard_code="BURN_OUTPUT_TOO_LARGE",
            cwd=render_dir,
        )
    except VideoMediaError as exc:
        raise VideoBurnError(exc.code, exc.message) from exc
    if completed.returncode != 0:
        raise VideoBurnError(
            "BURN_RENDER_FAILED",
            "ffmpeg could not burn the subtitles into the video",
        )


def _validate_rendered_output(
    path: Path,
    *,
    source_width: int,
    source_height: int,
    source_duration_ms: int,
    limits: YouTubeLimits,
    ffprobe_bin: str,
    ffmpeg_bin: str,
) -> tuple[str, int]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise VideoBurnError(
            "BURN_OUTPUT_MISSING", "ffmpeg did not produce an output video"
        ) from exc
    if size == 0:
        raise VideoBurnError("BURN_OUTPUT_INVALID", "the rendered video is empty")
    if size > limits.max_render_bytes:
        raise VideoBurnError(
            "BURN_OUTPUT_TOO_LARGE",
            f"the rendered video is {size} bytes, exceeding the limit "
            f"{limits.max_render_bytes}",
        )
    payload = _probe_media(path, ffprobe_bin=ffprobe_bin, timeout=limits.probe_timeout_seconds)
    video, audio = _streams(payload)
    if video is None or audio is None:
        raise VideoBurnError(
            "BURN_OUTPUT_INVALID",
            "the rendered output must contain exactly one video and one audio stream",
        )
    width = _positive_int(video.get("width"), label="output width")
    height = _positive_int(video.get("height"), label="output height")
    if width != source_width or height != source_height:
        raise VideoBurnError(
            "BURN_OUTPUT_INVALID",
            f"the rendered resolution {width}x{height} does not preserve the source "
            f"{source_width}x{source_height}",
        )
    duration_ms = _duration_ms(payload)
    tolerance = max(_MIN_DURATION_TOLERANCE_MS, source_duration_ms // 200)
    if abs(duration_ms - source_duration_ms) > tolerance:
        raise VideoBurnError(
            "BURN_OUTPUT_INCOMPLETE",
            f"the rendered video is {duration_ms}ms but the source is "
            f"{source_duration_ms}ms (beyond {tolerance}ms); refusing a truncated "
            "output",
        )
    _decode_check(path, ffmpeg_bin=ffmpeg_bin, timeout=limits.decode_timeout_seconds)
    return _sha256_path(path), size


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _reuse_publication(
    video_dir: Path,
    *,
    origin: YouTubeOriginSettings,
    artifact: TimedSubtitleArtifact,
    toolchain: BurnToolchain,
    settings: RenderSettings,
) -> BurnOutcome | None:
    manifest = load_video_manifest(video_dir, origin.video_id)
    if manifest is None or manifest.video_publication_version != VIDEO_PUBLICATION_VERSION:
        return None
    # The manifest is read by the *expected* file name, but its own recorded
    # identity must confirm the same video: a manifest that names a different
    # video_id (for example a renamed/tampered record) is never reused.
    if manifest.video_id != origin.video_id:
        return None
    if manifest.render_settings != settings.payload():
        return None
    if (
        manifest.ffmpeg_version != toolchain.ffmpeg_version
        or manifest.timed_artifact_sha256 != artifact.integrity_sha256
        or manifest.source_transcript_sha256 != artifact.source_transcript_sha256
    ):
        return None
    # A persisted review signal is part of the publication contract: a manifest
    # that disagrees with the artifact it was rendered from is never reused (it is
    # re-rendered and republished with the correct signal).
    if (
        manifest.needs_review != artifact.needs_review
        or manifest.cue_flags != _cue_flags(artifact)
    ):
        return None
    ass_text = render_ass_document(
        cues_from_artifact(artifact), width=manifest.width, height=manifest.height
    )
    ass_sha256 = _sha256_text(ass_text)
    if ass_sha256 != manifest.ass_sha256:
        return None
    identity = _render_identity(
        origin=origin,
        artifact=artifact,
        ass_sha256=ass_sha256,
        width=manifest.width,
        height=manifest.height,
        toolchain=toolchain,
        settings=settings,
    )
    if identity != manifest.render_identity:
        return None
    target = video_dir / video_output_name(origin.video_id)
    if not _published_video_is_complete(target, manifest, expected_dir=video_dir):
        return None
    return BurnOutcome(
        video_path=target,
        manifest_path=video_dir / video_manifest_name(origin.video_id),
        timed_artifact=artifact,
        cue_count=len(artifact.cues),
        reused=True,
        source_video_reused=False,
        needs_review=artifact.needs_review,
        cue_flags=_cue_flags(artifact),
    )


def burn_turkish_subtitles(
    store: JobStore,
    stored: StoredInput,
    *,
    timed_route: Any | None = None,
    quality_settings: QualitySettings | None = None,
    ytdlp_bin: str = "yt-dlp",
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    video_dir: str | os.PathLike[str] | None = None,
    progress: Callable[[str], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    now: Callable[[], Any] = utc_now,
) -> BurnOutcome:
    """Produce (or reuse) the burned-in Turkish subtitle MP4 for one job.

    The store must be unlocked; this function holds the writer lock for the whole
    operation. A valid canonical timed transcript is mandatory: a job that can
    only recover a full text raises a typed review refusal and no video is made.
    The final MP4 and its manifest are published (archiving a differing previous
    publication) only after every validation passes.

    ``timed_route`` is the Google Translation Basic v2 timed route; it reuses or
    creates its provider-scoped timed artifact.
    """

    if store.locked:
        raise VideoBurnError(
            "BURN_ALREADY_LOCKED",
            "the job store must not already be locked before a video burn",
        )
    if timed_route is None:
        raise VideoBurnError(
            "TIMED_SUBTITLE_PROVIDER_MISSING",
            "a timed translation route is required to translate the timed "
            "transcript",
        )
    origin = stored.config.youtube_origin
    if origin is None:
        raise VideoBurnError(
            "BURN_NO_YOUTUBE_ORIGIN",
            "the burned-in subtitle video is only available for a YouTube source job",
        )
    limits = stored.config.youtube
    toolchain = preflight_burn_toolchain(ffmpeg_bin, timeout=limits.probe_timeout_seconds)
    if video_dir is None:
        from subtitle_flow.output_paths import default_video_dir

        directory = default_video_dir()
    else:
        directory = Path(video_dir).expanduser()
    directory = _confine_output_dir(directory)

    with store:
        try:
            transcript = recover_timed_transcript(store, stored)
        except MarkdownExportError as exc:
            raise VideoBurnError(
                "TIMED_SUBTITLE_TRANSCRIPT_UNAVAILABLE",
                "no verified canonical timed transcript is available for this job; "
                "a subtitle video cannot be produced without real segment timings "
                f"({exc.code})",
            ) from exc

        artifact = timed_route.reuse(
            stored, transcript, quality_settings=quality_settings
        )
        if artifact is None:
            artifact = timed_route.translate(
                stored,
                transcript,
                quality_settings=quality_settings,
                monotonic=monotonic,
                now=now,
                progress=progress,
            )

        reused = _reuse_publication(
            directory,
            origin=origin,
            artifact=artifact,
            toolchain=toolchain,
            settings=RENDER_SETTINGS,
        )
        if reused is not None:
            return reused

        cues = cues_from_artifact(artifact)
        if not cues:
            raise VideoBurnError(
                "TIMED_SUBTITLE_NO_SEGMENTS",
                "the timed translation carries no cues; refusing to render an "
                "empty subtitle video",
            )
        info, source_reused = _acquire_full_video(
            store,
            origin=origin,
            ytdlp_bin=ytdlp_bin,
            ffprobe_bin=ffprobe_bin,
            limits=limits,
            progress=progress,
        )
        ass_text = render_ass_document(cues, width=info.width, height=info.height)
        ass_sha256 = _sha256_text(ass_text)
        store.write_artifact(f"{_RENDER_DIRNAME}/{_ASS_FILENAME}", ass_text.encode("utf-8"))
        render_dir = store.resolve(_RENDER_DIRNAME)

        if progress is not None:
            progress("Türkçe altyazılar videoya kalıcı olarak işleniyor…")
        handle, temp_name = tempfile.mkstemp(
            prefix=f".{video_output_name(origin.video_id)}.",
            suffix=".tmp.mp4",
            dir=str(directory),
        )
        os.close(handle)
        temp_path = Path(temp_name)
        try:
            _burn_ffmpeg(
                video_path=info.path,
                render_dir=render_dir,
                out_path=temp_path,
                ffmpeg_bin=ffmpeg_bin,
                limits=limits,
                settings=RENDER_SETTINGS,
            )
            output_sha256, output_size = _validate_rendered_output(
                temp_path,
                source_width=info.width,
                source_height=info.height,
                source_duration_ms=info.duration_ms,
                limits=limits,
                ffprobe_bin=ffprobe_bin,
                ffmpeg_bin=ffmpeg_bin,
            )
            identity = _render_identity(
                origin=origin,
                artifact=artifact,
                ass_sha256=ass_sha256,
                width=info.width,
                height=info.height,
                toolchain=toolchain,
                settings=RENDER_SETTINGS,
            )
            manifest = VideoPublicationManifest(
                video_publication_version=VIDEO_PUBLICATION_VERSION,
                video_id=origin.video_id,
                canonical_url=origin.canonical_url,
                output_name=video_output_name(origin.video_id),
                output_sha256=output_sha256,
                output_size_bytes=output_size,
                render_identity=identity,
                source_transcript_sha256=artifact.source_transcript_sha256,
                timed_artifact_sha256=artifact.integrity_sha256,
                ass_sha256=ass_sha256,
                width=info.width,
                height=info.height,
                source_video_sha256=info.sha256,
                source_video_duration_ms=info.duration_ms,
                render_settings=RENDER_SETTINGS.payload(),
                ffmpeg_version=toolchain.ffmpeg_version,
                needs_review=artifact.needs_review,
                cue_flags=_cue_flags(artifact),
            )
            published = _publish_video(
                temp_path,
                video_dir=directory,
                video_id=origin.video_id,
                expected_sha256=output_sha256,
                expected_size=output_size,
            )
            manifest_path = directory / video_manifest_name(origin.video_id)
            _publish_manifest(manifest, manifest_path)
        except BaseException:
            try:
                temp_path.unlink()
            except OSError:
                pass
            raise
        if progress is not None:
            progress("Videolar klasörüne yayımlandı.")
        return BurnOutcome(
            video_path=published,
            manifest_path=manifest_path,
            timed_artifact=artifact,
            cue_count=len(artifact.cues),
            reused=False,
            source_video_reused=source_reused,
            needs_review=artifact.needs_review,
            cue_flags=_cue_flags(artifact),
        )


def _publish_manifest(
    manifest: VideoPublicationManifest,
    manifest_path: Path,
) -> None:
    """Atomically publish the small manifest next to the final video."""

    payload = manifest.model_dump_json(indent=2).encode("utf-8")
    target = Path(manifest_path)
    if target.is_symlink():
        raise VideoBurnError("VIDEO_TARGET_UNSAFE", "the manifest target is a symlink")
    if target.exists() and not target.is_file():
        raise VideoBurnError(
            "VIDEO_TARGET_UNSAFE",
            "the manifest target exists but is not a regular file",
        )
    handle, temp_name = tempfile.mkstemp(
        prefix=".manifest.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        platform_compat.replace_file_atomically(temp_name, str(target))
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
