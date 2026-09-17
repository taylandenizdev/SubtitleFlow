"""Local video acceptance and deterministic audio extraction (Phase 5).

This module is the only place that understands a video *container*. It accepts a
local, regular, explicitly supplied source file, proves what the file actually
is with ``ffprobe`` (never by extension), rejects anything that cannot be mapped
without ambiguity, and extracts a deterministic mono 16 kHz PCM WAV or explicit
FLAC that the existing ready-audio path can consume unchanged.

Design guarantees
-----------------
* **Local only.** Both ``ffprobe`` and ``ffmpeg`` run with an argument list (no
  shell), ``-nostdin`` and an explicit ``-protocol_whitelist`` that admits only
  local protocols, so a local playlist or disguised container cannot trigger a
  remote fetch while being probed.
* **Bound before work.** Byte size, duration, probe/decode/extraction timeouts
  and the output byte budget are enforced before and around any model work. The
  extraction output size is watched *while ffmpeg writes*; reaching the budget
  kills the process group and the partial file is discarded, so an oversized or
  clipped result is never validated or published.
* **Bounded tool output.** ``ffprobe``/``ffmpeg`` stdout and stderr are drained
  incrementally with an explicit byte ceiling. A child that emits more than the
  ceiling is killed with its process group and reported as a typed overflow
  instead of being buffered without bound.
* **Source is read-only and bound.** The source file itself must not be a
  symlink; its parent aliases (for example macOS ``/tmp`` -> ``/private/tmp``)
  are resolved once to a stable canonical absolute path, and every probe, hash
  and extraction runs against that bound path with before/after identity checks.
* **Timeline is explicit.** The format start, the selected video stream start
  and the selected audio stream start must be present, finite and within the
  configured tolerance of zero. A delay expressed as real leading silence does
  not move a stream start and therefore survives extraction with its offset
  intact. Genuine nonzero offsets are refused with a typed error instead of
  being silently normalised, because the extracted output cannot carry a source
  offset and no mapping is proven in this version.
* **One audio stream.** Exactly one audio stream is required; a multi-audio file
  is rejected before extraction rather than silently selecting the first stream.
* **Deterministic output.** The extraction identity binds the source SHA-256,
  the selected stream, the output settings, the extraction version and the tool
  version (never a temporary path). The original file is never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import threading
import time
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, BinaryIO, Final

from pydantic import BaseModel, ConfigDict, Field

from subtitle_flow import platform_compat
from subtitle_flow.config import (
    ExtractionSettings,
    MediaLimits,
    VideoLimits,
    VideoOriginSettings,
)
from subtitle_flow.media import AudioInfo, MediaError, sha256_file, validate_audio
from subtitle_flow.schemas import UtcDatetime
from subtitle_flow.storage import StoredArtifact, utc_now

__all__ = [
    "ACCEPTED_VIDEO_CONTAINER_TOKENS",
    "EXTRACTION_VERSION",
    "LOCAL_PROTOCOLS",
    "ExtractionManifest",
    "ExtractionResult",
    "VideoInfo",
    "VideoMediaError",
    "build_extraction_manifest",
    "compute_extraction_id",
    "extract_audio",
    "extraction_locator",
    "ffmpeg_tool_version",
    "recheck_video",
    "resolve_source_path",
    "validate_video",
]

#: Version of the extraction algorithm/timeline policy. A change invalidates
#: every previously cached extraction identity.
EXTRACTION_VERSION: Final[str] = "1"

#: Container family tokens this build was actually tested against. Acceptance
#: is by ``ffprobe``'s ``format_name`` tokens, never by the file extension.
#: Documented as build-tested facts, not a universal format promise.
ACCEPTED_VIDEO_CONTAINER_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "mov",
        "mp4",
        "m4a",
        "3gp",
        "3g2",
        "mj2",
        "matroska",
        "webm",
    }
)

#: Only local protocols are admitted to the media tools. ``file``/``pipe``/``fd``
#: cover a regular local file plus a local pipe; playlists and network protocols
#: (``http``, ``https``, ``hls``, ``tcp`` ...) are excluded by omission.
LOCAL_PROTOCOLS: Final[str] = "file,pipe,fd"

#: Explicit byte ceilings for one external tool invocation. Both child streams
#: are drained as they are produced; a stream that exceeds its ceiling makes the
#: whole process group be killed and the call fail with a typed overflow error.
#: The ceilings are deliberately far above any valid ``ffprobe`` JSON or
#: ``ffmpeg`` diagnostic for a bounded local file and far below an unbounded
#: diagnostic flood.
TOOL_STDOUT_CEILING_BYTES: Final[int] = 1024 * 1024
TOOL_STDERR_CEILING_BYTES: Final[int] = 1024 * 1024

_TOOL_READ_CHUNK: Final[int] = 64 * 1024
_TOOL_POLL_INTERVAL_SECONDS: Final[float] = 0.02
_TOOL_JOIN_TIMEOUT_SECONDS: Final[float] = 10.0
_TOOL_REAP_TIMEOUT_SECONDS: Final[float] = 10.0

#: Completeness tolerance when the source audio-stream duration is known. The
#: extracted PCM duration is compared with the *source audio* duration (never the
#: video format duration, which is a different, legitimate quantity) and the
#: allowance covers one codec frame plus one output frame. ``2048`` samples
#: covers AAC (1024), MP3 (1152) and Opus (960) with margin; the floor of 50 ms
#: absorbs rounding at high sample rates. Synthetic fixtures measured a maximum
#: drift of 0.022 ms (see the Phase 5 boundary-fix evidence).
AUDIO_COMPLETENESS_FRAME_SAMPLES: Final[int] = 2048
AUDIO_COMPLETENESS_MIN_TOLERANCE_MS: Final[int] = 50

_EXTRACTED_LOCATORS: Final[dict[str, str]] = {
    "wav": "artifacts/extracted_audio.wav",
    "flac": "artifacts/extracted_audio.flac",
}

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class VideoMediaError(ValueError):
    """Raised when a video input or its extraction is not acceptable.

    Carries a stable ``code`` plus a human-readable ``message``; callers map the
    code to a documented exit code without ever echoing a raw traceback.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class VideoInfo(BaseModel):
    """Verified, immutable media facts for one accepted source video."""

    model_config = _FROZEN

    path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, gt=0)
    duration_ms: int = Field(strict=True, gt=0)
    container: str = Field(min_length=1)
    video_codec: str = Field(min_length=1)
    audio_codec: str = Field(min_length=1)
    audio_sample_rate: int = Field(strict=True, gt=0)
    audio_channels: int = Field(strict=True, gt=0)
    #: Reported duration of the *selected audio stream* when the container
    #: provides a reliable one; ``None`` when it is unknown (for example a
    #: matroska remux whose stream duration is absent). It is used only for a
    #: completeness comparison against the extracted audio and is never
    #: substituted by the video/format duration.
    audio_duration_ms: Annotated[int, Field(strict=True, gt=0)] | None = None
    selected_audio_stream_index: int = Field(strict=True, ge=0)
    format_start_ms: int
    video_start_ms: int
    audio_start_ms: int


class ExtractionResult(BaseModel):
    """One deterministic extraction of an accepted video."""

    model_config = _FROZEN

    extraction_id: str = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    settings: ExtractionSettings
    audio: AudioInfo
    already_present: bool = False


class ExtractionManifest(BaseModel):
    """Durable proof that the extracted artifact belongs to a frozen origin.

    Written **last**, after the artifact is atomically published, so a crash
    between the two is detectable and recoverable rather than being presented as
    success.
    """

    model_config = _FROZEN

    extraction_version: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: int = Field(strict=True, gt=0)
    source_duration_ms: int = Field(strict=True, gt=0)
    selected_audio_stream_index: int = Field(strict=True, ge=0)
    settings: ExtractionSettings
    extraction_id: str = Field(min_length=1)
    audio: StoredArtifact
    completed_at_utc: UtcDatetime


def extraction_locator(settings: ExtractionSettings) -> str:
    """Return the confined job-relative locator for the extracted audio."""

    return _EXTRACTED_LOCATORS[settings.container]


def compute_extraction_id(
    video: VideoInfo,
    settings: ExtractionSettings,
    tool_version: str,
) -> str:
    """Return the deterministic cache identity for one extraction.

    Binds the source SHA-256, the selected stream, the exact output settings,
    the extraction version and the tool version. It deliberately excludes any
    temporary path or timestamp so two independent runs agree.
    """

    payload = {
        "extraction_version": EXTRACTION_VERSION,
        "source_video_sha256": video.video_sha256,
        "selected_audio_stream_index": video.selected_audio_stream_index,
        "settings": settings.model_dump(mode="json"),
        "tool_version": tool_version,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_extraction_manifest(
    origin: VideoOriginSettings,
    settings: ExtractionSettings,
    audio: StoredArtifact,
    *,
    completed_at: datetime | None = None,
) -> ExtractionManifest:
    """Assemble the durable extraction manifest from a frozen origin."""

    return ExtractionManifest(
        extraction_version=origin.extraction_version,
        tool_version=origin.tool_version,
        source_path=origin.source_path,
        source_sha256=origin.source_sha256,
        source_size_bytes=origin.source_size_bytes,
        source_duration_ms=origin.source_duration_ms,
        selected_audio_stream_index=origin.selected_audio_stream_index,
        settings=settings,
        extraction_id=origin.extraction_id,
        audio=audio,
        completed_at_utc=completed_at if completed_at is not None else utc_now(),
    )


def resolve_source_path(path: str | os.PathLike[str]) -> Path:
    """Return a stable canonical absolute path for a local source file.

    The source file itself must not be a symlink. Its parent directory aliases
    are resolved exactly once (macOS ``/tmp`` -> ``/private/tmp`` is legitimate),
    after which every media tool runs against the bound path so the resolution
    cannot change underneath the probe/hash/extract sequence.
    """

    raw = os.fspath(path)
    if not isinstance(raw, str) or raw.strip() == "":
        raise VideoMediaError("VIDEO_PATH_INVALID", "video path must not be empty")
    candidate = Path(os.path.abspath(raw))
    if os.path.islink(candidate):
        raise VideoMediaError(
            "VIDEO_SYMLINK",
            "the source video must be a regular file, not a symlink",
        )
    try:
        info = os.stat(candidate)
    except FileNotFoundError as exc:
        raise VideoMediaError("VIDEO_MISSING", "video file does not exist") from exc
    except OSError as exc:
        raise VideoMediaError("VIDEO_UNREADABLE", "video file is not readable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise VideoMediaError(
            "VIDEO_NOT_REGULAR", "video path is not a regular file"
        )
    parent = os.path.realpath(os.path.dirname(candidate) or ".")
    resolved = Path(parent) / candidate.name
    if os.path.islink(resolved):
        raise VideoMediaError(
            "VIDEO_SYMLINK",
            "the resolved source video is a symlink; refusing an unbound path",
        )
    return resolved


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _verify_source_binding(video: VideoInfo) -> tuple[int, int, int, int, int]:
    """Re-prove the validated source bytes and return their stat identity.

    Reuses :func:`recheck_video` for the SHA-256/size proof against the validated
    :class:`VideoInfo`, then records the file's full stat identity
    (``dev``/``inode``/``size``/``mtime_ns``/``ctime_ns``). Extraction compares
    this value before and after ``ffmpeg`` reads the source, so a concurrent
    writer cannot make the published artifact belong to different bytes than the
    recorded origin: a replace-then-restore race (A->C->A) leaves the hash equal
    but still changes at least the inode or ctime and is refused. Only the source
    is read; the file and its timestamps are never written.
    """

    recheck_video(video.path, video)
    candidate = resolve_source_path(video.path)
    try:
        info = os.stat(candidate)
    except OSError as exc:
        raise VideoMediaError(
            "VIDEO_UNREADABLE", "cannot stat the source video"
        ) from exc
    if info.st_size != video.size_bytes:
        raise VideoMediaError(
            "VIDEO_CHANGED",
            "video size changed after the source was validated",
        )
    return _stat_identity(info)


def _drain_stream(
    stream: BinaryIO,
    buffer: bytearray,
    ceiling: int,
    overflow: threading.Event,
) -> None:
    """Drain one child pipe, retaining at most ``ceiling`` bytes.

    Bytes beyond the ceiling are discarded but the pipe keeps being drained so a
    killed child can never block on a full pipe. Any surplus sets ``overflow``,
    which the caller observes and turns into a typed failure after killing the
    child's process group.
    """

    try:
        descriptor = stream.fileno()
    except (OSError, ValueError):
        return
    retained = 0
    while True:
        try:
            block = os.read(descriptor, _TOOL_READ_CHUNK)
        except InterruptedError:
            continue
        except OSError:
            break
        if not block:
            break
        room = ceiling - retained
        if room >= len(block):
            buffer.extend(block)
            retained += len(block)
        else:
            if room > 0:
                buffer.extend(block[:room])
                retained += room
            overflow.set()
    try:
        stream.close()
    except OSError:
        pass


def _kill_process_group(
    process: "subprocess.Popen[bytes]", pgid: int | None
) -> None:
    """Kill every process still owned by this child; never raises.

    POSIX keeps the captured process-group semantics (``pgid`` is the direct
    child's own group, captured at spawn, so a descendant is reached even after
    the leader has been reaped). Windows has no group kill, so the platform
    helper targets the owned pid's tree while the leader is still alive. A
    failed cleanup never replaces the typed error the caller is about to raise.
    """

    platform_compat.kill_owned_process_group(process, pgid)


def _reap_process(process: "subprocess.Popen[bytes]") -> None:
    """Reap a direct child, escalating to a hard kill if it will not exit."""

    try:
        process.wait(timeout=_TOOL_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=_TOOL_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass


def _join_drainers(threads: list[threading.Thread], timeout: float) -> bool:
    """Join every pipe-drain thread within one total budget.

    Returns ``True`` only when all drain threads have observed EOF, which means
    no process (including a descendant that inherited stdout/stderr) still holds
    the pipes open. ``False`` means at least one reader is still blocked on an
    open pipe and the caller must fail closed instead of returning truncated
    output while an orphan writer lives on.
    """

    deadline = time.monotonic() + max(0.0, timeout)
    for thread in threads:
        remaining = deadline - time.monotonic()
        thread.join(timeout=remaining if remaining > 0 else 0.0)
    return all(not thread.is_alive() for thread in threads)


def _terminate_process_group(
    process: "subprocess.Popen[bytes]", pgid: int | None
) -> None:
    """Kill the group captured at spawn, then reap the direct leader.

    The group id is passed in rather than recomputed: once the leader has been
    reaped its pid no longer resolves, while a surviving descendant would still
    belong to the same known group. Only that owned group is ever signalled.
    """

    _kill_process_group(process, pgid)
    _reap_process(process)


def _run_tool(
    args: list[str],
    *,
    timeout: float,
    missing_code: str,
    missing_label: str,
    timeout_code: str,
    failed_code: str,
    overflow_code: str,
    size_guard: tuple[Path, int] | None = None,
    size_guard_code: str = "VIDEO_OUTPUT_TOO_LARGE",
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one external media tool with bounded output and a hard deadline.

    The child runs in its own session/process group with an argument list (no
    shell). Both stdout and stderr are drained incrementally with an explicit
    byte ceiling; on a timeout, an output-overflow, or an output-file size guard
    trigger the whole process group is killed and reaped before a typed
    :class:`VideoMediaError` is raised, so no pipe can hang and no unbounded
    output is retained. ``size_guard`` is a ``(path, max_bytes)`` pair checked
    while the child is still writing (used for the extraction byte budget).

    The owned group id is captured at spawn, so it still names surviving
    descendants after the direct leader has been reaped. When the leader exits
    but a descendant keeps stdout/stderr open, the call waits for every drain
    thread to reach EOF within the same deadline; a pipe that stays open is a
    typed timeout refusal rather than a success with truncated output.
    """

    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=(str(cwd) if cwd is not None else None),
            **platform_compat.owned_process_kwargs(),
        )
    except FileNotFoundError as exc:
        raise VideoMediaError(
            missing_code, f"{missing_label} not found; install ffmpeg"
        ) from exc
    except OSError as exc:
        raise VideoMediaError(
            failed_code, f"{missing_label} could not start"
        ) from exc

    # The platform helper makes the direct child the leader of its own process
    # group on POSIX (a fresh process group on Windows). Capture that group id
    # now: after the leader is reaped the pid can no longer be resolved here, but
    # the owned group id still names any surviving descendant and nothing outside
    # this owned group is ever signalled.
    pgid = process.pid
    assert process.stdout is not None and process.stderr is not None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow = threading.Event()
    threads = [
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_buffer, TOOL_STDOUT_CEILING_BYTES, overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_buffer, TOOL_STDERR_CEILING_BYTES, overflow),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    deadline = time.monotonic() + timeout
    reason: str | None = None
    leader_exited = False
    while True:
        if overflow.is_set():
            reason = "overflow"
            break
        if size_guard is not None:
            guard_path, guard_limit = size_guard
            try:
                if guard_path.stat().st_size > guard_limit:
                    reason = "size_guard"
                    break
            except OSError:
                pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            reason = "timeout"
            break
        try:
            process.wait(timeout=min(_TOOL_POLL_INTERVAL_SECONDS, remaining))
            leader_exited = True
            break
        except subprocess.TimeoutExpired:
            continue

    pipe_closed_in_time = True
    if reason is not None:
        _terminate_process_group(process, pgid)
        _join_drainers(threads, _TOOL_JOIN_TIMEOUT_SECONDS)
    elif leader_exited:
        # The direct leader exited, but a descendant may have inherited
        # stdout/stderr and still hold the pipes open. Do not report success
        # until every drain thread has reached EOF within the same deadline; a
        # lingering inherited pipe is a bounded refusal, never a success with
        # truncated output or an orphan writer.
        pipe_closed_in_time = _join_drainers(threads, deadline - time.monotonic())
        if not pipe_closed_in_time:
            _terminate_process_group(process, pgid)
            _join_drainers(threads, _TOOL_JOIN_TIMEOUT_SECONDS)
    # Reap the leader; a no-op when ``wait`` already collected it above.
    try:
        process.wait(timeout=_TOOL_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process, pgid)

    # A child can finish between polls; make the same limits deterministic for a
    # completed run as for a killed one.
    if reason is None and overflow.is_set():
        reason = "overflow"
    if reason is None and size_guard is not None:
        guard_path, guard_limit = size_guard
        try:
            if guard_path.stat().st_size > guard_limit:
                reason = "size_guard"
        except OSError:
            pass
    if reason is None and not pipe_closed_in_time:
        # A descendant that inherited stdout/stderr kept a pipe open past the
        # deadline. The owned group was killed above; fail closed rather than
        # returning output that may be truncated while an orphan writer lived on.
        reason = "timeout"

    stdout = bytes(stdout_buffer).decode("utf-8", errors="replace")
    stderr = bytes(stderr_buffer).decode("utf-8", errors="replace")

    if reason == "overflow":
        raise VideoMediaError(
            overflow_code,
            f"{missing_label} produced more than the bounded "
            f"{TOOL_STDOUT_CEILING_BYTES} byte stdout/stderr ceiling; "
            "the process was stopped instead of buffering unbounded output",
        )
    if reason == "size_guard":
        assert size_guard is not None
        guard_limit = size_guard[1]
        raise VideoMediaError(
            size_guard_code,
            f"{missing_label} wrote more than the {guard_limit} byte extraction "
            "budget; the partial output was stopped and is refused",
        )
    if reason == "timeout":
        raise VideoMediaError(
            timeout_code, f"{missing_label} timed out after {timeout}s"
        )

    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def ffmpeg_tool_version(ffmpeg_bin: str = "ffmpeg", *, timeout: float = 30.0) -> str:
    """Return the installed ``ffmpeg`` version string, recorded as provenance."""

    completed = _run_tool(
        [ffmpeg_bin, "-version"],
        timeout=timeout,
        missing_code="FFMPEG_MISSING",
        missing_label="ffmpeg",
        timeout_code="FFMPEG_TIMEOUT",
        failed_code="FFMPEG_FAILED",
        overflow_code="FFMPEG_OUTPUT_OVERFLOW",
    )
    if completed.returncode != 0:
        raise VideoMediaError("FFMPEG_FAILED", "ffmpeg -version failed")
    first_line = (completed.stdout or "").strip().splitlines()
    if not first_line or not first_line[0].strip():
        raise VideoMediaError("FFMPEG_FAILED", "ffmpeg did not report its version")
    return first_line[0].strip()[:255]


def _probe_video(
    path: Path, *, ffprobe_bin: str, timeout: float
) -> dict[str, object]:
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
    completed = _run_tool(
        args,
        timeout=timeout,
        missing_code="FFPROBE_MISSING",
        missing_label="ffprobe",
        timeout_code="FFPROBE_TIMEOUT",
        failed_code="FFPROBE_FAILED",
        overflow_code="FFPROBE_OUTPUT_OVERFLOW",
    )
    if completed.returncode != 0:
        raise VideoMediaError(
            "FFPROBE_FAILED",
            "ffprobe could not read the file; the container is invalid, "
            "unsupported or uses a non-local protocol",
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise VideoMediaError("FFPROBE_FAILED", "ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise VideoMediaError("FFPROBE_FAILED", "ffprobe returned an unexpected payload")
    return payload


def _finite_float(value: object, *, label: str) -> float:
    if value is None or value == "" or value == "N/A":
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", f"{label} is missing from the probe result"
        )
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", f"{label} is not a number"
        ) from exc
    if not math.isfinite(parsed):
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", f"{label} must be a finite number"
        )
    return parsed


def _required_start_seconds(value: object, *, label: str) -> Decimal:
    """Parse a required start time as an exact finite decimal.

    The comparison against the supported tolerance happens on this exact source
    value *before* any rounding to integer milliseconds, so ``0.0014`` cannot be
    rounded to ``1`` and accepted. A missing, unparsable or non-finite start time
    means the source timeline cannot be proven zero-based and is reported as an
    unsupported offset rather than a generic metadata error.
    """

    if value is None or value == "" or value == "N/A":
        raise VideoMediaError(
            "VIDEO_UNSUPPORTED_OFFSET",
            f"the {label} start time is not reported; the timeline cannot be "
            "proven zero-based",
        )
    try:
        seconds = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise VideoMediaError(
            "VIDEO_UNSUPPORTED_OFFSET",
            f"the {label} start time is not a number",
        ) from exc
    if not seconds.is_finite():
        raise VideoMediaError(
            "VIDEO_UNSUPPORTED_OFFSET",
            f"the {label} start time is not finite",
        )
    return seconds


def _round_start_ms(seconds: Decimal) -> int:
    """Round an already-accepted start time to integer milliseconds.

    Called only after the exact source value passed the tolerance comparison, so
    the rounding can never enlarge the accepted window.
    """

    return int((seconds * 1000).to_integral_value(rounding=ROUND_HALF_EVEN))


def _duration_ms_from_stream(stream: dict[str, object]) -> int | None:
    """Return a reliable audio-stream duration in ms, or ``None`` when unknown.

    ``duration`` is preferred; when it is absent the ``duration_ts``/``time_base``
    pair is used if present. A container that reports neither (for example a
    matroska remux) yields ``None`` and no duration is invented from the video or
    format duration.
    """

    raw = stream.get("duration")
    if raw not in (None, "", "N/A"):
        try:
            seconds = float(str(raw))
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None and math.isfinite(seconds) and seconds > 0:
            return max(1, int(round(seconds * 1000)))
    raw_ts = stream.get("duration_ts")
    raw_tb = stream.get("time_base")
    if raw_ts in (None, "", "N/A") or not isinstance(raw_tb, str):
        return None
    numerator_text, _separator, denominator_text = raw_tb.partition("/")
    try:
        duration_ts = int(str(raw_ts))
        numerator = int(numerator_text)
        denominator = int(denominator_text)
    except (TypeError, ValueError):
        return None
    if duration_ts <= 0 or numerator <= 0 or denominator <= 0:
        return None
    seconds = duration_ts * numerator / denominator
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return max(1, int(round(seconds * 1000)))


def _parse_video_metadata(
    payload: dict[str, object], *, limits: VideoLimits
) -> dict[str, object]:
    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", "ffprobe did not report a stream list"
        )
    streams: list[dict[str, object]] = []
    for entry in raw_streams:
        if not isinstance(entry, dict):
            raise VideoMediaError(
                "VIDEO_METADATA_INVALID", "ffprobe reported a malformed stream entry"
            )
        streams.append(entry)

    format_info = payload.get("format")
    format_map = format_info if isinstance(format_info, dict) else {}

    format_name = str(format_map.get("format_name") or "").lower()
    # Keep the demuxer's own order (for example ``mov,mp4,m4a,...``) so the
    # recorded container token is deterministic run to run.
    names = [name.strip() for name in format_name.split(",") if name.strip()]
    container = next(
        (name for name in names if name in ACCEPTED_VIDEO_CONTAINER_TOKENS), None
    )
    if container is None:
        raise VideoMediaError(
            "VIDEO_UNSUPPORTED_FORMAT",
            "container is not a supported local video container; accepted "
            "families are mp4/mov and matroska/webm",
        )

    def _is_real_video(entry: dict[str, object]) -> bool:
        if entry.get("codec_type") != "video":
            return False
        disposition = entry.get("disposition")
        if isinstance(disposition, dict) and disposition.get("attached_pic") == 1:
            return False
        return True

    video_streams = [entry for entry in streams if _is_real_video(entry)]
    audio_streams = [entry for entry in streams if entry.get("codec_type") == "audio"]

    if not video_streams:
        raise VideoMediaError(
            "VIDEO_NO_VIDEO", "the file has no video stream; use the audio path"
        )
    if not audio_streams:
        raise VideoMediaError(
            "VIDEO_NO_AUDIO", "the video file has no audio stream"
        )
    if len(audio_streams) > 1:
        raise VideoMediaError(
            "VIDEO_MULTIPLE_AUDIO",
            f"the video file has {len(audio_streams)} audio streams; exactly one "
            "is required and no default-language stream is selected silently",
        )

    video_stream = video_streams[0]
    audio_stream = audio_streams[0]
    try:
        selected_audio_index = int(str(audio_stream.get("index")))
    except (TypeError, ValueError):
        selected_audio_index = -1
    if selected_audio_index < 0:
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", "audio stream has no numeric index"
        )

    video_codec = video_stream.get("codec_name")
    audio_codec = audio_stream.get("codec_name")
    if not isinstance(video_codec, str) or not video_codec:
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", "video stream has no codec name"
        )
    if not isinstance(audio_codec, str) or not audio_codec:
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", "audio stream has no codec name"
        )

    def _positive_int(value: object, label: str) -> int:
        try:
            parsed = int(str(value))
        except (TypeError, ValueError) as exc:
            raise VideoMediaError(
                "VIDEO_METADATA_INVALID", f"audio stream {label} is not a positive integer"
            ) from exc
        if parsed <= 0:
            raise VideoMediaError(
                "VIDEO_METADATA_INVALID", f"audio stream {label} must be positive"
            )
        return parsed

    sample_rate = _positive_int(audio_stream.get("sample_rate"), "sample_rate")
    channels = _positive_int(audio_stream.get("channels"), "channels")

    raw_duration = format_map.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = video_stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = audio_stream.get("duration")
    seconds = _finite_float(raw_duration, label="duration")
    if seconds <= 0:
        raise VideoMediaError(
            "VIDEO_METADATA_INVALID", "video duration must be a positive number"
        )
    duration_ms = int(round(seconds * 1000))
    if duration_ms < limits.min_duration_ms:
        raise VideoMediaError(
            "VIDEO_DURATION_INVALID",
            f"video duration {duration_ms}ms is below the minimum "
            f"{limits.min_duration_ms}ms",
        )
    if duration_ms > limits.max_duration_ms:
        raise VideoMediaError(
            "VIDEO_TOO_LONG",
            f"video duration {duration_ms}ms exceeds the limit "
            f"{limits.max_duration_ms}ms",
        )

    tolerance_ms = limits.start_tolerance_ms
    tolerance_seconds = Decimal(tolerance_ms) / Decimal(1000)
    starts: dict[str, int] = {}
    for label, raw in (
        ("format", format_map.get("start_time")),
        ("video stream", video_stream.get("start_time")),
        ("audio stream", audio_stream.get("start_time")),
    ):
        exact = _required_start_seconds(raw, label=label)
        # Compare the exact finite source value before any millisecond rounding.
        if abs(exact) > tolerance_seconds:
            raise VideoMediaError(
                "VIDEO_UNSUPPORTED_OFFSET",
                f"the {label} start time is {exact}s, outside the supported "
                f"+/-{tolerance_ms}ms window; a nonzero offset cannot be carried "
                "by the extracted audio and no mapping is proven in this version",
            )
        starts[label] = _round_start_ms(exact)

    return {
        "container": container,
        "duration_ms": duration_ms,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "audio_sample_rate": sample_rate,
        "audio_channels": channels,
        "audio_duration_ms": _duration_ms_from_stream(audio_stream),
        "selected_audio_stream_index": selected_audio_index,
        "format_start_ms": starts["format"],
        "video_start_ms": starts["video stream"],
        "audio_start_ms": starts["audio stream"],
    }


def _decode_all(path: Path, *, ffmpeg_bin: str, timeout: float) -> None:
    """Decode every video/audio stream to ``null`` with ``-xerror``.

    A faststart-truncated file can probe cleanly; only a full strict decode
    proves the container is complete. Any nonzero exit is a corrupt/truncated
    rejection before any provider is reached.
    """

    args = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-xerror",
        "-protocol_whitelist",
        LOCAL_PROTOCOLS,
        "-i",
        str(path),
        "-map",
        "0:v",
        "-map",
        "0:a",
        "-f",
        "null",
        "-",
    ]
    completed = _run_tool(
        args,
        timeout=timeout,
        missing_code="FFMPEG_MISSING",
        missing_label="ffmpeg",
        timeout_code="FFMPEG_TIMEOUT",
        failed_code="FFMPEG_FAILED",
        overflow_code="FFMPEG_OUTPUT_OVERFLOW",
    )
    if completed.returncode != 0:
        raise VideoMediaError(
            "VIDEO_CORRUPT",
            "the video did not decode cleanly; it is corrupt or truncated",
        )


def validate_video(
    path: str | os.PathLike[str],
    *,
    limits: VideoLimits | None = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
) -> VideoInfo:
    """Validate one local video file and return its measured facts.

    The bound canonical path is hashed before probing/decoding and again
    afterwards while its stat identity stays stable, so the returned hash and
    metadata describe the exact bytes that decoded successfully. The original
    file is only ever read.
    """

    limits = limits if limits is not None else VideoLimits()
    candidate = resolve_source_path(path)
    initial = os.stat(candidate)
    size = initial.st_size
    if size == 0:
        raise VideoMediaError("VIDEO_EMPTY", "video file is empty")
    if size > limits.max_bytes:
        raise VideoMediaError(
            "VIDEO_TOO_LARGE",
            f"video file is {size} bytes, exceeding the limit {limits.max_bytes}",
        )

    sha_before, hashed_before = sha256_file(candidate)
    if hashed_before != size:
        raise VideoMediaError(
            "VIDEO_CHANGED",
            "video file changed size while being validated; retry with a stable file",
        )

    payload = _probe_video(
        candidate, ffprobe_bin=ffprobe_bin, timeout=limits.probe_timeout_seconds
    )
    facts = _parse_video_metadata(payload, limits=limits)
    _decode_all(
        candidate, ffmpeg_bin=ffmpeg_bin, timeout=limits.decode_timeout_seconds
    )

    final = os.stat(candidate)
    if _stat_identity(final) != _stat_identity(initial):
        raise VideoMediaError(
            "VIDEO_CHANGED",
            "video file changed while being validated; retry with a stable file",
        )
    sha_after, hashed_after = sha256_file(candidate)
    if sha_after != sha_before or hashed_after != hashed_before:
        raise VideoMediaError(
            "VIDEO_CHANGED",
            "video file changed while being validated; retry with a stable file",
        )

    return VideoInfo(
        path=str(candidate),
        original_filename=candidate.name,
        video_sha256=sha_after,
        size_bytes=hashed_after,
        duration_ms=int(facts["duration_ms"]),
        container=str(facts["container"]),
        video_codec=str(facts["video_codec"]),
        audio_codec=str(facts["audio_codec"]),
        audio_sample_rate=int(facts["audio_sample_rate"]),
        audio_channels=int(facts["audio_channels"]),
        audio_duration_ms=(
            int(facts["audio_duration_ms"])
            if facts["audio_duration_ms"] is not None
            else None
        ),
        selected_audio_stream_index=int(facts["selected_audio_stream_index"]),
        format_start_ms=int(facts["format_start_ms"]),
        video_start_ms=int(facts["video_start_ms"]),
        audio_start_ms=int(facts["audio_start_ms"]),
    )


def _extraction_args(
    video: VideoInfo,
    settings: ExtractionSettings,
    out_path: Path,
    *,
    ffmpeg_bin: str,
) -> list[str]:
    if settings.container == "wav":
        codec = "pcm_s16le"
        sample_fmt = "s16"
        format_name = "wav"
    elif settings.container == "flac":
        codec = "flac"
        sample_fmt = "s16"
        format_name = "flac"
    else:  # pragma: no cover - Literal-constrained by ExtractionSettings
        raise VideoMediaError(
            "VIDEO_SETTINGS_INVALID", f"unsupported container {settings.container!r}"
        )
    return [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-xerror",
        "-protocol_whitelist",
        LOCAL_PROTOCOLS,
        "-i",
        video.path,
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        str(settings.channels),
        "-ar",
        str(settings.sample_rate),
        "-sample_fmt",
        sample_fmt,
        "-c:a",
        codec,
        "-f",
        format_name,
        "-y",
        str(out_path),
    ]


def _completeness_tolerance_ms(sample_rate: int) -> int:
    """Return the codec/resampler allowance for a source audio duration check."""

    frame_ms = math.ceil(AUDIO_COMPLETENESS_FRAME_SAMPLES * 1000 / sample_rate)
    return max(AUDIO_COMPLETENESS_MIN_TOLERANCE_MS, frame_ms + 1)


def _check_source_audio_completeness(video: VideoInfo, audio: AudioInfo) -> None:
    """Refuse an extracted duration that does not match a known source audio stream.

    Only the *source audio* duration is used; the video format duration is a
    different, legitimate quantity and is never substituted. When the container
    reports no reliable audio-stream duration the check is skipped (no duration is
    invented), and the output is still required to pass a full strict decode and
    the deterministic in-write byte guard.
    """

    source_ms = video.audio_duration_ms
    if source_ms is None:
        return
    tolerance_ms = _completeness_tolerance_ms(video.audio_sample_rate)
    drift_ms = abs(audio.duration_ms - source_ms)
    if drift_ms > tolerance_ms:
        raise VideoMediaError(
            "VIDEO_OUTPUT_INCOMPLETE",
            f"the extracted audio is {audio.duration_ms}ms but the source audio "
            f"stream is {source_ms}ms ({drift_ms}ms drift, beyond the "
            f"{tolerance_ms}ms codec/resampler tolerance); refusing a truncated "
            "or clipped extraction",
        )


def _discard_partial_output(out_path: Path) -> None:
    """Remove a failed/over-limit extraction temp so it is never published."""

    try:
        out_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def extract_audio(
    video: VideoInfo,
    *,
    settings: ExtractionSettings,
    out_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    tool_version: str,
    limits: VideoLimits | None = None,
    media_limits: MediaLimits | None = None,
    ffprobe_bin: str = "ffprobe",
) -> ExtractionResult:
    """Extract deterministic mono 16 kHz audio from an accepted video.

    The output byte budget is enforced *while* ffmpeg writes: the running child
    is killed (with its process group) as soon as the temp file exceeds
    ``limits.max_output_bytes`` and the partial file is removed. Only after a
    zero exit and a byte check is the output full-decoded and validated by the
    existing ready-audio validator; a known source audio-stream duration is then
    compared for completeness. ``-fs``/``-t`` are deliberately not used, so a
    clipped file is never accepted as success. The original video is never
    modified and the extraction identity is a pure function of the source hash,
    the settings and the tool version.

    Extraction is bound to the validated source identity: the source bytes and
    their full stat identity are re-proved immediately before ``ffmpeg`` runs and
    again immediately after it returns. If the source changed in between (or was
    replaced and later restored while ``ffmpeg`` was reading it), the call is
    refused with a typed :class:`VideoMediaError` and the partial output is
    discarded, so no artifact/manifest can be published under a mismatched origin.
    """

    limits = limits if limits is not None else VideoLimits()
    media = media_limits if media_limits is not None else MediaLimits()
    out_path = Path(out_path)
    args = _extraction_args(video, settings, out_path, ffmpeg_bin=ffmpeg_bin)
    try:
        # Prove the source is still the validated bytes and record its stat
        # identity before handing the path to ffmpeg.
        baseline_identity = _verify_source_binding(video)
        completed = _run_tool(
            args,
            timeout=limits.extraction_timeout_seconds,
            missing_code="FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="FFMPEG_TIMEOUT",
            failed_code="FFMPEG_FAILED",
            overflow_code="FFMPEG_OUTPUT_OVERFLOW",
            size_guard=(out_path, limits.max_output_bytes),
            size_guard_code="VIDEO_OUTPUT_TOO_LARGE",
        )
        # Prove the source did not change (or change and change back) while
        # ffmpeg was reading it. Comparing the hash alone would miss an
        # A->C->A restore; the full stat identity (inode/ctime) does not.
        if _verify_source_binding(video) != baseline_identity:
            raise VideoMediaError(
                "VIDEO_CHANGED",
                "the source video was modified or replaced during extraction; "
                "refusing to publish audio that was not derived from the "
                "validated source",
            )
        if completed.returncode != 0:
            raise VideoMediaError(
                "VIDEO_EXTRACT_FAILED",
                "ffmpeg could not extract a complete audio stream from the video",
            )
        try:
            output_size = out_path.stat().st_size
        except OSError as exc:
            raise VideoMediaError(
                "VIDEO_EXTRACT_FAILED", "ffmpeg did not produce an audio file"
            ) from exc
        if output_size == 0:
            raise VideoMediaError(
                "VIDEO_EXTRACT_FAILED", "ffmpeg produced an empty audio file"
            )
        # Check bytes before invoking the full decode/validation.
        if output_size > limits.max_output_bytes:
            raise VideoMediaError(
                "VIDEO_OUTPUT_TOO_LARGE",
                f"the extracted audio is {output_size} bytes, exceeding the "
                f"limit {limits.max_output_bytes}",
            )
        try:
            audio = validate_audio(
                out_path,
                limits=media,
                ffprobe_bin=ffprobe_bin,
                ffmpeg_bin=ffmpeg_bin,
            )
        except MediaError as exc:
            raise VideoMediaError(
                "VIDEO_OUTPUT_INVALID",
                f"the extracted audio was rejected ({exc.code})",
            ) from exc
        _check_source_audio_completeness(video, audio)
        return ExtractionResult(
            extraction_id=compute_extraction_id(video, settings, tool_version),
            extraction_version=EXTRACTION_VERSION,
            tool_version=tool_version,
            settings=settings,
            audio=audio,
        )
    except BaseException:
        # A failed or over-limit temp is never published or manifested.
        _discard_partial_output(out_path)
        raise


def recheck_video(path: str | os.PathLike[str], expected: VideoInfo) -> VideoInfo:
    """Re-measure identity and fail if the bytes differ from ``expected``."""

    candidate = resolve_source_path(path)
    try:
        size = os.path.getsize(candidate)
    except OSError as exc:
        raise VideoMediaError(
            "VIDEO_UNREADABLE", "cannot stat the source video"
        ) from exc
    if size != expected.size_bytes:
        raise VideoMediaError(
            "VIDEO_CHANGED",
            f"video size changed from {expected.size_bytes} to {size}",
        )
    digest, measured = sha256_file(candidate)
    if digest != expected.video_sha256 or measured != expected.size_bytes:
        raise VideoMediaError(
            "VIDEO_CHANGED",
            "video content no longer matches the recorded SHA-256",
        )
    return expected
