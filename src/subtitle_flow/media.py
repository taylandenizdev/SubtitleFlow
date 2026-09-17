"""Ready-audio validation for WAV/FLAC inputs.

Acceptance is based on the actual container and audio stream, never on the file
extension. A WAV/FLAC file must contain exactly one audio stream and no other
streams; video, attached cover art and other non-audio streams are rejected
explicitly for this phase. ``ffprobe`` supplies duration, codec, channels,
sample rate and container; a full ``ffmpeg`` decode to ``null`` then proves the
file is not a truncated or corrupt container that only *looks* valid from its
header. The original file is only ever read, never rewritten.

The hash and probed metadata are bound to the same bytes that were decoded: the
file is hashed before probing/decoding and re-hashed afterwards while its stat
identity (device, inode, size, mtime, ctime) is required to stay stable, so an
ordinary same-size replacement or in-place rewrite is rejected rather than
described by stale metadata.

External processes are launched with an argument list (no shell) and a timeout,
so shell metacharacters are inert and a hung probe cannot block a job forever.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from subtitle_flow.config import MediaLimits

__all__ = [
    "ACCEPTED_CONTAINERS",
    "AudioInfo",
    "MediaError",
    "recheck_audio",
    "sha256_file",
    "validate_audio",
]

#: Container families accepted by the first pilot. Extension is irrelevant.
ACCEPTED_CONTAINERS: Final[frozenset[str]] = frozenset({"wav", "flac"})

_READ_CHUNK: Final[int] = 1024 * 1024


class MediaError(ValueError):
    """Raised when an input file is not an acceptable ready-audio file."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class AudioInfo(BaseModel):
    """Verified, immutable media facts for one accepted input file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(strict=True, gt=0)
    duration_ms: int = Field(strict=True, gt=0)
    container: str = Field(min_length=1)
    codec_name: str = Field(min_length=1)
    sample_rate: int = Field(strict=True, gt=0)
    channels: int = Field(strict=True, gt=0)


def sha256_file(path: str | Path) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` by streaming the file once."""

    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _require_regular_file(path: Path) -> os.stat_result:
    try:
        info = os.stat(path)
    except FileNotFoundError as exc:
        raise MediaError("AUDIO_MISSING", f"audio file does not exist: {path}") from exc
    except OSError as exc:
        raise MediaError("AUDIO_UNREADABLE", f"audio file is not readable: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise MediaError("AUDIO_NOT_REGULAR", f"audio path is not a regular file: {path}")
    return info


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the stat fields that must stay stable while validating a file."""

    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _run_probe(
    args: list[str],
    *,
    timeout: float,
    missing_code: str,
    missing_label: str,
    timeout_code: str,
    failed_code: str,
) -> subprocess.CompletedProcess[str]:
    """Run one external media tool, mapping process failures to ``MediaError``."""

    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaError(missing_code, f"{missing_label} not found: {args[0]!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(
            timeout_code, f"{missing_label} timed out after {timeout}s: {args[-1]}"
        ) from exc
    except OSError as exc:
        raise MediaError(failed_code, f"{missing_label} could not start: {exc}") from exc


def _probe_media(path: Path, *, ffprobe_bin: str, timeout: float) -> dict[str, object]:
    args = [
        ffprobe_bin,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = _run_probe(
        args,
        timeout=timeout,
        missing_code="FFPROBE_MISSING",
        missing_label="ffprobe",
        timeout_code="FFPROBE_TIMEOUT",
        failed_code="FFPROBE_FAILED",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise MediaError(
            "FFPROBE_FAILED",
            f"ffprobe could not read the file: {detail[-1] if detail else 'unknown error'}",
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError("FFPROBE_FAILED", "ffprobe returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise MediaError("FFPROBE_FAILED", "ffprobe returned an unexpected payload")
    return payload


def _parse_stream_metadata(
    payload: dict[str, object], *, limits: MediaLimits
) -> tuple[str, str, int, int, int]:
    """Return ``(container, codec, sample_rate, channels, duration_ms)``."""

    raw_streams = payload.get("streams")
    if not isinstance(raw_streams, list):
        raise MediaError("AUDIO_METADATA_INVALID", "ffprobe did not report a stream list")
    streams: list[dict[str, object]] = []
    for entry in raw_streams:
        if not isinstance(entry, dict):
            raise MediaError(
                "AUDIO_METADATA_INVALID", "ffprobe reported a malformed stream entry"
            )
        codec_type = entry.get("codec_type")
        if not isinstance(codec_type, str) or not codec_type:
            raise MediaError(
                "AUDIO_METADATA_INVALID", "ffprobe stream is missing its codec_type"
            )
        streams.append(entry)

    format_info = payload.get("format")
    format_map = format_info if isinstance(format_info, dict) else {}

    format_name = str(format_map.get("format_name") or "").lower()
    names = {name.strip() for name in format_name.split(",") if name.strip()}
    audio_streams = [entry for entry in streams if entry.get("codec_type") == "audio"]
    non_audio_streams = [
        entry for entry in streams if entry.get("codec_type") != "audio"
    ]

    has_video = any(entry.get("codec_type") == "video" for entry in streams)
    container = next((name for name in names if name in ACCEPTED_CONTAINERS), None)
    if container is None:
        if has_video and not audio_streams:
            raise MediaError(
                "AUDIO_VIDEO_INPUT",
                "video input is not accepted; only ready WAV/FLAC audio is supported",
            )
        shown = format_name or "unknown"
        raise MediaError(
            "AUDIO_UNSUPPORTED_FORMAT",
            f"container {shown!r} is not accepted; only WAV/FLAC are supported",
        )
    if not audio_streams:
        raise MediaError(
            "AUDIO_NO_STREAM", f"{container} file has no audio stream: {format_name!r}"
        )
    if len(audio_streams) > 1:
        raise MediaError(
            "AUDIO_MULTIPLE_STREAMS",
            f"{container} file has {len(audio_streams)} audio streams; "
            "exactly one audio stream is required",
        )
    if non_audio_streams:
        kinds = sorted(
            {str(entry.get("codec_type") or "unknown") for entry in non_audio_streams}
        )
        raise MediaError(
            "AUDIO_NON_AUDIO_STREAM",
            f"{container} file contains non-audio stream(s) {kinds}; video, cover "
            "art and other non-audio streams are not supported",
        )

    stream = audio_streams[0]

    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec:
        raise MediaError("AUDIO_METADATA_INVALID", "audio stream has no codec name")

    def _positive_int(value: object, label: str) -> int:
        try:
            parsed = int(str(value))
        except (TypeError, ValueError) as exc:
            raise MediaError(
                "AUDIO_METADATA_INVALID", f"audio stream {label} is not a positive integer"
            ) from exc
        if parsed <= 0:
            raise MediaError(
                "AUDIO_METADATA_INVALID", f"audio stream {label} must be positive"
            )
        return parsed

    sample_rate = _positive_int(stream.get("sample_rate"), "sample_rate")
    channels = _positive_int(stream.get("channels"), "channels")

    raw_duration = format_map.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = stream.get("duration")
    try:
        seconds = float(str(raw_duration))
    except (TypeError, ValueError) as exc:
        raise MediaError(
            "AUDIO_METADATA_INVALID", "media duration is missing or not a number"
        ) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise MediaError(
            "AUDIO_METADATA_INVALID", "media duration must be a finite positive number"
        )
    duration_ms = int(round(seconds * 1000))
    if duration_ms < limits.min_duration_ms:
        raise MediaError(
            "AUDIO_DURATION_INVALID",
            f"media duration {duration_ms}ms is below the minimum "
            f"{limits.min_duration_ms}ms",
        )
    if duration_ms > limits.max_duration_ms:
        raise MediaError(
            "AUDIO_TOO_LONG",
            f"media duration {duration_ms}ms exceeds the limit "
            f"{limits.max_duration_ms}ms",
        )
    return container, codec, sample_rate, channels, duration_ms


def _verify_decode(path: Path, *, ffmpeg_bin: str, timeout: float) -> None:
    """Decode the first audio stream to ``null`` to prove the file is intact."""

    args = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-vn",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    completed = _run_probe(
        args,
        timeout=timeout,
        missing_code="FFMPEG_MISSING",
        missing_label="ffmpeg",
        timeout_code="FFMPEG_TIMEOUT",
        failed_code="FFMPEG_FAILED",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise MediaError(
            "AUDIO_CORRUPT",
            "audio stream did not decode cleanly: "
            f"{detail[-1] if detail else 'unknown decode error'}",
        )


def validate_audio(
    path: str | Path,
    *,
    limits: MediaLimits | None = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    verify_decode: bool | None = None,
) -> AudioInfo:
    """Validate one ready-audio file and return its measured facts.

    The file is hashed before probing/decoding and again afterwards, and its stat
    identity (device, inode, size, mtime, ctime) must stay stable across every
    step. This binds the recorded hash, size and probed metadata to the exact
    bytes that were successfully decoded, so a same-size replacement or an
    in-place rewrite during validation is rejected instead of being described by
    stale metadata. Ordinary concurrent mutations are detected; this is not an
    adversarial kernel-race guarantee.

    Args:
        path: Local candidate file. Extension is ignored for acceptance; the
            stored path is absolute so later rechecks do not depend on the
            caller's working directory (this also makes a leading ``-`` filename
            safe for the media tools).
        limits: Byte/duration/timeout ceilings. Defaults to :class:`MediaLimits`.
        ffprobe_bin: ``ffprobe`` executable (argument list, never a shell string).
        ffmpeg_bin: ``ffmpeg`` executable used for the decode check.
        verify_decode: Only ``True``/``None`` are accepted. ``False`` is rejected
            because accepted audio must always be decode-verified.

    Raises:
        MediaError: with a stable ``code`` for every rejection reason.
    """

    limits = limits if limits is not None else MediaLimits()
    if verify_decode is not None and not verify_decode:
        raise MediaError(
            "AUDIO_DECODE_REQUIRED",
            "decode verification cannot be disabled; a full ffmpeg decode is "
            "required before audio is accepted",
        )

    candidate = Path(os.path.abspath(os.fspath(path)))
    initial = _require_regular_file(candidate)
    size = initial.st_size
    if size == 0:
        raise MediaError("AUDIO_EMPTY", f"audio file is empty: {candidate}")
    if size > limits.max_bytes:
        raise MediaError(
            "AUDIO_TOO_LARGE",
            f"audio file is {size} bytes, exceeding the limit {limits.max_bytes}",
        )

    sha_before, hashed_before = sha256_file(candidate)
    if hashed_before != size:
        raise MediaError(
            "AUDIO_CHANGED",
            "audio file changed size while being validated; retry with a stable file",
        )

    payload = _probe_media(
        candidate, ffprobe_bin=ffprobe_bin, timeout=limits.probe_timeout_seconds
    )
    container, codec, sample_rate, channels, duration_ms = _parse_stream_metadata(
        payload, limits=limits
    )

    _verify_decode(
        candidate, ffmpeg_bin=ffmpeg_bin, timeout=limits.decode_timeout_seconds
    )

    final = _require_regular_file(candidate)
    if _stat_identity(final) != _stat_identity(initial):
        raise MediaError(
            "AUDIO_CHANGED",
            "audio file changed while being validated; retry with a stable file",
        )
    sha_after, hashed_after = sha256_file(candidate)
    if sha_after != sha_before or hashed_after != hashed_before:
        raise MediaError(
            "AUDIO_CHANGED",
            "audio file changed while being validated; retry with a stable file",
        )

    return AudioInfo(
        path=str(candidate),
        original_filename=candidate.name,
        audio_sha256=sha_after,
        size_bytes=hashed_after,
        duration_ms=duration_ms,
        container=container,
        codec_name=codec,
        sample_rate=sample_rate,
        channels=channels,
    )


def recheck_audio(path: str | Path, expected: AudioInfo) -> AudioInfo:
    """Re-measure identity and fail if the bytes differ from ``expected``.

    Used at processing and resume points so validation can never silently refer
    to a different file than the one the job was created from.
    """

    candidate = Path(path)
    _require_regular_file(candidate)
    try:
        size = os.path.getsize(candidate)
    except OSError as exc:
        raise MediaError("AUDIO_UNREADABLE", f"cannot stat audio file: {candidate}") from exc
    if size != expected.size_bytes:
        raise MediaError(
            "AUDIO_CHANGED",
            f"audio file size changed from {expected.size_bytes} to {size}",
        )
    sha256, measured_size = sha256_file(candidate)
    if sha256 != expected.audio_sha256 or measured_size != expected.size_bytes:
        raise MediaError(
            "AUDIO_CHANGED",
            "audio file content no longer matches the job's recorded SHA-256",
        )
    return expected
