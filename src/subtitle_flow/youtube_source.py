"""Single public YouTube source support for the accepted audio pipeline.

This module is the only place that understands a YouTube URL. It performs three
tightly scoped jobs and nothing else:

1. **Offline URL acceptance.** Only a single public YouTube video URL in the
   documented forms is accepted and canonicalized to
   ``https://www.youtube.com/watch?v=ID``. Playlists, live/upcoming URLs,
   channels, lookalike hosts, userinfo, non-default ports, extra/ambiguous
   ``v`` parameters, unsupported query parameters and control characters are
   refused before any process runs.
2. **Bounded audio-only acquisition.** The external ``yt-dlp`` binary (through
   :mod:`subtitle_flow.yt_dlp_runner`) downloads only ``bestaudio`` into a
   private working directory. The downloaded intermediate is probed, fully
   decoded and its duration measured, then a deterministic mono 16 kHz PCM
   canonical audio artifact is derived with the same settings the local video
   path uses. Nothing is downloaded from a video stream and no full video is
   requested or kept.
3. **Durable, verifiable source cache.** The canonical artifact and a provenance
   manifest are published atomically under a source cache outside the repository,
   guarded by a single-writer lock. A repeated source command reuses a verified
   entry with zero network work; a completed job resumes from the job-owned copy
   with no ``yt-dlp`` call at all. A missing or corrupt cached artifact fails
   closed instead of being silently re-downloaded under an existing identity.

Honesty notes
-------------
* Audio bytes must be transferred from YouTube; this module never claims "no
  download". It does not fetch the video stream and does not keep a full video.
* The canonical artifact is the *extracted audio* timeline (zero-based). It is not
  claimed to be exactly aligned to the arbitrary YouTube video timeline.
* YouTube is a live third-party service: an offline fixture cannot prove that
  extraction works today. That requires a real network run and may break at any
  time.
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
from typing import Final
from urllib.parse import parse_qsl, urlsplit

from subtitle_flow import platform_compat
from subtitle_flow.config import (
    ExtractionSettings,
    MediaLimits,
    YouTubeLimits,
    YouTubeOriginSettings,
)
from subtitle_flow.media import AudioInfo, MediaError, sha256_file, validate_audio
from subtitle_flow.storage import utc_now
from subtitle_flow.video import (
    LOCAL_PROTOCOLS,
    VideoMediaError,
    _run_tool,
    ffmpeg_tool_version,
)
from subtitle_flow.yt_dlp_runner import (
    YtDlpError,
    YtDlpLimits,
    download_audio,
    ytdlp_version,
)

__all__ = [
    "YOUTUBE_EXTRACTION_VERSION",
    "YouTubeMediaError",
    "YouTubeMetadata",
    "YouTubeReference",
    "YouTubeSourcePurge",
    "YouTubeSourceResult",
    "acquire_youtube_source",
    "canonicalize_youtube_url",
    "parse_youtube_metadata",
    "purge_cached_youtube_source",
    "youtube_locator",
]

#: Version of the YouTube audio acquisition/derivation policy. A change
#: invalidates every previously cached source identity.
YOUTUBE_EXTRACTION_VERSION: Final[str] = "1"

#: Confined, deterministic job-relative locators for the canonical artifact.
_YOUTUBE_LOCATORS: Final[dict[str, str]] = {
    "wav": "artifacts/source_audio.wav",
    "flac": "artifacts/source_audio.flac",
}


def youtube_locator(settings: ExtractionSettings) -> str:
    """Return the confined job-relative locator for the canonical source audio."""

    return _YOUTUBE_LOCATORS[settings.container]

_VIDEO_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ALLOWED_HOSTS: Final[frozenset[str]] = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
)
#: Query parameters that carry no source identity and are dropped from the
#: canonical URL. Anything else is refused so a semantic parameter (a start time,
#: a playlist position) can never be silently ignored.
_BENIGN_QUERY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "si",
        "feature",
        "pp",
        "ab_channel",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
    }
)
#: Parameters that change what is played (a playlist, an index, a start time) or
#: select a live stream. They are refused explicitly rather than dropped by the
#: per-key loop in :func:`canonicalize_youtube_url`.
_METADATA_TITLE_LIMIT: Final[int] = 300

_COMPLETENESS_FRAME_SAMPLES: Final[int] = 2048
_COMPLETENESS_MIN_TOLERANCE_MS: Final[int] = 50

#: The provenance manifest is a small, flat JSON document. It is read with a
#: hard byte ceiling so a poisoned or corrupt manifest cannot force an unbounded
#: allocation, and a partial read is never accepted as valid JSON.
_CACHE_READ_CHUNK: Final[int] = 64 * 1024
_MANIFEST_READ_LIMIT_BYTES: Final[int] = 256 * 1024


class YouTubeMediaError(ValueError):
    """A YouTube source, its download or its derivation was refused.

    Carries a stable ``code`` plus a sanitized ``message``. The message never
    contains a signed URL, cookie, header or raw extractor blob.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class YouTubeReference:
    """One canonicalized, accepted YouTube video reference."""

    video_id: str
    canonical_url: str


@dataclass(frozen=True)
class YouTubeMetadata:
    """Allowlisted metadata for one downloaded video."""

    video_id: str
    title: str
    duration_ms: int
    extractor: str


@dataclass(frozen=True)
class YouTubeSourceResult:
    """A verified canonical audio artifact plus its typed origin provenance."""

    audio: AudioInfo
    origin: YouTubeOriginSettings
    from_cache: bool


@dataclass(frozen=True)
class YouTubeSourcePurge:
    """Outcome of removing one exact source-cache entry.

    ``removed`` is ``True`` only when the exact entry was proven to match the
    recorded origin and then deleted. ``reason`` is a short, sanitized Turkish
    explanation when nothing was removed (already absent, mismatched, a symlink
    or an unexpected tree), so a caller can surface an honest warning instead of
    claiming a cleanup that did not happen.
    """

    removed: bool
    reason: str | None = None


# --------------------------------------------------------------------------- #
# Offline URL acceptance
# --------------------------------------------------------------------------- #
def _reject(code: str, message: str) -> "YouTubeMediaError":
    return YouTubeMediaError(code, message)


def canonicalize_youtube_url(raw: str) -> YouTubeReference:
    """Validate and canonicalize one public YouTube video URL.

    The accepted forms are ``youtube.com/watch?v=ID`` (with the ``www.`` and
    ``m.`` hosts), ``youtu.be/ID``, ``youtube.com/shorts/ID`` and
    ``youtube.com/embed/ID``. Everything else is refused.
    """

    if not isinstance(raw, str) or raw.strip() == "":
        raise _reject("YOUTUBE_URL_INVALID", "a YouTube URL is required")
    if raw != raw.strip() or any(ch.isspace() for ch in raw):
        raise _reject(
            "YOUTUBE_URL_INVALID",
            "the URL must not contain leading/trailing or embedded whitespace",
        )
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in raw):
        raise _reject(
            "YOUTUBE_URL_INVALID", "the URL must not contain control characters"
        )

    try:
        split = urlsplit(raw)
    except ValueError as exc:
        raise _reject("YOUTUBE_URL_INVALID", "the URL could not be parsed") from exc

    if split.scheme.lower() != "https":
        raise _reject(
            "YOUTUBE_URL_UNSUPPORTED",
            "only https YouTube URLs are accepted",
        )
    if split.username is not None or split.password is not None:
        raise _reject(
            "YOUTUBE_URL_UNSUPPORTED", "URLs with user information are not accepted"
        )
    try:
        port = split.port
    except ValueError as exc:
        raise _reject("YOUTUBE_URL_INVALID", "the URL port is invalid") from exc
    if port is not None:
        raise _reject(
            "YOUTUBE_URL_UNSUPPORTED", "an explicit port is not accepted"
        )
    host = split.hostname
    if host is None or host.lower() not in _ALLOWED_HOSTS:
        raise _reject(
            "YOUTUBE_URL_UNSUPPORTED",
            "only youtube.com/www.youtube.com/m.youtube.com and youtu.be are "
            "accepted",
        )
    host = host.lower()
    if split.fragment != "":
        raise _reject(
            "YOUTUBE_URL_UNSUPPORTED",
            "URL fragments (for example a start time) are not supported",
        )

    pairs = parse_qsl(split.query, keep_blank_values=True)
    seen: dict[str, list[str]] = {}
    for key, value in pairs:
        seen.setdefault(key, []).append(value)
    for key in seen:
        if key in _BENIGN_QUERY_KEYS:
            continue
        if key == "v":
            if len(seen[key]) > 1:
                raise _reject(
                    "YOUTUBE_URL_AMBIGUOUS",
                    "the URL carries more than one video id",
                )
            continue
        raise _reject(
            "YOUTUBE_URL_UNSUPPORTED",
            f"the query parameter {key!r} is not supported for a single video",
        )

    query_id = seen.get("v", [None])[0]
    path_id = _path_video_id(split.path, host=host)
    if path_id is not None and query_id is not None:
        raise _reject(
            "YOUTUBE_URL_AMBIGUOUS",
            "the URL carries two different video id locations",
        )
    video_id = path_id if path_id is not None else query_id
    if not video_id or not _VIDEO_ID_RE.match(video_id):
        raise _reject(
            "YOUTUBE_URL_INVALID",
            "the video id must be exactly 11 characters of [A-Za-z0-9_-]",
        )
    return YouTubeReference(
        video_id=video_id,
        canonical_url=f"https://www.youtube.com/watch?v={video_id}",
    )


def _path_video_id(path: str, *, host: str) -> str | None:
    segments = [segment for segment in path.split("/") if segment != ""]
    if host == "youtu.be":
        if len(segments) != 1:
            raise _reject(
                "YOUTUBE_URL_UNSUPPORTED", "youtu.be URLs must be /<video-id> only"
            )
        return segments[0]
    if not segments:
        return None
    head = segments[0].lower()
    if head == "watch" and len(segments) == 1:
        return None
    if head in {"shorts", "embed"}:
        if len(segments) != 2:
            raise _reject(
                "YOUTUBE_URL_UNSUPPORTED",
                f"/{head}/ URLs must be /{head}/<video-id> only",
            )
        return segments[1]
    # Channels, playlists, live pages, @handles and any other path are refused.
    raise _reject(
        "YOUTUBE_URL_UNSUPPORTED",
        "only single-video watch/youtu.be/shorts/embed URLs are accepted",
    )


# --------------------------------------------------------------------------- #
# Metadata validation (allowlisted fields only)
# --------------------------------------------------------------------------- #
def _sanitize_title(raw: str) -> str:
    cleaned = "".join(ch for ch in raw if ch >= " " and ch != "\x7f")
    return cleaned.strip()[:_METADATA_TITLE_LIMIT]


def _parse_positive_number(raw: str, *, label: str) -> float:
    text = (raw or "").strip()
    if text in {"", "NA", "None", "null"}:
        raise _reject(
            "YOUTUBE_METADATA_INVALID", f"{label} is missing from the metadata"
        )
    try:
        value = float(text)
    except ValueError as exc:
        raise _reject(
            "YOUTUBE_METADATA_INVALID", f"{label} is not a number"
        ) from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise _reject(
            "YOUTUBE_METADATA_INVALID", f"{label} must be a finite number"
        )
    return value


def _parse_flag(raw: str) -> bool:
    return (raw or "").strip().lower() in {"true", "1", "yes"}


def parse_youtube_metadata(
    metadata: dict[str, str],
    reference: YouTubeReference,
    *,
    limits: YouTubeLimits,
) -> YouTubeMetadata:
    """Validate the allowlisted metadata against the canonical reference."""

    reported_id = (metadata.get("id") or "").strip()
    if reported_id != reference.video_id:
        raise _reject(
            "YOUTUBE_ID_MISMATCH",
            "the downloaded video id does not match the requested video id",
        )
    extractor = (metadata.get("extractor") or "").strip()
    if not extractor.lower().startswith("youtube"):
        raise _reject(
            "YOUTUBE_METADATA_INVALID", "the extractor is not the YouTube extractor"
        )
    availability = (metadata.get("availability") or "").strip().lower()
    if availability != "public":
        raise _reject(
            "YOUTUBE_NOT_PUBLIC",
            "only a public video is accepted; the recorded availability is not "
            "'public'",
        )
    live_status = (metadata.get("live_status") or "").strip().lower()
    if live_status not in {"not_live", "was_live"}:
        raise _reject(
            "YOUTUBE_LIVE_UNSUPPORTED",
            "live, upcoming or still-processing streams are not supported",
        )
    if _parse_flag(metadata.get("is_live", "")):
        raise _reject(
            "YOUTUBE_LIVE_UNSUPPORTED", "an active live stream is not supported"
        )

    seconds = _parse_positive_number(metadata.get("duration", ""), label="duration")
    if seconds <= 0:
        raise _reject(
            "YOUTUBE_METADATA_INVALID", "duration must be a positive number"
        )
    duration_ms = int(round(seconds * 1000))
    if duration_ms < limits.min_duration_ms:
        raise _reject(
            "YOUTUBE_SOURCE_TOO_SHORT",
            f"the source duration {duration_ms}ms is below the minimum "
            f"{limits.min_duration_ms}ms",
        )
    if duration_ms > limits.max_duration_ms:
        raise _reject(
            "YOUTUBE_SOURCE_TOO_LONG",
            f"the source duration {duration_ms}ms exceeds the limit "
            f"{limits.max_duration_ms}ms",
        )
    return YouTubeMetadata(
        video_id=reported_id,
        title=_sanitize_title(metadata.get("title", "")),
        duration_ms=duration_ms,
        extractor=extractor,
    )


# --------------------------------------------------------------------------- #
# Cache identity, locking and durable publication
# --------------------------------------------------------------------------- #
def _canonical_cache_root(cache_root: str | os.PathLike[str]) -> Path:
    """Return the canonical confinement base for the durable source cache.

    The user-supplied root is resolved exactly once so a legitimate parent alias
    (macOS ``/tmp`` -> ``/private/tmp``) is handled. Every nested, untrusted
    component added later (an entry directory, an entry file or the lock file)
    is checked separately and must resolve to a real child of this base.
    """

    root = Path(cache_root).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _reject(
            "YOUTUBE_CACHE_UNSAFE", "the source cache root could not be created"
        ) from exc
    return Path(os.path.realpath(root))


def _reject_if_symlink(path: Path, *, label: str) -> None:
    if os.path.islink(path):
        raise _reject(
            "YOUTUBE_CACHE_UNSAFE",
            f"the source cache {label} is a symlink; refusing to read, write or "
            "erase through a link outside the cache",
        )


def _confined_cache_entry(root: Path, name: str) -> Path:
    """Return ``root/name`` only when it is a real child of the canonical root.

    ``name`` is a single untrusted component. A symlink (even one pointing at a
    same-content directory outside the cache) or a path that escapes the
    canonical root is refused before any read, hash, write, reuse or cleanup.
    """

    candidate = root / name
    _reject_if_symlink(candidate, label=f"entry {name!r}")
    resolved = Path(os.path.realpath(candidate))
    if resolved != candidate or resolved.parent != root:
        raise _reject(
            "YOUTUBE_CACHE_UNSAFE",
            f"the source cache entry {name!r} does not resolve inside the "
            "canonical cache root",
        )
    return candidate


def _bounded_read_manifest_bytes(path: Path) -> bytes:
    """Read the manifest with a hard byte ceiling and no link following."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT", "the source manifest is unreadable"
        ) from exc
    try:
        buffer = bytearray()
        while len(buffer) <= _MANIFEST_READ_LIMIT_BYTES:
            chunk = os.read(
                descriptor,
                min(_CACHE_READ_CHUNK, _MANIFEST_READ_LIMIT_BYTES + 1 - len(buffer)),
            )
            if not chunk:
                break
            buffer.extend(chunk)
    finally:
        os.close(descriptor)
    if len(buffer) > _MANIFEST_READ_LIMIT_BYTES:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the source manifest exceeds the bounded size limit",
        )
    return bytes(buffer)


def _cache_identity_from_parts(
    video_id: str,
    settings: ExtractionSettings,
    *,
    extraction_version: str,
    ytdlp_version_value: str,
    ffmpeg_version_value: str,
) -> str:
    """Hash the exact cache identity from its recorded parts.

    This is the single implementation used both when a source is written and
    when a completed job's recorded origin is used to locate its exact cache
    entry for cleanup. The recorded origin carries every input (settings,
    extraction version and tool versions), so a purge never has to execute an
    external tool or guess.
    """

    payload = {
        "video_id": video_id,
        "settings": settings.model_dump(mode="json"),
        "extraction_version": extraction_version,
        "ytdlp_version": ytdlp_version_value,
        "ffmpeg_version": ffmpeg_version_value,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_identity(
    video_id: str,
    settings: ExtractionSettings,
    *,
    ytdlp_version_value: str,
    ffmpeg_version_value: str,
) -> str:
    return _cache_identity_from_parts(
        video_id,
        settings,
        extraction_version=YOUTUBE_EXTRACTION_VERSION,
        ytdlp_version_value=ytdlp_version_value,
        ffmpeg_version_value=ffmpeg_version_value,
    )


def _cache_entry_name(origin: YouTubeOriginSettings) -> str:
    """Return the deterministic entry directory name recorded for ``origin``."""

    identity = _cache_identity_from_parts(
        origin.video_id,
        origin.settings,
        extraction_version=origin.extraction_version,
        ytdlp_version_value=origin.ytdlp_version,
        ffmpeg_version_value=origin.ffmpeg_version,
    )
    return f"{origin.video_id}-{identity[:16]}"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=target.parent)
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
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


class _CacheLock:
    """A best-effort single-writer lock for one source cache entry."""

    def __init__(self, path: Path, *, timeout: float = 60.0) -> None:
        self._path = path
        self._timeout = timeout
        self._handle = None

    def __enter__(self) -> "_CacheLock":
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _reject(
                "YOUTUBE_CACHE_UNSAFE", "the source cache lock directory is not usable"
            ) from exc
        _reject_if_symlink(self._path, label="lock file")
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except OSError as exc:
            raise _reject(
                "YOUTUBE_CACHE_UNSAFE", "the source cache lock file is not usable"
            ) from exc
        self._handle = os.fdopen(descriptor, "a+")
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                platform_compat.lock_fd_exclusive_nonblocking(
                    self._handle.fileno()
                )
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    self._handle.close()
                    self._handle = None
                    raise _reject(
                        "YOUTUBE_CACHE_LOCKED",
                        "another process is writing this source cache entry",
                    )
                time.sleep(0.1)

    def __exit__(self, *_exc: object) -> None:
        if self._handle is not None:
            try:
                platform_compat.unlock_fd(self._handle.fileno())
            except OSError:
                pass
            self._handle.close()
            self._handle = None


def _read_manifest(path: Path) -> YouTubeOriginSettings:
    _reject_if_symlink(path, label="manifest")
    raw = _bounded_read_manifest_bytes(path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT", "the source manifest is not valid UTF-8"
        ) from exc
    try:
        return YouTubeOriginSettings.model_validate_json(text)
    except ValueError as exc:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT", "the source manifest is invalid"
        ) from exc


def _verify_manifest_identity(
    origin: YouTubeOriginSettings,
    reference: YouTubeReference,
    settings: ExtractionSettings,
    *,
    ytdlp_version_value: str,
    ffmpeg_version_value: str,
) -> None:
    if origin.video_id != reference.video_id:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached manifest belongs to a different video id",
        )
    if origin.settings != settings:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached manifest was produced with different extraction settings",
        )
    if origin.ytdlp_version != ytdlp_version_value:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached manifest was produced by a different yt-dlp version",
        )
    if origin.ffmpeg_version != ffmpeg_version_value:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached manifest was produced by a different ffmpeg version",
        )
    if origin.extraction_version != YOUTUBE_EXTRACTION_VERSION:
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached manifest uses a different extraction version",
        )


def _unexpected_cache_tree(entry: Path, *, settings: ExtractionSettings) -> str | None:
    """Return a reason when ``entry`` is not the exact expected, link-free tree.

    A cache entry may only contain its provenance manifest and the single
    canonical artifact. Any symlink, subdirectory or unexpected file makes the
    entry ambiguous to erase safely, so it is refused (never followed, never
    deleted) and reported instead.
    """

    expected_files = {"source.manifest.json", f"canonical.{settings.container}"}
    try:
        entries = list(os.scandir(entry))
    except OSError:
        return "önbellek girdisi okunamadı"
    for child in entries:
        try:
            if child.is_symlink():
                return f"önbellek girdisinde sembolik bağlantı: {child.name}"
            if child.is_dir(follow_symlinks=False):
                return f"önbellek girdisinde beklenmeyen alt dizin: {child.name}"
        except OSError:
            return f"önbellek girdisi incelenemedi: {child.name}"
        if child.name not in expected_files:
            return f"önbellek girdisinde beklenmeyen dosya: {child.name}"
    if not (entry / "source.manifest.json").is_file():
        return "önbellek manifesti eksik"
    return None


def purge_cached_youtube_source(
    origin: YouTubeOriginSettings,
    *,
    cache_root: str | os.PathLike[str] | None = None,
) -> YouTubeSourcePurge:
    """Remove the exact, verified source-cache entry recorded for ``origin``.

    The entry is located from the origin's own recorded identity (video id,
    extraction settings, extraction version and both tool versions), so no
    external tool is executed and no glob over an untrusted URL/title is used.
    The stored manifest is re-read and must match the origin byte-for-byte on
    its identity payload before anything is removed, and the confined path
    helpers reject a symlink or an escaped path. Nothing is ever followed or
    removed outside the canonical cache root.

    Only the entry directory is removed. The sibling 0-byte ``<entry>.lock``
    marker is intentionally left in place: it is the stable inode that concurrent
    writers lock on inside ``_CacheLock``, so unlinking it would reintroduce a
    lock race. ``removed=True`` therefore means "the entry is gone", not "the
    cache directory is byte-for-byte empty".
    """

    if cache_root is None:
        from subtitle_flow.config import default_youtube_cache_root

        cache_root = default_youtube_cache_root()
    try:
        root = _canonical_cache_root(cache_root)
    except YouTubeMediaError as exc:
        return YouTubeSourcePurge(False, f"kaynak önbellek kökü kullanılamadı ({exc.code})")

    entry_name = _cache_entry_name(origin)
    try:
        entry = _confined_cache_entry(root, entry_name)
        lock_path = _confined_cache_entry(root, f"{entry_name}.lock")
    except YouTubeMediaError as exc:
        return YouTubeSourcePurge(False, f"önbellek yolu güvenli değil ({exc.code})")

    try:
        with _CacheLock(lock_path):
            if not entry.exists():
                return YouTubeSourcePurge(False, "önbellek girdisi zaten yok")
            _reject_if_symlink(entry, label="entry")
            if not entry.is_dir():
                return YouTubeSourcePurge(False, "önbellek girdisi gerçek bir dizin değil")
            manifest_path = entry / "source.manifest.json"
            _reject_if_symlink(manifest_path, label="manifest")
            if not manifest_path.is_file():
                return YouTubeSourcePurge(False, "önbellek manifesti eksik")
            try:
                manifest = _read_manifest(manifest_path)
            except YouTubeMediaError as exc:
                return YouTubeSourcePurge(False, f"önbellek manifesti okunamadı ({exc.code})")
            try:
                _verify_manifest_identity(
                    manifest,
                    YouTubeReference(origin.video_id, origin.canonical_url),
                    origin.settings,
                    ytdlp_version_value=origin.ytdlp_version,
                    ffmpeg_version_value=origin.ffmpeg_version,
                )
            except YouTubeMediaError as exc:
                return YouTubeSourcePurge(False, f"önbellek kimliği eşleşmiyor ({exc.code})")
            if manifest.identity_payload() != origin.identity_payload():
                return YouTubeSourcePurge(False, "önbellek manifesti kaydedilen kaynakla eşleşmiyor")
            unexpected = _unexpected_cache_tree(entry, settings=origin.settings)
            if unexpected is not None:
                return YouTubeSourcePurge(False, unexpected)
            try:
                shutil.rmtree(entry)
            except OSError:
                return YouTubeSourcePurge(False, "önbellek girdisi silinemedi")
            if entry.exists():
                return YouTubeSourcePurge(False, "önbellek girdisi tamamen silinemedi")
            return YouTubeSourcePurge(True)
    except YouTubeMediaError as exc:
        return YouTubeSourcePurge(False, f"önbellek kilidi alınamadı ({exc.code})")


# --------------------------------------------------------------------------- #
# Intermediate validation and canonical derivation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Intermediate:
    path: Path
    codec_name: str
    ext: str
    duration_ms: int
    size_bytes: int
    sha256: str


def _probe_audio_only(
    path: Path, *, ffprobe_bin: str, timeout: float
) -> tuple[str, int]:
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
            missing_code="YOUTUBE_FFPROBE_MISSING",
            missing_label="ffprobe",
            timeout_code="YOUTUBE_FFPROBE_TIMEOUT",
            failed_code="YOUTUBE_FFPROBE_FAILED",
            overflow_code="YOUTUBE_OUTPUT_OVERFLOW",
        )
    except VideoMediaError as exc:
        raise _reject(exc.code, exc.message) from exc
    if completed.returncode != 0:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID",
            "ffprobe could not read the downloaded audio; the container is "
            "invalid, unsupported or uses a non-local protocol",
        )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID", "ffprobe returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise _reject(
            "YOUTUBE_SOURCE_INVALID", "ffprobe returned an unexpected payload"
        )
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise _reject(
            "YOUTUBE_SOURCE_INVALID", "ffprobe reported no stream list"
        )
    audio_streams = [
        entry
        for entry in streams
        if isinstance(entry, dict) and entry.get("codec_type") == "audio"
    ]
    other_streams = [
        entry
        for entry in streams
        if isinstance(entry, dict) and entry.get("codec_type") != "audio"
    ]
    if not audio_streams:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID", "the downloaded file has no audio stream"
        )
    if len(audio_streams) > 1:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID",
            "the downloaded file has more than one audio stream; exactly one is "
            "required",
        )
    if other_streams:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID",
            "the downloaded file contains a non-audio stream; only audio-only "
            "is accepted",
        )
    stream = audio_streams[0]
    codec = stream.get("codec_name")
    if not isinstance(codec, str) or not codec:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID", "the audio stream has no codec name"
        )
    format_map = payload.get("format")
    format_map = format_map if isinstance(format_map, dict) else {}
    raw_duration = format_map.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = stream.get("duration")
    try:
        seconds = float(str(raw_duration))
    except (TypeError, ValueError) as exc:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID", "the downloaded audio duration is unknown"
        ) from exc
    if seconds != seconds or seconds in (float("inf"), float("-inf")) or seconds <= 0:
        raise _reject(
            "YOUTUBE_SOURCE_INVALID",
            "the downloaded audio duration must be a finite positive number",
        )
    return codec, int(round(seconds * 1000))


def _decode_audio(path: Path, *, ffmpeg_bin: str, timeout: float) -> None:
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
        "-vn",
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
            missing_code="YOUTUBE_FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="YOUTUBE_FFMPEG_TIMEOUT",
            failed_code="YOUTUBE_FFMPEG_FAILED",
            overflow_code="YOUTUBE_OUTPUT_OVERFLOW",
        )
    except VideoMediaError as exc:
        raise _reject(exc.code, exc.message) from exc
    if completed.returncode != 0:
        raise _reject(
            "YOUTUBE_SOURCE_CORRUPT",
            "the downloaded audio did not decode cleanly; it is corrupt or "
            "truncated",
        )


def _validate_intermediate(
    path: Path,
    *,
    limits: YouTubeLimits,
    ffprobe_bin: str,
    ffmpeg_bin: str,
) -> _Intermediate:
    try:
        info = os.stat(path)
    except OSError as exc:
        raise _reject(
            "YOUTUBE_NO_OUTPUT", "the downloaded audio file is missing"
        ) from exc
    if info.st_size == 0:
        raise _reject("YOUTUBE_NO_OUTPUT", "the downloaded audio file is empty")
    if info.st_size > limits.max_source_bytes:
        raise _reject(
            "YOUTUBE_SOURCE_TOO_LARGE",
            f"the downloaded audio is {info.st_size} bytes, exceeding the limit "
            f"{limits.max_source_bytes}",
        )
    codec, duration_ms = _probe_audio_only(
        path,
        ffprobe_bin=ffprobe_bin,
        timeout=limits.probe_timeout_seconds,
    )
    _decode_audio(
        path, ffmpeg_bin=ffmpeg_bin, timeout=limits.decode_timeout_seconds
    )
    digest, size = sha256_file(path)
    ext = path.suffix.lstrip(".").lower() or "bin"
    return _Intermediate(
        path=path,
        codec_name=codec,
        ext=ext,
        duration_ms=duration_ms,
        size_bytes=size,
        sha256=digest,
    )


def _completeness_tolerance_ms(sample_rate: int) -> int:
    frame_ms = -(-_COMPLETENESS_FRAME_SAMPLES * 1000 // sample_rate)
    return max(_COMPLETENESS_MIN_TOLERANCE_MS, frame_ms + 1)


def _derive_canonical(
    source: Path,
    settings: ExtractionSettings,
    out_path: Path,
    *,
    ffmpeg_bin: str,
    limits: YouTubeLimits,
) -> None:
    if settings.container == "wav":
        codec, sample_fmt, format_name = "pcm_s16le", "s16", "wav"
    elif settings.container == "flac":
        codec, sample_fmt, format_name = "flac", "s16", "flac"
    else:  # pragma: no cover - Literal-constrained by ExtractionSettings
        raise _reject(
            "YOUTUBE_SETTINGS_INVALID",
            f"unsupported canonical container {settings.container!r}",
        )
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
        str(source),
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
    try:
        completed = _run_tool(
            args,
            timeout=limits.extraction_timeout_seconds,
            missing_code="YOUTUBE_FFMPEG_MISSING",
            missing_label="ffmpeg",
            timeout_code="YOUTUBE_FFMPEG_TIMEOUT",
            failed_code="YOUTUBE_FFMPEG_FAILED",
            overflow_code="YOUTUBE_OUTPUT_OVERFLOW",
            size_guard=(out_path, limits.max_output_bytes),
            size_guard_code="YOUTUBE_OUTPUT_TOO_LARGE",
        )
    except VideoMediaError as exc:
        raise _reject(exc.code, exc.message) from exc
    if completed.returncode != 0:
        raise _reject(
            "YOUTUBE_EXTRACT_FAILED",
            "ffmpeg could not derive a complete canonical audio artifact",
        )
    try:
        if out_path.stat().st_size == 0:
            raise _reject(
                "YOUTUBE_EXTRACT_FAILED", "the derived canonical audio is empty"
            )
    except OSError as exc:
        raise _reject(
            "YOUTUBE_EXTRACT_FAILED", "ffmpeg did not produce a canonical artifact"
        ) from exc


def _validate_canonical(
    path: Path, *, media_limits: MediaLimits, ffprobe_bin: str, ffmpeg_bin: str
) -> AudioInfo:
    try:
        return validate_audio(
            path,
            limits=media_limits,
            ffprobe_bin=ffprobe_bin,
            ffmpeg_bin=ffmpeg_bin,
        )
    except MediaError as exc:
        raise _reject(
            "YOUTUBE_OUTPUT_INVALID",
            f"the derived canonical audio was rejected ({exc.code})",
        ) from exc


def _rebind_canonical(audio: AudioInfo, path: Path) -> AudioInfo:
    return AudioInfo.model_validate(
        {**audio.model_dump(), "path": str(path), "original_filename": path.name}
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def acquire_youtube_source(
    reference: YouTubeReference,
    *,
    ytdlp_bin: str,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    limits: YouTubeLimits | None = None,
    settings: ExtractionSettings | None = None,
    cache_root: str | os.PathLike[str] | None = None,
    media_limits: MediaLimits | None = None,
) -> YouTubeSourceResult:
    """Acquire, verify and cache the canonical audio for one YouTube video.

    The identity binds the canonical video id, the deterministic extraction
    settings, the extraction version and the *installed* external tool versions.
    A verified cache entry is reused with no network call. A missing manifest
    (an interrupted first attempt) is rebuilt from scratch under the entry lock;
    a present-but-invalid entry fails closed and is never silently redownloaded.
    """

    limits = limits if limits is not None else YouTubeLimits()
    settings = settings if settings is not None else ExtractionSettings()
    media_limits = media_limits if media_limits is not None else MediaLimits()
    if cache_root is None:
        from subtitle_flow.config import default_youtube_cache_root

        cache_root = default_youtube_cache_root()
    cache_root = _canonical_cache_root(cache_root)

    ytdlp_version_value = ytdlp_version(
        ytdlp_bin, timeout=limits.probe_timeout_seconds
    )
    ffmpeg_version_value = ffmpeg_tool_version(
        ffmpeg_bin, timeout=limits.probe_timeout_seconds
    )
    identity = _cache_identity(
        reference.video_id,
        settings,
        ytdlp_version_value=ytdlp_version_value,
        ffmpeg_version_value=ffmpeg_version_value,
    )
    entry_name = f"{reference.video_id}-{identity[:16]}"
    entry = _confined_cache_entry(cache_root, entry_name)
    lock_path = _confined_cache_entry(cache_root, f"{entry_name}.lock")
    audio_name = f"canonical.{settings.container}"
    audio_path = entry / audio_name
    manifest_path = entry / "source.manifest.json"

    with _CacheLock(lock_path):
        # An entry, its canonical audio, its manifest and the lock path are all
        # untrusted, user-writable locations. Refusing a symlink here is what
        # keeps a same-content link from being read as a valid cache and keeps a
        # poisoned entry from being written through into an external directory.
        _reject_if_symlink(entry, label="entry")
        _reject_if_symlink(audio_path, label="canonical audio")
        _reject_if_symlink(manifest_path, label="manifest")
        if manifest_path.is_file():
            origin = _read_manifest(manifest_path)
            _verify_manifest_identity(
                origin,
                reference,
                settings,
                ytdlp_version_value=ytdlp_version_value,
                ffmpeg_version_value=ffmpeg_version_value,
            )
            audio = _reuse_cached_audio(
                audio_path,
                origin,
                media_limits=media_limits,
                ffprobe_bin=ffprobe_bin,
                ffmpeg_bin=ffmpeg_bin,
            )
            return YouTubeSourceResult(audio=audio, origin=origin, from_cache=True)

        # A previous attempt crashed before publishing its manifest: the entry is
        # a partial and is cleared under the lock, then rebuilt. A present valid
        # manifest is never reached here and is therefore never overwritten, and
        # the entry is proven to be a real directory (not a link) above.
        if entry.exists():
            shutil.rmtree(entry, ignore_errors=True)
        entry.mkdir(parents=True, exist_ok=True)
        _reject_if_symlink(entry, label="entry")

        work = Path(tempfile.mkdtemp(prefix=".work-", dir=str(cache_root)))
        try:
            result = _download_and_derive(
                reference,
                work_dir=work,
                ytdlp_bin=ytdlp_bin,
                ffprobe_bin=ffprobe_bin,
                ffmpeg_bin=ffmpeg_bin,
                limits=limits,
                settings=settings,
                media_limits=media_limits,
                ytdlp_version_value=ytdlp_version_value,
                ffmpeg_version_value=ffmpeg_version_value,
            )
            canonical = result.audio
            # Publish the canonical artifact first, then the manifest last, so a
            # crash in between is detectable as a partial rather than presented
            # as a verified entry.
            platform_compat.replace_file_atomically(canonical.path, audio_path)
            _atomic_write_bytes(
                manifest_path, result.origin.model_dump_json(indent=2).encode("utf-8")
            )
            rebound = _rebind_canonical(canonical, audio_path)
            return YouTubeSourceResult(
                audio=rebound, origin=result.origin, from_cache=False
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _reuse_cached_audio(
    audio_path: Path,
    origin: YouTubeOriginSettings,
    *,
    media_limits: MediaLimits,
    ffprobe_bin: str,
    ffmpeg_bin: str,
) -> AudioInfo:
    _reject_if_symlink(audio_path, label="canonical audio")
    if not audio_path.is_file():
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached canonical audio is missing while its manifest exists; "
            "refusing to silently re-download under the same identity",
        )
    audio = _validate_canonical(
        audio_path,
        media_limits=media_limits,
        ffprobe_bin=ffprobe_bin,
        ffmpeg_bin=ffmpeg_bin,
    )
    if (
        audio.audio_sha256 != origin.canonical_audio_sha256
        or audio.size_bytes != origin.canonical_audio_size_bytes
    ):
        raise _reject(
            "YOUTUBE_CACHE_CORRUPT",
            "the cached canonical audio does not match its recorded hash; "
            "refusing to silently re-download or relabel it",
        )
    return audio


def _download_and_derive(
    reference: YouTubeReference,
    *,
    work_dir: Path,
    ytdlp_bin: str,
    ffprobe_bin: str,
    ffmpeg_bin: str,
    limits: YouTubeLimits,
    settings: ExtractionSettings,
    media_limits: MediaLimits,
    ytdlp_version_value: str,
    ffmpeg_version_value: str,
) -> YouTubeSourceResult:
    ytdlp_limits = YtDlpLimits(
        max_directory_bytes=limits.max_source_bytes,
        download_timeout_seconds=limits.download_timeout_seconds,
        max_stdout_bytes=limits.max_stdout_bytes,
        max_stderr_bytes=limits.max_stderr_bytes,
        retries=limits.max_retries,
        fragment_retries=limits.max_fragment_retries,
        socket_timeout_seconds=limits.socket_timeout_seconds,
        max_duration_seconds=max(1, limits.max_duration_ms // 1000),
    )
    try:
        download = download_audio(
            ytdlp_bin, reference.canonical_url, work_dir=work_dir, limits=ytdlp_limits
        )
    except YtDlpError as exc:
        raise _reject(exc.code, exc.message) from exc
    metadata = parse_youtube_metadata(download.metadata, reference, limits=limits)
    intermediate = _validate_intermediate(
        download.output_path,
        limits=limits,
        ffprobe_bin=ffprobe_bin,
        ffmpeg_bin=ffmpeg_bin,
    )
    canonical_tmp = work_dir / f"canonical.{settings.container}"
    _derive_canonical(
        intermediate.path,
        settings,
        canonical_tmp,
        ffmpeg_bin=ffmpeg_bin,
        limits=limits,
    )
    canonical = _validate_canonical(
        canonical_tmp,
        media_limits=media_limits,
        ffprobe_bin=ffprobe_bin,
        ffmpeg_bin=ffmpeg_bin,
    )
    tolerance_ms = _completeness_tolerance_ms(canonical.sample_rate)
    if abs(canonical.duration_ms - metadata.duration_ms) > tolerance_ms:
        raise _reject(
            "YOUTUBE_OUTPUT_INCOMPLETE",
            "the derived canonical audio duration does not match the recorded "
            "source duration; refusing a truncated or clipped derivation",
        )
    origin = YouTubeOriginSettings(
        video_id=reference.video_id,
        canonical_url=reference.canonical_url,
        title=metadata.title,
        source_duration_ms=metadata.duration_ms,
        retrieved_at_utc=utc_now(),
        ytdlp_version=ytdlp_version_value,
        ffmpeg_version=ffmpeg_version_value,
        extraction_version=YOUTUBE_EXTRACTION_VERSION,
        intermediate_codec=intermediate.codec_name,
        intermediate_ext=intermediate.ext,
        intermediate_sha256=intermediate.sha256,
        intermediate_size_bytes=intermediate.size_bytes,
        canonical_audio_sha256=canonical.audio_sha256,
        canonical_audio_size_bytes=canonical.size_bytes,
        settings=settings,
    )
    return YouTubeSourceResult(audio=canonical, origin=origin, from_cache=False)
