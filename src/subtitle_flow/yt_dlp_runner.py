"""Bounded, hardened subprocess runner for the external ``yt-dlp`` binary.

The runner is deliberately narrow: it knows how to ask the *installed* external
``yt-dlp`` for its version and how to download **one audio-only stream** into a
private working directory. It never imports ``yt_dlp`` as a Python module, never
uses a shell and never reads a user configuration file.

Hardening guarantees
--------------------
* **Explicit safe arguments.** Every invocation is an argument list (no shell)
  with an explicit, audited set of flags: user config and plugin directories are
  disabled, remote components are disallowed, cookies/netrc/browser identity and
  mark-watched are off, playlists are refused, TLS certificate verification stays
  on (``--no-check-certificates`` is never passed) and ``file://`` stays
  disabled.
* **Audio-only.** ``-f bestaudio`` (= ``best*[vcodec=none]``) never selects a
  muxed video stream. ``bestaudio*`` is never used.
* **Minimal environment.** The child receives a freshly built environment with a
  private ``HOME``/``TMPDIR`` and no proxy or credential variables, so an
  unrelated shell setting cannot change egress.
* **Bounded while it runs.** Wall-clock deadline, per-file/directory byte growth
  monitor, retry counts and stdout/stderr byte ceilings are enforced *during* the
  run. On a timeout, an output ceiling overflow or a directory-size guard trip,
  the whole owned process group is killed and reaped, so no orphan download keeps
  writing.
* **No raw JSON.** Only an explicit allowlist of metadata fields is requested and
  each is written by ``yt-dlp`` itself to a separate small file; the signed
  ``formats``/``url``/``http_headers`` payload is never requested, parsed or
  persisted.

Honesty limits: byte growth is sampled on a poll interval, so a single burst can
overshoot the directory budget before the next sample kills the child. The
overshoot is bounded by poll interval times write rate and the partial output is
discarded, but the runner does not claim an exact instantaneous cap.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from subtitle_flow import platform_compat

__all__ = [
    "DEFAULT_YTDLP_RETRIES",
    "DEFAULT_YTDLP_FRAGMENT_RETRIES",
    "DEFAULT_YTDLP_SOCKET_TIMEOUT_SECONDS",
    "FORMAT_AUDIO",
    "FORMAT_VIDEO",
    "METADATA_FIELDS",
    "YtDlpDownload",
    "YtDlpError",
    "YtDlpLimits",
    "download_audio",
    "download_video",
    "ytdlp_version",
]

#: Audio-only stream selector (documented, unchanged). ``bestaudio`` equals
#: ``best*[vcodec=none]`` and never selects a muxed video stream.
FORMAT_AUDIO: Final[str] = "bestaudio"

#: Full-video stream selector for the opt-in burned-in subtitle feature. It
#: prefers a separate video+audio pair merged to MP4 and falls back to the best
#: single muxed stream. This is intentionally *not* the accepted audio selector
#: and never changes the audio cache identity.
FORMAT_VIDEO: Final[str] = "bestvideo*+bestaudio/best"

#: Metadata field allowlist. Each is requested as its own ``--print-to-file``
#: target so the untrusted title can never break a single-line protocol and no
#: raw extractor JSON (which carries signed URLs and headers) is ever produced.
METADATA_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "title",
    "duration",
    "extractor",
    "availability",
    "live_status",
    "is_live",
    "was_live",
)

DEFAULT_YTDLP_RETRIES: Final[int] = 3
DEFAULT_YTDLP_FRAGMENT_RETRIES: Final[int] = 3
DEFAULT_YTDLP_SOCKET_TIMEOUT_SECONDS: Final[float] = 20.0

_READ_CHUNK: Final[int] = 64 * 1024
_POLL_INTERVAL_SECONDS: Final[float] = 0.05
_JOIN_TIMEOUT_SECONDS: Final[float] = 10.0
_REAP_TIMEOUT_SECONDS: Final[float] = 10.0
_VERSION_CEILING_BYTES: Final[int] = 64 * 1024
#: Per-field metadata and output-path reporting are small, allowlisted
#: single-value files. They are read with a hard byte ceiling so a hostile or
#: buggy child cannot force an unbounded allocation before the value is parsed.
_METADATA_READ_LIMIT_BYTES: Final[int] = 8192
_FILEPATH_READ_LIMIT_BYTES: Final[int] = 4096

#: Marker substrings used to classify a failed child into a *generic* typed
#: error. The child text is never copied into the message, so no signed URL,
#: cookie, header or token can survive regardless of newline style.
_AUTH_MARKERS: Final[tuple[str, ...]] = (
    "sign in",
    "login required",
    "private video",
    "members-only",
    "confirm your age",
    "authentication",
    "unauthorized",
    "http error 401",
    "http error 403",
)
_TOOL_MARKERS: Final[tuple[str, ...]] = (
    "ffmpeg",
    "javascript runtime",
    "js runtime",
    "could not find a javascript",
)


class YtDlpError(RuntimeError):
    """A bounded ``yt-dlp`` invocation failed or was refused.

    Carries a stable ``code`` plus a human-readable, **sanitized** message. The
    message never contains a signed URL, cookie, header or raw extractor blob.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class YtDlpLimits:
    """Hard bounds for one download attempt."""

    max_directory_bytes: int
    download_timeout_seconds: float
    max_stdout_bytes: int
    max_stderr_bytes: int
    retries: int = DEFAULT_YTDLP_RETRIES
    fragment_retries: int = DEFAULT_YTDLP_FRAGMENT_RETRIES
    socket_timeout_seconds: float = DEFAULT_YTDLP_SOCKET_TIMEOUT_SECONDS
    max_duration_seconds: int | None = None


@dataclass(frozen=True)
class YtDlpDownload:
    """One successful bounded download of a single audio-only output.

    The child's raw stdout/stderr are deliberately **not** carried: they may
    contain a signed URL, cookie or header and are never needed downstream. Only
    the allowlisted metadata and the confirmed output path are returned.
    """

    output_path: Path
    metadata: dict[str, str]


def _classify_failure(stderr: str, *, route: str = "audio") -> tuple[str, str]:
    """Map a failed child to a generic typed error without copying its text.

    Only fixed marker substrings are inspected; the returned message is a
    constant, so a signed URL, cookie, header or token in the child stderr can
    never reach an error, export or status record. The wording is route-aware so
    a failed *video* acquisition never claims an audio-only download.
    """

    lowered = (stderr or "").lower()
    if any(marker in lowered for marker in _AUTH_MARKERS):
        return (
            "YTDLP_AUTH_REQUIRED",
            "yt-dlp reported that this video requires authentication or sign-in; "
            "only a public, completed video is supported",
        )
    if any(marker in lowered for marker in _TOOL_MARKERS):
        return (
            "YTDLP_TOOL_MISSING",
            "yt-dlp reported a missing required external tool (for example "
            "ffmpeg or a JavaScript runtime)",
        )
    label = "video" if route == "video" else "audio-only"
    return (
        "YTDLP_DOWNLOAD_FAILED",
        f"yt-dlp could not download the requested {label} stream",
    )



def _build_env(work_dir: Path) -> dict[str, str]:
    """Return a minimal child environment with a private HOME/TMPDIR.

    The child never inherits the parent environment wholesale. Only the few
    variables an external binary needs to start, resolve sibling tools and run
    its TLS/socket stack are forwarded; proxy, credential and language-affecting
    variables are deliberately dropped so an unrelated shell cannot redirect
    egress or change message text.

    On Windows the private ``HOME``/``TMPDIR`` are joined by the profile and temp
    locations Windows tools fall back to (``TEMP``/``TMP``/``USERPROFILE``/
    ``APPDATA``/``LOCALAPPDATA``), each pointing inside the private working
    directory, plus the system variables (``SystemRoot``/``windir``/``PATHEXT``)
    without which child creation and its TLS stack can fail.
    """

    private = str(work_dir)
    env = {
        "HOME": private,
        "TMPDIR": private,
        "NO_COLOR": "1",
        "LANG": "C",
        "LC_ALL": "C",
        "YTDLP_NO_PLUGINS": "1",
    }
    if platform_compat.is_windows():
        env.update(_windows_env(work_dir))
    else:
        env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    return env


def _windows_env(work_dir: Path) -> dict[str, str]:
    """Windows child variables, keeping every writable location private."""

    system_root = (
        os.environ.get("SystemRoot")
        or os.environ.get("SYSTEMROOT")
        or os.environ.get("windir")
        or "C:\\Windows"
    )
    private = str(work_dir)
    return {
        "PATH": os.environ.get("PATH") or f"{system_root}\\system32;{system_root}",
        "PATHEXT": os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD"),
        "SystemRoot": system_root,
        "windir": system_root,
        "TEMP": private,
        "TMP": private,
        "USERPROFILE": private,
        "APPDATA": private,
        "LOCALAPPDATA": private,
    }


def _drain_stream(
    stream: BinaryIO,
    buffer: bytearray,
    ceiling: int,
    overflow: threading.Event,
) -> None:
    try:
        descriptor = stream.fileno()
    except (OSError, ValueError):
        return
    retained = 0
    while True:
        try:
            block = os.read(descriptor, _READ_CHUNK)
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


def _directory_bytes(root: Path) -> int:
    total = 0
    try:
        entries = list(root.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                total += _directory_bytes(entry)
            else:
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def _kill_group(
    process: "subprocess.Popen[bytes]", pgid: int | None
) -> None:
    """Kill the owned child group/tree; never lets cleanup mask a typed error."""

    platform_compat.kill_owned_process_group(process, pgid)


def _reap(process: "subprocess.Popen[bytes]") -> None:
    try:
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass


def _join_drainers(threads: list[threading.Thread], timeout: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout)
    for thread in threads:
        remaining = deadline - time.monotonic()
        thread.join(timeout=remaining if remaining > 0 else 0.0)
    return all(not thread.is_alive() for thread in threads)


def _run_bounded(
    args: list[str],
    *,
    env: dict[str, str],
    timeout: float,
    stdout_ceiling: int,
    stderr_ceiling: int,
    directory_guard: tuple[Path, int] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one child in its own process group with bounded output and deadline."""

    try:
        process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            **platform_compat.owned_process_kwargs(),
        )
    except FileNotFoundError as exc:
        raise YtDlpError(
            "YTDLP_MISSING",
            "yt-dlp was not found; install it and/or set YOUTUBE_YTDLP_BIN",
        ) from exc
    except OSError as exc:
        raise YtDlpError("YTDLP_FAILED", "yt-dlp could not start") from exc

    pgid = process.pid
    assert process.stdout is not None and process.stderr is not None
    stdout_buffer = bytearray()
    stderr_buffer = bytearray()
    overflow = threading.Event()
    threads = [
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_buffer, stdout_ceiling, overflow),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_buffer, stderr_ceiling, overflow),
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
        if directory_guard is not None:
            guard_root, guard_limit = directory_guard
            if _directory_bytes(guard_root) > guard_limit:
                reason = "size_guard"
                break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            reason = "timeout"
            break
        try:
            process.wait(timeout=min(_POLL_INTERVAL_SECONDS, remaining))
            leader_exited = True
            break
        except subprocess.TimeoutExpired:
            continue

    pipe_closed_in_time = True
    if reason is not None:
        _kill_group(process, pgid)
        _reap(process)
        _join_drainers(threads, _JOIN_TIMEOUT_SECONDS)
    elif leader_exited:
        pipe_closed_in_time = _join_drainers(threads, deadline - time.monotonic())
        if not pipe_closed_in_time:
            _kill_group(process, pgid)
            _join_drainers(threads, _JOIN_TIMEOUT_SECONDS)
    try:
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_group(process, pgid)
        _reap(process)

    if reason is None and overflow.is_set():
        reason = "overflow"
    if reason is None and directory_guard is not None:
        guard_root, guard_limit = directory_guard
        if _directory_bytes(guard_root) > guard_limit:
            reason = "size_guard"
    if reason is None and not pipe_closed_in_time:
        reason = "timeout"

    stdout = bytes(stdout_buffer).decode("utf-8", errors="replace")
    stderr = bytes(stderr_buffer).decode("utf-8", errors="replace")

    if reason == "overflow":
        raise YtDlpError(
            "YTDLP_OUTPUT_OVERFLOW",
            "yt-dlp produced more than the bounded stdout/stderr ceiling; the "
            "process was stopped instead of buffering unbounded output",
        )
    if reason == "size_guard":
        assert directory_guard is not None
        raise YtDlpError(
            "YTDLP_SOURCE_TOO_LARGE",
            "the download exceeded the configured source byte budget while it "
            "was running; the process was stopped and the partial output is "
            "discarded",
        )
    if reason == "timeout":
        raise YtDlpError(
            "YTDLP_TIMEOUT", f"yt-dlp exceeded the {timeout}s download deadline"
        )
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def ytdlp_version(binary: str = "yt-dlp", *, timeout: float = 30.0) -> str:
    """Return the installed ``yt-dlp`` version string, recorded as provenance.

    This is a local, offline call. It never updates the tool and never fetches
    anything.
    """

    if not isinstance(binary, str) or binary.strip() == "":
        raise YtDlpError("YTDLP_BIN_INVALID", "yt-dlp binary path must not be empty")
    completed = _run_bounded(
        [binary, "--ignore-config", "--version"],
        env=_build_env(Path(os.getcwd())),
        timeout=timeout,
        stdout_ceiling=_VERSION_CEILING_BYTES,
        stderr_ceiling=_VERSION_CEILING_BYTES,
    )
    if completed.returncode != 0:
        raise YtDlpError("YTDLP_FAILED", "yt-dlp --version failed")
    lines = (completed.stdout or "").strip().splitlines()
    if not lines or not lines[0].strip():
        raise YtDlpError("YTDLP_FAILED", "yt-dlp did not report its version")
    return lines[0].strip()[:255]


def _metadata_args(fields: dict[str, Path]) -> list[str]:
    args: list[str] = []
    for field in METADATA_FIELDS:
        args += ["--print-to-file", f"%({field})s", str(fields[field])]
    return args


def _bounded_read_text(
    path: Path,
    limit: int,
    *,
    oversize_code: str,
    oversize_message: str,
) -> str | None:
    """Read at most ``limit`` bytes from a small file, or refuse it.

    The file is opened with ``O_NOFOLLOW`` where available so a symlinked
    allowlisted read cannot be redirected outside the private working directory.
    A missing or unreadable file returns ``None``; a file larger than ``limit``
    raises a typed error instead of being silently truncated, so a partial
    identifier can never be accepted. The allocation is bounded by ``limit + 1``
    bytes *before* any decode.
    """

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        buffer = bytearray()
        while len(buffer) <= limit:
            chunk = os.read(descriptor, min(_READ_CHUNK, limit + 1 - len(buffer)))
            if not chunk:
                break
            buffer.extend(chunk)
    finally:
        os.close(descriptor)
    if len(buffer) > limit:
        raise YtDlpError(oversize_code, oversize_message)
    return bytes(buffer).decode("utf-8", errors="replace")


def _read_metadata(fields: dict[str, Path]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for name, path in fields.items():
        raw = _bounded_read_text(
            path,
            _METADATA_READ_LIMIT_BYTES,
            oversize_code="YTDLP_METADATA_OVERSIZE",
            oversize_message=(
                "an allowlisted metadata value exceeded the bounded read limit; "
                "refusing a partial identifier"
            ),
        )
        metadata[name] = "" if raw is None else raw.strip()
    return metadata


def _build_argv(
    binary: str,
    url: str,
    *,
    work_dir: Path,
    fields: dict[str, Path],
    filepath_file: Path,
    limits: YtDlpLimits,
) -> list[str]:
    # Only operators documented for yt-dlp match filters are used: ``!field``
    # (field absent) and numeric ``<=``. Public availability and completion are
    # re-validated from the allowlisted metadata after the download, so this
    # pre-download filter is a conservative guard, not the only gate. An unknown
    # duration fails the comparison and therefore fails closed.
    match_filters = "!is_live"
    if limits.max_duration_seconds is not None:
        match_filters += f" & duration <= {int(limits.max_duration_seconds)}"
    argv = _hardened_argv(
        binary,
        url,
        work_dir=work_dir,
        fields=fields,
        filepath_file=filepath_file,
        limits=limits,
        format_selector=FORMAT_AUDIO,
        match_filters=match_filters,
    )
    return argv


def _hardened_argv(
    binary: str,
    url: str,
    *,
    work_dir: Path,
    fields: dict[str, Path],
    filepath_file: Path,
    limits: YtDlpLimits,
    format_selector: str,
    match_filters: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    """Assemble one hardened yt-dlp argument list (no shell, no user config).

    The audio path passes ``FORMAT_AUDIO`` and a ``None`` ``extra_args``, so its
    argv is byte-identical to the accepted audio-only command. The opt-in video
    path passes ``FORMAT_VIDEO`` plus merge options and shares every hardening
    flag, the private working directory, the exact metadata allowlist and the
    output-path report.
    """

    argv = [
        binary,
        "--ignore-config",
        "--no-config-locations",
        "--no-plugin-dirs",
        "--no-remote-components",
        "--no-cookies",
        "--no-cookies-from-browser",
        "--no-mark-watched",
        "--no-playlist",
        "--abort-on-error",
        "-f",
        format_selector,
        "--restrict-filenames",
        "--no-overwrites",
        "--no-part",
        "--no-progress",
        "--no-simulate",
        "--newline",
        "-P",
        f"home:{work_dir}",
        "-P",
        f"temp:{work_dir}",
        "-o",
        "source.%(ext)s",
        "-R",
        str(max(0, limits.retries)),
        "--fragment-retries",
        str(max(0, limits.fragment_retries)),
        "--socket-timeout",
        str(limits.socket_timeout_seconds),
        "--match-filters",
        match_filters,
        "--print-to-file",
        "after_move:filepath",
        str(filepath_file),
    ]
    if extra_args:
        argv += extra_args
    argv += _metadata_args(fields)
    argv.append(url)
    return argv


def _build_video_argv(
    binary: str,
    url: str,
    *,
    work_dir: Path,
    fields: dict[str, Path],
    filepath_file: Path,
    limits: YtDlpLimits,
) -> list[str]:
    """Build the hardened full-video argv for the opt-in subtitle feature.

    Uses the same match filter as the audio path (refuse live, bound duration)
    and asks yt-dlp to merge a video+audio pair into MP4. TikTok/MITM bypass
    flags, cookies, browser identity, proxies and remote components stay off.
    """

    match_filters = "!is_live"
    if limits.max_duration_seconds is not None:
        match_filters += f" & duration <= {int(limits.max_duration_seconds)}"
    return _hardened_argv(
        binary,
        url,
        work_dir=work_dir,
        fields=fields,
        filepath_file=filepath_file,
        limits=limits,
        format_selector=FORMAT_VIDEO,
        match_filters=match_filters,
        extra_args=["--merge-output-format", "mp4"],
    )


def _finalize_download(
    completed: subprocess.CompletedProcess[str],
    *,
    fields: dict[str, Path],
    filepath_file: Path,
    work_dir: Path,
    route: str = "audio",
) -> YtDlpDownload:
    """Validate one bounded download's metadata and reported output path."""

    metadata = _read_metadata(fields)
    if completed.returncode != 0:
        # The child's stderr may carry a signed URL, cookie, header or token; it
        # is never copied into the raised error. Only a fixed, generic typed
        # message is returned.
        code, message = _classify_failure(completed.stderr or "", route=route)
        raise YtDlpError(code, message)

    reported = _bounded_read_text(
        filepath_file,
        _FILEPATH_READ_LIMIT_BYTES,
        oversize_code="YTDLP_OUTPUT_INVALID",
        oversize_message=(
            "the reported yt-dlp output path exceeded the bounded read limit"
        ),
    )
    reported = "" if reported is None else reported.strip()
    if not reported:
        raise YtDlpError(
            "YTDLP_NO_OUTPUT", "yt-dlp did not report a produced output path"
        )
    output = Path(reported)
    if not output.is_absolute():
        output = work_dir / output
    try:
        canonical_work = work_dir.resolve()
        canonical_output = output.resolve()
    except OSError as exc:
        raise YtDlpError("YTDLP_NO_OUTPUT", "yt-dlp output path is not resolvable") from exc
    if canonical_output.parent != canonical_work:
        raise YtDlpError(
            "YTDLP_OUTPUT_ESCAPE",
            "yt-dlp reported an output outside the private working directory",
        )
    if output.is_symlink() or not output.is_file():
        raise YtDlpError(
            "YTDLP_OUTPUT_INVALID",
            "the reported yt-dlp output is not a regular file",
        )
    extras = [
        entry
        for entry in canonical_work.iterdir()
        if entry.name.startswith("source") and entry.resolve() != canonical_output
    ]
    if extras:
        raise YtDlpError(
            "YTDLP_MULTIPLE_OUTPUTS",
            "yt-dlp produced more than one source output; refusing an ambiguous "
            "download",
        )
    return YtDlpDownload(output_path=canonical_output, metadata=metadata)


def download_audio(
    binary: str,
    url: str,
    *,
    work_dir: Path,
    limits: YtDlpLimits,
) -> YtDlpDownload:
    """Download the single best audio-only stream for ``url`` into ``work_dir``.

    The canonical URL must already have been validated and canonicalized by the
    caller; the runner does not parse or trust the URL. Existing files in
    ``work_dir`` are never reused or overwritten (``--no-overwrites``); the caller
    is responsible for providing an empty private directory.
    """

    if not isinstance(url, str) or url.strip() == "":
        raise YtDlpError("YTDLP_URL_INVALID", "canonical URL must not be empty")
    work_dir = Path(work_dir)
    fields = {name: work_dir / f".meta.{name}.txt" for name in METADATA_FIELDS}
    filepath_file = work_dir / ".meta.filepath.txt"
    argv = _build_argv(
        binary,
        url,
        work_dir=work_dir,
        fields=fields,
        filepath_file=filepath_file,
        limits=limits,
    )
    completed = _run_bounded(
        argv,
        env=_build_env(work_dir),
        timeout=limits.download_timeout_seconds,
        stdout_ceiling=limits.max_stdout_bytes,
        stderr_ceiling=limits.max_stderr_bytes,
        directory_guard=(work_dir, limits.max_directory_bytes),
    )
    return _finalize_download(
        completed,
        fields=fields,
        filepath_file=filepath_file,
        work_dir=work_dir,
        route="audio",
    )


def download_video(
    binary: str,
    url: str,
    *,
    work_dir: Path,
    limits: YtDlpLimits,
) -> YtDlpDownload:
    """Download the best full video (video+audio) for ``url`` into ``work_dir``.

    This is the opt-in companion to :func:`download_audio` for the burned-in
    subtitle feature. It shares every hardening guarantee (shell-free argv, user
    config/plugin/cookie/browser/proxy refusal, TLS verification, private
    environment, process-group kill on timeout/overflow/size trip, allowlisted
    metadata, symlink/path confinement) while selecting a *video* stream and
    merging it to MP4. The caller is responsible for providing an empty private
    directory; the accepted audio argv and cache identity are untouched.
    """

    if not isinstance(url, str) or url.strip() == "":
        raise YtDlpError("YTDLP_URL_INVALID", "canonical URL must not be empty")
    work_dir = Path(work_dir)
    fields = {name: work_dir / f".meta.{name}.txt" for name in METADATA_FIELDS}
    filepath_file = work_dir / ".meta.filepath.txt"
    argv = _build_video_argv(
        binary,
        url,
        work_dir=work_dir,
        fields=fields,
        filepath_file=filepath_file,
        limits=limits,
    )
    completed = _run_bounded(
        argv,
        env=_build_env(work_dir),
        timeout=limits.download_timeout_seconds,
        stdout_ceiling=limits.max_stdout_bytes,
        stderr_ceiling=limits.max_stderr_bytes,
        directory_guard=(work_dir, limits.max_directory_bytes),
    )
    return _finalize_download(
        completed,
        fields=fields,
        filepath_file=filepath_file,
        work_dir=work_dir,
        route="video",
    )
