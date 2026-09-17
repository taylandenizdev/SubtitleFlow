"""Minimal platform shims for the two POSIX-only OS dependencies.

The pipeline is written against POSIX process and file primitives. Two of them
are unavailable on Windows and previously sat directly on the import and runtime
paths:

* ``fcntl.flock`` for the single-writer job and source-cache locks, and
* ``os.killpg`` / ``signal.SIGKILL`` plus ``start_new_session`` for the bounded
  ``yt-dlp`` / ``ffmpeg`` child lifecycle.

This module keeps the POSIX behaviour unchanged and adds a small Windows
implementation so the package imports and the pipeline runs there. It is
deliberately small and standard-library only (``msvcrt`` plus the ``taskkill``
utility): no third-party process library and no large, unchecked ``ctypes``
surface. Nothing here is imported at package import time except this module
itself; ``fcntl`` is imported lazily inside the POSIX lock path, so a Windows
interpreter never touches it.

Honesty limits on Windows
-------------------------
* ``msvcrt.locking`` is a *region* lock, not ``flock``: it is advisory and bound
  to a byte range. Every acquisition locks byte 0 and every release unlocks the
  same byte 0, so a lock is always matched, but the full BSD ``flock`` semantics
  are not reproduced. The one property the store needs is preserved: a second
  writer is refused while the first holds the byte.
* There is no Windows equivalent of an ``os.killpg`` group kill. While the
  owned direct child is still alive, ``taskkill /T /F /PID <owned pid>`` is used
  to take down its tree, and only that owned pid is ever targeted. If the leader
  has already exited the tree can no longer be named safely (the pid may have
  been reused), so no ``taskkill`` is attempted and surviving descendants cannot
  be guaranteed to be reaped. That remainder is not hidden here.
* ``os.replace`` fails on Windows with a sharing violation while the destination
  is open without delete sharing. :func:`replace_file_atomically` retries that
  exact error for a bounded time on Windows; the POSIX path is a plain
  ``os.replace``.
* None of the Windows code paths were executed on Windows. Only the POSIX path
  and the simulated Windows backend (a fake ``msvcrt`` and a fake process) were
  exercised.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Final

__all__ = [
    "kill_owned_process_group",
    "lock_fd_exclusive_nonblocking",
    "owned_process_kwargs",
    "replace_file_atomically",
    "unlock_fd",
]

#: The one predicate every backend branch reads. It is a module attribute so the
#: POSIX-only test environment can exercise the Windows branches with a fake
#: backend without pretending the host is Windows.
_IS_WINDOWS: bool = os.name == "nt"

#: ``msvcrt`` exists only on Windows. The guarded import keeps the module
#: importable everywhere and leaves ``None`` on POSIX.
_msvcrt: Any
try:  # pragma: no cover - importable only on Windows
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - normal on POSIX
    _msvcrt = None

#: ``subprocess.CREATE_NEW_PROCESS_GROUP`` is only defined on Windows, so the
#: documented value is carried here and referenced under the Windows branch only.
_WINDOWS_CREATE_NEW_PROCESS_GROUP: Final[int] = 0x00000200

#: ``taskkill /T /F`` on one owned pid is given this long before being abandoned;
#: a failed tree kill is best effort and never masks the typed child error.
_TASKKILL_TIMEOUT_SECONDS: Final[float] = 10.0

#: ``os.replace`` sharing-violation retry budget, Windows only. The reader that
#: holds the destination open closes it almost immediately, so a short bounded
#: wait turns a spurious failure into a durable write without masking a real one.
_REPLACE_RETRY_ATTEMPTS: Final[int] = 25
_REPLACE_RETRY_DELAY_SECONDS: Final[float] = 0.02


def is_windows() -> bool:
    """Whether the Windows backend is selected for this interpreter."""

    return _IS_WINDOWS


# --------------------------------------------------------------------------- #
# File locks
# --------------------------------------------------------------------------- #
def lock_fd_exclusive_nonblocking(fd: int) -> None:
    """Take an exclusive, non-blocking lock on ``fd`` or raise ``OSError``.

    POSIX uses ``fcntl.flock`` with ``LOCK_EX | LOCK_NB``. Windows locks byte 0
    with ``msvcrt.locking(LK_NBLCK)`` after seeking to byte 0, so the acquired
    range is deterministic regardless of the descriptor's prior file position.
    """

    if _IS_WINDOWS:
        _lock_fd_windows(fd)
    else:
        _lock_fd_posix(fd)


def unlock_fd(fd: int) -> None:
    """Release a lock previously taken by :func:`lock_fd_exclusive_nonblocking`."""

    if _IS_WINDOWS:
        _unlock_fd_windows(fd)
    else:
        _unlock_fd_posix(fd)


def _lock_fd_posix(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd_posix(fd: int) -> None:
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _lock_fd_windows(fd: int) -> None:
    if _msvcrt is None:
        raise OSError("the Windows file-locking backend is unavailable")
    os.lseek(fd, 0, os.SEEK_SET)
    _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)


def _unlock_fd_windows(fd: int) -> None:
    if _msvcrt is None:
        raise OSError("the Windows file-locking backend is unavailable")
    os.lseek(fd, 0, os.SEEK_SET)
    _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)


# --------------------------------------------------------------------------- #
# Child process ownership and termination
# --------------------------------------------------------------------------- #
def owned_process_kwargs() -> dict[str, object]:
    """Return the ``Popen`` keyword that places a child in its own group.

    POSIX keeps ``start_new_session=True``. Windows, where that argument is
    documented as POSIX-only and ignored, asks for a new process group instead.
    """

    if _IS_WINDOWS:
        return {"creationflags": _WINDOWS_CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_owned_process_group(
    process: "subprocess.Popen[bytes]", pgid: int | None
) -> None:
    """Best-effort kill of the child that this pipeline itself started.

    Only the owned group/pid is ever signalled. The call never raises: a failed
    or unavailable cleanup must never replace the typed timeout/overflow error
    the caller is about to report. On Windows a tree kill is attempted only
    while the leader is still alive, because a reaped leader's pid can be reused.
    """

    try:
        if _IS_WINDOWS:
            _kill_windows_tree(process)
        else:
            _kill_posix_group(process, pgid)
    except (OSError, AttributeError, ValueError, subprocess.SubprocessError):
        pass


def _kill_posix_group(
    process: "subprocess.Popen[bytes]", pgid: int | None
) -> None:
    if pgid is None:
        _safe_kill(process)
        return
    os.killpg(pgid, signal.SIGKILL)


def _kill_windows_tree(process: "subprocess.Popen[bytes]") -> None:
    if process.poll() is None:
        # The leader owns the tree and its pid is still ours; hand that exact pid
        # to taskkill. Once poll() reports an exit the pid may already name an
        # unrelated process, so no tree kill is attempted.
        _terminate_windows_tree(process.pid)
    _safe_kill(process)


def _terminate_windows_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_TASKKILL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _safe_kill(process: "subprocess.Popen[bytes]") -> None:
    try:
        process.kill()
    except (ProcessLookupError, OSError):
        pass


# --------------------------------------------------------------------------- #
# Atomic publication
# --------------------------------------------------------------------------- #
def replace_file_atomically(
    source: str | os.PathLike[str], destination: str | os.PathLike[str]
) -> None:
    """``os.replace`` with a bounded Windows sharing-violation retry.

    On POSIX this is exactly ``os.replace``. On Windows a concurrent reader of
    the destination can hold it without delete sharing for a very short window;
    that specific ``PermissionError`` is retried for a bounded time, after which
    the original error is raised rather than hidden.
    """

    if not _IS_WINDOWS:
        os.replace(source, destination)
        return
    last_error: PermissionError | None = None
    for _ in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)
    assert last_error is not None
    raise last_error
