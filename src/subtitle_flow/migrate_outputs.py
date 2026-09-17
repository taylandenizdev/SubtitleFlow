"""Lossless, idempotent migration of the legacy ``transkriptler/`` folder.

Historically the source Markdown lived at the repository root in
``transkriptler/``. The canonical location is now ``outputs/transkriptler/`` (with
``outputs/ceviriler/`` beside it). This module moves the legacy content without
losing a single byte:

* every file is copied to a fresh temporary file, ``fsync``-ed and hash-verified
  against the source before it is published atomically;
* an existing destination with *identical* content is accepted and the source
  copy is removed (idempotent re-run); an existing destination with *different*
  content is a reported collision and is **never** overwritten, while the source
  is left untouched;
* symlinks, non-regular entries and escaping paths are refused, never followed;
* a source file that is being written while it is copied (size/`mtime_ns`/inode
  change) is reported as a running-write collision instead of being published;
* an empty or absent source is a no-op.

The migration is safe to run twice and never deletes a source file before its
verified destination exists.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from subtitle_flow import platform_compat
from subtitle_flow.output_paths import unsafe_symlink_ancestor

__all__ = [
    "MigrationCollision",
    "MigrationError",
    "MigrationReport",
    "migrate_transcripts",
]

F_MIGRATION_LOCK: Final[str] = ".migration.lock"


class MigrationError(Exception):
    """A typed, actionable migration failure (nothing unsafe was published)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class MigrationCollision:
    """One source entry that was intentionally left untouched."""

    relative: str
    reason: str


@dataclass(frozen=True)
class MigrationReport:
    """Deterministic result of one migration pass."""

    source: str
    target: str
    moved: tuple[str, ...] = ()
    removed_identical: tuple[str, ...] = ()
    collisions: tuple[MigrationCollision, ...] = ()
    skipped: tuple[str, ...] = ()
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.collisions

    @property
    def changed(self) -> bool:
        return bool(self.moved or self.removed_identical)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _is_within(base: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _acquire_lock(directory: Path) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / F_MIGRATION_LOCK
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    try:
        platform_compat.lock_fd_exclusive_nonblocking(fd)
    except OSError as exc:
        os.close(fd)
        raise MigrationError(
            "MIGRATION_LOCKED",
            "another migration is already running (or the lock is held)",
        ) from exc
    return fd


def _publish(source: Path, relative: str, target_root: Path) -> str:
    """Copy one file into the target after verification; delete source on success."""

    target = target_root / relative
    target_parent = target.parent
    target_parent.mkdir(parents=True, exist_ok=True)
    if not _is_within(target_root, target_parent):
        raise MigrationError(
            "MIGRATION_PATH_ESCAPE", f"target path escapes the output root: {relative}"
        )
    if target.is_symlink():
        raise MigrationError(
            "MIGRATION_TARGET_SYMLINK",
            f"destination is a symlink and is never written: {relative}",
        )

    before = _stat_identity(source)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".migrate-", dir=str(target_parent)
    )
    handle = os.fdopen(descriptor, "wb")
    try:
        with source.open("rb") as reader:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()

    after = _stat_identity(source)
    if before != after:
        os.unlink(temporary)
        raise MigrationError(
            "MIGRATION_SOURCE_CHANGED",
            f"source changed while it was copied: {relative}",
        )
    if _sha256_file(source) != _sha256_file(Path(temporary)):
        os.unlink(temporary)
        raise MigrationError(
            "MIGRATION_HASH_MISMATCH",
            f"copied bytes do not match the source: {relative}",
        )
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise MigrationError(
            "MIGRATION_TARGET_RACE",
            f"destination appeared during the migration: {relative}",
        ) from exc
    except OSError as exc:
        raise MigrationError(
            "MIGRATION_PUBLISH_FAILED",
            f"the verified copy could not be published for {relative}: {exc}",
        ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    _fsync_dir(target_parent)
    os.unlink(source)
    return relative


def _fsync_dir(directory: Path) -> None:
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


def migrate_transcripts(
    source_dir: str | Path,
    target_dir: str | Path,
    *,
    dry_run: bool = False,
) -> MigrationReport:
    """Move ``source_dir`` content into ``target_dir`` losslessly and idempotently.

    ``dry_run`` reports exactly what would happen without writing or deleting.
    A differing same-name destination, a symlink entry, a non-regular entry or a
    source file changing mid-copy becomes a reported collision; the colliding
    source is always left in place.
    """

    source = Path(source_dir).expanduser()
    target = Path(target_dir).expanduser()
    moved: list[str] = []
    removed_identical: list[str] = []
    collisions: list[MigrationCollision] = []
    skipped: list[str] = []

    if not source.exists():
        return MigrationReport(
            source=str(source), target=str(target), skipped=("source-missing",),
            dry_run=dry_run,
        )
    if source.is_symlink() or not source.is_dir():
        raise MigrationError(
            "MIGRATION_SOURCE_INVALID",
            "the migration source must be a real directory, not a symlink",
        )
    if target.is_symlink():
        raise MigrationError(
            "MIGRATION_TARGET_SYMLINK",
            "the migration target must not be a symlink",
        )
    unsafe = unsafe_symlink_ancestor(target)
    if unsafe is not None:
        raise MigrationError(
            "MIGRATION_TARGET_UNSAFE",
            "the migration target is below a symlinked path component and is "
            f"never written through: {unsafe}",
        )

    if not dry_run:
        lock_fd = _acquire_lock(target.parent)
    else:
        lock_fd = None
    try:
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(source, followlinks=False):
            root_path = Path(root)
            # A symlinked directory must never be followed or removed silently.
            real_dirs: list[str] = []
            for name in sorted(dirs):
                candidate = root_path / name
                if candidate.is_symlink():
                    relative = str(candidate.relative_to(source))
                    collisions.append(
                        MigrationCollision(relative, "symlinked directory")
                    )
                    continue
                real_dirs.append(name)
            dirs[:] = real_dirs
            for name in sorted(files):
                candidate = root_path / name
                relative = str(candidate.relative_to(source))
                if candidate.is_symlink():
                    collisions.append(MigrationCollision(relative, "symlink"))
                    continue
                if not candidate.is_file():
                    collisions.append(MigrationCollision(relative, "not a regular file"))
                    continue
                destination = target / relative
                if destination.is_symlink():
                    collisions.append(
                        MigrationCollision(relative, "destination is a symlink")
                    )
                    continue
                if destination.exists():
                    if destination.is_dir():
                        collisions.append(
                            MigrationCollision(relative, "destination is a directory")
                        )
                        continue
                    if _sha256_file(candidate) == _sha256_file(destination):
                        removed_identical.append(relative)
                        if not dry_run:
                            os.unlink(candidate)
                        continue
                    collisions.append(
                        MigrationCollision(relative, "destination has different content")
                    )
                    continue
                if dry_run:
                    moved.append(relative)
                    continue
                try:
                    moved.append(_publish(candidate, relative, target))
                except MigrationError as exc:
                    collisions.append(MigrationCollision(relative, exc.code))
                except OSError as exc:
                    # Any unexpected filesystem error is reported per entry so a
                    # single bad file never aborts the rest of the migration.
                    collisions.append(
                        MigrationCollision(relative, f"MIGRATION_IO_ERROR:{exc.errno}")
                    )

        if not dry_run:
            _remove_empty_dirs(source)
    finally:
        if lock_fd is not None:
            try:
                platform_compat.unlock_fd(lock_fd)
            finally:
                os.close(lock_fd)

    return MigrationReport(
        source=str(source),
        target=str(target),
        moved=tuple(moved),
        removed_identical=tuple(removed_identical),
        collisions=tuple(collisions),
        skipped=tuple(skipped),
        dry_run=dry_run,
    )


def _remove_empty_dirs(root: Path) -> None:
    """Remove now-empty directories bottom-up, including the source root."""

    for current, dirs, _files in os.walk(root, topdown=False):
        path = Path(current)
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
            except OSError:
                pass
        except OSError:
            continue
