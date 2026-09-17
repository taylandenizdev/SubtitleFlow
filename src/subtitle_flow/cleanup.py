"""Automatic per-video disk cleanup for *completed* desktop YouTube jobs.

The desktop YouTube -> Scribe STT -> Google Basic v2 Turkish MT chain keeps a
number of durable
artefacts per video: the job directory (raw STT/MT bodies, canonical
``artifacts/`` and manifests) and a shared source-cache entry under the
``youtube-sources`` cache. Once the current run has safely published **both**
human-facing Markdown documents (source and Turkish), the operator's explicit
choice is that everything except the Markdown is removed for that completed
video. This module implements exactly that, and only that:

* The job to erase is the *exact* finished job id (never a glob); it is only
  erased after its stored YouTube origin is read back and cross-checked.
* Every *other* job that records the same validated video id is erased only when
  it too carries a terminal Turkish MT artifact; a source-only, partial, locked,
  symlinked or otherwise ambiguous job is preserved for recovery.
* The cache entry to erase is the *exact* entry recorded by each erased job's
  origin, and it is verified against its stored manifest before deletion.
* Both directories are confined, symlink-free and lock-serialized. A symlink,
  an escaped path, a locked job or an unexpected tree is refused and reported,
  never followed or guessed at.
* A cache entry still referenced by another (for example incomplete/partial)
  job is retained so that job remains resumable.
* Failure is never allowed to mask the successful documents: callers get an
  honest warning and any remaining data is kept.
* Only the cache *entry* is removed; the race-safe 0-byte ``<entry>.lock``
  marker is intentionally kept because it is the stable inode that serializes
  concurrent writers, and unlinking it would reintroduce a lock race.

The callers decide *when* this is safe: the CLI/desktop layer verifies that the
current run really published both documents under the selected output
directories before invoking :func:`clean_completed_youtube_job`. This module
performs no document verification of its own beyond that contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from subtitle_flow import platform_compat
from subtitle_flow.config import YouTubeOriginSettings
from subtitle_flow.output_paths import unsafe_symlink_ancestor
from subtitle_flow.storage import (
    JobLockedError,
    JobStore,
    StorageError,
    StoredInput,
)
from subtitle_flow.youtube_source import (
    YouTubeMediaError,
    purge_cached_youtube_source,
)

__all__ = [
    "CleanupOutcome",
    "clean_completed_youtube_job",
    "verify_published_document",
]

_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class CleanupOutcome:
    """Result of one automatic cleanup attempt.

    ``warning_text`` is ``None`` only when every eligible job (the named one and
    its completed same-video siblings) and each exclusive cache entry were
    removed (or there was genuinely nothing to remove). Any partial or refused
    cleanup produces a short Turkish warning that callers must surface; it never
    claims a cleanup that did not happen.
    """

    attempted: bool = False
    job_removed: bool = False
    cache_removed: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.attempted and self.job_removed and not self.warnings

    @property
    def warning_text(self) -> str | None:
        if not self.warnings:
            return None
        return "Otomatik disk temizliği tamamlanamadı: " + " ".join(self.warnings)


def verify_published_document(
    path: str | os.PathLike[str] | None,
    *,
    expected_dir: str | os.PathLike[str],
    video_id: str | None,
    expected_text: str | None = None,
) -> str | None:
    """Return ``None`` when ``path`` is a real, safe document, else a reason.

    A document only counts for cleanup when it is a regular, non-symlinked
    ``<video_id>.md`` file directly inside the selected output directory, below
    no unsafe symlinked component and non-empty. A stale pre-existing file at an
    unrelated location can never satisfy this check.

    ``expected_text`` adds a byte-identity proof: when given, the file's bytes
    must hash exactly to ``expected_text`` (UTF-8). This binds a translate-only
    cleanup to the precise source document this run translated, so a stale or
    different-provider ``<video_id>.md`` can never authorize deleting the raw
    evidence.
    """

    if not isinstance(video_id, str) or _SAFE_NAME_RE.match(video_id) is None:
        return "video kimliği eksik veya güvenli değil"
    if not path:
        return "yayımlanmış Markdown yolu yok"
    candidate = Path(path)
    if candidate.name != f"{video_id}.md":
        return "Markdown dosya adı video kimliğiyle eşleşmiyor"
    if os.path.islink(candidate):
        return "Markdown dosyası sembolik bağlantı"
    if not candidate.is_file():
        return "Markdown dosyası bulunamadı"
    unsafe = unsafe_symlink_ancestor(candidate)
    if unsafe is not None:
        return f"Markdown yolu güvenli olmayan sembolik bileşen içeriyor: {unsafe}"
    try:
        if os.path.realpath(candidate.parent) != os.path.realpath(
            Path(expected_dir).expanduser()
        ):
            return "Markdown seçilen çıktı dizininin dışında"
        data = candidate.read_bytes()
    except OSError:
        return "Markdown dosyası okunamadı"
    if not data:
        return "Markdown dosyası boş"
    if expected_text is not None:
        expected = expected_text.encode("utf-8")
        if hashlib.sha256(data).digest() != hashlib.sha256(expected).digest():
            return "Markdown içeriği bu çalıştırmada çevrilen kaynakla eşleşmiyor"
    return None


def _find_symlink(root: Path) -> str | None:
    """Return the first symlink found in the tree rooted at ``root`` (or ``None``)."""

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            if os.path.islink(os.path.join(dirpath, name)):
                return os.path.join(dirpath, name)
    return None


def _remove_tree(path: Path) -> str | None:
    """Best-effort recursive delete; return a reason when something remains."""

    shutil.rmtree(path, ignore_errors=True)
    if not os.path.lexists(path):
        return None
    # A read-only file or directory can survive a best-effort delete; relax the
    # permissions once and retry. Any remainder is reported, never hidden.
    for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
        for name in (*dirnames, *filenames):
            target = os.path.join(dirpath, name)
            if os.path.islink(target):
                continue
            try:
                os.chmod(target, 0o700)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)
    if os.path.lexists(path):
        return "bazı dosyalar silinemedi"
    return None


def _remove_job_under_lock(
    store: JobStore,
    *,
    validate: Callable[[], str | None] | None = None,
) -> tuple[bool, str | None]:
    """Delete one verified job directory under its writer lock.

    ``validate`` is re-run *while the lock is held*, immediately before the tree
    is touched, so a job can never be erased after it stopped qualifying (for
    example a translation that turned out to be partial, a changed origin or a
    newly appeared symlink). It returns a reason to refuse or ``None``.
    """

    try:
        store.acquire_lock()
    except JobLockedError:
        return False, "iş dizini başka bir yazar tarafından kilitli; veri korundu"
    except StorageError:
        return False, "iş kilidi alınamadı; veri korundu"
    try:
        if validate is not None:
            refusal = validate()
            if refusal is not None:
                return False, refusal
        linked = _find_symlink(store.job_dir)
        if linked is not None:
            return False, f"iş dizininde beklenmeyen sembolik bağlantı; silinmedi: {linked}"
        if platform_compat.is_windows():
            # Windows cannot remove a directory tree while its lock file is
            # open; release first, then delete best-effort.
            store.release_lock()
        reason = _remove_tree(store.job_dir)
    finally:
        store.release_lock()
    if reason is not None:
        return False, f"iş dizini tamamen silinemedi: {reason}"
    return True, None


def _has_fulltext_translation(store: JobStore) -> bool:
    """Whether a terminal, validated full-text MT artifact is present.

    Only the desktop route's ``artifacts/mt.fulltext*.json`` evidence counts; a
    symlinked, unreadable or non-terminal file returns ``False`` so the job is
    preserved rather than guessed safe.
    """

    from subtitle_flow.fulltext_mt import FullTextTranslationArtifact

    artifacts_dir = store.job_dir / "artifacts"
    if os.path.islink(artifacts_dir) or not artifacts_dir.is_dir():
        return False
    try:
        children = list(os.scandir(artifacts_dir))
    except OSError:
        return False
    found = False
    for child in children:
        name = child.name
        if not (name.startswith("mt.fulltext") and name.endswith(".json")):
            continue
        try:
            if child.is_symlink() or not child.is_file():
                return False
        except OSError:
            return False
        try:
            data = Path(child.path).read_bytes()
        except OSError:
            return False
        try:
            artifact = FullTextTranslationArtifact.model_validate_json(data)
        except ValueError:
            return False
        if artifact.status not in {"translated", "skipped_same_language"}:
            return False
        found = True
    return found


def _job_has_completed_translation(store: JobStore) -> bool:
    """Whether ``store`` carries a terminal, validated Turkish MT artifact.

    This is deliberately stronger than "STT finished": a source-only or partial
    job must stay resumable, so only a fully persisted translation (the desktop
    full-text artifact, or a complete segment MT manifest) counts. Any unreadable,
    unvalidated or non-terminal evidence returns ``False`` and the job is
    preserved.
    """

    if _has_fulltext_translation(store):
        return True
    try:
        manifest = store.read_mt_manifest()
    except (StorageError, OSError):
        return False
    if manifest is None or manifest.status != "complete" or manifest.translation is None:
        return False
    try:
        store.verify_artifact(manifest.translation)
    except StorageError:
        return False
    return True


def _candidate_ineligible_reason(store: JobStore) -> str | None:
    """A reason why a same-video sibling must not be auto-erased (or ``None``)."""

    linked = _find_symlink(store.job_dir)
    if linked is not None:
        return f"iş dizininde beklenmeyen sembolik bağlantı: {linked}"
    if not _job_has_completed_translation(store):
        return "tamamlanmış Türkçe çeviri kanıtı yok"
    return None


def _validate_completed_job(store: JobStore, *, video_id: str) -> str | None:
    """Re-check, under the held lock, that a sibling is still eligible."""

    try:
        stored = store.read_input()
    except (StorageError, OSError):
        return "iş dizini doğrulanamadı; veri korundu"
    origin = stored.config.youtube_origin
    if origin is None or origin.video_id != video_id:
        return "kayıtlı YouTube video kimliği beklenenle eşleşmiyor; veri korundu"
    if not _job_has_completed_translation(store):
        return "tamamlanmış Türkçe çeviri kanıtı yok; veri korundu"
    return None


@dataclass(frozen=True)
class _SameVideoJob:
    """One job directory that records the target video, plus its eligibility."""

    job_id: str
    origin: YouTubeOriginSettings
    cache_root: str
    refusal: str | None


def _gather_same_video_jobs(
    job_root: Path,
    *,
    current_id: str,
    current_stored: StoredInput,
    video_id: str,
) -> list[_SameVideoJob]:
    """Gather every job directory recording exactly ``video_id`` as its origin.

    Only the stored, validated origin is used; an unrelated video, a symlinked
    directory, an unreadable snapshot or a foreign file is skipped untouched.
    The explicitly requested current job is always included (its documents were
    just verified by the caller); every other same-video job carries a refusal
    reason unless it holds a terminal Turkish MT artifact.
    """

    candidates: list[_SameVideoJob] = []
    try:
        entries = sorted(job_root.iterdir())
    except OSError:
        return candidates
    for entry in entries:
        if entry.is_symlink() or not entry.is_dir():
            continue
        if entry.name == current_id:
            stored = current_stored
            refusal: str | None = None
        else:
            try:
                store = JobStore(job_root, entry.name)
                if not store.exists():
                    continue
                stored = store.read_input()
            except (StorageError, OSError):
                continue
            refusal = _candidate_ineligible_reason(store)
        origin = stored.config.youtube_origin
        if origin is None or origin.video_id != video_id:
            continue
        candidates.append(
            _SameVideoJob(
                job_id=entry.name,
                origin=origin,
                cache_root=stored.config.youtube_cache_root,
                refusal=refusal,
            )
        )
    return candidates


def _purge_origin_cache(
    origin: YouTubeOriginSettings, *, cache_root: str
) -> tuple[bool, str | None]:
    """Purge one origin's exact cache entry, treating "already gone" as success.

    Returns ``(cache_gone, warning)``. The race-safe ``<entry>.lock`` marker is
    intentionally left in place: it is the stable inode other writers lock on,
    and unlinking it would reintroduce a lock race for no benefit.
    """

    try:
        purge = purge_cached_youtube_source(origin, cache_root=cache_root)
    except (YouTubeMediaError, OSError):
        return False, "kaynak önbelleği temizlenemedi"
    if purge.removed:
        return True, None
    if purge.reason and "zaten yok" in purge.reason:
        return True, None
    if purge.reason:
        return False, f"kaynak önbelleği temizlenemedi: {purge.reason}"
    return False, "kaynak önbelleği temizlenemedi"


def _other_job_references_origin(
    job_root: Path,
    *,
    exclude_id: str,
    origin: YouTubeOriginSettings,
) -> bool:
    """Whether any *other* job still records this exact source identity.

    The source cache is shared by every job that acquired the same canonical
    audio, regardless of STT/MT provider. A still-present (for example
    incomplete or partial) job must keep that cache so it stays resumable.
    """

    try:
        entries = sorted(job_root.iterdir())
    except OSError:
        return False
    for entry in entries:
        if entry.name == exclude_id or entry.is_symlink() or not entry.is_dir():
            continue
        try:
            store = JobStore(job_root, entry.name)
            if not store.exists():
                continue
            stored = store.read_input()
        except (StorageError, OSError):
            continue
        other = stored.config.youtube_origin
        if other is not None and other.identity_payload() == origin.identity_payload():
            return True
    return False


def clean_completed_youtube_job(
    *,
    job_root: str | os.PathLike[str],
    job_id: str,
    expected_video_id: str | None = None,
) -> CleanupOutcome:
    """Erase the exact completed job, its completed same-video siblings and cache.

    The named ``job_id`` is the run whose documents the caller just verified, so
    it is erased once its stored ``input.json`` records the expected video id.
    Every *other* job directory recording that same validated video id is erased
    only when it also carries a terminal Turkish MT artifact, so a source-only,
    partial, locked, symlinked or otherwise ambiguous job is preserved for
    recovery. Afterwards each erased job's exclusive source-cache entry is purged
    under its own cache lock; an entry still referenced by any remaining job is
    kept and reported.

    Every failure path returns a warning instead of raising, so a completed run's
    published documents are never masked by a cleanup problem.
    """

    warnings: list[str] = []
    try:
        current = JobStore(job_root, job_id)
    except StorageError:
        return CleanupOutcome(
            attempted=True, warnings=("iş kimliği güvenli değil; temizlik yapılmadı",)
        )
    try:
        if not current.exists():
            return CleanupOutcome(
                attempted=True, warnings=("iş dizini bulunamadı; temizlik yapılmadı",)
            )
        stored = current.read_input()
    except (StorageError, OSError):
        return CleanupOutcome(
            attempted=True, warnings=("iş dizini doğrulanamadı; veri korundu",)
        )

    origin = stored.config.youtube_origin
    if origin is None:
        return CleanupOutcome(
            attempted=True,
            warnings=("kayıtlı YouTube kaynağı yok; temizlik yapılmadı",),
        )
    if expected_video_id is not None and origin.video_id != expected_video_id:
        return CleanupOutcome(
            attempted=True,
            warnings=("kayıtlı YouTube video kimliği beklenenle eşleşmiyor; veri korundu",),
        )
    video_id = origin.video_id
    root = Path(job_root)

    # Gather first (read-only), then erase under per-job locks. A same-video job
    # without terminal translation evidence is skipped silently: it is preserved
    # for recovery and nothing was expected from it.
    candidates = _gather_same_video_jobs(
        root, current_id=job_id, current_stored=stored, video_id=video_id
    )
    removed: list[_SameVideoJob] = []
    for candidate in candidates:
        if candidate.refusal is not None:
            if candidate.job_id == job_id:
                warnings.append(f"iş dizini silinmedi: {candidate.refusal}")
            continue
        try:
            store = (
                current
                if candidate.job_id == job_id
                else JobStore(job_root, candidate.job_id)
            )
        except StorageError:
            warnings.append(
                f"{candidate.job_id}: iş kimliği güvenli değil; veri korundu"
            )
            continue
        if candidate.job_id == job_id:
            job_removed, job_warning = _remove_job_under_lock(store)
        else:
            job_removed, job_warning = _remove_job_under_lock(
                store,
                validate=lambda s=store: _validate_completed_job(s, video_id=video_id),
            )
        if job_removed:
            removed.append(candidate)
        elif job_warning is not None:
            warnings.append(f"{candidate.job_id}: {job_warning}")

    distinct: dict[str, tuple[YouTubeOriginSettings, str]] = {}
    for candidate in removed:
        key = json.dumps(candidate.origin.identity_payload(), sort_keys=True)
        distinct.setdefault(key, (candidate.origin, candidate.cache_root))
    cache_removed = bool(removed)
    for _payload, (candidate_origin, cache_root) in distinct.items():
        if _other_job_references_origin(
            root, exclude_id=job_id, origin=candidate_origin
        ):
            warnings.append(
                "kaynak önbelleği başka bir iş tarafından paylaşıldığı için korundu"
            )
            cache_removed = False
            continue
        cache_gone, cache_warning = _purge_origin_cache(
            candidate_origin, cache_root=cache_root
        )
        if cache_warning is not None:
            warnings.append(cache_warning)
            cache_removed = False
        elif not cache_gone:
            cache_removed = False

    if not removed and not warnings:
        warnings.append("silinebilir tamamlanmış iş bulunamadı; veri korundu")

    return CleanupOutcome(
        attempted=True,
        job_removed=bool(removed),
        cache_removed=cache_removed,
        warnings=tuple(warnings),
    )
