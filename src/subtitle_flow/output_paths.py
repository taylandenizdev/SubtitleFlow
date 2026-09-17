"""Canonical on-disk locations for source and Turkish output Markdown.

The desktop/CLI workflow keeps two parallel, human-facing document sets apart:

* ``outputs/transkriptler/<video_id>.md`` -- the exact original-language source
  Markdown (one file per video), and
* ``outputs/ceviriler/<video_id>.md`` -- the Turkish machine translation.

The original job directories (``jobs/<job_id>/raw`` evidence, canonical
artifacts and manifests) are the durable machine evidence and are never
rewritten by these exports. ``outputs/`` is Git-ignored.

The helpers here are deliberately pure path helpers: they do no I/O and never
touch the filesystem. Callers that write documents use the safe, atomic writers
in :mod:`subtitle_flow.transcript_markdown` and
:mod:`subtitle_flow.fulltext_mt`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_SOURCE_DIRNAME",
    "DEFAULT_TRANSLATION_DIRNAME",
    "DEFAULT_VIDEO_DIRNAME",
    "default_source_dir",
    "default_translation_dir",
    "default_video_dir",
    "translation_dir_for_source",
    "unsafe_symlink_ancestor",
    "video_dir_for_source",
]

#: Default repository-relative directory for source Markdown documents.
DEFAULT_SOURCE_DIRNAME: Final[str] = "outputs/transkriptler"

#: Default repository-relative directory for Turkish translation documents.
DEFAULT_TRANSLATION_DIRNAME: Final[str] = "outputs/ceviriler"

#: Default repository-relative directory for burned-in Turkish subtitle videos.
DEFAULT_VIDEO_DIRNAME: Final[str] = "outputs/videolar"

#: The historical, pre-migration directory name that the migration moves out of.
LEGACY_SOURCE_DIRNAME: Final[str] = "transkriptler"


def default_source_dir(base: str | Path | None = None) -> Path:
    """Return the canonical source-Markdown directory under ``base``.

    ``base`` defaults to the current working directory so a launcher anchored at
    the repository root always resolves the same absolute location.
    """

    root = Path.cwd() if base is None else Path(base)
    return root / DEFAULT_SOURCE_DIRNAME


def default_translation_dir(base: str | Path | None = None) -> Path:
    """Return the canonical Turkish-Markdown directory under ``base``."""

    root = Path.cwd() if base is None else Path(base)
    return root / DEFAULT_TRANSLATION_DIRNAME


def default_video_dir(base: str | Path | None = None) -> Path:
    """Return the canonical burned-in subtitle video directory under ``base``."""

    root = Path.cwd() if base is None else Path(base)
    return root / DEFAULT_VIDEO_DIRNAME


def video_dir_for_source(source_dir: str | Path) -> Path:
    """Derive the sibling ``videolar`` directory for a source-directory override."""

    resolved = Path(source_dir).expanduser()
    return resolved.parent / "videolar"


def translation_dir_for_source(source_dir: str | Path) -> Path:
    """Derive the sibling Turkish directory for a source-directory override.

    A deliberate custom ``--transcript-dir`` is honoured: if the given directory
    is named ``transkriptler`` (the historical/default source folder) the Turkish
    output is the sibling ``ceviriler`` directory; otherwise the Turkish output
    is a sibling ``ceviriler`` directory next to the custom path. This keeps the
    two document sets side by side without ever writing into the source folder.
    """

    resolved = Path(source_dir).expanduser()
    return resolved.parent / "ceviriler"


#: Standard OS symlink aliases (macOS ``/var`` -> ``private/var`` and friends)
#: that are stable, OS-provided paths rather than an operator/attacker redirect.
#: They are allowed so an ordinary system temp directory keeps working.
_SYSTEM_SYMLINK_TARGETS: Final[dict[str, frozenset[str]]] = {
    "darwin": frozenset({"/private/var", "/private/tmp", "/private/etc"}),
}


def _is_standard_system_alias(link: Path) -> bool:
    allowed = _SYSTEM_SYMLINK_TARGETS.get(sys.platform, frozenset())
    if not allowed:
        return False
    try:
        target = os.readlink(link)
    except OSError:
        return False
    if not os.path.isabs(target):
        target = os.path.join(str(link.parent), target)
    return os.path.normpath(target) in allowed


def unsafe_symlink_ancestor(path: str | Path) -> Path | None:
    """Return the first symlinked component of ``path`` that is not a system alias.

    ``path`` itself and every existing parent component are inspected *without*
    following the link. Standard OS temp aliases (e.g. macOS ``/var`` ->
    ``/private/var``) are accepted so an ordinary temp directory keeps working;
    any other symlinked component is returned so the caller can refuse to write
    through it. ``None`` means the path has no unsafe symlinked component.
    """

    absolute = Path(os.path.abspath(os.fspath(path)))
    for candidate in (absolute, *absolute.parents):
        if candidate == candidate.parent:
            continue
        if not os.path.lexists(candidate):
            continue
        if os.path.islink(candidate) and not _is_standard_system_alias(candidate):
            return candidate
    return None
