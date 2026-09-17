"""Cloud-neutral full-text Turkish MT artifacts and deterministic chunking.

The desktop chain produces two human-facing Markdown documents from a single
full-text pass plus a *separate* timed pass. This module owns the provider-neutral
full-text artifact contract and the deterministic, contiguous chunking shared by
the route:

* the exact source text is split into deterministic, contiguous chunks with
  explicit source character spans (``start_char``/``end_char``) and stable chunk
  ids; the concatenation of the chunks reproduces the input byte-for-byte;
* the accepted output, the source SHA-256/language, per-stage durations and
  completeness flags are persisted in the job directory;
* the Turkish Markdown is rendered and published atomically with no-clobber
  archiving of a differing earlier document.

No timestamp is ever fabricated; this module only chunks text and persists the
accepted cloud artifact.
"""

from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from subtitle_flow.schemas import (
    SCHEMA_VERSION,
    RawArtifactRef,
    UtcDatetime,
)
from subtitle_flow.transcript_markdown import (
    MarkdownExportError,
    SourceTranscript,
    archive_markdown_document,
    write_markdown_document,
)

__all__ = [
    "FULLTEXT_MT_VERSION",
    "FullTextChunk",
    "FullTextChunkRecord",
    "FullTextMTError",
    "FullTextTranslationArtifact",
    "chunk_text_with_budget",
    "render_translation_markdown",
    "write_translation_markdown",
]

#: Format version of the full-text MT artifact. A change makes old artifacts
#: non-reusable (they are regenerated, never silently reinterpreted).
FULLTEXT_MT_VERSION: Final[str] = "1"

_TARGET_LANGUAGE: Final[str] = "tr"

_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Sentence/paragraph boundaries a chunk may start after, so chunks follow the
#: text's own structure instead of an arbitrary character count.
_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?…])\s+|\n\s*\n")

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class FullTextMTError(Exception):
    """A typed failure to chunk, translate or persist a full-text translation."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class FullTextChunk:
    """One deterministic, contiguous slice of the exact source full text."""

    chunk_id: str
    start_char: int
    end_char: int
    text: str


class FullTextChunkRecord(BaseModel):
    """Persisted chunk: exact source slice, accepted output and raw evidence."""

    model_config = _FROZEN

    chunk_id: str = Field(min_length=1)
    start_char: int = Field(strict=True, ge=0)
    end_char: int = Field(strict=True, ge=0)
    source_text: str
    translated_text_tr: str | None = None
    flags: tuple[str, ...] = ()
    raw_reference: RawArtifactRef | None = None


class FullTextTranslationArtifact(BaseModel):
    """Durable, verifiable full-text Turkish translation for one job source."""

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    fulltext_mt_version: str = FULLTEXT_MT_VERSION
    kind: str = "fulltext"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    target_language: str = _TARGET_LANGUAGE
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_char_count: int = Field(strict=True, ge=0)
    status: str = Field(min_length=1)
    needs_review: bool = False
    model_identity: dict[str, Any] = Field(default_factory=dict)
    #: Exact output-affecting generation settings (route/batch/quality). A change
    #: makes the stored output non-reusable so a changed request budget or quality
    #: threshold can never silently reuse an incompatible translation.
    generation_settings: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int | None = Field(default=None, strict=True, ge=0)
    chunks: tuple[FullTextChunkRecord, ...] = ()
    created_at_utc: UtcDatetime

    @model_validator(mode="after")
    def _check_contract(self) -> "FullTextTranslationArtifact":
        if self.kind != "fulltext":
            raise ValueError("full-text artifact kind must be 'fulltext'")
        if self.target_language != _TARGET_LANGUAGE:
            raise ValueError("full-text artifact target language must be Turkish")
        if self.status not in {"translated", "skipped_same_language"}:
            raise ValueError("full-text artifact status is not a terminal success")
        if self.status == "skipped_same_language":
            if self.source_language != _TARGET_LANGUAGE:
                raise ValueError(
                    "a same-language full-text artifact must have a Turkish source"
                )
        if not self.chunks:
            raise ValueError("full-text artifact must carry at least one chunk")
        cursor = 0
        ids: set[str] = set()
        for chunk in self.chunks:
            if chunk.chunk_id in ids:
                raise ValueError("full-text chunk ids must be unique")
            ids.add(chunk.chunk_id)
            if chunk.start_char != cursor or chunk.end_char <= chunk.start_char:
                raise ValueError(
                    "full-text chunks must be contiguous and strictly increasing"
                )
            if chunk.translated_text_tr is None:
                raise ValueError("a terminal full-text chunk must carry its output")
            cursor = chunk.end_char
        if cursor != self.source_char_count:
            raise ValueError(
                "full-text chunks must cover the entire source character span"
            )
        return self


# --------------------------------------------------------------------------- #
# Deterministic chunking
# --------------------------------------------------------------------------- #
def _candidate_starts(text: str) -> list[int]:
    starts = {0}
    for match in _BOUNDARY_RE.finditer(text):
        starts.add(match.end())
    starts.discard(len(text))
    return sorted(starts)


def _fits(text: str, char_budget: int, byte_budget: int) -> bool:
    return len(text) <= char_budget and len(text.encode("utf-8")) <= byte_budget


def _split_at_whitespace(
    text: str, cursor: int, char_budget: int, byte_budget: int
) -> int | None:
    """Return the largest whitespace boundary that keeps the chunk within budget."""

    limit = min(len(text), cursor + char_budget)
    # Trim further while the UTF-8 byte budget is exceeded.
    while limit > cursor and len(text[cursor:limit].encode("utf-8")) > byte_budget:
        limit -= 1
    if limit <= cursor:
        return None
    index = limit
    while index > cursor and not text[index - 1].isspace():
        index -= 1
    if index <= cursor:
        return None
    return index


def chunk_text_with_budget(
    text: str, *, char_budget: int, byte_budget: int
) -> tuple[FullTextChunk, ...]:
    """Split exact text into contiguous, budget-bounded chunks (route-agnostic).

    The concatenation of ``chunk.text`` reproduces the input byte-for-byte and
    the spans are half-open ``[start_char, end_char)`` character offsets. A
    single token that cannot fit is a typed refusal; nothing is truncated or
    split mid-word.
    """

    if not isinstance(text, str) or text == "":
        raise FullTextMTError(
            "FULLTEXT_MT_EMPTY", "the recovered source text is empty"
        )
    if char_budget <= 0 or byte_budget <= 0:
        raise FullTextMTError(
            "FULLTEXT_MT_BUDGET_INVALID",
            "the full-text request budget cannot hold the fixed template overhead",
        )

    total = len(text)
    starts = _candidate_starts(text)
    chunks: list[FullTextChunk] = []
    cursor = 0
    index = 0
    while cursor < total:
        remaining = text[cursor:]
        if _fits(remaining, char_budget, byte_budget):
            end = total
        else:
            position = bisect_right(starts, cursor)
            best: int | None = None
            while position < len(starts):
                candidate = starts[position]
                if candidate - cursor > char_budget:
                    break
                if len(text[cursor:candidate].encode("utf-8")) > byte_budget:
                    break
                best = candidate
                position += 1
            if best is None:
                best = _split_at_whitespace(
                    text, cursor, char_budget, byte_budget
                )
            if best is None or best <= cursor:
                raise FullTextMTError(
                    "FULLTEXT_MT_INPUT_TOO_LONG",
                    "a single indivisible token exceeds the MT request budget; "
                    "refusing to truncate or split it",
                )
            end = best
        chunks.append(
            FullTextChunk(
                chunk_id=f"chunk_{index:04d}",
                start_char=cursor,
                end_char=end,
                text=text[cursor:end],
            )
        )
        cursor = end
        index += 1
    return tuple(chunks)


# --------------------------------------------------------------------------- #
# Rendering / publishing
# --------------------------------------------------------------------------- #
def _language_label(code: str) -> str:
    names = {
        "en": "İngilizce",
        "ar": "Arapça",
        "fa": "Farsça",
        "ru": "Rusça",
        "de": "Almanca",
        "he": "İbranice",
        "tr": "Türkçe",
    }
    name = names.get(code)
    return f"{name} (`{code}`)" if name else f"`{code}`"


def render_translation_markdown(
    source: SourceTranscript, artifact: FullTextTranslationArtifact
) -> str:
    """Render the Turkish document for one verified full-text translation."""

    title = (source.title or "").strip()
    if title == "":
        title = (
            f"YouTube video {source.video_id}"
            if source.video_id
            else "Türkçe çeviri"
        )
    title = " ".join(title.replace("\r", " ").replace("\n", " ").split())
    lines = [f"# {title} — Türkçe çeviri", ""]
    if source.source_url:
        lines.append(f"**Kaynak:** {source.source_url}")
    lines.append(
        f"**Diller:** {_language_label(artifact.source_language)} → Türkçe"
    )
    lines.append(f"**Model:** {artifact.provider} / {artifact.model}")
    if artifact.provider == "google-basic":
        lines.append(
            "**Not:** Bu metin Google Translation LLM (Basic v2 API) ile "
            "üretilmiştir; insan incelemesi ve dil kalitesi değerlendirmesi "
            "ayrıdır."
        )
    else:
        lines.append(
            "**Not:** Bu metin makine çevirisidir; insan incelemesi ve dil "
            "kalitesi değerlendirmesi ayrıdır."
        )
    if artifact.needs_review:
        lines.append(
            "**İnceleme:** Otomatik kalite kontrolleri bazı bölümleri "
            "işaretledi; anlam doğruluğu iddia edilmez."
        )
    lines.append("")
    body = "\n\n".join(
        chunk.translated_text_tr.strip()
        for chunk in artifact.chunks
        if chunk.translated_text_tr and chunk.translated_text_tr.strip()
    )
    lines.append(body)
    return "\n".join(lines) + "\n"


def write_translation_markdown(
    source: SourceTranscript,
    artifact: FullTextTranslationArtifact,
    *,
    translation_dir: str | Path | None = None,
    fallback_name: str,
    overwrite: bool = False,
) -> Path:
    """Publish the Turkish Markdown atomically without clobbering a differing file.

    ``translation_dir`` defaults to the sibling ``ceviriler`` directory of the
    default ``outputs/transkriptler`` source folder.

    The default ``<id>.md`` is the current run's translation, but it is never
    clobbered blindly. A differing existing document is first archived under
    ``_arsiv/`` with a collision-safe content-hash + model name, then the new,
    complete document is published atomically with ``overwrite``. A failure
    before this point never reaches the publisher, so a failed new run always
    leaves the earlier document intact.
    """

    if translation_dir is None:
        from subtitle_flow.output_paths import default_translation_dir

        directory: str | Path = default_translation_dir()
    else:
        directory = translation_dir
    name = source.video_id or fallback_name
    if not isinstance(name, str) or _SAFE_NAME_RE.match(name) is None:
        raise MarkdownExportError(
            "TRANSLATION_NAME_INVALID",
            "the translation file name is not a safe identifier",
        )
    payload = render_translation_markdown(source, artifact).encode("utf-8")
    target = Path(str(directory)).expanduser() / f"{name}.md"
    if target.exists() and not target.is_symlink():
        try:
            existing = target.read_bytes()
        except OSError as exc:
            raise MarkdownExportError(
                "TRANSLATION_TARGET_UNREADABLE",
                f"cannot read the existing Markdown file: {exc}",
            ) from exc
        if existing != payload:
            # Preserve the differing earlier output before publishing this run's
            # document as the new default. The archive is itself no-clobber, so
            # an even earlier archive with the same content+model is reused and a
            # genuinely different one is never overwritten.
            archive_markdown_document(
                directory=directory, name=name, content=existing
            )
            overwrite = True
    return write_markdown_document(
        directory=directory,
        name=name,
        payload=payload,
        overwrite=overwrite,
        name_code="TRANSLATION_NAME_INVALID",
        dir_unsafe_code="TRANSLATION_DIR_UNSAFE",
        dir_invalid_code="TRANSLATION_DIR_INVALID",
        target_unsafe_code="TRANSLATION_TARGET_UNSAFE",
        target_unreadable_code="TRANSLATION_TARGET_UNREADABLE",
        exists_code="TRANSLATION_EXISTS",
        write_code="TRANSLATION_WRITE_FAILED",
    )
