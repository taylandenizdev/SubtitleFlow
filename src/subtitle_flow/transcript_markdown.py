"""Standalone full-text Markdown export for a source-language transcript.

A standalone transcription (``transcribe-youtube``) can succeed even when the
optional timed alignment cannot be produced: the provider's verified archived
response still carries the complete source wording. This module recovers that
full text from durable, already-verified evidence -- the append-only raw archive
first, then the committed canonical transcript -- and writes one Markdown file
per video. It never calls a provider, downloads media or fabricates segments.

Only exact, consistent wording is accepted. A malformed, partial, wrong-language
or tampered archived body is refused, never turned into a false success.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from subtitle_flow.exports import ExportError, _read_verified_transcript
from subtitle_flow.languages import normalize_language
from subtitle_flow.output_paths import unsafe_symlink_ancestor
from subtitle_flow.schemas import RawArtifactRef, StageKind, Transcript
from subtitle_flow.storage import (
    ApiTraceKind,
    ApiTraceRecord,
    JobStore,
    StorageCorruptionError,
    StoredInput,
)

__all__ = [
    "ARCHIVE_DIRNAME",
    "MarkdownExportError",
    "SourceTranscript",
    "archive_markdown_document",
    "detect_document_model",
    "render_markdown",
    "recover_source_transcript",
    "slugify_identifier",
    "write_markdown_document",
    "write_transcript_markdown",
    "TIMED_TRANSCRIPT_UNAVAILABLE",
    "TRANSCRIPT_NOT_RECOVERABLE",
    "recover_timed_transcript",
]

#: Recovery found no usable verified evidence (as opposed to corrupt evidence).
TRANSCRIPT_NOT_RECOVERABLE: Final[str] = "TRANSCRIPT_NOT_RECOVERABLE"

#: Subdirectory that preserves a superseded document when the canonical
#: ``<id>.md`` is republished by a different model. The library scan ignores
#: directories, so an archived document never reappears as a current entry.
ARCHIVE_DIRNAME: Final[str] = "_arsiv"

#: Matches the ``**Model:** provider / model`` header the renderers write.
_MODEL_HEADER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\*\*Model:\*\*\s*(\S+)\s*/\s*(\S+)\s*$", re.MULTILINE
)

_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: Terminal STT trace outcomes that may authorize presenting an archived body as
#: a verified full text. ``complete`` is a proven HTTP 200 success; an
#: ``invalid_response`` is accepted only for the timestamp-normalization codes
#: below, after the body itself is independently proven coherent.
_ACCEPTED_TRACE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"complete", "invalid_response"}
)

#: The only post-HTTP-200 failure codes whose body may still carry the complete
#: spoken wording: the segmentation/normalization step failed, not the provider.
_SEGMENTATION_REVIEW_CODES: Final[frozenset[str]] = frozenset(
    {"STT_SEGMENTATION_INVALID", "STT_SEGMENTATION_NEEDS_REVIEW"}
)

_LANGUAGE_NAMES: Final[dict[str, str]] = {
    "en": "İngilizce",
    "ar": "Arapça",
    "fa": "Farsça",
    "ru": "Rusça",
    "de": "Almanca",
    "he": "İbranice",
    "tr": "Türkçe",
}


class MarkdownExportError(Exception):
    """A typed failure to produce or safely persist the full-text Markdown."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class SourceTranscript:
    """A verified source-language full text plus its bound provenance."""

    full_text: str
    source_language: str
    provider: str
    model: str
    title: str | None
    source_url: str | None
    video_id: str | None
    recovered_from: str


def _language_label(code: str) -> str:
    name = _LANGUAGE_NAMES.get(code)
    return f"{name} (`{code}`)" if name else f"`{code}`"


def render_markdown(source: SourceTranscript) -> str:
    """Render the friendly Markdown document for one source transcript."""

    title = (source.title or "").strip()
    if title == "":
        title = (
            f"YouTube video {source.video_id}"
            if source.video_id
            else "Kaynak transkript"
        )
    title = " ".join(title.replace("\r", " ").replace("\n", " ").split())
    lines = [
        f"# {title}",
        "",
    ]
    if source.source_url:
        lines.append(f"**Kaynak:** {source.source_url}")
    lines.append(f"**Dil:** {_language_label(source.source_language)}")
    lines.append(f"**Model:** {source.provider} / {source.model}")
    lines.extend(["", source.full_text])
    return "\n".join(lines) + "\n"


def _finite_probability(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _source_from_scribe_body(
    data: bytes, stored: StoredInput
) -> SourceTranscript | None:
    """Extract the exact full text from a registered Scribe response body.

    Returns ``None`` for any body that is not a complete, internally consistent
    Scribe success; it never repairs, truncates or relabels the wording.
    """

    try:
        payload = json.loads(data)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    if "transcripts" in payload:
        return None
    words = payload.get("words")
    text = payload.get("text")
    if not isinstance(words, list) or not isinstance(text, str):
        return None
    if text.strip() == "":
        return None

    parts: list[str] = []
    has_spoken_word = False
    for token in words:
        if not isinstance(token, Mapping):
            return None
        token_text = token.get("text")
        if not isinstance(token_text, str):
            return None
        parts.append(token_text)
        if token.get("type") == "word" and token_text.strip() != "":
            has_spoken_word = True
    if not has_spoken_word:
        return None
    if "".join(parts) != text:
        return None

    raw_code = payload.get("language_code")
    normalized = normalize_language(raw_code if isinstance(raw_code, str) else None)
    canonical = normalized.canonical_code
    if canonical is None or normalized.uncertain:
        return None

    probability = _finite_probability(payload.get("language_probability"))
    minimum = stored.config.api.scribe.min_language_probability
    if probability is None or probability < minimum:
        return None

    hint = stored.config.source_language_hint
    if hint is not None and hint != canonical:
        return None

    origin = stored.config.youtube_origin
    return SourceTranscript(
        full_text=text,
        source_language=canonical,
        provider=stored.config.stt.provider,
        model=stored.config.stt.model,
        title=origin.title if origin is not None else None,
        source_url=origin.canonical_url if origin is not None else None,
        video_id=origin.video_id if origin is not None else None,
        recovered_from="archived_response",
    )


def _authorizing_trace(
    store: JobStore,
    reference: RawArtifactRef,
    stored: StoredInput,
    *,
    attempt: int | None = None,
) -> ApiTraceRecord | None:
    """Return the durable STT trace that authorizes this archived raw body.

    An archived body is only presentable as a verified full text when the
    append-only API trace proves a terminal HTTP 200 outcome for the exact same
    raw reference (locator + hash + request id), for the configured provider and
    model, with no segment group, and bound to the dispatch attempt. A success is
    accepted as-is; a post-200 ``invalid_response`` is accepted only for the
    timestamp-normalization codes (the provider body is complete, only the
    optional point timing failed). Anything else -- a non-200 status, a
    ``remote_unknown``, an unledgered body or a model/provider mismatch -- never
    authorizes a body. The caller validates the body's wording and language.
    """

    provider = stored.config.stt.provider
    model = stored.config.stt.model
    for record in reversed(store.read_api_trace(stage=StageKind.stt)):
        if record.kind is not ApiTraceKind.outcome:
            continue
        if record.group_id is not None:
            continue
        if record.http_status != 200:
            continue
        if record.provider != provider or record.model != model:
            continue
        if record.outcome not in _ACCEPTED_TRACE_OUTCOMES:
            continue
        if (
            record.outcome == "invalid_response"
            and record.error_code not in _SEGMENTATION_REVIEW_CODES
        ):
            continue
        if record.pipeline_attempt is None:
            continue
        if attempt is not None and record.pipeline_attempt != attempt:
            continue
        bound = record.raw_reference
        if (
            bound is None
            or bound.locator != reference.locator
            or bound.sha256 != reference.sha256
            or bound.kind != reference.kind
            or bound.request_id != reference.request_id
        ):
            continue
        return record
    return None


def _recover_from_scribe_raw(
    store: JobStore, stored: StoredInput
) -> SourceTranscript | None:
    if stored.config.stt.provider != "elevenlabs":
        return None
    candidates = [
        record
        for record in store.raw_registrations(stage=StageKind.stt)
        if record.content_subtype == "json"
    ]
    for registration in reversed(candidates):
        reference = RawArtifactRef(
            kind=registration.kind,
            locator=registration.locator,
            sha256=registration.sha256,
            request_id=registration.request_id,
        )
        if _authorizing_trace(
            store, reference, stored, attempt=registration.attempt
        ) is None:
            # No durable terminal 200 trace binds this body: it is not evidence
            # of a successful STT response and is never presented as one.
            continue
        try:
            data = store.verify_raw(
                reference,
                stage=StageKind.stt,
                group_id=None,
                attempt=registration.attempt,
            )
        except StorageCorruptionError as exc:
            raise MarkdownExportError(
                "TRANSCRIPT_EVIDENCE_CORRUPT",
                f"archived STT response is not usable: {exc}",
            ) from exc
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not isinstance(payload, dict) or "transcripts" in payload:
            continue
        if not isinstance(payload.get("words"), list) or not isinstance(
            payload.get("text"), str
        ):
            continue
        source = _source_from_scribe_body(data, stored)
        if source is None:
            raise MarkdownExportError(
                "TRANSCRIPT_EVIDENCE_CORRUPT",
                "the archived Scribe response is internally inconsistent or "
                "cannot be presented as a verified full text",
            )
        return source
    return None


def _canonical_raw_is_authorized(
    store: JobStore, reference: RawArtifactRef, stored: StoredInput
) -> bool:
    """Whether the committed transcript's raw body may be read as full text.

    A local-route body has no HTTP trace and is unaffected. For the paid Scribe
    route the committed manifest already proves a successful normalization, but
    the raw *body* is still only branchable when the append-only API trace binds
    it to a terminal HTTP 200 outcome; otherwise the caller falls back to the
    committed segments, so a mismatched or legacy raw can never bypass the
    status/ledger gate applied to direct archive recovery.
    """

    if stored.config.stt.provider != "elevenlabs":
        return True
    return _authorizing_trace(store, reference, stored) is not None


def _recover_from_canonical_transcript(
    store: JobStore, stored: StoredInput
) -> SourceTranscript | None:
    manifest = store.read_stt_manifest()
    if manifest is None:
        return None
    try:
        transcript = _read_verified_transcript(store, manifest, stored)
    except (ExportError, StorageCorruptionError) as exc:
        raise MarkdownExportError(
            "TRANSCRIPT_EVIDENCE_CORRUPT",
            f"committed transcript evidence is not usable: {exc}",
        ) from exc

    if transcript.raw_reference is not None and _canonical_raw_is_authorized(
        store, transcript.raw_reference, stored
    ):
        try:
            data = store.verify_raw(
                transcript.raw_reference,
                stage=StageKind.stt,
                group_id=None,
                attempt=None,
            )
        except StorageCorruptionError as exc:
            raise MarkdownExportError(
                "TRANSCRIPT_EVIDENCE_CORRUPT",
                f"committed transcript raw body is not usable: {exc}",
            ) from exc
        source = _source_from_scribe_body(data, stored)
        if source is not None:
            return source

    joined = "".join(segment.source_text for segment in transcript.segments)
    if joined.strip() == "":
        candidate = transcript.request_metadata.get("provider_full_text")
        if isinstance(candidate, str):
            joined = candidate
    if joined.strip() == "":
        return None
    canonical = transcript.source_language
    if canonical is None or transcript.language_uncertain:
        return None
    hint = stored.config.source_language_hint
    if hint is not None and hint != canonical:
        return None
    origin = stored.config.youtube_origin
    return SourceTranscript(
        full_text=joined,
        source_language=canonical,
        provider=stored.config.stt.provider,
        model=stored.config.stt.model,
        title=origin.title if origin is not None else None,
        source_url=origin.canonical_url if origin is not None else None,
        video_id=origin.video_id if origin is not None else None,
        recovered_from="canonical_transcript",
    )


#: The job has no valid canonical *timed* transcript (only a full text, or
#: nothing). A burned-in subtitle video cannot be produced from it.
TIMED_TRANSCRIPT_UNAVAILABLE: Final[str] = "TIMED_TRANSCRIPT_UNAVAILABLE"


def recover_timed_transcript(store: JobStore, stored: StoredInput) -> Transcript:
    """Return the verified canonical timed transcript for one job.

    Unlike a full-text recovery, this requires the committed STT manifest and a
    re-verified transcript artifact with at least one real timed segment. A
    point-timestamp ``needs_review`` job, a full-text-only recovery or a job
    without an STT manifest raises a typed refusal instead of inventing timings.
    """

    manifest = store.read_stt_manifest()
    if manifest is None:
        raise MarkdownExportError(
            TIMED_TRANSCRIPT_UNAVAILABLE,
            "no committed STT manifest exists for this job",
        )
    try:
        transcript = _read_verified_transcript(store, manifest, stored)
    except (ExportError, StorageCorruptionError) as exc:
        raise MarkdownExportError(
            TIMED_TRANSCRIPT_UNAVAILABLE,
            f"the committed timed transcript is not usable: {exc}",
        ) from exc
    if not transcript.segments:
        raise MarkdownExportError(
            TIMED_TRANSCRIPT_UNAVAILABLE,
            "the committed transcript carries no timed segments",
        )
    return transcript


def recover_source_transcript(
    store: JobStore,
    stored: StoredInput,
    *,
    requested_hint: str | None = None,
) -> SourceTranscript:
    """Recover the verified source-language full text for one job.

    Raises :class:`MarkdownExportError` with ``TRANSCRIPT_NOT_RECOVERABLE`` when
    no usable evidence exists, and with ``TRANSCRIPT_LANGUAGE_MISMATCH`` when the
    recovered language contradicts an explicit caller hint. Corrupt or tampered
    registered evidence fails closed instead of being silently skipped.
    """

    source = _recover_from_scribe_raw(store, stored)
    if source is None:
        source = _recover_from_canonical_transcript(store, stored)
    if source is None:
        raise MarkdownExportError(
            TRANSCRIPT_NOT_RECOVERABLE,
            "no verified source transcript is available for this job",
        )
    if requested_hint is not None and requested_hint != source.source_language:
        raise MarkdownExportError(
            "TRANSCRIPT_LANGUAGE_MISMATCH",
            "the recovered source language does not match the requested language "
            "hint; refusing to relabel the transcript",
        )
    return source


def write_markdown_document(
    *,
    directory: str | Path,
    name: str,
    payload: bytes,
    overwrite: bool = False,
    name_code: str = "TRANSCRIPT_NAME_INVALID",
    dir_unsafe_code: str = "TRANSCRIPT_DIR_UNSAFE",
    dir_invalid_code: str = "TRANSCRIPT_DIR_INVALID",
    target_unsafe_code: str = "TRANSCRIPT_TARGET_UNSAFE",
    target_unreadable_code: str = "TRANSCRIPT_TARGET_UNREADABLE",
    exists_code: str = "TRANSCRIPT_EXISTS",
    write_code: str = "TRANSCRIPT_WRITE_FAILED",
) -> Path:
    """Write a Markdown document safely, atomically and idempotently.

    The file name is a validated identifier. An existing file with identical
    content is a no-op; an existing file with different content is refused
    unless ``overwrite`` is set. ``*_code`` lets the source and Turkish writers
    keep their own typed error codes while sharing this single audited path.
    """

    if not isinstance(name, str) or _SAFE_NAME_RE.match(name) is None:
        raise MarkdownExportError(
            name_code, "the Markdown file name is not a safe identifier"
        )

    base = Path(directory).expanduser()
    unsafe = unsafe_symlink_ancestor(base)
    if unsafe is not None:
        raise MarkdownExportError(
            dir_unsafe_code,
            "the Markdown directory is below a symlinked path component and is "
            f"never written through: {unsafe}",
        )
    if base.exists() and base.is_symlink():
        raise MarkdownExportError(
            dir_unsafe_code, "the Markdown directory is a symlink"
        )
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MarkdownExportError(
            dir_invalid_code, f"cannot create the Markdown directory: {exc}"
        ) from exc
    if base.is_symlink() or not base.is_dir():
        raise MarkdownExportError(
            dir_unsafe_code, "the Markdown directory is not a real directory"
        )

    target = base / f"{name}.md"
    if target.is_symlink():
        raise MarkdownExportError(
            target_unsafe_code, "the Markdown target is a symlink"
        )

    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError as exc:
            raise MarkdownExportError(
                target_unreadable_code,
                f"cannot read the existing Markdown file: {exc}",
            ) from exc
        if existing == payload:
            return target
        if not overwrite:
            raise MarkdownExportError(
                exists_code,
                "a different Markdown file already exists; pass the explicit "
                "overwrite option to replace it",
            )

    handle = None
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".md.tmp", dir=str(base)
        )
        handle = os.fdopen(descriptor, "wb")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        if overwrite:
            os.replace(temporary, target)
            temporary = None
        else:
            # Atomic no-clobber publish: the fully written, fsynced temp file is
            # hard-linked into place, which fails with ``FileExistsError`` if any
            # concurrent writer won the race. The temp file is never truncated
            # directly at the target, so a partial body is never observable.
            try:
                os.link(temporary, target)
            except FileExistsError:
                os.unlink(temporary)
                temporary = None
                try:
                    existing = target.read_bytes()
                except OSError as exc:
                    raise MarkdownExportError(
                        target_unreadable_code,
                        f"cannot read the existing Markdown file: {exc}",
                    ) from exc
                if existing == payload:
                    return target
                raise MarkdownExportError(
                    exists_code,
                    "a different Markdown file already exists; pass the explicit "
                    "overwrite option to replace it",
                )
            else:
                os.unlink(temporary)
                temporary = None
    except OSError as exc:
        raise MarkdownExportError(
            write_code, f"cannot write the Markdown document: {exc}"
        ) from exc
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return target


def detect_document_model(content: bytes) -> str | None:
    """Return the model tag recorded in a rendered document's header, if any.

    Only the exact ``**Model:** provider / model`` line is read; a document
    without it (or with undecodable bytes) yields ``None`` and is archived under
    an ``unknown`` slug rather than guessed.
    """

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None
    match = _MODEL_HEADER_RE.search(text)
    if match is None:
        return None
    return match.group(2)


def slugify_identifier(value: str) -> str:
    """Return a filename-safe slug for an arbitrary model/identifier string."""

    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return slug or "unknown"


def archive_markdown_document(
    *,
    directory: str | Path,
    name: str,
    content: bytes,
    name_code: str = "TRANSLATION_ARCHIVE_NAME_INVALID",
    dir_unsafe_code: str = "TRANSLATION_ARCHIVE_DIR_UNSAFE",
    dir_invalid_code: str = "TRANSLATION_ARCHIVE_DIR_INVALID",
    target_unsafe_code: str = "TRANSLATION_ARCHIVE_TARGET_UNSAFE",
    target_unreadable_code: str = "TRANSLATION_ARCHIVE_TARGET_UNREADABLE",
    exists_code: str = "TRANSLATION_ARCHIVE_COLLISION",
    write_code: str = "TRANSLATION_ARCHIVE_WRITE_FAILED",
) -> Path:
    """Preserve a superseded document under ``_arsiv`` without clobbering.

    The archive file name binds the *exact content hash* and the document's own
    model tag, so re-archiving identical content is a no-op while a genuinely
    different earlier archive is never overwritten (it raises a typed collision
    instead). The atomic, no-clobber writer is reused so a partial archive is
    never observable. ``*_code`` lets the source and Turkish writers keep their
    own typed error codes while sharing this single audited archive path.
    """

    base = Path(directory).expanduser()
    archive_dir = base / ARCHIVE_DIRNAME
    model = detect_document_model(content)
    digest = hashlib.sha256(content).hexdigest()[:16]
    # Digest first so the uniqueness key survives the length cap; the video/id
    # name is only a trailing convenience.
    archive_name = f"{digest}_{slugify_identifier(model or 'unknown')}_{name}"
    if len(archive_name) > 64:
        archive_name = archive_name[:64]
    return write_markdown_document(
        directory=archive_dir,
        name=archive_name,
        payload=content,
        overwrite=False,
        name_code=name_code,
        dir_unsafe_code=dir_unsafe_code,
        dir_invalid_code=dir_invalid_code,
        target_unsafe_code=target_unsafe_code,
        target_unreadable_code=target_unreadable_code,
        exists_code=exists_code,
        write_code=write_code,
    )


def write_transcript_markdown(
    source: SourceTranscript,
    *,
    transcript_dir: str | Path,
    fallback_name: str,
    overwrite: bool = False,
) -> Path:
    """Write the source Markdown document safely, atomically and idempotently.

    The file name is the validated video id (or the validated job id fallback).

    The default ``<id>.md`` is never clobbered. With the explicit ``overwrite``
    opt-in a differing earlier source document is first archived under
    ``_arsiv/`` with a collision-safe content-hash + model name, then the new,
    complete document is published atomically. An identical payload stays a
    no-op and is never archived, and an archive failure leaves the canonical old
    source untouched because the replacement only follows a successful archive.
    """

    name = source.video_id or fallback_name
    if not isinstance(name, str) or _SAFE_NAME_RE.match(name) is None:
        raise MarkdownExportError(
            "TRANSCRIPT_NAME_INVALID",
            "the Markdown file name is not a safe identifier",
        )
    payload = render_markdown(source).encode("utf-8")
    if overwrite:
        target = Path(str(transcript_dir)).expanduser() / f"{name}.md"
        if target.exists() and not target.is_symlink():
            try:
                existing = target.read_bytes()
            except OSError as exc:
                raise MarkdownExportError(
                    "TRANSCRIPT_TARGET_UNREADABLE",
                    f"cannot read the existing Markdown file: {exc}",
                ) from exc
            if existing != payload:
                # Preserve the differing earlier source before publishing this
                # run's document as the new default. The archive is itself
                # no-clobber, so an even earlier archive with the same
                # content+model is reused and a genuinely different one is never
                # overwritten.
                archive_markdown_document(
                    directory=transcript_dir,
                    name=name,
                    content=existing,
                    name_code="TRANSCRIPT_ARCHIVE_NAME_INVALID",
                    dir_unsafe_code="TRANSCRIPT_ARCHIVE_DIR_UNSAFE",
                    dir_invalid_code="TRANSCRIPT_ARCHIVE_DIR_INVALID",
                    target_unsafe_code="TRANSCRIPT_ARCHIVE_TARGET_UNSAFE",
                    target_unreadable_code="TRANSCRIPT_ARCHIVE_TARGET_UNREADABLE",
                    exists_code="TRANSCRIPT_ARCHIVE_COLLISION",
                    write_code="TRANSCRIPT_ARCHIVE_WRITE_FAILED",
                )
    return write_markdown_document(
        directory=transcript_dir,
        name=name,
        payload=payload,
        overwrite=overwrite,
    )
