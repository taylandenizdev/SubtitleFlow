"""Cloud-neutral timed subtitle artifact for burned-in Turkish subtitles.

The desktop full chain already produces the two human-facing Markdown documents.
A burned-in Turkish video needs a *different* translation artifact: one that keeps
each cue's real ``segment_id`` and the exact ``start_ms``/``end_ms`` of the
canonical source transcript segment it came from. This module owns exactly that
provider-neutral artifact contract and its validation:

* the exact canonical timed :class:`~subtitle_flow.schemas.Transcript` is the
  only accepted input; when no valid timed transcript exists (for example a
  Scribe point-timestamp ``needs_review`` job whose only recovery is the full
  text) a typed :class:`TimedSubtitleError` is raised so the caller can surface a
  review result instead of inventing timings;
* the accepted output, the exact cue times, quality/review flags and the source
  transcript identity are persisted together with an integrity hash over the
  whole record.

It never approximates, fabricates, proportionally distributes, clamps or infers a
timestamp, and it never changes the full-text Markdown translation. The Google
Translation Basic v2 route owns translation and evidence reuse; this module only
defines the shared artifact shape and validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from subtitle_flow.schemas import (
    SCHEMA_VERSION,
    RawArtifactRef,
    Transcript,
    UtcDatetime,
)

__all__ = [
    "TIMED_SUBTITLE_VERSION",
    "TimedCue",
    "TimedCueRecord",
    "TimedSubtitleArtifact",
    "TimedSubtitleError",
    "cues_from_artifact",
    "require_timed_transcript",
    "timed_transcript_sha256",
]

#: Format version of the timed subtitle artifact. A change makes old artifacts
#: non-reusable (they are regenerated, never silently reinterpreted). Version 2
#: binds the transcript's ``language_uncertain`` state into the source identity,
#: so a language-confidence change invalidates prior evidence.
TIMED_SUBTITLE_VERSION: Final[str] = "2"

_TARGET_LANGUAGE: Final[str] = "tr"

#: Languages the product accepts as a source. An unknown source is refused with
#: a clear, non-dispatching error instead of a guessed translation.
_KNOWN_LANGUAGES: Final[frozenset[str]] = frozenset(
    {"en", "ar", "fa", "ru", "de", "he", "tr"}
)

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class TimedSubtitleError(Exception):
    """A typed failure to produce, reuse or persist a timed subtitle artifact."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def timed_transcript_sha256(transcript: Transcript) -> str:
    """Return the deterministic identity hash of one canonical timed transcript.

    Binds the provider/model provenance and every segment's exact id, real time
    range, language, source text and translation input. Unlike a full-text hash it
    includes the actual ``start_ms``/``end_ms`` values, so a transcript whose
    timings changed can never be confused with the one a stored artifact was
    produced from. The ``language_uncertain`` confidence flag is part of the
    identity as well: a transcript that later becomes (or stops being) uncertain
    can never reuse evidence produced under the other state.
    """

    payload = {
        "provider": transcript.provider,
        "model": transcript.model,
        "source_language": transcript.source_language,
        "provider_language_code": transcript.provider_language_code,
        "language_uncertain": transcript.language_uncertain,
        "audio_duration_ms": transcript.audio_duration_ms,
        "segments": [
            {
                "segment_id": segment.segment_id,
                "start_ms": segment.start_ms,
                "end_ms": segment.end_ms,
                "source_language": segment.source_language,
                "source_text": segment.source_text,
                "translation_input": segment.translation_input,
                "speaker": segment.speaker,
            }
            for segment in transcript.segments
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_timed_transcript(transcript: Transcript) -> str:
    """Validate that ``transcript`` is a usable canonical timed transcript."""

    if not transcript.segments:
        raise TimedSubtitleError(
            "TIMED_SUBTITLE_NO_SEGMENTS",
            "the canonical transcript carries no timed segments; a subtitle video "
            "cannot be produced without real segment timings",
        )
    canonical = transcript.source_language
    if canonical is None or transcript.language_uncertain:
        raise TimedSubtitleError(
            "TIMED_SUBTITLE_LANGUAGE_UNCERTAIN",
            "the canonical transcript language is unknown, uncertain or mixed; "
            "refusing to burn subtitles that cannot be safely labeled",
        )
    if canonical not in _KNOWN_LANGUAGES:
        raise TimedSubtitleError(
            "TIMED_SUBTITLE_UNSUPPORTED_LANGUAGE",
            f"the source language {canonical!r} is not supported; refusing to "
            "guess a translation",
        )
    for segment in transcript.segments:
        if segment.end_ms <= segment.start_ms:
            raise TimedSubtitleError(
                "TIMED_SUBTITLE_TIMING_INVALID",
                f"segment {segment.segment_id!r} has an invalid time range",
            )
    return canonical


@dataclass(frozen=True)
class TimedCue:
    """One timed cue: an exact segment id/time range plus its Turkish output."""

    segment_id: str
    start_ms: int
    end_ms: int
    source_text: str
    translation_input: str | None
    translated_text_tr: str
    flags: tuple[str, ...] = ()


class TimedCueRecord(BaseModel):
    """Persisted cue: exact source identity, real times, accepted output."""

    model_config = _FROZEN

    segment_id: str = Field(min_length=1)
    start_ms: int = Field(strict=True, ge=0)
    end_ms: int = Field(strict=True, gt=0)
    source_text: str
    translation_input: str | None = None
    translated_text_tr: str | None = None
    flags: tuple[str, ...] = ()
    raw_reference: RawArtifactRef | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "TimedCueRecord":
        if self.end_ms <= self.start_ms:
            raise ValueError("cue end_ms must be strictly greater than start_ms")
        return self


class TimedSubtitleArtifact(BaseModel):
    """Durable, verifiable timed Turkish translation for one canonical source."""

    model_config = _FROZEN

    schema_version: str = SCHEMA_VERSION
    timed_subtitle_version: str = TIMED_SUBTITLE_VERSION
    kind: str = "timed"
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_provider: str = Field(min_length=1)
    source_model: str = Field(min_length=1)
    source_language: str = Field(min_length=1)
    target_language: str = _TARGET_LANGUAGE
    source_transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_segment_count: int = Field(strict=True, ge=0)
    status: str = Field(min_length=1)
    needs_review: bool = False
    model_identity: dict[str, Any] = Field(default_factory=dict)
    generation_settings: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int | None = Field(default=None, strict=True, ge=0)
    cues: tuple[TimedCueRecord, ...] = ()
    created_at_utc: UtcDatetime
    #: Integrity hash over every other field (see :func:`_compute_integrity`).
    #: A tampered or truncated record fails validation instead of being reused.
    integrity_sha256: str = ""

    @model_validator(mode="after")
    def _check_contract(self) -> "TimedSubtitleArtifact":
        if self.kind != "timed":
            raise ValueError("timed subtitle artifact kind must be 'timed'")
        if self.target_language != _TARGET_LANGUAGE:
            raise ValueError("timed subtitle artifact target language must be Turkish")
        if self.status not in {"translated", "skipped_same_language"}:
            raise ValueError("timed subtitle artifact status is not a terminal success")
        if self.status == "skipped_same_language" and self.source_language != _TARGET_LANGUAGE:
            raise ValueError(
                "a same-language timed artifact must have a Turkish source"
            )
        if not self.cues:
            raise ValueError("timed subtitle artifact must carry at least one cue")
        if len(self.cues) != self.source_segment_count:
            raise ValueError(
                "timed subtitle artifact cue count must equal source_segment_count"
            )
        ids: set[str] = set()
        for cue in self.cues:
            if cue.segment_id in ids:
                raise ValueError("timed cue segment ids must be unique")
            ids.add(cue.segment_id)
            if cue.translated_text_tr is None:
                raise ValueError("a terminal timed cue must carry its output")
            if (
                self.status == "skipped_same_language"
                and cue.translated_text_tr != cue.source_text
            ):
                raise ValueError(
                    "a same-language timed cue must preserve the source text"
                )
        expected = _compute_integrity(self)
        if self.integrity_sha256 == "":
            object.__setattr__(self, "integrity_sha256", expected)
        elif self.integrity_sha256 != expected:
            raise ValueError(
                "the timed subtitle artifact integrity hash does not match its "
                "content; refusing a tampered or corrupt artifact"
            )
        return self


def _compute_integrity(artifact: "TimedSubtitleArtifact") -> str:
    """Return the canonical integrity hash over every non-hash field."""

    payload = artifact.model_dump(mode="json")
    payload.pop("integrity_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cues_from_artifact(artifact: TimedSubtitleArtifact) -> tuple[TimedCue, ...]:
    """Return the ordered render cues for a validated timed artifact."""

    return tuple(
        TimedCue(
            segment_id=record.segment_id,
            start_ms=record.start_ms,
            end_ms=record.end_ms,
            source_text=record.source_text,
            translation_input=record.translation_input,
            translated_text_tr=record.translated_text_tr or record.source_text,
            flags=record.flags,
        )
        for record in artifact.cues
    )
