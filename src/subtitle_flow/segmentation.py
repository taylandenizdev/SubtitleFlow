"""Deterministic Scribe word-token normalization into source segments.

This module is the Phase 2B groundwork between a real ElevenLabs Scribe
provider response and the common :class:`~subtitle_flow.schemas.Segment`
contract. It is a *pure* function: it opens no network connection, reads no
file, loads no model and imports no provider SDK. The adapter that owns the
HTTP call parses the raw body, keeps the raw payload, and hands the ``words``
list plus ``text`` here.

Input contract (official Scribe ``words`` entries)
--------------------------------------------------
Every entry is a JSON object with:

* ``type``: ``"word"``, ``"spacing"`` or ``"audio_event"`` (required).
* ``text``: the literal transcription fragment (required, must be ``str``).
* ``start`` / ``end``: seconds, optional and nullable in the provider schema.
  Spoken ``word`` entries must carry both; ``spacing`` and ``audio_event``
  entries may omit them.
* ``speaker_id``: optional and nullable.

Anything else the provider sends (``logprob``, ``characters``,
``channel_index``, ...) is ignored, never interpreted and never persisted.

Literal preservation
--------------------
The exact text of every ``word`` and ``spacing`` token is preserved byte for
byte, including Unicode, punctuation attached to words, zero-width and
bidirectional marks, diacritics, and leading/inter-word/trailing whitespace.
The concatenation of all emitted ``Segment.source_text`` values equals the
concatenation of the speech (``word`` + ``spacing``) token texts in provider
order, and the same exact stream is always available as
``SegmentationResult.preserved_speech_text`` / ``metadata["preserved_speech_text"]``
even when no usable spoken word exists. No separator is invented, no whitespace
is stripped or normalized and no token is silently dropped. ``audio_event``
tokens are the only ones that may be omitted from the segments; they are
recorded with trace counts in metadata.

Token well-formedness is enforced, never guessed at:

* a ``word`` token must contain non-whitespace spoken content; empty or
  whitespace-only text raises ``SegmentationTokenError`` instead of
  manufacturing a whitespace-only speech segment;
* a ``spacing`` token may only carry Unicode whitespace plus a narrow set of
  zero-width/bidirectional format marks (:data:`_SPACING_FORMAT_MARKS`); any
  other (spoken-looking) content raises ``SegmentationTokenError`` instead of
  being silently lost. Empty spacing is valid;
* unknown or malformed token ``type`` values raise.

Determinism
-----------
Timings are converted from provider seconds to integer milliseconds with true
decimal half-up rounding: the incoming JSON number is read as the decimal it
represents (``Decimal(str(value))``) and multiplied by 1000 with
``ROUND_HALF_UP``, so a provider decimal tie such as ``0.5005`` s / ``2.0035`` s
rounds to ``501`` / ``2004`` ms. Timing bounds are compared in the same decimal
seconds contract, both before and after rounding. A word is never clamped and is
never repaired: reversed, missing, non-finite, boolean, string, out-of-audio or
zero-after-rounding spoken timings raise explicitly.

Segment boundaries are driven only by explicit, deterministic rules:
sentence-terminating punctuation, an inter-word silence gap, a speaker change,
the maximum segment duration and the maximum segment character count. Between
two words the spacing run belongs to the *preceding* word's segment.

Algorithm versions
------------------
Two deterministic algorithms share this module and are selected explicitly:

* :data:`SEGMENTER_ALGORITHM_VERSION_V2` (``scribe-word-v2``): the original
  stricter behaviour. A spoken ``word`` must satisfy ``start < end`` and must not
  collapse to an equal millisecond after half-up rounding; such a token raises.
  Existing job snapshots that recorded v2 keep exactly this behaviour, so an old
  transcript is never re-interpreted under a new algorithm.
* :data:`SEGMENTER_ALGORITHM_VERSION` (``scribe-word-v3``, the default for new
  jobs): a zero-duration word (``start == end``) or a positive sub-millisecond
  word whose rounded ``start_ms`` and ``end_ms`` are equal is treated as an
  *uncertain point timestamp*. The raw values and all literal text are kept and
  the token is aggregated into an adjacent same-speaker segment. Every segment
  that contains a point is emitted with an interval that covers the real
  provider spans **and** the known zero-width point positions of its own units,
  so ``start_ms <= point_ms <= end_ms`` always holds; a point contributes its
  exact known position and never an invented duration. No ``+1`` millisecond is
  invented, no value is clamped and no word is dropped, edited or merged across
  speakers. The gap/character/duration bounds are evaluated over real spans and
  known point positions together, so a long point chain cannot escape the
  segment duration limit. When a point token cannot be supported by any adjacent
  same-speaker real span within the configured gap/character/duration bounds,
  the response fails closed with :class:`SegmentationNeedsReviewError` (the raw
  body is already archived), so an isolated zero-only utterance is reported for
  review instead of being given a fabricated duration. Point timings, the number
  of segments that contain them and an ``uncertain_word_timings`` quality flag
  are recorded in metadata.

Positive-only token streams produce byte-identical segments under v2 and v3; only
zero/sub-millisecond timings differ. Reversed, out-of-audio, negative,
non-finite, boolean, string and missing timings are rejected by both versions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from subtitle_flow.languages import require_supported_language
from subtitle_flow.schemas import Segment

__all__ = [
    "POINT_TIMING_FLAG",
    "SEGMENTER_ALGORITHM_VERSION",
    "SEGMENTER_ALGORITHM_VERSION_V2",
    "SUPPORTED_SEGMENTER_VERSIONS",
    "ExcludedAudioEvent",
    "PointTiming",
    "SegmentationError",
    "SegmentationLimitError",
    "SegmentationNeedsReviewError",
    "SegmentationNoSpeechError",
    "SegmentationResult",
    "SegmentationTimingError",
    "SegmentationTokenError",
    "SegmenterSettings",
    "SentenceBoundaryPolicy",
    "segment_scribe_tokens",
]

#: Algorithm identity of the default algorithm for *new* jobs. Persist this next
#: to the segmenter settings so a change here can never reuse an unrelated older
#: segmentation result. Bumped to v3 when zero-duration and sub-millisecond
#: point timestamps became supported as uncertain timings that are aggregated
#: with adjacent same-speaker text instead of being rejected; a v3 segment span
#: covers the known point positions it carries as well as the real provider
#: spans.
SEGMENTER_ALGORITHM_VERSION: Final[str] = "scribe-word-v3"

#: The previous algorithm. It is retained verbatim for existing job snapshots:
#: a stored v2 fingerprint/config keeps producing strict v2 segments, and no old
#: output is silently relabeled or re-segmented as v3.
SEGMENTER_ALGORITHM_VERSION_V2: Final[str] = "scribe-word-v2"

#: Every algorithm this build can deterministically reproduce. A truly unknown
#: stored version (including a hypothetical v1) is refused rather than guessed.
SUPPORTED_SEGMENTER_VERSIONS: Final[frozenset[str]] = frozenset(
    {SEGMENTER_ALGORITHM_VERSION_V2, SEGMENTER_ALGORITHM_VERSION}
)

#: Quality flag attached to a segment whose span had to absorb one or more
#: uncertain point timings. It explicitly does not claim accurate word durations.
POINT_TIMING_FLAG: Final[str] = "uncertain_word_timings"

#: Exact decimal conversion factor, kept as :class:`~decimal.Decimal` so no
#: binary float enters the millisecond rounding contract.
_MS_PER_SECOND: Final[Decimal] = Decimal(1000)

#: Narrow, explicit policy for non-whitespace characters that may legitimately
#: accompany whitespace inside a ``spacing`` token. These are the zero-width and
#: bidirectional formatting marks that carry no spoken content but that a
#: provider may emit around real whitespace (ZWNJ in Persian, RTL marks in
#: Hebrew, BOM/word-joiner at stream edges, and the bidi embedding/isolate
#: controls). Anything outside Unicode whitespace plus this set is treated as
#: spoken-looking content and rejected rather than silently dropped.
_SPACING_FORMAT_MARKS: Final[frozenset[str]] = frozenset(
    "\u200b"  # ZERO WIDTH SPACE
    "\u200c"  # ZERO WIDTH NON-JOINER
    "\u200d"  # ZERO WIDTH JOINER
    "\u200e"  # LEFT-TO-RIGHT MARK
    "\u200f"  # RIGHT-TO-LEFT MARK
    "\u202a"  # LEFT-TO-RIGHT EMBEDDING
    "\u202b"  # RIGHT-TO-LEFT EMBEDDING
    "\u202c"  # POP DIRECTIONAL FORMATTING
    "\u202d"  # LEFT-TO-RIGHT OVERRIDE
    "\u202e"  # RIGHT-TO-LEFT OVERRIDE
    "\u2060"  # WORD JOINER
    "\u2061"  # FUNCTION APPLICATION
    "\u2062"  # INVISIBLE TIMES
    "\u2063"  # INVISIBLE SEPARATOR
    "\u2064"  # INVISIBLE PLUS
    "\u2066"  # LEFT-TO-RIGHT ISOLATE
    "\u2067"  # RIGHT-TO-LEFT ISOLATE
    "\u2068"  # FIRST STRONG ISOLATE
    "\u2069"  # POP DIRECTIONAL ISOLATE
    "\ufeff"  # ZERO WIDTH NO-BREAK SPACE (BOM)
)

_KNOWN_TOKEN_TYPES: Final[frozenset[str]] = frozenset(
    {"word", "spacing", "audio_event"}
)

#: Characters that terminate a sentence in the six pilot languages.
#: ``.``/``!``/``?`` cover English, German, Russian and Hebrew; ``…`` is a
#: common ellipsis; ``؟`` is the Arabic/Persian question mark; ``۔`` is the
#: Arabic/Urdu full stop; ``׃`` is the Hebrew sof pasuq; ``。`` is the CJK full
#: stop (kept harmless for mixed-script audio).
_SENTENCE_TERMINATORS: Final[frozenset[str]] = frozenset(
    {".", "!", "?", "\u2026", "\u061f", "\u06d4", "\u05c3", "\u3002"}
)

#: Characters that may follow a terminator (closing quotes/brackets) and must
#: not hide it, e.g. ``world."`` or German ``»so!«``.
_TRAILING_CLOSERS: Final[str] = "".join(
    sorted(
        {
            '"',
            "'",
            "\u2019",
            "\u201d",
            "\u201c",
            "\u201e",
            "\u00bb",
            "\u203a",
            "\u00ab",
            "\u2039",
            ")",
            "]",
            "}",
        }
    )
)


class SentenceBoundaryPolicy(StrEnum):
    """Whether sentence-terminating punctuation forces a segment boundary."""

    after_terminator = "after_terminator"
    never = "never"


class SegmenterSettings(BaseModel):
    """Frozen, validated, JSON-serializable segmentation thresholds.

    The settings are a deliberate part of the job fingerprint: equal settings
    always yield equal segmentation for equal tokens. All fields are strict
    integers, so booleans, floats and numeric strings are rejected instead of
    being coerced.

    Defaults are conservative pilot values (roughly subtitle-sized) and may be
    tuned by later measurement; changing any field changes the fingerprint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_segment_chars: Annotated[
        int, Field(strict=True, ge=1, le=1_000_000)
    ] = 500
    max_segment_duration_ms: Annotated[
        int, Field(strict=True, ge=1, le=21_600_000)
    ] = 15_000
    max_gap_ms: Annotated[
        int, Field(strict=True, ge=0, le=21_600_000)
    ] = 700
    sentence_boundary_policy: SentenceBoundaryPolicy = (
        SentenceBoundaryPolicy.after_terminator
    )


class SegmentationError(ValueError):
    """Base class for every deterministic segmentation failure."""


class SegmentationTokenError(SegmentationError):
    """A token entry is structurally malformed or has an unknown type."""


class SegmentationTimingError(SegmentationError):
    """A provided timing value is invalid; no clamping or repair happens."""


class SegmentationLimitError(SegmentationError):
    """An indivisible word/spacing run exceeds a hard segment limit."""


class SegmentationNoSpeechError(SegmentationError):
    """Spoken full text exists but no usable spoken word token can carry it."""


class SegmentationNeedsReviewError(SegmentationError):
    """A point timing cannot be supported by any real adjacent span.

    Raised by the v3 algorithm for a zero-duration/sub-millisecond word that has
    no same-speaker neighbour within the configured gap/character/duration
    bounds. The literal text is never dropped or given a fabricated duration; the
    caller routes the job to human review with the archived raw response.
    """


@dataclass(frozen=True)
class PointTiming:
    """An uncertain provider point timestamp kept verbatim for traceability.

    ``start_seconds``/``end_seconds`` are the exact provider values, ``point_ms``
    is the shared rounded millisecond. This is *not* an accurate word duration.
    """

    index: int
    start_seconds: float
    end_seconds: float
    point_ms: int
    speaker_id: str | None


@dataclass(frozen=True)
class ExcludedAudioEvent:
    """An ``audio_event`` token excluded from segments, kept for traceability."""

    index: int
    text: str
    start_seconds: float | None
    end_seconds: float | None
    speaker_id: str | None


@dataclass(frozen=True)
class SegmentationResult:
    """Segments plus JSON-valid provenance metadata for one Scribe response.

    ``segments`` is the only value the pipeline persists as normalized output;
    ``metadata`` is JSON-serializable and records the excluded events, the
    full-text discrepancy, no-speech state, token counts, settings, the exact
    preserved speech stream and the algorithm version. The adapter owns the full
    ``Transcript``/``RawArtifactRef``.

    ``preserved_speech_text`` is the exact concatenation of the ``word`` and
    ``spacing`` token texts in provider order (``audio_event`` excluded). It is
    always kept, including in the no-speech branch, so a spacing-only provider
    response can never lose its literal text even when no segment is emitted.
    """

    segments: tuple[Segment, ...]
    source_language: str
    audio_duration_ms: int
    settings: SegmenterSettings
    no_speech: bool
    full_text_mismatch: bool
    provider_full_text: str
    spoken_word_count: int
    spacing_token_count: int
    audio_event_count: int
    preserved_speech_text: str = ""
    excluded_audio_events: tuple[ExcludedAudioEvent, ...] = ()
    algorithm_version: str = SEGMENTER_ALGORITHM_VERSION
    #: Uncertain point timings (v3 only); empty for v2 and positive-only streams.
    point_timings: tuple[PointTiming, ...] = ()
    #: Number of emitted segments whose interval covers one or more point
    #: timings (the interval is widened to the known point positions it carries).
    point_grouped_segment_count: int = 0
    #: Non-empty when the result relies on uncertain timings.
    quality_flags: tuple[str, ...] = ()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a fresh, JSON-valid metadata mapping."""

        return {
            "algorithm_version": self.algorithm_version,
            "source_language": self.source_language,
            "audio_duration_ms": self.audio_duration_ms,
            "settings": self.settings.model_dump(mode="json"),
            "no_speech": self.no_speech,
            "full_text_mismatch": self.full_text_mismatch,
            "provider_full_text": self.provider_full_text,
            "preserved_speech_text": self.preserved_speech_text,
            "segment_count": len(self.segments),
            "point_timing_count": len(self.point_timings),
            "point_grouped_segment_count": self.point_grouped_segment_count,
            "quality_flags": list(self.quality_flags),
            "point_timings": [
                {
                    "index": timing.index,
                    "start_seconds": timing.start_seconds,
                    "end_seconds": timing.end_seconds,
                    "point_ms": timing.point_ms,
                    "speaker_id": timing.speaker_id,
                }
                for timing in self.point_timings
            ],
            "token_counts": {
                "word": self.spoken_word_count,
                "spacing": self.spacing_token_count,
                "audio_event": self.audio_event_count,
                "total": (
                    self.spoken_word_count
                    + self.spacing_token_count
                    + self.audio_event_count
                ),
            },
            "excluded_audio_events": [
                {
                    "index": event.index,
                    "text": event.text,
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "speaker_id": event.speaker_id,
                }
                for event in self.excluded_audio_events
            ],
        }


# --------------------------------------------------------------------------- #
# Internal parsed representations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Word:
    index: int
    text: str
    start_ms: int
    end_ms: int
    speaker: str | None
    #: True for a v3 uncertain point timing (``start_ms == end_ms``).
    is_point: bool = False


@dataclass
class _Unit:
    """One spoken word with the spacing runs attached to it.

    ``leading`` exists only for the first word of the stream (spacing before it
    has no preceding segment); ``trailing`` holds the spacing run between this
    word and the next spoken word (or to end of stream).
    """

    word: _Word
    leading: tuple[str, ...] = ()
    trailing: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return (
            "".join(self.leading) + self.word.text + "".join(self.trailing)
        )

    @property
    def is_point(self) -> bool:
        return self.word.is_point


# --------------------------------------------------------------------------- #
# Input validation helpers (clear, non-swallowing)
# --------------------------------------------------------------------------- #
def _require_token_sequence(tokens: object) -> Sequence[Any]:
    if isinstance(tokens, (str, bytes)) or not isinstance(tokens, Sequence):
        raise TypeError(
            "tokens must be a sequence of Scribe word mappings, got "
            f"{type(tokens).__name__}"
        )
    return tokens


def _require_mapping(raw: object, index: int) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise SegmentationTokenError(
            f"token {index} must be a mapping, got {type(raw).__name__}"
        )
    for key in raw:
        if not isinstance(key, str):
            raise SegmentationTokenError(
                f"token {index} has a non-string mapping key "
                f"({type(key).__name__})"
            )
    return raw


def _require_token_type(token: Mapping[str, Any], index: int) -> str:
    if "type" not in token:
        raise SegmentationTokenError(f"token {index} is missing required 'type'")
    value = token["type"]
    if not isinstance(value, str):
        raise SegmentationTokenError(
            f"token {index} 'type' must be str, got {type(value).__name__}"
        )
    if value not in _KNOWN_TOKEN_TYPES:
        raise SegmentationTokenError(
            f"token {index} has unsupported type {value!r}; expected one of "
            f"{sorted(_KNOWN_TOKEN_TYPES)}"
        )
    return value


def _require_text(token: Mapping[str, Any], index: int, token_type: str) -> str:
    if "text" not in token:
        raise SegmentationTokenError(f"token {index} is missing required 'text'")
    value = token["text"]
    if not isinstance(value, str):
        raise SegmentationTokenError(
            f"token {index} 'text' must be str, got {type(value).__name__}"
        )
    if token_type == "word" and value.strip() == "":
        raise SegmentationTokenError(
            f"word token {index} has empty or whitespace-only text "
            f"({value!r}); a spoken word must contain non-whitespace content"
        )
    if token_type == "spacing":
        _require_spacing_text(value, index)
    return value


def _is_spacing_only(text: str) -> bool:
    """Return ``True`` if ``text`` carries no spoken content.

    Spoken content means any character that is not Unicode whitespace and not
    one of the narrow :data:`_SPACING_FORMAT_MARKS` (an empty string is
    spacing-only). This is used both to police ``spacing`` token text and to
    decide whether a ``full_text`` with no word tokens is legitimate no-speech.
    """

    return all(
        char.isspace() or char in _SPACING_FORMAT_MARKS for char in text
    )


def _require_spacing_text(text: str, index: int) -> None:
    """Reject a spacing token that carries spoken-looking content.

    Allowed characters are Unicode whitespace and the narrow
    :data:`_SPACING_FORMAT_MARKS` set. Empty text is valid. The literal text is
    never trimmed, normalized or rewritten; the check only decides whether the
    token may be preserved as pure spacing.
    """

    if _is_spacing_only(text):
        return
    raise SegmentationTokenError(
        f"spacing token {index} text {text!r} contains non-whitespace "
        "content; spacing tokens may only carry whitespace and zero-width/"
        "bidirectional format marks"
    )


def _require_speaker(token: Mapping[str, Any], index: int) -> str | None:
    if "speaker_id" not in token:
        return None
    value = token["speaker_id"]
    if value is None:
        return None
    if not isinstance(value, str):
        raise SegmentationTokenError(
            f"token {index} 'speaker_id' must be str or null, got "
            f"{type(value).__name__}"
        )
    return value


def _require_seconds(value: object, *, index: int, field_name: str) -> Decimal:
    """Validate and return a genuine seconds value as an exact decimal.

    Only real ``int``/``float`` JSON numbers are accepted (booleans and numeric
    strings are rejected, never coerced). The returned :class:`~decimal.Decimal`
    is the decimal the provider number represents (``Decimal(str(value))``), so
    the later millisecond rounding and all timing-bound comparisons happen in one
    consistent decimal contract with no binary-float tie errors.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentationTimingError(
            f"token {index} '{field_name}' must be a finite number of seconds, "
            f"got {type(value).__name__} ({value!r})"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise SegmentationTimingError(
            f"token {index} '{field_name}' is too large to represent as seconds"
        ) from None
    if not math.isfinite(number):
        raise SegmentationTimingError(
            f"token {index} '{field_name}' must be finite, got {value!r}"
        )
    if number < 0:
        raise SegmentationTimingError(
            f"token {index} '{field_name}' must be non-negative, got {value!r}"
        )
    return Decimal(str(value))


def _seconds_to_ms(seconds: Decimal | int | float) -> int:
    """Deterministic decimal half-up rounding to integer milliseconds.

    The number is read as the decimal it represents (``Decimal(str(value))``)
    and multiplied by 1000 with ``ROUND_HALF_UP``, so provider decimal ties such
    as ``0.5005`` round to ``501`` instead of being lost to a binary-float
    ``floor(x * 1000 + 0.5)`` calculation.
    """

    if isinstance(seconds, bool) or not isinstance(
        seconds, (Decimal, int, float)
    ):
        raise TypeError(
            "seconds must be a Decimal or a real number, got "
            f"{type(seconds).__name__}"
        )
    decimal_seconds = (
        seconds if isinstance(seconds, Decimal) else Decimal(str(seconds))
    )
    return int(
        (decimal_seconds * _MS_PER_SECOND).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )


def _require_duration_ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "audio_duration_ms must be a non-negative integer, got "
            f"{type(value).__name__}"
        )
    if value < 0:
        raise ValueError("audio_duration_ms must be non-negative")
    try:
        float(value)
    except (OverflowError, ValueError):
        raise ValueError(
            "audio_duration_ms is too large to represent in seconds"
        ) from None
    return value


def _require_canonical_language(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"source_language must be str, got {type(value).__name__}"
        )
    canonical = require_supported_language(value, field="source_language")
    if canonical != value:
        raise ValueError(
            f"source_language must be a canonical code, got {value!r}; "
            f"use {canonical!r} and keep the provider code separately"
        )
    return canonical


def _parse_word(
    token: Mapping[str, Any], index: int, text: str, duration_ms: int
) -> _Word:
    if "start" not in token or "end" not in token:
        raise SegmentationTimingError(
            f"word token {index} must provide 'start' and 'end'"
        )
    start_raw = token["start"]
    end_raw = token["end"]
    if start_raw is None or end_raw is None:
        raise SegmentationTimingError(
            f"word token {index} must provide non-null 'start' and 'end'"
        )
    start_sec = _require_seconds(start_raw, index=index, field_name="start")
    end_sec = _require_seconds(end_raw, index=index, field_name="end")
    duration_sec = Decimal(duration_ms) / _MS_PER_SECOND
    if start_sec > duration_sec or end_sec > duration_sec:
        raise SegmentationTimingError(
            f"word token {index} timing [{start_sec}, {end_sec}] is outside the "
            f"{duration_sec} s audio duration"
        )
    if start_sec >= end_sec:
        raise SegmentationTimingError(
            f"word token {index} must have start < end, got "
            f"[{start_sec}, {end_sec}]"
        )
    start_ms = _seconds_to_ms(start_sec)
    end_ms = _seconds_to_ms(end_sec)
    if start_ms >= end_ms:
        raise SegmentationTimingError(
            f"word token {index} collapses to zero duration after millisecond "
            f"rounding (start_ms={start_ms}, end_ms={end_ms})"
        )
    if end_ms > duration_ms:
        raise SegmentationTimingError(
            f"word token {index} end_ms={end_ms} exceeds audio_duration_ms="
            f"{duration_ms} after rounding"
        )
    return _Word(
        index=index,
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        speaker=_require_speaker(token, index),
    )


def _parse_word_v3(
    token: Mapping[str, Any], index: int, text: str, duration_ms: int
) -> _Word:
    """Parse one spoken word for the v3 algorithm.

    Timing validation stays strict (finite, non-negative, bounded by the audio
    duration, ``start <= end``); unlike v2 a zero-duration word, or a positive
    word that collapses to an equal millisecond after half-up rounding, is not a
    failure. It becomes an uncertain *point* timing whose raw values are kept:

    * ``start > end`` (reversed), negative, non-finite, boolean, string, null,
      missing or out-of-audio timings still raise;
    * ``start == end`` and positive sub-millisecond collapse are accepted as a
      point (``start_ms == end_ms``); nothing is clamped or widened.
    """

    if "start" not in token or "end" not in token:
        raise SegmentationTimingError(
            f"word token {index} must provide 'start' and 'end'"
        )
    start_raw = token["start"]
    end_raw = token["end"]
    if start_raw is None or end_raw is None:
        raise SegmentationTimingError(
            f"word token {index} must provide non-null 'start' and 'end'"
        )
    start_sec = _require_seconds(start_raw, index=index, field_name="start")
    end_sec = _require_seconds(end_raw, index=index, field_name="end")
    duration_sec = Decimal(duration_ms) / _MS_PER_SECOND
    if start_sec > duration_sec or end_sec > duration_sec:
        raise SegmentationTimingError(
            f"word token {index} timing [{start_sec}, {end_sec}] is outside the "
            f"{duration_sec} s audio duration"
        )
    if start_sec > end_sec:
        raise SegmentationTimingError(
            f"word token {index} must have start <= end, got "
            f"[{start_sec}, {end_sec}]"
        )
    start_ms = _seconds_to_ms(start_sec)
    end_ms = _seconds_to_ms(end_sec)
    if end_ms > duration_ms:
        raise SegmentationTimingError(
            f"word token {index} end_ms={end_ms} exceeds audio_duration_ms="
            f"{duration_ms} after rounding"
        )
    is_point = start_ms >= end_ms
    point_ms = start_ms
    return _Word(
        index=index,
        text=text,
        start_ms=point_ms if is_point else start_ms,
        end_ms=point_ms if is_point else end_ms,
        speaker=_require_speaker(token, index),
        is_point=is_point,
    )


def _validate_optional_timing(
    token: Mapping[str, Any], index: int, duration_ms: int
) -> tuple[Decimal | None, Decimal | None]:
    """Validate spacing/event times *where present*; zero duration is allowed."""

    start_present = "start" in token and token["start"] is not None
    end_present = "end" in token and token["end"] is not None
    if not start_present and not end_present:
        return None, None
    if not (start_present and end_present):
        raise SegmentationTimingError(
            f"token {index} must provide both 'start' and 'end' or neither"
        )
    start_sec = _require_seconds(token["start"], index=index, field_name="start")
    end_sec = _require_seconds(token["end"], index=index, field_name="end")
    duration_sec = Decimal(duration_ms) / _MS_PER_SECOND
    if start_sec > duration_sec or end_sec > duration_sec:
        raise SegmentationTimingError(
            f"token {index} timing [{start_sec}, {end_sec}] is outside the "
            f"{duration_sec} s audio duration"
        )
    if start_sec > end_sec:
        raise SegmentationTimingError(
            f"token {index} must have start <= end, got [{start_sec}, {end_sec}]"
        )
    start_ms = _seconds_to_ms(start_sec)
    end_ms = _seconds_to_ms(end_sec)
    if start_ms > end_ms:
        raise SegmentationTimingError(
            f"token {index} timing inverts after millisecond rounding "
            f"(start_ms={start_ms}, end_ms={end_ms})"
        )
    if end_ms > duration_ms:
        raise SegmentationTimingError(
            f"token {index} end_ms={end_ms} exceeds audio_duration_ms="
            f"{duration_ms} after rounding"
        )
    return start_sec, end_sec


def _parse_event(
    token: Mapping[str, Any], index: int, text: str, duration_ms: int
) -> ExcludedAudioEvent:
    start_sec, end_sec = _validate_optional_timing(token, index, duration_ms)
    return ExcludedAudioEvent(
        index=index,
        text=text,
        start_seconds=None if start_sec is None else float(start_sec),
        end_seconds=None if end_sec is None else float(end_sec),
        speaker_id=_require_speaker(token, index),
    )


def _ends_sentence(text: str) -> bool:
    candidate = text.rstrip().rstrip(_TRAILING_CLOSERS)
    if not candidate:
        return False
    return candidate[-1] in _SENTENCE_TERMINATORS


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
class _SegmentBuilder:
    """Accumulates units into one segment and flushes deterministic records."""

    __slots__ = (
        "_language",
        "_settings",
        "_units",
        "start_ms",
        "end_ms",
        "speaker",
        "last_word_end_ms",
        "ends_sentence",
        "_char_count",
    )

    def __init__(self, language: str, settings: SegmenterSettings) -> None:
        self._language = language
        self._settings = settings
        self._units: list[_Unit] = []
        self.start_ms = 0
        self.end_ms = 0
        self.speaker: str | None = None
        self.last_word_end_ms = 0
        self.ends_sentence = False
        self._char_count = 0

    @property
    def active(self) -> bool:
        return bool(self._units)

    @property
    def char_count(self) -> int:
        return self._char_count

    def would_exceed(self, unit: _Unit) -> bool:
        candidate_start = min(self.start_ms, unit.word.start_ms)
        candidate_end = max(self.end_ms, unit.word.end_ms)
        return (
            unit.word.speaker != self.speaker
            or unit.word.start_ms - self.last_word_end_ms
            > self._settings.max_gap_ms
            or (
                self._settings.sentence_boundary_policy
                is SentenceBoundaryPolicy.after_terminator
                and self.ends_sentence
            )
            or self._char_count + len(unit.text)
            > self._settings.max_segment_chars
            or candidate_end - candidate_start
            > self._settings.max_segment_duration_ms
        )

    def add(self, unit: _Unit) -> None:
        if not self._units:
            self.start_ms = unit.word.start_ms
            self.end_ms = unit.word.end_ms
            self.speaker = unit.word.speaker
        else:
            self.start_ms = min(self.start_ms, unit.word.start_ms)
            self.end_ms = max(self.end_ms, unit.word.end_ms)
        self.last_word_end_ms = unit.word.end_ms
        self.ends_sentence = _ends_sentence(unit.word.text)
        self._units.append(unit)
        self._char_count += len(unit.text)

    def flush(self, index: int) -> Segment:
        return Segment(
            segment_id=f"seg_{index:04d}",
            start_ms=self.start_ms,
            end_ms=self.end_ms,
            source_language=self._language,
            source_text="".join(unit.text for unit in self._units),
            speaker=self.speaker,
        )


def _build_segments(
    units: Sequence[_Unit],
    language: str,
    settings: SegmenterSettings,
) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    builder = _SegmentBuilder(language, settings)
    for unit in units:
        unit_length = len(unit.text)
        if unit_length > settings.max_segment_chars:
            raise SegmentationLimitError(
                f"word token {unit.word.index} with its attached spacing needs "
                f"{unit_length} characters, exceeding max_segment_chars="
                f"{settings.max_segment_chars}; the run is indivisible"
            )
        word_duration = unit.word.end_ms - unit.word.start_ms
        if word_duration > settings.max_segment_duration_ms:
            raise SegmentationLimitError(
                f"word token {unit.word.index} lasts {word_duration} ms, "
                f"exceeding max_segment_duration_ms="
                f"{settings.max_segment_duration_ms}; the word is indivisible"
            )
        if builder.active and builder.would_exceed(unit):
            segments.append(builder.flush(len(segments) + 1))
            builder = _SegmentBuilder(language, settings)
        builder.add(unit)
    if builder.active:
        segments.append(builder.flush(len(segments) + 1))
    return tuple(segments)


class _SegmentBuilderV3:
    """Accumulates v3 units; a segment interval covers real and known points.

    Point units contribute their literal text and their exact known position to
    the deterministic boundary decisions. A point never invents a duration, but
    the emitted interval is widened to the minimum/maximum known position of the
    group's own units, so ``start_ms <= point_ms <= end_ms`` holds for every
    point a segment carries. :meth:`flush` raises
    :class:`SegmentationNeedsReviewError` when a group has no real span at all,
    so a zero-only group is never emitted with a fabricated duration.
    """

    __slots__ = (
        "_language",
        "_settings",
        "_units",
        "real_start_ms",
        "real_end_ms",
        "span_start_ms",
        "span_end_ms",
        "speaker",
        "last_word_end_ms",
        "ends_sentence",
        "_char_count",
        "_real_word_count",
        "_point_count",
        "_trailing_point_forward_anchor",
    )

    def __init__(self, language: str, settings: SegmenterSettings) -> None:
        self._language = language
        self._settings = settings
        self._units: list[_Unit] = []
        self.real_start_ms = 0
        self.real_end_ms = 0
        #: Emitted interval: min/max over real spans *and* known point positions.
        self.span_start_ms = 0
        self.span_end_ms = 0
        self.speaker: str | None = None
        self.last_word_end_ms = 0
        self.ends_sentence = False
        self._char_count = 0
        self._real_word_count = 0
        self._point_count = 0
        self._trailing_point_forward_anchor = False

    @property
    def active(self) -> bool:
        return bool(self._units)

    @property
    def char_count(self) -> int:
        return self._char_count

    @property
    def point_count(self) -> int:
        return self._point_count

    def would_exceed(self, unit: _Unit, *, point_forward_anchor: bool) -> bool:
        if unit.word.speaker != self.speaker:
            return True
        if (
            unit.word.start_ms - self.last_word_end_ms
            > self._settings.max_gap_ms
        ):
            return True
        if (
            self._settings.sentence_boundary_policy
            is SentenceBoundaryPolicy.after_terminator
            and self.ends_sentence
        ):
            if unit.is_point and not point_forward_anchor:
                # A point without a forward anchor must stay with the preceding
                # run so it is never stranded without a real span.
                pass
            elif (
                self._real_word_count == 0
                and self._trailing_point_forward_anchor
            ):
                # An all-point group whose last unit is a point with a valid
                # forward anchor must wait for that anchor: splitting it off
                # would flush a point-only group with no honest interval.
                pass
            else:
                # A point with a valid forward anchor may start the next segment
                # and preserve the sentence boundary.
                return True
        if self._char_count + len(unit.text) > self._settings.max_segment_chars:
            return True
        # The duration bound is evaluated over real spans *and* known point
        # positions, so a long point chain cannot evade the segment limit.
        candidate_start = min(self.span_start_ms, unit.word.start_ms)
        candidate_end = max(self.span_end_ms, unit.word.end_ms)
        if (
            candidate_end - candidate_start
            > self._settings.max_segment_duration_ms
        ):
            return True
        return False

    def add(self, unit: _Unit, *, forward_anchor: bool = False) -> None:
        if not self._units:
            self.speaker = unit.word.speaker
            self.span_start_ms = unit.word.start_ms
            self.span_end_ms = unit.word.end_ms
        else:
            self.span_start_ms = min(self.span_start_ms, unit.word.start_ms)
            self.span_end_ms = max(self.span_end_ms, unit.word.end_ms)
        if unit.is_point:
            self._point_count += 1
        elif self._real_word_count == 0:
            self.real_start_ms = unit.word.start_ms
            self.real_end_ms = unit.word.end_ms
            self._real_word_count = 1
        else:
            self.real_start_ms = min(self.real_start_ms, unit.word.start_ms)
            self.real_end_ms = max(self.real_end_ms, unit.word.end_ms)
            self._real_word_count += 1
        self._trailing_point_forward_anchor = bool(
            unit.is_point and forward_anchor
        )
        self.last_word_end_ms = unit.word.end_ms
        self.ends_sentence = _ends_sentence(unit.word.text)
        self._units.append(unit)
        self._char_count += len(unit.text)

    def flush(self, index: int) -> Segment:
        if self._real_word_count == 0:
            raise SegmentationNeedsReviewError(
                "a segment contains only zero-duration point timings with no "
                "adjacent same-speaker real span within the configured bounds; "
                "refusing to fabricate a duration, the literal text is preserved "
                "in the archived raw response and the job needs review"
            )
        flags = (POINT_TIMING_FLAG,) if self._point_count else ()
        return Segment(
            segment_id=f"seg_{index:04d}",
            start_ms=self.span_start_ms,
            end_ms=self.span_end_ms,
            source_language=self._language,
            source_text="".join(unit.text for unit in self._units),
            speaker=self.speaker,
            flags=flags,
        )


def _v3_point_forward_anchors(
    units: Sequence[_Unit], settings: SegmenterSettings
) -> list[bool]:
    """Return, per unit index, whether a point unit has a valid forward anchor.

    A point at index ``i`` has a forward anchor when the nearest real span to its
    right has the same speaker as every unit between (no cross-speaker merge) and
    lies within ``max_gap_ms``. Only point indices can be ``True``; the flag
    decides whether a sentence boundary may split before the point, so a point
    that cannot be supported forward stays with the preceding run instead of
    being stranded, and a trailing point punctuation with a valid forward anchor
    stays with its anchor instead of being flushed as a point-only group.
    """

    count = len(units)
    anchors = [False] * count
    if count == 0:
        return anchors
    same_run_end = [0] * count
    nearest_real: list[int | None] = [None] * (count + 1)
    for i in range(count - 1, -1, -1):
        if (
            i + 1 < count
            and units[i + 1].word.speaker == units[i].word.speaker
        ):
            same_run_end[i] = same_run_end[i + 1]
        else:
            same_run_end[i] = i
        nearest_real[i] = i if not units[i].is_point else nearest_real[i + 1]
    for i in range(count):
        if not units[i].is_point:
            continue
        anchor = nearest_real[i + 1]
        if anchor is None or anchor > same_run_end[i]:
            continue
        gap = units[anchor].word.start_ms - units[i].word.end_ms
        if gap <= settings.max_gap_ms:
            anchors[i] = True
    return anchors


def _build_segments_v3(
    units: Sequence[_Unit],
    language: str,
    settings: SegmenterSettings,
) -> tuple[tuple[Segment, ...], int]:
    """Build v3 segments; return ``(segments, point_grouped_segment_count)``."""

    segments: list[Segment] = []
    point_grouped = 0
    anchors = _v3_point_forward_anchors(units, settings)
    builder = _SegmentBuilderV3(language, settings)
    for index, unit in enumerate(units):
        unit_length = len(unit.text)
        if unit_length > settings.max_segment_chars:
            raise SegmentationLimitError(
                f"word token {unit.word.index} with its attached spacing needs "
                f"{unit_length} characters, exceeding max_segment_chars="
                f"{settings.max_segment_chars}; the run is indivisible"
            )
        word_duration = unit.word.end_ms - unit.word.start_ms
        if word_duration > settings.max_segment_duration_ms:
            raise SegmentationLimitError(
                f"word token {unit.word.index} lasts {word_duration} ms, "
                f"exceeding max_segment_duration_ms="
                f"{settings.max_segment_duration_ms}; the word is indivisible"
            )
        if builder.active and builder.would_exceed(
            unit, point_forward_anchor=anchors[index]
        ):
            if builder.point_count:
                point_grouped += 1
            segments.append(builder.flush(len(segments) + 1))
            builder = _SegmentBuilderV3(language, settings)
        builder.add(unit, forward_anchor=anchors[index])
    if builder.active:
        if builder.point_count:
            point_grouped += 1
        segments.append(builder.flush(len(segments) + 1))
    return tuple(segments), point_grouped


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def segment_scribe_tokens(
    *,
    tokens: Sequence[Mapping[str, Any]],
    full_text: str,
    source_language: str,
    audio_duration_ms: int,
    settings: SegmenterSettings,
    algorithm_version: str = SEGMENTER_ALGORITHM_VERSION,
) -> SegmentationResult:
    """Convert parsed Scribe ``words`` + ``text`` into source segments.

    Args:
        tokens: The provider ``words`` list, in provider order.
        full_text: The provider ``text`` value, preserved exactly.
        source_language: Canonical pilot code (``en``/``ar``/``fa``/``ru``/
            ``de``/``he``/``tr``); unknown or non-canonical codes raise.
        audio_duration_ms: Verified audio duration in integer milliseconds.
        settings: Frozen :class:`SegmenterSettings` snapshot.
        algorithm_version: One of :data:`SUPPORTED_SEGMENTER_VERSIONS`. The
            default is the current algorithm for new jobs; an existing job
            snapshot's stored version must be passed explicitly so old work is
            never silently re-segmented under a newer algorithm.

    Returns:
        A :class:`SegmentationResult` whose ``segments`` are ready to persist
        and whose ``metadata`` is JSON-valid. The adapter owns building the
        full :class:`~subtitle_flow.schemas.Transcript` and raw reference.

    Raises:
        TypeError: for structurally wrong inputs (null payloads, non-sequence
            tokens, non-string text, non-integer duration, wrong settings type).
        SegmentationError: for an unknown algorithm version, malformed tokens,
            invalid timings, indivisible oversized runs, spoken text without
            usable word tokens, or a v3 point timing that has no adjacent
            same-speaker span to borrow a real span from.
    """

    if algorithm_version not in SUPPORTED_SEGMENTER_VERSIONS:
        raise SegmentationError(
            f"unsupported segmenter algorithm version {algorithm_version!r}; "
            f"supported versions are {sorted(SUPPORTED_SEGMENTER_VERSIONS)}"
        )
    if not isinstance(settings, SegmenterSettings):
        raise TypeError(
            "settings must be a validated SegmenterSettings instance, got "
            f"{type(settings).__name__}"
        )
    # ``model_copy``/``model_construct`` can bypass field validation on the
    # public model, so re-validate the dumped settings at this boundary. A
    # mutated enum value or an out-of-range limit can then never silently change
    # segmentation behaviour; it fails instead.
    try:
        settings = SegmenterSettings.model_validate(
            settings.model_dump(warnings=False)
        )
    except ValidationError as exc:
        raise SegmentationError(
            "settings failed re-validation at the segmentation boundary; "
            "mutated or unvalidated settings are rejected"
        ) from exc
    if not isinstance(full_text, str):
        raise TypeError(
            f"full_text must be str, got {type(full_text).__name__}"
        )
    canonical_language = _require_canonical_language(source_language)
    duration_ms = _require_duration_ms(audio_duration_ms)
    token_list = _require_token_sequence(tokens)

    units: list[_Unit] = []
    pending: list[str] = []
    events: list[ExcludedAudioEvent] = []
    preserved_parts: list[str] = []
    point_timings: list[PointTiming] = []
    word_count = 0
    spacing_count = 0
    event_count = 0
    is_v3 = algorithm_version == SEGMENTER_ALGORITHM_VERSION

    for index, raw in enumerate(token_list):
        token = _require_mapping(raw, index)
        token_type = _require_token_type(token, index)
        text = _require_text(token, index, token_type)
        if token_type == "word":
            if is_v3:
                word = _parse_word_v3(token, index, text, duration_ms)
                if word.is_point:
                    point_timings.append(
                        PointTiming(
                            index=index,
                            start_seconds=float(
                                _require_seconds(
                                    token["start"], index=index, field_name="start"
                                )
                            ),
                            end_seconds=float(
                                _require_seconds(
                                    token["end"], index=index, field_name="end"
                                )
                            ),
                            point_ms=word.start_ms,
                            speaker_id=word.speaker,
                        )
                    )
            else:
                word = _parse_word(token, index, text, duration_ms)
            if units:
                units[-1].trailing = tuple(pending)
            leading = tuple(pending) if not units else ()
            pending = []
            units.append(_Unit(word=word, leading=leading))
            word_count += 1
            preserved_parts.append(text)
        elif token_type == "spacing":
            _validate_optional_timing(token, index, duration_ms)
            pending.append(text)
            spacing_count += 1
            preserved_parts.append(text)
        else:  # audio_event
            events.append(_parse_event(token, index, text, duration_ms))
            event_count += 1

    if units:
        units[-1].trailing = tuple(pending)
    preserved_stream = "".join(preserved_parts)

    if not units:
        if _is_spacing_only(full_text):
            return SegmentationResult(
                segments=(),
                source_language=canonical_language,
                audio_duration_ms=duration_ms,
                settings=settings,
                no_speech=True,
                full_text_mismatch=preserved_stream != full_text,
                provider_full_text=full_text,
                spoken_word_count=0,
                spacing_token_count=spacing_count,
                audio_event_count=event_count,
                preserved_speech_text=preserved_stream,
                excluded_audio_events=tuple(events),
                algorithm_version=algorithm_version,
            )
        raise SegmentationNoSpeechError(
            "full_text is non-empty but the response contains no usable word "
            "tokens; refusing to report a false success"
        )

    if is_v3:
        segments, point_grouped = _build_segments_v3(
            units, canonical_language, settings
        )
    else:
        segments = _build_segments(units, canonical_language, settings)
        point_grouped = 0
    return SegmentationResult(
        segments=segments,
        source_language=canonical_language,
        audio_duration_ms=duration_ms,
        settings=settings,
        no_speech=False,
        full_text_mismatch=preserved_stream != full_text,
        provider_full_text=full_text,
        spoken_word_count=word_count,
        spacing_token_count=spacing_count,
        audio_event_count=event_count,
        preserved_speech_text=preserved_stream,
        excluded_audio_events=tuple(events),
        algorithm_version=algorithm_version,
        point_timings=tuple(point_timings),
        point_grouped_segment_count=point_grouped,
        quality_flags=(
            (POINT_TIMING_FLAG,) if point_timings else ()
        ),
    )
