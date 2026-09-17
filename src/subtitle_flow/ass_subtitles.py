"""Sanitized, deterministic ASS subtitle generation for burned-in Turkish text.

The generated file is deliberately simple and resolution-aware:

* bottom-center alignment with safe horizontal/bottom margins;
* yellow primary text, bold and italic, with a black outline and shadow;
* a font size derived from the target height and manual word wrapping so a cue
  renders as at most two visually balanced lines whenever it can, without ever
  truncating a character;
* real cue times (rounded only to the ASS centisecond resolution) -- a cue is
  never moved, clamped, distributed or fabricated.

Security
--------
Cue text is untrusted (it is provider output). ASS has no escape for ``{``/``}``
(override blocks) and treats ``\\N``/``\\n``/``\\h`` as control sequences even
outside an override block, so the sanitizer replaces backslash and both braces
with their visually near-identical fullwidth forms. Control characters and
embedded newlines are normalised. After sanitization only text, spaces and the
sanitizer's own ``\\N`` line breaks can appear in the dialogue payload, so an
override tag can never be injected and a newline/control sequence can never
break the event line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Sequence

from subtitle_flow.timed_subtitles import TimedCue

__all__ = [
    "ASS_STYLE_NAME",
    "AssSubtitleError",
    "ass_timestamp",
    "font_size_for_height",
    "max_chars_per_line",
    "render_ass_document",
    "sanitize_ass_text",
    "validate_font_name",
    "wrap_cue_text",
]

ASS_STYLE_NAME: Final[str] = "Turkce"

#: Minimum rendered font size so a very small frame stays legible.
MIN_FONT_SIZE: Final[int] = 18

#: Font height as a fraction of the frame height (1080p -> ~52px).
FONT_HEIGHT_FRACTION: Final[float] = 0.048

#: Safe margins as a fraction of the frame width/height.
MARGIN_X_FRACTION: Final[float] = 0.05
MARGIN_BOTTOM_FRACTION: Final[float] = 0.055

#: Average glyph advance as a fraction of the font size, used for wrapping.
_GLYPH_ADVANCE: Final[float] = 0.56

#: Fullwidth replacements that neutralise ASS control characters while staying
#: visually close to the original (ASS has no escaping mechanism for them).
_BACKSLASH: Final[str] = "\uff3c"  # ＼
_LEFT_BRACE: Final[str] = "\uff5b"  # ｛
_RIGHT_BRACE: Final[str] = "\uff5d"  # ｝

_CONTROL_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

#: A font-family name is interpolated verbatim into the ASS style line, so only a
#: bounded, unambiguous character set is accepted. Commas (the style field
#: separator), newlines, braces and backslashes (override/escape syntax) and all
#: control characters are structurally impossible in a valid name.
_FONT_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")

#: ASS override-block/metadata delimiters and the dialogue field separator are
#: never allowed to survive from untrusted text.
_FORBIDDEN_IN_TEXT: Final[dict[int, str]] = {
    ord("\\"): _BACKSLASH,
    ord("{"): _LEFT_BRACE,
    ord("}"): _RIGHT_BRACE,
}


class AssSubtitleError(ValueError):
    """A typed failure to build a safe ASS document."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def sanitize_ass_text(text: str) -> str:
    """Return ``text`` as a single ASS-safe line with no control characters."""

    if not isinstance(text, str):
        raise AssSubtitleError("ASS_TEXT_INVALID", "cue text must be a string")
    cleaned = _CONTROL_RE.sub(" ", text)
    cleaned = cleaned.translate(_FORBIDDEN_IN_TEXT)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def validate_font_name(font_name: str) -> str:
    """Return ``font_name`` when it is a bounded, injection-safe family name.

    The name is written straight into the ASS ``Style:`` line without any ASS
    escaping mechanism, so a comma, newline/control character, brace or
    backslash is refused with a typed error instead of corrupting or extending
    the style/event grammar.
    """

    if not isinstance(font_name, str):
        raise AssSubtitleError("ASS_FONT_INVALID", "the font name must be a string")
    if _FONT_NAME_RE.match(font_name) is None:
        raise AssSubtitleError(
            "ASS_FONT_INVALID",
            "the font name contains characters that are not allowed in a safe "
            "font-family name",
        )
    return font_name


def font_size_for_height(height: int) -> int:
    """Return the resolution-aware subtitle font size for one frame height."""

    if not isinstance(height, int) or height <= 0:
        raise AssSubtitleError("ASS_RESOLUTION_INVALID", "the frame height must be positive")
    return max(MIN_FONT_SIZE, int(round(height * FONT_HEIGHT_FRACTION)))


def max_chars_per_line(width: int, font_size: int) -> int:
    """Return a deterministic per-line character budget for manual wrapping."""

    if not isinstance(width, int) or width <= 0:
        raise AssSubtitleError("ASS_RESOLUTION_INVALID", "the frame width must be positive")
    advance = max(1.0, font_size * _GLYPH_ADVANCE)
    return max(12, int(width / advance))


def _split_words(text: str) -> list[str]:
    return [word for word in text.split(" ") if word != ""]


def _hard_split(word: str, limit: int) -> list[str]:
    """Split a single over-long word into bounded pieces (never truncates)."""

    return [word[index : index + limit] for index in range(0, len(word), limit)]


def _greedy_lines(words: Sequence[str], limit: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in words:
        pieces = _hard_split(word, limit) if len(word) > limit else [word]
        for piece in pieces:
            candidate = piece if current == "" else f"{current} {piece}"
            if len(candidate) <= limit:
                current = candidate
            else:
                if current != "":
                    lines.append(current)
                current = piece
    if current != "":
        lines.append(current)
    return lines


def _balance_two_lines(words: Sequence[str], limit: int) -> list[str] | None:
    """Split words into two balanced lines each within ``limit``, or ``None``."""

    if not words:
        return None
    best: list[str] | None = None
    best_delta: int | None = None
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        right = " ".join(words[index:])
        if len(left) > limit or len(right) > limit:
            continue
        delta = abs(len(left) - len(right))
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best = [left, right]
    return best


def wrap_cue_text(text: str, *, max_chars: int) -> str:
    """Wrap sanitized text into at most two balanced lines where feasible.

    Returns the ASS payload with ``\\N`` between lines. A cue too long to fit two
    lines keeps as many lines as needed; no character is ever dropped.
    """

    if max_chars <= 0:
        raise AssSubtitleError("ASS_WRAP_INVALID", "the wrap budget must be positive")
    words = _split_words(text)
    if not words:
        return ""
    greedy = _greedy_lines(words, max_chars)
    if len(greedy) > 2:
        balanced = _balance_two_lines(words, max_chars)
        if balanced is not None:
            return "\\N".join(balanced)
        return "\\N".join(greedy)
    if len(greedy) == 2:
        return "\\N".join(greedy)
    return greedy[0]


def ass_timestamp(ms: int) -> str:
    """Format milliseconds as an ASS ``H:MM:SS.cc`` timestamp.

    The value is rounded to the ASS centisecond resolution (the format cannot
    represent milliseconds); the real millisecond times stay authoritative in the
    timed artifact. No clamping, redistribution or rounding to a different cue.
    """

    if not isinstance(ms, int) or ms < 0:
        raise AssSubtitleError("ASS_CUE_TIMING_INVALID", "cue times must be non-negative")
    centiseconds = (ms + 5) // 10
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, hundredths = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{hundredths:02d}"


@dataclass(frozen=True)
class _AssStyle:
    name: str
    font_name: str
    font_size: int
    margin_l: int
    margin_r: int
    margin_v: int


def _style_for(width: int, height: int) -> _AssStyle:
    font_size = font_size_for_height(height)
    margin_x = max(10, int(round(width * MARGIN_X_FRACTION)))
    margin_v = max(10, int(round(height * MARGIN_BOTTOM_FRACTION)))
    return _AssStyle(
        name=ASS_STYLE_NAME,
        font_name="Arial",
        font_size=font_size,
        margin_l=margin_x,
        margin_r=margin_x,
        margin_v=margin_v,
    )


def render_ass_document(
    cues: Sequence[TimedCue],
    *,
    width: int,
    height: int,
    font_name: str = "Arial",
) -> str:
    """Render a complete, sanitized ASS document for ``cues``.

    Cue order is preserved exactly; overlapping or non-monotonic cues render
    faithfully at their real times (libass supports concurrent events) and are
    never sorted, merged or repaired. A cue with an invalid time range is a typed
    refusal rather than a fabricated replacement.
    """

    font_name = validate_font_name(font_name)
    style = _style_for(width, height)
    if font_name != style.font_name:
        style = _AssStyle(
            name=style.name,
            font_name=font_name,
            font_size=style.font_size,
            margin_l=style.margin_l,
            margin_r=style.margin_r,
            margin_v=style.margin_v,
        )
    limit = max_chars_per_line(width, style.font_size)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: {style.name},{style.font_name},{style.font_size},"
            "&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,-1,0,0,"
            "100,100,0,0,1,3,1,2,"
            f"{style.margin_l},{style.margin_r},{style.margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        if cue.end_ms <= cue.start_ms or cue.start_ms < 0:
            raise AssSubtitleError(
                "ASS_CUE_TIMING_INVALID",
                f"cue {cue.segment_id!r} has an invalid time range; refusing to "
                "invent a replacement time",
            )
        payload = wrap_cue_text(
            sanitize_ass_text(cue.translated_text_tr), max_chars=limit
        )
        if payload == "":
            continue
        lines.append(
            "Dialogue: 0,"
            f"{ass_timestamp(cue.start_ms)},{ass_timestamp(cue.end_ms)},"
            f"{style.name},,0,0,0,,{payload}"
        )
    return "\n".join(lines) + "\n"
