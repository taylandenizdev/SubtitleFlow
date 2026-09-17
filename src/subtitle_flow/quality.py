"""Deterministic, provisional quality flags for aligned STT -> MT segments.

The checks here are intentionally conservative and local: they compare the
preserved source text against the produced Turkish text and never rewrite either
side. A missing flag is **not** evidence of semantic accuracy; the checks only
surface a small set of mechanically detectable risks (lost/changed numbers,
dates and percentages, dropped URLs/handles/hashtags, empty or unchanged
translations, suspicious length ratios, adjacent repeated target sentences and
uncertain language). Human WER/CER and semantic adequacy are out of scope and
remain explicitly unmeasured until reference data exists.

Documented limitations (all thresholds are provisional pilot values, not
measured quality):

* Spelled-out numbers (``12`` vs ``on iki``) cannot be matched.
* Locale separator ambiguity is unavoidable: ``1.234`` may be a thousands
  grouping or a decimal value, and ``1,234`` may be the reverse. A separator
  followed by exactly three digits is treated as a possible grouping and is
  compared against the plain digits (``1.234`` == ``1234``), which prefers a
  false negative (no flag) over a false positive (wrong flag). A separator that
  is not a three-digit group is treated as a decimal point and compared exactly
  (``1.5`` != ``15``). Explicit signs and mixed decimal+grouping separators are
  always compared exactly.
* Day/month order in numeric dates is not inferred; only the normalised digit
  sequence is compared.
* Named entities and ordinary words are not checked; only URLs, ``@handles`` and
  hashtags.
* A handle is an ``@`` at a token boundary (not preceded by a word character or
  ``.``, so email interiors such as ``first.last@example.com`` are excluded)
  followed by one or more Unicode word characters (letters, digits, underscore).
  Handles are matched as whole tokens, never by a partial prefix.
* Percentages are compared by their ``%``/``٪``/``％`` marker (which may precede
  or follow the number) or a known percent word (for example ``yüzde`` or
  ``percent``); spelled-out percent in other languages is not matched.
* Translation may reorder numbers, so numbers and percentages are compared by
  occurrence count (multiset), not by position.
* A Turkish source that is legitimately left unchanged (``skipped_same_language``)
  never receives an ``UNCHANGED_TRANSLATION`` flag.
* Equally repeated source content is never treated as definite corruption; an
  adjacent repetition is only flagged when the corresponding source segments
  differ.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "FLAG_DATE_MISMATCH",
    "FLAG_EMPTY_TRANSLATION",
    "FLAG_HANDLE_MISSING",
    "FLAG_HASHTAG_MISSING",
    "FLAG_LANGUAGE_UNCERTAIN",
    "FLAG_NUMBER_MISMATCH",
    "FLAG_PERCENTAGE_MISMATCH",
    "FLAG_POSSIBLE_OMISSION",
    "FLAG_POSSIBLE_REPETITION",
    "FLAG_SUSPICIOUS_LENGTH_RATIO",
    "FLAG_UNCHANGED_TRANSLATION",
    "FLAG_URL_MISSING",
    "QUALITY_LIMITATIONS",
    "QualityInput",
    "QualityReport",
    "QualitySettings",
    "SegmentQuality",
    "assess_quality",
]

FLAG_NUMBER_MISMATCH: Final[str] = "NUMBER_MISMATCH"
FLAG_DATE_MISMATCH: Final[str] = "DATE_MISMATCH"
FLAG_PERCENTAGE_MISMATCH: Final[str] = "PERCENTAGE_MISMATCH"
FLAG_URL_MISSING: Final[str] = "URL_MISSING"
FLAG_HANDLE_MISSING: Final[str] = "HANDLE_MISSING"
FLAG_HASHTAG_MISSING: Final[str] = "HASHTAG_MISSING"
FLAG_EMPTY_TRANSLATION: Final[str] = "EMPTY_TRANSLATION"
FLAG_UNCHANGED_TRANSLATION: Final[str] = "UNCHANGED_TRANSLATION"
FLAG_POSSIBLE_OMISSION: Final[str] = "POSSIBLE_OMISSION"
FLAG_SUSPICIOUS_LENGTH_RATIO: Final[str] = "SUSPICIOUS_LENGTH_RATIO"
FLAG_POSSIBLE_REPETITION: Final[str] = "POSSIBLE_REPETITION"
FLAG_LANGUAGE_UNCERTAIN: Final[str] = "LANGUAGE_UNCERTAIN"

QUALITY_LIMITATIONS: Final[tuple[str, ...]] = (
    "spelled-out numbers (for example '12' vs 'on iki') are not matched",
    "locale separator ambiguity (1.234 vs 1,234) is canonicalised, which may "
    "hide a real change; separators followed by exactly three digits are "
    "treated as possible grouping",
    "numeric day/month order is not inferred for dates",
    "named entities and ordinary wording are not checked semantically",
    "handles are matched as whole Unicode word tokens at an '@' boundary; "
    "email interiors are not treated as handles",
    "percentages are compared by symbol (% and Arabic/Persian equivalents) or "
    "known percent words only; spelled-out percent in other languages is not "
    "matched",
    "numbers and percentages may change order in translation, so occurrences "
    "are compared as a multiset rather than by position",
    "a missing flag is not evidence of semantic accuracy",
    "thresholds are provisional pilot values, not measured quality",
)

_FROZEN = ConfigDict(extra="forbid", frozen=True)

#: Deterministic order used when a segment carries more than one flag.
_FLAG_ORDER: Final[tuple[str, ...]] = (
    FLAG_LANGUAGE_UNCERTAIN,
    FLAG_EMPTY_TRANSLATION,
    FLAG_UNCHANGED_TRANSLATION,
    FLAG_NUMBER_MISMATCH,
    FLAG_DATE_MISMATCH,
    FLAG_PERCENTAGE_MISMATCH,
    FLAG_URL_MISSING,
    FLAG_HANDLE_MISSING,
    FLAG_HASHTAG_MISSING,
    FLAG_POSSIBLE_OMISSION,
    FLAG_SUSPICIOUS_LENGTH_RATIO,
    FLAG_POSSIBLE_REPETITION,
)

_SEPARATORS: Final[str] = ",.'\u2019\u066b\u066c\u00a0\u202f_"
_NUMBER_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"[+\-\u2212]?\d[" + re.escape(_SEPARATORS) + r"\d]*(?<=\d)"
)
_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b"
)
_URL_RE: Final[re.Pattern[str]] = re.compile(r"https?://[^\s<>\"'\)\]\}]+")
#: A handle is an ``@`` at a token boundary (not preceded by a word character,
#: another ``@`` or a ``.`` so email interiors are excluded) followed by one or
#: more Unicode word characters.
_HANDLE_RE: Final[re.Pattern[str]] = re.compile(r"(?<![\w@.])@(\w{1,60})", re.UNICODE)
_HASHTAG_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\w)#(\w{1,60})", re.UNICODE)
_PERCENT_CHARS: Final[str] = "%\u066a\uff05"
#: Words that mark the preceding number as a percentage when no symbol is used.
_PERCENT_WORDS: Final[frozenset[str]] = frozenset(
    {
        "yüzde",
        "yuzde",
        "percent",
        "percentage",
        "pourcent",
        "prozent",
        "procent",
        "procento",
        "porciento",
        "\u043f\u0440\u043e\u0446\u0435\u043d\u0442",
    }
)
_PERCENT_WORD_RE: Final[re.Pattern[str]] = re.compile(r"([^\W\d_]+)\s*$", re.UNICODE)
_URL_TRAILING: Final[str] = ".,;:!?)]}"


def _to_ascii_digits(text: str) -> str:
    """Map Unicode decimal digits to ASCII without touching other characters."""

    out: list[str] = []
    for char in text:
        if "0" <= char <= "9":
            out.append(char)
            continue
        try:
            value = unicodedata.decimal(char)
        except (TypeError, ValueError):
            out.append(char)
            continue
        out.append(str(value))
    return "".join(out)


def _normalize_number(raw: str) -> str:
    """Canonicalise one numeric token to a comparable ASCII decimal string."""

    sign = ""
    body = raw
    if body and body[0] in "+-\u2212":
        sign = "-" if body[0] in "-\u2212" else ""
        body = body[1:]
    body = (
        body.replace("\u066b", ".")
        .replace("\u066c", ",")
        .replace("\u00a0", "")
        .replace("\u202f", "")
        .replace("_", "")
        .replace("'", "")
        .replace("\u2019", "")
    )
    has_dot = "." in body
    has_comma = "," in body
    if has_dot and has_comma:
        if body.rfind(".") > body.rfind(","):
            body = body.replace(",", "")
        else:
            body = body.replace(".", "").replace(",", ".")
    elif has_dot:
        if body.count(".") > 1:
            body = body.replace(".", "")
    elif has_comma:
        if body.count(",") > 1:
            body = body.replace(",", "")
        else:
            body = body.replace(",", ".")
    if "." in body:
        integer, fraction = body.split(".", 1)
        integer = integer.lstrip("0") or "0"
        fraction = fraction.rstrip("0")
        # A zero fraction is dropped so "12", "12.0" and "12.00" are equal.
        body = integer if not fraction else f"{integer}.{fraction}"
    else:
        body = body.lstrip("0") or "0"
    return sign + body


def _normalize_date(raw: str) -> str:
    parts = re.split(r"[-/.]", raw)
    normalized: list[str] = []
    for part in parts:
        stripped = part.lstrip("0") or "0"
        normalized.append(stripped)
    return "-".join(normalized)


def _number_tokens(text: str) -> list[str]:
    """Return raw numeric tokens, excluding any token inside a date."""

    ascii_text = _to_ascii_digits(text)
    date_spans = [match.span() for match in _DATE_RE.finditer(ascii_text)]
    tokens: list[str] = []
    for match in _NUMBER_TOKEN_RE.finditer(ascii_text):
        start, end = match.span()
        if any(start < date_end and date_start < end for date_start, date_end in date_spans):
            continue
        tokens.append(match.group(0))
    return tokens


def _normalized_numbers(text: str) -> list[str]:
    """Return canonical numeric tokens, excluding any token inside a date."""

    return [_normalize_number(token) for token in _number_tokens(text)]


def _group_key(raw: str) -> str | None:
    """Return a sign-preserving grouping-insensitive key, or ``None``.

    ``None`` means the token carries a decimal point or mixed separators that
    cannot be explained as plain thousands grouping, so it must be compared by
    its exact canonical value instead.
    """

    sign = ""
    body = raw
    if body and body[0] in "+-\u2212":
        sign = "-" if body[0] in "-\u2212" else ""
        body = body[1:]
    body = (
        body.replace("\u066b", ".")
        .replace("\u066c", ",")
        .replace("\u00a0", "")
        .replace("\u202f", "")
        .replace("_", "")
        .replace("'", "")
        .replace("\u2019", "")
    )
    if "." not in body and "," not in body:
        if not body.isdigit():
            return None
        return sign + (body.lstrip("0") or "0")
    separators = {char for char in body if char in ".,"}
    if len(separators) != 1:
        return None
    separator = separators.pop()
    if not re.fullmatch(rf"\d{{1,3}}(?:{re.escape(separator)}\d{{3}})+", body):
        return None
    return sign + (body.replace(separator, "").lstrip("0") or "0")


def _numbers_match(source_text: str, target_text: str) -> bool:
    """Compare numeric occurrences, tolerating only grouping ambiguity.

    Exact canonical values are compared first. Only when they differ does the
    grouping-insensitive key apply, and then only if every token on both sides
    is a plausible grouped integer, so signs and unambiguous decimals are never
    erased. Occurrence counts (not order) are compared.
    """

    if Counter(_normalized_numbers(source_text)) == Counter(
        _normalized_numbers(target_text)
    ):
        return True
    source_grouping = [_group_key(token) for token in _number_tokens(source_text)]
    target_grouping = [_group_key(token) for token in _number_tokens(target_text)]
    if None in source_grouping or None in target_grouping:
        return False
    return Counter(source_grouping) == Counter(target_grouping)


def _normalized_dates(text: str) -> list[str]:
    ascii_text = _to_ascii_digits(text)
    return [_normalize_date(match.group(0)) for match in _DATE_RE.finditer(ascii_text)]


def _percent_occurrences(text: str) -> list[str]:
    """Return canonical numbers marked as percentages, occurrence by occurrence.

    A number is percent-marked when a ``%``/``٪``/``％`` symbol is directly
    adjacent (before or after, ignoring spaces) or when an immediately preceding
    known percent word is present.
    """

    ascii_text = _to_ascii_digits(text)
    results: list[str] = []
    for match in _NUMBER_TOKEN_RE.finditer(ascii_text):
        before = ascii_text[: match.start()].rstrip()
        after = ascii_text[match.end() :].lstrip()
        has_symbol = (before and before[-1] in _PERCENT_CHARS) or (
            after and after[0] in _PERCENT_CHARS
        )
        if not has_symbol:
            word = _PERCENT_WORD_RE.search(before)
            has_symbol = (
                word is not None and word.group(1).casefold() in _PERCENT_WORDS
            )
        if has_symbol:
            results.append(_normalize_number(match.group(0)))
    return results


def _urls(text: str) -> list[str]:
    found: list[str] = []
    for match in _URL_RE.finditer(text):
        token = match.group(0).rstrip(_URL_TRAILING)
        if token:
            found.append(token)
    return found


def _handles(text: str) -> list[str]:
    return [f"@{match.group(1)}" for match in _HANDLE_RE.finditer(text)]


def _hashtags(text: str) -> list[str]:
    return [f"#{match.group(1)}" for match in _HASHTAG_RE.finditer(text)]


def _missing_tokens(source_tokens: list[str], target: str) -> list[str]:
    folded_target = target.casefold()
    return [token for token in source_tokens if token.casefold() not in folded_target]


def _missing_handles(source_tokens: list[str], target: str) -> list[str]:
    """Return source handles whose whole token is absent from the target.

    Unlike a plain substring test this never accepts a partial ASCII prefix as
    equal, so ``@kullanıcı`` is not considered present in ``@kullanan``.
    """

    target_tokens = {token.casefold() for token in _handles(target)}
    return [token for token in source_tokens if token.casefold() not in target_tokens]


class QualitySettings(BaseModel):
    """Provisional, frozen thresholds. Changing them changes the flags."""

    model_config = _FROZEN

    min_length_ratio: Annotated[
        float, Field(strict=True, gt=0.0, allow_inf_nan=False)
    ] = 0.30
    max_length_ratio: Annotated[
        float, Field(strict=True, gt=0.0, allow_inf_nan=False)
    ] = 4.0
    min_length_chars: Annotated[int, Field(strict=True, ge=1)] = 12
    repetition_min_chars: Annotated[int, Field(strict=True, ge=1)] = 12


@dataclass(frozen=True)
class QualityInput:
    """One source/translation pair plus any flags already attached upstream."""

    segment_id: str
    source_text: str
    translated_text_tr: str | None
    translation_input: str | None = None
    existing_flags: tuple[str, ...] = ()


class SegmentQuality(BaseModel):
    """Final flag set for one segment (existing flags first, then checks)."""

    model_config = _FROZEN

    segment_id: str = Field(min_length=1)
    flags: tuple[str, ...] = ()


class QualityReport(BaseModel):
    """Per-segment flags plus honest coverage and limitation notes."""

    model_config = _FROZEN

    settings: QualitySettings
    same_language: bool
    language_uncertain: bool
    checked_segment_count: int = Field(strict=True, ge=0)
    translated_segment_count: int = Field(strict=True, ge=0)
    flagged_segment_count: int = Field(strict=True, ge=0)
    flag_counts: dict[str, int] = Field(default_factory=dict)
    segments: tuple[SegmentQuality, ...] = ()
    limitations: tuple[str, ...] = QUALITY_LIMITATIONS

    @property
    def review_required(self) -> bool:
        return self.flagged_segment_count > 0

    def flags_for(self, segment_id: str) -> tuple[str, ...]:
        for entry in self.segments:
            if entry.segment_id == segment_id:
                return entry.flags
        return ()


def _ordered_flags(flags: set[str]) -> tuple[str, ...]:
    ordered = [flag for flag in _FLAG_ORDER if flag in flags]
    extras = sorted(flag for flag in flags if flag not in _FLAG_ORDER)
    return tuple(ordered + extras)


def _content_flags(
    source_text: str,
    translated: str,
    *,
    settings: QualitySettings,
) -> set[str]:
    flags: set[str] = set()
    stripped_target = translated.strip()
    stripped_source = source_text.strip()
    if stripped_target == "":
        flags.add(FLAG_EMPTY_TRANSLATION)
        return flags
    if stripped_source != "" and stripped_target == stripped_source:
        flags.add(FLAG_UNCHANGED_TRANSLATION)

    # Compare numeric occurrences exactly; only a plausible thousands-grouping
    # difference is tolerated, so signs and unambiguous decimals are preserved.
    if not _numbers_match(source_text, translated):
        flags.add(FLAG_NUMBER_MISMATCH)

    source_dates = _normalized_dates(source_text)
    target_dates = _normalized_dates(translated)
    if sorted(source_dates) != sorted(target_dates):
        flags.add(FLAG_DATE_MISMATCH)

    # Percent units are compared occurrence by occurrence so a lost or gained
    # ``%`` is reported even when the digits match.
    if Counter(_percent_occurrences(source_text)) != Counter(
        _percent_occurrences(translated)
    ):
        flags.add(FLAG_PERCENTAGE_MISMATCH)

    if _missing_tokens(_urls(source_text), translated):
        flags.add(FLAG_URL_MISSING)
    if _missing_handles(_handles(source_text), translated):
        flags.add(FLAG_HANDLE_MISSING)
    if _missing_tokens(_hashtags(source_text), translated):
        flags.add(FLAG_HASHTAG_MISSING)

    if len(source_text) >= settings.min_length_chars:
        ratio = len(translated) / len(source_text)
        if ratio < settings.min_length_ratio:
            flags.add(FLAG_POSSIBLE_OMISSION)
        elif ratio > settings.max_length_ratio:
            flags.add(FLAG_SUSPICIOUS_LENGTH_RATIO)
    return flags


def assess_quality(
    items: list[QualityInput],
    *,
    settings: QualitySettings | None = None,
    same_language: bool = False,
    language_uncertain: bool = False,
) -> QualityReport:
    """Compute deterministic per-segment flags for an ordered segment list.

    ``same_language`` marks a verified ``skipped_same_language`` result (source
    equals target by contract); content comparison is skipped so an unchanged
    Turkish transcript is never reported as a translation defect. ``translated``
    of ``None`` means the segment was not translated at all (for example a
    source-only export) and content checks are skipped, never treated as empty.
    """

    resolved = settings if settings is not None else QualitySettings()
    results: list[SegmentQuality] = []
    counts: dict[str, int] = {}
    flagged = 0
    translated_count = 0
    for index, item in enumerate(items):
        check_flags: set[str] = set()
        if language_uncertain:
            check_flags.add(FLAG_LANGUAGE_UNCERTAIN)
        translated = item.translated_text_tr
        if translated is not None:
            translated_count += 1
            if not same_language:
                check_flags |= _content_flags(
                    item.source_text, translated, settings=resolved
                )
            elif translated.strip() == "":
                check_flags.add(FLAG_EMPTY_TRANSLATION)
        if index > 0 and translated is not None:
            previous = items[index - 1]
            previous_translated = previous.translated_text_tr
            if (
                not same_language
                and previous_translated is not None
                and translated.strip() != ""
                and translated == previous_translated
                and len(translated) >= resolved.repetition_min_chars
                and item.source_text != previous.source_text
            ):
                check_flags.add(FLAG_POSSIBLE_REPETITION)
        # Existing upstream flags (for example ``same_language``) are preserved
        # on the segment but never counted as a quality defect on their own.
        ordered = _ordered_flags(set(item.existing_flags) | check_flags)
        if check_flags:
            flagged += 1
        for flag in check_flags:
            counts[flag] = counts.get(flag, 0) + 1
        results.append(SegmentQuality(segment_id=item.segment_id, flags=ordered))

    return QualityReport(
        settings=resolved,
        same_language=same_language,
        language_uncertain=language_uncertain,
        checked_segment_count=len(items),
        translated_segment_count=translated_count,
        flagged_segment_count=flagged,
        flag_counts=counts,
        segments=tuple(results),
    )
