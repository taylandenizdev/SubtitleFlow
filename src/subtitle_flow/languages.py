"""Language-code normalization for the SubtitleFlow pilot.

The canonical internal contract uses BCP-47 style two-letter codes for the six
pilot source languages (``en``, ``ar``, ``fa``, ``ru``, ``de``, ``he``) plus the
Turkish target (``tr``). Provider-supplied codes are preserved separately so the
raw request/response metadata never loses the original value.

Normalization never guesses. A supplied code that is not explicitly known yields
``canonical_code=None`` together with ``uncertain=True``; it is never mapped to a
"closest" language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "CANONICAL_LANGUAGE_CODES",
    "PILOT_SOURCE_LANGUAGE_CODES",
    "TARGET_LANGUAGE_CODE",
    "NormalizedLanguage",
    "canonical_language_or_none",
    "is_supported_language",
    "normalize_language",
    "require_supported_language",
]

TARGET_LANGUAGE_CODE: Final[str] = "tr"

#: Canonical codes understood by the Phase 1 contract.
CANONICAL_LANGUAGE_CODES: Final[frozenset[str]] = frozenset(
    {"en", "ar", "fa", "ru", "de", "he", "tr"}
)

#: Canonical pilot source languages (target is Turkish).
PILOT_SOURCE_LANGUAGE_CODES: Final[frozenset[str]] = frozenset(
    {"en", "ar", "fa", "ru", "de", "he"}
)

#: Explicit alias -> canonical map. Includes ISO 639-2/T and 639-2/B alpha-3
#: codes, legacy codes seen from providers (``iw``) and full English names.
_ALIASES: Final[dict[str, str]] = {
    # English
    "en": "en",
    "eng": "en",
    "english": "en",
    # Arabic
    "ar": "ar",
    "ara": "ar",
    "arabic": "ar",
    # Persian (Farsi)
    "fa": "fa",
    "fas": "fa",
    "per": "fa",
    "farsi": "fa",
    "persian": "fa",
    # Russian
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    # German
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    # Hebrew (legacy ``iw`` is still emitted by some providers)
    "he": "he",
    "heb": "he",
    "iw": "he",
    "hebrew": "he",
    # Turkish
    "tr": "tr",
    "tur": "tr",
    "turkish": "tr",
}


@dataclass(frozen=True)
class NormalizedLanguage:
    """Result of normalizing a provider-supplied language code.

    Attributes:
        provider_code: The original string exactly as received (including any
            surrounding whitespace and an all-blank value), or ``None`` when no
            code was supplied. A missing code is never conflated with a supplied
            but blank code.
        canonical_code: Canonical pilot code, or ``None`` when the code is
            missing or unknown/unsupported.
        uncertain: ``True`` only when a supplied code could not be mapped to a
            canonical pilot language (including a blank string). A missing code
            is not marked uncertain because nothing was claimed.
    """

    provider_code: str | None
    canonical_code: str | None
    uncertain: bool

    @property
    def supported(self) -> bool:
        return self.canonical_code is not None


def normalize_language(raw: str | None) -> NormalizedLanguage:
    """Normalize ``raw`` to a canonical pilot code without guessing.

    ``provider_code`` always keeps ``raw`` byte-for-byte. A stripped, lowercased
    copy is used only for the alias lookup, so region/script subtags
    (``en-US`` -> ``en``, ``iw-IL`` -> ``he``) and case are handled without
    rewriting the recorded provider value. Unknown or unsupported primary
    subtags return ``canonical_code=None`` with ``uncertain=True``; a blank
    string is ``uncertain=True`` but still keeps its exact text. Non-string input
    is rejected instead of being stringified and mislabeled as provider data.
    """

    if raw is None:
        return NormalizedLanguage(provider_code=None, canonical_code=None, uncertain=False)
    if not isinstance(raw, str):
        raise TypeError(
            f"language code must be str or None, got {type(raw).__name__}"
        )

    lookup = raw.strip()
    if lookup == "":
        return NormalizedLanguage(
            provider_code=raw, canonical_code=None, uncertain=True
        )

    key = lookup.lower().replace("_", "-")
    primary = key.split("-", 1)[0]
    canonical = _ALIASES.get(primary)
    if canonical is None:
        return NormalizedLanguage(
            provider_code=raw, canonical_code=None, uncertain=True
        )
    return NormalizedLanguage(
        provider_code=raw, canonical_code=canonical, uncertain=False
    )


def canonical_language_or_none(raw: str | None) -> str | None:
    """Return the canonical code for ``raw``, or ``None`` if unknown/unsupported."""

    return normalize_language(raw).canonical_code


def is_supported_language(raw: str | None) -> bool:
    """Return ``True`` only for codes mapping to a canonical pilot language."""

    return normalize_language(raw).supported


def require_supported_language(raw: str | None, *, field: str = "language") -> str:
    """Return the canonical code for ``raw`` or raise :class:`ValueError`.

    Use this where an unsupported language must fail explicitly instead of
    starting a paid or unsupported operation.
    """

    normalized = normalize_language(raw)
    if normalized.canonical_code is None:
        raise ValueError(
            f"{field} is not a supported pilot language: {raw!r} "
            f"(supported: {sorted(CANONICAL_LANGUAGE_CODES)})"
        )
    return normalized.canonical_code
