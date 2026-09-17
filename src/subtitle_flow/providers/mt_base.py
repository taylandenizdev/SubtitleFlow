"""Machine-translation provider contract.

The baseline is deliberately synchronous and batch-oriented: one call translates
an ordered list of source segments into Turkish. Providers must not open a
network connection, load a model or read files during import. There is no silent
routing or fallback between providers.

Turkish is the only supported output. Before any adapter call or same-language
shortcut, :meth:`MachineTranslationProvider.translate` normalizes the declared
source and rejects duplicate incoming segment IDs and any segment whose
``source_language`` disagrees with that source. This stops Arabic (or any other)
text from being relabeled Turkish. A successful adapter result is checked against
the requested provider/model/language identity and must return each original
``source_text`` and ``translation_input`` byte-exact, per retained ID and in the
original order. Failed and remote-unknown results stay explicitly failed with
their error trace and no segments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import isabstract
from typing import ClassVar, Sequence

from subtitle_flow.languages import TARGET_LANGUAGE_CODE, normalize_language
from subtitle_flow.schemas import (
    ProviderIdentity,
    ProviderKind,
    Segment,
    SegmentContractError,
    TranslatedSegment,
    Translation,
    TranslationStatus,
    validate_segment_contract,
)

__all__ = [
    "MachineTranslationProvider",
    "UnsupportedLanguageError",
    "require_provider_identity",
]


def require_provider_identity(
    cls: type, provider_name: str | None, model_name: str | None
) -> None:
    """Validate that a concrete provider declares explicit identity attributes."""

    for attr, value in (("provider_name", provider_name), ("model_name", model_name)):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(
                f"{cls.__name__} must define a non-empty class attribute {attr!r}"
            )


def _with_flag(flags: tuple[str, ...], flag: str) -> tuple[str, ...]:
    if flag in flags:
        return flags
    return (*flags, flag)


class UnsupportedLanguageError(ValueError):
    """Raised instead of guessing when a language is unknown or unsupported."""


class MachineTranslationProvider(ABC):
    """Abstract synchronous MT provider targeting Turkish."""

    provider_name: ClassVar[str]
    model_name: ClassVar[str]
    default_target_language: ClassVar[str] = TARGET_LANGUAGE_CODE

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            return
        require_provider_identity(
            cls, getattr(cls, "provider_name", None), getattr(cls, "model_name", None)
        )

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity(
            kind=ProviderKind.mt,
            provider=self.provider_name,
            model=self.model_name,
        )

    def translate(
        self,
        segments: Sequence[Segment],
        source_language: str,
        target_language: str | None = None,
    ) -> Translation:
        """Translate ordered source segments into Turkish.

        ``source_language`` accepts any supported pilot code or alias.
        ``target_language`` may be omitted (defaults to ``'tr'``) or given as a
        supported Turkish alias (``'TR'``/``'tur'``); every other target is
        rejected before the adapter is involved. A Turkish source returns the
        original text as ``skipped_same_language`` with no adapter call.

        Raises:
            UnsupportedLanguageError: unknown source or a non-Turkish target.
            SegmentContractError: duplicate source IDs or a segment whose
                ``source_language`` disagrees with the declared source.
        """

        resolved_target = (
            target_language if target_language is not None else self.default_target_language
        )
        source = normalize_language(source_language)
        target = normalize_language(resolved_target)
        if source.canonical_code is None:
            raise UnsupportedLanguageError(
                f"unsupported source language: {source_language!r}"
            )
        if target.canonical_code is None:
            raise UnsupportedLanguageError(
                f"unsupported target language: {resolved_target!r}"
            )
        if target.canonical_code != TARGET_LANGUAGE_CODE:
            raise UnsupportedLanguageError(
                "only Turkish output is supported; "
                f"unsupported target language: {resolved_target!r}"
            )

        ordered = tuple(segments)
        self._validate_source_segments(ordered, source.canonical_code)

        if source.canonical_code == target.canonical_code:
            return self._same_language_translation(
                ordered, canonical_language=source.canonical_code
            )

        result = self._translate(
            ordered,
            source_language=source.canonical_code,
            target_language=target.canonical_code,
        )
        self._validate_result(
            result,
            ordered=ordered,
            source_language=source.canonical_code,
            target_language=target.canonical_code,
        )
        return result

    def _validate_source_segments(
        self, segments: Sequence[Segment], source_language: str
    ) -> None:
        """Reject duplicate IDs and segment/declared source disagreement.

        Runs before any adapter call or same-language shortcut, so a miscoded
        request fails with zero provider calls instead of producing a fabricated
        translation.
        """

        issues: list[str] = []
        seen: set[str] = set()
        for segment in segments:
            if segment.segment_id in seen:
                issues.append(f"duplicate source segment id: {segment.segment_id!r}")
            seen.add(segment.segment_id)
            if segment.source_language != source_language:
                issues.append(
                    f"segment {segment.segment_id!r} source_language "
                    f"{segment.source_language!r} does not match declared source "
                    f"language {source_language!r}"
                )
        if issues:
            raise SegmentContractError(issues, context="source segments")

    def _same_language_translation(
        self,
        segments: Sequence[Segment],
        *,
        canonical_language: str,
    ) -> Translation:
        """Build the explicit no-op translation used when source == target.

        Only called after :meth:`_validate_source_segments` verified the incoming
        identity. ``source_text``, ``translation_input`` and existing flags are
        retained exactly and the ``same_language`` flag is added.
        """

        translated = tuple(
            TranslatedSegment(
                segment_id=segment.segment_id,
                source_text=segment.source_text,
                translation_input=segment.translation_input,
                translated_text_tr=segment.source_text,
                flags=_with_flag(segment.flags, "same_language"),
            )
            for segment in segments
        )
        return Translation(
            provider=self.provider_name,
            model=self.model_name,
            status=TranslationStatus.skipped_same_language,
            source_language=canonical_language,
            target_language=canonical_language,
            segments=translated,
        )

    def _validate_result(
        self,
        result: Translation,
        *,
        ordered: Sequence[Segment],
        source_language: str,
        target_language: str,
    ) -> None:
        """Reject identity drift and fabricated or altered input provenance.

        Provider/model identity is always checked. Source/target language is
        checked for every non-failed result. A non-failed result must also match
        the incoming segment IDs and order and return ``source_text`` and
        ``translation_input`` byte-exact. Failed and remote-unknown results are
        accepted by identity and otherwise returned unchanged: the schema already
        requires an error trace and forbids segments. Nothing is repaired here.
        """

        issues: list[str] = []
        if result.provider != self.provider_name:
            issues.append(
                f"result provider {result.provider!r} does not match adapter "
                f"{self.provider_name!r}"
            )
        if result.model != self.model_name:
            issues.append(
                f"result model {result.model!r} does not match adapter "
                f"{self.model_name!r}"
            )
        if result.status is not TranslationStatus.failed:
            if result.source_language != source_language:
                issues.append(
                    f"result source_language {result.source_language!r} does not "
                    f"match request {source_language!r}"
                )
            if result.target_language != target_language:
                issues.append(
                    f"result target_language {result.target_language!r} does not "
                    f"match request {target_language!r}"
                )
        if issues:
            raise SegmentContractError(issues, context="translation identity")

        if result.status is TranslationStatus.failed:
            return

        validate_segment_contract(
            [segment.segment_id for segment in ordered],
            [segment.segment_id for segment in result.segments],
            context="translation segments",
        )
        for source_segment, translated in zip(ordered, result.segments):
            if translated.source_text != source_segment.source_text:
                issues.append(
                    f"segment {source_segment.segment_id!r} source_text was altered"
                )
            if translated.translation_input != source_segment.translation_input:
                issues.append(
                    f"segment {source_segment.segment_id!r} translation_input was "
                    "altered or dropped"
                )
        if issues:
            raise SegmentContractError(issues, context="translation provenance")

    @abstractmethod
    def _translate(
        self,
        segments: Sequence[Segment],
        *,
        source_language: str,
        target_language: str,
    ) -> Translation:
        """Provider-specific translation with documented canonical codes.

        Only called for distinct languages. ``source_language`` is the canonical
        pilot code (for example ``ar``) and ``target_language`` is ``tr``. Any
        original provider code is kept separately as data.
        """

        raise NotImplementedError
