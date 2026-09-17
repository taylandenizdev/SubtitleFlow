"""Speech-to-text provider contract.

The baseline is deliberately synchronous: one call transcribes one local audio
file. Providers must not open a network connection, load a model or read media
during import; all of that happens inside :meth:`transcribe` when explicitly
invoked. There is no silent routing or fallback between providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import isabstract
from pathlib import Path
from typing import ClassVar, Sequence

from subtitle_flow.schemas import ProviderIdentity, ProviderKind, Transcript

__all__ = ["SpeechToTextProvider", "require_provider_identity"]


def require_provider_identity(
    cls: type, provider_name: str | None, model_name: str | None
) -> None:
    """Validate that a concrete provider declares explicit identity attributes."""

    for attr, value in (("provider_name", provider_name), ("model_name", model_name)):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(
                f"{cls.__name__} must define a non-empty class attribute {attr!r}"
            )


class SpeechToTextProvider(ABC):
    """Abstract synchronous STT provider.

    Concrete providers MUST declare ``provider_name`` and ``model_name`` class
    attributes. :meth:`transcribe` returns a :class:`~subtitle_flow.schemas.Transcript`
    and must not repair invalid provider timings silently.
    """

    provider_name: ClassVar[str]
    model_name: ClassVar[str]

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
            kind=ProviderKind.stt,
            provider=self.provider_name,
            model=self.model_name,
        )

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language_hint: str | None = None,
        keyterms: Sequence[str] | None = None,
    ) -> Transcript:
        """Transcribe one local audio file into the common transcript schema.

        Args:
            audio_path: Local audio file. Video and extraction are out of scope.
            language_hint: Optional provider language code or alias. Not guessed.
            keyterms: Optional known names/terms to bias recognition.

        Returns:
            A validated transcript preserving segment order exactly as produced.
        """

        raise NotImplementedError
