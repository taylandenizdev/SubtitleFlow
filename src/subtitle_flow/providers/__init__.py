"""Provider adapter contracts and the concrete paid adapters.

The base contracts are dependency-free. The concrete Scribe/Google adapters
resolve ``httpx`` lazily, so importing this package never opens a socket, reads a
secret or loads a model.
"""

from subtitle_flow.providers.api_common import (
    ApiBindingError,
    ApiContext,
    ApiExtraMissingError,
    ExpectedSegment,
)
from subtitle_flow.providers.google_basic import GoogleTranslationBasicProvider
from subtitle_flow.providers.mt_base import (
    MachineTranslationProvider,
    UnsupportedLanguageError,
)
from subtitle_flow.providers.scribe import ScribeV2STTProvider
from subtitle_flow.providers.stt_base import SpeechToTextProvider

__all__ = [
    "ApiBindingError",
    "ApiContext",
    "ApiExtraMissingError",
    "ExpectedSegment",
    "GoogleTranslationBasicProvider",
    "MachineTranslationProvider",
    "ScribeV2STTProvider",
    "SpeechToTextProvider",
    "UnsupportedLanguageError",
]
