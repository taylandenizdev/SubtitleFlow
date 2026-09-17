"""SubtitleFlow common foundation.

Phase 1 exposes validated schemas, language normalization and provider
interfaces. Phase 2A adds ready-audio validation, immutable job configuration,
durable per-job storage and a synchronous resumable pipeline. Importing this
package must not open network connections, load models or read media.
"""

from importlib.metadata import PackageNotFoundError, version

from subtitle_flow.config import (
    ApiLimits,
    ApiSettings,
    BatchSettings,
    ExtractionSettings,
    GoogleBasicSettings,
    MediaLimits,
    PaidApiPolicy,
    PipelineConfig,
    ScribeSettings,
    StageOptions,
    VideoLimits,
    VideoOriginSettings,
    YouTubeLimits,
    YouTubeOriginSettings,
)
from subtitle_flow.exports import (
    AlignedArtifact,
    AlignedSegment,
    AlignmentProvenance,
    ExportResult,
    ExportsManifest,
    JobMetrics,
    build_alignment,
    export_job,
)
from subtitle_flow.languages import (
    CANONICAL_LANGUAGE_CODES,
    PILOT_SOURCE_LANGUAGE_CODES,
    TARGET_LANGUAGE_CODE,
    normalize_language,
)
from subtitle_flow.media import AudioInfo, MediaError, recheck_audio, validate_audio
from subtitle_flow.pipeline import (
    AudioPipeline,
    PipelineResult,
    group_segments,
)
from subtitle_flow.providers import (
    MachineTranslationProvider,
    SpeechToTextProvider,
    UnsupportedLanguageError,
)
from subtitle_flow.providers.api_common import (
    ApiBindingError,
    ApiContext,
    ApiExtraMissingError,
)
from subtitle_flow.providers.errors import ProviderCallError
from subtitle_flow.providers.google_basic import GoogleTranslationBasicProvider
from subtitle_flow.providers.scribe import ScribeV2STTProvider
from subtitle_flow.quality import (
    QualityReport,
    QualitySettings,
    assess_quality,
)
from subtitle_flow.schemas import (
    SCHEMA_VERSION,
    JobInput,
    JobRecord,
    JobStatus,
    Segment,
    StageResult,
    Transcript,
    Translation,
    TranslationStatus,
)
from subtitle_flow.storage import (
    JobStore,
    MTGroupRecord,
    MTManifest,
    STTManifest,
    StoredInput,
)
from subtitle_flow.video import (
    EXTRACTION_VERSION,
    ExtractionManifest,
    ExtractionResult,
    VideoInfo,
    VideoMediaError,
    compute_extraction_id,
    extract_audio,
    extraction_locator,
    ffmpeg_tool_version,
    recheck_video,
    validate_video,
)
from subtitle_flow.youtube_source import (
    YOUTUBE_EXTRACTION_VERSION,
    YouTubeMediaError,
    YouTubeReference,
    YouTubeSourceResult,
    acquire_youtube_source,
    canonicalize_youtube_url,
    youtube_locator,
)
from subtitle_flow.yt_dlp_runner import (
    YtDlpDownload,
    YtDlpError,
    YtDlpLimits,
    download_audio,
    ytdlp_version,
)

try:
    __version__ = version("subtitle-flow")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = [
    "CANONICAL_LANGUAGE_CODES",
    "PILOT_SOURCE_LANGUAGE_CODES",
    "SCHEMA_VERSION",
    "TARGET_LANGUAGE_CODE",
    "AlignedArtifact",
    "AlignedSegment",
    "AlignmentProvenance",
    "ApiBindingError",
    "ApiContext",
    "ApiExtraMissingError",
    "ApiLimits",
    "ApiSettings",
    "AudioInfo",
    "AudioPipeline",
    "BatchSettings",
    "EXTRACTION_VERSION",
    "ExportResult",
    "ExportsManifest",
    "ExtractionManifest",
    "ExtractionResult",
    "ExtractionSettings",
    "GoogleBasicSettings",
    "GoogleTranslationBasicProvider",
    "JobInput",
    "JobMetrics",
    "JobRecord",
    "JobStatus",
    "JobStore",
    "MTGroupRecord",
    "MTManifest",
    "MachineTranslationProvider",
    "MediaError",
    "MediaLimits",
    "PaidApiPolicy",
    "PipelineConfig",
    "PipelineResult",
    "ProviderCallError",
    "QualityReport",
    "QualitySettings",
    "STTManifest",
    "ScribeSettings",
    "ScribeV2STTProvider",
    "Segment",
    "SpeechToTextProvider",
    "StageOptions",
    "StageResult",
    "StoredInput",
    "Transcript",
    "Translation",
    "TranslationStatus",
    "UnsupportedLanguageError",
    "VideoInfo",
    "VideoLimits",
    "VideoMediaError",
    "VideoOriginSettings",
    "YOUTUBE_EXTRACTION_VERSION",
    "YtDlpDownload",
    "YtDlpError",
    "YtDlpLimits",
    "YouTubeLimits",
    "YouTubeMediaError",
    "YouTubeOriginSettings",
    "YouTubeReference",
    "YouTubeSourceResult",
    "__version__",
    "acquire_youtube_source",
    "assess_quality",
    "build_alignment",
    "canonicalize_youtube_url",
    "compute_extraction_id",
    "download_audio",
    "export_job",
    "extract_audio",
    "extraction_locator",
    "ffmpeg_tool_version",
    "group_segments",
    "normalize_language",
    "recheck_audio",
    "recheck_video",
    "validate_audio",
    "validate_video",
    "ytdlp_version",
    "youtube_locator",
]
