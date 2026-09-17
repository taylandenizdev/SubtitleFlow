"""Validated, immutable job configuration and deterministic cache fingerprints.

A job's behavior is fully described by an explicit, frozen snapshot: languages,
provider identities, keyterms and versions, schema/segmenter versions, stage
options, batching settings and media limits. Secrets never belong here; the
snapshot contains no credentials and no API keys are read at construction time.

The fingerprint is deterministic and covers the audio SHA-256 together with every
setting that can change results, so an old output can never be presented as the
result of a new configuration. ``config_fingerprint`` excludes the audio hash to
detect configuration-only changes; ``job_fingerprint`` includes it.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from subtitle_flow.languages import TARGET_LANGUAGE_CODE, normalize_language
from subtitle_flow.schemas import (
    SCHEMA_VERSION,
    ProviderIdentity,
    ProviderKind,
    Sha256Hex,
    UtcDatetime,
)
from subtitle_flow.segmentation import SEGMENTER_ALGORITHM_VERSION, SegmenterSettings

__all__ = [
    "CONFIG_VERSION",
    "FINGERPRINT_VERSION",
    "GOOGLE_BASIC_MAX_CODEPOINTS",
    "GOOGLE_BASIC_MAX_ITEMS",
    "GOOGLE_BASIC_MODEL_FAMILY",
    "GOOGLE_BASIC_OFFICIAL_BASE_URL",
    "GOOGLE_BASIC_OFFICIAL_PATH",
    "GOOGLE_BASIC_PROVIDER",
    "GOOGLE_BASIC_ROUTE_VERSION",
    "MONEY_MAX_ABSOLUTE_EXPONENT",
    "MONEY_MAX_DECIMAL_PRECISION",
    "SCRIBE_MAX_AUDIO_BYTES",
    "SCRIBE_MAX_KEYTERMS",
    "SCRIBE_MAX_KEYTERM_LENGTH",
    "SCRIBE_MAX_KEYTERM_WORDS",
    "SCRIBE_MIN_AUDIO_DURATION_MS",
    "SCRIBE_OFFICIAL_BASE_URL",
    "VIDEO_DEFAULT_MAX_BYTES",
    "VIDEO_START_TOLERANCE_MS",
    "ApiLimits",
    "ApiSettings",
    "BatchSettings",
    "ExtractionSettings",
    "GoogleBasicSettings",
    "MediaLimits",
    "PaidApiPolicy",
    "PipelineConfig",
    "ScribeSettings",
    "StageOptions",
    "VideoLimits",
    "VideoOriginSettings",
    "YouTubeLimits",
    "YouTubeOriginSettings",
    "default_job_root",
    "default_youtube_cache_root",
]

#: Version of the fingerprint payload shape. A change here intentionally
#: invalidates every previously stored fingerprint.
FINGERPRINT_VERSION: Final[str] = "1"

#: Storage format version for the job snapshot.
CONFIG_VERSION: Final[str] = "1"

_FROZEN = ConfigDict(extra="forbid", frozen=True)

StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]

#: Exact canonical provider endpoints accepted in this phase. Regional or
#: alternate hosts are intentionally *not* configurable yet; a lookalike,
#: userinfo, path, port, query or fragment is rejected at validation time.
SCRIBE_OFFICIAL_BASE_URL: Final[str] = "https://api.elevenlabs.io"

#: The single Google Cloud Translation **Basic v2** route. It serves the
#: standard Translation LLM through ``POST /language/translate/v2`` with the
#: model resource in the request body. The API key is presented only through the
#: ``X-goog-api-key`` header and is never placed in the URL or persisted.
GOOGLE_BASIC_OFFICIAL_BASE_URL: Final[str] = "https://translation.googleapis.com"
GOOGLE_BASIC_OFFICIAL_PATH: Final[str] = "/language/translate/v2"
GOOGLE_BASIC_PROVIDER: Final[str] = "google-basic"
GOOGLE_BASIC_MODEL_FAMILY: Final[str] = "general/translation-llm"
#: Route-level version so a future request/parse change invalidates stored
#: Basic-route artifacts.
GOOGLE_BASIC_ROUTE_VERSION: Final[str] = "google-basic-v1"

#: Conservative Basic v2 request ceilings. They are deliberately *not* larger
#: than the accepted pipeline batching (20 items / 4000 code points per group),
#: so selecting this route can never authorize a bigger request than the current
#: `BatchSettings` already allows. Oversized indivisible groups are refused
#: before the credential resolver or any HTTP request.
GOOGLE_BASIC_MAX_ITEMS: Final[int] = 20
GOOGLE_BASIC_MAX_CODEPOINTS: Final[int] = 4000

#: Official provider maxima/minima. Configuration may only *tighten* these; a
#: value outside the documented bound is rejected at construction time, and the
#: adapters re-check the same bounds independently at the call boundary.
SCRIBE_MAX_KEYTERMS: Final[int] = 1000
SCRIBE_MAX_KEYTERM_LENGTH: Final[int] = 50
SCRIBE_MAX_KEYTERM_WORDS: Final[int] = 5
SCRIBE_MIN_AUDIO_DURATION_MS: Final[int] = 100
SCRIBE_MAX_AUDIO_BYTES: Final[int] = 5_000_000_000

#: Significant digits and base-10 exponent bound an accepted monetary value may
#: carry. Values outside these bounds are rejected at construction instead of
#: being silently rounded near a durable budget cap (which would let an
#: unsupported cap become an invented remote-unknown failure at send time).
MONEY_MAX_DECIMAL_PRECISION: Final[int] = 28
MONEY_MAX_ABSOLUTE_EXPONENT: Final[int] = 1000

#: Default ceiling for a local source video and for the extracted audio artifact.
VIDEO_DEFAULT_MAX_BYTES: Final[int] = 2 * 1024 * 1024 * 1024

#: Phase 5 timeline policy: the container/format start, the selected video stream
#: start and the selected audio stream start must each be within this many
#: milliseconds of zero. Leading silence embedded as samples does not move a
#: stream start and therefore stays valid; genuine nonzero start times are
#: rejected with a typed error because the extracted WAV/FLAC cannot carry the
#: source offset and no mapping is proven in this version.
VIDEO_START_TOLERANCE_MS: Final[int] = 1


def _canonical_official_url(value: str, official: str) -> str:
    """Accept only the exact canonical endpoint (an optional trailing slash)."""

    if not isinstance(value, str) or value.rstrip("/") != official:
        raise ValueError(
            f"base_url must be the exact official endpoint {official!r}; "
            "regional or alternate hosts are not configurable in this phase"
        )
    return official


def default_job_root() -> str:
    """Return the default job root outside the repository.

    Callers are expected to override this with an explicit, user-owned directory.
    The default is a dedicated application-data directory so jobs are never
    confused with, mixed into or overwritten by unrelated artifacts. A job is
    resumed explicitly by pointing ``--job-root``/``JOB_DIR`` at its own
    directory.
    """

    return str(Path.home() / ".subtitleflow" / "jobs")


def default_youtube_cache_root() -> str:
    """Return the durable YouTube source cache root (outside the repository).

    The cache holds the canonical derived audio and its provenance manifest so a
    completed job can be resumed, and a repeated source command reused, without a
    fresh download. It is deliberately separate from the job root so identical
    source audio shared by several jobs is stored once.
    """

    return str(Path.home() / ".subtitleflow" / "youtube-sources")


class MediaLimits(BaseModel):
    """Hard ceilings applied before any provider call."""

    model_config = _FROZEN

    max_bytes: StrictPositiveInt = Field(default=512 * 1024 * 1024)
    max_duration_ms: StrictPositiveInt = Field(default=6 * 60 * 60 * 1000)
    min_duration_ms: StrictPositiveInt = Field(default=1)
    probe_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 30.0
    decode_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 300.0

    @model_validator(mode="after")
    def _check_bounds(self) -> "MediaLimits":
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms must not exceed max_duration_ms")
        return self


class VideoLimits(BaseModel):
    """Hard ceilings applied before any video probe, decode or extraction.

    These are safety bounds only: they reject an oversized, over-long or hung
    input *before* any model work. ``start_tolerance_ms`` encodes the Phase 5
    timeline policy (the format/video/audio start times must be present, finite
    and within the tolerance of zero). The supported window is at most
    ``+/-1`` ms: the field is capped at ``1`` here so an in-process caller cannot
    enlarge the tolerance and label a genuine one-second offset as zero-based,
    and a stricter ``0`` is allowed. A value outside the accepted range is
    refused at construction rather than silently mapped.
    """

    model_config = _FROZEN

    max_bytes: StrictPositiveInt = Field(default=VIDEO_DEFAULT_MAX_BYTES)
    max_duration_ms: StrictPositiveInt = Field(default=6 * 60 * 60 * 1000)
    min_duration_ms: StrictPositiveInt = Field(default=1)
    max_output_bytes: StrictPositiveInt = Field(default=VIDEO_DEFAULT_MAX_BYTES)
    start_tolerance_ms: Annotated[int, Field(strict=True, ge=0, le=1)] = (
        VIDEO_START_TOLERANCE_MS
    )
    probe_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 30.0
    decode_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 300.0
    extraction_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 900.0

    @model_validator(mode="after")
    def _check_bounds(self) -> "VideoLimits":
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms must not exceed max_duration_ms")
        return self


class ExtractionSettings(BaseModel):
    """Exact, deterministic audio-extraction output settings.

    The first version pins mono 16 kHz 16-bit output so a cached artifact is
    reproducible. ``container`` selects WAV (``pcm_s16le``) or FLAC (``s16``);
    every field here changes the extracted bytes and is therefore part of the
    video job fingerprint.
    """

    model_config = _FROZEN

    container: Literal["wav", "flac"] = "wav"
    sample_rate: Literal[16000] = 16000
    channels: Literal[1] = 1


class VideoOriginSettings(BaseModel):
    """Frozen provenance of one accepted video and its deterministic extraction.

    The source path is stored for resume but deliberately excluded from
    :meth:`identity_payload`, so the same bytes at a different path share the
    same extraction identity. The identity binds the source video SHA-256, the
    selected stream, the output settings, the extraction version and the tool
    version (never a temporary path).
    """

    model_config = _FROZEN

    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_size_bytes: StrictPositiveInt
    source_duration_ms: StrictPositiveInt
    selected_audio_stream_index: Annotated[int, Field(strict=True, ge=0)]
    extraction_id: str = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)

    def identity_payload(self) -> dict[str, object]:
        """Return the path-independent part of the origin used for fingerprints."""

        return {
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "source_duration_ms": self.source_duration_ms,
            "selected_audio_stream_index": self.selected_audio_stream_index,
            "extraction_id": self.extraction_id,
            "extraction_version": self.extraction_version,
            "tool_version": self.tool_version,
        }


class YouTubeLimits(BaseModel):
    """Hard ceilings applied around a single YouTube audio download.

    These are safety bounds only (like :class:`VideoLimits`) and never enter the
    fingerprint. ``max_source_bytes`` bounds the downloaded intermediate while it
    is being written, not only after; ``max_stdout_bytes``/``max_stderr_bytes``
    bound the child's pipes; ``max_retries``/``max_fragment_retries``/
    ``socket_timeout_seconds`` bound network retries.
    """

    model_config = _FROZEN

    max_source_bytes: StrictPositiveInt = Field(default=512 * 1024 * 1024)
    max_output_bytes: StrictPositiveInt = Field(default=VIDEO_DEFAULT_MAX_BYTES)
    max_duration_ms: StrictPositiveInt = Field(default=6 * 60 * 60 * 1000)
    min_duration_ms: StrictPositiveInt = Field(default=1)
    download_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 900.0
    probe_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 30.0
    decode_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 300.0
    extraction_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 900.0
    max_stdout_bytes: StrictPositiveInt = Field(default=1024 * 1024)
    max_stderr_bytes: StrictPositiveInt = Field(default=1024 * 1024)
    max_retries: Annotated[int, Field(strict=True, ge=0, le=20)] = 3
    max_fragment_retries: Annotated[int, Field(strict=True, ge=0, le=20)] = 3
    socket_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 20.0
    #: Opt-in full-video ceilings for the burned-in Turkish subtitle feature.
    #: They are safety bounds only and never enter any fingerprint; the accepted
    #: audio download keeps using ``max_source_bytes``/``download_timeout_seconds``
    #: unchanged.
    max_video_bytes: StrictPositiveInt = Field(default=4 * 1024 * 1024 * 1024)
    video_download_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 1800.0
    render_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 1800.0
    max_render_bytes: StrictPositiveInt = Field(default=4 * 1024 * 1024 * 1024)

    @model_validator(mode="after")
    def _check_bounds(self) -> "YouTubeLimits":
        if self.min_duration_ms > self.max_duration_ms:
            raise ValueError("min_duration_ms must not exceed max_duration_ms")
        return self


class YouTubeOriginSettings(BaseModel):
    """Frozen provenance of one YouTube source and its derived canonical audio.

    This is deliberately a *distinct* type from :class:`VideoOriginSettings`: a
    YouTube source is not a local video and its ``video_*`` semantics stay
    untouched. The record binds the immutable canonical video identity, the
    measured source duration, the retrieval time, the exact external tool
    versions, the downloaded intermediate's hash/size/format and the derived
    canonical audio's hash/size, plus the deterministic extraction settings.

    :meth:`identity_payload` deliberately **excludes** the retrieval timestamp and
    the untrusted title, so an equivalent replay against the same verified cache
    produces the same job identity and never depends on wall-clock time.
    """

    model_config = _FROZEN

    video_id: str = Field(pattern=r"^[A-Za-z0-9_-]{11}$")
    canonical_url: str = Field(min_length=1)
    #: Bounded, control-character-stripped, display-only title. It is data, never
    #: an instruction and never a filesystem path component.
    title: str = Field(default="", max_length=300)
    source_duration_ms: StrictPositiveInt
    retrieved_at_utc: UtcDatetime
    ytdlp_version: str = Field(min_length=1)
    ffmpeg_version: str = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    intermediate_codec: str = Field(min_length=1)
    intermediate_ext: str = Field(min_length=1)
    intermediate_sha256: Sha256Hex
    intermediate_size_bytes: StrictPositiveInt
    canonical_audio_sha256: Sha256Hex
    canonical_audio_size_bytes: StrictPositiveInt
    settings: ExtractionSettings

    @field_validator("canonical_url")
    @classmethod
    def _canonical_watch_url(cls, value: str) -> str:
        if not value.startswith("https://www.youtube.com/watch?v="):
            raise ValueError(
                "canonical_url must be the canonical "
                "https://www.youtube.com/watch?v=ID form"
            )
        return value

    @field_validator("title")
    @classmethod
    def _safe_title(cls, value: str) -> str:
        cleaned = "".join(ch for ch in value if ch >= " " and ch != "\x7f")
        return cleaned.strip()[:300]

    @model_validator(mode="after")
    def _check_origin(self) -> "YouTubeOriginSettings":
        expected = f"https://www.youtube.com/watch?v={self.video_id}"
        if self.canonical_url != expected:
            raise ValueError(
                "canonical_url must match the recorded 11-character video id"
            )
        return self

    def identity_payload(self) -> dict[str, object]:
        """Return the path/timestamp-independent part used for fingerprints."""

        return {
            "video_id": self.video_id,
            "canonical_url": self.canonical_url,
            "source_duration_ms": self.source_duration_ms,
            "ytdlp_version": self.ytdlp_version,
            "ffmpeg_version": self.ffmpeg_version,
            "extraction_version": self.extraction_version,
            "intermediate_codec": self.intermediate_codec,
            "intermediate_ext": self.intermediate_ext,
            "intermediate_sha256": self.intermediate_sha256,
            "intermediate_size_bytes": self.intermediate_size_bytes,
            "canonical_audio_sha256": self.canonical_audio_sha256,
            "canonical_audio_size_bytes": self.canonical_audio_size_bytes,
            "settings": self.settings.model_dump(mode="json"),
        }


class BatchSettings(BaseModel):
    """Deterministic MT grouping bounds. Segments are never split."""

    model_config = _FROZEN

    max_items_per_group: StrictPositiveInt = Field(default=20)
    max_chars_per_group: StrictPositiveInt = Field(default=4000)


class StageOptions(BaseModel):
    """Toggles that affect stage execution and therefore the fingerprint.

    ``verify_audio_decode`` is intentionally not configurable: the processing
    pipeline always requires a full decode of the audio stream, and a disabled
    decode would let a truncated or corrupt file reach a paid provider. The
    stricter ``Literal[True]`` type rejects ``False`` at construction time.
    """

    model_config = _FROZEN

    stt_send_keyterms: bool = True
    verify_audio_decode: Literal[True] = True


NonNegativeDecimal = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


def _reject_bool_numeric(value: object) -> object:
    """A boolean is never an acceptable monetary cap or estimate."""

    if isinstance(value, bool):
        raise ValueError("monetary limits must be finite decimals, not booleans")
    return value


class ScribeSettings(BaseModel):
    """Exact output-affecting ElevenLabs Scribe v2 request options.

    The provider endpoint is fixed to the canonical official host; this phase
    does not accept a broad host configuration. Every field here changes the
    request or the response contract and is therefore part of the job
    fingerprint. The cap fields may only *tighten* the official maxima.
    """

    model_config = _FROZEN

    base_url: str = Field(default=SCRIBE_OFFICIAL_BASE_URL, min_length=1)
    model_id: Literal["scribe_v2"] = "scribe_v2"
    timestamps_granularity: Literal["word"] = "word"
    diarize: bool = True
    tag_audio_events: bool = False
    #: ``enable_logging`` defaults to true at the provider; ``false`` is
    #: enterprise-only, so the *actual chosen* value is recorded, never assumed.
    enable_logging: bool = True
    #: Conservative confidence floor for non-empty speech. This is an initial
    #: policy threshold, not a measured-quality result; a known-but-low value is
    #: routed to human review instead of paid MT.
    min_language_probability: Annotated[
        float, Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False)
    ] = 0.8
    min_audio_duration_ms: Annotated[
        int, Field(strict=True, ge=SCRIBE_MIN_AUDIO_DURATION_MS)
    ] = SCRIBE_MIN_AUDIO_DURATION_MS
    max_audio_bytes: Annotated[
        int, Field(strict=True, gt=0, le=SCRIBE_MAX_AUDIO_BYTES)
    ] = SCRIBE_MAX_AUDIO_BYTES
    max_keyterms: Annotated[
        int, Field(strict=True, gt=0, le=SCRIBE_MAX_KEYTERMS)
    ] = SCRIBE_MAX_KEYTERMS
    max_keyterm_length: Annotated[
        int, Field(strict=True, gt=0, le=SCRIBE_MAX_KEYTERM_LENGTH)
    ] = SCRIBE_MAX_KEYTERM_LENGTH
    max_keyterm_words: Annotated[
        int, Field(strict=True, gt=0, le=SCRIBE_MAX_KEYTERM_WORDS)
    ] = SCRIBE_MAX_KEYTERM_WORDS

    @field_validator("base_url")
    @classmethod
    def _official_only(cls, value: str) -> str:
        return _canonical_official_url(value, SCRIBE_OFFICIAL_BASE_URL)


class GoogleBasicSettings(BaseModel):
    """Exact output-affecting Google Translation **Basic v2** request options.

    This is the product's single MT route. It authenticates with an API key
    presented only through the ``X-goog-api-key`` header, sends the
    ordered text list as ``q`` together with ``source``/``target``/``format`` and
    the full ``general/translation-llm`` model resource, and parses
    ``data.translations[].translatedText``. Basic v2 may omit the returned model;
    the requested resource stays in the stored identity and the missing reported
    model is recorded honestly instead of being fabricated.

    The region is fixed to ``global`` and the model family to
    ``general/translation-llm``: there is no user-configurable arbitrary endpoint
    or model. The request ceilings may only *tighten* the conservative Basic
    bounds and are part of the job fingerprint. The API key never lives here.
    """

    model_config = _FROZEN

    base_url: str = Field(default=GOOGLE_BASIC_OFFICIAL_BASE_URL, min_length=1)
    project: str | None = Field(default=None, min_length=1)
    location: Literal["global"] = "global"
    model_family: Literal["general/translation-llm"] = GOOGLE_BASIC_MODEL_FAMILY
    target_language: str = TARGET_LANGUAGE_CODE
    route_version: str = Field(default=GOOGLE_BASIC_ROUTE_VERSION, min_length=1)
    max_items_per_request: Annotated[
        int, Field(strict=True, gt=0, le=GOOGLE_BASIC_MAX_ITEMS)
    ] = GOOGLE_BASIC_MAX_ITEMS
    max_codepoints_per_request: Annotated[
        int, Field(strict=True, gt=0, le=GOOGLE_BASIC_MAX_CODEPOINTS)
    ] = GOOGLE_BASIC_MAX_CODEPOINTS

    @field_validator("base_url")
    @classmethod
    def _official_only(cls, value: str) -> str:
        return _canonical_official_url(value, GOOGLE_BASIC_OFFICIAL_BASE_URL)

    @field_validator("project")
    @classmethod
    def _project_charset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        if not set(value) <= allowed:
            raise ValueError("project id contains unexpected characters")
        return value

    @field_validator("target_language")
    @classmethod
    def _turkish_only(cls, value: str) -> str:
        normalized = normalize_language(value)
        if normalized.canonical_code != TARGET_LANGUAGE_CODE:
            raise ValueError("Google Basic target must be Turkish ('tr')")
        return TARGET_LANGUAGE_CODE

    @property
    def model_resource(self) -> str:
        """Canonical requested model resource (requires a configured project)."""

        if self.project is None:
            raise ValueError("google basic project is not configured")
        return f"projects/{self.project}/locations/{self.location}/models/{self.model_family}"

    @property
    def provider_name(self) -> str:
        return GOOGLE_BASIC_PROVIDER

    @property
    def model_name(self) -> str:
        return self.model_family


class ApiLimits(BaseModel):
    """Bounded HTTP timeouts, attempts and backoff for provider adapters."""

    model_config = _FROZEN

    connect_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 10.0
    read_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 300.0
    write_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 60.0
    pool_timeout_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 10.0
    max_attempts: Annotated[int, Field(strict=True, ge=1, le=10)] = 4
    backoff_base_seconds: Annotated[
        float, Field(strict=True, ge=0, le=60, allow_inf_nan=False)
    ] = 0.5
    backoff_max_seconds: Annotated[
        float, Field(strict=True, gt=0, le=600, allow_inf_nan=False)
    ] = 30.0
    retry_after_ceiling_seconds: Annotated[
        float, Field(strict=True, gt=0, le=3600, allow_inf_nan=False)
    ] = 120.0

    @model_validator(mode="after")
    def _check_backoff(self) -> "ApiLimits":
        if self.backoff_base_seconds > self.backoff_max_seconds:
            raise ValueError("backoff_base_seconds must not exceed backoff_max_seconds")
        return self

    def timeout(self):  # noqa: ANN201 - httpx.Timeout built lazily by callers
        """Return the four timeout components as a plain tuple of seconds."""

        return (
            self.connect_timeout_seconds,
            self.read_timeout_seconds,
            self.write_timeout_seconds,
            self.pool_timeout_seconds,
        )


class PaidApiPolicy(BaseModel):
    """Explicit opt-in and durable reservation caps for paid provider calls.

    A hard-coded price is never stored. ``per_call_upper_bound`` is a caller
    supplied conservative worst-case amount for one HTTP attempt; it is a local
    reservation only and never claims to predict the provider invoice. All
    amounts are finite decimals; booleans are rejected.
    """

    model_config = _FROZEN

    #: ``strict=True`` rejects truthy strings/ints so a paid opt-in can never be
    #: enabled by ``"false"`` or ``1``.
    allow_paid_api_calls: Annotated[bool, Field(strict=True)] = False
    total_reservation_cap: NonNegativeDecimal | None = None
    per_call_cap: NonNegativeDecimal | None = None
    per_call_upper_bound: NonNegativeDecimal | None = None
    currency: str | None = None

    @field_validator(
        "total_reservation_cap",
        "per_call_cap",
        "per_call_upper_bound",
        mode="before",
    )
    @classmethod
    def _no_bool_amounts(cls, value: object) -> object:
        return _reject_bool_numeric(value)

    @field_validator(
        "total_reservation_cap",
        "per_call_cap",
        "per_call_upper_bound",
    )
    @classmethod
    def _representable_amounts(cls, value: Decimal | None) -> Decimal | None:
        """Reject precision/scale/exponents that exact arithmetic cannot accept.

        A cap carrying more than ``MONEY_MAX_DECIMAL_PRECISION`` significant
        digits, or an exponent beyond ``MONEY_MAX_ABSOLUTE_EXPONENT``, is refused
        here, before any credential resolver, reservation or HTTP request, so an
        invalid policy can never surface as an invented remote-unknown outcome.
        """

        if value is None:
            return None
        if not value.is_finite() or value < 0:
            raise ValueError("monetary limits must be finite non-negative decimals")
        _sign, digits, exponent = value.as_tuple()
        if len(digits) > MONEY_MAX_DECIMAL_PRECISION:
            raise ValueError(
                f"monetary limits must carry at most {MONEY_MAX_DECIMAL_PRECISION} "
                "significant digits; refusing an amount that cannot be summed exactly"
            )
        if abs(int(exponent)) > MONEY_MAX_ABSOLUTE_EXPONENT:
            raise ValueError(
                f"monetary limits must stay within a +/-{MONEY_MAX_ABSOLUTE_EXPONENT} "
                "base-10 exponent bound"
            )
        return value

    @field_validator("currency")
    @classmethod
    def _currency_shape(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 3 or not value.isalpha() or not value.isupper():
            raise ValueError("currency must be a three-letter uppercase code")
        return value

    @property
    def configured(self) -> bool:
        return (
            self.total_reservation_cap is not None
            and self.per_call_cap is not None
            and self.per_call_upper_bound is not None
            and self.currency is not None
        )

    @model_validator(mode="after")
    def _check_policy(self) -> "PaidApiPolicy":
        amounts = (
            self.total_reservation_cap,
            self.per_call_cap,
            self.per_call_upper_bound,
        )
        if any(amount is not None for amount in amounts) and not self.configured:
            raise ValueError(
                "paid-call policy requires total_reservation_cap, per_call_cap, "
                "per_call_upper_bound and currency together"
            )
        if self.configured:
            assert self.total_reservation_cap is not None
            assert self.per_call_cap is not None
            assert self.per_call_upper_bound is not None
            if self.total_reservation_cap <= 0 or self.per_call_cap <= 0:
                raise ValueError("reservation caps must be strictly positive")
            if self.per_call_upper_bound <= 0:
                raise ValueError("per_call_upper_bound must be strictly positive")
            if self.per_call_upper_bound > self.per_call_cap:
                raise ValueError(
                    "per_call_upper_bound must not exceed the per-call cap"
                )
            if self.per_call_cap > self.total_reservation_cap:
                raise ValueError(
                    "per_call_cap must not exceed total_reservation_cap"
                )
        return self


class ApiSettings(BaseModel):
    """Typed provider/adapter configuration carried by the job snapshot."""

    model_config = _FROZEN

    scribe: ScribeSettings = Field(default_factory=ScribeSettings)
    limits: ApiLimits = Field(default_factory=ApiLimits)
    paid: PaidApiPolicy = Field(default_factory=PaidApiPolicy)


class PipelineConfig(BaseModel):
    """Immutable snapshot of everything that affects a job's output."""

    model_config = _FROZEN

    config_version: str = Field(default=CONFIG_VERSION, min_length=1)
    schema_version: str = Field(default=SCHEMA_VERSION, min_length=1)
    segmenter_version: str = Field(
        default=SEGMENTER_ALGORITHM_VERSION, min_length=1
    )
    source_language_hint: str | None = None
    target_language: str = TARGET_LANGUAGE_CODE
    stt: ProviderIdentity
    mt: ProviderIdentity
    keyterms: tuple[str, ...] = ()
    keyterms_version: str | None = None
    stage_options: StageOptions = Field(default_factory=StageOptions)
    batch: BatchSettings = Field(default_factory=BatchSettings)
    media: MediaLimits = Field(default_factory=MediaLimits)
    #: Deterministic segmenter thresholds; changing any of them changes the
    #: fingerprint even if ``segmenter_version`` was left at its default.
    segmenter: SegmenterSettings = Field(default_factory=SegmenterSettings)
    #: Provider adapter options, HTTP bounds and the paid-call reservation
    #: policy. Secrets never live here.
    api: ApiSettings = Field(default_factory=ApiSettings)
    #: Fixed Google Translation **Basic v2** (API key) MT route settings. There
    #: is no other MT route: the snapshot always carries it. It binds the
    #: project/model/route/batch/paid policy and deliberately excludes the key.
    google_basic: GoogleBasicSettings = Field(default_factory=GoogleBasicSettings)
    #: Video safety ceilings and exact extraction output settings. ``video`` is
    #: only a bound and never enters the fingerprint; ``video_extraction``
    #: changes the extracted bytes and is included only for a video job.
    video: VideoLimits = Field(default_factory=VideoLimits)
    video_extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    #: Explicit video origin. ``None`` means an ordinary ready-audio job and
    #: keeps every Phase 2/3/4 fingerprint byte-identical.
    video_origin: VideoOriginSettings | None = None
    #: YouTube source safety ceilings and the durable source cache root. Both are
    #: bounds/locations only and never enter the fingerprint.
    youtube: YouTubeLimits = Field(default_factory=YouTubeLimits)
    youtube_cache_root: str = Field(
        default_factory=default_youtube_cache_root, min_length=1
    )
    #: Explicit YouTube source origin. ``None`` means an ordinary ready-audio or
    #: local-video job; only a genuine YouTube source adds the conditional
    #: fingerprint block, so every inherited fingerprint stays byte-identical.
    youtube_origin: YouTubeOriginSettings | None = None
    job_root: str = Field(default_factory=default_job_root, min_length=1)

    @field_validator("config_version")
    @classmethod
    def _supported_config_version(cls, value: str) -> str:
        if value != CONFIG_VERSION:
            raise ValueError(
                f"unsupported config_version {value!r}; this build writes and "
                f"reads {CONFIG_VERSION!r}"
            )
        return value

    @field_validator("schema_version")
    @classmethod
    def _supported_schema_version(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value!r}; this build writes and "
                f"reads {SCHEMA_VERSION!r}"
            )
        return value

    @field_validator("source_language_hint")
    @classmethod
    def _canonical_hint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = normalize_language(value)
        if normalized.canonical_code is None:
            raise ValueError(
                f"source_language_hint is not a supported pilot language: {value!r}"
            )
        return normalized.canonical_code

    @field_validator("target_language")
    @classmethod
    def _turkish_target(cls, value: str) -> str:
        normalized = normalize_language(value)
        if normalized.canonical_code != TARGET_LANGUAGE_CODE:
            raise ValueError(
                f"target_language must be Turkish ({TARGET_LANGUAGE_CODE!r}); "
                f"got {value!r}"
            )
        return TARGET_LANGUAGE_CODE

    @field_validator("keyterms")
    @classmethod
    def _validate_keyterms(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for term in value:
            if not isinstance(term, str):
                raise ValueError("keyterms must be strings")
            if term == "":
                raise ValueError("keyterms must not be empty strings")
            if term in seen:
                raise ValueError(f"duplicate keyterm: {term!r}")
            seen.add(term)
        return value

    @model_validator(mode="after")
    def _check_identities(self) -> "PipelineConfig":
        if self.stt.kind is not ProviderKind.stt:
            raise ValueError("stt identity must have kind='stt'")
        if self.mt.kind is not ProviderKind.mt:
            raise ValueError("mt identity must have kind='mt'")
        if self.video_origin is not None and self.youtube_origin is not None:
            raise ValueError(
                "a job cannot be bound to both a local video origin and a "
                "YouTube source origin"
            )
        return self

    @property
    def job_root_path(self) -> Path:
        return Path(self.job_root)

    def with_job_root(self, job_root: str | Path) -> "PipelineConfig":
        """Return a copy bound to a different job root (not part of the fingerprint).

        The copy is re-validated from the dumped dictionary so a config that was
        previously produced by ``model_copy(update=...)`` (which bypasses field
        validators) cannot smuggle an invalid nested setting through this trust
        boundary.
        """

        return PipelineConfig.model_validate(
            {**self.model_dump(), "job_root": str(job_root)}
        )

    def with_video_origin(self, origin: VideoOriginSettings) -> "PipelineConfig":
        """Return a copy carrying an explicit video origin.

        Re-validated from the dumped form like :meth:`with_job_root`, so a
        ``model_copy``-produced snapshot cannot smuggle an invalid nested origin
        through this trust boundary. The origin is what binds a video job's
        fingerprint to the original source identity and the deterministic
        extraction, distinct from an ordinary audio-only job.
        """

        return PipelineConfig.model_validate(
            {**self.model_dump(), "video_origin": origin.model_dump(mode="json")}
        )

    def with_youtube_origin(self, origin: "YouTubeOriginSettings") -> "PipelineConfig":
        """Return a copy carrying an explicit YouTube source origin.

        Re-validated from the dumped form like :meth:`with_video_origin`, so a
        ``model_copy``-produced snapshot cannot smuggle an invalid nested origin
        through this trust boundary.
        """

        return PipelineConfig.model_validate(
            {**self.model_dump(), "youtube_origin": origin.model_dump(mode="json")}
        )

    # ------------------------------------------------------------------ #
    # Deterministic fingerprints
    # ------------------------------------------------------------------ #
    def _fingerprint_payload(self, audio_sha256: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "fingerprint_version": FINGERPRINT_VERSION,
            "config_version": self.config_version,
            "schema_version": self.schema_version,
            "segmenter_version": self.segmenter_version,
            "source_language_hint": self.source_language_hint,
            "target_language": self.target_language,
            "stt": {
                "kind": self.stt.kind.value,
                "provider": self.stt.provider,
                "model": self.stt.model,
            },
            "mt": {
                "kind": self.mt.kind.value,
                "provider": self.mt.provider,
                "model": self.mt.model,
            },
            "keyterms": list(self.keyterms),
            "keyterms_version": self.keyterms_version,
            "stage_options": self.stage_options.model_dump(mode="json"),
            "batch": self.batch.model_dump(mode="json"),
            "media": self.media.model_dump(mode="json"),
            "segmenter": self.segmenter.model_dump(mode="json"),
            "api": self.api.model_dump(mode="json"),
            "audio_sha256": audio_sha256,
        }
        # The fixed Google Basic MT route is always part of the identity. It
        # carries no secret: the key lives solely in a lazy resolver closure.
        payload["google_basic"] = self.google_basic.model_dump(mode="json")
        # Only a genuine video origin adds the extraction identity to the
        # fingerprint. The source path is excluded so the same bytes at another
        # path are the same job; the audio-only payload above is untouched.
        if self.video_origin is not None:
            payload["video_extraction"] = self.video_extraction.model_dump(mode="json")
            payload["video_origin"] = self.video_origin.identity_payload()
        # Only a genuine YouTube source adds the (timestamp/path-independent)
        # source identity, so two different videos that derive to byte-identical
        # audio remain distinct jobs while every audio-only and local-video
        # fingerprint stays byte-identical.
        if self.youtube_origin is not None:
            payload["youtube_origin"] = self.youtube_origin.identity_payload()
        return payload

    def _digest(self, audio_sha256: str | None) -> str:
        payload = self._fingerprint_payload(audio_sha256)
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def config_fingerprint(self) -> str:
        """Fingerprint of every setting except the audio identity."""

        return self._digest(None)

    def job_fingerprint(self, audio_sha256: str) -> str:
        """Fingerprint of the configuration plus one audio file's SHA-256."""

        if not isinstance(audio_sha256, str) or len(audio_sha256) != 64:
            raise ValueError("audio_sha256 must be a 64-character hex string")
        return self._digest(audio_sha256.lower())
