"""CLI-facing environment loading and runtime configuration.

The CLI only. Importing :mod:`subtitle_flow` never touches this module, and this
module never imports ``typer`` or ``httpx`` at import time, so a core
installation stays dependency-light.

The product has one fixed inference route: ElevenLabs Scribe v2 for speech to
text and the Google Translation **Basic v2** API-key route (Translation LLM) for
Turkish translation. There is no route, provider or model selection: the STT and
MT identities are derived here, never asked of the user.

Precedence is explicit and documented: **CLI argument > process environment >
the chosen ``.env`` file > safe default**. A blank placeholder (``KEY=``) always
means "unset" and never becomes an empty string, ``"false"`` is never treated as
truthy, and only the documented keys below are ever read. Secrets are kept out of
every snapshot, dataclass and exception: they live only inside lazy credential
resolver closures that read the environment dict at call time.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from subtitle_flow.config import (
    GOOGLE_BASIC_MODEL_FAMILY,
    GOOGLE_BASIC_PROVIDER,
    ApiSettings,
    ExtractionSettings,
    GoogleBasicSettings,
    PaidApiPolicy,
    PipelineConfig,
    ScribeSettings,
)
from subtitle_flow.languages import normalize_language
from subtitle_flow.providers.google_basic import GoogleTranslationBasicProvider
from subtitle_flow.providers.mt_base import MachineTranslationProvider
from subtitle_flow.providers.scribe import ScribeV2STTProvider
from subtitle_flow.providers.stt_base import SpeechToTextProvider
from subtitle_flow.schemas import ProviderIdentity, ProviderKind

__all__ = [
    "CANONICAL_GOOGLE_BASIC_MT_IDENTITY",
    "CANONICAL_MT_IDENTITY",
    "CANONICAL_STT_IDENTITY",
    "DEFAULT_ENV_FILE_NAME",
    "KNOWN_ENV_KEYS",
    "CliEnvironment",
    "CliOverrides",
    "CliSettings",
    "ConfigError",
    "RuntimeProviders",
    "build_pipeline_config",
    "build_settings",
    "default_runtime_factory",
    "elevenlabs_resolver",
    "google_basic_resolver",
    "load_environment",
    "parse_strict_bool",
    "parse_strict_decimal",
    "resolve_value",
]

DEFAULT_ENV_FILE_NAME: Final[str] = ".env"

CANONICAL_STT_IDENTITY: Final[ProviderIdentity] = ProviderIdentity(
    kind=ProviderKind.stt, provider="elevenlabs", model="scribe_v2"
)
#: The fixed MT identity (``provider`` names the transport, the request model is
#: the full project resource family).
CANONICAL_GOOGLE_BASIC_MT_IDENTITY: Final[ProviderIdentity] = ProviderIdentity(
    kind=ProviderKind.mt,
    provider=GOOGLE_BASIC_PROVIDER,
    model=GOOGLE_BASIC_MODEL_FAMILY,
)
#: Backwards-compatible alias for the single canonical MT identity.
CANONICAL_MT_IDENTITY: Final[ProviderIdentity] = CANONICAL_GOOGLE_BASIC_MT_IDENTITY

#: Only these names are ever read. Anything else in ``.env`` is ignored.
KNOWN_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ELEVENLABS_API_KEY",
        "GOOGLE_TRANSLATION_API_KEY",
        "GOOGLE_TRANSLATION_PROJECT",
        "JOB_DIR",
        "SOURCE_LANGUAGE",
        "ALLOW_PAID_API_CALLS",
        "TOTAL_RESERVATION_CAP",
        "PER_CALL_CAP",
        "PER_CALL_UPPER_BOUND",
        "RESERVATION_CURRENCY",
        "VIDEO_CONTAINER",
        "YOUTUBE_YTDLP_BIN",
        "YOUTUBE_CACHE_DIR",
    }
)

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    """A CLI configuration value is missing, malformed or unsupported."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CliEnvironment:
    """Resolved ``.env``/process values with crediting-safe ``repr``.

    Process-environment values take precedence over ``.env`` values. A value is
    only present when it is a non-blank string, so a blank placeholder is
    indistinguishable from a missing key.
    """

    def __init__(
        self,
        *,
        file_values: Mapping[str, str],
        process_values: Mapping[str, str],
        env_file: str | None,
        used_env_file: bool,
    ) -> None:
        self._file_values = dict(file_values)
        self._process_values = dict(process_values)
        self.env_file = env_file
        self.used_env_file = used_env_file

    def __repr__(self) -> str:
        return (
            f"CliEnvironment(env_file={self.env_file!r}, "
            f"used_env_file={self.used_env_file}, "
            f"keys={sorted(KNOWN_ENV_KEYS)})"
        )

    def __str__(self) -> str:  # pragma: no cover - defensive mirror of repr
        return self.__repr__()

    def lookup(self, key: str) -> str | None:
        """Return the non-blank value for ``key`` (process env wins)."""

        if key not in KNOWN_ENV_KEYS:
            raise ConfigError("UNKNOWN_KEY", f"{key!r} is not a documented setting")
        value = self._process_values.get(key)
        if value is None:
            value = self._file_values.get(key)
        return value


@dataclass(frozen=True)
class CliOverrides:
    """Values explicitly supplied on the command line (never secrets)."""

    source_language: str | None = None
    keyterms: tuple[str, ...] = ()
    job_root: str | None = None
    allow_paid_api_calls: bool | None = None
    total_reservation_cap: str | None = None
    per_call_cap: str | None = None
    per_call_upper_bound: str | None = None
    currency: str | None = None
    video_container: str | None = None
    youtube_ytdlp_bin: str | None = None


@dataclass(frozen=True)
class CliSettings:
    """Secret-free runtime configuration derived from the CLI invocation."""

    job_root: str
    source_language: str | None
    keyterms: tuple[str, ...]
    stt: ProviderIdentity
    mt: ProviderIdentity
    api: ApiSettings
    google_basic: GoogleBasicSettings
    env_file: str | None
    used_env_file: bool
    has_elevenlabs_key: bool
    has_google_translation_key: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
    video_extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    video_container_explicit: bool = False
    youtube_ytdlp_bin: str = "yt-dlp"
    youtube_cache_root: str = ""

    def __repr__(self) -> str:
        return (
            f"CliSettings(job_root={self.job_root!r}, "
            f"source_language={self.source_language!r}, "
            f"keyterms={len(self.keyterms)}, stt={self.stt!r}, mt={self.mt!r}, "
            f"env_file={self.env_file!r}, used_env_file={self.used_env_file}, "
            f"has_elevenlabs_key={self.has_elevenlabs_key}, "
            f"has_google_translation_key={self.has_google_translation_key}, "
            f"notes={len(self.notes)})"
        )


@dataclass(frozen=True)
class RuntimeProviders:
    """The concrete provider pair bound to one job configuration."""

    stt: SpeechToTextProvider
    mt: MachineTranslationProvider


def parse_strict_bool(value: str, *, name: str) -> bool:
    """Parse a strict boolean; ``"false"``/``"0"`` are never truthy."""

    text = value.strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(
        "INVALID_BOOLEAN",
        f"{name} must be one of true/false; refusing to guess",
    )


def parse_strict_decimal(value: str, *, name: str) -> Decimal:
    """Parse a finite, non-negative decimal without silent coercion."""

    text = value.strip()
    if text == "":
        raise ConfigError("INVALID_DECIMAL", f"{name} must not be blank")
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ConfigError(
            "INVALID_DECIMAL", f"{name} must be a decimal number"
        ) from exc
    if not parsed.is_finite() or parsed < 0:
        raise ConfigError(
            "INVALID_DECIMAL",
            f"{name} must be a finite non-negative decimal",
        )
    return parsed


def load_environment(
    env_file: str | Path | None = None,
    *,
    explicit: bool = False,
    use_default: bool = True,
    environ: Mapping[str, str] | None = None,
) -> CliEnvironment:
    """Load the chosen ``.env`` deliberately, with interpolation disabled.

    ``explicit`` marks a user-supplied path: a missing file is then an error
    instead of the silent absence of the default file. Only the current working
    directory's ``.env`` is considered by default (unless ``use_default`` is
    false); parent directories are never searched implicitly.
    """

    process = dict(os.environ if environ is None else environ)
    process_values: dict[str, str] = {}
    for key in KNOWN_ENV_KEYS:
        raw = process.get(key)
        if isinstance(raw, str) and raw.strip() != "":
            process_values[key] = raw

    selected: Path | None
    if env_file is not None:
        selected = Path(env_file)
    elif use_default:
        selected = Path.cwd() / DEFAULT_ENV_FILE_NAME
    else:
        selected = None

    if explicit and selected is not None and not selected.is_file():
        raise ConfigError(
            "ENV_FILE_MISSING", f"the requested env file does not exist: {selected}"
        )

    file_values: dict[str, str] = {}
    used_env_file = False
    if selected is not None and selected.is_file():
        file_values = _read_dotenv(selected)
        used_env_file = True

    return CliEnvironment(
        file_values=file_values,
        process_values=process_values,
        env_file=str(selected) if selected is not None and used_env_file else None,
        used_env_file=used_env_file,
    )


def _read_dotenv(path: Path) -> dict[str, str]:
    try:
        from dotenv import dotenv_values
    except ImportError as exc:  # pragma: no cover - cli extra is required
        raise ConfigError(
            "CLI_EXTRA_MISSING",
            "reading .env requires python-dotenv; install subtitle-flow[cli]",
        ) from exc
    try:
        # interpolation=False: values are taken literally, never expanded.
        raw = dotenv_values(path, interpolate=False, encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        # Normalize raw decoder/dotenv failures into a safe configuration error:
        # the offending bytes (which may be a mistyped secret) are never echoed.
        raise ConfigError(
            "ENV_FILE_UNREADABLE",
            f"the selected env file could not be read as UTF-8 text: {path}",
        ) from exc
    values: dict[str, str] = {}
    for key, value in raw.items():
        if key not in KNOWN_ENV_KEYS:
            continue
        if value is None or value.strip() == "":
            continue
        values[key] = value
    return values


def resolve_value(
    environment: CliEnvironment,
    key: str,
    cli_value: str | None,
    default: str | None = None,
) -> str | None:
    """Resolve one setting with CLI > process env > ``.env`` > default order."""

    if cli_value is not None:
        return cli_value
    value = environment.lookup(key)
    if value is None:
        return default
    return value


def _build_paid_policy(
    *,
    allow_paid: bool | None,
    total_cap: str | None,
    per_call_cap: str | None,
    per_call_upper_bound: str | None,
    currency: str | None,
) -> PaidApiPolicy:
    if not allow_paid:
        return PaidApiPolicy(allow_paid_api_calls=False)
    amounts = {
        "TOTAL_RESERVATION_CAP": total_cap,
        "PER_CALL_CAP": per_call_cap,
        "PER_CALL_UPPER_BOUND": per_call_upper_bound,
        "RESERVATION_CURRENCY": currency,
    }
    missing = [name for name, value in amounts.items() if value is None]
    if missing:
        raise ConfigError(
            "PAID_POLICY_INCOMPLETE",
            "enabling paid API calls requires " + ", ".join(sorted(missing)),
        )
    try:
        return PaidApiPolicy(
            allow_paid_api_calls=True,
            total_reservation_cap=parse_strict_decimal(
                total_cap, name="TOTAL_RESERVATION_CAP"
            ),
            per_call_cap=parse_strict_decimal(per_call_cap, name="PER_CALL_CAP"),
            per_call_upper_bound=parse_strict_decimal(
                per_call_upper_bound, name="PER_CALL_UPPER_BOUND"
            ),
            currency=(currency or "").strip(),
        )
    except ValidationError as exc:
        raise ConfigError(
            "PAID_POLICY_INVALID",
            "the paid-call reservation caps are not valid; check the amounts, "
            "currency and ordering (total >= per-call >= upper bound)",
        ) from exc


def _build_video_extraction(
    environment: CliEnvironment, cli: CliOverrides
) -> tuple[ExtractionSettings, bool]:
    """Resolve the deterministic extraction container (wav/flac).

    Only the container is configurable; the sample rate and channel count are
    pinned by :class:`ExtractionSettings` so a cached extraction stays
    reproducible.
    """

    raw = resolve_value(environment, "VIDEO_CONTAINER", cli.video_container, None)
    explicit = raw is not None
    if raw is None:
        return ExtractionSettings(), False
    container = raw.strip().lower()
    if container not in {"wav", "flac"}:
        raise ConfigError(
            "INVALID_VIDEO_CONTAINER",
            f"VIDEO_CONTAINER must be 'wav' or 'flac' (got {container!r})",
        )
    return ExtractionSettings(container=container), explicit


def _build_youtube_locations(
    environment: CliEnvironment, cli: CliOverrides
) -> tuple[str, str]:
    """Resolve the external ``yt-dlp`` binary and the durable source cache root.

    Neither value is a secret and neither enters a fingerprint. The binary must
    be an installed external tool: this project does not import ``yt_dlp`` and
    never updates it. The cache root defaults outside the repository.
    """

    from subtitle_flow.config import default_youtube_cache_root

    binary = resolve_value(
        environment, "YOUTUBE_YTDLP_BIN", cli.youtube_ytdlp_bin, "yt-dlp"
    )
    assert binary is not None
    binary = binary.strip() or "yt-dlp"
    cache_root = resolve_value(environment, "YOUTUBE_CACHE_DIR", None, None)
    if cache_root is None or cache_root.strip() == "":
        cache_root = default_youtube_cache_root()
    cache_root = str(Path(cache_root).expanduser())
    return binary, cache_root


def build_settings(
    environment: CliEnvironment, *, overrides: CliOverrides | None = None
) -> CliSettings:
    """Resolve a secret-free :class:`CliSettings` from the environment and CLI.

    The fixed Scribe/Google Basic identities are derived here; the user never
    selects a provider or model. Missing credentials and a disabled paid policy
    are reported in ``notes`` and refused later, immediately before dispatch, so
    an offline reuse of a completed job never demands a fresh paid opt-in.
    """

    cli = overrides if overrides is not None else CliOverrides()
    notes: list[str] = []
    video_extraction, video_container_explicit = _build_video_extraction(
        environment, cli
    )
    youtube_ytdlp_bin, youtube_cache_root = _build_youtube_locations(environment, cli)

    job_root = resolve_value(environment, "JOB_DIR", cli.job_root, None)
    if job_root is None or job_root.strip() == "":
        from subtitle_flow.config import default_job_root

        job_root = default_job_root()
    job_root = str(Path(job_root).expanduser())

    source_language_raw = resolve_value(
        environment, "SOURCE_LANGUAGE", cli.source_language, None
    )
    source_language: str | None = None
    if source_language_raw is not None and source_language_raw.strip() != "":
        normalized = normalize_language(source_language_raw)
        if normalized.canonical_code is None:
            raise ConfigError(
                "UNSUPPORTED_LANGUAGE",
                f"source language is not a supported pilot language: "
                f"{source_language_raw!r}",
            )
        source_language = normalized.canonical_code

    allow_paid_raw = resolve_value(environment, "ALLOW_PAID_API_CALLS", None, None)
    allow_paid: bool | None = None
    if cli.allow_paid_api_calls is not None:
        allow_paid = cli.allow_paid_api_calls
    elif allow_paid_raw is not None:
        allow_paid = parse_strict_bool(allow_paid_raw, name="ALLOW_PAID_API_CALLS")

    paid = _build_paid_policy(
        allow_paid=allow_paid,
        total_cap=resolve_value(
            environment, "TOTAL_RESERVATION_CAP", cli.total_reservation_cap, None
        ),
        per_call_cap=resolve_value(
            environment, "PER_CALL_CAP", cli.per_call_cap, None
        ),
        per_call_upper_bound=resolve_value(
            environment, "PER_CALL_UPPER_BOUND", cli.per_call_upper_bound, None
        ),
        currency=resolve_value(
            environment, "RESERVATION_CURRENCY", cli.currency, None
        ),
    )

    project = environment.lookup("GOOGLE_TRANSLATION_PROJECT")
    project = project.strip() if project is not None and project.strip() != "" else None
    try:
        api = ApiSettings(scribe=ScribeSettings(), paid=paid)
        google_basic = GoogleBasicSettings(project=project)
    except ValidationError as exc:
        raise ConfigError(
            "INVALID_CONFIG",
            "GOOGLE_TRANSLATION_PROJECT or another adapter setting is invalid",
        ) from exc

    has_elevenlabs_key = environment.lookup("ELEVENLABS_API_KEY") is not None
    has_google_translation_key = (
        environment.lookup("GOOGLE_TRANSLATION_API_KEY") is not None
    )
    if project is None:
        notes.append(
            "GOOGLE_TRANSLATION_PROJECT is unset; the paid MT route cannot run yet"
        )
    if not has_elevenlabs_key:
        notes.append("ELEVENLABS_API_KEY is unset; the paid STT route cannot run yet")
    if not has_google_translation_key:
        notes.append(
            "GOOGLE_TRANSLATION_API_KEY is unset; the paid MT route cannot run yet"
        )
    if not paid.allow_paid_api_calls:
        notes.append("paid API calls are disabled by default; no request will be sent")

    return CliSettings(
        job_root=job_root,
        source_language=source_language,
        keyterms=tuple(cli.keyterms),
        stt=CANONICAL_STT_IDENTITY,
        mt=CANONICAL_GOOGLE_BASIC_MT_IDENTITY,
        api=api,
        google_basic=google_basic,
        env_file=environment.env_file,
        used_env_file=environment.used_env_file,
        has_elevenlabs_key=has_elevenlabs_key,
        has_google_translation_key=has_google_translation_key,
        notes=tuple(notes),
        video_extraction=video_extraction,
        video_container_explicit=video_container_explicit,
        youtube_ytdlp_bin=youtube_ytdlp_bin,
        youtube_cache_root=youtube_cache_root,
    )


def build_pipeline_config(settings: CliSettings) -> PipelineConfig:
    """Build the immutable snapshot used for one CLI invocation."""

    try:
        return PipelineConfig(
            job_root=settings.job_root,
            source_language_hint=settings.source_language,
            stt=settings.stt,
            mt=settings.mt,
            keyterms=settings.keyterms,
            api=settings.api,
            google_basic=settings.google_basic,
            video_extraction=settings.video_extraction,
            youtube_cache_root=settings.youtube_cache_root,
        )
    except ValidationError as exc:
        raise ConfigError(
            "INVALID_CONFIG",
            f"the job configuration is invalid: {exc}",
        ) from exc


def elevenlabs_resolver(environment: CliEnvironment) -> Callable[[], str]:
    """Return a lazy resolver that reads the STT key only when dispatched."""

    def resolve() -> str:
        key = environment.lookup("ELEVENLABS_API_KEY")
        if not key:
            raise ConfigError(
                "MISSING_CREDENTIAL",
                "ELEVENLABS_API_KEY is not configured; refusing to send",
            )
        return key

    return resolve


def google_basic_resolver(environment: CliEnvironment) -> Callable[[], str]:
    """Return a lazy resolver that reads the Basic API key only when dispatched.

    The key is read from the environment mapping at call time and is never stored
    on any snapshot, returned to the browser, logged or written to an artifact.
    Only :func:`build_settings` checks *whether* the key is configured, and it
    reports that as a boolean.
    """

    def resolve() -> str:
        key = environment.lookup("GOOGLE_TRANSLATION_API_KEY")
        if not key:
            raise ConfigError(
                "MISSING_CREDENTIAL",
                "GOOGLE_TRANSLATION_API_KEY is not configured; refusing to send",
            )
        return key

    return resolve


def default_runtime_factory(
    config: PipelineConfig, environment: CliEnvironment
) -> RuntimeProviders:
    """Build the fixed Scribe + Google Translation Basic v2 adapters.

    The identities are derived from the snapshot; there is no route selection and
    no fallback. The Google Basic adapter authenticates with the API key through
    a lazy resolver, so no secret is read until dispatch.
    """

    if config.mt != CANONICAL_GOOGLE_BASIC_MT_IDENTITY:
        raise ConfigError(
            "MT_ROUTE_INVALID",
            "the job snapshot does not carry the fixed Google Basic MT identity; "
            "refusing to build a runtime",
        )
    return RuntimeProviders(
        stt=ScribeV2STTProvider(
            api=config.api, credential_resolver=elevenlabs_resolver(environment)
        ),
        mt=GoogleTranslationBasicProvider(
            api=config.api,
            basic=config.google_basic,
            credential_resolver=google_basic_resolver(environment),
        ),
    )
