"""``subtitle-flow`` command line interface.

Entry point for the four planned verbs — ``process``, ``transcribe``,
``translate`` and ``resume`` — over the existing shared per-job store. The CLI
only wires configuration, adapters, the pipeline and the derived exports
together; it contains no provider logic and makes no network call by itself.

Offline guarantees
------------------
``--help`` and every local validation work without credentials or network. Paid
API calls stay disabled unless explicitly opted in. The Google Basic paid/config
gate is applied at the last responsible moment: a run that can entirely reuse an
already complete, verified job does so with zero credential, reservation or HTTP
work, while a genuinely required remote call still fails closed before sending.
There is one fixed inference route: ElevenLabs Scribe v2 for STT and Google
Translation Basic v2 (API key) for Turkish MT.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Final

import typer

from subtitle_flow import __version__
from subtitle_flow.cleanup import (
    clean_completed_youtube_job,
    verify_published_document,
)
from subtitle_flow.cli_config import (
    CANONICAL_MT_IDENTITY,
    CliEnvironment,
    CliOverrides,
    CliSettings,
    ConfigError,
    RuntimeProviders,
    build_pipeline_config,
    build_settings,
    default_runtime_factory,
    elevenlabs_resolver,
    google_basic_resolver,
    load_environment,
)
from subtitle_flow.config import PipelineConfig
from subtitle_flow.google_basic_mt import (
    GoogleBasicMTError,
    GoogleBasicTimedRoute,
    reuse_google_basic_fulltext,
    translate_google_basic_fulltext,
)
from subtitle_flow.exports import (
    ExportError,
    ExportResult,
    F_ALIGNED,
    F_METRICS,
    F_TRANSCRIPT_TXT,
    F_TRANSLATION_TXT,
    JobMetrics,
    ModelIdentity,
    StageTiming,
    export_job,
)
from subtitle_flow.fulltext_mt import (
    FullTextMTError,
    FullTextTranslationArtifact,
    write_translation_markdown,
)
from subtitle_flow.languages import TARGET_LANGUAGE_CODE
from subtitle_flow.media import MediaError
from subtitle_flow.migrate_outputs import MigrationError, migrate_transcripts
from subtitle_flow.output_paths import (
    default_source_dir,
    default_translation_dir,
    default_video_dir,
    translation_dir_for_source,
    video_dir_for_source,
)
from subtitle_flow.pipeline import (
    AudioPipeline,
    PipelineError,
    PipelineNeedsReviewError,
    PipelineProviderError,
    PreflightError,
    ProviderBindingError,
    ProviderResultError,
    RemoteStatusUnknownError,
    ReplayRefused,
    ResumeRefused,
    SegmentGroupingError,
)
from subtitle_flow.providers import (
    MachineTranslationProvider,
    ScribeV2STTProvider,
    UnsupportedLanguageError,
)
from subtitle_flow.providers.errors import ProviderCallError
from subtitle_flow.schemas import JobStatus, Segment, Translation
from subtitle_flow.storage import (
    JobLockedError,
    JobMissingError,
    JobStore,
    StorageError,
    StoredInput,
)
from subtitle_flow.timed_subtitles import TimedSubtitleError
from subtitle_flow.transcript_markdown import (
    MarkdownExportError,
    SourceTranscript,
    TRANSCRIPT_NOT_RECOVERABLE,
    recover_source_transcript,
    render_markdown,
    write_transcript_markdown,
)
from subtitle_flow.video import VideoMediaError
from subtitle_flow.video_burn import (
    BurnOutcome,
    VideoBurnError,
    burn_turkish_subtitles,
    preflight_burn_toolchain,
    verify_published_video,
)
from subtitle_flow.youtube_source import (
    YouTubeMediaError,
    YouTubeReference,
    canonicalize_youtube_url,
)

__all__ = [
    "ACTION_FULLCHAIN",
    "ACTION_TRANSLATE_ARCHIVED",
    "ACTION_PROCESS",
    "ACTION_RESUME",
    "ACTION_TRANSCRIBE",
    "ACTION_TRANSLATE",
    "CommandResult",
    "ExitCode",
    "app",
    "execute",
    "main",
    "use_desktop_stt_factory",
    "use_google_basic_factory",
]

ACTION_PROCESS: Final[str] = "process"
ACTION_TRANSCRIBE: Final[str] = "transcribe"
ACTION_TRANSLATE: Final[str] = "translate"
ACTION_RESUME: Final[str] = "resume"
#: Desktop full chain: source STT (Scribe) then Google Basic Turkish MT.
ACTION_FULLCHAIN: Final[str] = "fullchain"
#: Turkish MT only, from an already archived source (no audio, no STT).
ACTION_TRANSLATE_ARCHIVED: Final[str] = "translate-archived"

#: Full-text source MT failure codes that are actionable configuration/runtime
#: refusals (no fabricated success) rather than review items.
_FULLTEXT_MT_CONFIG_CODES: Final[frozenset[str]] = frozenset(
    {
        "FULLTEXT_MT_NO_SOURCE",
        "FULLTEXT_MT_EMPTY",
        "FULLTEXT_MT_BUDGET_INVALID",
        "FULLTEXT_MT_INPUT_TOO_LONG",
        "FULLTEXT_MT_EMPTY_SOURCE",
        "FULLTEXT_MT_NO_VIDEO",
    }
)
_SOURCE_MT_CONFIG_CODES: Final[frozenset[str]] = _FULLTEXT_MT_CONFIG_CODES

#: Google Translation LLM (Basic v2) failures that are actionable pre-dispatch
#: configuration/input refusals: no request was sent (or it was rejected as
#: invalid) and no human review is needed.
_GOOGLE_BASIC_CONFIG_CODES: Final[frozenset[str]] = frozenset(
    {
        "PAID_CALLS_DISABLED",
        "PAID_POLICY_INCOMPLETE",
        "PAID_POLICY_INVALID",
        "PAID_PER_CALL_LIMIT",
        "PAID_TOTAL_LIMIT",
        "API_CREDENTIAL_MISSING",
        "MT_PROJECT_MISSING",
        "MT_ENDPOINT_INVALID",
        "MT_EMPTY_GROUP",
        "MT_GROUP_TOO_LARGE",
        "MT_SEGMENT_TOO_LONG",
        "MT_INPUT_TOO_LONG",
        "MT_ARGUMENT_MISMATCH",
    }
)

#: Google Translation LLM provider statuses that are clearly request/auth or
#: configuration problems. The request was dispatched and rejected as such, so
#: the honest outcome is actionable invalid input/configuration (no automatic
#: resend), never a human review item. Quota/rate-limit and remote-unknown stay
#: classified separately.
_GOOGLE_BASIC_PROVIDER_REJECT_CODES: Final[frozenset[str]] = frozenset(
    {
        "INVALID_ARGUMENT",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "FAILED_PRECONDITION",
    }
)

#: Google Translation LLM evidence/provenance problems that require a human
#: review decision rather than an automatic retry.
_GOOGLE_BASIC_REVIEW_CODES: Final[frozenset[str]] = frozenset(
    {
        "GOOGLE_BASIC_FULLTEXT_CORRUPT",
        "GOOGLE_BASIC_EVIDENCE_CORRUPT",
        "GOOGLE_BASIC_PROVENANCE",
        "GOOGLE_BASIC_INCOMPLETE",
        "GOOGLE_BASIC_UNLOCKED",
        # A prior paid outcome exists but its artifact is not reusable: the
        # default replay is refused (no auto-resend). The remote-unknown variant
        # is classified as INCOMPLETE by its ``remote_status_unknown`` flag.
        "GOOGLE_BASIC_REPLAY_REFUSED",
        "GOOGLE_BASIC_REMOTE_STATUS_UNKNOWN",
    }
)

_CONFIG_CODES: Final[frozenset[str]] = frozenset(
    {
        "PAID_CALLS_DISABLED",
        "PAID_POLICY_INCOMPLETE",
        "PAID_POLICY_INVALID",
        "PAID_PER_CALL_LIMIT",
        "PAID_TOTAL_LIMIT",
        "API_CREDENTIAL_MISSING",
        "MT_PROJECT_MISSING",
        "STT_ENDPOINT_INVALID",
        "MT_ENDPOINT_INVALID",
        "STT_LANGUAGE_HINT_INVALID",
        "MT_STATUS_INVALID",
        "SKIP_STATE_INVALID",
        "MT_INPUT_TOO_LONG",
        "STT_DURATION_UNKNOWN",
    }
)

#: Video rejections that are invalid *input* (no work was attempted), as opposed
#: to a tool/environment or extraction failure. The code set is explicit so a
#: missing ffmpeg never masquerades as bad user input.
_VIDEO_INPUT_CODES: Final[frozenset[str]] = frozenset(
    {
        "VIDEO_PATH_INVALID",
        "VIDEO_MISSING",
        "VIDEO_UNREADABLE",
        "VIDEO_NOT_REGULAR",
        "VIDEO_SYMLINK",
        "VIDEO_EMPTY",
        "VIDEO_TOO_LARGE",
        "VIDEO_TOO_LONG",
        "VIDEO_DURATION_INVALID",
        "VIDEO_UNSUPPORTED_FORMAT",
        "VIDEO_UNSUPPORTED_OFFSET",
        "VIDEO_NO_VIDEO",
        "VIDEO_NO_AUDIO",
        "VIDEO_MULTIPLE_AUDIO",
        "VIDEO_CORRUPT",
        "VIDEO_CHANGED",
        "VIDEO_METADATA_INVALID",
        "VIDEO_SETTINGS_INVALID",
        # A nonzero ffprobe exit is a bad container/stream or a blocked
        # non-local protocol, i.e. rejected input. A *missing* tool stays in the
        # failure bucket so an uninstalled ffmpeg is never called bad input.
        "FFPROBE_FAILED",
    }
)


#: YouTube rejections that are invalid *input* (the URL, its metadata or a
#: downloaded source that does not satisfy the contract), as opposed to a
#: missing tool or a transient network/tool failure.
_YOUTUBE_INPUT_CODES: Final[frozenset[str]] = frozenset(
    {
        "YOUTUBE_URL_INVALID",
        "YOUTUBE_URL_UNSUPPORTED",
        "YOUTUBE_URL_AMBIGUOUS",
        "YOUTUBE_METADATA_INVALID",
        "YOUTUBE_ID_MISMATCH",
        "YOUTUBE_NOT_PUBLIC",
        "YOUTUBE_LIVE_UNSUPPORTED",
        "YOUTUBE_SOURCE_TOO_SHORT",
        "YOUTUBE_SOURCE_TOO_LONG",
        "YOUTUBE_SOURCE_TOO_LARGE",
        "YOUTUBE_SOURCE_INVALID",
        "YOUTUBE_SOURCE_CORRUPT",
        "YOUTUBE_OUTPUT_INVALID",
        "YOUTUBE_OUTPUT_TOO_LARGE",
        "YOUTUBE_OUTPUT_INCOMPLETE",
        "YOUTUBE_SETTINGS_INVALID",
    }
)


#: Burned-in subtitle refusals that are invalid *input/configuration* (the
#: operator can act on them) rather than a tool/environment failure. The
#: ``YTDLP_*`` codes are deliberately absent: the established audio route maps a
#: failed ``yt-dlp`` download through ``YouTubeMediaError`` to ``FAILURE``, and
#: the burned-in route must not silently assign a different exit code to the same
#: failure. A missing/invalid model or provider is actionable config, so it stays
#: in this bucket; a missing media toolchain stays a ``FAILURE`` (matching the
#: existing video/media route).
_BURN_INPUT_CODES: Final[frozenset[str]] = frozenset(
    {
        "VIDEO_NAME_INVALID",
        "VIDEO_DIR_INVALID",
        "VIDEO_FULL_MISSING",
        "VIDEO_FULL_INVALID",
        "VIDEO_FULL_TOO_LARGE",
        "VIDEO_FULL_INCOMPLETE",
        "VIDEO_TARGET_UNSAFE",
        "VIDEO_DIR_UNSAFE",
        "BURN_MEDIA_INVALID",
        "BURN_NO_YOUTUBE_ORIGIN",
        "BURN_ALREADY_LOCKED",
        "TIMED_SUBTITLE_NO_SEGMENTS",
        "TIMED_SUBTITLE_LANGUAGE_UNCERTAIN",
        "TIMED_SUBTITLE_UNSUPPORTED_LANGUAGE",
        "TIMED_SUBTITLE_TIMING_INVALID",
        "TIMED_SUBTITLE_PROVIDER_MISSING",
        "TIMED_SUBTITLE_MODEL_INVALID",
    }
)

#: Burned-in subtitle refusals that are a transient/environment failure (a local
#: model call that did not complete, or a lock/state problem), never a human
#: review decision. Mapping these to ``REVIEW_REQUIRED`` would misreport a
#: retryable failure as a completed-with-review result.
_BURN_FAILURE_CODES: Final[frozenset[str]] = frozenset(
    {
        "TIMED_SUBTITLE_FAILED",
        "TIMED_SUBTITLE_UNLOCKED",
    }
)

#: Burned-in subtitle refusals that require a human review decision: a stored
#: artifact or archived evidence is present but not safely reusable.
_BURN_REVIEW_CODES: Final[frozenset[str]] = frozenset(
    {
        "TIMED_SUBTITLE_TRANSCRIPT_UNAVAILABLE",
        "TIMED_SUBTITLE_CORRUPT",
        "TIMED_SUBTITLE_EVIDENCE_CORRUPT",
        "TIMED_SUBTITLE_IDENTITY_INVALID",
        "TIMED_SUBTITLE_PROVENANCE",
        "TIMED_SUBTITLE_INCOMPLETE",
        "VIDEO_ARCHIVE_COLLISION",
    }
)


class ExitCode(IntEnum):
    """Documented process exit codes.

    * ``0`` success (job complete, no review flag)
    * ``2`` invalid input or configuration (no provider call was made)
    * ``3`` review required (quality flags, uncertain language, pending review)
    * ``4`` incomplete, interrupted or unknown; safe to resume with ``resume``
    * ``5`` failure (corruption, lock collision, unexpected error)
    """

    SUCCESS = 0
    INVALID_INPUT = 2
    REVIEW_REQUIRED = 3
    INCOMPLETE = 4
    FAILURE = 5


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one CLI command, including the exact exit code.

    ``source_markdown`` and ``translation_markdown`` are the published document
    paths (if any); ``partial`` is ``True`` when a full-chain run produced the
    source document but the Turkish translation did not complete, so the
    caller can present the source result and retry only the Turkish MT later.
    ``cleanup_warning`` carries an honest, non-masking message when the explicit
    post-completion disk cleanup could not finish; it is ``None`` on a complete
    cleanup (and always ``None`` when cleanup was not requested).
    """

    exit_code: int
    job_id: str | None
    message: str
    export: ExportResult | None = None
    video_id: str | None = None
    source_markdown: str | None = None
    translation_markdown: str | None = None
    #: Published burned-in Turkish subtitle MP4 (opt-in feature only).
    video_file: str | None = None
    partial: bool = False
    translation_error: str | None = None
    video_error: str | None = None
    cleanup_warning: str | None = None

    @property
    def metrics(self) -> JobMetrics | None:
        return self.export.metrics if self.export is not None else None


def _map_exception(exc: BaseException) -> tuple[ExitCode, str]:
    """Translate an exception into a documented exit code and Turkish message."""

    if isinstance(exc, ConfigError):
        # Keep the stable code in the surfaced text: the UI selects its specific
        # actionable hint (for example the paid-consent checkbox for
        # ``PAID_CALLS_DISABLED``) by matching that code.
        return ExitCode.INVALID_INPUT, f"yapılandırma hatası ({exc.code}): {exc.message}"
    if isinstance(exc, FullTextMTError):
        if exc.code in _SOURCE_MT_CONFIG_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"kaynak Türkçe çeviri başlatılamadı ({exc.code}): {exc.message}",
            )
        if exc.code in {"FULLTEXT_MT_CORRUPT", "FULLTEXT_MT_EVIDENCE_CORRUPT",
                        "FULLTEXT_MT_IDENTITY_INVALID", "FULLTEXT_MT_PROVENANCE",
                        "FULLTEXT_MT_INCOMPLETE"}:
            return (
                ExitCode.REVIEW_REQUIRED,
                f"kaynak Türkçe çeviri kanıtı incelenmeli ({exc.code}): {exc.message}",
            )
        return ExitCode.REVIEW_REQUIRED, f"kaynak Türkçe çeviri hatası ({exc.code}): {exc.message}"
    if isinstance(exc, GoogleBasicMTError):
        # The Google Basic adapter preserves its dispatch/unknown classification;
        # the CLI reports the selected provider honestly (never "yerel").
        if exc.code == "PAID_TOTAL_LIMIT":
            # A total-cap refusal is about the *next/current* request only; a
            # multi-group job may already have sent (and durably reserved) earlier
            # requests, so never imply the whole job sent nothing.
            return (
                ExitCode.INVALID_INPUT,
                "Google Translation LLM bütçe üst sınırına ulaşıldı "
                "(PAID_TOTAL_LIMIT): sıradaki istek gönderilmedi; önceki "
                "istekler için kalıcı rezervasyonlar bulunabilir.",
            )
        if exc.code in _GOOGLE_BASIC_PROVIDER_REJECT_CODES:
            # A provider auth/argument/configuration rejection is actionable
            # invalid input; the request is not resent automatically.
            return (
                ExitCode.INVALID_INPUT,
                f"Google Translation LLM isteği sağlayıcı tarafından reddedildi "
                f"({exc.code}); ayarları/girdiyi denetleyin, istek yeniden "
                "gönderilmedi.",
            )
        if exc.code in _GOOGLE_BASIC_CONFIG_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"Google Translation LLM yapılandırması eksik/geçersiz "
                f"({exc.code}): istek gönderilmedi.",
            )
        if exc.remote_status_unknown:
            return (
                ExitCode.INCOMPLETE,
                f"Google Translation LLM durumu bilinmiyor ({exc.code}); otomatik "
                "yeniden gönderim yok.",
            )
        if exc.retryable:
            return (
                ExitCode.INCOMPLETE,
                f"Google Translation LLM yeniden denenebilir ({exc.code}); tekrar "
                "deneyin.",
            )
        if exc.code in _GOOGLE_BASIC_REVIEW_CODES:
            return (
                ExitCode.REVIEW_REQUIRED,
                f"Google Translation LLM kanıtı incelenmeli ({exc.code}): "
                f"{exc.message}",
            )
        return (
            ExitCode.REVIEW_REQUIRED,
            f"Google Translation LLM hatası ({exc.code}): inceleme gerekli.",
        )
    if isinstance(exc, VideoBurnError):
        if exc.code in _BURN_INPUT_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"altyazılı video oluşturulamadı ({exc.code}): {exc.message}",
            )
        if exc.code in _BURN_REVIEW_CODES:
            return (
                ExitCode.REVIEW_REQUIRED,
                f"altyazılı video inceleme gerektiriyor ({exc.code}): {exc.message}",
            )
        return (
            ExitCode.FAILURE,
            f"altyazılı video işlemi başarısız ({exc.code}): {exc.message}",
        )
    if isinstance(exc, TimedSubtitleError):
        if exc.code in _BURN_INPUT_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"zaman kodlu çeviri başlatılamadı ({exc.code}): {exc.message}",
            )
        if exc.code in _BURN_REVIEW_CODES:
            return (
                ExitCode.REVIEW_REQUIRED,
                f"zaman kodlu çeviri incelenmeli ({exc.code}): {exc.message}",
            )
        # Transient MT failures and an unlocked store are environment
        # failures, not a review decision (and never a silent success).
        if exc.code in _BURN_FAILURE_CODES:
            return (
                ExitCode.FAILURE,
                f"zaman kodlu çeviri başarısız ({exc.code}): {exc.message}",
            )
        return (
            ExitCode.FAILURE,
            f"zaman kodlu çeviri hatası ({exc.code}): {exc.message}",
        )
    if isinstance(exc, ProviderCallError):
        code = exc.error.code
        if code == "PAID_TOTAL_LIMIT":
            return (
                ExitCode.INVALID_INPUT,
                "bütçe üst sınırına ulaşıldı (PAID_TOTAL_LIMIT): sıradaki istek "
                "gönderilmedi; önceki istekler için kalıcı rezervasyonlar "
                "bulunabilir.",
            )
        if code in _CONFIG_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"çeviri yapılandırması eksik/geçersiz ({code}): istek "
                "gönderilmedi veya kabul edilmedi.",
            )
        if exc.error.remote_status_unknown:
            return (
                ExitCode.INCOMPLETE,
                f"çeviri durumu bilinmiyor ({code}); otomatik yeniden "
                "gönderim yok.",
            )
        if exc.error.retryable:
            return (
                ExitCode.INCOMPLETE,
                f"çeviri yeniden denenebilir ({code}); tekrar deneyin.",
            )
        return ExitCode.REVIEW_REQUIRED, f"çeviri hatası ({code})."
    if isinstance(exc, MediaError):
        return (
            ExitCode.INVALID_INPUT,
            f"ses dosyası reddedildi ({exc.code}): {exc.message}",
        )
    if isinstance(exc, VideoMediaError):
        if exc.code in _VIDEO_INPUT_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"video reddedildi ({exc.code}): {exc.message}",
            )
        return (
            ExitCode.FAILURE,
            f"video işlemi başarısız ({exc.code}); iş veya ortam incelenmeli.",
        )
    if isinstance(exc, YouTubeMediaError):
        if exc.code in _YOUTUBE_INPUT_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"YouTube kaynağı reddedildi ({exc.code}): {exc.message}",
            )
        return (
            ExitCode.FAILURE,
            f"YouTube kaynak işlemi başarısız ({exc.code}); iş veya ortam "
            "incelenmeli.",
        )
    if isinstance(exc, UnsupportedLanguageError):
        return ExitCode.INVALID_INPUT, f"desteklenmeyen dil: {exc}"
    if isinstance(exc, SegmentGroupingError):
        return ExitCode.INVALID_INPUT, f"segment gruplaması başarısız: {exc}"
    if isinstance(exc, ReplayRefused):
        return (
            ExitCode.REVIEW_REQUIRED,
            "önceki deneme incelenmeli; otomatik yeniden gönderim durduruldu.",
        )
    if isinstance(exc, RemoteStatusUnknownError):
        return (
            ExitCode.INCOMPLETE,
            "uzak sağlayıcı durumu bilinmiyor; açık onay olmadan yeniden "
            "gönderilmez. `resume` ile veya inceleme sonrası sürdürün.",
        )
    if isinstance(exc, PipelineNeedsReviewError):
        return ExitCode.REVIEW_REQUIRED, f"insan incelemesi gerekli: {exc}"
    if isinstance(exc, ProviderResultError):
        return ExitCode.REVIEW_REQUIRED, f"sağlayıcı sonucu geçersiz: {exc}"
    if isinstance(exc, PipelineProviderError):
        code = exc.error.code
        if code == "PAID_TOTAL_LIMIT":
            return (
                ExitCode.INVALID_INPUT,
                "bütçe üst sınırına ulaşıldı (PAID_TOTAL_LIMIT): sıradaki istek "
                "gönderilmedi; önceki istekler için kalıcı rezervasyonlar "
                "bulunabilir.",
            )
        if code in _CONFIG_CODES:
            return (
                ExitCode.INVALID_INPUT,
                f"yapılandırma eksik/geçersiz ({code}): sağlayıcıya istek "
                "gönderilmedi.",
            )
        if exc.error.remote_status_unknown:
            return (
                ExitCode.INCOMPLETE,
                f"uzak durum bilinmiyor ({code}); otomatik yeniden gönderim yok.",
            )
        if exc.error.retryable:
            return (
                ExitCode.INCOMPLETE,
                f"yeniden denelenebilir aşama hatası ({code}); `resume` ile "
                "sürdürebilirsiniz.",
            )
        return ExitCode.REVIEW_REQUIRED, f"aşama hatası ({code}): inceleme gerekli."
    if isinstance(exc, (PreflightError, ResumeRefused, ProviderBindingError)):
        return ExitCode.INVALID_INPUT, f"iş ön kontrolü başarısız: {exc}"
    if isinstance(exc, JobLockedError):
        return (
            ExitCode.FAILURE,
            "iş başka bir yazar tarafından kilitli; aynı anda tek yazar çalışır.",
        )
    if isinstance(exc, JobMissingError):
        return (
            ExitCode.INVALID_INPUT,
            "belirtilen iş bulunamadı; job_id ve JOB_DIR değerlerini kontrol edin.",
        )
    if isinstance(exc, ExportError):
        return ExitCode.FAILURE, f"dışa aktarma başarısız: {exc}"
    if isinstance(exc, MarkdownExportError):
        if exc.code == "TRANSCRIPT_LANGUAGE_MISMATCH":
            return ExitCode.INVALID_INPUT, f"dil uyuşmazlığı: {exc.message}"
        return (
            ExitCode.FAILURE,
            f"tam metin Markdown yazılamadı ({exc.code}): {exc.message}",
        )
    if isinstance(exc, StorageError):
        return ExitCode.FAILURE, f"kalıcı depo hatası: {exc}"
    if isinstance(exc, PipelineError):
        return ExitCode.FAILURE, f"iş hattı hatası: {exc}"
    if isinstance(exc, OSError):
        return ExitCode.FAILURE, f"girdi/çıktı hatası: {exc}"
    return (
        ExitCode.FAILURE,
        f"beklenmeyen hata: {type(exc).__name__}; iş yeniden çalıştırılabilir.",
    )


def _stage_line(label: str, timing: StageTiming) -> str:
    if timing.elapsed_ms is None:
        reason = timing.reason or "ölçülmedi"
        return f"{label}: ölçülmedi ({reason})"
    return f"{label}: {timing.elapsed_ms / 1000:.1f} sn"


def _model_line(label: str, identity: ModelIdentity) -> str:
    if identity.reported_model:
        return f"{label}: {identity.requested_model} (bildirilen: {identity.reported_model})"
    reason = identity.reported_model_reason or "bildirilmedi"
    return f"{label}: {identity.requested_model} (dönen model {reason})"


def _cost_line(metrics: JobMetrics) -> str:
    cost = metrics.cost
    if cost.status == "unknown":
        return f"Maliyet: ölçülmedi ({cost.unknown_reason or 'gerekçe yok'})"
    parts = []
    if cost.estimated is not None:
        parts.append(f"tahmini {cost.estimated} {cost.currency or ''}".strip())
    if cost.actual is not None:
        parts.append(f"gerçek {cost.actual} {cost.currency or ''}".strip())
    return "Maliyet: " + ", ".join(parts or ["ölçülmedi"])


def _render_summary(settings: CliSettings, export: ExportResult) -> str:
    metrics = export.metrics
    lines = [
        f"İş: {export.job_id}",
        f"Durum: {metrics.status}",
        f"Algılanan dil: {metrics.source_language or 'bilinmiyor'}"
        + (" (belirsiz)" if metrics.language_uncertain else ""),
        f"Ses süresi: {metrics.audio_duration_ms / 1000:.1f} sn",
        _stage_line("STT süresi", metrics.stt_timing),
        _stage_line("MT süresi", metrics.mt_timing),
    ]
    if metrics.total_stage_elapsed_ms is not None:
        lines.append(f"Toplam aşama süresi: {metrics.total_stage_elapsed_ms / 1000:.1f} sn")
    lines.extend(
        [
            _model_line("STT modeli", metrics.stt),
            _model_line("MT modeli", metrics.mt),
            "İşaretlenen segment: "
            f"{metrics.quality.flagged_segments}/{metrics.quality.checked_segments}",
            f"İnceleme gerekli: {'evet' if export.review_required else 'hayır'}",
            _cost_line(metrics),
        ]
    )
    if metrics.cost.reservation_total is not None:
        lines.append(
            "Yerel rezervasyon (fatura değil): "
            f"{metrics.cost.reservation_total} {metrics.cost.reservation_currency or ''}".strip()
        )
    lines.append("WER/CER: ölçülmedi (referans transkript yok)")
    exported = [F_TRANSCRIPT_TXT]
    if export.translation_included:
        exported.append(F_TRANSLATION_TXT)
    exported.extend([F_ALIGNED, F_METRICS])
    lines.append("Çıktılar: " + ", ".join(exported))
    lines.append(f"Çıktı dizini: {settings.job_root}/{export.job_id}")
    if export.review_required:
        lines.append(
            "Not: kalite bayrakları inceleme gerektirir; geçerli aşama kanıtı korunur."
        )
    return "\n".join(lines)


def _exit_for(export: ExportResult) -> ExitCode:
    if export.review_required:
        return ExitCode.REVIEW_REQUIRED
    status = export.metrics.status
    if status == JobStatus.complete.value:
        return ExitCode.SUCCESS
    if status == JobStatus.needs_review.value:
        return ExitCode.REVIEW_REQUIRED
    return ExitCode.INCOMPLETE


#: Test seam: the real Typer commands call :func:`execute` with the active
#: factory. Tests inject the same real adapters over a mocked HTTP transport;
#: production always falls back to :func:`default_runtime_factory` and never
#: silently changes the paid route.
_runtime_factory_override: Callable[..., RuntimeProviders] | None = None


def use_runtime_factory(factory: Callable[..., RuntimeProviders] | None) -> None:
    """Install (or with ``None`` clear) the factory the commands resolve.

    This is an offline test seam only: it selects *how* the canonical Scribe and
    Google adapters are constructed, never *which* providers run.
    """

    global _runtime_factory_override
    _runtime_factory_override = factory


def _active_runtime_factory() -> Callable[..., RuntimeProviders]:
    return _runtime_factory_override or default_runtime_factory


def _resume_config(settings: CliSettings, stored) -> PipelineConfig:
    """Bind an existing job to its stored immutable snapshot.

    Output-affecting options (source hint, repeated keyterms, provider
    identities, batching, media, segmenter and adapter settings) come from the
    stored snapshot so an existing job can be continued without recreating the
    original command line. Only explicitly supplied CLI values are checked
    against the snapshot, and a divergence fails closed here -- before any
    credential resolver, runtime adapter or HTTP call. The paid-call policy is
    part of that snapshot: a stored authorization is never inherited silently,
    and a current invocation that collides with it is refused rather than
    loosened, reset or re-fingerprinted.
    """

    stored_config = stored.config
    issues: list[str] = []
    if (
        settings.source_language is not None
        and settings.source_language != stored_config.source_language_hint
    ):
        issues.append("source language")
    if settings.keyterms and tuple(settings.keyterms) != tuple(stored_config.keyterms):
        issues.append("keyterms")
    if settings.api.paid != stored_config.api.paid:
        issues.append("paid-call policy")
    if (
        settings.video_container_explicit
        and settings.video_extraction != stored_config.video_extraction
    ):
        issues.append("video extraction container")
    if issues:
        raise ConfigError(
            "JOB_CONFIG_COLLISION",
            "the stored job snapshot is authoritative; the current invocation "
            "collides on " + ", ".join(issues) + ". Omit the colliding option "
            "to reuse the stored value, or start a new job. Paid authorization is "
            "never inherited: re-enable it explicitly with reservation caps equal "
            "to the stored policy.",
        )
    return PipelineConfig.model_validate(
        {**stored_config.model_dump(), "job_root": settings.job_root}
    )


def _config_for_action(
    action: str, settings: CliSettings, job_id: str | None
) -> PipelineConfig:
    if action in {ACTION_TRANSLATE, ACTION_RESUME}:
        assert job_id is not None
        probe = JobStore(settings.job_root, job_id)
        if not probe.exists():
            raise JobMissingError(
                f"job {job_id!r} does not exist under {settings.job_root!r}"
            )
        return _resume_config(settings, probe.read_input())
    return build_pipeline_config(settings)


def _validate_youtube_reuse(
    reference: YouTubeReference, stored: StoredInput
) -> None:
    """Prove an explicitly named job really is this video before any reuse.

    An explicit ``--job-id`` must never let a request for one video return
    another video's transcript. The stored immutable snapshot is already
    re-validated field by field when it is read; here the requested reference is
    checked against the job's frozen YouTube origin (and a job that is not a
    YouTube job at all is refused).
    """

    origin = stored.config.youtube_origin
    if origin is None:
        raise ConfigError(
            "JOB_CONFIG_COLLISION",
            "the stored job is not a YouTube-source job; refusing to reuse it for "
            "a YouTube transcript",
        )
    if origin.video_id != reference.video_id:
        raise ConfigError(
            "JOB_CONFIG_COLLISION",
            "the requested video does not match the stored job's YouTube origin; "
            "refusing to return a transcript for a different video",
        )


def _validate_reuse(settings: CliSettings, stored: StoredInput) -> None:
    """Validate caller options against a stored snapshot for an offline reuse.

    The paid-call policy is deliberately *not* checked here: a source-only or
    full-text-only reuse sends nothing, so it must not require re-authorizing a
    paid call. Provider/model, source-language hint, keyterms and route are
    checked so a caller selection is never silently ignored.
    """

    stored_config = stored.config
    issues: list[str] = []
    if (
        settings.source_language is not None
        and stored_config.source_language_hint is not None
        and settings.source_language != stored_config.source_language_hint
    ):
        issues.append("source language")
    if settings.keyterms and tuple(settings.keyterms) != tuple(stored_config.keyterms):
        issues.append("keyterms")
    if settings.stt != stored_config.stt or settings.mt != stored_config.mt:
        issues.append("provider/model")
    if (
        settings.video_container_explicit
        and settings.video_extraction != stored_config.video_extraction
    ):
        issues.append("video extraction container")
    if issues:
        raise ConfigError(
            "JOB_CONFIG_COLLISION",
            "the stored job snapshot is authoritative; the current invocation "
            "collides on " + ", ".join(issues) + ".",
        )


def _try_recover(
    job_root: str, job_id: str, *, requested_hint: str | None
) -> SourceTranscript | None:
    """Recover a full text from a job's durable evidence, or ``None``.

    Corrupt or mismatched evidence is *not* swallowed: only the explicit
    "nothing recoverable" outcome becomes ``None`` for a caller to fall through.
    """

    store = JobStore(job_root, job_id)
    with store:
        stored = store.read_input()
        try:
            return recover_source_transcript(
                store, stored, requested_hint=requested_hint
            )
        except MarkdownExportError as exc:
            if exc.code == TRANSCRIPT_NOT_RECOVERABLE:
                return None
            raise


def _find_youtube_job(
    settings: CliSettings, reference: YouTubeReference
) -> tuple[str, StoredInput] | None:
    """Find an existing compatible job for this video with a recoverable text.

    This makes a repeated ``transcribe-youtube`` request practical without a
    second paid call even when the source was first acquired under a different
    (for example explicitly named) job id. Only a job whose stored source matches
    the requested video, whose settings are compatible, and whose verified
    evidence actually yields a full text is reused.
    """

    root = Path(settings.job_root)
    if not root.is_dir():
        return None
    candidates: list[tuple[object, str, StoredInput]] = []
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            store = JobStore(settings.job_root, entry.name)
        except StorageError:
            continue
        if not store.exists():
            continue
        try:
            stored = store.read_input()
        except (StorageError, OSError):
            continue
        origin = stored.config.youtube_origin
        if origin is None or origin.video_id != reference.video_id:
            continue
        try:
            _validate_reuse(settings, stored)
        except ConfigError:
            continue
        candidates.append((stored.stored_at_utc, entry.name, stored))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _stored_at, job_id, _stored in candidates:
        try:
            source = _try_recover(
                settings.job_root, job_id, requested_hint=settings.source_language
            )
        except MarkdownExportError as exc:
            if exc.code == "TRANSCRIPT_LANGUAGE_MISMATCH":
                continue
            raise
        if source is not None:
            return job_id, _stored
    return None


def _markdown_message(
    job_id: str, source: SourceTranscript, path: Path
) -> str:
    lines = [f"İş: {job_id}", "Durum: kaynak tam metin Markdown üretildi"]
    if source.source_url:
        lines.append(f"Kaynak: {source.source_url}")
    lines.append(f"Dil: {source.source_language}")
    lines.append(f"Model: {source.provider} / {source.model}")
    lines.append(f"Markdown: {path}")
    lines.append(
        "Not: zaman hizalaması ve çeviri isteğe bağlıdır; tam metin bunlardan "
        "bağımsız üretilir."
    )
    return "\n".join(lines)


def _write_markdown_result(
    job_id: str,
    source: SourceTranscript,
    transcript_dir: str | None,
    overwrite: bool,
    *,
    export: ExportResult | None = None,
) -> CommandResult:
    directory = transcript_dir or str(default_source_dir())
    path = write_transcript_markdown(
        source,
        transcript_dir=directory,
        fallback_name=job_id,
        overwrite=overwrite,
    )
    return CommandResult(
        exit_code=ExitCode.SUCCESS,
        job_id=job_id,
        message=_markdown_message(job_id, source, path),
        export=export,
        video_id=source.video_id,
        source_markdown=str(path),
    )


def _call_once(callback: Callable[[], None] | None) -> Callable[[], None] | None:
    """Wrap a pure pre-dispatch gate so it is invoked at most once.

    The fullchain calls the gate early, before acquisition, to prove the paid MT
    route is dispatchable; the same wrapper is then threaded to the pipeline so
    the genuine later Scribe dispatch does not repeat it. A raising callback is
    never recorded as done: the error propagates and a later call would retry
    rather than falsely reporting success.
    """

    if callback is None:
        return None
    done = False

    def once() -> None:
        nonlocal done
        if done:
            return
        callback()
        done = True

    return once


def _transcribe_only(
    settings: CliSettings,
    environment: CliEnvironment,
    *,
    audio: str | None,
    job_id: str | None,
    input_kind: str,
    youtube_url: str | None,
    ytdlp_bin: str | None,
    allow_remote_retry: bool,
    runtime_factory: Callable[..., RuntimeProviders],
    transcript_dir: str | None,
    overwrite_transcript: bool,
    before_stt_dispatch: Callable[[], None] | None = None,
) -> CommandResult:
    """Run a source-only transcription.

    A YouTube request produces the standalone full-text Markdown from the
    verified archived response independent of the optional timed alignment, and
    exits ``0`` when that document is written. Audio/video transcription keeps
    its existing behaviour (canonical exports; ``4`` until the job is complete).

    ``before_stt_dispatch`` is the optional caller gate threaded to
    :meth:`AudioPipeline.transcribe`; the full chain passes the Google Basic
    paid/config gate so a fresh Scribe dispatch fails closed before it is billed
    when the accompanying MT route is not dispatchable, while a reused STT stage
    never runs it. For a brand-new YouTube job the same gate runs even earlier,
    before the yt-dlp acquisition, so a missing Google key/consent never
    downloads audio; the call-once wrapper keeps it from running twice.
    """

    if input_kind != "youtube":
        config = build_pipeline_config(settings)
        runtime = runtime_factory(config, environment)
        pipeline = AudioPipeline(
            config,
            stt_provider=runtime.stt,
            mt_provider=runtime.mt,
            ytdlp_bin=(ytdlp_bin or settings.youtube_ytdlp_bin),
        )
        started = pipeline.start(audio, job_id=job_id, input_kind=input_kind)
        pipeline.transcribe(
            started.job_id,
            allow_remote_retry=allow_remote_retry,
            before_dispatch=before_stt_dispatch,
        )
        with JobStore(config.job_root, started.job_id) as store:
            export = export_job(store)
        return CommandResult(
            exit_code=_exit_for(export),
            job_id=started.job_id,
            message=_render_summary(settings, export),
            export=export,
        )

    reference = canonicalize_youtube_url(youtube_url or "")
    resolved_id = job_id
    stored: StoredInput | None = None
    if resolved_id is not None:
        probe = JobStore(settings.job_root, resolved_id)
        if probe.exists():
            stored = probe.read_input()
            _validate_youtube_reuse(reference, stored)
            _validate_reuse(settings, stored)
    else:
        found = _find_youtube_job(settings, reference)
        if found is not None:
            resolved_id, stored = found

    if stored is not None:
        assert resolved_id is not None
        recovered = _try_recover(
            settings.job_root, resolved_id, requested_hint=settings.source_language
        )
        if recovered is not None:
            return _write_markdown_result(
                resolved_id,
                recovered,
                transcript_dir,
                overwrite_transcript,
            )
        # Offline recovery had nothing usable, so this request must dispatch.
        # Bind it to the stored immutable snapshot with the full strict resume
        # check: the *current* invocation must explicitly re-authorize the paid
        # policy (allow + caps). A stored paid opt-in is never inherited.
        config = _resume_config(settings, stored)
        # An existing job keeps its replay/remote-unknown ordering authoritative:
        # the gate runs only when the pipeline reaches a genuine dispatch.
        dispatch_gate = before_stt_dispatch
    else:
        config = build_pipeline_config(settings)
        # A brand-new job has no replay history to consult, so the caller gate
        # must fail closed *before* the yt-dlp acquisition: a missing Google
        # key/consent must not download audio. The call-once wrapper keeps the
        # same pure gate from running again when the pipeline reaches its own
        # dispatch point.
        dispatch_gate = _call_once(before_stt_dispatch)
        if dispatch_gate is not None:
            dispatch_gate()

    runtime = runtime_factory(config, environment)
    pipeline = AudioPipeline(
        config,
        stt_provider=runtime.stt,
        mt_provider=runtime.mt,
        ytdlp_bin=(ytdlp_bin or settings.youtube_ytdlp_bin),
    )
    started = pipeline.start_youtube(
        youtube_url or "", job_id=resolved_id, require_mt_project=False
    )
    effective_job_id = started.job_id
    failed: BaseException | None = None
    try:
        pipeline.transcribe(
            effective_job_id,
            allow_remote_retry=allow_remote_retry,
            before_dispatch=dispatch_gate,
        )
    except (PipelineProviderError, ReplayRefused) as exc:
        failed = exc

    recovered = _try_recover(
        config.job_root, effective_job_id, requested_hint=settings.source_language
    )
    if recovered is None:
        if failed is not None:
            raise failed
        raise MarkdownExportError(
            TRANSCRIPT_NOT_RECOVERABLE,
            "no verified source transcript is available for this job",
        )

    export: ExportResult | None = None
    if failed is None:
        with JobStore(config.job_root, effective_job_id) as store:
            export = export_job(store)
    return _write_markdown_result(
        effective_job_id,
        recovered,
        transcript_dir,
        overwrite_transcript,
        export=export,
    )


class _ForbiddenDesktopMTProvider(MachineTranslationProvider):
    """A never-called MT placeholder for the desktop STT-only step.

    The desktop chain translates through the Google Basic full-text route, never
    through this placeholder. Constructing this placeholder (instead of a real
    paid adapter) means no paid MT adapter is built on the STT-only step, while
    the pipeline's provider-identity check still sees the configured identity it
    never dispatches.
    """

    provider_name: ClassVar[str] = CANONICAL_MT_IDENTITY.provider
    model_name: ClassVar[str] = CANONICAL_MT_IDENTITY.model

    def _translate(  # pragma: no cover - proves the route is never entered
        self,
        segments: Sequence[Segment],
        *,
        source_language: str,
        target_language: str,
    ) -> Translation:
        raise RuntimeError(
            "the desktop chain never constructs or calls this placeholder; the "
            "full-text route must be used instead"
        )


def _stt_only_runtime_factory(
    config: PipelineConfig, environment: CliEnvironment
) -> RuntimeProviders:
    """Build only the STT adapter for the desktop source step.

    The desktop source step always uses Scribe plus the never-called placeholder,
    so the desktop chain can transcribe without constructing a Google MT adapter.
    """

    return RuntimeProviders(
        stt=ScribeV2STTProvider(
            api=config.api, credential_resolver=elevenlabs_resolver(environment)
        ),
        mt=_ForbiddenDesktopMTProvider(),
    )


#: Test seam for the desktop source STT adapter (offline injection only). It
#: selects *how* the Scribe adapter is constructed (for example with a mocked
#: HTTP transport), never *which* providers run.
_desktop_stt_factory_override: (
    Callable[..., RuntimeProviders] | None
) = None


def use_desktop_stt_factory(
    factory: Callable[..., RuntimeProviders] | None,
) -> None:
    global _desktop_stt_factory_override
    _desktop_stt_factory_override = factory


def _active_desktop_stt_factory() -> Callable[..., RuntimeProviders]:
    return _desktop_stt_factory_override or _stt_only_runtime_factory


#: Test seam for the Google Basic MT provider (offline injection only). It
#: selects *how* the Basic provider is constructed (for example over a mocked
#: HTTP transport), never *which* route runs.
_google_basic_factory_override: (
    Callable[[Any, Any, CliEnvironment], Any] | None
) = None


def use_google_basic_factory(
    factory: Callable[[Any, Any, CliEnvironment], Any] | None,
) -> None:
    global _google_basic_factory_override
    _google_basic_factory_override = factory


def _active_google_basic_factory() -> Callable[[Any, Any, CliEnvironment], Any]:
    if _google_basic_factory_override is not None:
        return _google_basic_factory_override
    from subtitle_flow.providers.google_basic import GoogleTranslationBasicProvider

    def factory(api: Any, basic: Any, environment: CliEnvironment) -> Any:
        return GoogleTranslationBasicProvider(
            api=api,
            basic=basic,
            credential_resolver=google_basic_resolver(environment),
        )

    return factory


def _require_google_basic(settings: CliSettings) -> Any:
    """Fail closed when the fixed paid MT route is not dispatchable."""

    basic = settings.google_basic
    if basic.project is None:
        raise ConfigError(
            "GOOGLE_BASIC_PROJECT_MISSING",
            "GOOGLE_TRANSLATION_PROJECT is not configured; no request will be sent",
        )
    if not settings.has_google_translation_key:
        raise ConfigError(
            "GOOGLE_BASIC_KEY_MISSING",
            "GOOGLE_TRANSLATION_API_KEY is not configured; no request will be sent",
        )
    if not settings.api.paid.allow_paid_api_calls:
        raise ConfigError(
            "PAID_CALLS_DISABLED",
            "paid API calls are disabled; no request will be sent",
        )
    return basic


def _resolve_translation_dir(
    translation_dir: str | None, transcript_dir: str | None
) -> str:
    if translation_dir:
        return translation_dir
    if transcript_dir:
        return str(translation_dir_for_source(transcript_dir))
    return str(default_translation_dir())


def _auto_cleanup_completed(
    settings: CliSettings,
    *,
    job_id: str,
    video_id: str | None,
    source_markdown: str | None,
    translation_markdown: str | None,
    transcript_dir: str | None,
    translation_dir: str | None,
    expected_source_text: str | None = None,
    published_video: str | None = None,
    video_dir: str | None = None,
) -> str | None:
    """Verify both documents then erase the completed video's jobs and cache.

    Both current-run documents must be real, non-symlinked ``<video_id>.md``
    files directly inside the selected output directories; a stale pre-existing
    document, a partial run or a single-document result never triggers cleanup.
    When ``expected_source_text`` is given (the translate-only route), the
    published source document must additionally be byte-identical to the source
    this run translated, so a stale or different-provider document can never
    authorize deleting the raw evidence. After verification the completed job and
    its safely-completed same-video siblings are erased (see
    :func:`clean_completed_youtube_job`). Returns an honest Turkish warning when
    cleanup could not run to completion (and never raises), so a completed run's
    documents are never masked.
    """

    if video_id is None:
        return "Otomatik disk temizliği yapılmadı: video kimliği yok."
    source_expected = transcript_dir or str(default_source_dir())
    translation_expected = _resolve_translation_dir(translation_dir, transcript_dir)
    reasons: list[str] = []
    source_reason = verify_published_document(
        source_markdown,
        expected_dir=source_expected,
        video_id=video_id,
        expected_text=expected_source_text,
    )
    if source_reason is not None:
        reasons.append(f"kaynak belge doğrulanamadı ({source_reason})")
    translation_reason = verify_published_document(
        translation_markdown,
        expected_dir=translation_expected,
        video_id=video_id,
    )
    if translation_reason is not None:
        reasons.append(f"Türkçe belge doğrulanamadı ({translation_reason})")
    if published_video is not None:
        video_expected = video_dir or str(default_video_dir())
        video_reason = verify_published_video(
            published_video, expected_dir=video_expected, video_id=video_id
        )
        if video_reason is not None:
            reasons.append(f"altyazılı video doğrulanamadı ({video_reason})")
    if reasons:
        return "Otomatik disk temizliği yapılmadı: " + "; ".join(reasons) + "."
    try:
        outcome = clean_completed_youtube_job(
            job_root=settings.job_root, job_id=job_id, expected_video_id=video_id
        )
    except Exception:  # noqa: BLE001 - cleanup must never mask the documents
        return "Otomatik disk temizliği başarısız; veri korundu."
    return outcome.warning_text


def _find_source_job(
    settings: CliSettings, video_id: str, requested_hint: str | None
) -> tuple[str, SourceTranscript] | None:
    """Find the newest job whose archived evidence yields this video's source.

    This never downloads media and never calls STT: it only reads already durable
    job evidence. The STT provider/model is deliberately not compared, because a
    local Turkish MT retry only needs the recovered source text.
    """

    root = Path(settings.job_root)
    if not root.is_dir():
        return None
    candidates: list[tuple[object, str]] = []
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        try:
            store = JobStore(settings.job_root, entry.name)
        except StorageError:
            continue
        if not store.exists():
            continue
        try:
            stored = store.read_input()
        except (StorageError, OSError):
            continue
        origin = stored.config.youtube_origin
        if origin is None or origin.video_id != video_id:
            continue
        candidates.append((stored.stored_at_utc, entry.name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _stored_at, job_id in candidates:
        try:
            source = _try_recover(
                settings.job_root, job_id, requested_hint=requested_hint
            )
        except MarkdownExportError as exc:
            if exc.code == "TRANSCRIPT_LANGUAGE_MISMATCH":
                continue
            raise
        if source is not None:
            return job_id, source
    return None
def _translate_archived_google_basic(
    settings: CliSettings,
    environment: CliEnvironment,
    *,
    job_id: str | None,
    youtube_url: str | None,
    video_id: str | None,
    translation_dir: str | None,
    transcript_dir: str | None,
    overwrite_translation: bool,
    allow_remote_retry: bool = False,
    progress: Callable[[str], None] | None = None,
    cleanup: bool = False,
) -> CommandResult:
    """Run the fixed Google Translation LLM (Basic v2) full-text MT route.

    Neither audio nor STT is touched: the verified source full text is recovered
    from durable evidence and translated through the Basic v2 API-key adapter (or
    a verified provider-scoped artifact is reused with zero HTTP). The Turkish
    Markdown is published atomically and a differing earlier document is archived,
    never destroyed.

    ``allow_remote_retry`` authorizes one fresh paid attempt for a full-text group
    whose durable trace already records an unreusable dispatched outcome; it
    defaults to ``False`` and is never inferred from a rerun.
    """

    mt_api, basic = settings.api, settings.google_basic
    resolved_id = job_id
    if resolved_id is not None:
        probe = JobStore(settings.job_root, resolved_id)
        if not probe.exists():
            raise JobMissingError(
                f"job {resolved_id!r} does not exist under {settings.job_root!r}"
            )
    else:
        target_video = video_id
        if target_video is None and youtube_url:
            target_video = canonicalize_youtube_url(youtube_url).video_id
        if not target_video:
            raise FullTextMTError(
                "FULLTEXT_MT_NO_VIDEO",
                "Google çeviri için bir YouTube video kimliği veya bağlantısı gerekli",
            )
        found = _find_source_job(settings, target_video, settings.source_language)
        if found is None:
            raise FullTextMTError(
                "FULLTEXT_MT_NO_SOURCE",
                "Bu video için arşivlenmiş kaynak transkript bulunamadı; önce "
                "kaynak transkripti alın.",
            )
        resolved_id = found[0]

    with JobStore(settings.job_root, resolved_id) as store:
        stored = store.read_input()
        try:
            source = recover_source_transcript(
                store, stored, requested_hint=settings.source_language
            )
        except MarkdownExportError as exc:
            if exc.code == TRANSCRIPT_NOT_RECOVERABLE:
                raise FullTextMTError(
                    "FULLTEXT_MT_NO_SOURCE",
                    "bu işte doğrulanmış kaynak tam metin bulunamadı.",
                ) from exc
            raise
        artifact = reuse_google_basic_fulltext(
            store, source, basic=basic
        )
        if artifact is None:
            # A remote paid MT dispatch (or, for a Turkish source, the zero-HTTP
            # same-language no-op) is genuinely required now. Gate the paid
            # route's project/key/consent here -- after a completed artifact has
            # been reused with zero credential/HTTP/reservation work, and before
            # the provider can send anything.
            if source.source_language != TARGET_LANGUAGE_CODE:
                _require_google_basic(settings)
            provider = _active_google_basic_factory()(
                mt_api, basic, environment
            )
            if progress is not None:
                progress("Google Translation LLM çevirisi başlıyor…")
            artifact = translate_google_basic_fulltext(
                store,
                stored,
                source,
                provider=provider,
                basic=basic,
                allow_remote_retry=allow_remote_retry,
                progress=progress,
            )
        directory = _resolve_translation_dir(translation_dir, transcript_dir)
        path = write_translation_markdown(
            source,
            artifact,
            translation_dir=directory,
            fallback_name=resolved_id,
            overwrite=overwrite_translation,
        )

    exit_code = (
        ExitCode.REVIEW_REQUIRED if artifact.needs_review else ExitCode.SUCCESS
    )
    cleanup_warning: str | None = None
    if cleanup:
        source_expected_dir = transcript_dir or str(default_source_dir())
        source_document = (
            str(Path(source_expected_dir).expanduser() / f"{source.video_id}.md")
            if source.video_id
            else None
        )
        cleanup_warning = _auto_cleanup_completed(
            settings,
            job_id=resolved_id,
            video_id=source.video_id,
            source_markdown=source_document,
            translation_markdown=str(path),
            transcript_dir=transcript_dir,
            translation_dir=translation_dir,
            expected_source_text=render_markdown(source),
        )
    lines = [
        f"İş: {resolved_id}",
        (
            "Durum: Google Translation LLM (Basic v2) Türkçe çevirisi hazır"
            if artifact.status == "translated"
            else "Durum: kaynak Türkçe; çeviri gerekmedi (aynı dil)"
        ),
        f"Dil: {source.source_language} → Türkçe",
        f"Model: {artifact.provider} / {artifact.model}",
        f"Türkçe Markdown: {path}",
    ]
    if artifact.needs_review:
        lines.append(
            "Not: otomatik kalite kontrolleri bazı bölümleri işaretledi; insan "
            "incelemesi gerekir, anlam doğruluğu iddia edilmez."
        )
    if cleanup_warning:
        lines.append(f"Not: {cleanup_warning}")
    return CommandResult(
        exit_code=exit_code,
        job_id=resolved_id,
        message="\n".join(lines),
        video_id=source.video_id,
        translation_markdown=str(path),
        cleanup_warning=cleanup_warning,
    )


def _run_burn_stage(
    settings: CliSettings,
    environment: CliEnvironment,
    *,
    job_id: str,
    video_dir: str | None,
    ytdlp_bin: str | None,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    progress: Callable[[str], None] | None,
    allow_remote_retry: bool = False,
) -> BurnOutcome:
    """Acquire the full video, burn the timed Turkish subtitles and publish.

    The timed translation is produced (or a verified matching artifact reused)
    through the fixed Google Translation LLM (Basic v2) route. The function never
    fabricates a timestamp: a job without a valid canonical timed transcript fails
    with a typed review refusal instead.

    ``allow_remote_retry`` is forwarded to the Basic v2 timed route. It never
    authorizes a paid resend by itself at the caller: it must be explicitly
    requested.
    """

    store = JobStore(settings.job_root, job_id)
    stored = store.read_input()
    provider = _active_google_basic_factory()(
        settings.api, settings.google_basic, environment
    )
    timed_route = GoogleBasicTimedRoute(
        store=store,
        provider=provider,
        basic=settings.google_basic,
        allow_remote_retry=allow_remote_retry,
        # The paid/config gate runs only if the timed route must actually
        # dispatch: a verified timed artifact is reused with zero credential,
        # reservation or HTTP work, and a Turkish source stays a no-op.
        before_dispatch=lambda: _require_google_basic(settings),
    )
    return burn_turkish_subtitles(
        store,
        stored,
        timed_route=timed_route,
        ytdlp_bin=ytdlp_bin or settings.youtube_ytdlp_bin,
        ffprobe_bin=ffprobe_bin,
        ffmpeg_bin=ffmpeg_bin,
        video_dir=video_dir,
        progress=progress,
    )


def _fullchain(
    settings: CliSettings,
    environment: CliEnvironment,
    *,
    audio: str | None,
    job_id: str | None,
    input_kind: str,
    youtube_url: str | None,
    ytdlp_bin: str | None,
    allow_remote_retry: bool,
    transcript_dir: str | None,
    translation_dir: str | None,
    overwrite_transcript: bool,
    overwrite_translation: bool,
    allow_mt_remote_retry: bool = False,
    progress: Callable[[str], None] | None = None,
    cleanup: bool = False,
    burn_video: bool = False,
    video_dir: str | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> CommandResult:
    """Source STT then Turkish MT, with an honest partial-failure result.

    ``allow_remote_retry`` authorizes the Scribe STT stage. ``allow_mt_remote_retry``
    separately authorizes a fresh *paid* Basic MT attempt; it defaults ``False``
    and is never inferred from ``--burn-video``.

    ``cleanup`` is the desktop auto-cleanup opt-in; when set and both documents
    are safely published, the completed job, its safely-completed same-video
    siblings and their exclusive source cache entries are erased (see
    :func:`_auto_cleanup_completed`). A cleanup problem is surfaced in
    ``cleanup_warning`` without masking the published documents.

    ``burn_video`` is the separate opt-in for a permanently burned-in Turkish
    subtitle MP4. It runs a second, timed Google Basic v2 translation and a
    full-video local render; the automatic cleanup only runs after *both*
    Markdown documents and
    the final MP4 have been published and verified, and a video failure keeps all
    job evidence for a safe retry while still reporting the published Markdown.
    """

    if progress is not None:
        progress("Kaynak transkript alınıyor…")
    source_result = _transcribe_only(
        settings,
        environment,
        audio=audio,
        job_id=job_id,
        input_kind=input_kind,
        youtube_url=youtube_url,
        ytdlp_bin=ytdlp_bin,
        allow_remote_retry=allow_remote_retry,
        runtime_factory=_active_desktop_stt_factory(),
        transcript_dir=transcript_dir,
        overwrite_transcript=overwrite_transcript,
        # The full chain will translate through the fixed Google Basic route, so
        # a genuinely required fresh Scribe dispatch must not be billed unless
        # that route's project/key/consent are dispatchable now. A verified STT
        # reuse never runs this gate, and a blocked/unknown prior attempt still
        # raises its replay refusal first.
        before_stt_dispatch=lambda: _require_google_basic(settings),
    )
    if source_result.exit_code != int(ExitCode.SUCCESS):
        return source_result
    if source_result.job_id is None:
        return CommandResult(
            exit_code=int(ExitCode.FAILURE),
            job_id=None,
            message="kaynak transkript işi kimliği alınamadı.",
        )
    if progress is not None:
        progress(
            "Kaynak transkript hazır; Google Translation LLM çevirisi başlıyor…"
        )
    try:
        mt_result = _translate_archived_google_basic(
            settings,
            environment,
            job_id=source_result.job_id,
            youtube_url=None,
            video_id=source_result.video_id,
            translation_dir=translation_dir,
            transcript_dir=transcript_dir,
            overwrite_translation=overwrite_translation,
            allow_remote_retry=allow_mt_remote_retry,
            progress=progress,
            cleanup=False,
        )
    except BaseException as exc:  # noqa: BLE001 - mapped to a documented code
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code, message = _map_exception(exc)
        return CommandResult(
            exit_code=code,
            job_id=source_result.job_id,
            message=(
                "Kaynak transkript hazır; "
                "Google Translation LLM çevirisi tamamlanamadı.\n"
                + f"{source_result.message}\n{message}"
            ),
            export=source_result.export,
            video_id=source_result.video_id,
            source_markdown=source_result.source_markdown,
            partial=True,
            translation_error=message,
        )
    video_file: str | None = None
    video_error: str | None = None
    video_review = False
    if burn_video:
        # A custom transcript directory keeps the two Markdown sets and the
        # burned-in video side by side (sibling ``videolar``), exactly like the
        # Turkish directory; an explicit ``--video-dir`` always wins.
        if video_dir is None and transcript_dir:
            video_dir = str(video_dir_for_source(transcript_dir))
        if progress is not None:
            progress("Altyazılı video hazırlanıyor…")
        try:
            burn = _run_burn_stage(
                settings,
                environment,
                job_id=source_result.job_id,
                video_dir=video_dir,
                ytdlp_bin=ytdlp_bin,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                progress=progress,
                allow_remote_retry=allow_mt_remote_retry,
            )
            video_file = str(burn.video_path)
            video_review = bool(burn.needs_review)
        except BaseException as exc:  # noqa: BLE001 - mapped to a documented code
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            code, message = _map_exception(exc)
            video_error = message
            # The two Markdown documents are already published; the video stage
            # failed. Keep every job/intermediate so the render can be retried
            # safely, and never mask the published documents or run cleanup. The
            # mapped code is preserved so a review refusal stays a review result.
            return CommandResult(
                exit_code=code,
                job_id=source_result.job_id,
                message=(
                    f"{source_result.message}\n{mt_result.message}\n"
                    "Kaynak ve Türkçe Markdown yayımlandı; altyazılı video "
                    f"tamamlanamadı.\n{message}"
                ),
                export=source_result.export,
                video_id=source_result.video_id,
                source_markdown=source_result.source_markdown,
                translation_markdown=mt_result.translation_markdown,
                partial=True,
                video_error=message,
            )

    cleanup_warning: str | None = None
    if cleanup:
        if burn_video and video_file is None:
            cleanup_warning = (
                "Otomatik disk temizliği yapılmadı: altyazılı video yayımlanmadı."
            )
        else:
            cleanup_warning = _auto_cleanup_completed(
                settings,
                job_id=source_result.job_id,
                video_id=source_result.video_id,
                source_markdown=source_result.source_markdown,
                translation_markdown=mt_result.translation_markdown,
                transcript_dir=transcript_dir,
                translation_dir=translation_dir,
                published_video=video_file,
                video_dir=video_dir,
            )
    message = f"{source_result.message}\n{mt_result.message}"
    if video_file is not None:
        message = f"{message}\nAltyazılı video: {video_file}"
    if video_review:
        message = (
            f"{message}\nNot: altyazı çevirisinde otomatik kalite denetimi bazı "
            "bölümleri işaretledi; insan incelemesi gerekir, anlam doğruluğu "
            "iddia edilmez."
        )
    if cleanup_warning:
        message = f"{message}\nNot: {cleanup_warning}"
    # A video that was produced while the timed translation requires review is a
    # review-required completion, not a silent success: every output is present,
    # nothing is partial, and the established cleanup rule may still run.
    final_exit = int(ExitCode.REVIEW_REQUIRED) if video_review else mt_result.exit_code
    return CommandResult(
        exit_code=final_exit,
        job_id=source_result.job_id,
        message=message,
        export=source_result.export,
        video_id=source_result.video_id,
        source_markdown=source_result.source_markdown,
        translation_markdown=mt_result.translation_markdown,
        video_file=video_file,
        partial=False,
        video_error=video_error,
        cleanup_warning=cleanup_warning,
    )


def execute(
    action: str,
    *,
    settings: CliSettings,
    environment: CliEnvironment,
    audio: str | None = None,
    job_id: str | None = None,
    allow_remote_retry: bool = False,
    allow_mt_remote_retry: bool | None = None,
    input_kind: str = "audio",
    youtube_url: str | None = None,
    ytdlp_bin: str | None = None,
    transcript_dir: str | None = None,
    overwrite_transcript: bool = False,
    translation_dir: str | None = None,
    overwrite_translation: bool = False,
    video_id: str | None = None,
    runtime_factory=default_runtime_factory,
    progress: Callable[[str], None] | None = None,
    cleanup: bool = False,
    burn_video: bool = False,
    video_dir: str | None = None,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> CommandResult:
    """Run one command end to end and return its documented exit code.

    The ``runtime_factory`` seam lets tests supply the same real adapters over a
    mocked HTTP transport instead of the default credential-resolving factory.
    It is not a provider switch: the fixed Scribe + Google Basic identities are
    always required. The desktop ``fullchain``/``translate-archived`` actions
    translate through the Google Basic full-text route.
    """

    if action not in {
        ACTION_PROCESS,
        ACTION_TRANSCRIBE,
        ACTION_TRANSLATE,
        ACTION_RESUME,
        ACTION_FULLCHAIN,
        ACTION_TRANSLATE_ARCHIVED,
    }:
        return CommandResult(
            exit_code=ExitCode.INVALID_INPUT,
            job_id=job_id,
            message=f"bilinmeyen komut: {action}",
        )
    if input_kind not in {"audio", "video", "youtube"}:
        return CommandResult(
            exit_code=ExitCode.INVALID_INPUT,
            job_id=job_id,
            message="--input-kind 'audio', 'video' veya 'youtube' olmalı",
        )
    if action == ACTION_TRANSLATE_ARCHIVED:
        if (
            job_id is None
            and not youtube_url
            and not (input_kind == "youtube" and video_id)
        ):
            return CommandResult(
                exit_code=ExitCode.INVALID_INPUT,
                job_id=job_id,
                message="çeviri için job_id ya da YouTube bağlantısı/kimliği gerekli",
            )
    elif action == ACTION_FULLCHAIN:
        if input_kind == "youtube":
            if not youtube_url:
                return CommandResult(
                    exit_code=ExitCode.INVALID_INPUT,
                    job_id=job_id,
                    message="bir YouTube video bağlantısı gerekli",
                )
        elif not audio:
            return CommandResult(
                exit_code=ExitCode.INVALID_INPUT,
                job_id=job_id,
                message="ses dosyası yolu gerekli",
            )
    elif input_kind == "youtube":
        if action in {ACTION_PROCESS, ACTION_TRANSCRIBE} and not youtube_url:
            return CommandResult(
                exit_code=ExitCode.INVALID_INPUT,
                job_id=job_id,
                message="bir YouTube video bağlantısı gerekli",
            )
    elif action in {ACTION_PROCESS, ACTION_TRANSCRIBE} and not audio:
        return CommandResult(
            exit_code=ExitCode.INVALID_INPUT,
            job_id=job_id,
            message="ses dosyası yolu gerekli",
        )
    if action in {ACTION_TRANSLATE, ACTION_RESUME} and not job_id:
        return CommandResult(
            exit_code=ExitCode.INVALID_INPUT,
            job_id=job_id,
            message="bu komut için job_id gerekli",
        )
    if burn_video and not (
        action == ACTION_FULLCHAIN and input_kind == "youtube"
    ):
        # The burned-in subtitle video is an opt-in of the fullchain YouTube route
        # only: a source-only, translate-only, local-video or ready-audio run has
        # no accepted full-video acquisition contract.
        return CommandResult(
            exit_code=ExitCode.INVALID_INPUT,
            job_id=job_id,
            message=(
                "--burn-video yalnız tam zincir YouTube akışında kullanılır "
                "(fullchain + --input-kind youtube)."
            ),
        )
    if action == ACTION_FULLCHAIN:
        # The Google Basic paid/config gate is deliberately *not* applied here: a
        # run that can entirely reuse already complete, verified local artifacts
        # must not demand a fresh paid opt-in, credential resolution, reservation
        # or HTTP. The gate is deferred to the point where a genuinely required
        # remote MT call is about to be dispatched (see
        # :func:`_translate_archived_google_basic` and :func:`_run_burn_stage`),
        # so a new/incomplete job still fails closed with the same
        # ``PAID_CALLS_DISABLED`` code. ``_require_google_basic`` is pure config
        # validation and never a network probe.
        if burn_video:
            # The opt-in video render needs a local ffmpeg with the libass
            # ``subtitles`` filter and the H.264/AAC encoders. Validate that
            # offline *before* the paid source STT call, so a missing tool is an
            # actionable refusal instead of a paid run that fails at the end.
            try:
                preflight_burn_toolchain(ffmpeg_bin or "ffmpeg")
            except VideoBurnError as exc:
                code, message = _map_exception(exc)
                return CommandResult(exit_code=code, job_id=job_id, message=message)

    # The paid-Basic MT authorization is independent of the STT one. The CLI
    # keeps them equal (one ``--allow-remote-retry``), while a caller such as the
    # desktop UI can withhold the paid MT resend without losing an explicit
    # STT replay authorization. It is never inferred from the route.
    effective_mt_retry = (
        allow_remote_retry if allow_mt_remote_retry is None else allow_mt_remote_retry
    )

    try:
        if action == ACTION_TRANSCRIBE:
            return _transcribe_only(
                settings,
                environment,
                audio=audio,
                job_id=job_id,
                input_kind=input_kind,
                youtube_url=youtube_url,
                ytdlp_bin=ytdlp_bin,
                allow_remote_retry=allow_remote_retry,
                runtime_factory=runtime_factory,
                transcript_dir=transcript_dir,
                overwrite_transcript=overwrite_transcript,
            )
        if action == ACTION_FULLCHAIN:
            return _fullchain(
                settings,
                environment,
                audio=audio,
                job_id=job_id,
                input_kind=input_kind,
                youtube_url=youtube_url,
                ytdlp_bin=ytdlp_bin,
                allow_remote_retry=allow_remote_retry,
                allow_mt_remote_retry=effective_mt_retry,
                transcript_dir=transcript_dir,
                translation_dir=translation_dir,
                overwrite_transcript=overwrite_transcript,
                overwrite_translation=overwrite_translation,
                progress=progress,
                cleanup=cleanup,
                burn_video=burn_video,
                video_dir=video_dir,
                ffmpeg_bin=ffmpeg_bin or "ffmpeg",
                ffprobe_bin=ffprobe_bin or "ffprobe",
            )
        if action == ACTION_TRANSLATE_ARCHIVED:
            return _translate_archived_google_basic(
                settings,
                environment,
                job_id=job_id,
                youtube_url=youtube_url,
                video_id=video_id,
                translation_dir=translation_dir,
                transcript_dir=transcript_dir,
                overwrite_translation=overwrite_translation,
                allow_remote_retry=effective_mt_retry,
                progress=progress,
                cleanup=cleanup,
            )
        config = _config_for_action(action, settings, job_id)
        runtime = runtime_factory(config, environment)
        pipeline = AudioPipeline(
            config,
            stt_provider=runtime.stt,
            mt_provider=runtime.mt,
            ytdlp_bin=(ytdlp_bin or settings.youtube_ytdlp_bin),
        )
        effective_job_id: str
        if action == ACTION_PROCESS:
            if input_kind == "youtube":
                result = pipeline.process_youtube(
                    youtube_url or "",
                    job_id=job_id,
                    allow_remote_retry=allow_remote_retry,
                )
            else:
                result = pipeline.process(
                    audio,
                    job_id=job_id,
                    allow_remote_retry=allow_remote_retry,
                    input_kind=input_kind,
                )
            effective_job_id = result.job_id
        elif action == ACTION_TRANSLATE:
            # ``AudioPipeline.translate`` returns a Translation without a job id;
            # the caller-supplied id is the authoritative job identity.
            pipeline.translate(job_id, allow_remote_retry=allow_remote_retry)
            effective_job_id = job_id
        else:
            result = pipeline.resume(job_id, allow_remote_retry=allow_remote_retry)
            effective_job_id = result.job_id

        with JobStore(config.job_root, effective_job_id) as store:
            export = export_job(store)
    except BaseException as exc:  # noqa: BLE001 - mapped to documented exit codes
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        code, message = _map_exception(exc)
        return CommandResult(exit_code=code, job_id=job_id, message=message)

    return CommandResult(
        exit_code=_exit_for(export),
        job_id=effective_job_id,
        message=_render_summary(settings, export),
        export=export,
    )


# --------------------------------------------------------------------------- #
# Typer wiring
# --------------------------------------------------------------------------- #
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help=(
        "YouTube/yerel ses veya video için Scribe v2 kaynak transkripti ve "
        "Google Translation Basic v2 Türkçe çevirisi."
    ),
    epilog=(
        "Çıkış kodları: 0 başarı; 2 geçersiz girdi/yapılandırma (çağrı yok); "
        "3 inceleme gerekli (kalite bayrağı, belirsiz dil veya bekleyen inceleme); "
        "4 eksik/kesintili/bilinmeyen (resume ile sürdürülebilir); "
        "5 hata (bozulma, kilit çakışması veya beklenmeyen hata)."
    ),
)

_AUDIO_HELP = "Hazır WAV/FLAC ses dosyası (ya da --input-kind video ile video dosyası)."
_INPUT_KIND_HELP = (
    "Girdi türü: 'audio' (varsayılan, değişmez WAV/FLAC yolu) veya 'video' "
    "(ffprobe ile doğrulanan yerel video; ses çıkarılıp aynı hatta verilir)."
)
_VIDEO_CONTAINER_HELP = (
    "Video çıkarımı çıktı kapsayıcısı: 'wav' (varsayılan, mono 16k PCM16) veya "
    "'flac'. Yalnız --input-kind video ile anlamlıdır."
)
_JOB_ROOT_HELP = "İş dizini kökü (varsayılan: JOB_DIR veya ~/.subtitleflow/jobs)."
_ENV_HELP = "Kullanılacak .env dosyası (varsayılan: çalışma dizinindeki .env)."
_YTDLP_BIN_HELP = (
    "Harici yt-dlp yürütücüsü (varsayılan: YOUTUBE_YTDLP_BIN veya PATH'teki "
    "yt-dlp). Proje yt-dlp'yi güncellemez ve Python modülü olarak import etmez."
)
_YOUTUBE_URL_HELP = (
    "Tek, herkese açık YouTube video bağlantısı (watch/youtu.be/shorts/embed). "
    "Oynatma listesi, canlı yayın, kanal ve özel içerik kapsam dışıdır."
)
_PAID_HELP = (
    "Ücretli API çağrılarını açıkça etkinleştir; rezervasyon üst sınırları da "
    "gerekir. Varsayılan kapalıdır."
)
_SOURCE_HELP = "Kaynak dil kodu/alias (örn. ar). Verilmezse algılamaya bırakılır."
_KEYTERM_HELP = "STT için özel terim; birden çok kez verilebilir."
_RETRY_HELP = (
    "Önceki/yarım denemeyi açıkça yeniden göndermeye izin ver. Ücretli bir "
    "rotada (Scribe veya Google Basic) yeni bir ücretli istek doğurabilir; "
    "varsayılan kapalıdır ve --burn-video/onay kutusu bunu açmaz."
)
_TRANSCRIPT_DIR_HELP = (
    "Kaynak tam metin Markdown çıktı dizini (varsayılan: depo kökü "
    "'outputs/transkriptler/')."
)
_OVERWRITE_TRANSCRIPT_HELP = (
    "Aynı video için içeriği değişmiş mevcut kaynak Markdown dosyasını açıkça "
    "değiştir."
)
_TRANSLATION_DIR_HELP = (
    "Türkçe çeviri Markdown çıktı dizini (varsayılan: 'outputs/ceviriler/'; "
    "--transcript-dir verilirse onun kardeş 'ceviriler' klasörü)."
)
_OVERWRITE_TRANSLATION_HELP = (
    "Geriye dönük uyumluluk seçeneği; davranışı değiştirmez. Farklı içerikli "
    "eski Türkçe çeviri her durumda yayından önce 'outputs/ceviriler/_arsiv/' "
    "altına arşivlenir."
)
_TRANSLATE_HELP = (
    "Kaynak transkriptten sonra tam metin Türkçe çevirisi adımını etkinleştir. "
    "Çeviri her zaman Google Translation LLM (Basic v2 API anahtarı) ile yapılır."
)
_CLEANUP_HELP = (
    "Her iki Markdown güvenle yayımlandıktan sonra tamamlanan işin dizinini ve "
    "ona ait kaynak ses önbelleğini sil (masaüstü varsayılanı). Yalnız tam "
    "zincir çeviri akışında anlamlıdır; yarım/başarısız işlere dokunulmaz ve "
    "temizlik başarısız olursa çıktılar korunur."
)
_BURN_VIDEO_HELP = (
    "Tam videoyu indirip Türkçe altyazıyı kalıcı olarak gömülü (burn-in) bir MP4 "
    "üret. Varsayılan kapalıdır ve yalnız tam zincir çeviri akışında kullanılır; "
    "zaman kodları kanonik transkriptten gelir, uydurulmaz."
)
_VIDEO_DIR_HELP = (
    "Altyazılı video çıktı dizini (varsayılan: 'outputs/videolar/')."
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"subtitle-flow {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Sürümü yaz ve çık.",
    ),
) -> None:
    """SubtitleFlow komut satırı."""


def _load(
    *,
    source_language: str | None,
    keyterm: list[str],
    job_root: str | None,
    env_file: str | None,
    no_env_file: bool,
    allow_paid: bool,
    total_cap: str | None,
    per_call_cap: str | None,
    per_call_upper: str | None,
    currency: str | None,
    video_container: str | None = None,
    youtube_ytdlp_bin: str | None = None,
) -> tuple[CliEnvironment, CliSettings]:
    if env_file is not None and no_env_file:
        raise ConfigError(
            "ENV_FILE_COLLISION",
            "--env-file ile --no-env-file birlikte kullanılamaz",
        )
    environment = load_environment(
        env_file,
        explicit=env_file is not None,
        use_default=not no_env_file,
    )
    overrides = CliOverrides(
        source_language=source_language,
        keyterms=tuple(keyterm),
        job_root=job_root,
        allow_paid_api_calls=True if allow_paid else None,
        total_reservation_cap=total_cap,
        per_call_cap=per_call_cap,
        per_call_upper_bound=per_call_upper,
        currency=currency,
        video_container=video_container,
        youtube_ytdlp_bin=youtube_ytdlp_bin,
    )
    return environment, build_settings(environment, overrides=overrides)


def _finish(result: CommandResult) -> None:
    typer.echo(result.message)
    raise typer.Exit(result.exit_code)


def _abort(message: str) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(ExitCode.INVALID_INPUT)


@app.command()
def process(
    audio: str = typer.Argument(..., help=_AUDIO_HELP),
    source_language: str | None = typer.Option(None, "--source-language", help=_SOURCE_HELP),
    keyterm: list[str] = typer.Option([], "--keyterm", help=_KEYTERM_HELP),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    job_id: str | None = typer.Option(None, "--job-id", help="Açık iş kimliği."),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    input_kind: str = typer.Option("audio", "--input-kind", help=_INPUT_KIND_HELP),
    video_container: str | None = typer.Option(
        None, "--video-container", help=_VIDEO_CONTAINER_HELP
    ),
) -> None:
    """Sesi doğrula, STT ve MT çalıştır, çıktıları üret."""

    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=keyterm,
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
            video_container=video_container,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_PROCESS,
            settings=settings,
            environment=environment,
            audio=audio,
            job_id=job_id,
            allow_remote_retry=allow_remote_retry,
            input_kind=input_kind,
            runtime_factory=_active_runtime_factory(),
        )
    )


@app.command()
def transcribe(
    audio: str = typer.Argument(..., help=_AUDIO_HELP),
    source_language: str | None = typer.Option(None, "--source-language", help=_SOURCE_HELP),
    keyterm: list[str] = typer.Option([], "--keyterm", help=_KEYTERM_HELP),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    job_id: str | None = typer.Option(None, "--job-id", help="Açık iş kimliği."),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    input_kind: str = typer.Option("audio", "--input-kind", help=_INPUT_KIND_HELP),
    video_container: str | None = typer.Option(
        None, "--video-container", help=_VIDEO_CONTAINER_HELP
    ),
) -> None:
    """Yalnız STT aşamasını çalıştır; çeviri sonraki komuta kalır."""

    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=keyterm,
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
            video_container=video_container,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_TRANSCRIBE,
            settings=settings,
            environment=environment,
            audio=audio,
            job_id=job_id,
            allow_remote_retry=allow_remote_retry,
            input_kind=input_kind,
            runtime_factory=_active_runtime_factory(),
        )
    )


@app.command()
def process_youtube(
    url: str = typer.Argument(..., help=_YOUTUBE_URL_HELP),
    source_language: str | None = typer.Option(None, "--source-language", help=_SOURCE_HELP),
    keyterm: list[str] = typer.Option([], "--keyterm", help=_KEYTERM_HELP),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    job_id: str | None = typer.Option(None, "--job-id", help="Açık iş kimliği."),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    ytdlp_bin: str | None = typer.Option(None, "--ytdlp-bin", help=_YTDLP_BIN_HELP),
) -> None:
    """Tek YouTube videosunun sesini alıp STT ve MT çalıştır, çıktıları üret."""

    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=keyterm,
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
            youtube_ytdlp_bin=ytdlp_bin,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_PROCESS,
            settings=settings,
            environment=environment,
            job_id=job_id,
            allow_remote_retry=allow_remote_retry,
            input_kind="youtube",
            youtube_url=url,
            ytdlp_bin=settings.youtube_ytdlp_bin,
            runtime_factory=_active_runtime_factory(),
        )
    )


@app.command()
def transcribe_youtube(
    url: str = typer.Argument(..., help=_YOUTUBE_URL_HELP),
    source_language: str | None = typer.Option(None, "--source-language", help=_SOURCE_HELP),
    keyterm: list[str] = typer.Option([], "--keyterm", help=_KEYTERM_HELP),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    job_id: str | None = typer.Option(None, "--job-id", help="Açık iş kimliği."),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    ytdlp_bin: str | None = typer.Option(None, "--ytdlp-bin", help=_YTDLP_BIN_HELP),
    transcript_dir: str | None = typer.Option(
        None, "--transcript-dir", help=_TRANSCRIPT_DIR_HELP
    ),
    overwrite_transcript: bool = typer.Option(
        False, "--overwrite-transcript", help=_OVERWRITE_TRANSCRIPT_HELP
    ),
    translate: bool = typer.Option(
        False, "--translate", help=_TRANSLATE_HELP
    ),
    translation_dir: str | None = typer.Option(
        None, "--translation-dir", help=_TRANSLATION_DIR_HELP
    ),
    overwrite_translation: bool = typer.Option(
        False, "--overwrite-translation", help=_OVERWRITE_TRANSLATION_HELP
    ),
    cleanup: bool = typer.Option(False, "--cleanup", help=_CLEANUP_HELP),
    burn_video: bool = typer.Option(False, "--burn-video", help=_BURN_VIDEO_HELP),
    video_dir: str | None = typer.Option(
        None, "--video-dir", help=_VIDEO_DIR_HELP
    ),
) -> None:
    """Tek YouTube videosunda STT çalıştır (ve isteğe bağlı Türkçe çeviri)."""

    if cleanup and not translate:
        _abort(
            "yapılandırma hatası: --cleanup yalnız --translate ile "
            "birlikte kullanılır"
        )
    if burn_video and not translate:
        _abort(
            "yapılandırma hatası: --burn-video yalnız --translate ile "
            "birlikte kullanılır"
        )
    if video_dir is not None and not burn_video:
        # An explicit video output directory is only meaningful for the burned-in
        # video opt-in; without it the flag would be silently ignored. Refuse it
        # instead of pretending it took effect.
        _abort(
            "yapılandırma hatası: --video-dir yalnız --burn-video ile birlikte "
            "kullanılır"
        )
    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=keyterm,
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
            youtube_ytdlp_bin=ytdlp_bin,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_FULLCHAIN if translate else ACTION_TRANSCRIBE,
            settings=settings,
            environment=environment,
            job_id=job_id,
            allow_remote_retry=allow_remote_retry,
            input_kind="youtube",
            youtube_url=url,
            ytdlp_bin=settings.youtube_ytdlp_bin,
            transcript_dir=transcript_dir,
            overwrite_transcript=overwrite_transcript,
            translation_dir=translation_dir,
            overwrite_translation=overwrite_translation,
            runtime_factory=_active_runtime_factory(),
            cleanup=cleanup,
            burn_video=burn_video,
            video_dir=video_dir,
        )
    )


@app.command()
def translate_youtube(
    url: str = typer.Argument(..., help=_YOUTUBE_URL_HELP),
    source_language: str | None = typer.Option(None, "--source-language", help=_SOURCE_HELP),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    job_id: str | None = typer.Option(None, "--job-id", help="Açık iş kimliği."),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    transcript_dir: str | None = typer.Option(
        None, "--transcript-dir", help=_TRANSCRIPT_DIR_HELP
    ),
    translation_dir: str | None = typer.Option(
        None, "--translation-dir", help=_TRANSLATION_DIR_HELP
    ),
    overwrite_translation: bool = typer.Option(
        False, "--overwrite-translation", help=_OVERWRITE_TRANSLATION_HELP
    ),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    cleanup: bool = typer.Option(False, "--cleanup", help=_CLEANUP_HELP),
) -> None:
    """Arşivlenmiş kaynak transkriptten Türkçe çeviri üret (ses/STT yok).

    Yalnız doğrulanmış yerel kanıt okunur; yeni indirme veya STT çağrısı yapılmaz.
    Çeviri her zaman Google Translation LLM (Basic v2 API anahtarı) ile yapılır.
    Daha önce gönderilmiş ama yeniden kullanılamayan bir grup varsayılan olarak
    yeniden gönderilmez; açık --allow-remote-retry yeni bir ücretli deneme
    yetkilendirebilir.
    """

    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=[],
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_TRANSLATE_ARCHIVED,
            settings=settings,
            environment=environment,
            job_id=job_id,
            input_kind="youtube",
            youtube_url=url,
            transcript_dir=transcript_dir,
            translation_dir=translation_dir,
            overwrite_translation=overwrite_translation,
            allow_remote_retry=allow_remote_retry,
            cleanup=cleanup,
        )
    )


@app.command()
def translate(
    job_id: str = typer.Argument(..., help="Mevcut iş kimliği."),
    source_language: str | None = typer.Option(
        None, "--source-language", help="Yalnız saklanan değeri doğrulamak için."
    ),
    keyterm: list[str] = typer.Option(
        [], "--keyterm", help="Yalnız saklanan terimleri doğrulamak için."
    ),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    video_container: str | None = typer.Option(
        None, "--video-container", help="Yalnız saklanan değeri doğrulamak için."
    ),
) -> None:
    """Mevcut işin yalnız MT aşamasını çalıştır (STT yeniden çağrılmaz)."""

    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=keyterm,
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
            video_container=video_container,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_TRANSLATE,
            settings=settings,
            environment=environment,
            job_id=job_id,
            allow_remote_retry=allow_remote_retry,
            runtime_factory=_active_runtime_factory(),
        )
    )


@app.command()
def resume(
    job_id: str = typer.Argument(..., help="Mevcut iş kimliği."),
    source_language: str | None = typer.Option(
        None, "--source-language", help="Yalnız saklanan değeri doğrulamak için."
    ),
    keyterm: list[str] = typer.Option(
        [], "--keyterm", help="Yalnız saklanan terimleri doğrulamak için."
    ),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    allow_paid: bool = typer.Option(False, "--allow-paid-api-calls", help=_PAID_HELP),
    total_cap: str | None = typer.Option(None, "--total-reservation-cap"),
    per_call_cap: str | None = typer.Option(None, "--per-call-cap"),
    per_call_upper: str | None = typer.Option(None, "--per-call-upper-bound"),
    currency: str | None = typer.Option(None, "--currency"),
    allow_remote_retry: bool = typer.Option(False, "--allow-remote-retry", help=_RETRY_HELP),
    video_container: str | None = typer.Option(
        None, "--video-container", help="Yalnız saklanan değeri doğrulamak için."
    ),
) -> None:
    """Mevcut işi doğrulanmış aşamaları yeniden kullanarak sürdür."""

    try:
        environment, settings = _load(
            source_language=source_language,
            keyterm=keyterm,
            job_root=job_root,
            env_file=env_file,
            no_env_file=no_env_file,
            allow_paid=allow_paid,
            total_cap=total_cap,
            per_call_cap=per_call_cap,
            per_call_upper=per_call_upper,
            currency=currency,
            video_container=video_container,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    _finish(
        execute(
            ACTION_RESUME,
            settings=settings,
            environment=environment,
            job_id=job_id,
            allow_remote_retry=allow_remote_retry,
            runtime_factory=_active_runtime_factory(),
        )
    )


@app.command()
def ui(
    port: int | None = typer.Option(
        None, "--port", help="Yerel arayüz portu (yalnız 127.0.0.1)."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Tarayıcıyı otomatik açma."
    ),
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    transcript_dir: str | None = typer.Option(
        None, "--transcript-dir", help=_TRANSCRIPT_DIR_HELP
    ),
    translation_dir: str | None = typer.Option(
        None, "--translation-dir", help=_TRANSLATION_DIR_HELP
    ),
    video_dir: str | None = typer.Option(None, "--video-dir", help=_VIDEO_DIR_HELP),
) -> None:
    """Yerel tarayıcı arayüzünü başlat (yalnız 127.0.0.1).

    Arayüz tek seferde tek iş çalıştırır: kaynak STT ElevenLabs Scribe v2,
    Türkçe çeviri Google Translation LLM (Basic v2 API anahtarı). Rota/sağlayıcı
    seçimi yoktur; sağlayıcıya yalnız açık onayla gidilir.
    """

    if env_file is not None and no_env_file:
        _abort(
            "yapılandırma hatası: --env-file ile --no-env-file birlikte "
            "kullanılamaz"
        )
    try:
        environment = load_environment(
            env_file,
            explicit=env_file is not None,
            use_default=not no_env_file,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")

    from subtitle_flow import web_ui

    resolved_port = web_ui.DEFAULT_PORT if port is None else port
    if not 0 <= resolved_port <= 65535:
        _abort("yapılandırma hatası: port 0-65535 aralığında olmalı")
    try:
        web_ui.serve(
            environment=environment,
            transcript_dir=transcript_dir,
            translation_dir=translation_dir,
            video_dir=video_dir,
            job_root=job_root,
            port=resolved_port,
            open_browser=not no_browser,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    except OSError as exc:
        _abort(f"arayüz başlatılamadı: {exc}")


@app.command()
def desktop(
    env_file: str | None = typer.Option(None, "--env-file", help=_ENV_HELP),
    no_env_file: bool = typer.Option(False, "--no-env-file", help=".env yükleme."),
    job_root: str | None = typer.Option(None, "--job-root", help=_JOB_ROOT_HELP),
    transcript_dir: str | None = typer.Option(
        None, "--transcript-dir", help=_TRANSCRIPT_DIR_HELP
    ),
    translation_dir: str | None = typer.Option(
        None, "--translation-dir", help=_TRANSLATION_DIR_HELP
    ),
    video_dir: str | None = typer.Option(None, "--video-dir", help=_VIDEO_DIR_HELP),
) -> None:
    """Yerel arayüzü kendi masaüstü penceresinde başlat (tarayıcı açmaz).

    Aynı ``127.0.0.1`` arayüzünü ve aynı kaynak-STT + Türkçe-MT zincirini
    kullanır; pencere kapanana kadar dinler. Sağlayıcı çağrısı ve model yükleme
    yalnız açık onayla, iş sırasında yapılır.
    """

    if env_file is not None and no_env_file:
        _abort(
            "yapılandırma hatası: --env-file ile --no-env-file birlikte "
            "kullanılamaz"
        )
    try:
        environment = load_environment(
            env_file,
            explicit=env_file is not None,
            use_default=not no_env_file,
        )
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")

    from subtitle_flow import desktop as desktop_host

    try:
        desktop_host.run_desktop(
            environment=environment,
            transcript_dir=transcript_dir,
            translation_dir=translation_dir,
            video_dir=video_dir,
            job_root=job_root,
        )
    except desktop_host.DesktopUnavailableError as exc:
        _abort(str(exc))
    except desktop_host.DesktopStartupError as exc:
        _abort(f"masaüstü penceresi başlatılamadı: {exc}")
    except ConfigError as exc:
        _abort(f"yapılandırma hatası: {exc.message}")
    except OSError as exc:
        _abort(f"masaüstü penceresi başlatılamadı: {exc}")


@app.command()
def migrate_outputs(
    source_dir: str | None = typer.Option(
        None,
        "--source-dir",
        help="Eski transkript klasörü (varsayılan: çalışma dizini 'transkriptler/').",
    ),
    target_dir: str | None = typer.Option(
        None,
        "--target-dir",
        help="Yeni hedef klasör (varsayılan: 'outputs/transkriptler/').",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Taşımayı gerçekten uygula; varsayılan yalnız rapor (dry-run).",
    ),
) -> None:
    """Eski ``transkriptler/`` içeriğini kayıpsız, idempotent biçimde taşı."""

    source = Path(source_dir).expanduser() if source_dir else Path.cwd() / "transkriptler"
    target = (
        Path(target_dir).expanduser()
        if target_dir
        else Path.cwd() / "outputs" / "transkriptler"
    )
    try:
        report = migrate_transcripts(source, target, dry_run=not apply)
    except MigrationError as exc:
        _abort(f"taşıma hatası ({exc.code}): {exc.message}")
    lines = [
        f"Kaynak: {report.source}",
        f"Hedef: {report.target}",
        f"Mod: {'uygulandı' if not report.dry_run else 'yalnız rapor (dry-run)'}",
        f"Taşınan: {len(report.moved)}",
        f"Aynı içerik nedeniyle kaldırılan kaynak: {len(report.removed_identical)}",
        f"Çakışma: {len(report.collisions)}",
    ]
    for collision in report.collisions:
        lines.append(f"  ! {collision.relative}: {collision.reason}")
    if report.skipped:
        lines.append("Atlandı: " + ", ".join(report.skipped))
    if report.collisions:
        lines.append(
            "Not: çakışan kaynaklar değiştirilmedi ve silinmedi; elle incelenmeli."
        )
    typer.echo("\n".join(lines))
    raise typer.Exit(0 if report.ok else int(ExitCode.REVIEW_REQUIRED))


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":  # pragma: no cover - module execution smoke
    main()
