"""Loopback-only local web UI for the YouTube transcript + Turkish MT workflow.

The UI stays a *thin wrapper* over the existing command line: one background job
at a time reuses :func:`subtitle_flow.cli.execute` -- the desktop full chain
(``fullchain``: source STT via ElevenLabs Scribe, then Google Translation Basic
v2 Turkish MT) or the retranslation of an already archived source
(``translate-archived``). No pipeline, provider or model logic is re-implemented
here, and nothing is called at start-up.

Product rules enforced here
--------------------------
* The inference route is fixed: ElevenLabs Scribe v2 for STT and Google
  Translation Basic v2 (API key) for Turkish MT. Every dispatch is paid and needs
  the explicit paid consent with reservation caps.
* Source (``outputs/transkriptler/<id>.md``) and Turkish
  (``outputs/ceviriler/<id>.md``) documents are separate views; the library can
  reopen either and start/retry the Turkish MT from an archived source without
  downloading audio or calling STT.
* One native full-chain job at a time; the server and UI stay responsive and
  report honest stage messages (source-only partial failure is explicit).

Hardening notes
---------------
* The server binds ``127.0.0.1`` only -- never a LAN address, never ``0.0.0.0``.
* Mutations require the per-process random CSRF token plus a loopback ``Host``
  and same-origin ``Origin`` check; there is no permissive CORS.
* Request bodies are bounded and must be JSON; only validated ``<video_id>.md``
  files under the two configured output directories can be read or downloaded.
* Credentials, ``.env`` contents and the full internal configuration are never
  sent to the browser, logged or rendered.
"""

from __future__ import annotations

import json
import re
import secrets
import socketserver
import threading
import time
import webbrowser
from dataclasses import dataclass, field, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Final

from subtitle_flow.cli import (
    ACTION_FULLCHAIN,
    ACTION_TRANSLATE_ARCHIVED,
    ExitCode,
    execute,
)
from subtitle_flow.cli_config import (
    CliEnvironment,
    CliOverrides,
    ConfigError,
    build_settings,
    parse_strict_decimal,
)
from subtitle_flow.config import default_job_root
from subtitle_flow.output_paths import (
    default_source_dir,
    translation_dir_for_source,
    video_dir_for_source,
)
from subtitle_flow.storage import JobStore, StorageError
from subtitle_flow.youtube_source import YouTubeMediaError, canonicalize_youtube_url

__all__ = [
    "DEFAULT_PORT",
    "UiOutcome",
    "UiRequest",
    "create_server",
    "serve",
]

#: Default loopback port. If it is busy the server falls back to an ephemeral
#: port so a double-clicked launcher never fails on a stale instance.
DEFAULT_PORT: Final[int] = 8765

#: Only a small JSON control body is ever accepted.
_MAX_BODY_BYTES: Final[int] = 64 * 1024

#: A document file name is the validated video id (the same shape the Markdown
#: exporter enforces), never an arbitrary path component.
_SAFE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_MAX_PREVIEW_BYTES: Final[int] = 2 * 1024 * 1024

#: The browser-visible pilot languages (labels only; the codes are canonical).
_PILOT_LANGUAGES: Final[tuple[tuple[str, str], ...]] = (
    ("en", "İngilizce"),
    ("ar", "Arapça"),
    ("fa", "Farsça"),
    ("ru", "Rusça"),
    ("de", "Almanca"),
    ("he", "İbranice"),
)

#: Static assets served from the package. Everything else is a 404.
_ASSETS: Final[dict[str, tuple[str, str]]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

#: Actionable Turkish hints appended to a mapped failure message.
_HINTS: Final[tuple[tuple[str, str], ...]] = (
    (
        "PAID_CALLS_DISABLED",
        "'Ücretli sağlayıcı kullanımına izin ver' kutusunu işaretleyin.",
    ),
    (
        "FULLTEXT_MT_NO_SOURCE",
        "Bu video için arşivlenmiş kaynak yok; önce 'Transkripti al' ile kaynak "
        "transkripti üretin.",
    ),
    (
        "MT_INPUT_TOO_LONG",
        "Kaynak metin tek bir çeviri isteğine sığmadı (tek bir kelime/parça bile "
        "bütçeyi aşıyor); metin sessizce kesilmez veya bölünmez. Daha kısa bir "
        "kaynakla yeniden deneyin.",
    ),
    (
        "otomatik yeniden gönderim durduruldu",
        "Bu iş daha önce yarım kaldı ve güvenlik gereği otomatik yeniden "
        "gönderim durdu; inceleyip komut satırından `--allow-remote-retry` ile "
        "açıkça sürdürün.",
    ),
)


@dataclass(frozen=True)
class UiRequest:
    """One validated UI request."""

    url: str
    video_id: str
    allow_paid: bool
    #: Total local reservation cap shared by every selected paid stage. It is a
    #: local ledger bound only and never a provider-invoice guarantee.
    reservation: str | None
    source_language: str | None
    #: Per-call upper reservation bound. It is independent of the total cap:
    #: ``per_call_cap == per_call_upper_bound``, while the total stays separate,
    #: so a multi-call job is not stopped after its first paid dispatch.
    per_call_reservation: str | None = None
    #: ``fullchain`` runs source STT + MT; ``translate`` runs MT only.
    action: str = "fullchain"
    #: Opt-in burned-in Turkish subtitle video. Only valid for ``fullchain``.
    burn_video: bool = False


@dataclass(frozen=True)
class UiOutcome:
    """The result of running one UI job through the shared command layer."""

    exit_code: int
    message: str
    video_id: str | None = None
    file_name: str | None = None
    translation_file_name: str | None = None
    #: Published burned-in Turkish subtitle MP4 file name (opt-in only).
    video_file_name: str | None = None
    partial: bool = False
    needs_review: bool = False
    translation_error: str | None = None
    video_error: str | None = None
    cleanup_warning: str | None = None


@dataclass
class _UiJob:
    job_id: str
    url: str
    video_id: str
    action: str = "fullchain"
    state: str = "running"
    stage: str = ""
    message: str = ""
    exit_code: int | None = None
    source_file_name: str | None = None
    translation_file_name: str | None = None
    video_file_name: str | None = None
    burn_video: bool = False
    has_source: bool = False
    has_translation: bool = False
    has_video: bool = False
    partial: bool = False
    needs_review: bool = False
    created_at: float = field(default_factory=time.time)

    def as_payload(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "stage": self.stage,
            "message": self.message,
            "exit_code": self.exit_code,
            "video_id": self.video_id,
            "action": self.action,
            "burn_video": self.burn_video,
            "source_file_name": self.source_file_name,
            "translation_file_name": self.translation_file_name,
            "video_file_name": self.video_file_name,
            "has_source": self.has_source,
            "has_translation": self.has_translation,
            "has_video": self.has_video,
            "partial": self.partial,
            "needs_review": self.needs_review,
        }


class _UiState:
    """Shared server state: one job at a time, guarded by a single lock."""

    def __init__(
        self,
        *,
        environment: CliEnvironment,
        source_dir: Path,
        translation_dir: Path,
        video_dir: Path | None = None,
        job_root: str | None = None,
    ) -> None:
        self.environment = environment
        self.source_dir = source_dir
        #: Backwards-compatible alias for the source document directory.
        self.transcript_dir = source_dir
        self.translation_dir = translation_dir
        #: Burned-in subtitle video directory; defaults to the source sibling so
        #: an offline caller that does not care about the opt-in keeps working.
        self.video_dir = (
            Path(video_dir)
            if video_dir is not None
            else video_dir_for_source(source_dir)
        )
        self.job_root = job_root
        self.csrf_token = secrets.token_urlsafe(32)
        self.jobs: dict[str, _UiJob] = {}
        self._lock = threading.Lock()
        self._busy = False
        self._closing = False

    # -- configuration ---------------------------------------------------- #

    def _settings(self, request: UiRequest):
        # One fixed paid route: Scribe STT and Google Basic MT share the same
        # two-value local reservation policy, taken from the explicit UI consent
        # and the two reservation values. A missing consent caps nothing and the
        # provider refuses before dispatch.
        total = request.reservation if request.allow_paid else None
        per_call = (
            _resolve_per_call(request.per_call_reservation)
            if request.allow_paid
            else None
        )
        overrides = CliOverrides(
            source_language=request.source_language,
            job_root=self.job_root,
            allow_paid_api_calls=True if request.allow_paid else False,
            total_reservation_cap=total,
            per_call_cap=per_call,
            per_call_upper_bound=per_call,
            currency="USD" if request.allow_paid else None,
        )
        return build_settings(self.environment, overrides=overrides)

    def run_request(
        self, request: UiRequest, *, progress: Callable[[str], None] | None = None
    ) -> UiOutcome:
        """Run one request through the real CLI executor and report the result.

        The outcome is derived from the returned :class:`CommandResult` and the
        exact documents it bound -- never from the mere presence of a file on
        disk -- so a stale previous output can never turn the current failure
        into a success. ``progress`` forwards the executor's honest stage text.
        """

        try:
            settings = self._settings(request)
        except ConfigError as exc:
            return UiOutcome(
                exit_code=int(ExitCode.INVALID_INPUT),
                message=_augment(f"{exc.message} ({exc.code})"),
            )
        try:
            if request.action == "translate":
                result = execute(
                    ACTION_TRANSLATE_ARCHIVED,
                    settings=settings,
                    environment=self.environment,
                    input_kind="youtube",
                    youtube_url=request.url or None,
                    video_id=request.video_id,
                    transcript_dir=str(self.source_dir),
                    translation_dir=str(self.translation_dir),
                    overwrite_translation=False,
                    progress=progress,
                    cleanup=True,
                )
            else:
                # A cloud Scribe route stays False for a paid resend (an explicit
                # decision is required) and the pipeline still validates the
                # durable history/hashes before anything is re-dispatched. The
                # Google Basic MT route is paid, so the separate MT knob is forced
                # closed: a paid Basic resend needs the explicit CLI
                # ``--allow-remote-retry``.
                #
                # The press is also the explicit opt-in to replace the canonical
                # source document with Scribe's result. The source writer archives
                # the differing earlier document under ``_arsiv/`` before the
                # atomic replacement; the CLI default stays no-clobber.
                result = execute(
                    ACTION_FULLCHAIN,
                    settings=settings,
                    environment=self.environment,
                    input_kind="youtube",
                    youtube_url=request.url,
                    transcript_dir=str(self.source_dir),
                    translation_dir=str(self.translation_dir),
                    allow_remote_retry=False,
                    allow_mt_remote_retry=False,
                    overwrite_transcript=True,
                    overwrite_translation=False,
                    progress=progress,
                    cleanup=True,
                    burn_video=request.burn_video,
                    video_dir=str(self.video_dir),
                )
        except ConfigError as exc:  # defensive: execute maps its own failures
            return UiOutcome(
                exit_code=int(ExitCode.INVALID_INPUT),
                message=_augment(f"{exc.message} ({exc.code})"),
            )

        video_id = result.video_id or request.video_id
        source_file = (
            Path(result.source_markdown).name if result.source_markdown else None
        )
        translation_file = (
            Path(result.translation_markdown).name
            if result.translation_markdown
            else None
        )
        video_file = Path(result.video_file).name if result.video_file else None
        needs_review = result.exit_code == int(ExitCode.REVIEW_REQUIRED)
        outcome = UiOutcome(
            exit_code=int(result.exit_code),
            message=result.message or "İş tamamlanamadı.",
            video_id=video_id,
            file_name=source_file,
            translation_file_name=translation_file,
            video_file_name=video_file,
            partial=bool(result.partial),
            needs_review=needs_review,
            translation_error=result.translation_error,
            video_error=result.video_error,
            cleanup_warning=result.cleanup_warning,
        )
        return replace(outcome, message=_augment(_ui_message(outcome)))

    # -- jobs ------------------------------------------------------------- #

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def begin_close(self) -> bool:
        """Atomically settle a close request against job start.

        Returns ``False`` while a job is running so the native window can veto
        the close and ask the operator to wait; otherwise marks the state as
        closing so that no *new* job can be admitted between this check and the
        window actually disappearing. This is the single gate that prevents a
        paid call from being silently dropped mid-flight.
        """

        with self._lock:
            if self._busy:
                return False
            self._closing = True
            return True

    def cancel_close(self) -> None:
        """Release the gate if a close request is abandoned without closing."""

        with self._lock:
            self._closing = False

    def mark_closing(self) -> bool:
        """Refuse new jobs immediately and report whether one is running.

        Used by a graceful interrupt (``Ctrl+C``): it sets the same ``_closing``
        gate as :meth:`begin_close` so no new job can start, and returns whether
        an in-flight job still needs to finish before the window is torn down.
        """

        with self._lock:
            self._closing = True
            return self._busy

    def wait_until_idle(self, *, timeout: float | None = None) -> bool:
        """Block until no job is active; ``False`` if ``timeout`` elapses first.

        The gate is expected to be closed already (see :meth:`mark_closing`), so
        this only waits for the running job to reach a final state. No job is
        cancelled, retried or replaced.
        """

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._busy:
                    return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def start(self, request: UiRequest) -> _UiJob | None:
        """Register a job and start its worker; ``None`` when one is running."""

        with self._lock:
            if self._busy or self._closing:
                return None
            self._busy = True
            job = _UiJob(
                job_id=secrets.token_hex(8),
                url=request.url,
                video_id=request.video_id,
                action=request.action,
                burn_video=request.burn_video,
            )
            self.jobs[job.job_id] = job
        thread = threading.Thread(
            target=self._worker, args=(job, request), name="subtitle-flow-ui-job", daemon=True
        )
        thread.start()
        return job

    def _worker(self, job: _UiJob, request: UiRequest) -> None:
        def report(stage: str) -> None:
            with self._lock:
                job.stage = stage

        try:
            outcome = self.run_request(request, progress=report)
        except Exception:  # noqa: BLE001 - never leak a traceback to the browser
            outcome = UiOutcome(
                exit_code=int(ExitCode.FAILURE),
                message="Beklenmeyen bir hata oluştu; iş yeniden çalıştırılabilir.",
            )
        with self._lock:
            try:
                job.exit_code = outcome.exit_code
                job.video_id = outcome.video_id or job.video_id
                job.source_file_name = outcome.file_name
                job.translation_file_name = outcome.translation_file_name
                job.video_file_name = outcome.video_file_name
                job.has_source = outcome.file_name is not None
                job.has_translation = outcome.translation_file_name is not None
                job.has_video = outcome.video_file_name is not None
                job.partial = outcome.partial
                job.needs_review = outcome.needs_review
                job.state = _final_state(outcome)
                job.message = outcome.message or _STATE_MESSAGE[job.state]
                job.stage = ""
            finally:
                self._busy = False

    def get_job(self, job_id: str) -> _UiJob | None:
        with self._lock:
            return self.jobs.get(job_id)

    # -- output files ----------------------------------------------------- #

    @staticmethod
    def _confined(base: Path, video_id: str) -> Path | None:
        if not isinstance(video_id, str) or _SAFE_NAME_RE.match(video_id) is None:
            return None
        resolved_base = base.resolve()
        target = (resolved_base / f"{video_id}.md").resolve()
        try:
            target.relative_to(resolved_base)
        except ValueError:
            return None
        return target

    def transcript_path(self, video_id: str) -> Path | None:
        """Resolve ``video_id`` to a confined source ``<id>.md`` path, or ``None``."""

        return self._confined(self.source_dir, video_id)

    def translation_path(self, video_id: str) -> Path | None:
        """Resolve ``video_id`` to a confined Turkish ``<id>.md`` path, or ``None``."""

        return self._confined(self.translation_dir, video_id)

    @staticmethod
    def _scan(base: Path) -> list[dict[str, Any]]:
        if not base.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for entry in base.iterdir():
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
            except OSError:
                continue
            if entry.suffix != ".md":
                continue
            video_id = entry.stem
            if _SAFE_NAME_RE.match(video_id) is None:
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            items.append(
                {
                    "video_id": video_id,
                    "title": _read_title(entry),
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        return items

    def list_library(self) -> list[dict[str, Any]]:
        """Merge both document sets into one library view keyed by video id.

        ``translatable`` reports whether durable job evidence for that video
        exists. A published Markdown document alone is not accepted as Turkish MT
        input, so the UI can make the unavailable action explicit instead of
        offering a doomed one.
        """

        merged: dict[str, dict[str, Any]] = {}
        for item in self._scan(self.source_dir):
            merged[item["video_id"]] = {
                "video_id": item["video_id"],
                "title": item["title"],
                "size_bytes": item["size_bytes"],
                "modified": item["modified"],
                "source": True,
                "translation": False,
            }
        for item in self._scan(self.translation_dir):
            entry = merged.get(item["video_id"])
            if entry is None:
                merged[item["video_id"]] = {
                    "video_id": item["video_id"],
                    "title": item["title"],
                    "size_bytes": item["size_bytes"],
                    "modified": item["modified"],
                    "source": False,
                    "translation": True,
                }
            else:
                entry["translation"] = True
                entry["modified"] = max(entry["modified"], item["modified"])
                if not entry["title"]:
                    entry["title"] = item["title"]
        archived = self._archived_video_ids()
        items = list(merged.values())
        for item in items:
            item["translatable"] = item["video_id"] in archived
        items.sort(key=lambda item: item["modified"], reverse=True)
        return items

    def _job_root_path(self) -> Path:
        if self.job_root:
            return Path(self.job_root).expanduser()
        return Path(default_job_root())

    def _archived_video_ids(self) -> set[str]:
        """Video ids that have durable job evidence usable as Turkish MT input.

        This is a read-only scan of the configured job root; it never downloads,
        calls a provider or mutates a job. A video listed here has an archived
        source that :func:`subtitle_flow.cli._find_source_job` can recover.
        """

        root = self._job_root_path()
        found: set[str] = set()
        try:
            entries = sorted(root.iterdir())
        except OSError:
            return found
        for entry in entries:
            try:
                if entry.is_symlink() or not entry.is_dir():
                    continue
                store = JobStore(root, entry.name)
                if not store.exists():
                    continue
                stored = store.read_input()
            except (StorageError, OSError):
                continue
            origin = stored.config.youtube_origin
            if origin is not None and origin.video_id:
                found.add(origin.video_id)
        return found

    def list_transcripts(self) -> list[dict[str, Any]]:
        """Legacy alias: the merged library view."""

        return self.list_library()

    def _read(self, path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                data = handle.read(_MAX_PREVIEW_BYTES + 1)
        except OSError:
            return None
        if len(data) > _MAX_PREVIEW_BYTES:
            return None
        return data.decode("utf-8", errors="replace")

    def read_transcript(self, video_id: str) -> str | None:
        return self._read(self.transcript_path(video_id))

    def read_translation(self, video_id: str) -> str | None:
        return self._read(self.translation_path(video_id))

    def bootstrap(self) -> dict[str, Any]:
        library = self.list_library()
        google_configured = google_translation_configured(self.environment)
        return {
            "csrf_token": self.csrf_token,
            "languages": [
                {"code": code, "name": name} for code, name in _PILOT_LANGUAGES
            ],
            "default_reservation": DEFAULT_TOTAL_RESERVATION,
            "default_total_reservation": DEFAULT_TOTAL_RESERVATION,
            "default_per_call_reservation": DEFAULT_PER_CALL_RESERVATION,
            # Only a configured boolean is sent; the API key itself is never
            # exposed.
            "google_translation_configured": google_configured,
            "library": library,
            "transcripts": library,
        }


def google_translation_configured(environment: CliEnvironment) -> bool:
    """Whether an explicit Basic v2 project and API key are configured.

    Only the boolean is ever exposed to the browser; the key value is never
    returned, logged or rendered.
    """

    # The fixed Basic v2 route derives its project *only* from the documented
    # ``GOOGLE_TRANSLATION_PROJECT``; no other project name is a fallback.
    project = environment.lookup("GOOGLE_TRANSLATION_PROJECT")
    return (
        project is not None
        and project.strip() != ""
        and environment.lookup("GOOGLE_TRANSLATION_API_KEY") is not None
    )


#: Visible desktop defaults for the two independent reservation values.
DEFAULT_TOTAL_RESERVATION: Final[str] = "1.00"
DEFAULT_PER_CALL_RESERVATION: Final[str] = "0.05"


def _resolve_per_call(value: str | None) -> str:
    """Return the explicit per-call reservation, or the documented default."""

    if value is None or value.strip() == "":
        return DEFAULT_PER_CALL_RESERVATION
    return value.strip()


def _augment(message: str) -> str:
    """Append one actionable hint when the mapped failure names a known code."""

    text = message or "İş başarısız."
    for marker, hint in _HINTS:
        if marker in text:
            return f"{text} {hint}"
    return text


#: Concise desktop status sentences. The CLI's multi-line summary (job id,
#: provider/model, absolute output paths) is intentionally never echoed here:
#: the status label shows one short Turkish sentence.
_DONE_MESSAGE: Final[str] = "İş tamamlandı."
_DONE_SOURCE_MESSAGE: Final[str] = "Kaynak transkript hazır."
_DONE_TRANSLATION_MESSAGE: Final[str] = "Türkçe çeviri hazır."
_DONE_BOTH_MESSAGE: Final[str] = "Kaynak transkript ve Türkçe çeviri hazır."
_DONE_VIDEO_MESSAGE: Final[str] = (
    "Kaynak transkript, Türkçe çeviri ve altyazılı video hazır."
)
#: Review is stated explicitly so the operator never mistakes it for a pass.
_REVIEW_MESSAGE: Final[str] = (
    "İş tamamlandı; otomatik kalite denetimi insan incelemesi gerektiriyor."
)
#: A partial run whose remaining failure is review-class keeps a visible review
#: signal instead of looking like an ordinary retryable error.
_REVIEW_PARTIAL_MESSAGE: Final[str] = (
    "İnceleme gerekiyor; kaynak ve Türkçe belge hazır, tamamlanamayan adım "
    "incelenmeli."
)
_PARTIAL_MESSAGE: Final[str] = (
    "Kaynak transkript hazır; Türkçe çeviri tamamlanamadı; yeniden deneyin."
)

#: Fallback message per terminal job state (used only when the result gave none).
_STATE_MESSAGE: Final[dict[str, str]] = {
    "done": _DONE_MESSAGE,
    "partial": _PARTIAL_MESSAGE,
    "error": "İş tamamlanamadı.",
}


def _ui_message(outcome: UiOutcome) -> str:
    """Map a terminal outcome to one concise Turkish sentence for the UI.

    A successful or review-required run never exposes the CLI's multi-line
    summary or internal paths; review is called out explicitly. A partial run
    keeps a short, actionable error (the mapped MT error, when present)
    and a failing run keeps its concise mapped error -- never a long success log.
    """

    state = _final_state(outcome)
    if state == "done":
        if outcome.needs_review:
            base = _REVIEW_MESSAGE
        elif outcome.video_file_name:
            base = _DONE_VIDEO_MESSAGE
        elif outcome.file_name and outcome.translation_file_name:
            base = _DONE_BOTH_MESSAGE
        elif outcome.translation_file_name:
            base = _DONE_TRANSLATION_MESSAGE
        elif outcome.file_name:
            base = _DONE_SOURCE_MESSAGE
        else:
            base = _DONE_MESSAGE
        if outcome.cleanup_warning:
            return f"{base} {outcome.cleanup_warning}"
        return base
    if state == "partial":
        detail = (outcome.video_error or outcome.translation_error or "").strip()
        if outcome.needs_review:
            return (
                f"{_REVIEW_PARTIAL_MESSAGE} {detail}".strip()
                if detail
                else _REVIEW_PARTIAL_MESSAGE
            )
        return detail or _PARTIAL_MESSAGE
    # A success result with no bound document is a closed error; never echo its
    # (success) summary. Multi-line CLI logs are never shown in the status label.
    detail = (outcome.message or "").strip()
    if outcome.exit_code == int(ExitCode.SUCCESS) or "\n" in detail:
        return _STATE_MESSAGE["error"]
    return detail or _STATE_MESSAGE["error"]


def _final_state(outcome: UiOutcome) -> str:
    """Derive the terminal state from the result and its bound documents.

    A prior document on disk is never evidence for the *current* run: only a
    document bound by this run's :class:`CommandResult` counts. A success or
    review-required result with no bound document is an error, not a false
    success.
    """

    produced = (
        outcome.file_name is not None
        or outcome.translation_file_name is not None
        or outcome.video_file_name is not None
    )
    code = outcome.exit_code
    if outcome.partial or outcome.video_error is not None:
        return "partial" if produced else "error"
    if code == int(ExitCode.SUCCESS):
        return "done" if produced else "error"
    if code == int(ExitCode.REVIEW_REQUIRED):
        return "done" if produced else "error"
    if code == int(ExitCode.INCOMPLETE):
        return "partial" if produced else "error"
    return "error"


def _read_title(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            head = handle.read(4096)
    except OSError:
        return ""
    text = head.decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()[:300]
    return ""


def _ui_dir() -> Path:
    return Path(__file__).resolve().parent / "ui"


class _UiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, state: _UiState) -> None:
        self.state = state
        super().__init__(address, handler)

    def server_bind(self) -> None:  # avoid a reverse-DNS lookup for the loopback
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


class _UiHandler(BaseHTTPRequestHandler):
    server_version = "SubtitleFlow/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> _UiState:
        return self.server.state  # type: ignore[attr-defined]

    # -- low-level helpers ------------------------------------------------ #

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Never echo request lines or client data into logs.
        return

    def _host_ok(self) -> bool:
        host = self.headers.get("Host", "")
        name = host.rsplit(":", 1)[0] if ":" in host else host
        return name in {"127.0.0.1", "localhost"}

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return True
        port = self.server.server_address[1]  # type: ignore[attr-defined]
        return origin in {
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        }

    def _send(
        self,
        status: HTTPStatus | int,
        content_type: str,
        body: bytes,
        *extra_headers: tuple[str, str],
    ) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in extra_headers:
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: HTTPStatus | int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _error(self, status: HTTPStatus, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    # -- request routing -------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_ok() or not self._origin_ok():
            self._error(HTTPStatus.FORBIDDEN, "FORBIDDEN", "İstek reddedildi.")
            return
        path = self.path.split("?", 1)[0]
        if path in _ASSETS:
            self._serve_asset(path)
            return
        if path == "/api/bootstrap":
            self._send_json(HTTPStatus.OK, self.state.bootstrap())
            return
        if path in {"/api/library", "/api/transcripts"}:
            self._send_json(
                HTTPStatus.OK, {"library": self.state.list_library()}
            )
            return
        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :]
            if _SAFE_NAME_RE.match(job_id) is None:
                self._error(HTTPStatus.BAD_REQUEST, "JOB_ID_INVALID", "Geçersiz iş kimliği.")
                return
            job = self.state.get_job(job_id)
            if job is None:
                self._error(HTTPStatus.NOT_FOUND, "JOB_MISSING", "İş bulunamadı.")
                return
            self._send_json(HTTPStatus.OK, job.as_payload())
            return
        if path.startswith("/api/source/"):
            self._serve_document(path, prefix="/api/source/", kind="source")
            return
        if path.startswith("/api/translation/"):
            self._serve_document(
                path, prefix="/api/translation/", kind="translation"
            )
            return
        if path.startswith("/api/transcripts/"):
            # Legacy alias for the source document.
            self._serve_document(path, prefix="/api/transcripts/", kind="source")
            return
        self._error(HTTPStatus.NOT_FOUND, "NOT_FOUND", "Bulunamadı.")

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_ok() or not self._origin_ok():
            self._error(HTTPStatus.FORBIDDEN, "FORBIDDEN", "İstek reddedildi.")
            return
        token = self.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(token, self.state.csrf_token):
            self._error(HTTPStatus.FORBIDDEN, "CSRF", "Oturum doğrulanamadı; sayfayı yenileyin.")
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/transcripts":
            self._create_job(action="fullchain")
            return
        if path == "/api/translations":
            self._create_job(action="translate")
            return
        self._error(HTTPStatus.NOT_FOUND, "NOT_FOUND", "Bulunamadı.")

    # -- handlers --------------------------------------------------------- #

    def _serve_asset(self, path: str) -> None:
        name, content_type = _ASSETS[path]
        target = _ui_dir() / name
        try:
            body = target.read_bytes()
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "ASSET_MISSING", "Arayüz dosyası bulunamadı.")
            return
        self._send(HTTPStatus.OK, content_type, body)

    def _serve_document(self, path: str, *, prefix: str, kind: str) -> None:
        rest = path[len(prefix) :]
        if rest.endswith("/download"):
            video_id = rest[: -len("/download")]
            download = True
        else:
            video_id = rest
            download = False
        if _SAFE_NAME_RE.match(video_id) is None:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "DOCUMENT_ID_INVALID",
                "Geçersiz belge kimliği.",
            )
            return
        if kind == "translation":
            content = self.state.read_translation(video_id)
        else:
            content = self.state.read_transcript(video_id)
        if content is None:
            self._error(
                HTTPStatus.NOT_FOUND,
                "DOCUMENT_MISSING",
                "Belge bulunamadı.",
            )
            return
        if download:
            body = content.encode("utf-8")
            self._send(
                HTTPStatus.OK,
                "text/markdown; charset=utf-8",
                body,
                ("Content-Disposition", f'attachment; filename="{video_id}.md"'),
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "video_id": video_id,
                "kind": kind,
                "file_name": f"{video_id}.md",
                "content": content,
            },
        )

    def _read_body(self) -> bytes | None:
        length_raw = self.headers.get("Content-Length")
        if length_raw is None:
            self.close_connection = True
            self._error(HTTPStatus.LENGTH_REQUIRED, "LENGTH_REQUIRED", "İstek gövdesi eksik.")
            return None
        try:
            length = int(length_raw)
        except ValueError:
            self.close_connection = True
            self._error(HTTPStatus.BAD_REQUEST, "LENGTH_INVALID", "Geçersiz istek gövdesi.")
            return None
        if length < 0 or length > _MAX_BODY_BYTES:
            self.close_connection = True
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "BODY_TOO_LARGE",
                "İstek gövdesi çok büyük.",
            )
            return None
        try:
            return self.rfile.read(length)
        except OSError:
            self.close_connection = True
            return None

    def _create_job(self, *, action: str) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/json"):
            self.close_connection = True
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "CONTENT_TYPE",
                "Yalnız JSON kabul edilir.",
            )
            return
        raw = self._read_body()
        if raw is None:
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError):
            self._error(HTTPStatus.BAD_REQUEST, "BAD_JSON", "İstek gövdesi okunamadı.")
            return
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "BAD_JSON", "İstek gövdesi bir nesne olmalı.")
            return
        try:
            request = _parse_request(payload, action=action)
        except _RequestError as exc:
            self._error(HTTPStatus.BAD_REQUEST, exc.code, exc.message)
            return
        if not google_translation_configured(self.state.environment):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "MT_UNCONFIGURED",
                "Google Translation LLM yapılandırılmamış: .env içinde "
                "GOOGLE_TRANSLATION_PROJECT ve GOOGLE_TRANSLATION_API_KEY "
                "tanımlanmalıdır.",
            )
            return
        job = self.state.start(request)
        if job is None:
            self._error(
                HTTPStatus(409),
                "BUSY",
                "Şu anda bir iş sürüyor; bitmesini bekleyip yeniden deneyin.",
            )
            return
        self._send_json(
            HTTPStatus.ACCEPTED,
            {"job_id": job.job_id, "video_id": job.video_id, "state": job.state},
        )


class _RequestError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def _parse_paid_consent(
    payload: dict[str, Any],
) -> tuple[bool, str | None, str | None]:
    """Parse the paid consent plus the two independent reservation values.

    Returns ``(allow_paid, total_reservation, per_call_reservation)``. The total
    defaults to ``1.00`` and the per-call upper bound to ``0.05``; both must be
    positive and the per-call value must not exceed the total, otherwise the
    request is refused before any job, media, STT or MT call.
    """

    allow_paid = payload.get("allow_paid", False)
    if not isinstance(allow_paid, bool):
        raise _RequestError("PAID_INVALID", "Geçersiz ücret onayı.")
    if not allow_paid:
        return False, None, None
    raw_total = payload.get("reservation", DEFAULT_TOTAL_RESERVATION)
    raw_per_call = payload.get("per_call_reservation", DEFAULT_PER_CALL_RESERVATION)
    if not isinstance(raw_total, str) or not isinstance(raw_per_call, str):
        raise _RequestError("RESERVATION_INVALID", "Geçersiz rezervasyon sınırı.")
    try:
        total = parse_strict_decimal(raw_total, name="toplam rezervasyon")
        per_call = parse_strict_decimal(raw_per_call, name="çağrı başı rezervasyon")
    except ConfigError as exc:
        raise _RequestError("RESERVATION_INVALID", "Geçersiz rezervasyon sınırı.") from exc
    if total <= 0 or per_call <= 0:
        raise _RequestError(
            "RESERVATION_INVALID", "Rezervasyon sınırları sıfırdan büyük olmalı."
        )
    if per_call > total:
        raise _RequestError(
            "RESERVATION_INVALID",
            "Çağrı başı üst sınır toplam rezervasyon sınırından büyük olamaz.",
        )
    return True, raw_total.strip(), raw_per_call.strip()


def _parse_request(payload: dict[str, Any], *, action: str = "fullchain") -> UiRequest:
    if action not in {"fullchain", "translate"}:
        raise _RequestError("ACTION_INVALID", "Geçersiz işlem.")

    if action == "translate":
        if payload.get("burn_video"):
            raise _RequestError(
                "BURN_VIDEO_UNSUPPORTED",
                "Altyazılı video yalnız tam zincir çalıştırmasında üretilir.",
            )
        raw_id = payload.get("video_id")
        if not isinstance(raw_id, str) or _SAFE_NAME_RE.match(raw_id.strip()) is None:
            raise _RequestError(
                "VIDEO_ID_INVALID", "Çeviri için geçerli bir video kimliği gerekli."
            )
        source_language = payload.get("source_language")
        if source_language in (None, ""):
            source_language = None
        elif not isinstance(source_language, str) or source_language not in {
            code for code, _name in _PILOT_LANGUAGES
        }:
            raise _RequestError("LANGUAGE_INVALID", "Geçersiz kaynak dil.")
        # A URL is optional for a translate-only request; the archived source is
        # located by video id, never by downloading anything.
        allow_paid, reservation, per_call = _parse_paid_consent(payload)
        return UiRequest(
            url="",
            video_id=raw_id.strip(),
            allow_paid=allow_paid,
            reservation=reservation,
            source_language=source_language,
            per_call_reservation=per_call,
            action="translate",
        )

    raw_url = payload.get("url")
    if not isinstance(raw_url, str) or raw_url.strip() == "":
        raise _RequestError("URL_REQUIRED", "Bir video bağlantısı girin.")
    if len(raw_url) > 2048:
        raise _RequestError("URL_TOO_LONG", "Bağlantı çok uzun.")
    try:
        reference = canonicalize_youtube_url(raw_url.strip())
    except YouTubeMediaError as exc:
        raise _RequestError(exc.code, exc.message) from exc

    allow_paid, reservation, per_call = _parse_paid_consent(payload)

    source_language = payload.get("source_language")
    if source_language in (None, ""):
        source_language = None
    elif not isinstance(source_language, str) or source_language not in {
        code for code, _name in _PILOT_LANGUAGES
    }:
        raise _RequestError("LANGUAGE_INVALID", "Geçersiz kaynak dil.")

    burn_video = payload.get("burn_video", False)
    if not isinstance(burn_video, bool):
        raise _RequestError("BURN_VIDEO_INVALID", "Geçersiz altyazılı video seçimi.")

    return UiRequest(
        url=reference.canonical_url,
        video_id=reference.video_id,
        allow_paid=bool(allow_paid),
        reservation=reservation,
        source_language=source_language,
        per_call_reservation=per_call,
        action="fullchain",
        burn_video=bool(burn_video),
    )


def _bind(host: str, port: int, state: _UiState) -> _UiServer:
    try:
        return _UiServer((host, port), _UiHandler, state)
    except OSError:
        if port == 0:
            raise
        # The documented default is busy (a stale instance); fall back to an
        # ephemeral loopback port rather than failing the launcher.
        return _UiServer((host, 0), _UiHandler, state)


def create_server(
    *,
    environment: CliEnvironment,
    transcript_dir: str | Path | None = None,
    translation_dir: str | Path | None = None,
    video_dir: str | Path | None = None,
    job_root: str | None = None,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
) -> _UiServer:
    """Create (but do not start) the loopback UI server.

    ``transcript_dir`` defaults to ``outputs/transkriptler/`` under the current
    working directory, ``translation_dir`` to its sibling ``outputs/ceviriler/``
    (or the sibling ``ceviriler`` of a custom source directory) and ``video_dir``
    to the sibling ``outputs/videolar/``. Only ``127.0.0.1`` is ever bound.
    """

    if host != "127.0.0.1":
        raise ConfigError("UI_HOST_INVALID", "the local UI only binds 127.0.0.1")
    source = (
        Path(transcript_dir).expanduser()
        if transcript_dir is not None
        else default_source_dir()
    )
    translation = (
        Path(translation_dir).expanduser()
        if translation_dir is not None
        else translation_dir_for_source(source)
    )
    video = (
        Path(video_dir).expanduser()
        if video_dir is not None
        else video_dir_for_source(source)
    )
    state = _UiState(
        environment=environment,
        source_dir=source,
        translation_dir=translation,
        video_dir=video,
        job_root=job_root,
    )
    return _bind(host, port, state)


def serve(
    *,
    environment: CliEnvironment,
    transcript_dir: str | Path | None = None,
    translation_dir: str | Path | None = None,
    video_dir: str | Path | None = None,
    job_root: str | None = None,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
) -> int:
    """Run the loopback UI server until interrupted."""

    server = create_server(
        environment=environment,
        transcript_dir=transcript_dir,
        translation_dir=translation_dir,
        video_dir=video_dir,
        job_root=job_root,
        port=port,
    )
    host, bound_port = server.server_address[0], server.server_address[1]
    url = f"http://{host}:{bound_port}/"
    print(f"Yerel arayüz: {url}", flush=True)
    print("Durdurmak için: Ctrl+C", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a headless host must still serve
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
