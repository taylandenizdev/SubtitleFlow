"""Native desktop window for the loopback YouTube transcript UI.

The window is a thin *host* for the exact same loopback UI the browser route
already serves: :func:`subtitle_flow.web_ui.create_server` is started on an
internal background thread and the OS-native webview (macOS ``WKWebView`` via
Cocoa, Windows ``WebView2`` via Edge Chromium) loads its ``127.0.0.1`` URL.
Nothing is re-implemented here -- no provider is called, no media is downloaded
and no model is loaded at start-up.

Hardening and lifecycle notes
-----------------------------
* The listener is always bound to ephemeral ``127.0.0.1:0`` and is owned by this
  process; no pre-existing ``8765`` server is ever reused.
* The original CSRF/origin/Host gates of :mod:`subtitle_flow.web_ui` stay in
  force because the webview loads the real loopback origin.
* Closing is vetoed while a job is active through the atomic gate in
  :meth:`subtitle_flow.web_ui._UiState.begin_close`; the operator is told to
  wait instead of silently dropping a call whose outcome is unknown.
* The internal server thread is always shut down on normal close, start-up
  failure and ``Ctrl+C``; a start-up failure never leaves a listener behind.
* ``Ctrl+C`` follows the same active-job rule as the window close gate: new
  jobs are refused and the running job is awaited, never silently cancelled.
* On Windows the Edge Chromium (WebView2) engine is requested explicitly; a
  missing runtime becomes an actionable :class:`DesktopStartupError` rather
  than a silent fallback to the legacy MSHTML engine.
* The optional ``pywebview`` dependency is imported lazily so the core package
  stays importable without the ``desktop`` extra.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from subtitle_flow import web_ui

__all__ = [
    "DesktopApi",
    "DesktopStartupError",
    "DesktopUnavailableError",
    "WINDOW_HEIGHT",
    "WINDOW_MIN_SIZE",
    "WINDOW_TITLE",
    "WINDOW_WIDTH",
    "run_desktop",
]

#: Normal, resizable desktop window with a sensible minimum size.
WINDOW_TITLE: Final[str] = "Transkript"
WINDOW_WIDTH: Final[int] = 960
WINDOW_HEIGHT: Final[int] = 840
WINDOW_MIN_SIZE: Final[tuple[int, int]] = (520, 600)

#: Injected into the loaded page when a close is vetoed. The page defines the
#: hook; a missing hook is a no-op (the close is still vetoed).
_BUSY_CLOSE_SCRIPT: Final[str] = (
    "window.transkriptDesktop && window.transkriptDesktop.busyClose "
    "&& window.transkriptDesktop.busyClose();"
)

_UNAVAILABLE_MESSAGE: Final[str] = (
    "Masaüstü penceresi için isteğe bağlı 'desktop' (pywebview) bağımlılığı "
    "kurulu değil. Depo kökünden kurun: uv sync --extra api --extra cli "
    "--extra desktop (veya pip install -e '.[api,cli,desktop]')."
)

#: Shown when the native window itself cannot be created (e.g. no display).
_STARTUP_FAILURE_MESSAGE: Final[str] = (
    "masaüstü penceresi oluşturulamadı; grafik ortamı kullanılabilir değil."
)

#: Actionable Windows hint: pywebview must use the Edge Chromium WebView2
#: engine (the legacy MSHTML engine has no ``fetch``, which the UI needs).
_WEBVIEW2_MESSAGE: Final[str] = (
    "masaüstü penceresi başlatılamadı; Windows'ta Microsoft Edge WebView2 "
    "Runtime gerekir. Kurulum: "
    "https://developer.microsoft.com/microsoft-edge/webview2/ "
    "(eski MSHTML motoru bu arayüzü çalıştıramaz)."
)


def _startup_message(platform: str) -> str:
    """Actionable start-up failure text without leaking raw configuration."""

    if platform == "win32":
        return _WEBVIEW2_MESSAGE
    return (
        "masaüstü penceresi başlatılamadı; grafik ortamının (macOS WKWebView) "
        "kurulu ve erişilebilir olduğundan emin olun."
    )


class DesktopUnavailableError(RuntimeError):
    """The optional native webview dependency is not installed."""


class DesktopStartupError(RuntimeError):
    """The native window could not be created or started."""


def _import_webview() -> Any:
    """Import ``pywebview`` lazily and turn its absence into a clear message."""

    try:
        import webview  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules
        raise DesktopUnavailableError(_UNAVAILABLE_MESSAGE) from exc
    return webview


def _open_directory(directory: Path) -> None:
    """Open a directory in the OS file manager (no shell, fixed argument)."""

    target = str(directory)
    if sys.platform == "darwin":
        subprocess.run(["open", target], check=False)
    elif sys.platform == "win32":  # pragma: no cover - Windows only
        os.startfile(target)  # type: ignore[attr-defined]
    else:  # pragma: no cover - not a supported desktop target
        subprocess.run(["xdg-open", target], check=False)


class DesktopApi:
    """Narrow JS bridge: save one allowlisted transcript, open its folder.

    Only a validated ``<video_id>.md`` under the configured transcript directory
    can be saved, and the destination is always chosen by the operator through
    the native save dialog. There is no arbitrary path read/write exposed.
    """

    def __init__(
        self,
        *,
        state: web_ui._UiState,
        open_path: Callable[[Path], None] = _open_directory,
    ) -> None:
        self._state = state
        self._open_path = open_path
        self._window: Any | None = None
        self._dialog_save: Any | None = None

    def _bind_window(self, window: Any, dialog_save: Any) -> None:
        """Attach the created native window and its save-dialog constant.

        Kept private so the JS bridge only exposes the two narrow actions.
        """

        self._window = window
        self._dialog_save = dialog_save

    #: The only two document kinds the bridge can ever touch. Any other value is
    #: rejected, so the bridge never becomes a generic filesystem action.
    _KINDS: Final[tuple[str, ...]] = ("source", "translation")

    #: Default file-name suffixes so the source and Turkish documents can never
    #: be confused in a save dialog (and a same-folder save cannot silently
    #: replace the other kind).
    _KIND_SUFFIX: Final[dict[str, str]] = {"source": "kaynak", "translation": "turkce"}

    def _document_path(self, video_id: Any, kind: Any) -> Path | None:
        if kind is None:
            kind = "source"
        if not isinstance(kind, str) or kind not in self._KINDS:
            return None
        if not isinstance(video_id, str):
            return None
        if kind == "translation":
            return self._state.translation_path(video_id)
        return self._state.transcript_path(video_id)

    #: Folder kinds the narrow bridge can reveal: the two Markdown document
    #: directories plus the burned-in subtitle video directory. No other path is
    #: reachable, and no document kind is implied by the folder list.
    _FOLDER_KINDS: Final[tuple[str, ...]] = ("source", "translation", "video")

    def _folder(self, kind: Any) -> Path | None:
        if kind is None:
            kind = "source"
        if not isinstance(kind, str) or kind not in self._FOLDER_KINDS:
            return None
        if kind == "translation":
            return Path(self._state.translation_dir)
        if kind == "video":
            return Path(self._state.video_dir)
        return Path(self._state.transcript_dir)

    def save_transcript(self, video_id: Any, kind: Any = "source") -> dict[str, Any]:
        """Save one validated document (``source`` or ``translation``).

        The default file name encodes the document kind, and a pre-existing
        destination with *different* content is refused instead of being replaced,
        so saving the Turkish document can never silently erase the source one.
        """

        resolved_kind = "source" if kind is None else kind
        if not isinstance(resolved_kind, str) or resolved_kind not in self._KINDS:
            return {"ok": False, "message": "Geçersiz belge türü."}
        path = self._document_path(video_id, resolved_kind)
        if path is None or not path.is_file():
            return {"ok": False, "message": "Belge bulunamadı."}
        if self._window is None:
            return {"ok": False, "message": "Pencere hazır değil."}
        try:
            selected: Sequence[str] | None = self._window.create_file_dialog(
                self._dialog_save,
                directory=str(path.parent),
                save_filename=f"{video_id}-{self._KIND_SUFFIX[resolved_kind]}.md",
                file_types=("Markdown (*.md)",),
            )
        except Exception:  # noqa: BLE001 - a dialog failure must not crash the app
            return {"ok": False, "message": "Kaydetme penceresi açılamadı."}
        if not selected:
            return {"ok": False, "cancelled": True}
        target = Path(selected[0] if isinstance(selected, (list, tuple)) else selected)
        try:
            source_bytes = path.read_bytes()
        except OSError:
            return {"ok": False, "message": "Belge okunamadı."}
        if target.exists():
            try:
                existing = target.read_bytes()
            except OSError:
                return {"ok": False, "message": "Hedef dosya okunamadı."}
            if existing == source_bytes:
                return {"ok": True, "path": str(target)}
            return {
                "ok": False,
                "message": "Bu konumda farklı içerikli bir dosya var; üzerine "
                "yazılmadı. Başka bir ad veya klasör seçin.",
            }
        try:
            target.write_bytes(source_bytes)
        except OSError:
            return {"ok": False, "message": "Dosya kaydedilemedi."}
        return {"ok": True, "path": str(target)}

    def open_transcripts_folder(self, kind: Any = "source") -> dict[str, Any]:
        """Open one fixed output directory (``source``/``translation``/``video``)."""

        directory = self._folder(kind)
        if directory is None:
            return {"ok": False, "message": "Geçersiz klasör türü."}
        if not directory.is_dir():
            return {"ok": False, "message": "Kayıt klasörü bulunamadı."}
        try:
            self._open_path(directory)
        except Exception:  # noqa: BLE001 - opening a folder must not crash the app
            return {"ok": False, "message": "Klasör açılamadı."}
        return {"ok": True}


def _notify_busy_close(window: Any) -> None:
    """Tell the page about a vetoed close without blocking the GUI thread.

    The closing handler runs synchronously on the GUI thread; ``evaluate_js``
    blocks until the GUI thread services it, so the notification is dispatched
    from a short-lived background thread instead of deadlocking the close.
    """

    def notify() -> None:
        try:
            window.evaluate_js(_BUSY_CLOSE_SCRIPT)
        except Exception:  # noqa: BLE001 - guidance is best effort only
            pass

    threading.Thread(
        target=notify, name="subtitle-flow-desktop-notify", daemon=True
    ).start()


def _attach_close_gate(window: Any, state: web_ui._UiState) -> Callable[..., bool]:
    """Wire the window's ``closing`` event to the atomic state gate."""

    def on_closing(*_args: Any) -> bool:
        if not state.begin_close():
            _notify_busy_close(window)
            return False
        return True

    window.events.closing += on_closing
    return on_closing


def _stop_server(server: Any, thread: threading.Thread) -> None:
    """Stop the internally owned listener and join its thread."""

    try:
        server.shutdown()
    except Exception:  # noqa: BLE001 - shutdown must never mask the outcome
        pass
    try:
        server.server_close()
    except Exception:  # noqa: BLE001
        pass
    thread.join(timeout=5)


def _start_gui(webview: Any, *, platform: str) -> None:
    """Start the native event loop, forcing Edge Chromium on Windows.

    Pinning ``gui="edgechromium"`` prevents a silent fallback to the legacy
    MSHTML engine, which cannot run the loopback UI's ``fetch`` calls; a missing
    WebView2 runtime then fails actionably instead of degrading quietly.
    """

    if platform == "win32":
        webview.start(gui="edgechromium")
    else:
        webview.start()


def _interrupt_gracefully(window: Any, state: web_ui._UiState) -> None:
    """Close the window on ``Ctrl+C`` without ever dropping an active job.

    Mirrors the window close gate: new jobs are refused at once, and a running
    job is allowed to reach its own final state (the page is asked to wait)
    before the window is destroyed. The job is never cancelled or retried.
    """

    state.mark_closing()
    if state.busy:
        _notify_busy_close(window)
        state.wait_until_idle()
    try:
        window.destroy()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def run_desktop(
    *,
    environment: Any,
    transcript_dir: str | Path | None = None,
    translation_dir: str | Path | None = None,
    video_dir: str | Path | None = None,
    job_root: str | None = None,
    window_title: str = WINDOW_TITLE,
    webview_module: Any | None = None,
    platform: str | None = None,
) -> int:
    """Run the native transcript window until it is closed.

    ``environment`` is a resolved :class:`~subtitle_flow.cli_config.CliEnvironment`;
    the loopback state is built lazily per request by the existing UI, so no
    provider call or model load happens here. The listener is always ephemeral
    (``127.0.0.1:0``) and owned by this process. ``platform`` defaults to
    :data:`sys.platform` and is injectable so the Windows engine gate is
    testable without a Windows host.
    """

    webview = webview_module if webview_module is not None else _import_webview()
    resolved_platform = sys.platform if platform is None else platform

    server = web_ui.create_server(
        environment=environment,
        transcript_dir=transcript_dir,
        translation_dir=translation_dir,
        video_dir=video_dir,
        job_root=job_root,
        port=0,
    )
    host, port = server.server_address[0], server.server_address[1]
    url = f"http://{host}:{port}/"
    thread = threading.Thread(
        target=server.serve_forever, name="subtitle-flow-desktop-server", daemon=True
    )
    thread.start()
    try:
        api = DesktopApi(state=server.state)
        try:
            window = webview.create_window(
                title=window_title,
                url=url,
                js_api=api,
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                min_size=WINDOW_MIN_SIZE,
                resizable=True,
                confirm_close=False,
                background_color="#f5f7fa",
                text_select=True,
            )
            if window is None:
                raise DesktopStartupError(_STARTUP_FAILURE_MESSAGE)
            api._bind_window(window, webview.FileDialog.SAVE)
            _attach_close_gate(window, server.state)
        except DesktopStartupError:
            raise
        except Exception as exc:  # noqa: BLE001 - no traceback/raw config leaks
            raise DesktopStartupError(_startup_message(resolved_platform)) from exc
        try:
            _start_gui(webview, platform=resolved_platform)
        except KeyboardInterrupt:
            _interrupt_gracefully(window, server.state)
        except Exception as exc:  # noqa: BLE001 - e.g. missing WebView2 runtime
            raise DesktopStartupError(_startup_message(resolved_platform)) from exc
        return 0
    finally:
        _stop_server(server, thread)
