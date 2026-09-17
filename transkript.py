#!/usr/bin/env python3
"""Transkript masaüstü uygulaması başlatıcısı.

Bu dosya ``python transkript.py`` ile (belgelenen bağımlılıklar kurulduktan
sonra) çalışır ve arayüzü kendi işletim sistemi penceresinde açar. Hangi dizinden
çağrılırsa çağrılsın depo köküne sabitlenir; ``.env`` yalnız depo kökünde
aranır (kopyalanmaz, yazdırılmaz).

Bu betik kendiliğinden hiçbir şey kurmaz ve başlangıçta sağlayıcı çağırmaz,
medya indirmez veya model yüklemez.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Missing-dependency guidance shown instead of a raw traceback. The project is
#: not published, so the ``pip`` form installs it from the repository root.
_GUIDANCE = (
    "Bağımlılıklar eksik. Depo kökünden kurun: uv sync --extra api --extra cli "
    "--extra desktop veya pip install -e '.[api,cli,desktop]'."
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_env_file(repo_root: Path) -> Path | None:
    """Return the repository's own ``.env`` when present, otherwise ``None``."""

    local = repo_root / ".env"
    if local.is_file():
        return local
    return None


def _launch(env_path: Path | None) -> int:
    from subtitle_flow import desktop
    from subtitle_flow.cli_config import ConfigError, load_environment

    try:
        environment = load_environment(
            str(env_path) if env_path is not None else None,
            explicit=env_path is not None,
            use_default=True,
        )
    except ConfigError as exc:
        sys.stderr.write(f"yapılandırma hatası: {exc.message}\n")
        return 2

    try:
        return desktop.run_desktop(environment=environment)
    except desktop.DesktopUnavailableError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except desktop.DesktopStartupError as exc:
        sys.stderr.write(f"masaüstü penceresi başlatılamadı: {exc}\n")
        return 2
    except ConfigError as exc:
        sys.stderr.write(f"yapılandırma hatası: {exc.message}\n")
        return 2


def main(argv: list[str] | None = None) -> int:
    del argv  # reserved: kept so the launcher stays callable as ``main([...])``
    repo_root = _repo_root()
    source = repo_root / "src"
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(repo_root)
    env_path = _resolve_env_file(repo_root)
    try:
        return _launch(env_path)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        sys.stderr.write(f"{_GUIDANCE} (eksik modül: {missing})\n")
        return 2


if __name__ == "__main__":  # pragma: no cover - process entry smoke
    raise SystemExit(main())
