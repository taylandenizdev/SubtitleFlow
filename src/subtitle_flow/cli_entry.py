"""Lazy console entry point for ``subtitle-flow``.

The core :mod:`subtitle_flow` package never imports ``typer``; this tiny
wrapper keeps the ``subtitle-flow`` script importable on a core-only
installation and turns a missing ``[cli]`` extra into one concise,
traceback-free message instead of a raw ``ModuleNotFoundError``.
"""

from __future__ import annotations

import sys
from typing import Final

#: Top-level modules that only the optional ``[cli]`` extra provides.
_CLI_MODULES: Final[frozenset[str]] = frozenset(
    {"typer", "click", "rich", "shellingham", "pygments", "dotenv", "python_dotenv"}
)

_CLI_GUIDANCE: Final[str] = (
    "subtitle-flow: the optional CLI dependencies are missing. "
    "Install them with: pip install 'subtitle-flow[cli]'"
)


def main() -> None:
    """Run the Typer application, or explain a missing ``[cli]`` extra."""

    try:
        from subtitle_flow.cli import main as cli_main
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing in _CLI_MODULES:
            sys.stderr.write(_CLI_GUIDANCE + "\n")
            raise SystemExit(2) from None
        raise
    cli_main()


if __name__ == "__main__":  # pragma: no cover - module execution smoke
    main()
