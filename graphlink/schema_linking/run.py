from __future__ import annotations

import runpy
import sys
from pathlib import Path

from .core import ensure_legacy_import_path


def main() -> None:
    """Delegate to the renamed GraphLink 0201 core CLI.

    The CLI accepts the same arguments as the original 0201 runner, but the public
    entrypoint is now `python -m graphlink.schema_linking.run`.
    """
    ensure_legacy_import_path()
    core_path = Path(__file__).resolve().with_name("_graphlink_core_0201.py")
    sys.argv[0] = "graphlink.schema_linking.run"
    runpy.run_path(str(core_path), run_name="__main__")


if __name__ == "__main__":
    main()
