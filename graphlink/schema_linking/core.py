from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_CORE_MODULE = "graphlink.schema_linking._graphlink_core_0201"


def ensure_legacy_import_path() -> None:
    """Expose the renamed 0201 compatibility files for legacy absolute imports."""
    compat_dir = Path(__file__).resolve().parent
    text = str(compat_dir)
    if text not in sys.path:
        sys.path.insert(0, text)


def load_core() -> ModuleType:
    ensure_legacy_import_path()
    return importlib.import_module(_CORE_MODULE)


def compute_metrics_sl(linked_json_path: str, db_path: str) -> None:
    load_core().compute_metrics_sl(linked_json_path, db_path)


def __getattr__(name: str):
    return getattr(load_core(), name)
