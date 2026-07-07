from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraphLinkPaths:
    data_root: Path
    spider_root: Path | None = None
    bird_root: Path | None = None
    spider2lite_root: Path | None = None
    examples_lite: Path | None = None
    database_graphs_dir: Path | None = None
    unified_schema_linking_dir: Path | None = None
    output_root: Path | None = None


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("PyYAML is required for YAML configs. Install pyyaml or use JSON config.") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_paths(path: str | Path) -> GraphLinkPaths:
    raw = _load_yaml_or_json(Path(path))
    paths = raw.get("paths", raw)
    if not isinstance(paths, dict):
        raise ValueError("Expected a top-level 'paths' mapping or a flat path mapping")

    def maybe(key: str) -> Path | None:
        value = paths.get(key)
        return Path(str(value)).expanduser() if value else None

    data_root = maybe("data_root") or Path("data")
    return GraphLinkPaths(
        data_root=data_root,
        spider_root=maybe("spider_root"),
        bird_root=maybe("bird_root"),
        spider2lite_root=maybe("spider2lite_root"),
        examples_lite=maybe("examples_lite"),
        database_graphs_dir=maybe("database_graphs_dir"),
        unified_schema_linking_dir=maybe("unified_schema_linking_dir"),
        output_root=maybe("output_root"),
    )


def validate_paths(paths: GraphLinkPaths, require_existing: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in paths.__dict__.items():
        if value is None:
            summary[key] = {"configured": False, "exists": False, "path": None}
            continue
        exists = value.exists()
        if require_existing and not exists:
            raise FileNotFoundError(f"Configured path does not exist: {key}={value}")
        summary[key] = {"configured": True, "exists": exists, "path": str(value)}
    return summary
