from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .format import unique_keep_order


def _table_key(value: Any) -> str:
    return str(value).strip().strip('`"\'').lower().split(".")[-1]


def _selected_keys(selected_tables: list[str]) -> set[str]:
    keys: set[str] = set()
    for table in selected_tables:
        raw = str(table).strip().strip('`"\'').lower()
        if raw:
            keys.add(raw)
            keys.add(raw.split(".")[-1])
    return keys


def graph_summary_candidates(graph_dir: str | Path, instance_id: str) -> list[Path]:
    root = Path(graph_dir)
    candidates = sorted(root.glob(f"{instance_id}*graph_summary.json"))
    if not candidates and instance_id.startswith("sf_"):
        candidates = sorted(root.glob(f"{instance_id[len('sf_'):]}*graph_summary.json"))
    return candidates


def collect_hint_items(graph_dir: str | Path, instance_id: str, selected_tables: list[str]) -> list[dict[str, Any]]:
    selected = _selected_keys(selected_tables)
    items: list[dict[str, Any]] = []
    for path in graph_summary_candidates(graph_dir, instance_id):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for edge in summary.get("edges", []):
            if not isinstance(edge, list) or len(edge) < 3 or not isinstance(edge[2], dict):
                continue
            left, right, meta = str(edge[0]), str(edge[1]), edge[2]
            if _table_key(left) not in selected and _table_key(right) not in selected:
                continue
            score = float(meta.get("weight") or meta.get("score") or 0.0)
            reason = meta.get("reason") or meta.get("edge_type") or meta.get("type") or "graph_relation"
            items.append({"left": left, "right": right, "reason": str(reason), "score": score, "meta": meta})
    items.sort(key=lambda item: item["score"], reverse=True)
    return items


def render_dependency_hints(items: list[dict[str, Any]], limit: int = 32) -> str:
    lines = ["GraphLink table dependency hints:"]
    seen: set[str] = set()
    for item in items:
        left, right = item["left"], item["right"]
        reason = item.get("reason", "graph_relation")
        text = f"- {left} -> {right} (reason={reason})"
        if text in seen:
            continue
        seen.add(text)
        lines.append(text)
        if len(lines) > limit:
            break
    return "\n".join(lines) if len(lines) > 1 else ""


def dependency_hints_for_instance(graph_dir: str | Path, instance_id: str, selected_tables: list[str], limit: int = 32) -> str:
    return render_dependency_hints(collect_hint_items(graph_dir, instance_id, selected_tables), limit=limit)
