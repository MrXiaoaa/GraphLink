from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def positive_answer(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"y", "yes", "true", "1"}


def unique_keep_order(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def normalize_items(items: list[Any]) -> dict[str, Any]:
    tables: list[str] = []
    columns: dict[str, list[str]] = {}
    selected: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not positive_answer(item.get("answer")):
            continue
        table = str(item.get("table name") or item.get("table_name") or "").strip()
        if not table:
            continue
        tables.append(table)
        if isinstance(item.get("columns"), list):
            cols = [str(col).strip() for col in item["columns"] if str(col).strip()]
            if cols:
                columns[table] = cols
        selected.append(item)
    return {
        "predicted_tables": unique_keep_order(tables),
        "predicted_columns": columns,
        "selected_linking_items": selected,
        "source_format": "graphlink_answer_list",
    }


def read_linking(path: str | Path) -> dict[str, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        out: dict[str, dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, list):
                out[str(key)] = normalize_items(value)
            elif isinstance(value, dict):
                out[str(key)] = value
        return out
    if isinstance(data, list):
        return {str(row["instance_id"]): row for row in data if isinstance(row, dict) and row.get("instance_id")}
    raise ValueError(f"Unsupported linking file format: {path}")


def lookup_instance(linking: dict[str, dict[str, Any]], instance_id: str) -> dict[str, Any]:
    keys = [instance_id]
    if instance_id.startswith(("sf_bq", "sf_ga")):
        keys.append(instance_id[len("sf_"):])
    for key in keys:
        if key in linking:
            return linking[key]
    return {}


def selected_tables_for_instance(linking: dict[str, dict[str, Any]], instance_id: str) -> list[str]:
    row = lookup_instance(linking, instance_id)
    return unique_keep_order(row.get("predicted_tables") or [])
