#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

from .reproduce_sql_generation import (
    AUTOLINK_RUN,
    DATASETS,
    REPO_ROOT,
    SCHEMA_LINKING_DIR,
    append_jsonl,
    db_display_id,
    format_schema,
    load_json,
    load_table_schema,
    make_tasks,
    sanitize_table_name,
    sqlite_foreign_keys,
)


NATIVE_API_METHODS = ["AutoLink", "CHESS", "KaSLA"]
DATASETS["spider2lite"] = {
    "query_file": AUTOLINK_RUN / "data" / "spider2lite" / "spider2_data.json",
    "schema_root": AUTOLINK_RUN / "resource" / "databases",
}
CHESS_TEMPLATE = (
    REPO_ROOT
    / "methods"
    / "baselines"
    / "CHESS"
    / "templates"
    / "template_generate_candidate_one.txt"
)
GRAPH_SUMMARY_DIRS = {
    "spider": REPO_ROOT / "datasets" / "database_graphs_spider",
    "bird": REPO_ROOT / "datasets" / "database_graphs_bird",
}
SPIDER2LITE_EXPANSION_PATH_DIR = REPO_ROOT / "data" / "expansion_paths"
SPIDER2LITE_GRAPH_SUMMARY_DIR = Path(os.environ.get("GRAPHLINK_SPIDER2_GRAPH_SUMMARY_DIR", str(REPO_ROOT / "data" / "database_graphs_0827")))
GRAPH_SUMMARY_CACHE: dict[str, Any] = {}


def load_autolink_config() -> Any:
    spec = importlib.util.spec_from_file_location("autolink_config", AUTOLINK_RUN / "config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load AutoLink config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_dialect_prompts() -> Any:
    spec = importlib.util.spec_from_file_location("dialect_prompt", Path(__file__).with_name("dialect_prompt.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load backend dialect prompt helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Prompts()


DIALECT_PROMPTS = load_dialect_prompts()


def sample_values(table_obj: dict[str, Any], col_name: str, limit: int = 2) -> list[Any]:
    values: list[Any] = []
    for row in table_obj.get("sample_rows", []):
        if isinstance(row, dict) and col_name in row and row[col_name] not in (None, ""):
            values.append(row[col_name])
        if len(values) >= limit:
            break
    return values


def selected_table_objects(schema_dir: Path, predicted_tables: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    seen: set[str] = set()
    tables: list[dict[str, Any]] = []
    missing: list[str] = []
    for table in predicted_tables:
        short = sanitize_table_name(table)
        key = short.lower()
        if not short or key in seen:
            continue
        seen.add(key)
        obj = load_table_schema(schema_dir, short)
        if obj is None:
            obj = load_table_schema_recursive(schema_dir, table)
        if obj is None:
            missing.append(short)
            continue
        tables.append(obj)
    return tables, missing


def load_table_schema_recursive(schema_dir: Path, table_name: str) -> dict[str, Any] | None:
    raw = str(table_name).strip().strip("`\"'")
    parts = [
        part.strip().strip("`\"'").strip()
        for part in raw.split(".")
        if part.strip().strip("`\"'").strip()
    ]
    if len(parts) >= 2:
        candidates = [
            schema_dir / ".".join(parts[:-1]) / f"{parts[-1]}.json",
            schema_dir / parts[-2] / f"{parts[-2]}.{parts[-1]}.json",
            schema_dir / parts[-2] / f"{parts[-1]}.json",
            schema_dir / f"{'.'.join(parts[-2:])}.json",
        ]
        for direct in candidates:
            if direct.exists():
                return load_json(direct)
    direct = schema_dir / f"{sanitize_table_name(raw)}.json"
    if direct.exists():
        return load_json(direct)

    table = sanitize_table_name(table_name)
    candidates = [table]
    if "." in str(table_name):
        candidates.append(str(table_name).split(".")[-1].strip("`\"'"))
    seen = {c.lower() for c in candidates if c}
    for path in schema_dir.rglob("*.json"):
        if path.stem.lower() in seen:
            return load_json(path)
    return None


def table_parts_for_prompt_match(value: Any) -> list[str]:
    raw = clean_identifier(value)
    return [
        part.strip().strip("`\"'").strip().lower()
        for part in raw.split(".")
        if part.strip().strip("`\"'").strip()
    ]


def shard_family_base(table_part: str) -> str | None:
    for pattern in (r"^(.+?)_\d{8}$", r"^(.+?)\d{8}$", r"^(.+?)_\d{6}$"):
        match = re.match(pattern, table_part)
        if match:
            return match.group(1)
    return None


def year_family_base(table_part: str) -> str | None:
    for pattern in (r"^(.+?)_\d{4}$", r"^(.+?)\d{4}$"):
        match = re.match(pattern, table_part)
        if match:
            return match.group(1)
    return None


def external_table_family(task: dict[str, Any] | None) -> tuple[str, str] | None:
    external = (task or {}).get("external_knowledge")
    if not external or not str(external).endswith(".md"):
        return None
    stem = Path(str(external)).name[:-3].lower()
    parts = stem.split(".")
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return None


def spider2lite_table_group_key(value: Any, task: dict[str, Any] | None = None) -> str:
    parts = table_parts_for_prompt_match(value)
    if not parts:
        return ""
    last = parts[-1]
    dataset = parts[-2] if len(parts) >= 2 else ""
    external_family = external_table_family(task)
    base = shard_family_base(last) or year_family_base(last)
    if external_family and base == external_family[1] and (not dataset or dataset == external_family[0]):
        return f"family:{external_family[0]}.{external_family[1]}"
    if base:
        return f"family:{dataset}.{base}" if dataset else f"family:{base}"
    return f"{dataset}.{last}" if dataset else last


def schema_object_for_prompt_table(schema_dir: Path, table_name: str) -> dict[str, Any] | None:
    obj = load_table_schema_recursive(schema_dir, table_name)
    if obj is not None:
        return obj
    parts = table_parts_for_prompt_match(table_name)
    if not parts:
        return None
    last = parts[-1]
    base = shard_family_base(last) or year_family_base(last)
    if not base:
        return None
    for path in schema_dir.rglob("*.json"):
        if path.stem.lower().startswith(base.lower()):
            return load_json(path)
    return None


def normalize_column_key(value: Any) -> str:
    return str(value).strip().strip("`\"'").lower()


def clean_identifier(value: Any) -> str:
    return str(value).strip().strip("`\"'").rstrip("`").strip()


def table_match_keys(value: Any) -> set[str]:
    raw = clean_identifier(value)
    if not raw:
        return set()
    compact = raw.lower()
    parts = [
        part.strip().strip("`\"'").strip()
        for part in raw.split(".")
        if part.strip().strip("`\"'").strip()
    ]
    keys = {compact, sanitize_table_name(raw).lower()}
    if parts:
        keys.add(parts[-1].lower())
    if len(parts) >= 2:
        keys.add(".".join(parts[-2:]).lower())
    return {key for key in keys if key}


def table_sequence_key(value: Any) -> str:
    keys = table_match_keys(value)
    if not keys:
        return normalize_column_key(value)
    raw = clean_identifier(value).lower()
    if raw in keys:
        return raw
    short = sanitize_table_name(value).lower()
    if short in keys:
        return short
    return sorted(keys, key=lambda key: (key.count("."), len(key), key))[0]


def unique_table_sequence(tables: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for table in tables:
        key = table_sequence_key(table)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(table)
    return unique


def spider2lite_grouped_table_sequence(
    tables: list[str],
    task: dict[str, Any] | None,
) -> list[tuple[str, list[str]]]:
    group_order: list[str] = []
    groups: dict[str, list[str]] = {}
    seen_tables_by_group: dict[str, set[str]] = {}
    for table in tables:
        group_key = spider2lite_table_group_key(table, task) or table_sequence_key(table)
        table_key = table_sequence_key(table)
        if not group_key or not table_key:
            continue
        if group_key not in groups:
            group_order.append(group_key)
            groups[group_key] = []
            seen_tables_by_group[group_key] = set()
        if table_key in seen_tables_by_group[group_key]:
            continue
        groups[group_key].append(table)
        seen_tables_by_group[group_key].add(table_key)
    return [(group_key, groups[group_key]) for group_key in group_order]


def flatten_table_groups(groups: list[tuple[str, list[str]]]) -> list[str]:
    tables: list[str] = []
    for _, group_tables in groups:
        tables.extend(group_tables)
    return tables


def selected_table_column_keys(table_name: str) -> set[str]:
    keys = {
        normalize_column_key(table_name),
        normalize_column_key(sanitize_table_name(table_name)),
        normalize_column_key(str(table_name).split(".")[-1].strip("`\"'")),
    }
    return {key for key in keys if key}


def copy_columns_for_table(predicted_columns: dict[str, Any], table_name: str) -> tuple[str, Any] | None:
    target_keys = selected_table_column_keys(table_name)
    for key, value in predicted_columns.items():
        if selected_table_column_keys(key) & target_keys:
            return key, value
    return None


def apply_table_count_limit_and_fill(
    predicted_tables: list[str],
    predicted_columns: dict[str, Any] | None,
    table_count_limit: int | None,
    fill_predicted: dict[str, Any] | None,
    table_count_limit_mode: str = "table",
    task: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any]]:
    primary_unique_tables = unique_table_sequence(predicted_tables)
    primary_groups = spider2lite_grouped_table_sequence(predicted_tables, task)
    if table_count_limit is None or table_count_limit <= 0:
        return predicted_tables, predicted_columns, {
            "table_count_limit": None,
            "table_count_limit_mode": table_count_limit_mode,
            "num_primary_unique_tables": len(primary_unique_tables),
            "num_primary_unique_table_groups": len(primary_groups),
            "num_effective_table_groups": len(primary_groups),
            "num_fill_tables_added": 0,
        }

    fill_added: list[str] = []
    selected_group_count: int | None = None
    if table_count_limit_mode == "spider2lite_group":
        selected_groups = primary_groups[:table_count_limit]
        selected_group_keys = {group_key for group_key, _ in selected_groups}
        fill_groups = spider2lite_grouped_table_sequence(
            (fill_predicted or {}).get("predicted_tables") or [],
            task,
        )
        for group_key, group_tables in fill_groups:
            if len(selected_groups) >= table_count_limit:
                break
            if not group_key or group_key in selected_group_keys:
                continue
            selected_groups.append((group_key, group_tables))
            selected_group_keys.add(group_key)
            fill_added.extend(group_tables)
        selected = flatten_table_groups(selected_groups)
        selected_group_count = len(selected_groups)
    else:
        selected = primary_unique_tables[:table_count_limit]
        selected_keys = {table_sequence_key(table) for table in selected}
        fill_tables = unique_table_sequence((fill_predicted or {}).get("predicted_tables") or [])
        for table in fill_tables:
            if len(selected) >= table_count_limit:
                break
            key = table_sequence_key(table)
            if not key or key in selected_keys:
                continue
            selected.append(table)
            selected_keys.add(key)
            fill_added.append(table)
        selected_group_count = len(spider2lite_grouped_table_sequence(selected, task))

    adjusted_columns = predicted_columns
    if predicted_columns is not None:
        adjusted_columns = {}
        for table in selected:
            existing = copy_columns_for_table(predicted_columns, table)
            if existing is not None:
                adjusted_columns[existing[0]] = existing[1]
                continue
            fill_columns = (fill_predicted or {}).get("predicted_columns")
            if isinstance(fill_columns, dict):
                filled = copy_columns_for_table(fill_columns, table)
                if filled is not None:
                    adjusted_columns[filled[0]] = filled[1]

    return selected, adjusted_columns, {
        "table_count_limit": table_count_limit,
        "table_count_limit_mode": table_count_limit_mode,
        "num_primary_unique_tables": len(primary_unique_tables),
        "num_primary_unique_table_groups": len(primary_groups),
        "num_effective_table_groups": selected_group_count,
        "num_fill_tables_added": len(fill_added),
        "fill_tables_added": fill_added,
    }


def selected_table_key_set(selected_tables: list[str]) -> set[str]:
    keys: set[str] = set()
    for table in selected_tables:
        keys.update(table_match_keys(table))
    return keys


def load_json_cached(path: Path) -> Any:
    key = str(path)
    if key not in GRAPH_SUMMARY_CACHE:
        GRAPH_SUMMARY_CACHE[key] = load_json(path)
    return GRAPH_SUMMARY_CACHE[key]


def graph_summary_paths_for_task(dataset: str, task: dict[str, Any], selected_tables: list[str]) -> list[Path]:
    instance_id = str(task["instance_id"])
    db_name = str(task["db_name"])
    if dataset in GRAPH_SUMMARY_DIRS:
        graph_dir = GRAPH_SUMMARY_DIRS[dataset]
        paths = sorted(graph_dir.glob(f"{instance_id}_*graph_summary.json"))
        if paths:
            return paths[:1]
        return sorted(graph_dir.glob(f"*{db_name}*graph_summary.json"))[:1]

    if dataset != "spider2lite":
        return []
    candidates = sorted(SPIDER2LITE_GRAPH_SUMMARY_DIR.glob(f"{instance_id}*graph_summary.json"))
    if not candidates:
        candidates = sorted(SPIDER2LITE_GRAPH_SUMMARY_DIR.glob(f"*{db_name}*graph_summary.json"))
    if not candidates:
        return []
    selected_keys = selected_table_key_set(selected_tables)
    scored: list[tuple[int, str, Path]] = []
    for path in candidates:
        try:
            summary = load_json_cached(path)
        except Exception:
            continue
        overlap = 0
        for table in summary.get("tables", []):
            if table_match_keys(table) & selected_keys:
                overlap += 1
        if overlap:
            scored.append((overlap, path.name, path))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [path for _score, _name, path in scored[:8]]


def relation_reason(data: dict[str, Any]) -> str:
    reason = str(data.get("reason") or "").replace("\n", " ").strip()
    if not reason:
        parts = []
        for key in ("ind_confidence", "desc_sim", "mh_sim", "weight"):
            if key in data:
                parts.append(f"{key}={data[key]}")
        reason = ", ".join(parts)
    return reason or "unknown"


def relation_type(data: dict[str, Any]) -> str:
    if data.get("fk_type"):
        return str(data["fk_type"])
    if data.get("fk_relation"):
        return "related"
    return "similarity"


def format_column_relation(t1: str, t2: str, data: dict[str, Any]) -> str | None:
    pair = data.get("ind_column_pair")
    if not isinstance(pair, list) or len(pair) < 2:
        return None
    direction = str(data.get("ind_direction") or "t1->t2")
    if direction == "t2->t1":
        left_table, right_table = t2, t1
    else:
        left_table, right_table = t1, t2
    return (
        f"{left_table}.{pair[0]} = {right_table}.{pair[1]} "
        f"(type={relation_type(data)}, reason={relation_reason(data)})"
    )


def format_explicit_fk_relation(t1: str, t2: str, summary: dict[str, Any], data: dict[str, Any]) -> str | None:
    for rel in summary.get("foreign_key_info", {}).get("fk_relations", []):
        fk_table = clean_identifier(rel.get("fk_table", ""))
        pk_table = clean_identifier(rel.get("pk_table", ""))
        if not fk_table or not pk_table:
            continue
        fk_matches_t1 = table_match_keys(fk_table) & table_match_keys(t1)
        pk_matches_t2 = table_match_keys(pk_table) & table_match_keys(t2)
        fk_matches_t2 = table_match_keys(fk_table) & table_match_keys(t2)
        pk_matches_t1 = table_match_keys(pk_table) & table_match_keys(t1)
        if (fk_matches_t1 and pk_matches_t2) or (fk_matches_t2 and pk_matches_t1):
            return (
                f"{fk_table}.{rel.get('fk_column')} = {pk_table}.{rel.get('pk_column')} "
                f"(type={relation_type(data)}, reason={relation_reason(data)})"
            )
    return None


def graph_edge_hint_item(t1: str, t2: str, data: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    line = format_column_relation(t1, t2, data)
    if line is None:
        line = format_explicit_fk_relation(t1, t2, summary, data)
    if line is None:
        line = f"{t1} -> {t2} (type={relation_type(data)}, reason={relation_reason(data)})"
    return {
        "line": line,
        "left_keys": table_match_keys(t1),
        "right_keys": table_match_keys(t2),
        "score": (
            1 if data.get("fk_type") == "explicit" else 0,
            1 if data.get("fk_relation") else 0,
            float(data.get("weight") or 0),
            float(data.get("ind_confidence") or 0),
            float(data.get("desc_sim") or 0),
            float(data.get("mh_sim") or 0),
        ),
    }


def collect_graph_summary_hint_items(dataset: str, task: dict[str, Any], selected_tables: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in graph_summary_paths_for_task(dataset, task, selected_tables):
        try:
            summary = load_json_cached(path)
        except Exception:
            continue
        for edge in summary.get("edges", []):
            if not isinstance(edge, list) or len(edge) < 3 or not isinstance(edge[2], dict):
                continue
            items.append(graph_edge_hint_item(str(edge[0]), str(edge[1]), edge[2], summary))
    items.sort(key=lambda item: item["score"], reverse=True)
    return items


def collect_expansion_path_hint_items(task: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    path = SPIDER2LITE_EXPANSION_PATH_DIR / f"{task['instance_id']}_expansion_paths.json"
    if not path.exists():
        return [], False
    try:
        data = load_json_cached(path)
    except Exception:
        return [], True
    items: list[dict[str, Any]] = []
    for parent_data in data.get("expansion_paths", {}).values():
        for path_data in parent_data.get("paths", []):
            for step in path_data.get("steps", []):
                left = str(step.get("from") or "")
                right = str(step.get("to") or "")
                if not left or not right:
                    continue
                edge_info = step.get("edge_info") if isinstance(step.get("edge_info"), dict) else {}
                reason = str(edge_info.get("reason") or "").strip()
                extras = []
                if reason:
                    extras.append(f"reason={reason}")
                if step.get("minhash_similarity"):
                    extras.append(f"minhash={step.get('minhash_similarity')}")
                if edge_info.get("desc_sim"):
                    extras.append(f"desc_sim={edge_info.get('desc_sim')}")
                detail = ", ".join(extras) or "reason=GraphLink expansion path"
                items.append(
                    {
                        "line": f"{left} -> {right} ({detail})",
                        "left_keys": table_match_keys(left),
                        "right_keys": table_match_keys(right),
                        "score": (0, 1, float(step.get("minhash_similarity") or 0), float(edge_info.get("desc_sim") or 0)),
                    }
                )
    items.sort(key=lambda item: item["score"], reverse=True)
    return items, True


def collect_dependency_hint_items(dataset: str, task: dict[str, Any], selected_tables: list[str]) -> list[dict[str, Any]]:
    if dataset == "spider2lite":
        path_items, has_expansion_path = collect_expansion_path_hint_items(task)
        if has_expansion_path:
            return path_items
    return collect_graph_summary_hint_items(dataset, task, selected_tables)


def render_dependency_hint_block(
    hint_items: list[dict[str, Any]],
    selected_tables: list[str],
    limit: int,
) -> str:
    if not hint_items or limit <= 0:
        return ""
    selected_keys = selected_table_key_set(selected_tables)
    lines: list[str] = []
    seen: set[str] = set()
    for item in hint_items:
        if not (item["left_keys"] & selected_keys and item["right_keys"] & selected_keys):
            continue
        line = item["line"]
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= limit:
            break
    if not lines:
        return ""
    numbered = [f"{idx}. {line.lstrip('- ').strip()}" for idx, line in enumerate(lines, start=1)]
    return (
        "Relevant join paths:\n"
        + "\n".join(numbered)
        + "\nUse these join paths when the question requires combining these tables."
    )


def append_dependency_hint_block(
    schema_text: str,
    hint_items: list[dict[str, Any]],
    selected_tables: list[str],
    limit: int,
) -> str:
    block = render_dependency_hint_block(hint_items, selected_tables, limit)
    if not block:
        return schema_text
    return schema_text + "\n\n" + block


def dependency_hint_line_count(prompt: str) -> int:
    markers = ("Relevant join paths:", "GraphLink table dependency hints:")
    positions = [prompt.find(marker) for marker in markers if prompt.find(marker) != -1]
    if not positions:
        return 0
    lines = prompt[min(positions):].splitlines()[1:]
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("Use these join paths"):
            break
        if stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped):
            count += 1
    return count


def column_filter_for_table(predicted_columns: dict[str, Any] | None, table_name: str) -> set[str] | None:
    if not predicted_columns:
        return None
    keys = {
        normalize_column_key(table_name),
        normalize_column_key(sanitize_table_name(table_name)),
        normalize_column_key(str(table_name).split(".")[-1].strip("`\"'")),
    }
    raw_cols = None
    for key, value in predicted_columns.items():
        candidate_keys = {
            normalize_column_key(key),
            normalize_column_key(sanitize_table_name(key)),
            normalize_column_key(str(key).split(".")[-1].strip("`\"'")),
        }
        if keys & candidate_keys:
            raw_cols = value
            break
    if raw_cols is None:
        return set()
    if isinstance(raw_cols, dict):
        raw_cols = list(raw_cols)
    if not isinstance(raw_cols, list):
        return set()
    allowed: set[str] = set()
    for col in raw_cols:
        norm = normalize_column_key(col)
        if norm:
            allowed.add(norm)
            allowed.add(norm.split(".")[-1])
    return allowed


def format_schema_as_create(
    schema_dir: Path,
    predicted_tables: list[str],
    predicted_columns: dict[str, Any] | None = None,
) -> str:
    tables, missing = selected_table_objects(schema_dir, predicted_tables)
    blocks: list[str] = []
    selected: set[str] = set()
    for obj in tables:
        table_name = str(obj.get("table_name") or "")
        if not table_name:
            continue
        selected.add(table_name)
        col_names = obj.get("column_names", [])
        col_types = obj.get("column_types", [])
        descriptions = obj.get("description", [])
        lines = [f"CREATE TABLE {table_name} ("]
        col_lines = []
        allowed_columns = column_filter_for_table(predicted_columns, table_name)
        for idx, col in enumerate(col_names):
            if allowed_columns is not None and normalize_column_key(col) not in allowed_columns:
                continue
            typ = col_types[idx] if idx < len(col_types) and col_types[idx] else "TEXT"
            desc = descriptions[idx] if idx < len(descriptions) else ""
            examples = sample_values(obj, col)
            comment_bits = []
            if desc:
                comment_bits.append(str(desc))
            if examples:
                comment_bits.append("examples: " + ", ".join(map(str, examples)))
            comment = " -- " + " | ".join(comment_bits) if comment_bits else ""
            col_lines.append(f"  {col} {typ}{comment}")
        lines.append(",\n".join(col_lines))
        lines.append(");")
        blocks.append("\n".join(lines))
    fk_text = sqlite_foreign_keys(schema_dir, selected)
    if fk_text:
        blocks.append("Foreign keys:\n" + fk_text)
    if missing:
        blocks.append("Predicted tables without local schema JSON: " + ", ".join(missing[:50]))
    return "\n\n".join(blocks)


def compact_text(value: Any, limit: int) -> str:
    text = str(value).replace("\n", " ").strip()
    if limit and len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def render_spider2lite_compact_schema_block(
    obj: dict[str, Any],
    requested_tables: list[str],
    predicted_columns: dict[str, Any] | None = None,
) -> str:
    requested_table = requested_tables[0] if requested_tables else ""
    table_name = str(obj.get("table_fullname") or obj.get("table_name") or requested_table).strip()
    col_names = obj.get("column_names") or []
    col_types = obj.get("column_types") or []
    descriptions = obj.get("description") or []
    allowed_columns = column_filter_for_table(predicted_columns, table_name)
    if allowed_columns is not None and not allowed_columns and requested_table:
        allowed_columns = column_filter_for_table(predicted_columns, requested_table)

    lines = [f"Table full name: {table_name}"]
    if requested_table and requested_table != table_name:
        lines.append(f"Representative schema for selected table: {requested_table}")
    if len(requested_tables) > 1:
        visible = ", ".join(map(str, requested_tables[:12]))
        suffix = f", ... ({len(requested_tables)} selected physical tables)" if len(requested_tables) > 12 else ""
        lines.append(f"Selected physical tables represented: {visible}{suffix}")

    for idx, col in enumerate(col_names):
        if allowed_columns is not None and normalize_column_key(col) not in allowed_columns:
            continue
        typ = col_types[idx] if idx < len(col_types) and col_types[idx] else "TEXT"
        desc = descriptions[idx] if idx < len(descriptions) else ""
        line = f"Column name: {col} Type: {typ}"
        if desc:
            line += f" Description: {compact_text(desc, 500)}"
        examples = sample_values(obj, col, limit=1)
        if examples:
            line += f" Example: {compact_text(examples[0], 120)}"
        lines.append(line)

    rows = obj.get("sample_rows") or []
    if rows:
        sample_text = json.dumps(rows[:2], ensure_ascii=False, default=str)
        lines.append("Sample rows:")
        lines.append(compact_text(sample_text, 2000))
    return "\n".join(lines)


def format_spider2lite_compact_schema(
    schema_dir: Path,
    predicted_tables: list[str],
    predicted_columns: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    group_order: list[str] = []
    groups: dict[str, list[str]] = {}
    for table in predicted_tables:
        group_key = spider2lite_table_group_key(table, task)
        if not group_key:
            group_key = table_sequence_key(table)
        if not group_key:
            continue
        if group_key not in groups:
            group_order.append(group_key)
            groups[group_key] = []
        groups[group_key].append(table)

    blocks: list[str] = []
    rendered_tables: list[str] = []
    missing: list[str] = []
    for group_key in group_order:
        requested_tables = groups[group_key]
        obj = schema_object_for_prompt_table(schema_dir, requested_tables[0])
        if obj is None:
            missing.append(str(requested_tables[0]))
            continue
        blocks.append(render_spider2lite_compact_schema_block(obj, requested_tables, predicted_columns))
        rendered_tables.append(str(obj.get("table_fullname") or obj.get("table_name") or requested_tables[0]))

    if missing:
        blocks.append("Predicted logical table groups without local schema JSON: " + ", ".join(missing[:50]))
    return "\n\n".join(blocks), rendered_tables


def fit_spider2lite_compact_schema_to_budget(
    schema_dir: Path,
    predicted_tables: list[str],
    schema_char_budget: int | None,
    predicted_columns: dict[str, Any] | None = None,
    schema_text_postprocessor=None,
    task: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    group_order: list[str] = []
    group_first_table: dict[str, str] = {}
    for table in predicted_tables:
        group_key = spider2lite_table_group_key(table, task)
        if not group_key:
            group_key = table_sequence_key(table)
        if not group_key or group_key in group_first_table:
            continue
        group_order.append(group_key)
        group_first_table[group_key] = table

    def render_schema(group_count: int) -> tuple[str, list[str]]:
        included_groups = set(group_order[:group_count])
        included_tables = [
            table
            for table in predicted_tables
            if (spider2lite_table_group_key(table, task) or table_sequence_key(table)) in included_groups
        ]
        text, rendered_tables = format_spider2lite_compact_schema(
            schema_dir,
            included_tables,
            predicted_columns,
            task,
        )
        if schema_text_postprocessor is not None:
            text = schema_text_postprocessor(text, predicted_tables)
        return text, rendered_tables

    if not schema_char_budget:
        return render_schema(len(group_order))

    best_text = ""
    best_tables: list[str] = []
    for idx in range(1, len(group_order) + 1):
        trial_text, trial_tables = render_schema(idx)
        if len(trial_text) <= schema_char_budget:
            best_text = trial_text
            best_tables = trial_tables
            continue
        if not best_text:
            best_text = (
                trial_text[:schema_char_budget]
                + "\n\n-- SCHEMA TRUNCATED TO FIT THE CHARACTER BUDGET. Use only visible tables/columns."
            )
            best_tables = trial_tables[:1]
        break
    return best_text, best_tables


def format_schema_kasla(
    schema_dir: Path,
    predicted_tables: list[str],
    predicted_columns: dict[str, Any] | None = None,
) -> str:
    tables, missing = selected_table_objects(schema_dir, predicted_tables)
    lines: list[str] = []
    selected: set[str] = set()
    for obj in tables:
        table_name = str(obj.get("table_name") or "")
        if not table_name:
            continue
        selected.add(table_name)
        col_names = obj.get("column_names", [])
        col_types = obj.get("column_types", [])
        descriptions = obj.get("description", [])
        col_parts = []
        allowed_columns = column_filter_for_table(predicted_columns, table_name)
        for idx, col in enumerate(col_names):
            if allowed_columns is not None and normalize_column_key(col) not in allowed_columns:
                continue
            attrs = []
            typ = col_types[idx] if idx < len(col_types) else ""
            if typ:
                attrs.append(str(typ))
            desc = descriptions[idx] if idx < len(descriptions) else ""
            if desc:
                attrs.append("comment : " + str(desc))
            examples = sample_values(obj, col)
            if examples:
                attrs.append("values : " + " , ".join(map(str, examples)))
            col_parts.append(f"{table_name}.{col} ( {' | '.join(attrs)} )")
        lines.append(f"table {table_name} , columns = [")
        lines.extend("  " + part for part in col_parts)
        lines.append("]")
    fk_text = sqlite_foreign_keys(schema_dir, selected)
    lines.append("foreign keys :\n" + fk_text if fk_text else "foreign keys : None")
    if missing:
        lines.append("missing predicted tables : " + ", ".join(missing[:50]))
    return "\n".join(lines)


def build_autolink_prompt(
    question: str,
    external_knowledge: str | None,
    schema_text: str,
    sql_type: str,
    dialect_optimization: str,
) -> str:
    cfg = build_autolink_prompt.cfg
    prompt = cfg.SQL_GENERATION.replace("{PROMPT}", schema_text)
    prompt = prompt.replace("{QUESTION}", question)
    prompt = prompt.replace("{SQL_TYPE}", sql_type)
    prompt = prompt.replace("{SQL_DIALECT_OPTIMIZATION}", dialect_optimization)
    if external_knowledge:
        prompt = prompt.replace(
            "Database Schema and External Knowledge:",
            "Database Schema and External Knowledge:",
        )
        prompt += "\n\nExternal Knowledge:\n" + external_knowledge
    return prompt


build_autolink_prompt.cfg = load_autolink_config()


def build_chess_prompt(question: str, external_knowledge: str | None, schema_text: str) -> str:
    template = CHESS_TEMPLATE.read_text(encoding="utf-8")
    return (
        template.replace("{DATABASE_SCHEMA}", schema_text)
        .replace("{QUESTION}", question)
        .replace("{HINT}", external_knowledge or "None")
    )


def build_kasla_prompt(question: str, external_knowledge: str | None, schema_text: str) -> str:
    parts = [
        "[Text-to-SQL task]",
        "database schema:",
        schema_text,
        "Question:",
        question,
    ]
    if external_knowledge:
        parts.extend(["Evidence:", external_knowledge])
    parts.append("Generate SQL to solve the above question:")
    return "\n".join(parts)


def api_for_task(dataset: str, task: dict[str, Any]) -> str:
    if dataset == "spider2lite":
        instance_id = str(task["instance_id"])
        if instance_id.startswith(("bq", "ga")):
            return "bigquery"
        if instance_id.startswith("sf"):
            return "snowflake"
    return "sqlite"


def build_backend_dialect_block(api: str, table_list_text: str) -> str:
    parts = [
        f"Please think step by step and answer only one complete SQL in {api} dialect in ```sql``` format.",
        f"SQL usage example: {DIALECT_PROMPTS.get_prompt_dialect_basic(api)}",
        "Here are some useful tips for answering:",
        DIALECT_PROMPTS.get_prompt_dialect_list_all_tables(table_list_text, api),
        DIALECT_PROMPTS.get_prompt_dialect_sql_safety(api),
        DIALECT_PROMPTS.get_prompt_dialect_nested(api),
        DIALECT_PROMPTS.get_prompt_convert_symbols(),
        DIALECT_PROMPTS.get_prompt_dialect_string_matching(api),
        "For time-related queries, given the variety of formats, avoid using time converting functions unless you are certain of the specific format being used.",
        "When generating SQLs, be aware of quotation matching; do not mix single and double quotes in string literals.",
        DIALECT_PROMPTS.get_prompt_knowledge(),
        "When asked something without stating name or id, return both of them. e.g. Which products ...? The answer should include product_name and product_id.",
        "When asked percentage decrease, you should return a positive value. e.g. How many percentage points in 2021 decrease compared to ...? The answer should be a positive value indicating the decreased number. Try to use ABS().",
        "If asked two tables, you should reply with the last one instead of combining two tables. e.g. Identifying the top five states ... examine the state that ranks fourth overall and identify its top five counties. You should only answer top five counties.",
        DIALECT_PROMPTS.get_prompt_decimal_places(),
    ]
    if api == "snowflake":
        parts.append("When using ORDER BY xxx DESC, add NULLS LAST to exclude null records: ORDER BY xxx DESC NULLS LAST.")
        parts.append("Use ST_DISTANCE to calculate distance between two geographic points for more accurate answer.")
    return "\n".join(part for part in parts if part)


def append_backend_dialect_prompt(prompt: str, api: str, predicted_tables: list[str]) -> str:
    table_list_text = ", ".join(map(str, predicted_tables[:200]))
    return prompt + "\n\nBackend-aware dialect instructions:\n" + build_backend_dialect_block(api, table_list_text)


def build_compact_chess_prompt(
    question: str,
    external_knowledge: str | None,
    schema_text: str,
    api: str,
    predicted_tables: list[str],
) -> str:
    parts = [
        "You are an experienced database expert.",
        "Generate one SQL query using a recursive divide-and-conquer approach.",
        "First reason about the main question and needed sub-questions, then assemble the final SQL.",
        "",
        "Database schema:",
        schema_text,
        "",
        "Question:",
        question,
    ]
    if external_knowledge:
        parts.extend(["", "Evidence:", external_knowledge])
    parts.extend(
        [
            "",
            "Database admin instructions:",
            "- Only use tables and columns from the provided schema.",
            "- Select only the information required by the question.",
            "- Prefer INNER JOIN over unnecessary nested queries.",
            "- Use DISTINCT when the question asks for unique entities.",
            "- Return one complete SQL query only in the final answer.",
            "",
            "Dialect instructions:",
            build_backend_dialect_block(api, ", ".join(map(str, predicted_tables[:200]))),
            "",
            "When you get to the final query, output the query string ONLY inside the XML delimiter <FINAL_ANSWER></FINAL_ANSWER>.",
        ]
    )
    return "\n".join(parts)


def build_compact_kasla_prompt(
    question: str,
    external_knowledge: str | None,
    schema_text: str,
    api: str,
    predicted_tables: list[str],
) -> str:
    parts = [
        "You are an expert text-to-SQL generator.",
        "Use the provided schema and generate one executable SQL query.",
        "",
        "Database schema:",
        schema_text,
        "",
        "Question:",
        question,
    ]
    if external_knowledge:
        parts.extend(["", "Evidence:", external_knowledge])
    parts.extend(
        [
            "",
            "Dialect instructions:",
            build_backend_dialect_block(api, ", ".join(map(str, predicted_tables[:200]))),
            "",
            "Output only the final SQL in a ```sql``` code block. Do not output unrelated text.",
        ]
    )
    return "\n".join(parts)


def extract_native_sql(text: str) -> str:
    text = text.strip()
    final_matches = re.findall(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", text, flags=re.I | re.S)
    for final_text in reversed(final_matches):
        if final_text.strip():
            text = final_text.strip()
            break
    fence_matches = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence_matches:
        text = fence_matches[-1].strip()
    tagged = re.search(r"<sql>\s*(.*?)\s*</sql>", text, flags=re.I | re.S)
    if tagged:
        text = tagged.group(1).strip()
    if text.lower().startswith("sql"):
        text = text[3:].strip()
    return text.strip()


def dialect_for_task(dataset: str, task: dict[str, Any]) -> tuple[str, str]:
    cfg = build_autolink_prompt.cfg
    if dataset == "spider2lite":
        instance_id = str(task["instance_id"])
        if instance_id.startswith(("bq", "ga")):
            return "BigQuery", cfg.BIGQUERY_DIALECT_OPTIMIZATION_SQL_GEN
        if instance_id.startswith("sf"):
            return "Snowflake", cfg.SNOWFLAKE_DIALECT_OPTIMIZATION_SQL_GEN
    return "SQLite", cfg.SQLITE_DIALECT_OPTIMIZATION_SQL_GEN


def schema_dir_for_task(dataset: str, task: dict[str, Any]) -> Path:
    if dataset != "spider2lite":
        return DATASETS[dataset]["schema_root"] / task["db_name"]
    root = DATASETS[dataset]["schema_root"]
    instance_id = str(task["instance_id"])
    db_name = str(task["db_name"])
    if instance_id.startswith(("bq", "ga")):
        return root / "bigquery" / db_name
    if instance_id.startswith("sf"):
        return root / "snowflake" / db_name
    return root / "sqlite" / db_name


def make_native_tasks(dataset: str) -> list[dict[str, Any]]:
    if dataset != "spider2lite":
        return make_tasks(dataset)
    query_data = load_json(DATASETS[dataset]["query_file"])
    tasks: list[dict[str, Any]] = []
    for index, (raw_key, item) in enumerate(query_data.items()):
        tasks.append(
            {
                "index": index,
                "raw_key": raw_key,
                "db_name": item["db_name"],
                "instance_id": raw_key,
                "question": item.get("question", ""),
                "external_knowledge": item.get("external_knowledge"),
            }
        )
    return tasks


def predicted_for_task(dataset: str, predictions: dict[str, Any], task: dict[str, Any]) -> dict[str, Any] | None:
    instance_id = str(task["instance_id"])
    candidate_ids = [instance_id]
    if dataset == "spider2lite" and instance_id.startswith(("sf_bq", "sf_ga")):
        candidate_ids.append(instance_id[len("sf_"):])
    for candidate in candidate_ids:
        if candidate in predictions:
            return predictions[candidate]
    if dataset == "spider2lite":
        for candidate in candidate_ids:
            prefix = candidate + "_"
            for key, value in predictions.items():
                if str(key).startswith(prefix):
                    return value
    return None


def fit_schema_to_budget(
    formatter,
    schema_dir: Path,
    predicted_tables: list[str],
    schema_char_budget: int | None,
    predicted_columns: dict[str, Any] | None = None,
    schema_text_postprocessor=None,
) -> tuple[str, list[str]]:
    def render_schema(tables: list[str]) -> str:
        text = formatter(schema_dir, tables, predicted_columns)
        if schema_text_postprocessor is not None:
            text = schema_text_postprocessor(text, tables)
        return text

    if schema_char_budget:
        best_text = ""
        best_tables: list[str] = []
        for idx, _table in enumerate(predicted_tables, start=1):
            trial_tables = predicted_tables[:idx]
            trial_text = render_schema(trial_tables)
            if len(trial_text) <= schema_char_budget:
                best_text = trial_text
                best_tables = trial_tables
                continue
            if not best_text:
                best_tables = trial_tables
                best_text = (
                    trial_text[:schema_char_budget]
                    + "\n\n-- SCHEMA TRUNCATED TO FIT THE CHARACTER BUDGET. Use only visible tables/columns."
                )
            break
        return best_text, best_tables

    schema_text = render_schema(predicted_tables)
    return schema_text, predicted_tables


def build_prompt(
    method: str,
    dataset: str,
    schema_dir: Path,
    task: dict[str, Any],
    predicted_tables: list[str],
    predicted_columns: dict[str, Any] | None,
    backend_dialect_prompts: bool,
    schema_char_budget: int | None,
    graphlink_dependency_hints: bool = False,
    graphlink_dependency_hint_limit: int = 80,
) -> tuple[str, str, int]:
    hint_items: list[dict[str, Any]] = []
    schema_text_postprocessor = None
    if graphlink_dependency_hints and method in {"AutoLink", "CHESS", "KaSLA"}:
        hint_items = collect_dependency_hint_items(dataset, task, predicted_tables)

        def schema_text_postprocessor(schema_text: str, selected_tables: list[str]) -> str:
            return append_dependency_hint_block(
                schema_text,
                hint_items,
                selected_tables,
                graphlink_dependency_hint_limit,
            )

    if method == "AutoLink":
        schema_text, prompt_tables = fit_schema_to_budget(
            format_schema,
            schema_dir,
            predicted_tables,
            schema_char_budget,
            schema_text_postprocessor=schema_text_postprocessor,
        )
        sql_type, dialect_optimization = dialect_for_task(dataset, task)
        prompt_style = "autolink_sql_generation"
        if graphlink_dependency_hints:
            prompt_style += "_graphlink_dependency_hints"
        return (
            build_autolink_prompt(
                task["question"],
                task.get("external_knowledge"),
                schema_text,
                sql_type,
                dialect_optimization,
            ),
            prompt_style,
            len(prompt_tables),
        )
    if method == "CHESS":
        if backend_dialect_prompts:
            if dataset == "spider2lite":
                schema_text, prompt_tables = fit_spider2lite_compact_schema_to_budget(
                    schema_dir,
                    predicted_tables,
                    schema_char_budget,
                    predicted_columns,
                    schema_text_postprocessor,
                    task,
                )
                style = "chess_candidate_generator_backend_dialect_compact_schema"
            else:
                schema_text, prompt_tables = fit_schema_to_budget(
                    format_schema_as_create,
                    schema_dir,
                    predicted_tables,
                    schema_char_budget,
                    predicted_columns,
                    schema_text_postprocessor,
                )
                style = "chess_candidate_generator_backend_dialect"
            prompt = build_compact_chess_prompt(
                task["question"],
                task.get("external_knowledge"),
                schema_text,
                api_for_task(dataset, task),
                predicted_tables,
            )
            if graphlink_dependency_hints:
                style += "_graphlink_dependency_hints"
            return prompt, style, len(prompt_tables)
        schema_text, prompt_tables = fit_schema_to_budget(
            format_schema_as_create,
            schema_dir,
            predicted_tables,
            schema_char_budget,
            predicted_columns,
            schema_text_postprocessor,
        )
        prompt = build_chess_prompt(task["question"], task.get("external_knowledge"), schema_text)
        style = "chess_candidate_generator"
        if graphlink_dependency_hints:
            style += "_graphlink_dependency_hints"
        return prompt, style, len(prompt_tables)
    if method == "KaSLA":
        dialect_tables = predicted_tables
        if backend_dialect_prompts and dataset == "spider2lite":
            schema_text, prompt_tables = fit_spider2lite_compact_schema_to_budget(
                schema_dir,
                predicted_tables,
                schema_char_budget,
                predicted_columns,
                schema_text_postprocessor,
                task,
            )
            style = "kasla_text_to_sql_backend_dialect_compact_schema"
        else:
            kasla_formatter = format_schema_as_create if backend_dialect_prompts else format_schema_kasla
            schema_text, prompt_tables = fit_schema_to_budget(
                kasla_formatter,
                schema_dir,
                predicted_tables,
                schema_char_budget,
                predicted_columns,
                schema_text_postprocessor,
            )
            dialect_tables = prompt_tables
            style = "kasla_text_to_sql_backend_dialect"
        if backend_dialect_prompts:
            prompt = build_compact_kasla_prompt(
                task["question"],
                task.get("external_knowledge"),
                schema_text,
                api_for_task(dataset, task),
                dialect_tables,
            )
            if graphlink_dependency_hints:
                style += "_graphlink_dependency_hints"
            return prompt, style, len(prompt_tables)
        prompt = build_kasla_prompt(task["question"], task.get("external_knowledge"), schema_text)
        style = "kasla_text_to_sql"
        if graphlink_dependency_hints:
            style += "_graphlink_dependency_hints"
        return prompt, style, len(prompt_tables)
    raise ValueError(f"Unsupported method for native API reproduction: {method}")


def generate_one(
    client: Any,
    provider: str,
    model: str,
    dataset: str,
    method: str,
    task: dict[str, Any],
    predicted: dict[str, Any] | None,
    fill_predicted: dict[str, Any] | None,
    table_count_limit: int | None,
    table_count_limit_mode: str,
    temperature: float,
    max_tokens: int,
    request_timeout_sec: float,
    retries: int,
    retry_delay: float,
    backend_dialect_prompts: bool,
    schema_char_budget: int | None,
    use_predicted_columns: bool,
    graphlink_dependency_hints: bool,
    graphlink_dependency_hint_limit: int,
) -> dict[str, Any]:
    schema_dir = schema_dir_for_task(dataset, task)
    raw_pred_tables = (predicted or {}).get("predicted_tables") or []
    pred_columns = (predicted or {}).get("predicted_columns") if use_predicted_columns else None
    pred_tables, pred_columns, table_limit_meta = apply_table_count_limit_and_fill(
        raw_pred_tables,
        pred_columns,
        table_count_limit,
        fill_predicted,
        table_count_limit_mode,
        task,
    )
    prompt, prompt_style, num_prompt_tables = build_prompt(
        method,
        dataset,
        schema_dir,
        task,
        pred_tables,
        pred_columns,
        backend_dialect_prompts,
        schema_char_budget,
        graphlink_dependency_hints,
        graphlink_dependency_hint_limit,
    )
    started = time.time()
    raw = ""
    sql = ""
    error = None
    for attempt in range(retries + 1):
        try:
            raw = call_model(client, provider, model, prompt, temperature, max_tokens, request_timeout_sec)
            sql = extract_native_sql(raw)
            error = None
            break
        except Exception as exc:
            error = repr(exc)
            if attempt < retries:
                time.sleep(retry_delay * (attempt + 1))
    return {
        "index": task["index"],
        "db_id": db_display_id(dataset, schema_dir, task["db_name"]),
        "instance_id": task["instance_id"],
        "question": task["question"],
        "method": method,
        "dataset": dataset,
        "prompt_style": prompt_style,
        "predicted_sql": sql,
        "error": error,
        "latency_sec": round(time.time() - started, 3),
        "num_linked_tables": len(raw_pred_tables),
        "num_effective_linked_tables": len(pred_tables),
        "num_prompt_tables": num_prompt_tables,
        "effective_predicted_tables": pred_tables,
        **table_limit_meta,
        "prompt_chars": len(prompt),
        "graphlink_dependency_hints": graphlink_dependency_hints,
        "dependency_hint_lines": dependency_hint_line_count(prompt),
        "raw_response": raw,
    }


def make_client(provider: str, api_key: str, base_url: str, client_max_retries: int) -> Any:
    if provider == "anthropic":
        return Anthropic(api_key=api_key, base_url=base_url, timeout=180, max_retries=client_max_retries)
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=client_max_retries)


def call_model(
    client: Any,
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    request_timeout_sec: float,
) -> str:
    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=request_timeout_sec,
        )
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text or "")
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(block))
        return "".join(parts)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout_sec,
    )
    return response.choices[0].message.content or ""


def is_successful_prediction(row: dict[str, Any]) -> bool:
    return not row.get("error") and bool(str(row.get("predicted_sql") or "").strip())


def completed_native_ids(jsonl_path: Path, retry_errors: bool = False) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            instance_id = str(row.get("instance_id") or "")
            if not instance_id:
                continue
            if retry_errors and not is_successful_prediction(row):
                continue
            done.add(instance_id)
    return done


def aggregate_native_json(jsonl_path: Path, json_path: Path) -> None:
    rows_by_id: dict[str, dict[str, Any]] = {}
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                instance_id = str(row.get("instance_id") or "")
                if instance_id:
                    rows_by_id[instance_id] = row
    rows = sorted(rows_by_id.values(), key=lambda x: x.get("index", 0))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def resolve_table_fill_linking_path(args: argparse.Namespace, dataset: str) -> Path | None:
    if args.table_fill_linking_file:
        return Path(args.table_fill_linking_file)
    if not args.table_fill_method:
        return None
    fill_dir = Path(args.table_fill_linking_dir or args.linking_dir)
    return fill_dir / f"{args.table_fill_method}_{dataset}.json"


def run_pair(args: argparse.Namespace, dataset: str, method: str) -> None:
    tasks = make_native_tasks(dataset)
    if args.include_ids:
        include_ids = set(args.include_ids)
        tasks = [task for task in tasks if str(task["instance_id"]) in include_ids]
    linking_path = Path(args.linking_dir) / f"{method}_{dataset}.json"
    predictions = load_json(linking_path)
    fill_path = resolve_table_fill_linking_path(args, dataset)
    fill_predictions = load_json(fill_path) if fill_path is not None else {}

    out_dir = Path(args.output_dir) / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "predictions.jsonl"
    json_path = out_dir / "predictions.json"
    done = completed_native_ids(jsonl_path, retry_errors=args.retry_errors)
    todo = [task for task in tasks if task["instance_id"] not in done]
    if args.limit_per_pair is not None:
        todo = todo[: args.limit_per_pair]

    print(
        f"[{dataset}/{method}] total={len(tasks)} done={len(done)} todo={len(todo)} "
        f"linking={len(predictions)} fill_linking={len(fill_predictions)} "
        f"table_limit={args.table_count_limit} table_limit_mode={args.table_count_limit_mode} "
        f"output={json_path}",
        flush=True,
    )

    client = make_client(args.provider, args.api_key, args.base_url, args.client_max_retries)
    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                generate_one,
                client,
                args.provider,
                args.model,
                dataset,
                method,
                task,
                predicted_for_task(dataset, predictions, task),
                predicted_for_task(dataset, fill_predictions, task),
                args.table_count_limit,
                args.table_count_limit_mode,
                args.temperature,
                args.max_tokens,
                args.request_timeout_sec,
                args.retries,
                args.retry_delay,
                args.backend_dialect_prompts,
                args.schema_char_budget,
                args.use_predicted_columns,
                args.graphlink_dependency_hints,
                args.graphlink_dependency_hint_limit,
            ): task
            for task in todo
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            append_jsonl(jsonl_path, result)
            finished += 1
            if finished % args.flush_every == 0 or finished == len(todo):
                aggregate_native_json(jsonl_path, json_path)
                print(f"[{dataset}/{method}] progress {len(done) + finished}/{len(tasks)}", flush=True)
    aggregate_native_json(jsonl_path, json_path)
    print(f"[{dataset}/{method}] completed -> {json_path}", flush=True)


def main() -> None:
    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(proxy_key, None)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["spider", "bird"], choices=sorted(DATASETS))
    parser.add_argument("--methods", nargs="+", default=NATIVE_API_METHODS, choices=NATIVE_API_METHODS)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--linking-dir", default=str(SCHEMA_LINKING_DIR))
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=os.environ.get("SQLGEN_PROVIDER", "openai"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--client-max-retries", type=int, default=2)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit-per-pair", type=int, default=None)
    parser.add_argument("--include-ids", nargs="+", default=None)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--backend-dialect-prompts", action="store_true")
    parser.add_argument("--schema-char-budget", type=int, default=None)
    parser.add_argument("--use-predicted-columns", action="store_true")
    parser.add_argument("--graphlink-dependency-hints", action="store_true")
    parser.add_argument("--graphlink-dependency-hint-limit", type=int, default=80)
    parser.add_argument(
        "--table-count-limit",
        type=int,
        default=None,
        help="Keep at most this many schema-linking tables/groups after optional fill.",
    )
    parser.add_argument(
        "--table-count-limit-mode",
        choices=["table", "spider2lite_group"],
        default="table",
        help=(
            "How to apply --table-count-limit. 'table' keeps the legacy unique-table "
            "behavior; 'spider2lite_group' first merges Spider2Lite physical shard "
            "tables into logical groups, then keeps the top-k groups."
        ),
    )
    parser.add_argument(
        "--table-fill-linking-file",
        default=None,
        help="Optional JSON linking file used to fill missing tables up to --table-count-limit.",
    )
    parser.add_argument(
        "--table-fill-linking-dir",
        default=None,
        help="Directory for --table-fill-method; defaults to --linking-dir.",
    )
    parser.add_argument(
        "--table-fill-method",
        default=None,
        help="Method name used with --table-fill-linking-dir as {method}_{dataset}.json.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry rows with an error or empty predicted_sql when resuming from predictions.jsonl.",
    )
    args = parser.parse_args()

    if args.provider == "anthropic":
        args.base_url = args.base_url or os.environ.get("ANTHROPIC_BASE_URL")
        args.api_key = args.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
        args.model = args.model or os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro")
    else:
        args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
        args.api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        args.model = args.model or os.environ.get("VLLM_MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507-FP8")
    if not args.api_key:
        raise RuntimeError(f"Missing API key for provider={args.provider}")

    print(f"provider={args.provider} base_url={args.base_url} model={args.model} workers={args.workers}", flush=True)
    for dataset in args.datasets:
        for method in args.methods:
            run_pair(args, dataset, method)


if __name__ == "__main__":
    main()
