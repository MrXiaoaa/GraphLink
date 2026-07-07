#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert generated QA JSONL (from generate_qa_from_tables.py) into a VERL-compatible Parquet
for schema-filtering + SQL-generation RL.

Input JSONL (one object per line) is expected to include at least:
  - query (str)
  - tables (List[str])                 # the selected tables used by the gold SQL  (gold_tables)
  - sql (str)                          # gold SQL
  - data (str|None)                    # execution result of gold SQL (CSV string recommended)
  - success (bool)                     # whether gold SQL executed successfully
  - source_example (str)               # example directory name, e.g., "bq006", "local001"

This script builds:
  - candidate_tables: all tables available in that example (from table_descriptions.json)
  - gold_tables: tables from the sample ("tables")
  - prompt: a single-turn chat prompt asking the model to output JSON with:
      {"selected_tables": [...], "sql": "..."}  (see schema_filtering_reward.parse_solution)

Output Parquet columns (safe for VERL / custom reward):
  - id (str)
  - prompt (list[dict])        # chat messages
  - ground_truth (dict)        # {query, candidate_tables, gold_tables, gold_sql, gold_data}
  - extra_info (dict)          # {api, source_example, sqlite_path, ...}

Usage:
  python convert2parquet.py \
    --input_jsonl generated_qa.jsonl \
    --examples_root /path/to/examples_lite \
    --output_parquet train.parquet \
    --only_success \
    --require_gold_data
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -------------------------
# IO helpers
# -------------------------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {ln} in {path}: {e}") from e


def load_table_descriptions(examples_root: str, source_example: str) -> Dict[str, str]:
    """
    Read examples_root/<source_example>/table_descriptions.json
    """
    p = Path(examples_root) / source_example / "table_descriptions.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing table_descriptions.json: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"table_descriptions.json must be a dict, got {type(data)} at {p}")
    # keys are table names, values are descriptions
    return {str(k): str(v) for k, v in data.items()}


def infer_api(source_example: str) -> str:
    """
    Match your utils.get_api_name() conventions.
    bq* -> bigquery, sf* -> snowflake, local* -> sqlite, else sqlite.
    """
    if source_example.startswith("bq"):
        return "bigquery"
    if source_example.startswith("sf") or source_example.startswith("snowflake"):
        return "snowflake"
    if source_example.startswith("local"):
        return "sqlite"
    return "sqlite"


def maybe_find_sqlite_path(examples_root: str, source_example: str) -> Optional[str]:
    """
    For local sqlite examples: find *.sqlite or *.db inside examples_root/<source_example>/.
    For non-sqlite, return None.
    """
    ex_dir = Path(examples_root) / source_example
    if not ex_dir.exists() or not ex_dir.is_dir():
        return None
    for p in sorted(ex_dir.iterdir()):
        if p.is_file() and p.suffix in [".sqlite", ".db"]:
            return str(p)
    return None


# -------------------------
# Prompt building
# -------------------------
def build_prompt(query: str, candidate_tables: List[str], table_descriptions: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Single-turn prompt. Output format aligned with schema_filtering_reward.parse_solution:
      {"selected_tables":[...], "sql":"..."}
    """
    # Keep table list concise and stable
    desc_lines = []
    for t in candidate_tables:
        desc = table_descriptions.get(t, "")
        # guard very long descriptions
        if len(desc) > 300:
            desc = desc[:300] + "..."
        desc_lines.append(f"- {t}: {desc}")

    tables_block = "\n".join(desc_lines)

    content = f"""You are a database analyst and a SQL expert.

## User Question
{query}

## Candidate Tables (you must select from this list only)
{tables_block}

## Task
1) Select the MINIMUM set of tables needed to answer the question (do NOT guess joins).
2) Write an executable SQL query that answers the question.

## Strict Rules
- selected_tables must be a subset of the candidate tables above.
- Do NOT invent tables or columns.
- Only join tables when the join key is obvious from naming (same key name or *_id style).
- Output MUST be valid JSON with exactly these keys: selected_tables, sql
- Output MUST contain NO extra text before or after the JSON.
- If you cannot answer without guessing, return selected_tables as [] and sql as "".

## Output JSON format
{{"selected_tables": ["t1","t2"], "sql": "SELECT ..."}}
"""
    return [{"role": "user", "content": content}]


# -------------------------
# Record conversion
# -------------------------
def normalize_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes"}
    if isinstance(x, (int, float)):
        return bool(x)
    return False


def make_record(
    sample: Dict[str, Any],
    examples_root: str,
    only_success: bool,
    require_gold_data: bool,
) -> Optional[Dict[str, Any]]:
    # ---- filter stage
    success = normalize_bool(sample.get("success", False))
    if only_success and not success:
        return None

    query = sample.get("query") or ""
    gold_tables = sample.get("tables") or []
    gold_sql = sample.get("sql") or ""
    gold_data = sample.get("data", None)
    source_example = sample.get("source_example") or ""

    if not query or not source_example:
        return None

    if not isinstance(gold_tables, list):
        gold_tables = []
    gold_tables = [str(x) for x in gold_tables]

    if not gold_sql or not str(gold_sql).strip():
        # even if success was true, keep it safe
        return None

    if require_gold_data:
        if gold_data is None:
            return None
        if isinstance(gold_data, str) and not gold_data.strip():
            return None

    # ---- candidate tables from descriptions
    table_desc = load_table_descriptions(examples_root, source_example)
    candidate_tables = list(table_desc.keys())

    # ---- build prompt
    prompt = build_prompt(query, candidate_tables, table_desc)

    api = infer_api(source_example)
    sqlite_path = maybe_find_sqlite_path(examples_root, source_example) if api == "sqlite" else None

    rid = f"{source_example}::{sample.get('query', '')[:50].strip()}".replace("\n", " ")

    return {
        "id": rid,
        "prompt": prompt,
        "ground_truth": {
            "query": query,
            "candidate_tables": candidate_tables,
            "gold_tables": gold_tables,
            "gold_sql": gold_sql,
            "gold_data": gold_data,
        },
        "extra_info": {
            "api": api,
            "source_example": source_example,
            "sqlite_path": sqlite_path,
            # optional debug / analysis fields
            "relevant_columns": sample.get("relevant_columns", []),
            "gold_exec_time": sample.get("execution_time", None),
            "gold_size": sample.get("size", None),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True, help="JSONL produced by generate_qa_from_tables.py")
    ap.add_argument("--examples_root", type=str, required=True, help="examples_lite root dir (contains per-example folders)")
    ap.add_argument("--output_parquet", type=str, required=True, help="Output parquet path")
    ap.add_argument("--only_success", action="store_true", help="Keep only success==true samples")
    ap.add_argument("--require_gold_data", action="store_true", help="Drop samples with missing/empty gold data")
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap for number of records written")
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    n_total = 0
    n_kept = 0
    n_skipped = 0

    for sample in iter_jsonl(str(in_path)):
        n_total += 1
        rec = make_record(
            sample=sample,
            examples_root=args.examples_root,
            only_success=args.only_success,
            require_gold_data=args.require_gold_data,
        )
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
        n_kept += 1
        if args.max_samples is not None and n_kept >= args.max_samples:
            break

    if not records:
        raise RuntimeError(
            f"No records written. total={n_total}, skipped={n_skipped}. "
            f"Tip: remove --only_success / --require_gold_data to debug input."
        )

    df = pd.DataFrame.from_records(records)
    df.to_parquet(str(out_path), index=False)

    print("✅ Done")
    print(f"Input:  {in_path}  (lines processed: {n_total})")
    print(f"Output: {out_path}")
    print(f"Kept:   {n_kept}")
    print(f"Skipped:{n_skipped}")
    print("\nColumns:", list(df.columns))


if __name__ == "__main__":
    main()
