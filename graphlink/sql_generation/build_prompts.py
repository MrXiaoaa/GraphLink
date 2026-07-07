from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from graphlink.schema_linking.dependency_hints import dependency_hints_for_instance
from graphlink.schema_linking.format import read_linking, selected_tables_for_instance

SEP = "-" * 50
STRUCT_MARKER = "The table structure information is"
EXTERNAL_MARKER = "External knowledge that might be helpful:"


def _blocks(prompt: str) -> list[str]:
    if STRUCT_MARKER not in prompt:
        return []
    body = prompt.split(STRUCT_MARKER, 1)[1]
    if EXTERNAL_MARKER in body:
        body = body.split(EXTERNAL_MARKER, 1)[0]
    return [block.strip() for block in body.split(SEP) if "Table full name:" in block]


def _table_name(block: str) -> str | None:
    match = re.search(r"^Table full name:\s*(.+?)\s*$", block, flags=re.M)
    return match.group(1).strip() if match else None


def _keys(table: str) -> set[str]:
    raw = str(table).strip().strip('`"\'').lower()
    return {raw, raw.split(".")[-1]}


def filter_schema_blocks(original_prompt: str, selected_tables: list[str]) -> tuple[str, int]:
    selected = set().union(*[_keys(table) for table in selected_tables]) if selected_tables else set()
    kept: list[str] = []
    for block in _blocks(original_prompt):
        name = _table_name(block)
        if name and (_keys(name) & selected):
            kept.append(block)
    schema_text = f"\n{SEP}\n".join(kept)
    return schema_text, len(kept)


def build_prompt(original_prompt: str, selected_tables: list[str], dependency_block: str = "", char_budget: int | None = None) -> tuple[str, int]:
    schema_text, kept_count = filter_schema_blocks(original_prompt, selected_tables)
    if not schema_text:
        schema_text = "\n".join(f"Table full name: {table}" for table in selected_tables)
    prefix = original_prompt.split(STRUCT_MARKER, 1)[0] + STRUCT_MARKER + "\n"
    suffix = ""
    if EXTERNAL_MARKER in original_prompt:
        suffix = EXTERNAL_MARKER + original_prompt.split(EXTERNAL_MARKER, 1)[1]
    prompt = prefix + schema_text.strip() + "\n"
    if dependency_block:
        prompt += "\n" + dependency_block.strip() + "\n"
    if suffix:
        prompt += "\n" + suffix
    if char_budget and len(prompt) > char_budget:
        prompt = prompt[:char_budget]
    return prompt, kept_count


def ensure_payload(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name == "prompts.txt":
            continue
        target = dst_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GraphLink selected-schema SQL generation examples.")
    parser.add_argument("--source-examples", type=Path, required=True)
    parser.add_argument("--output-examples", type=Path, required=True)
    parser.add_argument("--linking-file", type=Path, required=True)
    parser.add_argument("--database-graphs-dir", type=Path, default=None)
    parser.add_argument("--dependency-hints", action="store_true")
    parser.add_argument("--hint-limit", type=int, default=32)
    parser.add_argument("--prompt-char-budget", type=int, default=None)
    parser.add_argument("--include-ids", nargs="*", default=None)
    args = parser.parse_args()

    linking = read_linking(args.linking_file)
    include = set(args.include_ids or []) or None
    args.output_examples.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src_dir in sorted(path for path in args.source_examples.iterdir() if path.is_dir()):
        instance_id = src_dir.name
        if include and instance_id not in include:
            continue
        original = src_dir / "prompts.txt"
        if not original.exists():
            continue
        selected = selected_tables_for_instance(linking, instance_id)
        dep = ""
        if args.dependency_hints and args.database_graphs_dir:
            dep = dependency_hints_for_instance(args.database_graphs_dir, instance_id, selected, args.hint_limit)
        prompt, kept = build_prompt(original.read_text(encoding="utf-8"), selected, dep, args.prompt_char_budget)
        dst_dir = args.output_examples / instance_id
        ensure_payload(src_dir, dst_dir)
        (dst_dir / "prompts.txt").write_text(prompt, encoding="utf-8")
        manifest.append({"instance_id": instance_id, "selected_tables": len(selected), "kept_schema_blocks": kept, "prompt_chars": len(prompt), "dependency_hints": bool(dep)})
    (args.output_examples / "_graphlink_manifest.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in manifest) + "\n", encoding="utf-8")
    print(json.dumps({"examples": len(manifest), "output": str(args.output_examples)}, indent=2))


if __name__ == "__main__":
    main()
