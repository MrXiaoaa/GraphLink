from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOLINK_RUN = REPO_ROOT / "methods" / "baselines" / "AutoLink" / "run"
SCHEMA_LINKING_DIR = REPO_ROOT / "data" / "unified_schema_linking"

DATASETS = {
    "spider": {
        "query_file": AUTOLINK_RUN / "data" / "spider" / "spider2_data.json",
        "schema_root": AUTOLINK_RUN / "resource" / "databases" / "sqlite_spider",
    },
    "bird": {
        "query_file": AUTOLINK_RUN / "data" / "bird" / "spider2_data.json",
        "schema_root": AUTOLINK_RUN / "resource" / "databases" / "sqlite_bird",
    },
}

BASELINES = ["AutoLink", "CHESS", "GraphLink", "KaSLA", "LinkAlign"]
WRITE_LOCK = threading.Lock()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_table_name(name: str) -> str:
    text = str(name).strip().strip("`\"'")
    text = text.split(" ")[0]
    if "." in text:
        text = text.split(".")[-1]
    return text.strip().strip("`\"'")


def bird_instance_id(db_name: str, index: int) -> str:
    return f"local_{db_name}_{index}"


def make_tasks(dataset: str) -> list[dict[str, Any]]:
    query_data = load_json(DATASETS[dataset]["query_file"])
    tasks: list[dict[str, Any]] = []
    for raw_key, item in query_data.items():
        index = int(raw_key.split("_")[-1]) if "_" in raw_key else len(tasks)
        db_name = item["db_name"]
        instance_id = db_name if dataset == "spider" else bird_instance_id(db_name, index)
        tasks.append(
            {
                "index": index,
                "raw_key": raw_key,
                "db_name": db_name,
                "instance_id": instance_id,
                "question": item.get("question", ""),
                "external_knowledge": item.get("external_knowledge"),
            }
        )
    tasks.sort(key=lambda x: x["index"])
    return tasks


def db_display_id(dataset: str, schema_dir: Path, db_name: str) -> str:
    if dataset == "bird":
        return db_name
    sqlite_files = sorted(schema_dir.glob("*.sqlite"))
    if sqlite_files:
        return sqlite_files[0].stem
    name = db_name
    if name.startswith("local_"):
        name = name[len("local_") :]
    return re.sub(r"_\d+$", "", name)


def load_table_schema(schema_dir: Path, table_name: str) -> dict[str, Any] | None:
    table = sanitize_table_name(table_name)
    direct = schema_dir / f"{table}.json"
    if direct.exists():
        return load_json(direct)
    lower = table.lower()
    for path in schema_dir.glob("*.json"):
        if path.stem.lower() == lower:
            return load_json(path)
    return None


def sqlite_foreign_keys(schema_dir: Path, selected_tables: set[str]) -> str:
    sqlite_files = sorted(schema_dir.glob("*.sqlite"))
    if not sqlite_files:
        return ""
    selected_lower = {t.lower() for t in selected_tables}
    lines: list[str] = []
    try:
        conn = sqlite3.connect(str(sqlite_files[0]))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        for table in tables:
            if table.lower() not in selected_lower:
                continue
            try:
                cur.execute(f"PRAGMA foreign_key_list('{table}')")
                for row in cur.fetchall():
                    ref_table = row[2]
                    if ref_table.lower() in selected_lower:
                        lines.append(f"- {table}.{row[3]} -> {ref_table}.{row[4]}")
            except Exception:
                continue
    except Exception:
        return ""
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return "\n".join(lines)


def format_schema(schema_dir: Path, predicted_tables: list[str], max_sample_rows: int = 2) -> str:
    seen: set[str] = set()
    blocks: list[str] = []
    selected: set[str] = set()
    missing: list[str] = []

    for table in predicted_tables:
        short = sanitize_table_name(table)
        key = short.lower()
        if not short or key in seen:
            continue
        seen.add(key)
        obj = load_table_schema(schema_dir, short)
        if obj is None:
            missing.append(short)
            continue
        table_name = obj.get("table_name") or short
        selected.add(str(table_name))
        col_names = obj.get("column_names", [])
        col_types = obj.get("column_types", [])
        descriptions = obj.get("description", [])
        lines = [f"Table: {table_name}", "Columns:"]
        for idx, col in enumerate(col_names):
            typ = col_types[idx] if idx < len(col_types) else ""
            desc = descriptions[idx] if idx < len(descriptions) else ""
            desc_text = f" -- {desc}" if desc else ""
            lines.append(f"- {col} ({typ}){desc_text}")
        samples = obj.get("sample_rows", [])[:max_sample_rows]
        if samples:
            lines.append("Sample rows:")
            for row in samples:
                lines.append(json.dumps(row, ensure_ascii=False, default=str))
        blocks.append("\n".join(lines))

    fk_text = sqlite_foreign_keys(schema_dir, selected)
    if fk_text:
        blocks.append("Foreign keys:\n" + fk_text)
    if missing:
        blocks.append("Predicted tables without local schema JSON: " + ", ".join(missing[:50]))
    return "\n\n".join(blocks)


def extract_sql(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    tagged = re.search(r"<sql>\s*(.*?)\s*</sql>", text, flags=re.I | re.S)
    if tagged:
        text = tagged.group(1).strip()
    lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
    text = "\n".join(lines).strip()
    if text.lower().startswith("sql"):
        text = text[3:].strip()
    return text


def build_messages(question: str, external_knowledge: str | None, schema_text: str) -> list[dict[str, str]]:
    system = (
        "You are an expert text-to-SQL generator. Generate one valid SQLite SQL query. "
        "Use only the provided schema. Return SQL only, without explanation or markdown."
    )
    user_parts = [
        "Database schema:",
        schema_text if schema_text else "(No linked schema tables were provided.)",
        "",
        "Question:",
        question,
    ]
    if external_knowledge:
        user_parts.extend(["", "External knowledge:", external_knowledge])
    user_parts.extend(["", "Return only the final SQL query."])
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_parts)}]


def completed_ids(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            done.add(str(obj.get("instance_id")))
    return done


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_one(
    client: OpenAI,
    model: str,
    dataset: str,
    baseline: str,
    task: dict[str, Any],
    predicted: dict[str, Any] | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    schema_dir = DATASETS[dataset]["schema_root"] / task["db_name"]
    pred_tables = []
    if predicted:
        pred_tables = predicted.get("predicted_tables") or []
    schema_text = format_schema(schema_dir, pred_tables)
    messages = build_messages(task["question"], task.get("external_knowledge"), schema_text)
    error = None
    raw = ""
    sql = ""
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=180,
        )
        raw = response.choices[0].message.content or ""
        sql = extract_sql(raw)
    except Exception as exc:
        error = repr(exc)
    return {
        "index": task["index"],
        "db_id": db_display_id(dataset, schema_dir, task["db_name"]),
        "instance_id": task["instance_id"],
        "question": task["question"],
        "baseline": baseline,
        "dataset": dataset,
        "predicted_sql": sql,
        "error": error,
        "latency_sec": round(time.time() - started, 3),
        "num_linked_tables": len(pred_tables),
        "raw_response": raw,
    }


def aggregate_json(jsonl_path: Path, json_path: Path) -> None:
    rows = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rows.sort(key=lambda x: x.get("index", 0))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def run_pair(args: argparse.Namespace, dataset: str, baseline: str) -> None:
    tasks = make_tasks(dataset)
    linking_path = SCHEMA_LINKING_DIR / f"{baseline}_{dataset}.json"
    predictions = load_json(linking_path)

    out_dir = Path(args.output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{baseline}_{dataset}.jsonl"
    json_path = out_dir / f"{baseline}_{dataset}.json"
    done = completed_ids(jsonl_path)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    todo = [task for task in tasks if task["instance_id"] not in done]
    if args.limit_per_pair is not None:
        todo = todo[: args.limit_per_pair]
    print(
        f"[{dataset}/{baseline}] total={len(tasks)} done={len(done)} todo={len(todo)} "
        f"linking={len(predictions)} output={json_path}",
        flush=True,
    )

    finished = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                generate_one,
                client,
                args.model,
                dataset,
                baseline,
                task,
                predictions.get(task["instance_id"]),
                args.temperature,
                args.max_tokens,
            ): task
            for task in todo
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            append_jsonl(jsonl_path, result)
            finished += 1
            if finished % args.flush_every == 0 or finished == len(todo):
                aggregate_json(jsonl_path, json_path)
                errors = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if '"error": null' not in line)
                print(
                    f"[{dataset}/{baseline}] progress {len(done) + finished}/{len(tasks)} "
                    f"errors_seen={errors}",
                    flush=True,
                )
    aggregate_json(jsonl_path, json_path)
    print(f"[{dataset}/{baseline}] completed -> {json_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["spider", "bird"], choices=sorted(DATASETS))
    parser.add_argument("--baselines", nargs="+", default=BASELINES, choices=BASELINES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL_NAME", "Qwen3-235B-A22B-Instruct-2507-FP8"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit-per-pair", type=int, default=None)
    args = parser.parse_args()

    print(f"base_url={args.base_url} model={args.model} workers={args.workers}", flush=True)
    for dataset in args.datasets:
        for baseline in args.baselines:
            run_pair(args, dataset, baseline)


if __name__ == "__main__":
    main()
