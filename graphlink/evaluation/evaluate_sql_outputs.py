#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOLINK_RUN = Path(os.environ.get("GRAPHLINK_AUTOLINK_RUN", str(REPO_ROOT / "data" / "autolink_run")))
SUMMARY = REPO_ROOT / "outputs" / "sql_generation_eval_summary.csv"
DETAIL_DIR = REPO_ROOT / "outputs" / "eval_details"

DATASETS = {
    "spider": {
        "query_file": AUTOLINK_RUN / "data" / "spider" / "spider2_data.json",
        "gold_file": REPO_ROOT / "datasets" / "spider_full" / "spider_data" / "dev_gold.sql",
        "sqlite_root": AUTOLINK_RUN / "resource" / "databases" / "sqlite_spider",
        "prefix": "spider",
    },
    "bird": {
        "query_file": AUTOLINK_RUN / "data" / "bird" / "spider2_data.json",
        "gold_file": REPO_ROOT / "datasets" / "bird" / "bird_unzipped" / "dev_20240627" / "dev_gold.sql",
        "sqlite_root": AUTOLINK_RUN / "resource" / "databases" / "sqlite_bird",
        "prefix": "bird",
    },
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_sql(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    fence = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence[-1].strip()
    tagged = re.search(r"<sql>\s*(.*?)\s*</sql>", text, flags=re.I | re.S)
    if tagged:
        text = tagged.group(1).strip()
    final = re.search(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", text, flags=re.I | re.S)
    if final:
        text = final.group(1).strip()
    match = re.search(r"\b(WITH|SELECT|INSERT|UPDATE|DELETE)\b", text, flags=re.I)
    if match:
        text = text[match.start() :]
    return text.strip().rstrip(";") + ";" if text.strip() else ""


def normalize_result(rows: list[tuple[Any, ...]], ordered: bool):
    normalized = [tuple("NULL" if value is None else str(value) for value in row) for row in rows]
    if ordered:
        return normalized
    return collections.Counter(normalized)


def run_sql(db_path: Path, sql: str, timeout_sec: float) -> list[tuple[Any, ...]]:
    if not sql:
        raise ValueError("empty sql")
    conn = sqlite3.connect(str(db_path))
    start = time.time()

    def progress() -> int:
        return 1 if time.time() - start > timeout_sec else 0

    conn.set_progress_handler(progress, 10000)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()
    finally:
        conn.close()


def load_tasks(dataset: str) -> list[dict[str, Any]]:
    data = load_json(DATASETS[dataset]["query_file"])
    tasks: list[dict[str, Any]] = []
    for raw_key, item in data.items():
        index = int(raw_key.split("_")[-1]) if "_" in raw_key else len(tasks)
        tasks.append(
            {
                "index": index,
                "raw_key": raw_key,
                "db_name": item["db_name"],
                "question": item.get("question", ""),
            }
        )
    return sorted(tasks, key=lambda row: row["index"])


def load_gold(dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with DATASETS[dataset]["gold_file"].open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                sql, db_name = line.rsplit("\t", 1)
            else:
                sql, db_name = line, None
            rows.append({"index": index, "gold_sql": sql, "gold_db_name": db_name})
    return rows


def sqlite_path(dataset: str, task: dict[str, Any], gold: dict[str, Any]) -> Path:
    root = DATASETS[dataset]["sqlite_root"]
    db_name = str(task.get("db_name") or gold.get("gold_db_name") or "")
    schema_dir = root / db_name
    sqlite_files = sorted(schema_dir.glob("*.sqlite"))
    if sqlite_files:
        return sqlite_files[0]
    gold_db = str(gold.get("gold_db_name") or "")
    return root / gold_db / f"{gold_db}.sqlite"


def load_native_predictions(run_root: Path, dataset: str, method: str) -> list[dict[str, Any]]:
    pred_path = run_root / dataset / method / "predictions.json"
    if pred_path.exists():
        rows = load_json(pred_path)
    else:
        jsonl_path = run_root / dataset / method / "predictions.jsonl"
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
    return rows


def load_autolink_full_predictions(run_root: Path, dataset: str) -> list[dict[str, Any]]:
    final_dir = run_root / "sql_selection" / "final"
    prefix = DATASETS[dataset]["prefix"]
    rows: list[dict[str, Any]] = []
    for path in sorted(final_dir.glob(f"{prefix}_*/selected.sql")):
        raw_id = path.parent.name
        try:
            index = int(raw_id.split("_")[-1])
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "instance_id": raw_id,
                "predicted_sql": path.read_text(encoding="utf-8"),
                "error": None,
            }
        )
    rows.sort(key=lambda row: row["index"])
    return rows


def eval_one(
    dataset: str,
    pred: dict[str, Any],
    tasks: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    timeout_sec: float,
) -> dict[str, Any]:
    index = int(pred.get("index", -1))
    instance_id = str(pred.get("instance_id") or f"{DATASETS[dataset]['prefix']}_{index:04d}")
    if index < 0 or index >= len(gold_rows) or index >= len(tasks):
        return {"instance_id": instance_id, "index": index, "status": "missing_gold"}
    gold = gold_rows[index]
    task = tasks[index]
    db_path = sqlite_path(dataset, task, gold)
    pred_sql = extract_sql(pred.get("predicted_sql") or pred.get("sql") or pred.get("response"))
    gold_sql = extract_sql(gold["gold_sql"])
    ordered = bool(re.search(r"\border\s+by\b", gold_sql, flags=re.I))
    try:
        pred_rows = run_sql(db_path, pred_sql, timeout_sec=timeout_sec)
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "index": index,
            "status": "pred_exec_error",
            "error": repr(exc),
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "db_path": str(db_path),
        }
    try:
        gold_out = run_sql(db_path, gold_sql, timeout_sec=timeout_sec)
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "index": index,
            "status": "gold_exec_error",
            "error": repr(exc),
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "db_path": str(db_path),
        }
    ok = normalize_result(pred_rows, ordered) == normalize_result(gold_out, ordered)
    return {
        "instance_id": instance_id,
        "index": index,
        "status": "correct" if ok else "wrong",
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "db_path": str(db_path),
    }


def append_summary(row: dict[str, Any]) -> None:
    fields = [
        "task_id",
        "method",
        "schema_linking",
        "dataset",
        "finished",
        "total",
        "execution_correct",
        "execution_accuracy",
        "missing",
        "generation_errors",
        "pred_exec_errors",
        "gold_exec_errors",
        "detail_path",
    ]
    existing: list[dict[str, str]] = []
    if SUMMARY.exists():
        with SUMMARY.open("r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    key = (row["task_id"], row["method"], row["schema_linking"], row["dataset"])
    existing = [
        old
        for old in existing
        if (old.get("task_id"), old.get("method"), old.get("schema_linking"), old.get("dataset")) != key
    ]
    existing.append({field: str(row.get(field, "")) for field in fields})
    existing.sort(key=lambda item: (item["dataset"], item["schema_linking"], item["method"], item["task_id"]))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--source", required=True, choices=["native", "autolink-full"])
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--method", required=True, help="Method name written to the summary table.")
    parser.add_argument(
        "--prediction-method",
        default=None,
        help="Directory method name to read under native outputs. Defaults to --method.",
    )
    parser.add_argument("--schema-linking", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if args.source == "native":
        preds = load_native_predictions(run_root, args.dataset, args.prediction_method or args.method)
    else:
        preds = load_autolink_full_predictions(run_root, args.dataset)

    tasks = load_tasks(args.dataset)
    gold = load_gold(args.dataset)
    details: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(eval_one, args.dataset, pred, tasks, gold, args.timeout_sec) for pred in preds]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            details.append(future.result())
            if done % 100 == 0 or done == len(futures):
                print(f"[{args.task_id}] eval {done}/{len(futures)}", flush=True)
    details.sort(key=lambda row: row.get("index", 0))
    correct = sum(1 for row in details if row["status"] == "correct")
    missing = sum(1 for row in details if row["status"] == "missing_gold")
    pred_err = sum(1 for row in details if row["status"] == "pred_exec_error")
    gold_err = sum(1 for row in details if row["status"] == "gold_exec_error")
    gen_err = sum(1 for pred in preds if pred.get("error"))

    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = DETAIL_DIR / f"{args.method}_{args.schema_linking}_{args.dataset}_details.json"
    if args.source == "autolink-full":
        detail_path = DETAIL_DIR / f"{args.method}_{args.schema_linking}_{args.dataset}_full_details.json"
    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "task_id": args.task_id,
        "method": args.method,
        "schema_linking": args.schema_linking,
        "dataset": args.dataset,
        "finished": len(preds),
        "total": len(gold),
        "execution_correct": correct,
        "execution_accuracy": round(correct / len(gold), 6) if gold else 0,
        "missing": missing,
        "generation_errors": gen_err,
        "pred_exec_errors": pred_err,
        "gold_exec_errors": gold_err,
        "detail_path": str(detail_path),
    }
    append_summary(row)
    print(f"[{args.task_id}] Ex {correct}/{len(gold)} = {row['execution_accuracy']}", flush=True)


if __name__ == "__main__":
    main()
