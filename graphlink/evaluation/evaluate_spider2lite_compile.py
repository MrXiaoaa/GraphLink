#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from google.cloud import bigquery
from google.oauth2 import service_account
import snowflake.connector


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "spider2-lite" / "resource" / "spider2-lite.jsonl"
EXAMPLES_PARENT = Path(os.environ.get("GRAPHLINK_CREDENTIAL_EXAMPLES_ROOT", str(REPO_ROOT / "data" / "examples_lite"))).parent
SQLITE_ROOT = REPO_ROOT / "spider2-lite" / "resource" / "databasesUnified"
SUMMARY = REPO_ROOT / "outputs" / "spider2lite_online_compile_summary.csv"
DETAIL_DIR = REPO_ROOT / "outputs" / "eval_details"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                rows[str(item["instance_id"])] = item
    return rows


def load_predictions(run_root: Path, method: str) -> list[dict[str, Any]]:
    pred_path = run_root / "spider2lite" / method / "predictions.json"
    if pred_path.exists():
        rows = load_json(pred_path)
    else:
        jsonl_path = run_root / "spider2lite" / method / "predictions.jsonl"
        rows = [json.loads(line) for line in jsonl_path.open("r", encoding="utf-8") if line.strip()]
    rows.sort(key=lambda row: int(row.get("index", 10**9)))
    return rows


def extract_sql(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if not text:
        return ""
    final = re.search(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", text, flags=re.I | re.S)
    if final:
        text = final.group(1).strip()
    fence_matches = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence_matches:
        text = fence_matches[-1].strip()
    tagged = re.search(r"<sql>\s*(.*?)\s*</sql>", text, flags=re.I | re.S)
    if tagged:
        text = tagged.group(1).strip()

    starts_with_sql = re.match(r"^\s*(WITH|SELECT)\b", text, flags=re.I) is not None
    matches = list(re.finditer(r"\b(WITH|SELECT)\b", text, flags=re.I))
    if not matches:
        return ""
    if not starts_with_sql:
        text = text[matches[-1].start() :].strip()
    text = re.split(r"\n\s*(?:Explanation|Reasoning|Notes?)\s*:", text, flags=re.I)[0].strip()
    text = re.split(r"\n\s*[*-]\s+\*\*(?:Sub-question|Analysis|Step)", text, flags=re.I)[0].strip()
    if ";" in text:
        text = text[: text.find(";") + 1]
    return text.rstrip().rstrip(";")


def backend_for(instance_id: str) -> str:
    if instance_id.startswith(("bq", "ga")):
        return "bigquery"
    if instance_id.startswith("sf"):
        return "snowflake"
    return "sqlite"


def credential_path(instance_id: str, backend: str) -> Path | None:
    names = {
        "bigquery": "bigquery_credential.json",
        "snowflake": "snowflake_credential.json",
    }
    if backend not in names:
        return None
    for base in (EXAMPLES_PARENT / "examples_lite", EXAMPLES_PARENT / "examples_lite_test"):
        path = base / instance_id / names[backend]
        if path.exists():
            return path
    return None


def sqlite_path(db_name: str) -> Path | None:
    direct = SQLITE_ROOT / db_name / f"{db_name}.sqlite"
    if direct.exists():
        return direct
    files = sorted((SQLITE_ROOT / db_name).glob("*.sqlite")) if (SQLITE_ROOT / db_name).exists() else []
    if files:
        return files[0]
    lower = db_name.lower()
    for path in SQLITE_ROOT.glob("*/*.sqlite"):
        if path.stem.lower() == lower or path.parent.name.lower() == lower:
            return path
    return None


def check_bigquery(sql: str, cred_path: Path, timeout_sec: float) -> tuple[str, dict[str, Any]]:
    creds = service_account.Credentials.from_service_account_file(str(cred_path))
    client = bigquery.Client(credentials=creds, project=creds.project_id)
    config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = client.query(sql, job_config=config, timeout=timeout_sec)
    return "compile_ok", {"estimated_bytes": int(job.total_bytes_processed or 0)}


def check_snowflake(sql: str, cred_path: Path, timeout_sec: float) -> tuple[str, dict[str, Any]]:
    cfg = load_json(cred_path)
    cfg.setdefault("login_timeout", timeout_sec)
    cfg.setdefault("network_timeout", timeout_sec)
    cfg.setdefault("socket_timeout", timeout_sec)
    conn = snowflake.connector.connect(**cfg)
    try:
        cur = conn.cursor()
        try:
            cur.execute("EXPLAIN USING TEXT " + sql, timeout=timeout_sec)
            rows = cur.fetchmany(1)
            return "compile_ok", {"explain_rows": len(rows)}
        finally:
            cur.close()
    finally:
        conn.close()


def check_sqlite(sql: str, db_name: str) -> tuple[str, dict[str, Any]]:
    db_path = sqlite_path(db_name)
    if db_path is None:
        return "missing_sqlite_db", {}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("EXPLAIN QUERY PLAN " + sql)
        rows = cur.fetchmany(1)
        return "compile_ok", {"db_path": str(db_path), "explain_rows": len(rows)}
    finally:
        conn.close()


def eval_one(pred: dict[str, Any], metadata: dict[str, dict[str, Any]], timeout_sec: float) -> dict[str, Any]:
    instance_id = str(pred.get("instance_id") or "")
    item = metadata.get(instance_id)
    if item is None:
        return {"instance_id": instance_id, "backend": "unknown", "status": "missing_metadata"}
    backend = backend_for(instance_id)
    db_name = str(item["db"])
    sql = extract_sql(pred.get("predicted_sql") or pred.get("sql") or pred.get("response"))
    if not sql:
        return {"instance_id": instance_id, "backend": backend, "db": db_name, "status": "no_sql"}
    start = time.time()
    try:
        if backend == "bigquery":
            cred = credential_path(instance_id, backend)
            if cred is None:
                return {"instance_id": instance_id, "backend": backend, "db": db_name, "status": "missing_credential"}
            status, extra = check_bigquery(sql, cred, timeout_sec)
            extra["credential_path"] = str(cred)
        elif backend == "snowflake":
            cred = credential_path(instance_id, backend)
            if cred is None:
                return {"instance_id": instance_id, "backend": backend, "db": db_name, "status": "missing_credential"}
            status, extra = check_snowflake(sql, cred, timeout_sec)
            extra["credential_path"] = str(cred)
        else:
            status, extra = check_sqlite(sql, db_name)
        return {
            "instance_id": instance_id,
            "backend": backend,
            "db": db_name,
            "status": status,
            "elapsed_sec": round(time.time() - start, 3),
            "sql": sql,
            **extra,
        }
    except Exception as exc:
        return {
            "instance_id": instance_id,
            "backend": backend,
            "db": db_name,
            "status": "compile_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
            "elapsed_sec": round(time.time() - start, 3),
            "sql": sql,
        }


def write_summary(row: dict[str, Any]) -> None:
    fields = [
        "task_id",
        "method",
        "schema_linking",
        "dataset",
        "finished",
        "total",
        "compile_ok",
        "compile_rate",
        "bigquery_ok",
        "snowflake_ok",
        "sqlite_ok",
        "no_sql",
        "compile_errors",
        "missing_credentials",
        "missing_sqlite_db",
        "detail_path",
    ]
    existing: list[dict[str, str]] = []
    if SUMMARY.exists():
        existing = list(csv.DictReader(SUMMARY.open("r", encoding="utf-8")))
    key = (row["task_id"], row["method"], row["schema_linking"], row["dataset"])
    existing = [
        old
        for old in existing
        if (old.get("task_id"), old.get("method"), old.get("schema_linking"), old.get("dataset")) != key
    ]
    existing.append({field: str(row.get(field, "")) for field in fields})
    existing.sort(key=lambda item: (item["schema_linking"], item["method"], item["task_id"]))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def main() -> None:
    for proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(proxy_key, None)
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--method", required=True, choices=["AutoLink", "CHESS", "KaSLA"])
    parser.add_argument("--schema-linking", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-ids", nargs="+", default=None)
    args = parser.parse_args()

    metadata = load_metadata()
    predictions = load_predictions(Path(args.run_root), args.method)
    if args.include_ids:
        keep = set(args.include_ids)
        predictions = [row for row in predictions if str(row.get("instance_id")) in keep]
    if args.limit is not None:
        predictions = predictions[: args.limit]

    details: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(eval_one, pred, metadata, args.timeout_sec) for pred in predictions]
        for future in concurrent.futures.as_completed(futures):
            details.append(future.result())
            if len(details) % 25 == 0 or len(details) == len(predictions):
                print(f"[{args.task_id or args.run_root}] compile {len(details)}/{len(predictions)}", flush=True)
    details.sort(key=lambda row: row.get("instance_id", ""))

    counts: dict[str, int] = {}
    backend_ok = {"bigquery": 0, "snowflake": 0, "sqlite": 0}
    for row in details:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row["status"] == "compile_ok":
            backend_ok[row["backend"]] = backend_ok.get(row["backend"], 0) + 1

    task_id = args.task_id or str(args.run_root)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = DETAIL_DIR / f"{args.method}_{args.schema_linking}_spider2lite_online_compile_details.json"
    detail_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    finished = len(details)
    total = len(metadata) if args.limit is None and not args.include_ids else finished
    compile_ok = counts.get("compile_ok", 0)
    summary = {
        "task_id": task_id,
        "method": args.method,
        "schema_linking": args.schema_linking,
        "dataset": "spider2lite_online_compile",
        "finished": finished,
        "total": total,
        "compile_ok": compile_ok,
        "compile_rate": round(compile_ok / total, 6) if total else 0,
        "bigquery_ok": backend_ok.get("bigquery", 0),
        "snowflake_ok": backend_ok.get("snowflake", 0),
        "sqlite_ok": backend_ok.get("sqlite", 0),
        "no_sql": counts.get("no_sql", 0),
        "compile_errors": counts.get("compile_error", 0),
        "missing_credentials": counts.get("missing_credential", 0),
        "missing_sqlite_db": counts.get("missing_sqlite_db", 0),
        "detail_path": str(detail_path),
    }
    write_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
