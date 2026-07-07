#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import snowflake.connector


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PARENT = REPO_ROOT / "graphlink" / "spider2_compat"
DATA_FILE = REPO_ROOT / "spider2-lite" / "resource" / "spider2-lite.jsonl"
SQLITE_ROOT = REPO_ROOT / "spider2-lite" / "resource" / "databasesUnified"
GOLD_EXEC_DIR = REPO_ROOT / "spider2-lite" / "evaluation_suite" / "gold" / "exec_result"
DETAIL_DIR = REPO_ROOT / "outputs" / "eval_details"
RESULT_ROOT = DETAIL_DIR / "spider2_accuracy_results"
SUMMARY = REPO_ROOT / "outputs" / "spider2lite_accuracy_summary.csv"

from graphlink.spider2_compat.eval import evaluate_spider2sql  # noqa: E402
from graphlink.spider2_compat.sql import SqlEnv  # noqa: E402


class PerExampleCredentialSqlEnv(SqlEnv):
    def __init__(self) -> None:
        super().__init__()
        self.sf_credential_paths: dict[str, str] = {}

    def start_db_sf(self, ex_id):  # type: ignore[no-untyped-def]
        credential_path = self.sf_credential_paths.get(ex_id)
        if not credential_path:
            return super().start_db_sf(ex_id)
        with self.conn_lock:
            if ex_id not in self.conns:
                try:
                    cfg = load_json(Path(credential_path))
                    conn = snowflake.connector.connect(**cfg)
                    if "warehouse" not in cfg:
                        try:
                            cur = conn.cursor()
                            cur.execute("USE WAREHOUSE COMPUTE_WH_PARTICIPANT")
                            cur.close()
                        except Exception as warehouse_error:
                            self.logger.warning(f"Failed to set warehouse: {warehouse_error}")
                    self.conns[ex_id] = conn
                except Exception as exc:
                    self.logger.error(f"Failed to connect to Snowflake {ex_id}: {exc}")
                    return False
            return True


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


def load_compile_details(path: Path) -> dict[str, dict[str, Any]]:
    details = load_json(path)
    return {str(row.get("instance_id")): row for row in details}


def backend_for(instance_id: str) -> str:
    if instance_id.startswith(("bq", "ga")):
        return "bigquery"
    if instance_id.startswith("sf"):
        return "snowflake"
    return "sqlite"


def find_credential(instance_id: str, backend: str) -> str:
    name = {
        "bigquery": "bigquery_credential.json",
        "snowflake": "snowflake_credential.json",
    }.get(backend)
    if not name:
        return ""
    credential_root = Path(os.environ.get("GRAPHLINK_CREDENTIAL_EXAMPLES_ROOT", str(REPO_ROOT / "data" / "examples_lite")))
    for base in (credential_root,):
        path = base / instance_id / name
        if path.exists():
            return str(path)
    return ""


def sqlite_path(db_name: str) -> str:
    direct = SQLITE_ROOT / db_name / f"{db_name}.sqlite"
    if direct.exists():
        return str(direct)
    db_dir = SQLITE_ROOT / db_name
    if db_dir.exists():
        files = sorted(db_dir.glob("*.sqlite"))
        if files:
            return str(files[0])
    lower = db_name.lower()
    for path in SQLITE_ROOT.glob("*/*.sqlite"):
        if path.stem.lower() == lower or path.parent.name.lower() == lower:
            return str(path)
    return ""


def configure_credentials(sql_env: SqlEnv, row: dict[str, Any], credential_mode: str) -> None:
    if credential_mode != "per-example":
        return
    backend = row.get("backend")
    credential_path = row.get("credential_path")
    if not credential_path:
        return
    if backend == "bigquery":
        if sql_env.bq_client is not None:
            try:
                sql_env.bq_client.close()
            except Exception:
                pass
        sql_env.bq_client = None
        sql_env.bq_credentials = None
        sql_env.bq_credential_path = credential_path
    elif backend == "snowflake" and hasattr(sql_env, "sf_credential_paths"):
        sql_env.sf_credential_paths[row["instance_id"]] = credential_path  # type: ignore[attr-defined]


def score_csv(csv_path: str, instance_id: str) -> tuple[int, str]:
    if not csv_path:
        return 0, ""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        score = evaluate_spider2sql(str(GOLD_EXEC_DIR), csv_path, instance_id, task="lite")
    return int(score), buf.getvalue().strip()


def build_candidate(
    pred: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    compile_details: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    instance_id = str(pred.get("instance_id") or "")
    item = metadata.get(instance_id, {})
    compile_row = compile_details.get(instance_id, {})
    backend = str(compile_row.get("backend") or backend_for(instance_id))
    db_name = str(compile_row.get("db") or item.get("db") or "")
    credential_path = str(compile_row.get("credential_path") or find_credential(instance_id, backend))
    db_path = str(compile_row.get("db_path") or "")
    if backend == "sqlite" and not db_path and db_name:
        db_path = sqlite_path(db_name)
    return {
        "instance_id": instance_id,
        "backend": backend,
        "db": db_name,
        "compile_status": str(compile_row.get("status") or "missing_compile_detail"),
        "sql": str(compile_row.get("sql") or ""),
        "credential_path": credential_path,
        "sqlite_path": db_path,
    }


def execute_candidate(
    sql_env: SqlEnv,
    row: dict[str, Any],
    result_dir: Path,
    timeout_sec: int,
    max_len: int,
    credential_mode: str,
) -> dict[str, Any]:
    instance_id = row["instance_id"]
    out_csv = result_dir / instance_id / "result.csv"
    base_result = {
        "instance_id": instance_id,
        "backend": row.get("backend", ""),
        "db": row.get("db", ""),
        "compile_status": row.get("compile_status", ""),
        "execute_status": "",
        "output_csv": "",
        "output_bytes": 0,
        "score": 0,
        "score_message": "",
        "elapsed_sec": 0,
        "error_type": "",
        "error": "",
    }

    compile_status = row.get("compile_status")
    if compile_status != "compile_ok":
        status = "no_sql" if compile_status == "no_sql" else "skipped_compile_error"
        return {**base_result, "execute_status": status}
    if not row.get("sql"):
        return {**base_result, "execute_status": "no_sql"}
    if row.get("backend") in {"bigquery", "snowflake"} and not row.get("credential_path"):
        return {**base_result, "execute_status": "missing_credential"}
    if row.get("backend") == "sqlite" and not row.get("sqlite_path"):
        return {**base_result, "execute_status": "missing_sqlite_db"}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        configure_credentials(sql_env, row, credential_mode)
        result = sql_env.execute_sql_api(
            row["sql"],
            instance_id,
            save_path=str(out_csv),
            api=row["backend"],
            max_len=max_len,
            sqlite_path=row.get("sqlite_path"),
            timeout=timeout_sec,
        )
        elapsed = round(time.time() - start, 3)
        output_exists = out_csv.exists()
        output_csv = str(out_csv) if output_exists else ""
        output_bytes = out_csv.stat().st_size if output_exists else 0
        if isinstance(result, dict):
            return {
                **base_result,
                "execute_status": "execute_error",
                "elapsed_sec": elapsed,
                "output_csv": output_csv,
                "output_bytes": output_bytes,
                "error": str(result.get("error_msg") or result)[:1200],
            }
        execute_status = "execute_ok" if str(result) == "0" else "execute_empty"
        score, score_message = score_csv(output_csv, instance_id) if output_exists else (0, "")
        return {
            **base_result,
            "execute_status": execute_status,
            "output_csv": output_csv,
            "output_bytes": output_bytes,
            "score": score,
            "score_message": score_message[:500],
            "elapsed_sec": elapsed,
        }
    except Exception as exc:
        return {
            **base_result,
            "execute_status": "execute_error",
            "elapsed_sec": round(time.time() - start, 3),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
        }


def write_detail(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_existing_detail(path: Path) -> list[dict[str, Any]]:
    try:
        rows = load_json(path)
        if isinstance(rows, list):
            return rows
    except Exception as exc:
        print(f"Warning: ignoring unreadable resume detail {path}: {exc}", file=sys.stderr, flush=True)
    return []


def write_summary(row: dict[str, Any]) -> None:
    fields = [
        "task_id",
        "method",
        "schema_linking",
        "total",
        "finished",
        "correct",
        "accuracy",
        "execute_ok",
        "execute_empty",
        "execute_error",
        "skipped_compile_error",
        "no_sql",
        "missing_credential",
        "missing_sqlite_db",
        "detail_path",
        "result_dir",
    ]
    existing: list[dict[str, str]] = []
    if SUMMARY.exists():
        existing = list(csv.DictReader(SUMMARY.open("r", encoding="utf-8")))
    key = (row["task_id"], row["method"], row["schema_linking"])
    existing = [
        old
        for old in existing
        if (old.get("task_id"), old.get("method"), old.get("schema_linking")) != key
    ]
    existing.append({field: str(row.get(field, "")) for field in fields})
    existing.sort(key=lambda item: (item["method"], item["schema_linking"], item["task_id"]))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def summarize(
    task_id: str,
    method: str,
    schema_linking: str,
    total: int,
    rows: list[dict[str, Any]],
    detail_path: Path,
    result_dir: Path,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("execute_status", "")
        counts[status] = counts.get(status, 0) + 1
    correct = sum(int(row.get("score") or 0) for row in rows)
    summary = {
        "task_id": task_id,
        "method": method,
        "schema_linking": schema_linking,
        "total": total,
        "finished": len(rows),
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0,
        "execute_ok": counts.get("execute_ok", 0),
        "execute_empty": counts.get("execute_empty", 0),
        "execute_error": counts.get("execute_error", 0),
        "skipped_compile_error": counts.get("skipped_compile_error", 0),
        "no_sql": counts.get("no_sql", 0),
        "missing_credential": counts.get("missing_credential", 0),
        "missing_sqlite_db": counts.get("missing_sqlite_db", 0),
        "detail_path": str(detail_path),
        "result_dir": str(result_dir),
    }
    write_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--method", required=True, choices=["CHESS", "KaSLA"])
    parser.add_argument("--schema-linking", required=True)
    parser.add_argument("--compile-detail", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-len", type=int, default=30000)
    parser.add_argument("--credential-mode", choices=["examples-root", "per-example"], default="per-example")
    parser.add_argument("--include-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata()
    predictions = load_predictions(Path(args.run_root), args.method)
    compile_details = load_compile_details(Path(args.compile_detail))

    if args.include_ids:
        keep = set(args.include_ids)
        predictions = [row for row in predictions if str(row.get("instance_id")) in keep]
    if args.limit is not None:
        predictions = predictions[: args.limit]

    candidates = [build_candidate(pred, metadata, compile_details) for pred in predictions]
    expected_total = len(metadata) if args.include_ids is None and args.limit is None else len(candidates)
    result_dir = RESULT_ROOT / args.task_id
    detail_path = DETAIL_DIR / f"{args.task_id}_details.json"

    rows: list[dict[str, Any]] = []
    done: dict[str, dict[str, Any]] = {}
    if args.resume and detail_path.exists():
        rows = load_existing_detail(detail_path)
        done = {str(row.get("instance_id")): row for row in rows if row.get("execute_status")}

    sql_env: SqlEnv
    if args.credential_mode == "per-example":
        sql_env = PerExampleCredentialSqlEnv()
    else:
        sql_env = SqlEnv()

    try:
        rows_by_id = {str(row.get("instance_id")): row for row in rows}
        for index, candidate in enumerate(candidates, start=1):
            instance_id = candidate["instance_id"]
            if instance_id in done:
                print(f"[{index}/{len(candidates)}] resume {candidate['backend']} {instance_id}", flush=True)
                continue
            print(f"[{index}/{len(candidates)}] {candidate['backend']} {instance_id}", flush=True)
            result = execute_candidate(
                sql_env,
                candidate,
                result_dir,
                timeout_sec=args.timeout_sec,
                max_len=args.max_len,
                credential_mode=args.credential_mode,
            )
            rows_by_id[instance_id] = result
            ordered = [rows_by_id[candidate_row["instance_id"]] for candidate_row in candidates if candidate_row["instance_id"] in rows_by_id]
            write_detail(detail_path, ordered)
        final_rows = [rows_by_id[candidate["instance_id"]] for candidate in candidates if candidate["instance_id"] in rows_by_id]
    finally:
        sql_env.close_db()

    summary = summarize(
        args.task_id,
        args.method,
        args.schema_linking,
        expected_total,
        final_rows,
        detail_path,
        result_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
