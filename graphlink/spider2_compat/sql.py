import sqlite3
import io
import csv
import os
import glob
import random
import multiprocessing as mp
from queue import Empty
from .utils import hard_cut
from google.cloud import bigquery
from google.oauth2 import service_account
import google.auth.transport.requests
import snowflake.connector
import json
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
import threading
import logging


_BQ_ROTATION_LOCK = threading.Lock()
_BQ_CREDENTIAL_USAGE_COUNT = {}
_BQ_QUOTA_EXHAUSTED = set()
_BQ_CREDENTIAL_CACHE = {"key": None, "paths": []}


def _sqlite_worker(sql_query, save_path, max_len, sqlite_path, result_queue):
    conn = None
    cursor = None
    try:
        if not sqlite_path or not os.path.exists(sqlite_path):
            result_queue.put(("error", f"Database file does not exist: {sqlite_path}"))
            return

        uri = f"file:{sqlite_path}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA query_only = 1")
        conn.execute("PRAGMA temp_store = memory")

        cursor = conn.cursor()
        cursor.execute(sql_query)
        column_info = cursor.description
        rows = []
        current_len = 0
        for row in cursor:
            row_str = str(row)
            rows.append(row)
            if current_len + len(row_str) > max_len:
                break
            current_len += len(row_str)

        if not rows:
            result_queue.put(("ok", "No data found for the specified query.\n"))
            return

        columns = [desc[0] for desc in column_info] if column_info else []
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)
        csv_content = output.getvalue()
        output.close()

        if save_path:
            with open(save_path, 'w', newline='') as f:
                f.write(csv_content)
            result_queue.put(("ok", 0))
        else:
            result_queue.put(("ok", hard_cut(csv_content, max_len)))
    except Exception as e:
        result_queue.put(("error", str(e)))
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


class SqlEnv:
    def __init__(self):
        self.conns = {}
        self.conn_lock = threading.RLock()  # Use RLock for thread safety
        self.logger = logging.getLogger(__name__)
        
        # BigQuery 统一会话管理
        self.bq_client = None
        self.bq_credentials = None
        self.bq_credential_path = None
        self.bq_lock = threading.RLock()  # BigQuery 操作锁
        self.credential_root = os.environ.get("GRAPHLINK_CREDENTIAL_ROOT")

    def _credential_path_for(self, ex_id, filename):
        credential_root = os.environ.get("GRAPHLINK_CREDENTIAL_ROOT") or self.credential_root
        if credential_root and ex_id:
            ex_key = str(ex_id).split("/", 1)[0]
            per_example = os.path.join(credential_root, ex_key, filename)
            if os.path.exists(per_example):
                return per_example
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, filename)

    def _default_bq_credential_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(os.path.join(script_dir, "..", "baselines", "AutoLink", "run", "bigquery_credentials"))

    def _bq_rotation_enabled(self):
        return os.environ.get("GRAPHLINK_BQ_ROTATE_CREDENTIALS", "1").lower() not in {"0", "false", "no"}

    def _load_bq_rotation_credentials(self):
        if not self._bq_rotation_enabled():
            return []
        cred_dir = os.environ.get("GRAPHLINK_BQ_CREDENTIAL_DIR") or self._default_bq_credential_dir()
        include_old = os.environ.get("GRAPHLINK_BQ_INCLUDE_OLD_CREDENTIALS", "0").lower() in {"1", "true", "yes"}
        cache_key = (cred_dir, include_old)
        with _BQ_ROTATION_LOCK:
            if _BQ_CREDENTIAL_CACHE["key"] == cache_key:
                return list(_BQ_CREDENTIAL_CACHE["paths"])
            paths = sorted(glob.glob(os.path.join(cred_dir, "**", "*.json"), recursive=True))
            if not include_old:
                paths = [path for path in paths if "old_promising-era" not in path]
            _BQ_CREDENTIAL_CACHE["key"] = cache_key
            _BQ_CREDENTIAL_CACHE["paths"] = paths
            for path in paths:
                _BQ_CREDENTIAL_USAGE_COUNT.setdefault(path, 0)
            return list(paths)

    def _get_least_used_bq_credential(self, used_paths=None):
        paths = self._load_bq_rotation_credentials()
        if not paths:
            return None
        used = set(used_paths or [])
        with _BQ_ROTATION_LOCK:
            available = [path for path in paths if path not in used and path not in _BQ_QUOTA_EXHAUSTED]
            if not available:
                available = [path for path in paths if path not in used]
            if not available:
                return None
            for path in available:
                _BQ_CREDENTIAL_USAGE_COUNT.setdefault(path, 0)
            min_usage = min(_BQ_CREDENTIAL_USAGE_COUNT[path] for path in available)
            candidates = [path for path in available if _BQ_CREDENTIAL_USAGE_COUNT[path] == min_usage]
            selected = random.choice(candidates)
            _BQ_CREDENTIAL_USAGE_COUNT[selected] += 1
            return selected

    def _mark_bq_credential_quota_exhausted(self, credential_path):
        if not credential_path:
            return
        with _BQ_ROTATION_LOCK:
            _BQ_QUOTA_EXHAUSTED.add(credential_path)

    def _is_bq_quota_error(self, error_str):
        text = error_str.lower()
        return "quota" in text or "quotaexceeded" in text or "free query bytes scanned" in text

    def _is_bq_auth_error(self, error_str):
        text = error_str.lower()
        return "401" in text or "unauthenticated" in text or "credentials_missing" in text or "invalid_grant" in text

    def get_rows(self, cursor, max_len):
        rows = []
        current_len = 0
        for row in cursor:
            row_str = str(row)
            rows.append(row)
            if current_len + len(row_str) > max_len:
                break
            current_len += len(row_str)
        return rows

    def get_csv(self, columns, rows):
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)
        csv_content = output.getvalue()
        output.close()
        return csv_content
    
    def _init_bq_client(self, force_refresh=False, ex_id=None, credential_path=None):
        """
        初始化或刷新 BigQuery client（统一会话管理）
        
        Args:
            force_refresh: 是否强制刷新凭证（用于 401 错误重试）
        
        Returns:
            BigQuery client 实例
        """
        with self.bq_lock:
            import os
            
            credential_path = credential_path or self._credential_path_for(ex_id, "bigquery_credential.json")

            # 如果已有 client 且凭证路径一致，不需要强制刷新，直接返回
            if self.bq_client is not None and not force_refresh and self.bq_credential_path == credential_path:
                return self.bq_client

            # 如果切换了 per-example credential，关闭旧 client 后重新初始化
            if self.bq_client is not None and self.bq_credential_path != credential_path:
                try:
                    self.bq_client.close()
                except Exception:
                    pass
                self.bq_client = None
                self.bq_credentials = None
            self.bq_credential_path = credential_path
            
            try:
                # 加载凭证
                SCOPES = ["https://www.googleapis.com/auth/bigquery"]
                self.bq_credentials = service_account.Credentials.from_service_account_file(
                    self.bq_credential_path,
                    scopes=SCOPES
                )
                
                # 创建 client（统一会话）
                self.bq_client = bigquery.Client(
                    credentials=self.bq_credentials,
                    project=self.bq_credentials.project_id
                )
                
                self.logger.info(f"BigQuery client {'refreshed' if force_refresh else 'initialized'} successfully")
                return self.bq_client
                
            except Exception as e:
                self.logger.error(f"Failed to initialize BigQuery client: {e}")
                self.bq_client = None
                self.bq_credentials = None
                raise
    
    def _refresh_bq_credentials(self, ex_id=None):
        """
        刷新 BigQuery 凭证（用于 401 错误处理）
        """
        self.logger.info("Refreshing BigQuery credentials due to authentication error...")
        
        try:
            # 如果有现有凭证，尝试刷新
            if self.bq_credentials is not None:
                try:
                    # 刷新凭证（service account 凭证会自动刷新 token）
                    self.bq_credentials.refresh(google.auth.transport.requests.Request())
                    self.logger.info("BigQuery credentials refreshed successfully")
                    return True
                except Exception as refresh_error:
                    self.logger.warning(f"Failed to refresh existing credentials: {refresh_error}")
            
            # 如果刷新失败或没有现有凭证，重新初始化整个 client
            self._init_bq_client(force_refresh=True, ex_id=ex_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to refresh BigQuery credentials: {e}")
            return False

    def start_db_sqlite(self, sqlite_path):
        with self.conn_lock:
            if sqlite_path not in self.conns:
                try:
                    # Check if file exists before attempting to connect
                    import os
                    if not os.path.exists(sqlite_path):
                        self.logger.error(f"Database file does not exist: {sqlite_path}")
                        return False
                    
                    uri = f"file:{sqlite_path}?mode=ro"
                    # Add timeout and other connection parameters for stability
                    conn = sqlite3.connect(
                        uri, 
                        uri=True, 
                        check_same_thread=False,
                        timeout=30.0,  # 30 second timeout
                        isolation_level=None  # Autocommit mode for read-only
                    )
                    # Set pragmas for better performance and safety
                    conn.execute("PRAGMA query_only = 1")  # Read-only mode
                    conn.execute("PRAGMA temp_store = memory")  # Use memory for temp storage
                    
                    self.conns[sqlite_path] = conn
                    self.logger.info(f"Successfully connected to database: {sqlite_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to connect to database {sqlite_path}: {e}")
                    return False
            return True

    def start_db_sf(self, ex_id):
        with self.conn_lock:
            if ex_id not in self.conns.keys():
                try:
                    import os
                    credential_path = self._credential_path_for(ex_id, "snowflake_credential.json")
                    snowflake_credential = json.load(open(credential_path))
                    conn = snowflake.connector.connect(**snowflake_credential)
                    
                    # Set warehouse if not specified in credentials
                    if 'warehouse' not in snowflake_credential:
                        try:
                            cursor = conn.cursor()
                            cursor.execute("USE WAREHOUSE COMPUTE_WH_PARTICIPANT")
                            cursor.close()
                            self.logger.info(f"Set warehouse COMPUTE_WH_PARTICIPANT for {ex_id}")
                        except Exception as we:
                            self.logger.warning(f"Failed to set warehouse: {we}")
                    
                    self.conns[ex_id] = conn
                except Exception as e:
                    self.logger.error(f"Failed to connect to Snowflake {ex_id}: {e}")
                    return False
            return True

    def close_db(self):
        with self.conn_lock:
            self.logger.info(f"Closing {len(self.conns)} database connections")
            for key, conn in list(self.conns.items()):
                try:
                    if conn:
                        conn.close()
                        self.logger.debug(f"Connection {key} closed.")
                        del self.conns[key]
                except Exception as e:
                    self.logger.error(f"Error closing DB connection for {key}: {e}")
        
        # 清理 BigQuery client
        with self.bq_lock:
            if self.bq_client is not None:
                try:
                    self.bq_client.close()
                    self.logger.info("BigQuery client closed")
                except Exception as e:
                    self.logger.error(f"Error closing BigQuery client: {e}")
                finally:
                    self.bq_client = None
                    self.bq_credentials = None

    def _get_connection(self, sqlite_path):
        """Get a thread-safe database connection"""
        with self.conn_lock:
            if sqlite_path not in self.conns:
                if not self.start_db_sqlite(sqlite_path):
                    return None
            return self.conns.get(sqlite_path)

    def exec_sql_sqlite(self, sql_query, save_path=None, max_len=30000, sqlite_path=None):
        conn = self._get_connection(sqlite_path)
        if not conn:
            return "##ERROR## Failed to get database connection."

        cursor = conn.cursor()
        try:
            cursor.execute(sql_query)
            column_info = cursor.description
            rows = self.get_rows(cursor, max_len)
            columns = [desc[0] for desc in column_info]
        except Exception as e:
            return "##ERROR##"+str(e)
        finally:
            try:
                cursor.close()
            except Exception as e:
                print("Failed to close cursor:", e)

        if not rows:
            return "No data found for the specified query.\n"
        else:
            csv_content = self.get_csv(columns, rows)
            if save_path:
                with open(save_path, 'w', newline='') as f:
                    f.write(csv_content)
                return 0
            else:
                return hard_cut(csv_content, max_len)
            
    def exec_sql_sf(self, sql_query, save_path, max_len, ex_id):
        with self.conn_lock:
            if ex_id not in self.conns.keys():
                if not self.start_db_sf(ex_id):
                    return "##ERROR## Failed to connect to Snowflake database."
            conn = self.conns[ex_id]

        with conn.cursor() as cursor:
            try:
                cursor.execute(sql_query)
                column_info = cursor.description
                rows = self.get_rows(cursor, max_len)
                columns = [desc[0] for desc in column_info]
            except Exception as e:
                return "##ERROR##"+str(e)

        if not rows:
            return "No data found for the specified query.\n"
        else:
            csv_content = self.get_csv(columns, rows)
            if save_path:
                with open(save_path, 'w', newline='') as f:
                    f.write(csv_content)
                return 0
            else:
                return hard_cut(csv_content, max_len)

    def exec_sql_bq(self, sql_query, save_path, max_len, max_retries=1, timeout=60, ex_id=None):
        """Execute BigQuery SQL with AutoLink-style credential rotation on quota errors."""
        used_credentials = []
        rotation_paths = self._load_bq_rotation_credentials()
        max_attempts = len(rotation_paths) if rotation_paths else max_retries + 1
        if max_attempts <= 0:
            max_attempts = 1
        last_error = None

        for attempt in range(max_attempts):
            credential_path = None
            if rotation_paths:
                credential_path = self._get_least_used_bq_credential(used_credentials)
                if credential_path is None:
                    break
                used_credentials.append(credential_path)

            try:
                client = self._init_bq_client(
                    force_refresh=attempt > 0,
                    ex_id=ex_id,
                    credential_path=credential_path,
                )
                label = os.path.basename(credential_path) if credential_path else os.path.basename(self.bq_credential_path or "")
                self.logger.debug(
                    f"Executing BigQuery query (attempt {attempt + 1}/{max_attempts}, credential={label}, timeout={timeout}s)"
                )
                query_job = client.query(sql_query)
                try:
                    result_iterator = query_job.result(timeout=timeout)
                except Exception as timeout_error:
                    error_str = str(timeout_error)
                    if "timeout" in error_str.lower() or "timed out" in error_str.lower():
                        self.logger.warning(f"Query execution timeout after {timeout}s")
                        return f"##ERROR##QUERY_TOO_SLOW: Query execution exceeded {timeout} seconds timeout. Consider simplifying the query or reducing the data range."
                    raise

                rows = []
                current_len = 0
                for row in result_iterator:
                    if current_len > max_len:
                        break
                    current_len += len(str(dict(row)))
                    rows.append(dict(row))

                df = pd.DataFrame(rows)
                if df.empty:
                    return "No data found for the specified query.\n"
                if save_path:
                    df.to_csv(f"{save_path}", index=False)
                    return 0
                return hard_cut(df.to_csv(index=False), max_len)

            except Exception as e:
                error_str = str(e)
                last_error = error_str
                if self._is_bq_quota_error(error_str):
                    self.logger.warning(
                        f"BigQuery quota error on credential {os.path.basename(credential_path or self.bq_credential_path or '')}: {error_str[:200]}"
                    )
                    self._mark_bq_credential_quota_exhausted(credential_path or self.bq_credential_path)
                    if rotation_paths and attempt + 1 < max_attempts:
                        continue
                    return "##ERROR##" + error_str

                if self._is_bq_auth_error(error_str):
                    self.logger.warning(
                        f"BigQuery authentication error on credential {os.path.basename(credential_path or self.bq_credential_path or '')}: {error_str[:200]}"
                    )
                    if rotation_paths and attempt + 1 < max_attempts:
                        continue
                    if not rotation_paths and attempt < max_retries and self._refresh_bq_credentials(ex_id=ex_id):
                        continue
                    return "##ERROR##" + error_str

                self.logger.error(f"BigQuery query failed: {error_str[:200]}")
                return "##ERROR##" + error_str

        return "##ERROR##" + (last_error or "No usable BigQuery credential available")

    def _get_sqlite_timeout(self, timeout):
        if timeout is not None:
            return timeout
        raw_timeout = os.environ.get("GRAPHLINK_SQLITE_TIMEOUT") or os.environ.get("GRAPHLINK_SQL_TIMEOUT") or "60"
        try:
            return max(1, int(raw_timeout))
        except ValueError:
            self.logger.warning(f"Invalid GRAPHLINK_SQLITE_TIMEOUT={raw_timeout!r}; using 60s")
            return 60

    def _get_sqlite_mp_context(self):
        context_name = os.environ.get("GRAPHLINK_SQLITE_MP_CONTEXT", "fork" if hasattr(os, "fork") else "spawn")
        try:
            return mp.get_context(context_name)
        except ValueError:
            fallback = "fork" if hasattr(os, "fork") else "spawn"
            self.logger.warning(f"Invalid GRAPHLINK_SQLITE_MP_CONTEXT={context_name!r}; using {fallback}")
            return mp.get_context(fallback)

    def execute_sql_api(self, sql_query, ex_id, save_path=None, api="sqlite", max_len=30000, sqlite_path=None, timeout=None):
        try:
            if api == "bigquery":
                # 对BigQuery使用较短的默认超时（60秒），避免长时间等待
                bq_timeout = min(timeout, 60) if timeout else 60
                result = self.exec_sql_bq(sql_query, save_path, max_len, timeout=bq_timeout, ex_id=ex_id)
            elif api == "snowflake":
                result = self.exec_sql_sf(sql_query, save_path, max_len, ex_id)
            elif api == "sqlite":
                sqlite_timeout = self._get_sqlite_timeout(timeout)
                result = self.execute_sqlite_with_timeout(sql_query, save_path, max_len, sqlite_path, sqlite_timeout)
            else:
                return {"status": "error", "error_msg": f"##ERROR## Unsupported API: {api}"}

            if "##ERROR##" in str(result):
                return {"status": "error", "error_msg": str(result)}
            else:
                return str(result)
        except Exception as e:
            self.logger.error(f"Error in execute_sql_api: {e}")
            return {"status": "error", "error_msg": f"##ERROR## {str(e)}"}

    def execute_sqlite_with_timeout(self, sql_query, save_path, max_len, sqlite_path, timeout=300):
        ctx = self._get_sqlite_mp_context()
        result_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_sqlite_worker,
            args=(sql_query, save_path, max_len, sqlite_path, result_queue),
        )
        process.daemon = True
        process.start()
        process.join(timeout)
        sql_preview = sql_query[:100] + "..." if len(sql_query) > 100 else sql_query

        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            error_msg = f"##ERROR##Query execution timed out after {timeout}s. SQL preview: {sql_preview}"
            self.logger.warning(error_msg)
            return error_msg

        try:
            status, payload = result_queue.get_nowait()
        except Empty:
            if process.exitcode == 0:
                error_msg = f"##ERROR##SQLite worker returned no result. SQL preview: {sql_preview}"
            else:
                error_msg = f"##ERROR##SQLite worker exited with code {process.exitcode}. SQL preview: {sql_preview}"
            self.logger.warning(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"##ERROR##Query execution failed: {str(e)}. SQL preview: {sql_preview}"
            self.logger.warning(error_msg)
            return error_msg

        if status == "error":
            return f"##ERROR##{payload}"
        return str(payload)
