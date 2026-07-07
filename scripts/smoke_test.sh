#!/usr/bin/env bash
set -euo pipefail
${PYTHON:-python3} -m py_compile $(find graphlink -name '*.py' -print)
${PYTHON:-python3} -m graphlink.data.prepare --config configs/paths.example.yaml
${PYTHON:-python3} - <<'PY'
from graphlink.schema_linking.format import normalize_items
items = [{"answer": "Y", "table name": "db.table", "columns": []}, {"answer": "N", "table name": "x"}]
row = normalize_items(items)
assert row["predicted_tables"] == ["db.table"]
print("GraphLink smoke test passed")
PY
