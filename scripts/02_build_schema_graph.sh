#!/usr/bin/env bash
set -euo pipefail
: "${GRAPHLINK_EXAMPLES_LITE:?Set GRAPHLINK_EXAMPLES_LITE to the prepared examples directory}"
: "${GRAPHLINK_DATABASE_GRAPHS_DIR:=outputs/database_graphs}"
${PYTHON:-python3} -m graphlink.schema_linking.run \
  --task lite \
  --db_path "$GRAPHLINK_EXAMPLES_LITE" \
  --build_db_graphs \
  --db_graphs_output "$GRAPHLINK_DATABASE_GRAPHS_DIR"
