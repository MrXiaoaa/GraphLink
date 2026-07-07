#!/usr/bin/env bash
set -euo pipefail
: "${GRAPHLINK_EXAMPLES_LITE:?Set GRAPHLINK_EXAMPLES_LITE to the prepared examples directory}"
: "${GRAPHLINK_DATABASE_GRAPHS_DIR:?Set GRAPHLINK_DATABASE_GRAPHS_DIR to database graphs}"
mkdir -p outputs/schema_linking
${PYTHON:-python3} -m graphlink.schema_linking.run \
  --task lite \
  --db_path "$GRAPHLINK_EXAMPLES_LITE" \
  --linked_json_pth outputs/schema_linking/graphlink_linked.json \
  --database_graphs_dir "$GRAPHLINK_DATABASE_GRAPHS_DIR" \
  --use_semantic_graph_search \
  --use_subquery_decomposition \
  --enable_batch_rerank \
  --batch_size "${GRAPHLINK_BATCH_SIZE:-10}"
