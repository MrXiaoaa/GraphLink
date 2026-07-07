#!/usr/bin/env bash
set -euo pipefail
: "${GRAPHLINK_EXAMPLES_LITE:?Set GRAPHLINK_EXAMPLES_LITE to the prepared examples directory}"
: "${GRAPHLINK_DATABASE_GRAPHS_DIR:?Set GRAPHLINK_DATABASE_GRAPHS_DIR to database graphs}"
${PYTHON:-python3} -m graphlink.sql_generation.build_prompts \
  --source-examples "$GRAPHLINK_EXAMPLES_LITE" \
  --output-examples outputs/examples_lite_graphlink \
  --linking-file outputs/schema_linking/graphlink_linked.json \
  --database-graphs-dir "$GRAPHLINK_DATABASE_GRAPHS_DIR" \
  --dependency-hints \
  --prompt-char-budget "${GRAPHLINK_PROMPT_CHAR_BUDGET:-131072}"
