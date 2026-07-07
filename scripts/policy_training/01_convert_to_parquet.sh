#!/usr/bin/env bash
set -euo pipefail
: "${GRAPHLINK_EXAMPLES_LITE:?Set GRAPHLINK_EXAMPLES_LITE to examples_lite root}"
${PYTHON:-python3} -m graphlink.policy_training.convert2parquet \
  --input_jsonl "${GRAPHLINK_POLICY_QA_JSONL:-outputs/policy_training/generated_qa.jsonl}" \
  --examples_root "${GRAPHLINK_EXAMPLES_LITE}" \
  --output_parquet "${GRAPHLINK_POLICY_PARQUET:-outputs/policy_training/all.parquet}" \
  --only_success \
  --require_gold_data \
  ${GRAPHLINK_POLICY_CONVERT_EXTRA_ARGS:-}
