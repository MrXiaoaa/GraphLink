#!/usr/bin/env bash
set -euo pipefail
: "${GRAPHLINK_EXAMPLES_LITE:?Set GRAPHLINK_EXAMPLES_LITE to examples_lite root}"
mkdir -p outputs/policy_training
${PYTHON:-python3} -m graphlink.policy_training.generate_qa_from_tables \
  --input_path "${GRAPHLINK_EXAMPLES_LITE}" \
  --k_tables "${GRAPHLINK_POLICY_K_TABLES:-3}" \
  --n_questions "${GRAPHLINK_POLICY_N_QUESTIONS:-5}" \
  --output "${GRAPHLINK_POLICY_QA_JSONL:-outputs/policy_training/generated_qa.jsonl}" \
  ${GRAPHLINK_POLICY_EXTRA_ARGS:-}
