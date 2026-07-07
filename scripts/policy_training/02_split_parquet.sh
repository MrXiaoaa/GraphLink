#!/usr/bin/env bash
set -euo pipefail
${PYTHON:-python3} -m graphlink.policy_training.split_parquet \
  --input_parquet "${GRAPHLINK_POLICY_PARQUET:-outputs/policy_training/all.parquet}" \
  --train_parquet "${TRAIN_PARQUET:-outputs/policy_training/train.parquet}" \
  --val_parquet "${VAL_PARQUET:-outputs/policy_training/val.parquet}" \
  --val_ratio "${GRAPHLINK_POLICY_VAL_RATIO:-0.05}"
