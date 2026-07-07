# Policy Training

GraphLink includes a lightweight policy-training module for table-level schema filtering. The goal is to train a model to select the minimum relevant table set and optionally emit SQL.

## Bundled Sample

This release includes `examples/policy_training/train.parquet` as a small ready-to-inspect policy-training parquet sample. For real experiments, regenerate train/val parquet files from your local examples and pass them through `TRAIN_PARQUET` and `VAL_PARQUET`.

## Pipeline

1. Generate synthetic QA/SQL examples from table descriptions and schemas.
2. Convert QA JSONL to VERL-compatible Parquet.
3. Split Parquet into train/validation sets.
4. Train with VERL GRPO using the packaged reward function.

## Generate QA Data

```bash
export GRAPHLINK_EXAMPLES_LITE=/path/to/examples_lite
bash scripts/policy_training/00_generate_qa.sh
```

The generator reads `table_descriptions.json` from each example directory, asks an LLM to select tables and create questions, then validates SQL with the local/online SQL environment when credentials are available.

## Convert and Split

```bash
bash scripts/policy_training/01_convert_to_parquet.sh
bash scripts/policy_training/02_split_parquet.sh
```

Parquet columns:

- `id`: stable sample id.
- `prompt`: chat messages for the policy model.
- `ground_truth`: `{query, candidate_tables, gold_tables, gold_sql, gold_data}`.
- `extra_info`: backend/source metadata.

## Reward

`graphlink/policy_training/schema_filtering_reward.py` exposes `compute_score()`. It parses model output JSON:

```json
{"selected_tables": ["table_a"], "sql": "SELECT ..."}
```

The default reward uses table-level precision/recall/F1 plus a small JSON/SQL presence bonus. It does not require live database credentials in the release package. You can extend it with execution-result matching for your private environment.

## Train GRPO

Install VERL separately, then run:

```bash
export BASE_MODEL=/path/to/base/model
export TRAIN_PARQUET=outputs/policy_training/train.parquet
export VAL_PARQUET=outputs/policy_training/val.parquet
bash scripts/policy_training/03_train_grpo.sh
```

The training shell is intentionally environment-driven: no cluster paths, no internal endpoints, and no hard-coded API keys are stored in this release.
