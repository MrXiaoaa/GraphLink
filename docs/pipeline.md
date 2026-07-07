# Complete GraphLink Pipeline

This document describes the reproducibility pipeline from raw datasets to GraphLink schema linking, schema-context rendering, fixed-generator evaluation, and policy training.

## 0. Environment Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure model access through environment variables:

```bash
export OPENAI_BASE_URL=http://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your_key
export VLLM_MODEL_NAME=your_chat_model
export GRAPHLINK_EMBEDDING_MODEL=Qwen3-Embedding-8B
```

No API keys or credentials should be written into source files.

## 1. Dataset Download and Layout

GraphLink expects datasets to be prepared into example directories. Large datasets are not committed.

Typical layout:

```text
data/
  spider/
  bird/
  spider2-lite/
  examples_spider/
  examples_bird/
  examples_lite/
  database_graphs_spider/
  database_graphs_bird/
  database_graphs_0206_enhanced/
  unified_schema_linking/
outputs/
  schema_linking/
  sql_generation/
  eval_details/
```

Use:

```bash
bash scripts/00_download_data.sh
bash scripts/01_prepare_data.sh
```

`00_download_data.sh` only stages archives or local files. Dataset unpacking and license-controlled files must be handled locally.

## 2. Data Preparation

Each example directory should contain enough information to render table schemas and prompts:

```text
examples_lite/<instance_id>/
  prompts.txt                  # optional source prompt for backend-aware context rendering
  table_descriptions.json       # table-level descriptions
  <database/schema files>        # DDL.csv, table JSON, sqlite db, or online DB metadata
```

For Spider/BIRD, use converted local examples. For Spider2.0-Lite, keep per-example online credentials outside git and provide them only in local runtime directories.

Validate paths:

```bash
python3 -m graphlink.data.prepare --config configs/paths.yaml --create-output-dirs
```

## 3. Schema Graph Construction

GraphLink builds database-specific schema graphs. Nodes are tables; edges can come from:

- explicit foreign keys,
- MinHash/value overlap,
- table-description similarity,
- IND/AIND-style dependency signals,
- workload/conditional-function metadata when available.

Example:

```bash
export GRAPHLINK_EXAMPLES_LITE=/path/to/examples_lite
export GRAPHLINK_DATABASE_GRAPHS_DIR=outputs/database_graphs
bash scripts/02_build_schema_graph.sh
```

Equivalent direct command:

```bash
python3 -m graphlink.schema_linking.run \
  --task lite \
  --db_path "$GRAPHLINK_EXAMPLES_LITE" \
  --build_db_graphs \
  --db_graphs_output "$GRAPHLINK_DATABASE_GRAPHS_DIR"
```

## 4. GraphLink Schema Linking

Main setting:

```bash
python3 -m graphlink.schema_linking.run \
  --task lite \
  --db_path /path/to/examples_lite \
  --linked_json_pth outputs/schema_linking/graphlink_linked.json \
  --database_graphs_dir /path/to/database_graphs \
  --use_semantic_graph_search \
  --use_subquery_decomposition \
  --top_k_preselection 10 \
  --enable_topk_rerank \
  --enable_batch_rerank \
  --batch_size 10
```

Important flags:

- `--use_semantic_graph_search`: use graph-aware retrieval rather than the older flat retrieval path.
- `--use_subquery_decomposition`: decompose complex questions into subqueries before graph search.
- `--top_k_preselection`: initial candidate budget before LLM policy pruning.
- `--enable_topk_rerank`: ask the LLM/policy model to judge preselected candidates.
- `--enable_batch_rerank`: judge multiple candidate tables in one batch prompt.
- `--disable_graph_topology`: ablation flag; disables graph-topology text in batch pruning.

Output is a linked JSON keyed by instance id. Selected tables have `answer == "Y"`.

## 5. Schema Linking Metrics

```bash
python3 -m graphlink.schema_linking.metrics \
  --linked-json outputs/schema_linking/graphlink_linked.json \
  --db-path /path/to/examples_lite
```

Metrics are table-level precision, recall, and F1 against gold linked tables.

## 6. SQL Generation Inputs

GraphLink-selected schema can be rendered into backend-aware SQL-generation prompts:

```bash
python3 -m graphlink.sql_generation.build_prompts \
  --source-examples /path/to/examples_lite \
  --output-examples outputs/examples_lite_graphlink \
  --linking-file outputs/schema_linking/graphlink_linked.json \
  --database-graphs-dir /path/to/database_graphs \
  --dependency-hints \
  --prompt-char-budget 131072
```

This rewrites the table-structure block to contain GraphLink-selected tables and optionally appends dependency hints.

## 7. SQL Generation

Single-shot CHESS/KaSLA-compatible runner:

```bash
python3 -m graphlink.sql_generation.run \
  --datasets spider bird \
  --methods CHESS KaSLA \
  --output-dir outputs/sql_generation/native \
  --linking-dir /path/to/unified_schema_linking \
  --graphlink-dependency-hints \
  --backend-dialect-prompts \
  --workers 4 \
  --max-tokens 4096
```

Provider/model are controlled by environment variables or CLI:

```bash
export SQLGEN_PROVIDER=openai
export OPENAI_BASE_URL=http://your-endpoint/v1
export OPENAI_API_KEY=your_key
export VLLM_MODEL_NAME=your_model
```

## 8. Evaluation

### Spider/BIRD

```bash
python3 -m graphlink.evaluation.evaluate_sql_outputs \
  --dataset spider \
  --source native \
  --run-root outputs/sql_generation/native \
  --method CHESS \
  --schema-linking graphlink \
  --task-id spider_chess_graphlink
```

### Spider2.0-Lite Compile Validity

```bash
python3 -m graphlink.evaluation.evaluate_spider2lite_compile \
  --run-root outputs/sql_generation/native \
  --method CHESS \
  --schema-linking graphlink \
  --task-id spider2lite_chess_graphlink_compile
```

### Spider2.0-Lite Result Accuracy

```bash
python3 -m graphlink.evaluation.evaluate_spider2lite_accuracy \
  --run-root outputs/sql_generation/native \
  --method CHESS \
  --schema-linking graphlink \
  --compile-detail outputs/eval_details/<compile_detail>.json \
  --task-id spider2lite_chess_graphlink_accuracy \
  --credential-mode per-example \
  --resume
```

Online evaluation requires local BigQuery/Snowflake credentials. These are intentionally not included.

## 9. Policy Training

The policy-training module trains a model to select relevant tables from candidates.

### Generate QA JSONL

```bash
export GRAPHLINK_EXAMPLES_LITE=/path/to/examples_lite
bash scripts/policy_training/00_generate_qa.sh
```

### Convert to Parquet

```bash
bash scripts/policy_training/01_convert_to_parquet.sh
```

Expected parquet columns:

- `id`
- `prompt`
- `ground_truth`
- `extra_info`

### Split Train/Validation

```bash
bash scripts/policy_training/02_split_parquet.sh
```

### Train with VERL GRPO

```bash
export BASE_MODEL=/path/to/base/model
export TRAIN_PARQUET=outputs/policy_training/train.parquet
export VAL_PARQUET=outputs/policy_training/val.parquet
bash scripts/policy_training/03_train_grpo.sh
```

The packaged reward function is `graphlink/policy_training/schema_filtering_reward.py:compute_score`.

## 10. Outputs to Report

For paper tables, keep these outputs:

- schema linking linked JSON,
- schema linking precision/recall/F1 logs,
- SQL generation predictions,
- Spider/BIRD execution accuracy summary,
- Spider2.0-Lite compile validity summary,
- Spider2.0-Lite result accuracy summary,
- prompt length/table-count/token statistics if comparing context budgets.

Do not commit raw outputs to GitHub unless they are deliberately curated artifacts.

## 11. Release Checklist

Before publishing:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
rg -n "<your-private-endpoint-regex>|<your-private-path-regex>|<your-private-account-regex>|sk-[A-Za-z0-9_-]{20,}" .
find . -type f \( -name '*credential*' -o -name '*.pem' -o -name '*.key' -o -name '.env*' \)
bash scripts/smoke_test.sh
```

Only the bundled sample parquet under `examples/policy_training/` and `graphlink/policy_training/data/` is intended to be committed.
