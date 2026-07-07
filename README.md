# GraphLink

GraphLink is a graph-based, table-level schema linking pipeline for text-to-SQL. It builds database schema graphs, retrieves relevant tables for each natural-language question, optionally renders table dependency hints, and evaluates the resulting schema contexts on Spider, BIRD, and Spider2.0-Lite style benchmarks.

This repository is a cleaned release package extracted from a research workspace. It keeps the reproducible GraphLink schema-linking pipeline, context rendering utilities, and evaluation adapters, while excluding private credentials, large datasets, generated experiment outputs, and historical debugging scripts.

## Highlights

- **Graph-based schema linking**: table profiling, table descriptions, embedding retrieval, schema graph construction, personalized PageRank, subquery decomposition, and batch policy pruning.
- **Dependency-aware context rendering**: GraphLink can render table relation hints such as joins and graph dependencies for downstream generators.
- **Fixed-generator evaluation adapters**: utilities for CHESS/KaSLA-style single-shot evaluation and Spider2.0-Lite backend-aware prompt/evaluation compatibility.
- **Policy-training module**: generate QA/SQL data, convert to VERL-compatible parquet, split train/val data, and train a table-selection policy with GRPO.
- **GitHub-safe package**: no real BigQuery/Snowflake credentials, no internal endpoints, no experiment logs, and no full baseline repositories.

## Repository Layout

```text
GraphLink/
  graphlink/
    data/                    # path config validation and dataset preparation helpers
    schema_linking/          # GraphLink core, CLI, metrics, linked-file helpers
    sql_generation/          # schema-context renderers and fixed-generator adapters
    evaluation/              # Spider/BIRD and Spider2Lite evaluation adapters
    policy_training/         # QA generation, parquet conversion, reward, GRPO template
    spider2_compat/          # compatibility shims for Spider2.0-Lite evaluation
  configs/                   # example path/model/policy-training configs
  docs/                      # detailed pipeline and format documentation
  examples/policy_training/  # bundled sample train.parquet
  scripts/                   # staged runnable pipeline scripts
```

## Installation

```bash
git clone <your-github-repo-url>
cd GraphLink
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional but recommended for local development:

```bash
bash scripts/smoke_test.sh
```

The smoke test does not require full datasets. It validates imports, config loading, and linked-file normalization.

## Configuration

Copy the example configs and edit local paths:

```bash
cp configs/paths.example.yaml configs/paths.yaml
cp configs/model.example.yaml configs/model.yaml
cp configs/policy_training.example.yaml configs/policy_training.yaml
```

Model access is configured through environment variables, not committed files:

```bash
export OPENAI_BASE_URL=http://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=your_key
export VLLM_MODEL_NAME=your_chat_model
export GRAPHLINK_EMBEDDING_MODEL=Qwen3-Embedding-8B
```

For Spider2.0-Lite online execution, keep BigQuery/Snowflake credentials outside git and point your local examples directory to them. Do not commit credential JSON files.

## End-to-End Pipeline

The staged scripts are intended to be read and customized. They use environment variables for local paths and model settings.

```bash
# 0. Download or stage datasets.
bash scripts/00_download_data.sh

# 1. Validate local path config and create output directories.
bash scripts/01_prepare_data.sh

# 2. Build database-specific schema graphs.
export GRAPHLINK_EXAMPLES_LITE=/path/to/examples_lite
export GRAPHLINK_DATABASE_GRAPHS_DIR=/path/to/database_graphs
bash scripts/02_build_schema_graph.sh

# 3. Run GraphLink schema linking.
bash scripts/03_run_schema_linking.sh

# 4. Build selected-schema SQL-generation prompts with dependency hints.
bash scripts/04_build_sql_prompts.sh

# 5. Run dataset-specific evaluation commands.
bash scripts/05_evaluate.sh
```

For a detailed phase-by-phase explanation, see [docs/pipeline.md](docs/pipeline.md).

## Schema Linking

Run the public GraphLink CLI:

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

For BIRD-style local examples, use `--task local` and point `--db_path` to the prepared BIRD examples directory:

```bash
python3 -m graphlink.schema_linking.run \
  --db_path /path/to/examples_bird \
  --linked_json_pth outputs/schema_linking/graphlink_bird.json \
  --use_semantic_graph_search \
  --use_subquery_decomposition \
  --database_graphs_dir /path/to/database_graphs_bird \
  --use_desc_in_rerank \
  --task local \
  --model Qwen14B-rl-alldata-80-conditional-strict \
  --top_k_preselection 10 \
  --enable_topk_rerank \
  --enable_batch_rerank \
  --disable_graph_topology \
  --batch_size 10
```

Compute table-level metrics:

```bash
python3 -m graphlink.schema_linking.metrics \
  --linked-json outputs/schema_linking/graphlink_linked.json \
  --db-path /path/to/examples_lite
```

## Linked JSON Format

GraphLink outputs table decisions keyed by instance id:

```json
{
  "bq001": [
    {
      "answer": "Y",
      "table name": "project.dataset.table",
      "columns": [],
      "selection_method": "pagerank",
      "score": 0.91
    }
  ]
}
```

Downstream code should treat `answer == "Y"` as selected. Empty `columns` means table-level selection with all columns available.

More details: [docs/schema_linking_format.md](docs/schema_linking_format.md).

## SQL Generation

Build backend-aware selected-schema prompts:

```bash
python3 -m graphlink.sql_generation.build_prompts \
  --source-examples /path/to/examples_lite \
  --output-examples outputs/examples_lite_graphlink \
  --linking-file outputs/schema_linking/graphlink_linked.json \
  --database-graphs-dir /path/to/database_graphs \
  --dependency-hints \
  --prompt-char-budget 131072
```

Run CHESS/KaSLA-style single-shot SQL generation:

```bash
python3 -m graphlink.sql_generation.run \
  --datasets spider bird \
  --methods CHESS KaSLA \
  --output-dir outputs/sql_generation/native \
  --linking-dir /path/to/unified_schema_linking \
  --graphlink-dependency-hints \
  --backend-dialect-prompts
```

Dependency hints are rendered in prompts as:

```text
GraphLink table dependency hints:
- table_a.col_x = table_b.col_y (type=..., reason=...)
- table_a -> table_b (reason=...)
```

More details: [docs/sql_generation.md](docs/sql_generation.md).

## Evaluation

Spider/BIRD execution accuracy:

```bash
python3 -m graphlink.evaluation.evaluate_sql_outputs \
  --dataset spider \
  --source native \
  --run-root outputs/sql_generation/native \
  --method CHESS \
  --schema-linking graphlink \
  --task-id spider_chess_graphlink
```

Spider2.0-Lite compile validity:

```bash
python3 -m graphlink.evaluation.evaluate_spider2lite_compile \
  --run-root outputs/sql_generation/native \
  --method CHESS \
  --schema-linking graphlink \
  --task-id spider2lite_chess_graphlink_compile
```

Spider2.0-Lite result accuracy:

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

## Policy Training

GraphLink includes a policy-training module for table-level schema filtering. A small sample parquet is included at `examples/policy_training/train.parquet`.

Generate QA data and train with VERL/GRPO:

```bash
export GRAPHLINK_EXAMPLES_LITE=/path/to/examples_lite
bash scripts/policy_training/00_generate_qa.sh
bash scripts/policy_training/01_convert_to_parquet.sh
bash scripts/policy_training/02_split_parquet.sh

export BASE_MODEL=/path/to/base/model
export TRAIN_PARQUET=outputs/policy_training/train.parquet
export VAL_PARQUET=outputs/policy_training/val.parquet
bash scripts/policy_training/03_train_grpo.sh
```

The default reward is table-level precision/recall/F1 plus a JSON/SQL-format bonus. It intentionally does not require live database credentials. See [docs/policy_training.md](docs/policy_training.md).

## Third-Party Compatibility

Some small compatibility shims are included so GraphLink schema contexts can be evaluated with common Spider2.0-Lite execution utilities. They are isolated from the core schema-linking method. See [docs/compatibility.md](docs/compatibility.md) and [NOTICE](NOTICE).

## Citation

If you use GraphLink, cite the artifact below and replace it with the final paper citation once the paper is published.

```bibtex
@misc{xiao2026graphlink,
  title = {GraphLink: Graph-Based Schema Linking for Text-to-SQL},
  author = {Xiao, Qingfa},
  year = {2026},
  howpublished = {\url{https://github.com/MrXiaoaa/GraphLink}},
  note = {Code artifact}
}
```

This repository also includes small compatibility adapters for Spider2.0-style execution that are derived from ReFoRCE-style utilities. If you use those adapters, compare against ReFoRCE, or reuse its evaluation setting, please cite both the ReFoRCE paper and repository:

```bibtex
@article{deng2025reforce,
  title = {ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Consensus Enforcement, and Column Exploration},
  author = {Deng, Minghang and Ramachandran, Ashwin and Xu, Canwen and Hu, Lanxiang and Yao, Zhewei and Datta, Anupam and Zhang, Hao},
  journal = {arXiv preprint arXiv:2502.00675},
  year = {2025}
}

@misc{snowflakelabs2025reforcecode,
  title = {ReFoRCE},
  author = {{Snowflake-Labs}},
  year = {2025},
  howpublished = {\url{https://github.com/Snowflake-Labs/ReFoRCE}},
  note = {GitHub repository}
}
```

BibTeX entries for the third-party baselines and benchmarks used by the experiment settings can be copied directly from [docs/baseline_citations.md](docs/baseline_citations.md).
