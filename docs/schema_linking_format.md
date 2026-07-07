# Schema Linking Format

GraphLink uses a table-level linked JSON format. Each top-level key is an instance id. Each value is a list of table decisions.

Required fields for selected tables:

- `answer`: selected when this is `Y`, `yes`, `true`, or `1`.
- `table name`: full table name or display table name.
- `columns`: optional list; empty means all columns are available.

Useful metadata fields:

- `selection_method`: PageRank, policy pruning, fallback, ablation setting, etc.
- `score`: ranking or confidence score.

The helper `graphlink.schema_linking.format.read_linking()` normalizes both GraphLink answer-list JSON and row-oriented JSONL-style predictions.
