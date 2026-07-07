# Schema-Context Rendering and Evaluation

GraphLink supports two downstream modes.

## Single-shot CHESS/KaSLA-style generation

Use `python -m graphlink.sql_generation.run` with selected schema inputs. This preserves each SQL generator's prompt style while swapping the schema linking input.

## Backend-aware selected-schema prompts

Use `python -m graphlink.sql_generation.build_prompts` to take an existing examples directory and rewrite `prompts.txt` so that the table-structure section contains GraphLink-selected tables and optional dependency hints. This step only changes the serialized schema context; any column exploration, refinement, voting, or repair behavior belongs to the downstream SQL generator and is not part of GraphLink.

## Dependency hints

Dependency hints are rendered as:

```text
GraphLink table dependency hints:
- table_a.col_x = table_b.col_y (type=..., reason=...)
- table_a -> table_b (reason=...)
```

Only hints involving selected tables should be injected.
