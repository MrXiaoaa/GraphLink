# Schema Linking Baselines

This document defines the schema-linking baselines used to compare against GraphLink. It does not vendor baseline source code. Citation details are collected in [../../docs/baseline_citations.md](../../docs/baseline_citations.md).

## Baselines

| Name | Citation | Description | Included? | Notes |
|---|---|---|---:|---|
| GraphLink (ours) | GraphLink paper, pending | Graph search + subquery decomposition + personalized PageRank + policy pruning. | Yes | Main method in `graphlink/schema_linking/`. |
| GraphLink without dependency hints | GraphLink paper, pending | Pure selected-table schema linking output. | Yes | Dependency hints are not part of schema-linking metrics. |
| AutoLinkSL | `wang2025autolink` | AutoLink schema-linking output used as an external baseline. | No | Convert its output to unified linked JSON. |
| DE-SL | `karpukhin2020dpr` | Dense-retrieval schema-linking baseline. | No | Convert its output to unified linked JSON. |
| CE-SL | `khattab2020colbert` | Interaction/reranking schema-linking baseline. | No | Convert its output to unified linked JSON. |
| Oracle | Dataset citations | Gold tables from benchmark annotations. | No data committed | Used as upper-bound schema input. |
| CHESS native table-level | `talaei2024chess` | Optional CHESS-style table-level schema linking baseline. | Adapter only | Used only when explicitly studying native schema-linking behavior. |

## Required Output Format

All baselines should be normalized to the GraphLink linked JSON shape:

```json
{
  "instance_id": [
    {
      "answer": "Y",
      "table name": "database.schema.table",
      "columns": [],
      "selection_method": "baseline_name",
      "score": 0.0
    }
  ]
}
```

Rules:

- `answer == "Y"` means selected.
- `columns == []` means table-level/all-columns selection.
- Preserve original table names when possible.
- For Spider2.0-Lite `sf_bq*` / `sf_ga*` tasks, keep a consistent mapping between physical instance id and logical database/table names.

## What To Commit

Commit conversion scripts or small config files if needed. Do not commit baseline repositories, raw outputs, logs, credentials, or large generated datasets.
