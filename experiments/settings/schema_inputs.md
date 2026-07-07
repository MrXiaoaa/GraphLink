# Schema Linking Input Settings

This file defines schema-linking settings only.

## Full Settings

| Schema-linking input | Definition | Column policy |
|---|---|---|
| GraphLink | Tables with `answer == "Y"` from GraphLink linked JSON. | All columns unless `columns` is non-empty. |
| AutoLinkSL | Tables selected by AutoLink schema-linking baseline. | Usually table-level/full-table in main comparisons. |
| DE-SL | Tables selected by DE-SL schema-linking baseline. | Usually table-level/full-table in main comparisons. |
| Oracle | Gold relevant tables from benchmark annotations. | Gold table set; all columns available unless gold columns are used. |
| CHESS native table-level | CHESS-selected tables for schema-linking-only studies. | Table-level/all-columns. |

## Table-Budget Settings

Budgeted schema-linking settings evaluate selection quality under a fixed table budget.

| Dataset | Budgets | Notes |
|---|---|---|
| Spider | Top-3, Top-5, Top-8, Full | Use table-level selected table count. |
| BIRD | Top-3, Top-5, Top-8, Full | Use table-level selected table count. |
| Spider2.0-Lite | Top-10, Top-20, Top-30, Full | Use effective selected tables after Spider2 table-name normalization. |

## Fill Policy

Default: no fill.

If a setting uses fill, label it explicitly:

- `GraphLink + DE-SL fill`
- `AutoLinkSL + DE-SL fill`
- `DE-SL + DE-SL fill`

Filled settings should report both raw selected table count and effective table count.

## Metrics To Report

- Precision.
- Recall.
- F1.
- Average selected tables.
- Recall < 1 count.
- Optional: average judged candidates and failed samples.

For Spider2.0-Lite, include the exact id-normalization rule used for `sf_bq*` and `sf_ga*` examples.
