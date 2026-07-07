# Schema Linking Experiment Settings

This folder records only the schema-linking baseline settings used to compare GraphLink with other schema-linking methods.

It is intentionally limited to schema-linking inputs, baseline definitions, table-budget settings, and schema-linking command templates.

## Files

- `baselines/README.md`: schema-linking baseline definitions and inclusion policy.
- `settings/schema_inputs.md`: full/top-k schema input settings and linked JSON contracts.
- `commands/schema_linking.md`: GraphLink schema-linking command templates and metric commands.

## Inclusion Policy

We include settings and data contracts, not full third-party baseline repositories. Baseline implementations such as AutoLinkSL or DE-SL should be installed externally, and their outputs should be converted to the unified linked JSON format before comparison.

## Main Schema Linking Comparisons

| Method | Role | Output contract |
|---|---|---|
| GraphLink (ours) | Graph-based table-level schema linking. | `instance_id -> [{answer, table name, columns, selection_method, score}]` |
| AutoLinkSL | External schema-linking baseline. | Unified linked JSON after conversion. |
| DE-SL | External schema-linking baseline. | Unified linked JSON after conversion. |
| Oracle | Gold schema-linking upper bound. | Gold linked JSON after conversion. |
| CHESS native | Optional table-level baseline for Spider2.0-Lite schema-linking studies. | Unified linked JSON after conversion. |

## Metrics

Report table-level schema-linking precision, recall, and F1. For Spider2.0-Lite, also report average selected table count and prompt/rendered schema statistics when studying context budgets.
