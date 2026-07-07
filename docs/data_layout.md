# Data Layout

The repository does not commit datasets or credentials. A typical local layout is:

```text
data/
  spider/
  bird/
  spider2-lite/
  examples_lite/
  database_graphs_0206_enhanced/
  unified_schema_linking/
outputs/
  schema_linking/
  sql_generation/
  eval_details/
```

BigQuery and Snowflake credentials, when needed for Spider2Lite online evaluation, must stay outside git or be injected by the user locally.
