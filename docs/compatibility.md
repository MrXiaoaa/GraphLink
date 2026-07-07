# Third-Party Compatibility Notes

GraphLink is released as an independent schema-linking package. A few compatibility shims are included to interoperate with Spider2.0-Lite execution/evaluation utilities used in the research workspace:

- `graphlink/schema_linking/_graphlink_core_0201.py`: main GraphLink schema linking implementation extracted from the 0201 experiment version.
- `graphlink/schema_linking/chat.py`, `utils.py`, `reconstruct_data.py`: compatibility helpers required by the 0201 core.
- `graphlink/sql_generation/dialect_prompt.py`: backend dialect helper text used by the Spider2.0-Lite prompt adapter.

Historical experiment outputs, credentials, retry queues, and unrelated scripts are intentionally excluded. These compatibility files are not part of the GraphLink schema-linking method itself.
