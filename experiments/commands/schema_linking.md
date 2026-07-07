# Schema Linking Command Templates

## GraphLink on Spider2.0-Lite

```bash
python3 -m graphlink.schema_linking.run \
  --task lite \
  --db_path /path/to/examples_lite \
  --linked_json_pth outputs/schema_linking/graphlink_lite.json \
  --database_graphs_dir /path/to/database_graphs_0206_enhanced \
  --use_semantic_graph_search \
  --use_subquery_decomposition \
  --top_k_preselection 10 \
  --enable_topk_rerank \
  --enable_batch_rerank \
  --batch_size 10
```

## GraphLink on BIRD Local Examples

```bash
python3 -m graphlink.schema_linking.run \
  --task local \
  --db_path /path/to/examples_bird \
  --linked_json_pth outputs/schema_linking/graphlink_bird.json \
  --database_graphs_dir /path/to/database_graphs_bird \
  --use_semantic_graph_search \
  --use_subquery_decomposition \
  --use_desc_in_rerank \
  --model Qwen14B-rl-alldata-80-conditional-strict \
  --top_k_preselection 10 \
  --enable_topk_rerank \
  --enable_batch_rerank \
  --disable_graph_topology \
  --batch_size 10
```

## Metrics

```bash
python3 -m graphlink.schema_linking.metrics \
  --linked-json outputs/schema_linking/graphlink_lite.json \
  --db-path /path/to/examples_lite
```
