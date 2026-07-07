#!/usr/bin/env bash
set -euo pipefail
echo "Run dataset-specific evaluators from graphlink.evaluation after SQL generation finishes."
echo "Spider/BIRD: ${PYTHON:-python3} -m graphlink.evaluation.evaluate_sql_outputs --help"
echo "Spider2Lite compile: ${PYTHON:-python3} -m graphlink.evaluation.evaluate_spider2lite_compile --help"
echo "Spider2Lite accuracy: ${PYTHON:-python3} -m graphlink.evaluation.evaluate_spider2lite_accuracy --help"
