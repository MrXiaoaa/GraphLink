#!/usr/bin/env bash
set -euo pipefail
${PYTHON:-python3} -m graphlink.data.prepare --config "${GRAPHLINK_PATH_CONFIG:-configs/paths.yaml}" --create-output-dirs
