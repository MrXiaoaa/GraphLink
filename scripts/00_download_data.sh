#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw data/spider data/bird data/spider2-lite

fetch() {
  local name="$1"
  local src="${2:-}"
  local out="data/raw/${name}"
  if [[ -z "$src" ]]; then
    echo "[skip] ${name}: set ${name}_ZIP or place files manually."
    return 0
  fi
  if [[ -f "$src" ]]; then
    cp "$src" "$out"
  else
    curl -L "$src" -o "$out"
  fi
}

fetch SPIDER "${SPIDER_ZIP:-}"
fetch BIRD "${BIRD_ZIP:-}"
fetch SPIDER2LITE "${SPIDER2LITE_ZIP:-}"

echo "Dataset archives are staged under data/raw. Unpack them according to docs/data_layout.md."
