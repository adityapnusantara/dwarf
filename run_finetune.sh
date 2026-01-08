#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${ROOT}/deepseek-coder-1.3b-instruct.yml}"
DATA="${DATA:-${ROOT}/formatted_synthetic_sql.jsonl}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-0}"  # 0 = no limit

echo "[dwarf] Generating dataset: split=${SPLIT} limit=${LIMIT} -> ${DATA}"
python "${ROOT}/load_synthetic_sql.py" --split "${SPLIT}" --limit "${LIMIT}" --out "${DATA}"

cd "${ROOT}"
echo "[dwarf] Training..."
axolotl train "${CONFIG}" "$@"

echo "[dwarf] Merging LoRA..."
axolotl merge-lora "${CONFIG}"
