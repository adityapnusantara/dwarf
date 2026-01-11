#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/gemma-3-1b-qlora.yml`}"
DATA="${DATA:-${ROOT}/data/train.jsonl}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-0}"  # 0 = no limit

echo "[dwarf] Preparing data -> ${DATA} (split=${SPLIT}, limit=${LIMIT})"
python "${ROOT}/scripts/prepare_data.py" --split "${SPLIT}" --limit "${LIMIT}" --out "${DATA}"

cd "${ROOT}"
echo "[dwarf] Training..."
axolotl train "${CONFIG}" "$@"

if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
  echo "[dwarf] Merging LoRA..."
  axolotl merge-lora "${CONFIG}"
else
  echo "[dwarf] Skipping merge (SKIP_MERGE=1)"
fi
