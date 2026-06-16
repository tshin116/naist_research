#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
BLOCK_LENGTHS="${BLOCK_LENGTHS:-16 32 64 128 256 512}"

for block_length in $BLOCK_LENGTHS
do
  "$PYTHON" compare_source_tent_oftta.py \
    --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
    --resume ./ckpt_3class \
    --stream_mode synthetic_block \
    --block_length "$block_length" \
    --methods source norm tent ema_tent mi_dynamic_ema_tent \
    --run_name "WESAD3class_syntheticBlock${block_length}_MI_Dynamic_EMATent" \
    "$@"
done
