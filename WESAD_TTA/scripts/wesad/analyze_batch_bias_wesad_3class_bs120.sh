#!/bin/bash
set -euo pipefail

# Usage:
#   bash scripts/wesad/analyze_batch_bias_wesad_3class_bs120.sh <sequential_csv> <shuffle_csv>
#
# The analysis recomputes target-batch class composition with batch size 120
# and correlates it with sequential delta and shuffle gain.
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <sequential_comparison_csv> <shuffle_comparison_csv>" >&2
  exit 1
fi

cd /work/shinsaku-t/naist_reserch/WESAD_TTA
PYTHON=${PYTHON:-/work/shinsaku-t/miniconda3/envs/wesad_env/bin/python}

TMPDIR=/work/shinsaku-t/naist_reserch/tmp \
MPLCONFIGDIR=/work/shinsaku-t/naist_reserch/tmp/matplotlib \
CUDA_VISIBLE_DEVICES='' \
"$PYTHON" analyze_batch_bias.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --comparison_csv "$1" \
  --shuffle_comparison_csv "$2" \
  --batch_size 120 \
  --run_name WESAD3class_bs120_batchBias_vs_SourceTentOFTTA
