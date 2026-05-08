#!/bin/bash
set -euo pipefail

# WESAD 3-class diagnostic condition:
# put all target windows of one subject into a single test-time batch.
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
PYTHON=${PYTHON:-/work/shinsaku-t/miniconda3/envs/wesad_env/bin/python}

TMPDIR=/work/shinsaku-t/naist_reserch/tmp \
MPLCONFIGDIR=/work/shinsaku-t/naist_reserch/tmp/matplotlib \
CUDA_VISIBLE_DEVICES='' \
"$PYTHON" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --batch_size 100000 \
  --run_name WESAD3class_oneTargetSubjectOneBatch_SourceTentOFTTA \
  "$@"
