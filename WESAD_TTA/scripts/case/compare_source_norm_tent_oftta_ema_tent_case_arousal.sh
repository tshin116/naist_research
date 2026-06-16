#!/bin/bash

cd /home/shinsaku-t/work/naist_reserch/WESAD_TTA

PYTHON="${PYTHON:-python}"

"$PYTHON" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_arousal.yaml \
  --resume ./ckpt_case_arousal \
  --methods source norm tent oftta ema_tent \
  --run_name CASE_arousal_SourceNormTentOFTTAEMATent
