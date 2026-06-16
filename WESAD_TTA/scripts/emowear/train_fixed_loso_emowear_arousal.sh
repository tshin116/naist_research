#!/bin/bash

cd /home/shinsaku-t/work/naist_reserch/WESAD_TTA

PYTHON="${PYTHON:-python}"

"$PYTHON" train_fixed_loso.py \
  --dataset_cfg ./cfg/dataset/emowear_arousal.yaml \
  --algorithm_cfg ./cfg/algorithm/source_fixed.yaml \
  --resume ./ckpt_emowear_arousal
