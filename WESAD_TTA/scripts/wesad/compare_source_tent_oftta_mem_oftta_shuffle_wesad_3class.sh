#!/bin/bash

# 3分類 checkpoint を使い、target loader を shuffle して Source / Tent / OFTTA / MemOFTTA を比較する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class_target_shuffle.yaml \
  --resume ./ckpt_3class \
  --methods source tent oftta mem_oftta \
  "$@"
