#!/bin/bash

# 固定 epoch の3分類 checkpoint を使い、target loader を shuffle して Source、Tent、OFTTA を比較する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class_target_shuffle.yaml \
  --resume ./ckpt_fixed_3class \
  "$@"
