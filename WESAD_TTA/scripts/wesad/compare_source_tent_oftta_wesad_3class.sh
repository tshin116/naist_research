#!/bin/bash

# 3分類 checkpoint を使い、Source / Tent / OFTTA を被験者ごとに比較する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class
