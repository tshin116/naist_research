#!/bin/bash

# 内側 LOSO を使わず、WESAD 3分類 source model を固定 epoch で外側 LOSO 学習する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python train_fixed_loso.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --algorithm_cfg ./cfg/algorithm/source_fixed_3class.yaml
