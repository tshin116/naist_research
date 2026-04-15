#!/bin/bash

# WESAD を baseline / stress / amusement の 3 分類として外側 LOSO 学習する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python train.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
