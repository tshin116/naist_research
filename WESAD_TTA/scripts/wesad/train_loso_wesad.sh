#!/bin/bash

# WESAD の全被験者を対象に外側 LOSO 学習を実行する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python train.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
