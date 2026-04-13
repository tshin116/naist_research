#!/bin/bash

# WESAD の raw .pkl から、被験者ごとの前処理済み .npz を作成する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python preprocess.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
