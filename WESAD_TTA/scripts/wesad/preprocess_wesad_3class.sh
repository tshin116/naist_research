#!/bin/bash

# WESAD の raw .pkl から、3分類用の前処理済み .npz を作成する。
# 保存先は cfg/dataset/wesad_3class.yaml の processed_dir に従う。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python preprocess.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
