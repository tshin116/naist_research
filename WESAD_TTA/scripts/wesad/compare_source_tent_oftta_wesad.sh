#!/bin/bash

# Source、Tent、OFTTA を各被験者ごとに評価し、比較表を作成する。
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  "$@"
