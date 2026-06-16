#!/bin/bash

cd /home/shinsaku-t/work/naist_reserch/WESAD_TTA

PYTHON="${PYTHON:-python}"

"$PYTHON" compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/emowear_valence.yaml \
  --resume ./ckpt_emowear_valence \
  --methods source norm tent oftta ema_tent dynamix_ema_tent tema dua rotta note delta \
  --run_name EmoWear_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA
