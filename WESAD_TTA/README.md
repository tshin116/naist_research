# WESAD_TTA

This directory contains research code for wearable affective computing with Test-Time Adaptation.

Main comparison targets:

- Source
- Norm
- Tent
- OFTTA
- EMA-Tent
- DynaMix EMA-Tent
- TEMA
- DUA
- RoTTA
- NOTE
- DELTA

Main datasets:

- WESAD: stress/no-stress and baseline/stress/amusement
- CASE: arousal and valence
- EmoWear: arousal and valence

## First Read

Start from:

- `PROJECT_OVERVIEW_FOR_SUPERVISOR.md`

This file summarizes the directory structure, datasets, preprocessing, methods, DynaMix EMA-Tent, result locations, and major comparison tables.

Then read:

- `新手法の説明文/dynamix_ema_tent_method.md`
- `新手法の説明文/ema_tent_method.md`
- `docs/external_tta_methods_research_memo.md`
- `docs/emowear_full_experiment_results.md`
- `SHARING_GUIDE.md`

`Readme.md` is the older detailed WESAD implementation guide. `README.md` is the current entry point for sharing this repository.

## Main Code

- `train_fixed_loso.py`: LOSO source-model training
- `compare_source_tent_oftta.py`: unified Source/TTA comparison
- `models/cnn1d.py`: 1D-CNN with BatchNorm1d
- `data_processing/`: WESAD / CASE / EmoWear preprocessing and loaders
- `TTA/adapt_algorithm/`: TTA implementations
- `cfg/`: dataset and algorithm YAML configs
- `scripts/`: reproducible shell commands

## Main Results

Primary result directories are under:

- `logs/wesad/compare_source_tent_oftta/`
- `logs/case/compare_source_tent_oftta/`
- `logs/emowear/compare_source_tent_oftta/`

Important latest comparison logs:

- `logs/wesad/compare_source_tent_oftta/260616_195900_WESAD2class_stress_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/`
- `logs/wesad/compare_source_tent_oftta/260616_201019_WESAD3class_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/`
- `logs/case/compare_source_tent_oftta/260616_200625_CASE_arousal_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/`
- `logs/case/compare_source_tent_oftta/260616_200628_CASE_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/`
- `logs/emowear/compare_source_tent_oftta/260616_194526_EmoWear_arousal_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/`
- `logs/emowear/compare_source_tent_oftta/260616_194526_EmoWear_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/`

Each result directory contains:

- `source_tent_oftta_comparison.csv`
- `source_tent_oftta_comparison.md`
- `source_tent_oftta_average.png`
- `source_tent_oftta_macro_f1.png`
- `source_tent_oftta_accuracy.png`

## Data Policy

Raw data and processed data are not intended to be committed to Git.

Ignored locally:

- `WESAD_TTA/data/`
- raw WESAD / CASE / EmoWear data
- Python cache
- local virtual environments
- external cloned repositories

The dataset paths are controlled by YAML files in `cfg/dataset/`.

