# Sharing Guide

## Recommended Sharing Style

Use GitHub for code, configs, documentation, checkpoints, and lightweight result logs.

Do not put raw or processed datasets in GitHub.

Reason:

- Entire `WESAD_TTA/`: about 2.8 GB
- `WESAD_TTA/data/`: about 2.7 GB
- `WESAD_TTA/logs/`: about 65 MB
- checkpoints: small enough to share

So the practical split is:

- GitHub: code, configs, docs, scripts, logs, checkpoints
- Cloud/NAS: datasets and processed caches

## What To Share In Git

Include:

- `README.md`
- `PROJECT_OVERVIEW_FOR_SUPERVISOR.md`
- `SHARING_GUIDE.md`
- `Readme.md`
- `cfg/`
- `scripts/`
- `TTA/`
- `models/`
- `data_processing/`
- `docs/`
- `tools/`
- `新手法の説明文/`
- `logs/`
- `ckpt*`
- main Python entry points such as `compare_source_tent_oftta.py` and `train_fixed_loso.py`

Exclude:

- `data/`
- raw WESAD / CASE / EmoWear files
- `.venv/`
- `__pycache__/`
- external cloned repositories such as `DELTA/`, `DUA/`, `NOTE/`, `RoTTA/`, `RealisticTTA/`

## Data Paths

Dataset locations are configured by YAML files:

- `cfg/dataset/wesad_3class.yaml`
- `cfg/dataset/wesad_binary_processed.yaml`
- `cfg/dataset/case_arousal.yaml`
- `cfg/dataset/case_valence.yaml`
- `cfg/dataset/emowear_arousal.yaml`
- `cfg/dataset/emowear_valence.yaml`

When another machine runs the code, update the dataset path in these YAML files if necessary.

## Files To Read First

For a supervisor or collaborator:

1. `PROJECT_OVERVIEW_FOR_SUPERVISOR.md`
2. `README.md`
3. `新手法の説明文/dynamix_ema_tent_method.md`
4. `docs/external_tta_methods_research_memo.md`
5. Latest result directories listed in `README.md`

## If GitHub Rejects Large Files

If push fails because older commits already contain large data files, remove large files from Git history with `git filter-repo` or BFG Repo-Cleaner.

For normal future commits, `.gitignore` now prevents new dataset caches under `WESAD_TTA/data/` from being added.

## Optional Zip Sharing

For a one-time share, create an archive excluding datasets:

```bash
tar \
  --exclude='WESAD_TTA/data' \
  --exclude='WESAD_TTA/__pycache__' \
  --exclude='WESAD_TTA/.venv' \
  --exclude='WESAD_TTA_share_*.tar.gz' \
  -czf WESAD_TTA_share_YYYYMMDD.tar.gz WESAD_TTA
```

Then share the archive plus datasets separately if needed.
