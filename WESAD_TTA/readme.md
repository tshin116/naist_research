## WESAD 1D-CNN LOSO + Test-Time Adaptation

This directory reorganizes the WESAD stress detection experiment in an
OFTTA-style layout.

### Layout

- `cfg/`: YAML configs for datasets, algorithms, and default runtime settings.
- `data_processing/`: WESAD loading, windowing, label mapping, scaling, and loaders.
- `models/`: 1D-CNN model definitions.
- `TTA/`: Test-time adaptation setup and algorithms.
- `scripts/`: Reproducible shell entrypoints.
- `ckpt/`: LOSO model checkpoints.
- `logs/`: Training and adaptation logs.

### Train LOSO source models

```bash
bash train.sh
```

### Evaluate source and Tent

```bash
bash adapt.sh
```

The default dataset config expects the original WESAD data at
`../self_learning_wesad/WESAD`.
