# EmoWear Full LOSO and TTA Results

## 実験条件

- Dataset: EmoWear cleaned/synchronized CSV
- Subjects: 48
- Tasks:
  - Valence binary classification
  - Arousal binary classification
- Model: WESAD / CASE と同じ `StressCNN1D`
- Architecture: 1D-CNN + BatchNorm1d
- Input signals: 8 channels
  - E4 BVP
  - E4 EDA
  - E4 SKT
  - E4 ACC x/y/z
  - BH3 ECG
  - BH3 RSP
- Window: 10 s
- Stride: 5 s
- Sampling rate: 50 Hz
- Input shape: `(batch, 8, 500)`
- Source training: fixed-epoch LOSO, 20 epochs
- Batch size: 64
- Metric: Macro-F1

SensorTile ACC/GYRO は初期比較から除外した。理由は、欠損被験者が存在し、ファイルサイズが大きく、タイムスタンプが不規則であるためである。

## ラベル定義

Valence:

- `valence >= 5.1`: class 1
- `valence < 5.1`: class 0

Arousal:

- `arousal >= 5.0`: class 1
- `arousal < 5.0`: class 0

いずれもCSV全体で確認したグローバル中央値に基づく二値化である。

## 生成済みデータ

Processed cache:

- Valence: `data/emowear/processed_valence`
- Arousal: `data/emowear/processed_arousal`

全体のwindow数:

| Task | Subjects | Windows | Class 0 | Class 1 |
|---|---:|---:|---:|---:|
| Valence | 48 | 19303 | 9538 | 9765 |
| Arousal | 48 | 19303 | 9391 | 9912 |

注意点:

- Arousalでは `19-9UY7` が今回の閾値ではclass 0のみになった。
- そのため、Arousalの被験者別Macro-F1では、片クラスtargetが存在することを解釈時に明記する必要がある。

## 実行コマンド

Source model training:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/train_fixed_loso_emowear_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/train_fixed_loso_emowear_arousal.sh
```

TTA comparison:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/compare_source_tta_emowear_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/compare_source_tta_emowear_arousal.sh
```

## 保存先

Source LOSO:

- Valence: `logs/emowear/train_fixed_loso/260616_191732/loso_fixed_epoch_results.csv`
- Arousal: `logs/emowear/train_fixed_loso/260616_191734/loso_fixed_epoch_results.csv`

TTA comparison:

- Valence: `logs/emowear/compare_source_tent_oftta/260616_194526_EmoWear_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA`
- Arousal: `logs/emowear/compare_source_tent_oftta/260616_194526_EmoWear_arousal_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA`

各TTA比較ディレクトリには以下が保存される。

- `source_tent_oftta_comparison.csv`
- `source_tent_oftta_comparison.md`
- `source_tent_oftta_macro_f1.png`
- `source_tent_oftta_accuracy.png`
- `source_tent_oftta_average.png`
- `ema_tent_batch_diagnostics.csv`
- `ema_tent_gate_summary.md`
- `tta_stream_summary.csv`

## Valence Results

Average Macro-F1:

| Method | Macro-F1 |
|---|---:|
| DynaMix EMA-Tent | 0.4935 |
| EMA-Tent | 0.4915 |
| OFTTA | 0.4817 |
| TEMA | 0.4802 |
| DUA | 0.4794 |
| DELTA | 0.4777 |
| Tent | 0.4744 |
| Norm | 0.4742 |
| Source | 0.4722 |
| NOTE | 0.4719 |
| RoTTA | 0.4719 |

Best-subject count:

| Method | Count |
|---|---:|
| DELTA | 10 |
| OFTTA | 8 |
| Source | 8 |
| TEMA | 6 |
| DynaMix EMA-Tent | 5 |
| Norm | 4 |
| EMA-Tent | 4 |
| RoTTA | 3 |
| Tent | 1 |
| NOTE | 1 |

Interpretation:

Valenceでは、平均Macro-F1でDynaMix EMA-Tentが最良であった。ただし、差は大きくなく、EMA-Tentとの差は約0.002である。これは、EmoWear Valenceでは動的混合重みが性能を大きく押し上げるというより、固定混合比率のEMA-Tentと同程度以上を維持しつつ、固定ハイパーパラメータ依存を弱める方向に働いていると解釈できる。

一方で、被験者別の勝ち数ではDELTAが最も多く、平均性能と被験者別の最良手法は一致しない。これはEmoWearの被験者間差、ラベル分布差、動画ごとの反応差が大きく、単一のTTA手法が全被験者で一貫して優位になる状況ではないことを示す。

## Arousal Results

Average Macro-F1:

| Method | Macro-F1 |
|---|---:|
| DELTA | 0.4624 |
| EMA-Tent | 0.4552 |
| DynaMix EMA-Tent | 0.4549 |
| TEMA | 0.4516 |
| DUA | 0.4508 |
| Source | 0.4502 |
| Norm | 0.4501 |
| Tent | 0.4499 |
| NOTE | 0.4476 |
| RoTTA | 0.4473 |
| OFTTA | 0.4412 |

Best-subject count:

| Method | Count |
|---|---:|
| DELTA | 13 |
| OFTTA | 8 |
| Source | 8 |
| RoTTA | 5 |
| Norm | 5 |
| DynaMix EMA-Tent | 4 |
| TEMA | 3 |
| EMA-Tent | 2 |
| DUA | 2 |

Interpretation:

ArousalではDELTAが平均Macro-F1と勝ち数の両方で最良であった。EMA-TentとDynaMix EMA-TentはSourceよりはわずかに高いが、改善幅は限定的である。Arousalは一部被験者でラベルが大きく偏っており、特に `19-9UY7` は片クラスtargetである。このような条件では、BN統計の安定化だけではなく、予測分布やクラスバランスを明示的に制御する手法が有利になる可能性がある。

## Overall Notes

EmoWearでは、WESADで見られたようなEMA-Tent / DynaMix EMA-Tentの明確な優位性は限定的であった。ValenceではDynaMix EMA-Tentが平均Macro-F1で最良だったが、ArousalではDELTAが最良だった。

この結果から、DynaMix EMA-TentはWESADのようにブロック構造やBN統計の崩壊が強く効く条件では有効である一方、EmoWearのようにラベル分布や被験者差が異なるデータセットでは、クラス分布制御を含むTTA手法も強い比較対象になると考えられる。

論文で扱う場合は、EmoWearを外部データセット検証として位置付け、「提案手法がValenceでは最良平均を示したが、ArousalではDELTAが最良であり、タスク依存性がある」と記述するのが妥当である。
