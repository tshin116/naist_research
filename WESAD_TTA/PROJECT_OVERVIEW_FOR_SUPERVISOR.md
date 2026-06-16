# WESAD_TTA Project Overview for Supervisor

## このディレクトリの目的

このディレクトリは、ウェアラブル生体信号を用いた感情推定に対して、Test-Time Adaptation、特に Source / Norm / Tent / OFTTA / EMA-Tent / DynaMix EMA-Tent などを比較する研究コードである。

主な目的は以下である。

- WESAD の未知被験者感情推定に TTA が有効かを検証する。
- WESAD の通常順序に存在するブロック構造と batch 内ラベル偏りが、既存TTAを壊す原因になるかを分析する。
- その対策として EMA-Tent および DynaMix EMA-Tent を実装・評価する。
- CASE / EmoWear など外部感情データセットにも同じ 1D-CNN + BatchNorm1d の枠組みを適用し、提案手法の汎化性を確認する。
- TEMA / DUA / RoTTA / NOTE / DELTA などの既存TTA手法も比較対象として追加する。

## まず読むべきファイル

最初に読むべきファイルは以下である。

| File | 内容 |
|---|---|
| `PROJECT_OVERVIEW_FOR_SUPERVISOR.md` | このファイル。全体の案内。 |
| `paper_results_ema_tent.md` | WESADを中心とした論文用の結果・実験ストーリー。 |
| `新手法の説明文/ema_tent_method.md` | EMA-Tentの手法説明。 |
| `新手法の説明文/dynamix_ema_tent_method.md` | DynaMix EMA-Tentの手法説明。 |
| `docs/external_tta_methods_research_memo.md` | TEMA / DUA / RoTTA / NOTE / DELTA の実装上の対応メモ。 |
| `docs/emowear_feasibility_report.md` | EmoWearを使えるか確認した内容。 |
| `docs/emowear_full_experiment_results.md` | EmoWearのフルLOSO学習とTTA比較結果。 |

## ディレクトリ構成

重要な場所だけを示す。

```text
WESAD_TTA/
  cfg/
    default.yaml
    dataset/
      wesad.yaml
      wesad_3class.yaml
      wesad_3class_target_shuffle.yaml
      wesad_binary_processed.yaml
      case_arousal.yaml
      case_valence.yaml
      emowear_arousal.yaml
      emowear_valence.yaml
    algorithm/
      source.yaml
      source_fixed.yaml
      norm.yaml
      tent.yaml
      oftta.yaml
      ema_tent.yaml
      dynamix_ema_tent.yaml
      tema.yaml
      dua.yaml
      rotta.yaml
      note.yaml
      delta.yaml

  data_processing/
    wesad.py
    case.py
    emowear.py

  models/
    cnn1d.py

  TTA/
    setup.py
    adapt_algorithm/
      norm.py
      tent.py
      oftta.py
      ema_tent.py
      mi_dynamic_ema_tent.py
      tema.py
      dua.py
      rotta.py
      note.py
      delta.py

  scripts/
    wesad/
    case/
    emowear/

  logs/
    wesad/
    case/
    emowear/
    summary_all_tta/

  docs/
  新手法の説明文/
```

## モデル

モデルは `models/cnn1d.py` の `StressCNN1D` である。

入力形状は PyTorch の Conv1d に合わせて以下である。

```text
(batch, channels, time)
```

構造は以下である。

```text
Conv1d
ReLU
BatchNorm1d
MaxPool1d
Conv1d
ReLU
BatchNorm1d
MaxPool1d
Conv1d
ReLU
BatchNorm1d
MaxPool1d
Global Average Pooling
Dropout
Linear
ReLU
Dropout
Linear
```

TTA手法は主に BatchNorm1d 層を対象にする。Tent / EMA-Tent / DynaMix EMA-Tent では、Conv / Linear は更新せず、基本的に BatchNorm affine パラメータ `gamma, beta` のみを更新する。

## データセット

このプロジェクトでは、すべてのデータセットをできるだけ同じ形式にそろえている。

共通方針:

- 入力は固定長の多チャンネル時系列window。
- PyTorch Conv1d用に `(batch, channels, time)` へ変換する。
- モデルは同じ `StressCNN1D` を使う。
- 評価は被験者単位のLOSOを基本にする。
- Source modelはtarget被験者を除外して学習する。
- TTAではtarget被験者のtest streamを順に入力し、ラベルは適応には使わず評価のみに使う。
- 指標はMacro-F1を主指標にする。

データセットごとの大きな違いは、ラベルの作られ方、window長、サンプリング周波数、stream構造である。

| Dataset | Task | Subjects | Signal type | Window | Sampling rate | Channels | Label definition |
|---|---|---:|---|---:|---:|---:|---|
| WESAD | stress/no-stress | 15 | chest physiological signals | 5 s | 700 Hz | 8 | stress vs non-stress |
| WESAD | baseline/stress/amusement | 15 | chest physiological signals | 5 s | 700 Hz | 8 | 3-class from original WESAD labels |
| CASE | arousal/valence | 30+ | physiological signals | 10 s | 50 Hz | 8 | subject-wise annotation mean |
| EmoWear | arousal/valence | 48 | E4 + BioHarness signals | 10 s | 50 Hz | 8 | global median of SAM score |

### WESAD

設定:

- 3クラス: `cfg/dataset/wesad_3class.yaml`
- 3クラス shuffle: `cfg/dataset/wesad_3class_target_shuffle.yaml`
- 2クラス: `cfg/dataset/wesad_binary_processed.yaml`

処理:

- 実装: `data_processing/wesad.py`
- 元データ: WESAD の被験者ごとの `.pkl`。
- 使用データ: chest sensor。
- 使用チャンネル: ECG, EDA, EMG, Resp, Temp, ACC x, ACC y, ACC z。
- Sampling rate: 700 Hz。
- Window: 5秒、つまり `window_size=3500`。
- 入力形状: `(batch, 8, 3500)`。
- 被験者ごとに最初の `calibration_windows=60` windowで標準化器をfitし、その被験者内の全windowへ適用する。
- 2クラスでは stress vs no-stress を扱う。
- 3クラスでは baseline / stress / amusement を扱う。
- meditation は3クラス分類では除外する。
- 被験者ごとにwindow化し、1D-CNNに入力する。

ラベル:

- 2クラス stress/no-stress:
  - stress: original label `2`
  - no-stress: baseline `1`, amusement `3`, meditation `4`
- 3クラス:
  - baseline: original label `1`
  - stress: original label `2`
  - amusement: original label `3`
  - meditation `4` は除外

重要な分析:

- WESAD通常順序では、target stream の batch 内ラベル構成が強く偏る。
- 特に同一ラベルのwindowが連続するブロック構造がある。
- TentやOFTTAが通常順序で崩れる原因として、current batch BN統計が偏ったbatchに強く引きずられることが確認された。
- shuffle条件では既存手法の性能が上がるため、batch構成依存性が重要な問題である。

### CASE

設定:

- Arousal: `cfg/dataset/case_arousal.yaml`
- Valence: `cfg/dataset/case_valence.yaml`

処理:

- 実装: `data_processing/case.py`
- 元データ: `Personalized_Affective_Computing/archives/Case` のCASE raw data。
- 使用データ: physiological signal と joystick annotation。
- 使用チャンネル: ECG, BVP, GSR, RSP, TEMP, EMG zygomaticus, EMG corrugator, EMG trapezius。
- 10秒window、5秒stride、50 Hz。
- 入力形状: `(batch, 8, 500)`。
- 元の1000 Hz信号にwinsorization、low-pass filter、downsamplingを行い50 Hzへ変換する。
- 被験者ごとにmin-max normalizationを行う。
- 被験者内平均に基づき arousal / valence を二値化する。
- active annotation区間のみを使う。

ラベル:

- Arousal: joystick `axis1`
- Valence: joystick `axis2`
- 各被験者内の平均値より大きいwindowをclass 1、それ以下をclass 0とする。

### EmoWear

設定:

- Arousal: `cfg/dataset/emowear_arousal.yaml`
- Valence: `cfg/dataset/emowear_valence.yaml`

処理:

- 実装: `data_processing/emowear.py`
- データ場所: `/mnt/c/Users/kojo/Downloads/csv/csv`
- cleaned and synchronized CSV を使用。
- 被験者数: CSV上で48名。
- 総動画試行数: 1776。
- `surveys.csv` のSAM評価と `markers-phase2.csv` の動画区間を `seq, exp` で対応付ける。
- 動画視聴区間 `vidB` から `postB` の信号を切り出す。
- 50 Hzへ線形補間し、10秒window、5秒strideで分割する。
- 使用チャンネルは以下の8ch。

```text
E4 BVP
E4 EDA
E4 SKT
E4 ACC x
E4 ACC y
E4 ACC z
BH3 ECG
BH3 RSP
```

SensorTile ACC/GYRO は、欠損被験者があり、ファイルサイズが大きく、タイムスタンプも不規則であるため、初期比較からは除外した。

ラベル:

- Valence:
  - `valence >= 5.1`: class 1
  - `valence < 5.1`: class 0
- Arousal:
  - `arousal >= 5.0`: class 1
  - `arousal < 5.0`: class 0

注意:

- Arousalでは `19-9UY7` が今回の閾値では片クラスtargetになる。
- そのため、ArousalのMacro-F1解釈では片クラス被験者があることを明記する必要がある。

詳細:

- `docs/emowear_feasibility_report.md`
- `docs/emowear_full_experiment_results.md`

## 主要プログラム

| File | 役割 |
|---|---|
| `train_fixed_loso.py` | 固定epochでLOSO source modelを学習する。 |
| `compare_source_tent_oftta.py` | Source / Norm / Tent / OFTTA / EMA-Tent / DynaMix / 外部TTA手法を横並び評価する。 |
| `adapt.py` | 単一TTA手法を指定して実行する古い形式の入口。 |
| `config.py` | `cfg/default.yaml`, dataset yaml, algorithm yaml を統合する。 |
| `utils.py` | dataset module と model を選択する。 |
| `metrics.py` | Accuracy, per-class F1, Macro-F1, confusion matrix を計算する。 |
| `analyze_batch_bias.py` | batch内ラベル偏りとTTA性能の関係を分析する。 |
| `analyze_subject_tsne.py` | WESAD被験者差のt-SNE可視化を作成する。 |

## TTA手法の実装場所

| Method | 実装 | 設定 |
|---|---|---|
| Source | `TTA/setup.py` 内でadaptなし | `cfg/algorithm/source.yaml` |
| Norm | `TTA/adapt_algorithm/norm.py` | `cfg/algorithm/norm.yaml` |
| Tent | `TTA/adapt_algorithm/tent.py` | `cfg/algorithm/tent.yaml` |
| OFTTA | `TTA/adapt_algorithm/oftta.py` | `cfg/algorithm/oftta.yaml` |
| EMA-Tent | `TTA/adapt_algorithm/ema_tent.py` | `cfg/algorithm/ema_tent.yaml` |
| DynaMix EMA-Tent | `TTA/adapt_algorithm/mi_dynamic_ema_tent.py` | `cfg/algorithm/dynamix_ema_tent.yaml` |
| TEMA | `TTA/adapt_algorithm/tema.py` | `cfg/algorithm/tema.yaml` |
| DUA | `TTA/adapt_algorithm/dua.py` | `cfg/algorithm/dua.yaml` |
| RoTTA | `TTA/adapt_algorithm/rotta.py` | `cfg/algorithm/rotta.yaml` |
| NOTE | `TTA/adapt_algorithm/note.py` | `cfg/algorithm/note.yaml` |
| DELTA | `TTA/adapt_algorithm/delta.py` | `cfg/algorithm/delta.yaml` |

手法のdispatchは `TTA/setup.py` にある。

## DynaMix EMA-Tent の中身

DynaMix EMA-Tent は、EMA-Tentを基準に、BatchNorm統計の混合重みを動的に決める手法である。

詳細説明:

- `新手法の説明文/dynamix_ema_tent_method.md`

### EMA-Tentとの共通点

DynaMix EMA-Tentは以下をEMA-Tentから引き継ぐ。

- Source BN統計をanchorとして保持する。
- Target EMA統計を保持する。
- Current batch統計を取得する。
- BN正規化に `source / target EMA / current batch` の混合統計を使う。
- Entropy minimization でBN affineパラメータ `gamma, beta` を更新する。
- Conv / Linear は更新しない。

### EMA-Tentとの違い

EMA-Tentでは以下のように混合重みを固定する。

```text
w_source = 0.2
w_ema    = 0.5
w_batch  = 0.3
```

DynaMix EMA-Tentでは、この固定混合比をやめ、batchごとに以下で決める。

まず、batch平均予測分布のentropyからgateを計算する。

```text
p_bar = (1 / B) sum_i p_i
g = H(p_bar) / log C
```

`g` が大きいほど、batch予測が複数クラスに分散しているとみなす。`g` が小さい場合は、batchが単一クラスに偏っている可能性が高い。

Current batch統計の使用量:

```text
w_batch^t = w_batch_max * g_t
```

Target EMA統計の信頼度:

```text
N_eff^t = N_eff^(t-1) + B * g_t
rho_t = 1 - exp(-N_eff^t / K)
```

Source / EMAへの配分:

```text
w_ema^t = (1 - w_batch^t) * rho_t
w_source^t = (1 - w_batch^t) * (1 - rho_t)
```

そのため、DynaMix EMA-Tentは以下の挙動になる。

- 初期はsource統計を強めに使う。
- 信頼できるtarget batchを観測するほどtarget EMA統計を強める。
- batch予測が多様なときのみcurrent batch統計を使う。
- batch予測が単一クラス寄りのときはcurrent batch統計の寄与を下げる。

### DynaMix EMA-Tent の設定

設定ファイル:

```text
cfg/algorithm/dynamix_ema_tent.yaml
```

現在の主要設定:

```yaml
adaption: mi_dynamic_ema_tent
tent_lr: 0.0001
tent_steps: 1
episodic: false

ema_tent_source_weight: 0.2
ema_tent_ema_weight: 0.5
ema_tent_batch_weight: 0.3
ema_momentum: 0.80

mi_gate: false
gate_aware_ema: false
dynamic_mixture_weights: true
gate_threshold: 0.2

w_batch_max: 0.3
rho_k: 128
```

実装上は `mi_dynamic_ema_tent.py` を再利用しているが、DynaMix設定では `mi_gate: false`, `gate_aware_ema: false` としている。したがって、MI gateやgate-aware EMAを使う版ではなく、EMA-Tentに dynamic mixture weights だけを入れた版である。

## MI-gated Dynamic EMA-Tent との関係

最初に、より複雑な改良として MI-gated Dynamic EMA-Tent も実装した。

実装:

- `TTA/adapt_algorithm/mi_dynamic_ema_tent.py`

説明:

- `新手法の説明文/mi_dynamic_ema_tent_method.md`

ただし、WESADで部品ごとの有効性を検証した結果、MI gateやgate-aware EMAよりも、混合重みを動的に決める部分が最も重要だった。そのため、論文の主提案としては、よりシンプルな DynaMix EMA-Tent を採用している。

## 実行方法

### WESAD 3class Source学習

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/wesad/train_fixed_loso_wesad_3class.sh
```

### WESAD 3class TTA比較

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source tent oftta ema_tent dynamix_ema_tent \
  --run_name WESAD3class_TentOFTTAEMATent_DynaMixEMATent
```

外部TTA手法も含める場合:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/wesad_3class.yaml \
  --resume ./ckpt_3class \
  --methods source tent oftta dynamix_ema_tent tema dua rotta note delta \
  --run_name WESAD3class_SourceTentOFTTADynaMixTEMA_DUA_RoTTA_NOTE_DELTA
```

### CASE

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/case/train_fixed_loso_case_arousal.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/case/train_fixed_loso_case_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_arousal.yaml \
  --resume ./ckpt_case_arousal \
  --methods source norm tent oftta ema_tent dynamix_ema_tent tema dua rotta note delta \
  --run_name CASE_arousal_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  python compare_source_tent_oftta.py \
  --dataset_cfg ./cfg/dataset/case_valence.yaml \
  --resume ./ckpt_case_valence \
  --methods source norm tent oftta ema_tent dynamix_ema_tent tema dua rotta note delta \
  --run_name CASE_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA
```

### EmoWear

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/train_fixed_loso_emowear_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/train_fixed_loso_emowear_arousal.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/compare_source_tta_emowear_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/compare_source_tta_emowear_arousal.sh
```

## 主要結果

以下の表は、各CSVに保存されている `Average` 行のMacro-F1をそのまま記載している。`Std` 行は平均計算に含めていない。

### WESAD 2class stress/no-stress, all TTA

CASE / EmoWear の arousal / valence と同じ二値分類の比較として、WESADも stress vs no-stress の結果を用意した。

結果保存先:

```text
logs/wesad/compare_source_tent_oftta/260616_195900_WESAD2class_stress_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/
```

タスク:

- class 0: no-stress
- class 1: stress

平均Macro-F1:

| Method | Macro-F1 |
|---|---:|
| EMA-Tent | 0.7672 |
| DynaMix EMA-Tent | 0.7502 |
| TEMA | 0.7063 |
| DUA | 0.7061 |
| Source | 0.6737 |
| DELTA | 0.6719 |
| NOTE | 0.6112 |
| RoTTA | 0.6110 |
| Norm | 0.6063 |
| Tent | 0.6061 |
| OFTTA | 0.5646 |

解釈:

- WESADの二値stress分類では、EMA-Tentが最良で、DynaMix EMA-Tentが2番目に高い。
- WESAD 3classより分類問題は単純だが、通常順序streamのbatch偏りは残る。
- EMA系手法が強いことから、WESADではcurrent batch統計だけに依存するより、source統計と過去target統計を保持する設計が有効である。

### WESAD 3class, all TTA

結果保存先:

```text
logs/wesad/compare_source_tent_oftta/260616_201019_WESAD3class_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/
```

平均Macro-F1:

| Method | Macro-F1 |
|---|---:|
| EMA-Tent | 0.6169 |
| DynaMix EMA-Tent | 0.6133 |
| TEMA | 0.5629 |
| DUA | 0.5519 |
| Source | 0.5285 |
| OFTTA | 0.4499 |
| DELTA | 0.4059 |
| RoTTA | 0.4015 |
| NOTE | 0.4012 |
| Norm | 0.3964 |
| Tent | 0.3963 |

解釈:

- WESAD通常順序ではEMA-TentとDynaMix EMA-Tentが上位であり、両者の差は小さい。
- DynaMix EMA-Tentは固定混合比率を使わずに、ハイパーパラメータ調整済みのEMA-Tentに近い性能を示した。
- TentはNorm的なBN統計利用だけではなく、batch構成に強く依存し、通常順序では崩れやすい。
- OFTTAはsource anchoringにより過度な適応を抑える設計だが、WESADではtarget被験者への適応が必要であり、通常順序の偏りも重なって性能改善が限定的であった。

### CASE Arousal, all TTA

結果保存先:

```text
logs/case/compare_source_tent_oftta/260616_200625_CASE_arousal_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/
```

平均Macro-F1:

| Method | Macro-F1 |
|---|---:|
| Tent | 0.5622 |
| Norm | 0.5618 |
| NOTE | 0.5549 |
| RoTTA | 0.5548 |
| OFTTA | 0.5457 |
| DELTA | 0.5382 |
| EMA-Tent | 0.4969 |
| DynaMix EMA-Tent | 0.4700 |
| TEMA | 0.4308 |
| DUA | 0.4280 |
| Source | 0.4073 |

### CASE Valence, all TTA

結果保存先:

```text
logs/case/compare_source_tent_oftta/260616_200628_CASE_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/
```

平均Macro-F1:

| Method | Macro-F1 |
|---|---:|
| Norm | 0.5671 |
| Tent | 0.5670 |
| NOTE | 0.5666 |
| RoTTA | 0.5666 |
| OFTTA | 0.5544 |
| EMA-Tent | 0.5252 |
| DynaMix EMA-Tent | 0.5197 |
| DELTA | 0.4971 |
| TEMA | 0.4653 |
| DUA | 0.4629 |
| Source | 0.4545 |

解釈:

- CASEではTent / NOTE / RoTTA が強い。
- WESADと異なり、DynaMix EMA-Tentが常に最良ではない。
- これは、データセットごとのstream構造・ラベル分布・被験者差が異なるためと考えられる。
- CASEではNormとTentがほぼ同値であり、BN affine更新よりも推論時BN統計の利用が支配的に効いている可能性がある。

### EmoWear Valence

結果保存先:

```text
logs/emowear/compare_source_tent_oftta/260616_194526_EmoWear_valence_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/
```

平均Macro-F1:

| Method | Macro-F1 |
|---|---:|
| DynaMix EMA-Tent | 0.5026 |
| EMA-Tent | 0.5006 |
| OFTTA | 0.4904 |
| TEMA | 0.4887 |
| DUA | 0.4879 |
| DELTA | 0.4862 |
| Tent | 0.4830 |
| Norm | 0.4829 |
| Source | 0.4805 |
| NOTE | 0.4804 |
| RoTTA | 0.4804 |

### EmoWear Arousal

結果保存先:

```text
logs/emowear/compare_source_tent_oftta/260616_194526_EmoWear_arousal_SourceNormTentOFTTAEMATentDynaMixTEMA_DUA_RoTTA_NOTE_DELTA/
```

平均Macro-F1:

| Method | Macro-F1 |
|---|---:|
| DELTA | 0.4705 |
| EMA-Tent | 0.4633 |
| DynaMix EMA-Tent | 0.4630 |
| TEMA | 0.4596 |
| DUA | 0.4588 |
| Norm | 0.4583 |
| Tent | 0.4581 |
| Source | 0.4581 |
| NOTE | 0.4557 |
| RoTTA | 0.4554 |
| OFTTA | 0.4490 |

解釈:

- EmoWear ValenceではDynaMix EMA-Tentが平均Macro-F1で最良。
- EmoWear ArousalではDELTAが最良。
- Arousalでは `19-9UY7` が二値化後に片クラスtargetとなっており、Macro-F1解釈には注意が必要である。

### 結果ログと図

各比較ログには、CSV/Markdown表と図が保存されている。主に見るべきファイルは以下である。

| File | 内容 |
|---|---|
| `source_tent_oftta_comparison.csv` | 被験者別のAccuracy, class-wise F1, Macro-F1。 |
| `source_tent_oftta_comparison.md` | CSVをMarkdown表にしたもの。 |
| `source_tent_oftta_average.png` | 手法ごとの平均性能の棒グラフ。 |
| `source_tent_oftta_macro_f1.png` | 被験者別Macro-F1の比較図。 |
| `source_tent_oftta_accuracy.png` | 被験者別Accuracyの比較図。 |
| `ema_tent_batch_diagnostics.csv` | EMA-Tent / DynaMix EMA-Tent のbatch単位診断ログ。 |
| `tta_stream_summary.csv` | batch構成やstream統計の要約。 |

## WESADで重要な補助分析

### Batch内ラベル偏り

関連プログラム:

- `analyze_batch_bias.py`
- `scripts/wesad/analyze_batch_bias_wesad_3class.sh`
- `scripts/wesad/analyze_batch_bias_ema_tent_wesad_3class.sh`

関連ログ:

```text
logs/wesad/batch_bias_analysis/
```

目的:

- batch内クラス分布を可視化する。
- batch偏りとTTA性能変化の相関を見る。
- 通常順序とshuffle条件の差を確認する。

### 被験者差のt-SNE可視化

関連プログラム:

- `analyze_subject_tsne.py`

関連ログ:

```text
logs/wesad/subject_tsne/
```

目的:

- 同じ感情ラベルでも被験者間で生体信号特徴が大きく異なることを可視化する。
- 個人差が大きいため、未知被験者へのTTAが必要であるという導入の根拠にする。

### BN統計 vs affine更新の切り分け

関連ログ:

```text
logs/wesad/compare_source_tent_oftta/260526_003406_WESAD3class_SourceNormTent_sequential_BNstats_vs_affineUpdate/
logs/wesad/compare_source_tent_oftta/260526_003406_WESAD3class_SourceNormTent_shuffle_BNstats_vs_affineUpdate/
```

目的:

- Tentの性能変化が、BN統計のcurrent batch利用によるものか、BN affineパラメータ更新によるものかを切り分ける。
- WESAD通常順序では、current batch BN統計自体が偏ったbatchに引きずられることが大きい。

## 外部TTA手法についての注意

TEMA / DUA / RoTTA / NOTE / DELTA は、公開実装の思想を1D-CNN + BatchNorm1dの生体信号分類へ移植したものである。

詳細:

- `docs/external_tta_methods_research_memo.md`

重要な注意:

- 画像分類向けのaugmentationやmemory更新をそのまま再現しているわけではない。
- 生体信号用に簡略化・調整している。
- 論文では「official implementation完全再現」ではなく、「1D-CNN/BN設定に合わせた比較実装」と明記する必要がある。

## 現在の論文ストーリー案

1. 生体信号感情推定では被験者差が大きい。
2. Domain adaptation / domain generalization ではなく、未知被験者streamに対してTTAを使えるかを検証する。
3. 代表的TTAである Tent / OFTTA を WESAD に適用した。
4. そのままでは通常順序のWESADで性能が限定的、または崩れる。
5. 原因分析として、WESADの通常順序にはブロック構造があり、batch内ラベルが大きく偏ることを確認した。
6. Current batch BN統計だけに依存すると、単一クラス寄りbatchで正規化統計が崩れる。
7. EMA-Tentを導入し、source統計・target EMA統計・current batch統計を混合することで改善した。
8. さらに、固定混合重みへの依存を減らすため、DynaMix EMA-Tentを提案した。
9. WESADではEMA-Tent / DynaMix EMA-Tentが上位であり、source/EMA/current batch統計を組み合わせる設計が有効である。
10. DynaMix EMA-Tentは、固定混合比率を使うEMA-Tentと近い性能を維持しつつ、batchごとに混合比率を動的に決められる。
11. CASE / EmoWearでは最良手法はタスクによって異なるが、外部データセットでもDynaMix EMA-Tentは比較対象として成立する。

## 共有時の注意

- `logs/` と `ckpt_*` は容量が大きい可能性がある。
- GitHubに全ログやcheckpointをpushする場合は容量に注意する。
- `data/` 配下のprocessed cacheも含めると容量が増える。
- EmoWear raw data は `/mnt/c/Users/kojo/Downloads/csv/csv` を参照しており、このリポジトリ内にはraw dataをコピーしていない。
- WESAD raw data も別ディレクトリ参照であり、dataset yaml の `dataset_dir` を環境に合わせて変更する必要がある。

## 先生に見てもらいたいポイント

- WESAD通常順序で既存TTAが失敗する原因を、batch構成とBN統計の観点から説明できているか。
- EMA-Tent / DynaMix EMA-Tent の提案が、既存研究との差分として十分に明確か。
- DynaMix EMA-Tentを主提案にするか、EMA-Tentを主提案にしてDynaMixを拡張実験にするか。
- CASE / EmoWear の結果を主論文に入れるか、追加検証として扱うか。
- 外部TTA手法の比較をどこまで主張してよいか。
