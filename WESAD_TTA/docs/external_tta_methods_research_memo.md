# External TTA Methods Research Memo

## 目的

WESAD_TTA の既存比較に、外部リポジトリ由来の TTA 手法を追加した。

比較対象:

```text
Source
Tent
OFTTA
DynaMix EMA-Tent
TEMA
DUA
RoTTA
NOTE
DELTA
```

対象データセット:

```text
WESAD 3class
CASE arousal
CASE valence
```

## 参照した外部リポジトリ

```text
/home/shinsaku-t/work/naist_reserch/RealisticTTA
/home/shinsaku-t/work/naist_reserch/DUA
/home/shinsaku-t/work/naist_reserch/RoTTA
/home/shinsaku-t/work/naist_reserch/NOTE
/home/shinsaku-t/work/naist_reserch/DELTA
```

## 実装方針

外部リポジトリは主に画像分類、CIFAR/ImageNet、BatchNorm2d を前提としている。
そのため、WESAD_TTA ではコードをそのまま import せず、1D-CNN と BatchNorm1d に
合わせた軽量 wrapper として移植した。

TTA 計算ではラベルを使わない。ラベルは評価後の診断ログにのみ保存する。

## 追加ファイル

```text
TTA/adapt_algorithm/tema.py
TTA/adapt_algorithm/dua.py
TTA/adapt_algorithm/rotta.py
TTA/adapt_algorithm/note.py
TTA/adapt_algorithm/delta.py

cfg/algorithm/tema.yaml
cfg/algorithm/dua.yaml
cfg/algorithm/rotta.yaml
cfg/algorithm/note.yaml
cfg/algorithm/delta.yaml
```

既存ファイルの変更:

```text
TTA/setup.py
compare_source_tent_oftta.py
```

## 手法別の移植内容

### TEMA

参照元:

```text
RealisticTTA/methods/bn.py
RealisticTTA/methods/norm.py
```

RealisticTTA の `norm_ema` / `EMABatchNorm` を BatchNorm1d 用に移植した。

処理:

```text
1. train forward で test batch を使って BN running mean/var を EMA 更新
2. eval forward で更新後の BN 統計を使って予測
3. パラメータ更新は行わない
```

注意:

```text
TEMA は推論対象 batch 自身で BN running stats を更新した後、その更新後統計で同じ batch を推論する。
これは RealisticTTA 実装の norm_ema に合わせた挙動である。
```

### DUA

参照元:

```text
DUA/methods/dua.py
```

DUA は test-time に BN running stats を更新する手法である。
原著実装では画像 augmentation と複数反復を用いるが、WESAD/CASE の1D信号では同じ
augmentation をそのまま使えないため、以下の最小版にした。

処理:

```text
1. batch ごとに BN momentum を decay させる
2. train forward で BN running stats を更新
3. eval forward で更新後統計を使って予測
4. パラメータ更新は行わない
```

主な設定:

```text
dua_initial_momentum = 0.1
dua_decay_factor = 0.94
dua_min_momentum = 0.005
```

### RoTTA

参照元:

```text
RoTTA/core/adapter/rotta.py
RoTTA/core/utils/memory.py
RoTTA/core/utils/bn_layers.py
```

RoTTA の主要要素である class-balanced memory、teacher EMA、teacher-student consistency
を1D-CNN用に移植した。

処理:

```text
1. teacher EMA model で pseudo label と entropy を計算
2. pseudo label ごとの class-balanced memory に sample を保存
3. 一定間隔で memory 上の sample を使って student model の BN affine を更新
4. student から teacher EMA を更新
```

注意:

```text
原著RoTTAの画像用 strong augmentation は使っていない。
WESAD/CASEでは生体信号のaugmentation設計が別途必要なため、今回は同一sample上のconsistency更新にした。
```

### NOTE

参照元:

```text
NOTE/learner/note.py
NOTE/utils/memory.py
```

NOTE の FIFO memory と entropy minimization を1D-CNN用に移植した。

処理:

```text
1. 入力 batch の sample を FIFO memory に保存
2. memory が一定数たまったら memory 上で entropy minimization
3. 更新対象は BN affine パラメータのみ
```

注意:

```text
NOTE の原著実装には sample-wise online loop や複数memory形式がある。
今回は WESAD_TTA の batch-based evaluation loop に合わせ、FIFO memory 版として実装した。
```

### DELTA

参照元:

```text
DELTA/algorithms/delta.py
```

DELTA の TBR と entropy/class-distribution balancing を BatchNorm1d 用に移植した。

処理:

```text
1. BatchNorm1d を TBR1d に置換
2. batch 統計と過去 running 統計から renormalization を行う
3. entropy weighted loss で BN affine を更新
4. qhat による予測分布の移動平均を使い、class balance weight を加える
```

注意:

```text
原著DELTAはBatchNorm2d画像分類前提である。
今回は1D時系列信号向けにBatchNorm1d版TBRとして実装した。
```

## 平均結果

集計CSV:

```text
logs/summary_all_tta/source_tent_oftta_dynamix_tema_dua_rotta_note_delta_summary.csv
```

### WESAD 3class

| Method | Accuracy | Macro-F1 | Wins |
| --- | ---: | ---: | ---: |
| Source | 0.6915 | 0.5285 | 3 |
| Tent | 0.4553 | 0.3970 | 0 |
| OFTTA | 0.5231 | 0.4499 | 0 |
| DynaMix EMA-Tent | 0.7201 | 0.6172 | 9 |
| TEMA | 0.7135 | 0.5629 | 2 |
| DUA | 0.7054 | 0.5519 | 0 |
| RoTTA | 0.4597 | 0.4015 | 0 |
| NOTE | 0.4595 | 0.4012 | 0 |
| DELTA | 0.4712 | 0.4027 | 1 |

### CASE Arousal

| Method | Accuracy | Macro-F1 | Wins |
| --- | ---: | ---: | ---: |
| Source | 0.4959 | 0.4069 | 1 |
| Tent | 0.5673 | 0.5624 | 8 |
| OFTTA | 0.5663 | 0.5464 | 6 |
| DynaMix EMA-Tent | 0.5250 | 0.4738 | 4 |
| TEMA | 0.5047 | 0.4296 | 2 |
| DUA | 0.5028 | 0.4270 | 0 |
| RoTTA | 0.5589 | 0.5541 | 2 |
| NOTE | 0.5590 | 0.5542 | 2 |
| DELTA | 0.5421 | 0.5290 | 7 |

### CASE Valence

| Method | Accuracy | Macro-F1 | Wins |
| --- | ---: | ---: | ---: |
| Source | 0.5348 | 0.4540 | 4 |
| Tent | 0.5868 | 0.5719 | 7 |
| OFTTA | 0.5868 | 0.5531 | 4 |
| DynaMix EMA-Tent | 0.5649 | 0.5178 | 6 |
| TEMA | 0.5404 | 0.4649 | 1 |
| DUA | 0.5386 | 0.4622 | 1 |
| RoTTA | 0.5830 | 0.5675 | 5 |
| NOTE | 0.5825 | 0.5670 | 5 |
| DELTA | 0.5219 | 0.5057 | 3 |

## 図

横断比較:

```text
logs/summary_all_tta/all_datasets_all_tta_macro_f1.pdf
logs/summary_all_tta/all_datasets_all_tta_macro_f1.png
```

各データセット別:

```text
logs/wesad/compare_source_tent_oftta/260616_171511_WESAD3class_SourceTentOFTTADynaMixTEMA_DUA_RoTTA_NOTE_DELTA/wesad_3class_all_tta_macro_f1.pdf
logs/case/compare_source_tent_oftta/260616_171715_CASE_arousal_SourceTentOFTTADynaMixTEMA_DUA_RoTTA_NOTE_DELTA/case_arousal_all_tta_macro_f1.pdf
logs/case/compare_source_tent_oftta/260616_171734_CASE_valence_SourceTentOFTTADynaMixTEMA_DUA_RoTTA_NOTE_DELTA/case_valence_all_tta_macro_f1.pdf
```

## 現時点の解釈

WESAD では DynaMix EMA-Tent が最も高い平均 Macro-F1 を示した。
TEMA と DUA は Source より高い accuracy を維持しつつ Macro-F1 も改善するが、
DynaMix EMA-Tent には届かない。RoTTA、NOTE、DELTA は WESAD 通常順序では
平均性能が低く、単一クラス block stream での memory/entropy 更新が不安定になっている
可能性がある。

CASE では WESAD と異なり、Tent、RoTTA、NOTE、OFTTA が強い。
特に CASE arousal / valence では RoTTA と NOTE が Tent に近い平均 Macro-F1 を示した。
これは CASE では WESAD より current batch や短期memoryを使った更新が破綻しにくい
可能性を示す。

## 論文での注意点

外部手法はすべて元実装を完全再現したものではなく、WESAD_TTA の 1D-CNN /
BatchNorm1d / batch-based evaluation に合わせた移植実装である。

特に以下は明記する必要がある。

```text
DUA:
  画像augmentationは使わず、BN running statistics更新のみを移植した。

RoTTA:
  strong augmentationは使わず、class-balanced memory + teacher EMA consistencyを移植した。

NOTE:
  sample-wise online loopではなく、batch-based loopに合わせたFIFO memory entropy minimizationとした。

DELTA:
  BatchNorm2d用TBRをBatchNorm1dに拡張し、BN affine更新のみを行った。
```

このため、論文では「official implementationを1D生体信号モデルに適用可能な形へ
adaptedした比較」と表現するのが安全である。
