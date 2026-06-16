# EmoWear CSV Feasibility Report

## 確認対象

- データ場所: `/mnt/c/Users/kojo/Downloads/csv/csv`
- 形式: cleaned and synchronized CSV package
- 被験者ディレクトリ数: 48
- 欠番: `35`
- 主要ファイル:
  - `surveys.csv`
  - `markers-phase2.csv`
  - `markers-unique.csv`
  - `params.csv`
  - `signals-e4-*.csv`
  - `signals-bh3-*.csv`
  - `signals-front-*.csv`, `signals-back-*.csv`, `signals-water-*.csv`

## 結論

EmoWear は Valence / Arousal の分類タスクに利用可能である。

理由は、`surveys.csv` に各動画試行ごとの SAM 評価値があり、`markers-phase2.csv` に同じ `seq, exp` に対応する動画開始時刻 `vidB` と終了時刻 `postB` があるためである。したがって、各動画区間の生体信号を切り出し、対応する valence / arousal をラベルとして付与できる。

## ラベルと試行数

`surveys.csv` の列:

- `seq`
- `exp`
- `valence`
- `arousal`
- `dominance`
- `liking`
- `familiarity`

`markers-phase2.csv` の列:

- `seq`
- `exp`
- `newExp`
- `preB`
- `vidB`
- `postB`
- `surveyB`
- `walkB`
- `walkE`
- `walkDetect`
- `walkFinish`

`seq, exp` で `surveys.csv` と `markers-phase2.csv` を結合できる。

確認結果:

- 被験者数: 48
- 総動画試行数: 1776
- 45名は38試行
- 3名は一部欠損し、19 / 23 / 24試行
- 動画区間長: 平均約61.24秒

SAM値の概要:

| 指標 | 平均 | 中央値 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| valence | 5.286 | 5.1 | 1.0 | 9.0 |
| arousal | 4.702 | 5.0 | 1.0 | 9.0 |
| dominance | 5.770 | 6.0 | 1.0 | 9.0 |

Valence / Arousal を二値分類にする場合、グローバル中央値で分割すると比較的均衡する。

- Valence: `>= 5.1` が890試行、`< 5.1` が886試行
- Arousal: `>= 5.0` が908試行、`< 5.0` が868試行

## 信号ファイルの利用可能性

全48名に存在する信号:

- `signals-e4-bvp.csv`
- `signals-e4-eda.csv`
- `signals-e4-skt.csv`
- `signals-e4-acc.csv`
- `signals-bh3-ecg.csv`
- `signals-bh3-rsp.csv`
- `signals-bh3-acc.csv`

一部被験者のみ存在する信号:

- `signals-front-acc.csv`: 46名
- `signals-front-gyro.csv`: 46名
- `signals-back-acc.csv`: 44名
- `signals-back-gyro.csv`: 44名
- `signals-water-acc.csv`: 21名
- `signals-water-gyro.csv`: 21名

最初の比較実験では、全被験者に揃っている E4 + BioHarness 3 を使うのがよい。

## サンプリング間隔

代表被験者 `01-9TZK` で確認したサンプリング間隔:

| 信号 | 推定サンプリング間隔 | 推定周波数 |
|---|---:|---:|
| E4 BVP | 約0.0156秒 | 約64 Hz |
| E4 EDA | 約0.2497秒 | 約4 Hz |
| E4 SKT | 約0.2497秒 | 約4 Hz |
| E4 ACC | 約0.0312秒 | 約32 Hz |
| BH3 ECG | 約0.0040秒 | 約250 Hz |
| BH3 RSP | 約0.0400秒 | 約25 Hz |
| BH3 ACC | 約0.0100秒 | 約100 Hz |

E4 / BH3 は確認範囲ではほぼ等間隔であり、WESAD / CASE と同様の window 化が可能である。

## SensorTile ACC/GYRO について

ST SensorTile.box の `front`, `back`, `water` 系 ACC/GYRO は、ファイルサイズが非常に大きく、さらにタイムスタンプが不規則である。

確認した総サイズ:

- `signals-front-acc.csv`: 約13.70 GB
- `signals-front-gyro.csv`: 約5.64 GB
- `signals-back-acc.csv`: 約9.68 GB
- `signals-back-gyro.csv`: 約3.89 GB
- `signals-water-acc.csv`: 約5.06 GB
- `signals-water-gyro.csv`: 約1.83 GB

したがって、初回実験では SensorTile を除外し、E4 / BH3 で Valence / Arousal 分類を成立させるのが現実的である。SensorTile を使う場合は、各動画区間ごとに必要な範囲だけ読み出し、`numpy.interp` で等間隔化してから特徴量化または window 化する。

## 推奨する前処理方針

1. `surveys.csv` と `markers-phase2.csv` を `seq, exp` で結合する。
2. 各試行について `vidB` から `postB` までを動画視聴区間として切り出す。
3. E4 / BH3 の信号を共通周波数にリサンプリングする。
   - まずは 32 Hz または 64 Hz が扱いやすい。
   - ECGを重視する場合は 100 Hz 以上も検討する。
4. 一定長 window に分割する。
   - 例: 10秒 window、5秒 stride。
5. window に試行単位の valence / arousal ラベルを付与する。
6. 分類ラベルは最初は二値分類にする。
   - Valence: `valence >= 5.1` を high、`< 5.1` を low。
   - Arousal: `arousal >= 5.0` を high、`< 5.0` を low。
7. 評価は LOSO 形式が望ましい。
   - 未知被験者を target とし、Source / Tent / OFTTA / EMA-Tent / DynaMix EMA-Tent などを比較できる。

## 注意点

- 49名データセットと説明されているが、CSVディレクトリ上では48被験者分を確認した。
- 3名は38試行すべてが揃っていない。
- SensorTile は欠損被験者があり、全被験者比較に入れると対象被験者数が減る。
- TTA実験では、ラベルは評価にのみ使い、適応計算には使わない。
- SAM値の二値化は論文上で明確に定義する必要がある。現時点ではグローバル中央値分割が最も単純で、クラス数もほぼ均衡する。

## 次に実装する場合の最小構成

実装済みの最小構成:

- `data_processing/emowear.py`
- `cfg/dataset/emowear_valence.yaml`
- `cfg/dataset/emowear_arousal.yaml`
- `scripts/emowear/train_fixed_loso_emowear_valence.sh`
- `scripts/emowear/train_fixed_loso_emowear_arousal.sh`
- `scripts/emowear/compare_source_tta_emowear_valence.sh`
- `scripts/emowear/compare_source_tta_emowear_arousal.sh`

初期実験では E4 BVP / EDA / SKT / ACC と BH3 ECG / RSP / ACC を使い、SensorTile は後続の拡張として扱う。

## 実装した前処理

既存の WESAD / CASE 実験と同じ 1D-CNN + BatchNorm の入力にそろえるため、EmoWear も `(window, channel)` 形式の固定長時系列へ変換する。

使用する8チャンネル:

1. E4 BVP
2. E4 EDA
3. E4 SKT
4. E4 ACC x
5. E4 ACC y
6. E4 ACC z
7. BH3 ECG
8. BH3 RSP

この8チャンネルを選んだ理由:

- 全48被験者に存在する。
- WESAD / CASE と同様に生理信号中心の入力を構成できる。
- SensorTile ACC/GYRO と異なり、確認範囲ではタイムスタンプがほぼ等間隔である。
- 既存の `StressCNN1D` が `num_channels=8` でそのまま利用できる。

前処理手順:

1. 被験者フォルダ内の `surveys.csv` と `markers-phase2.csv` を `seq, exp` で結合する。
2. 各試行の動画視聴区間を `vidB` から `postB` と定義する。
3. 各動画区間を 50 Hz の共通時間グリッドへ変換する。
4. 各信号を `numpy.interp` で共通時間グリッドへ補間する。
5. 10秒 window、5秒 stride で分割する。
6. 各windowには、その動画試行のSAMラベルを付与する。
7. 被験者ごとに各チャンネルを `StandardScaler` で標準化する。
8. PyTorch Conv1d 用に `(batch, channel, time)` へ変換して DataLoader を作成する。

ラベル定義:

- Valence: `valence >= 5.1` を high、`valence < 5.1` を low。
- Arousal: `arousal >= 5.0` を high、`arousal < 5.0` を low。

この閾値は、確認したCSV全体のグローバル中央値である。二値分類としてクラス数がほぼ均衡するため、最初の比較実験ではこの設定を採用する。

## 1D-CNN + BN への接続

`utils.py` に `dataset == "emowear"` の分岐を追加し、既存の `train_fixed_loso.py` と `compare_source_tent_oftta.py` から EmoWear を呼べるようにした。

モデルは既存の `models/cnn1d.py` の `StressCNN1D` をそのまま使う。構造は以下である。

- Conv1d
- ReLU
- BatchNorm1d
- MaxPool1d
- Conv1d
- ReLU
- BatchNorm1d
- MaxPool1d
- Conv1d
- ReLU
- BatchNorm1d
- MaxPool1d
- Global Average Pooling
- Linear classifier

Tent / OFTTA / EMA-Tent / DynaMix EMA-Tent などのTTA手法は、この BatchNorm1d 層を対象として既存実装と同じ形式で適用できる。

## 実行コマンド

Source model の固定epoch LOSO学習:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/train_fixed_loso_emowear_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/train_fixed_loso_emowear_arousal.sh
```

TTA比較:

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/compare_source_tta_emowear_valence.sh
```

```bash
PYTHON=/home/shinsaku-t/work/naist_reserch/.venv/bin/python \
  bash scripts/emowear/compare_source_tta_emowear_arousal.sh
```

比較対象:

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

## 動作確認結果

以下の確認を実施した。

- `data_processing/emowear.py` と `utils.py` の `py_compile`
- `scripts/emowear/*.sh` の `bash -n`
- 代表被験者 `01-9TZK` の前処理
- 全48名分の processed cache 作成
- DataLoader から1 batchを取り出し、既存1D-CNNへ forward

代表被験者 `01-9TZK` の前処理結果:

- Valence: `X=(413, 500, 8)`, class counts `{0: 236, 1: 177}`
- Arousal: `X=(413, 500, 8)`, class counts `{0: 162, 1: 251}`

全被験者 processed cache:

| Task | subjects | windows | class 0 | class 1 | cache size |
|---|---:|---:|---:|---:|---:|
| Valence | 48 | 19303 | 9538 | 9765 | 約90 MB |
| Arousal | 48 | 19303 | 9391 | 9912 | 約90 MB |

DataLoader / model forward 確認:

- 入力batch形状: `(64, 8, 500)`
- ラベル形状: `(64,)`
- 1D-CNN出力形状: `(64,)`

注意点として、Arousalでは `19-9UY7` がグローバル中央値閾値では class 0 のみになった。LOSO全体のSource学習は可能だが、被験者別Macro-F1の解釈では、このような片クラスtarget被験者が存在することを明記する必要がある。

## 論文記載用メモ

EmoWearでは、全被験者に共通して利用可能なEmpatica E4およびZephyr BioHarness 3由来の生理信号を用いた。具体的には、BVP, EDA, SKT, E4 ACCの3軸, ECG, RSPの計8チャンネルを用いた。各被験者について、`surveys.csv` に記録されたSAM評価と `markers-phase2.csv` の動画提示区間を対応付け、動画視聴区間 `vidB` から `postB` の信号を抽出した。抽出した信号は50 Hzの共通時間グリッドへ線形補間し、10秒窓、5秒ストライドで分割した。各窓には対応する動画試行のvalenceまたはarousalスコアを付与し、グローバル中央値に基づいて二値化した。Valenceは5.1以上をhigh、5.1未満をlow、Arousalは5.0以上をhigh、5.0未満をlowとした。モデルはWESADおよびCASE実験と同じ1D-CNNを用い、各畳み込みブロックにBatchNorm1dを含む構造とした。これにより、Source, Tent, OFTTA, EMA-Tentおよび提案手法を同一のTTA評価枠組みで比較できる。

ST SensorTile.box のACC/GYROは、タイムスタンプが不規則であり、また欠損被験者が存在しファイルサイズも大きいため、初期比較からは除外した。これらを利用する場合は、各動画区間ごとに必要な範囲を読み出し、線形補間により等間隔化したうえで追加チャンネルとして扱う必要がある。
