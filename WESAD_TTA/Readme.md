# WESAD_TTA ディレクトリ構成ガイド

このディレクトリは、WESAD データセットを用いたストレス二値分類について、1D-CNN の LOSO
評価と Test-Time Adaptation を実行するための実験コードをまとめたものです。

元の `self_learning_wesad/stress_detection_1d_cnn_loso_pytorch.py` は、前処理、モデル定義、
学習、評価、設定値、保存処理が 1 ファイルに集約されていました。`WESAD_TTA` では OFTTA の構成を
参考に、責務ごとにファイルとディレクトリを分けています。これにより、前処理だけを修正したい場合、
モデルだけを差し替えたい場合、Tent 以外の TTA 手法を追加したい場合に、変更箇所を追いやすくしています。

## 全体の実行フロー

基本的な流れは次の通りです。

1. `cfg` の YAML ファイルからデータセット設定とアルゴリズム設定を読み込む
2. `data_processing` で WESAD の被験者データを読み込み、5 秒窓へ分割する
3. WESAD のラベルを「非ストレス」と「ストレス」の二値へ変換する
4. 被験者ごとに標準化し、Conv1d 用のテンソル形状へ変換する
5. `models` の 1D-CNN を使って外側 LOSO と内側 LOSO を実行する
6. 学習済みモデルを `ckpt` に保存する
7. `adapt.py` で source 評価、または各 TTA 手法によるテスト時適応評価を実行する
8. 結果と実行時設定を `logs` に保存する

前処理済みデータが `data/wesad/processed/` に存在する場合は、raw `.pkl` からの窓分割をやり直さず、
`.npz` を読み込んで実行します。

## ディレクトリ構成

### `cfg/`

実験設定を置くディレクトリです。コード内に固定値を散らさず、データセットやアルゴリズムごとの設定を
YAML として分離しています。

- `cfg/default.yaml`
  共通設定です。データセット名、モデル名、ログ出力先、チェックポイント保存先、乱数シードなどを定義します。

- `cfg/dataset/wesad.yaml`
  WESAD 固有の設定です。データセットパス、サンプリング周波数、窓幅、チャンネル数、バッチサイズなどを定義します。
  デフォルトでは raw data として `../self_learning_wesad/WESAD` を参照し、前処理済みデータは
  `./data/wesad/processed` に保存します。

- `cfg/dataset/wesad_target_shuffle.yaml`
  評価時の target loader だけを shuffle する設定です。Tent/OFTTA の batch composition 依存を調べるために使います。
  元の Tent 実装に合わせて `episodic=false` は変えず、batch の順序だけを変更します。

- `cfg/algorithm/source.yaml`
  1D-CNN の通常学習と source 評価に使う設定です。最大 epoch 数、学習率、early stopping の patience などを定義します。

- `cfg/algorithm/source_fixed.yaml`
  内側 LOSO を使わず、固定 epoch で source model を学習する設定です。既存の nested LOSO checkpoint を
  上書きしないよう、デフォルト保存先は `./ckpt_fixed` です。

- `cfg/algorithm/tent.yaml`
  Tent 適応に使う設定です。Tent の学習率、1 バッチあたりの更新回数、episodic adaptation の有無を定義します。

- `cfg/algorithm/*.yaml`
  `norm`, `pl`, `shot`, `sar`, `t3a`, `tast`, `tast_bn`, `oftta` など、各 TTA 手法の設定です。
  OFTTA 由来の多クラス・2D-CNN 前提の設定を、WESAD の二値分類・1D-CNN 用に分けています。

### `data_processing/`

WESAD の読み込みと前処理を担当します。

- `data_processing/wesad.py`
  被験者ごとの `.pkl` 読み込み、胸部センサ信号の 5 秒窓分割、ラベルの二値化、被験者ごとの標準化、
  PyTorch `DataLoader` 作成を行います。Conv1d は `(batch, channels, time)` を入力に取るため、
  元の `(window, time, channel)` から軸を入れ替える処理もここに集約しています。
  また、前処理済み `.npz` の保存と読み込み、評価専用 target loader の作成も担当します。

### `models/`

モデル定義を置くディレクトリです。

- `models/cnn1d.py`
  WESAD 用の 1D-CNN を定義します。入力は 8 チャンネルの胸部センサ時系列で、出力は二値分類用の
  1 次元 logits です。損失関数には `BCEWithLogitsLoss` を使う前提です。
  T3A/OFTTA/TAST 系の手法が feature を使えるよう、最後の線形分類層の直前の 64 次元 feature も
  取り出せるようにしています。

### `TTA/`

Test-Time Adaptation の設定とアルゴリズム実装を置くディレクトリです。

- `TTA/setup.py`
  `args.adaption` の値に応じて、通常の source 評価か Tent 評価かを切り替えます。

- `TTA/adapt_algorithm/tent.py`
  WESAD の 1D-CNN 向け Tent 実装です。元の Tent 実装は `BatchNorm2d` と多クラス softmax を想定することが多いですが、
  ここでは `BatchNorm1d` と二値 logits に合わせています。テストバッチごとに予測エントロピーを最小化し、
  BatchNorm の scale と bias だけを更新します。

- `TTA/adapt_algorithm/*.py`
  OFTTA ディレクトリにある TTA 手法を、WESAD の 1D-CNN と single-logit 二値分類に合わせて移植した実装です。
  追加済みの手法は `norm`, `pl`, `shot`, `sar`, `t3a`, `tast`, `tast_bn`, `oftta` です。
  `BatchNorm2d` 前提の処理は `BatchNorm1d` に変更し、多クラス softmax 前提の entropy や pseudo-label 処理は
  single-logit から 2 クラス logits を作って扱う形に変更しています。

### `scripts/`

実験を再現しやすくするためのシェルスクリプトです。

- `scripts/wesad/preprocess_wesad.sh`
  WESAD の raw `.pkl` から、被験者ごとの前処理済み `.npz` を作成します。

- `scripts/wesad/train_loso_wesad.sh`
  WESAD の LOSO 学習を実行します。

- `scripts/wesad/train_fixed_loso_wesad.sh`
  検証ユーザを置かず、固定 epoch で外側 LOSO 学習を実行します。

- `scripts/wesad/adapt_source_wesad.sh`
  各被験者をターゲットにして、適応なしの source 評価を実行します。

- `scripts/wesad/adapt_tent_wesad.sh`
  各被験者をターゲットにして、Tent によるテスト時適応評価を実行します。

- `scripts/wesad/adapt_*_wesad.sh`
  各 TTA 手法を全被験者に対して実行するスクリプトです。`adapt.sh` は source と全 TTA 手法を順に呼び出します。

- `scripts/wesad/compare_source_tent_oftta_shuffle_wesad.sh`
  target loader を shuffle して、Source、Tent、OFTTA の比較表とグラフを作成します。

- `scripts/wesad/compare_source_tent_oftta_fixed_shuffle_wesad.sh`
  固定 epoch checkpoint `./ckpt_fixed` を使い、target loader を shuffle して比較表とグラフを作成します。

### `ckpt/`

LOSO の各 fold で学習したモデルを保存するディレクトリです。保存形式は次のような階層です。

```text
ckpt/
  wesad/
    cnn1d/
      S2/
        wesad_S2_checkpoint.pt
```

`adapt.py` はこの保存形式を前提にチェックポイントを読み込みます。

### `logs/`

学習や評価の実行結果を保存するディレクトリです。実行ごとに日時付きのディレクトリを作り、
`config.yaml`、`log.txt`、`result.csv` などを保存します。

### `data/`

前処理済みデータの保存先です。raw WESAD は `../self_learning_wesad/WESAD` に置いたままにし、
`preprocess.py` または実行時の自動前処理によって、被験者ごとの `.npz` をここに作成します。

```text
data/
  wesad/
    processed/
      S2_ws3500_ch8_cal60.npz
      S3_ws3500_ch8_cal60.npz
      ...
```

`.npz` には次の配列が保存されます。

- `X`: shape `(num_windows, 3500, 8)` の前処理済み時系列
- `y`: shape `(num_windows,)` の二値ラベル

この保存単位は被験者ごとです。LOSO 評価で train/test をまたいだ標準化にならないよう、
各被験者の冒頭 `calibration_windows` 窓だけで scaler を fit しています。

## 主要ファイル

- `train.py`
  1D-CNN の LOSO 学習を行うメインプログラムです。外側 LOSO でテスト被験者を 1 名ずつ固定し、
  残りの被験者で内側 LOSO を回して最終学習 epoch 数を決定します。

- `train_fixed_loso.py`
  内側 LOSO を行わず、外側 LOSO の train subjects 全体で固定 epoch 学習するメインプログラムです。
  epoch 数が小さく選ばれすぎる影響を検証するために使います。

- `adapt.py`
  学習済みチェックポイントを読み込み、source 評価または Tent 評価を実行するメインプログラムです。
  評価時には target 被験者だけを読み込みます。たとえば `target_domain=S2` の場合、
  S2 以外の被験者データは読み込みません。

- `preprocess.py`
  raw WESAD から被験者ごとの前処理済み `.npz` を作成するメインプログラムです。

- `config.py`
  YAML 設定とコマンドライン引数を統合します。

- `utils.py`
  乱数シード固定、device 選択、モデル生成、データセット生成をまとめた補助関数です。

- `metrics.py`
  loss、accuracy、クラス別 F1、mean F1、混同行列などの評価指標を計算します。

## 実行方法

`wesad_env` 環境に必要な依存関係が入っている前提です。

### LOSO 学習

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env bash train.sh
```

または直接:

```bash
conda run -n wesad_env python train.py \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
```

### 固定 epoch の LOSO 学習

内側 LOSO による epoch 選択を行わず、`source_fixed.yaml` の `max_epochs` で固定学習します。

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env bash scripts/wesad/train_fixed_loso_wesad.sh
```

保存先はデフォルトで `ckpt_fixed/` です。固定 epoch 版 checkpoint を使って評価する場合は、
`adapt.py` に `--resume ./ckpt_fixed` を渡します。

```bash
conda run -n wesad_env python adapt.py \
  --target_domain S2 \
  --resume ./ckpt_fixed \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/source.yaml
```

### 前処理済みデータの作成

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env bash scripts/wesad/preprocess_wesad.sh
```

このコマンドを事前に実行しておくと、以後の学習や評価では
`data/wesad/processed/` の `.npz` を再利用できます。未作成の被験者がある場合は、
実行時に raw `.pkl` から自動生成して保存します。

### source 評価と Tent 評価

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env bash adapt.sh
```

### 特定の被験者だけ評価する例

```bash
conda run -n wesad_env python adapt.py \
  --target_domain S2 \
  --dataset_cfg ./cfg/dataset/wesad.yaml \
  --algorithm_cfg ./cfg/algorithm/oftta.yaml
```

`--algorithm_cfg` を `tent.yaml`, `norm.yaml`, `pl.yaml`, `shot.yaml`, `sar.yaml`,
`t3a.yaml`, `tast.yaml`, `tast_bn.yaml`, `oftta.yaml` に変えることで、同じ checkpoint に対して
各 TTA 手法を評価できます。

### target loader を shuffle した比較

通常の `wesad.yaml` では target window を時系列順に batch 化します。先頭の安静時 window が Tent/OFTTA に
与える影響を調べる場合は、target loader だけを shuffle する設定を使います。

```bash
cd /work/shinsaku-t/naist_reserch/WESAD_TTA
conda run -n wesad_env bash scripts/wesad/compare_source_tent_oftta_shuffle_wesad.sh
```

固定 epoch checkpoint を使う場合はこちらです。

```bash
conda run -n wesad_env bash scripts/wesad/compare_source_tent_oftta_fixed_shuffle_wesad.sh
```

## 注意点

- `adapt.py` を実行する前に、対応する被験者のチェックポイントを `train.py` で作成しておく必要があります。
- `adapt.py` は評価専用 loader を使うため、target 被験者だけを読み込みます。source 評価で target 以外の
  被験者データを読む必要はありません。
- WESAD は被験者間差が大きいため、通常のランダム分割ではなく LOSO で未知被験者性能を見る構成にしています。
- Tent は教師ラベルを使わずにテストバッチの予測エントロピーを下げます。そのため通常の評価と異なり、
  推論中も勾配計算を有効にします。
- この実装の Tent は `BatchNorm1d` の affine パラメータのみを更新します。Conv1d や Linear の重みは更新しません。
