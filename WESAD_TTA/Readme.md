# WESAD_TTA ディレクトリ構成ガイド

このディレクトリは、WESAD データセットを用いたストレス二値分類について、1D-CNN の LOSO
評価と Tent による Test-Time Adaptation を実行するための実験コードをまとめたものです。

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
7. `adapt.py` で source 評価、または Tent によるテスト時適応評価を実行する
8. 結果と実行時設定を `logs` に保存する

## ディレクトリ構成

### `cfg/`

実験設定を置くディレクトリです。コード内に固定値を散らさず、データセットやアルゴリズムごとの設定を
YAML として分離しています。

- `cfg/default.yaml`
  共通設定です。データセット名、モデル名、ログ出力先、チェックポイント保存先、乱数シードなどを定義します。

- `cfg/dataset/wesad.yaml`
  WESAD 固有の設定です。データセットパス、サンプリング周波数、窓幅、チャンネル数、バッチサイズなどを定義します。
  デフォルトでは `../self_learning_wesad/WESAD` を参照します。

- `cfg/algorithm/source.yaml`
  1D-CNN の通常学習と source 評価に使う設定です。最大 epoch 数、学習率、early stopping の patience などを定義します。

- `cfg/algorithm/tent.yaml`
  Tent 適応に使う設定です。Tent の学習率、1 バッチあたりの更新回数、episodic adaptation の有無を定義します。

### `data_processing/`

WESAD の読み込みと前処理を担当します。

- `data_processing/wesad.py`
  被験者ごとの `.pkl` 読み込み、胸部センサ信号の 5 秒窓分割、ラベルの二値化、被験者ごとの標準化、
  PyTorch `DataLoader` 作成を行います。Conv1d は `(batch, channels, time)` を入力に取るため、
  元の `(window, time, channel)` から軸を入れ替える処理もここに集約しています。

### `models/`

モデル定義を置くディレクトリです。

- `models/cnn1d.py`
  WESAD 用の 1D-CNN を定義します。入力は 8 チャンネルの胸部センサ時系列で、出力は二値分類用の
  1 次元 logits です。損失関数には `BCEWithLogitsLoss` を使う前提です。

### `TTA/`

Test-Time Adaptation の設定とアルゴリズム実装を置くディレクトリです。

- `TTA/setup.py`
  `args.adaption` の値に応じて、通常の source 評価か Tent 評価かを切り替えます。

- `TTA/adapt_algorithm/tent.py`
  WESAD の 1D-CNN 向け Tent 実装です。元の Tent 実装は `BatchNorm2d` と多クラス softmax を想定することが多いですが、
  ここでは `BatchNorm1d` と二値 logits に合わせています。テストバッチごとに予測エントロピーを最小化し、
  BatchNorm の scale と bias だけを更新します。

### `scripts/`

実験を再現しやすくするためのシェルスクリプトです。

- `scripts/wesad/train_loso_wesad.sh`
  WESAD の LOSO 学習を実行します。

- `scripts/wesad/adapt_source_wesad.sh`
  各被験者をターゲットにして、適応なしの source 評価を実行します。

- `scripts/wesad/adapt_tent_wesad.sh`
  各被験者をターゲットにして、Tent によるテスト時適応評価を実行します。

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

OFTTA 風の構成に合わせたデータ置き場です。現時点のデフォルト設定では、実データは
`../self_learning_wesad/WESAD` を参照しているため、このディレクトリに WESAD をコピーする必要はありません。

## 主要ファイル

- `train.py`
  1D-CNN の LOSO 学習を行うメインプログラムです。外側 LOSO でテスト被験者を 1 名ずつ固定し、
  残りの被験者で内側 LOSO を回して最終学習 epoch 数を決定します。

- `adapt.py`
  学習済みチェックポイントを読み込み、source 評価または Tent 評価を実行するメインプログラムです。

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
  --algorithm_cfg ./cfg/algorithm/tent.yaml
```

## 注意点

- `adapt.py` を実行する前に、対応する被験者のチェックポイントを `train.py` で作成しておく必要があります。
- WESAD は被験者間差が大きいため、通常のランダム分割ではなく LOSO で未知被験者性能を見る構成にしています。
- Tent は教師ラベルを使わずにテストバッチの予測エントロピーを下げます。そのため通常の評価と異なり、
  推論中も勾配計算を有効にします。
- この実装の Tent は `BatchNorm1d` の affine パラメータのみを更新します。Conv1d や Linear の重みは更新しません。
