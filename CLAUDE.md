# WESAD Test-Time Adaptation (TTA) Research Project

WESAD データセットを用いたストレス検出における Test-Time Adaptation 手法の研究リポジトリ。
胸部装着型ウェアラブルセンサの生体信号（ECG, EDA, EMG, Resp, Temp, ACC x/y/z）から、
3 クラス（中性 / ストレス / 楽しさ）を分類する 1D-CNN を LOSO で学習し、
TENT, EMA-TENT, OFTTA 等の TTA 手法で被験者間汎化を改善する。

## ディレクトリ構造

```
WESAD_TTA/
├── cfg/                          # 実験設定 (YAML)
│   ├── default.yaml              #   全実験共通のデフォルト (device, seed, resume path)
│   ├── dataset/
│   │   └── wesad.yaml            #   WESAD固有の設定 (window_size, batch_size, num_classes=3)
│   └── algorithm/
│       ├── source.yaml           #   Source学習 (max_epochs=20, patience=3)
│       ├── source_improved.yaml  #   改善版Source (max_epochs=50, patience=7, ckpt_improved/)
│       ├── tent.yaml             #   TENT: BNパラメータのエントロピー最小化
│       ├── tent_episodic.yaml    #   TENT (episodic=true): バッチごとにリセット
│       ├── ema_tent.yaml         #   EMA-TENT: BN統計量をEMAで平滑化 (momentum=0.9)
│       ├── ema_tent_episodic.yaml#   EMA-TENT (episodic=true)
│       ├── oftta.yaml            #   OFTTA: weighted BN + classifier adjustment
│       ├── t3a.yaml              #   T3A: 特徴空間でのクラシファイア調整
│       └── ...                   #   (sar, shot, pl, tast 等)
│
├── models/
│   └── cnn1d.py                  # StressCNN1D: 3層Conv1d + 2層FC (8ch→3class)
│                                 #   features部: Conv→ReLU→BN→Pool ×3 (8→32→64→128ch)
│                                 #   classifier部: AdaptiveAvgPool→Linear(128→64)→Linear(64→3)
│
├── data_processing/
│   └── wesad.py                  # WESAD前処理: pkl読み込み→窓分割→3クラスラベル変換→標準化
│                                 #   キャッシュ機能あり (data/wesad/processed/*.npz)
│
├── data/wesad/processed/         # 前処理済みデータ (被験者ごとの .npz)
│
├── TTA/                          # Test-Time Adaptation モジュール
│   ├── setup.py                  #   手法選択ファクトリ: get_adaptation(args, model)
│   └── adapt_algorithm/
│       ├── common.py             #     共通ユーティリティ (エントロピー計算, BNパラメータ収集等)
│       ├── tent.py               #     TENT: BN の γ/β をエントロピー最小化で更新
│       ├── ema_tent.py           #     EMA-TENT: EmaBN1d でBN統計量をEMA平滑化 + エントロピー最小化
│       ├── t3a.py                #     T3A: テストサンプルのサポート集合でクラシファイア重み再計算
│       └── oftta.py              #     OFTTA: weighted BN + support-based classifier
│
├── train.py                      # LOSO学習メインスクリプト
│                                 #   外側LOSO: 1名をテスト、残りで学習
│                                 #   内側LOSO: テスト以外で交差検証→最適epoch数を決定
│                                 #   → ckpt/{dataset}/{model}/{subject}/ にチェックポイント保存
│
├── adapt.py                      # 単一被験者の TTA 評価スクリプト
│                                 #   チェックポイント読み込み → Source評価 → TTA適用後評価
│
├── compare_source_tent_oftta.py  # 全被験者一括比較スクリプト
│                                 #   Source, TENT, EMA-TENT, OFTTA を15名分評価
│                                 #   → CSV, Markdown表, 比較棒グラフを出力
│
├── run_overnight.py              # 一括実験ランナー (5実験を順次実行)
│                                 #   Exp1: episodic比較, Exp2: バッチサイズ感度,
│                                 #   Exp3: Source再学習, Exp4: 改善Source全手法評価,
│                                 #   Exp5: shuffle条件 → overnight_results.md に結果保存
│
├── overnight_results.md          # 一括実験の結果レポート
├── metrics.py                    # 評価指標: accuracy, per-class F1, macro F1, 混同行列
├── utils.py                      # 共通ユーティリティ: seed固定, device取得, モデル/データ生成
├── config.py                     # YAML設定の読み込みとマージ (default → dataset → algorithm → CLI)
│
├── ckpt/                         # 元Sourceのチェックポイント (gitignore: *.pt)
├── ckpt_improved/                # 改善Sourceのチェックポイント (patience=7, max_epochs=50)
└── logs/                         # 実験ログ (タイムスタンプ付きディレクトリ)
```

## 設定の優先順位

`cfg/default.yaml` → `cfg/dataset/*.yaml` → `cfg/algorithm/*.yaml` → CLI引数
（後から読んだ設定が上書きする）

## 主な実験フロー

1. **学習**: `python train.py --dataset_cfg ./cfg/dataset/wesad.yaml --algorithm_cfg ./cfg/algorithm/source.yaml --device mps`
2. **評価**: `python compare_source_tent_oftta.py --device mps`
3. **一括実験**: `python run_overnight.py` (結果は overnight_results.md)

## EMA-TENT の要点

通常の TENT はテストバッチの統計量でBNを正規化するが、WESAD のように感情状態が
ブロック状に連続する時系列では、ブロック切り替え時に統計量が急変して性能が低下する。
EMA-TENT はソースモデルの running stats を初期値として EMA で平滑化し、この問題を緩和する。
最適な momentum は 0.9 付近（momentum 感度分析で確認済み）。

## 技術的な注意点

- Device: Apple Silicon の MPS を使用 (`--device mps`)
- 被験者: S2-S17 の15名 (S1, S12 は欠番)
- 前処理キャッシュ: `data/wesad/processed/` に `.npz` 形式で保存。設定変更時は再生成が必要
- チェックポイント: `*.pt` は `.gitignore` で除外。共有時は別途受け渡しが必要
