# NAIST Research Workspace

このリポジトリは研究用の作業ディレクトリをまとめたものです。現在の主なプロジェクトは `WESAD_TTA` と `self_learning_wesad` です。

## Main Projects

### `WESAD_TTA/`

WESAD データセットを用いた test-time adaptation (TTA) 実験の中心プロジェクトです。

主な内容は以下です。

- WESAD の 3 クラス感情推定実験
- 1D-CNN による LOSO 評価
- Source, Tent, OFTTA, MemOFTTA, EMA-Tent の比較
- batch composition bias の分析
- t-SNE による被験者差の可視化
- 論文用の実験結果・図・分析メモ

詳細は [`WESAD_TTA/Readme.md`](WESAD_TTA/Readme.md) を参照してください。

### `self_learning_wesad/`

WESAD 関連の自己学習・前処理・データ確認用の作業ディレクトリです。

現在は WESAD データ本体や補助的な実験コードを含む場合があります。データファイルはサイズが大きく、Git 管理に含めるべきでないものがあるため、コミット対象にする前に内容を確認してください。

## External or Reference Repositories

以下のディレクトリは、既存研究・実装確認・比較・移植のために外部から clone したもの、または参照用コードです。基本的には本研究の主実装ではありません。

- `OFTTA/`
- `tent/`
- `pytorch-classification/`
- `deep-residual-networks/`
- `wesad/`

これらは手法理解、実装移植、比較実験の参考として利用しています。変更が必要な場合でも、主プロジェクトである `WESAD_TTA/` への影響を確認してから扱います。

## Repository Notes

- 通常の開発・分析作業は `WESAD_TTA/` を中心に行います。
- `self_learning_wesad/WESAD/` のような生データは、原則として Git に追加しません。
- 外部 clone 由来のディレクトリに出ている差分は、研究本体の変更とは分けて扱います。
- コミット時は、`WESAD_TTA/` の実装・設定・論文用分析結果と、外部参照コードや生データを混ぜないようにします。

