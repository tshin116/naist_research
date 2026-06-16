# NAIST Research Workspace

このリポジトリは、研究用コードと実験結果をまとめた作業ディレクトリです。現在の主プロジェクトは `WESAD_TTA/` です。

## Main Project

### `WESAD_TTA/`

ウェアラブル生体信号を用いた感情推定に対して、Test-Time Adaptation (TTA) を比較する研究コードです。

対象データセット:

- WESAD
  - stress/no-stress 2クラス分類
  - baseline/stress/amusement 3クラス分類
- CASE
  - arousal 2クラス分類
  - valence 2クラス分類
- EmoWear
  - arousal 2クラス分類
  - valence 2クラス分類

比較手法:

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

主な実装:

- 1D-CNN + BatchNorm1d
- LOSO source model training
- target stream に対する test-time adaptation
- WESAD の batch composition bias 分析
- DynaMix EMA-Tent の提案・比較
- CASE / EmoWear への拡張評価

最初に読むファイル:

- `WESAD_TTA/PROJECT_OVERVIEW_FOR_SUPERVISOR.md`
- `WESAD_TTA/README.md`
- `WESAD_TTA/SHARING_GUIDE.md`
- `WESAD_TTA/新手法の説明文/dynamix_ema_tent_method.md`

古い詳細ガイド:

- `WESAD_TTA/Readme.md`

`Readme.md` は初期のWESAD中心の詳細ガイドです。現在の共有用入口は `WESAD_TTA/README.md` と `WESAD_TTA/PROJECT_OVERVIEW_FOR_SUPERVISOR.md` です。

## Sharing

GitHubには、コード・設定・説明文・checkpoint・主要ログを含めます。

データ本体は容量が大きいためGit管理には含めません。

Git管理から除外している主なもの:

- `WESAD_TTA/data/`
- raw dataset
- `.venv/`
- `__pycache__/`
- 共有用アーカイブ `*.tar.gz`
- 外部cloneした参照リポジトリ

Google Drive等で別共有するデータアーカイブ:

```text
WESAD_TTA_data_260616.tar.gz
```

このアーカイブは、`WESAD_TTA/` の1つ上の階層で展開します。

```bash
cd /path/to/naist_reserch
tar -xzf WESAD_TTA_data_260616.tar.gz
```

展開後に以下の構造になれば正しいです。

```text
naist_reserch/
  WESAD_TTA/
    data/
      case/
      emowear/
      wesad/
```

`WESAD_TTA/` の中で展開すると `WESAD_TTA/WESAD_TTA/data/` になってしまうため、避けてください。

## Share Archives

コード・設定・ログ・checkpointのみの共有アーカイブ:

```text
WESAD_TTA_share_260616.tar.gz
```

データのみの共有アーカイブ:

```text
WESAD_TTA_data_260616.tar.gz
```

GitHubを使う場合は、通常は `WESAD_TTA_share_260616.tar.gz` は不要です。GitHub clone後に `WESAD_TTA_data_260616.tar.gz` だけを展開すれば、processed data を使った再実行ができます。

## External or Reference Repositories

以下は既存研究の確認や比較実装の参考としてcloneした外部リポジトリです。主実装ではありません。

- `OFTTA/`
- `tent/`
- `pytorch-classification/`
- `deep-residual-networks/`
- `wesad/`
- `RealisticTTA/`
- `RoTTA/`
- `NOTE/`
- `DUA/`
- `DELTA/`
- `Personalized_Affective_Computing/`

これらは原則としてGit管理に含めず、実装上の対応内容は `WESAD_TTA/docs/` に研究メモとして残します。

## Other Directories

### `self_learning_wesad/`

WESADの初期実験、前処理確認、補助的な検証に使ったディレクトリです。現在の主実装は `WESAD_TTA/` に移っています。

### `case_dataset/`

CASE raw data 置き場です。raw data はGit管理対象外です。

## Notes

- 通常の研究コード確認は `WESAD_TTA/` だけ見れば十分です。
- 先生・共同研究者に共有する場合は、まず `WESAD_TTA/PROJECT_OVERVIEW_FOR_SUPERVISOR.md` を案内してください。
- データを含めて再実行する場合は、GitHub clone後に `WESAD_TTA_data_260616.tar.gz` を展開してください。
