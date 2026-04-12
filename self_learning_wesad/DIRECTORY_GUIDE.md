# self_learning_wesad ディレクトリ構成ガイド

このディレクトリは、WESADデータセットを用いたストレス検出モデル（1D-CNN）の学習、評価、およびテスト時適応（TTA）に関するプログラムとデータが含まれています。

## 1. 学習プログラム (Training)
*   **`stress_detection_1d_cnn.py`**: 
    *   全データをランダムに分割（70:30）して学習を行う標準的なスクリプト。
    *   モデルファイル `final_binary_1dcnn_class_weights.h5` を生成します。
*   **`stress_detection_1d_cnn_loso.py`**: 
    *   Leave-One-Subject-Out (LOSO) クロスバリデーションを行うスクリプト。
    *   未知のユーザーに対する性能を厳密に評価するために使用します。
*   **`stress-detection-with-1d-cnn-99-accuracy.ipynb`**: 
    *   1D-CNNを用いた学習プロセスのベースとなったノートブック。

## 2. 推論・テスト時適応 (Inference & TTA)
*   **`stress_inference.py`**: 
    *   学習済みモデルを読み込み、特定の被験者データに対してストレス判定を行う単体推論スクリプト。
*   **`stress_tent_tta.py`**: 
    *   TENTアルゴリズムを用いたテスト時適応（Test-Time Adaptation）を実行するスクリプト。
    *   教師ラベルなしで、推論中にモデルのBatchNormalization層を調整し、個人差を吸収します。
*   **`evalu_tent.py`**: 
    *   TTA（テスト時適応）の評価用スクリプト。

## 3. データセット (Dataset)
*   **`WESAD/`**: 
    *   実際のセンサーデータ（.pkl形式）が被験者ごとに格納されているディレクトリ。
*   **`wesad-full-dataset.zip` / `archive.zip`**: 
    *   WESADデータセットのアーカイブファイル。

## 4. 学習成果物・結果 (Artifacts & Results)
*   **`final_binary_1dcnn_class_weights.h5`**: 
    *   学習済みの1D-CNNモデル重みファイル。
*   **`loso_results.csv`**: 
    *   LOSO評価の結果（被験者ごとの精度、F1スコア）を記録したファイル。
*   **`training_curves.png`**: 
    *   学習時の損失（Loss）と精度（Accuracy）の推移グラフ。
*   **`confusion_matrix.png`**: 
    *   推論結果の混同行列（Confusion Matrix）画像。
*   **`stress-detection-loso-evaluation.ipynb`**: 
    *   LOSO評価の結果を分析するためのノートブック。

## 5. その他
*   **`README.txt` / `LICENSE.txt`**: 
    *   プロジェクトの説明とライセンスに関するドキュメント。
*   **`__pycache__/`**: 
    *   Pythonの実行キャッシュディレクトリ。
