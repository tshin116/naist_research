import os
# TensorFlowのログレベル設定と、CUDA/XLA関連のエラー回避
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_jit_profiling'] = 'false'

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from stress_inference import load_single_subject # データローダーを再利用

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "WESAD")

def setup_tent_model(model):
    """
    モデルをTENT用に設定します。
    BatchNormalization層（BN層）のパラメータ（scaleとshift）のみを学習可能にし、
    それ以外の全ての層の重みを固定（フリーズ）します。
    """
    model.trainable = False  # モデル全体の重みを一度固定
    for layer in model.layers:
        # TENTではBN層のみを適応対象とする
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
            # 注意: KerasではBN層の移動平均を使用せずバッチ統計量を使用するために、
            # 推論実行時に `training=True` を渡す必要があります（後述のtent_adapt_stepで実施）。
    return model

@tf.function
def tent_adapt_step(x, model, optimizer):
    """
    エントロピー最小化による1ステップの適応処理。
    推論データの予測エントロピー（不確実性）を算出し、それが最小になるようにBN層の重みを更新します。
    教師ラベルを使わずに「予測の自信」を高めるようにモデルを調整します。
    """
    with tf.GradientTape() as tape:
        # training=True を渡すことで、BN層が現在のテストバッチの統計量（平均・分散）を
        # 直接使用するように強制します（TENTの核心部分）。
        probs = model(x, training=True)
        
        # 二値分類のエントロピー計算: -[p*log(p) + (1-p)*log(1-p)]
        # 数値安定性のために微小な値 (1e-10) を加算
        entropy = - (probs * tf.math.log(probs + 1e-10) + (1 - probs) * tf.math.log(1 - probs + 1e-10))
        loss = tf.reduce_mean(entropy)
        
    # 学習可能（BN層のみ）に設定した変数に対して勾配を計算し、更新
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return probs

def run_tent_inference(model, X_test, batch_size=64, learning_rate=1e-4):
    """
    テスト時適応 (TTA: Test-Time Adaptation) を伴う推論を実行します。
    テストデータをバッチごとに処理し、予測を行う直前にモデルをそのデータに適応させます。
    """
    # 適応用のオプティマイザ（通常より小さめの学習率が推奨される）
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # モデルをTENT用にBN層のみ解凍
    model = setup_tent_model(model)
    
    all_probs = []
    print(f"{len(X_test)} 個のウィンドウに対して適応と推論を開始...")
    
    # テストデータをバッチ単位でループ
    for i in range(0, len(X_test), batch_size):
        x_batch = X_test[i : i + batch_size]
        
        # 現在のバッチに対して1ステップ適応を行い、その予測値を取得
        # (注: originalのTENT論文のように1バッチに対して複数回更新することも可能です)
        probs = tent_adapt_step(x_batch, model, optimizer)
        all_probs.append(probs.numpy())
        
        if (i // batch_size) % 10 == 0:
            print(f"バッチ {i // batch_size} 处理完了...")
            
    return np.vstack(all_probs)

if __name__ == "__main__":
    # モデルのロード (stress_detection_1d_cnn.py で保存されたもの)
    MODEL_PATH = os.path.join(SCRIPT_DIR, "final_binary_1dcnn_class_weights.h5")
    SUBJECT_ID = "S2" # 適応対象の未知のユーザー
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル {MODEL_PATH} が見つかりません。")
        exit(1)
        
    try:
        # テストデータの読み込み
        X_test, y_test = load_single_subject(SUBJECT_ID, DATASET_DIR)
        
        # 1. ベースライン (適応なしの通常の推論)
        print("\n--- 1. ベースライン推論 (適応なし) を実行中 ---")
        model_base = tf.keras.models.load_model(MODEL_PATH)
        y_probs_base = model_base.predict(X_test, batch_size=64, verbose=0)
        y_pred_base = (y_probs_base > 0.5).astype(int)
        f1_base = f1_score(y_test, y_pred_base)
        print(f"ベースライン F1スコア: {f1_base:.4f}")
        
        # 2. TENT (テスト時適応ありの推論)
        print("\n--- 2. TENT 推論 (テスト時適応あり) を実行中 ---")
        # モデルを再ロードして適応を開始
        model_tent = tf.keras.models.load_model(MODEL_PATH)
        y_probs_tent = run_tent_inference(model_tent, X_test, learning_rate=1e-4)
        y_pred_tent = (y_probs_tent > 0.5).astype(int)
        f1_tent = f1_score(y_test, y_pred_tent)
        
        # 結果の比較
        print("\n--- 比較結果まとめ ---")
        print(f"被験者ID: {SUBJECT_ID}")
        print(f"適応なし F1スコア: {f1_base:.4f}")
        print(f"適応あり (TENT) F1スコア: {f1_tent:.4f}")
        
        # 適応後の分類レポートを表示
        print(f"\n--- {SUBJECT_ID} の詳細レポート (TENT適応後) ---")
        print(classification_report(y_test, y_pred_tent, target_names=["Non-Stress", "Stress"]))
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
