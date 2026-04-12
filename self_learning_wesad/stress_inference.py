import os
# TensorFlowのログレベル設定と、CUDA/XLA関連のエラー（PTX不整合など）を回避するための設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_jit_profiling'] = 'false'

import pickle
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf

# --- グローバルパラメータ (学習時と一致させる) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "WESAD") # WESADデータセットの場所
WINDOW_SECONDS = 5  # ウィンドウサイズ（秒）
FS = 700            # サンプリング周波数 (Hz)
WINDOW_SIZE = WINDOW_SECONDS * FS # 3500サンプル
NUM_CHANNELS = 8    # センサーチャンネル数

def load_single_subject(subject_id, dataset_dir):
    """
    特定の被験者のデータを読み込み、前処理（ウィンドウ分割とスケーリング）を行います。
    最初の5分間のデータ（安静時を想定）を基準に標準化（StandardScaler）を行います。
    """
    path = os.path.join(dataset_dir, subject_id)
    files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f"被験者 {subject_id} の .pkl ファイルが見つかりません。")
        
    print(f"{subject_id} のデータを読み込み中...")
    with open(os.path.join(path, files[0]), 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    chest = data['signal']['chest']
    label_arr = data['label']
    
    total_samples = len(chest['ECG'].flatten())
    num_windows = total_samples // WINDOW_SIZE
    
    X, y = [], []
    for i in range(num_windows):
        start, end = i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE
        
        # 胸部センサーの各信号をスタック: ECG, EDA, EMG, Resp, Temp, ACC (x, y, z)
        signals = [
            chest['ECG'].ravel()[start:end],
            chest['EDA'].ravel()[start:end],
            chest['EMG'].ravel()[start:end],
            chest['Resp'].ravel()[start:end],
            chest['Temp'].ravel()[start:end],
            chest['ACC'][start:end, 0],
            chest['ACC'][start:end, 1],
            chest['ACC'][start:end, 2]
        ]
        
        window_data = np.stack(signals, axis=-1)
        # ウィンドウサイズが足りない場合はゼロパディング
        if len(window_data) < WINDOW_SIZE:
            window_data = np.pad(window_data, ((0, WINDOW_SIZE - len(window_data)), (0, 0)))
            
        X.append(window_data)
        # ウィンドウ内のラベルの最頻値を採用
        y.append(int(stats.mode(label_arr[start:end], keepdims=True).mode[0]))
    
    X, y = np.array(X), np.array(y)
    
    # ラベルのフィルタリングとマッピング: 
    # 1(Baseline), 3(Amusement), 4(Meditation) -> 0 (非ストレス)
    # 2(Stress) -> 1 (ストレス)
    valid = np.isin(y, [1, 2, 3, 4])
    X, y = X[valid], y[valid]
    y_mapped = np.array([1 if val == 2 else 0 for val in y])
    
    # キャリブレーション（最初の5分間 = 60ウィンドウ を使用してスケーラーを適合）
    CALIBRATION_WINDOWS = 60
    flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    
    if len(flat) > CALIBRATION_WINDOWS:
        scaler.fit(flat[:CALIBRATION_WINDOWS])
    else:
        scaler.fit(flat)
        
    # 全データをスケーリング
    scaled = scaler.transform(flat).astype(np.float32)
    X_scaled = scaled.reshape(-1, WINDOW_SIZE, NUM_CHANNELS)
    
    return X_scaled, y_mapped

if __name__ == "__main__":
    # 学習済みモデルのパス (stress_detection_1d_cnn.py で保存されたもの)
    MODEL_PATH = os.path.join(SCRIPT_DIR, "final_binary_1dcnn_class_weights.h5")
    SUBJECT_ID = "S2" # 推論対象の被験者ID
    
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル {MODEL_PATH} が見つかりません。先に学習スクリプトを実行してください。")
        exit(1)
        
    try:
        # モデルのロードとデータの準備
        model = tf.keras.models.load_model(MODEL_PATH)
        X_test, y_test = load_single_subject(SUBJECT_ID, DATASET_DIR)
        
        print(f"{len(X_test)} 個のウィンドウに対して推論を実行中...")
        # 推論の実行
        y_probs = model.predict(X_test, batch_size=64)
        y_pred = (y_probs > 0.5).astype(int)
        
        # 分類レポートの表示
        print(f"\n--- {SUBJECT_ID} の推論結果 (適応なし) ---")
        print(classification_report(y_test, y_pred, target_names=["Non-Stress", "Stress"]))
        
        f1 = f1_score(y_test, y_pred)
        print(f"F1スコア: {f1:.4f}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
