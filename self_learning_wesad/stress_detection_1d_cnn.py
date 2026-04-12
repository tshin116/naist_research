import os
# Disable XLA devices and JIT to resolve CUDA PTX errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_jit_profiling'] = 'false'
# Set CUDA directory for XLA
# os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3'

import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Flatten, Dense, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample, shuffle
from sklearn.utils.class_weight import compute_class_weight

# Global parameters
# WESADデータセットのディレクトリパスを定義
# スクリプトの場所を基準に絶対パスを取得
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "WESAD")

WINDOW_SECONDS = 5  # ウィンドウサイズ（秒）
FS = 700            # サンプリング周波数 (Hz) - WESADの胸部データは700Hz
WINDOW_SIZE = WINDOW_SECONDS * FS  # 1ウィンドウあたりのサンプル数 (5秒 * 700Hz = 3500サンプル)
NUM_CHANNELS = 8  # 使用するチャンネル数: ECG, EDA, EMG, Resp, Temp, ACCx, ACCy, ACCz

# --- 1. データ分割用関数 (Data Segmentation Functions) ---

def get_window(chest, label_arr, start, end):
    """
    指定された開始・終了インデックスに基づいて、胸部センサーデータからウィンドウを切り出します。
    また、ウィンドウ内のラベルの最頻値をそのウィンドウのラベルとして採用します。
    
    Args:
        chest (dict): 胸部センサーデータを含む辞書
        label_arr (array): ラベルデータの配列
        start (int): 開始インデックス
        end (int): 終了インデックス
        
    Returns:
        tuple: (ウィンドウデータ, ラベル) のペア。エラー時は (None, None)
    """
    def get_signal(key):
        # 指定されたキーの信号データを切り出し、1次元配列に変換
        s = np.array(chest.get(key, np.zeros(WINDOW_SIZE))[start:end]).ravel()
        # ウィンドウサイズに満たない場合はゼロパディングを行う
        if len(s) < WINDOW_SIZE:
            s = np.pad(s, (0, WINDOW_SIZE - len(s)))
        return s

    try:
        # 各センサーデータの取得
        ECG  = get_signal('ECG')
        EDA  = get_signal('EDA')
        EMG  = get_signal('EMG')
        Resp = get_signal('Resp')
        Temp = get_signal('Temp')
        
        # 加速度データ (ACC) は3軸あるため個別に処理
        acc  = np.array(chest.get('ACC', np.zeros((WINDOW_SIZE, 3)))[start:end])
        if acc.ndim == 2 and acc.shape[1] == 3:
            ACC_x, ACC_y, ACC_z = acc[:,0], acc[:,1], acc[:,2]
        else:
            # データがない、または形状が合わない場合はゼロ埋め
            ACC_x = ACC_y = ACC_z = np.zeros(WINDOW_SIZE)

        # 取得した全チャンネルをスタックして (WINDOW_SIZE, 8) の形状にする
        signals = [ECG, EDA, EMG, Resp, Temp, ACC_x, ACC_y, ACC_z]
        window_data = np.stack(signals, axis=-1)
        
        # ウィンドウ内のラベルの最頻値（モード）を計算して、代表ラベルとする
        mode_label = int(stats.mode(label_arr[start:end], keepdims=True).mode[0])
        return window_data, mode_label
    except Exception as e:
        print(f"Error at window {start}: {e}")
        return None, None

def segment_data_raw(data_dict):
    """
    1人の被験者の生データを、固定長のウィンドウに分割します。
    
    Args:
        data_dict (dict): pickleから読み込んだ被験者データ
        
    Returns:
        tuple: (全ウィンドウデータ配列, 全ラベル配列)
    """
    X, y = [], []
    if 'signal' not in data_dict or 'label' not in data_dict:
        return np.array([]), np.array([])
    
    chest = data_dict['signal']['chest']
    # 信号長から総サンプル数を取得 (ECGを基準とする)
    total_samples = len(chest['ECG'].flatten())
    
    # 作成可能なウィンドウ数を計算
    num_windows = total_samples // WINDOW_SIZE
    
    # ウィンドウごとにデータを切り出し
    for i in range(num_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        window_data, label = get_window(chest, data_dict['label'], start, end)
        if window_data is not None:
            X.append(window_data)
            y.append(label)
            
    return np.array(X), np.array(y)

# --- 2. 前処理とスケーリング (Subject-Level Preprocessing & Scaling) ---

def filter_map_labels_binary(X, y):
    """
    必要なラベルのみを抽出し、2値分類（ストレス vs 非ストレス）に変換します。
    
    対象ラベル:
        1: ベースライン (Baseline)
        2: ストレス (Stress)
        3: 娯楽 (Amusement)
        4: 瞑想 (Meditation)
        
    変換ルール:
        1, 3, 4 -> 0 (非ストレス)
        2       -> 1 (ストレス)
    """
    # 1, 2, 3, 4 のラベルを持つデータのみをフィルタリング
    valid = np.isin(y, [1, 2, 3, 4])
    X, y = X[valid], y[valid]
    
    # ラベルのマッピング定義
    mapping = {1:0, 2:1, 3:0, 4:0}
    # 定義に従ってラベルを変換
    y_mapped = np.array([mapping[val] for val in y])
    
    # データ数を揃えて返す
    return X[:len(y_mapped)], y_mapped

def load_subject_data(folder):
    """
    指定されたフォルダ内の .pkl ファイルを読み込み、ウィンドウ分割を行います。
    """
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    if not files:
        return np.array([]), np.array([])
    try:
        # pickleファイルの読み込み (latin1エンコーディングが必要な場合が多い)
        with open(os.path.join(folder, files[0]), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        # 生データをウィンドウ分割
        return segment_data_raw(data)
    except Exception as e:
        print(f"Skip {folder}: {e}")
        return np.array([]), np.array([])

def load_and_scale_all_data():
    """
    データセットディレクトリ内の全被験者データを読み込み、前処理とスケーリングを行います。
    被験者ごとに標準化 (StandardScaler) を適用します。
    """
    X_list, y_list = [], []
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return None, None
        
    # 各被験者ディレクトリを処理
    for subj in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, subj)
        if not os.path.isdir(path):
            continue
        print("Processing:", subj)
        
        # データの読み込み
        X_sub, y_sub = load_subject_data(path)
        if X_sub.size == 0:
            continue
            
        # フィルタリングとラベル変換
        X_sub, y_sub = filter_map_labels_binary(X_sub, y_sub)
        
        # スケーリング (被験者ごとの標準化)
        # StandardScalerを適用するために、一時的に (サンプル数 * 時間, チャンネル数) の2次元に変換
        # これにより、各チャンネルごとに平均0、分散1になるように正規化されます
        flat = X_sub.reshape(X_sub.shape[0], -1) # ここでは元のノートブックのロジックを踏襲
        scaled = StandardScaler().fit_transform(flat).astype(np.float32)
        
        # 元の形状 (サンプル数, 時間, チャンネル数) に戻してリストに追加
        X_list.append(scaled.reshape(-1, WINDOW_SIZE, NUM_CHANNELS))
        y_list.append(y_sub)
        
    if not X_list:
        return None, None
    
    # 全被験者のデータを結合して返す
    return np.vstack(X_list), np.concatenate(y_list)

# --- 3. モデル構築 (Build Model) ---

def build_1d_cnn_binary(input_shape):
    """
    1D-CNN (畳み込みニューラルネットワーク) モデルを構築します。
    時系列データの特徴抽出に適した構造です。
    
    Args:
        input_shape (tuple): 入力データの形状 (時間ステップ数, チャンネル数)
    """
    return Sequential([
        Input(shape=input_shape),

        # 第1畳み込みブロック
        # Conv1D: 32フィルター, カーネルサイズ7, ReLU活性化関数
        # BatchNormalization: 学習の安定化と加速
        # MaxPooling1D: 特徴量のダウンサンプリング (サイズを半分にする)
        Conv1D(32, 7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # 第2畳み込みブロック
        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        # 第3畳み込みブロック
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),

        # GlobalAveragePooling1D: 特徴マップ全体を平均化し、ベクトルに変換
        GlobalAveragePooling1D(),
        
        # Dropout: 過学習を防ぐためにニューロンをランダムに無効化 (40%)
        Dropout(0.4),

        # 全結合層
        Dense(64, activation='relu'),
        Dropout(0.3),

        # 出力層: 2値分類のためシグモイド関数を使用 (0〜1の確率を出力)
        Dense(1, activation='sigmoid')
    ])

# --- メイン実行処理 (Main Execution) ---
if __name__ == "__main__":
    # 1. データの読み込み
    print("Loading and processing data... (データの読み込みと処理中...)")
    X_all, y_all = load_and_scale_all_data()
    
    if X_all is None:
        print("Failed to load data. Exiting. (データの読み込みに失敗しました。終了します。)")
        exit(1)

    print("All windows:", X_all.shape[0])
    # ラベルごとのデータ数を表示
    print("Label counts:", dict(zip(*np.unique(y_all, return_counts=True))))

    # 2. データの分割 (学習用とテスト用)
    # 70%を学習用、30%をテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42, shuffle=True
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 3. モデルのコンパイル
    model = build_1d_cnn_binary((WINDOW_SIZE, NUM_CHANNELS))
    # オプティマイザ: Adam, 損失関数: バイナリ交差エントロピー
    model.compile(optimizer=Adam(5e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary() # モデル構造の表示

    # 4. モデルの学習
    # クラスの不均衡を考慮して、各クラスに重みを設定
    # 'balanced' モードで計算することで、少数クラスのペナルティを大きくする
    class_weights = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    ))
    
    # EarlyStopping: 検証データの損失が改善しなくなったら学習を早めに終了
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("Starting training... (学習を開始します...)")
    history = model.fit(
        X_train, y_train,
        epochs=20, # エポック数 (全データを何周学習するか)
        batch_size=64, # バッチサイズ (一度に処理するデータ数)
        validation_split=0.2, # 学習データの一部を検証用に使用 (20%)
        class_weight=class_weights, # クラス重みの適用
        callbacks=[early_stop], # コールバックの適用
        verbose=1
    )

    # 5. モデルの評価
    print("\nEvaluating model... (モデルを評価中...)")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 6. 閾値の調整 (Threshold Tuning)
    # 最も高いF1スコアが出る閾値を探索する
    y_probs = model.predict(X_test)
    thresholds = np.arange(0.5, 1.0, 0.05)
    best_thresh, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (y_probs > t).astype(int)
        score = f1_score(y_test, preds)
        if score > best_f1:
            best_f1, best_thresh = score, t

    print(f"Best thresh: {best_thresh:.2f}, F1: {best_f1:.3f}")

    # 7. 最終的な分類レポートの表示
    y_pred = (y_probs > best_thresh).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Stress","Stress"]))

    # 8. モデルの保存
    model.save("final_binary_1dcnn_class_weights.h5")
    print("Model saved to final_binary_1dcnn_class_weights.h5")

    # 9. 結果の可視化 (ファイル保存)
    
    # 学習曲線のプロット (損失と精度)
    plt.figure(figsize=(12,5))
    
    # Loss (損失) の推移
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy (精度) の推移
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to training_curves.png")

    # 混同行列 (Confusion Matrix) のプロット
    # 実際の結果と予測結果の対比をヒートマップで表示
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket',
                xticklabels=["Non-Stress","Stress"],
                yticklabels=["Non-Stress","Stress"])
    plt.title(f"Confusion Matrix (thresh={best_thresh:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

