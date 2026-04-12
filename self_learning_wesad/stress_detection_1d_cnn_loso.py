import os
# Disable XLA devices and JIT to resolve CUDA PTX errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_jit_profiling'] = 'false'
# Set CUDA directory for XLA
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3'

import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_curve, auc, accuracy_score
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "WESAD")

WINDOW_SECONDS = 5  # ウィンドウサイズ（秒）
FS = 700            # サンプリング周波数 (Hz) - WESADの胸部データは700Hz
WINDOW_SIZE = WINDOW_SECONDS * FS  # 1ウィンドウあたりのサンプル数 (5秒 * 700Hz = 3500サンプル)
NUM_CHANNELS = 8  # 使用するチャンネル数: ECG, EDA, EMG, Resp, Temp, ACCx, ACCy, ACCz

# --- 1. データ分割用関数 (Data Segmentation Functions) ---
# (ここは変更なし)

def get_window(chest, label_arr, start, end):
    """
    指定された開始・終了インデックスに基づいて、胸部センサーデータからウィンドウを切り出します。
    また、ウィンドウ内のラベルの最頻値をそのウィンドウのラベルとして採用します。
    """
    def get_signal(key):
        s = np.array(chest.get(key, np.zeros(WINDOW_SIZE))[start:end]).ravel()
        if len(s) < WINDOW_SIZE:
            s = np.pad(s, (0, WINDOW_SIZE - len(s)))
        return s

    try:
        ECG  = get_signal('ECG')
        EDA  = get_signal('EDA')
        EMG  = get_signal('EMG')
        Resp = get_signal('Resp')
        Temp = get_signal('Temp')
        
        acc  = np.array(chest.get('ACC', np.zeros((WINDOW_SIZE, 3)))[start:end])
        if acc.ndim == 2 and acc.shape[1] == 3:
            ACC_x, ACC_y, ACC_z = acc[:,0], acc[:,1], acc[:,2]
        else:
            ACC_x = ACC_y = ACC_z = np.zeros(WINDOW_SIZE)

        signals = [ECG, EDA, EMG, Resp, Temp, ACC_x, ACC_y, ACC_z]
        window_data = np.stack(signals, axis=-1)
        
        # mode_label = int(stats.mode(label_arr[start:end], keepdims=True).mode[0])

        segment = label_arr[start:end].astype(int)
        mode_label = np.bincount(segment).argmax()

        return window_data, mode_label
    except Exception as e:
        print(f"Error at window {start}: {e}")
        return None, None

def segment_data_raw(data_dict):
    """
    1人の被験者の生データを、固定長のウィンドウに分割します。
    """
    X, y = [], []
    if 'signal' not in data_dict or 'label' not in data_dict:
        return np.array([]), np.array([])
    
    chest = data_dict['signal']['chest']
    total_samples = len(chest['ECG'].flatten())
    num_windows = total_samples // WINDOW_SIZE
    
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
    """
    valid = np.isin(y, [1, 2, 3, 4])
    X, y = X[valid], y[valid]
    
    mapping = {1:0, 2:1, 3:0, 4:0}
    y_mapped = np.array([mapping[val] for val in y])
    
    return X[:len(y_mapped)], y_mapped

def load_subject_data(folder):
    """
    指定されたフォルダ内の .pkl ファイルを読み込み、ウィンドウ分割を行います。
    """
    files = [f for f in os.listdir(folder) if f.endswith('.pkl')]
    if not files:
        return np.array([]), np.array([])
    try:
        with open(os.path.join(folder, files[0]), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return segment_data_raw(data)
    except Exception as e:
        print(f"Skip {folder}: {e}")
        return np.array([]), np.array([])

def load_data_per_subject():
    """
    全被験者のデータを読み込み、被験者IDをキーとした辞書形式で返します。
    """
    subjects_data = {}
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return {}
        
    for subj in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, subj)
        if not os.path.isdir(path):
            continue
        print(f"Loading data for: {subj}")
        
        X_sub, y_sub = load_subject_data(path)
        if X_sub.size == 0:
            continue
            
        X_sub, y_sub = filter_map_labels_binary(X_sub, y_sub)
        
        # ---------------------------------------------------
        # 【正しい修正②】最初の5分間（60ウィンドウ）だけを使って標準化の基準を作る
        # ---------------------------------------------------
        CALIBRATION_WINDOWS = 60 # 最初の60ウィンドウ（5分間）
        
        flat = X_sub.reshape(X_sub.shape[0], -1)
        scaler = StandardScaler()
        
        # 単純に「最初の60行」だけを基準にするのが正解でした
        if len(flat) > CALIBRATION_WINDOWS:
            scaler.fit(flat[:CALIBRATION_WINDOWS])
        else:
            scaler.fit(flat) 
            
        scaled = scaler.transform(flat).astype(np.float32)
        X_sub_scaled = scaled.reshape(-1, WINDOW_SIZE, NUM_CHANNELS)
        
        subjects_data[subj] = {'X': X_sub_scaled, 'y': y_sub}
        
    return subjects_data

# --- 3. モデル構築 (Build Model) ---
# (ここは変更なし)

def build_1d_cnn_binary(input_shape):
    return Sequential([
        Input(shape=input_shape),
        Conv1D(32, 7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        GlobalAveragePooling1D(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

def stack_subjects(subjects_data, subject_list):
    """
    指定した被験者群のデータを結合して返します。
    """
    X_list = [subjects_data[subj]['X'] for subj in subject_list]
    y_list = [subjects_data[subj]['y'] for subj in subject_list]
    return np.vstack(X_list), np.concatenate(y_list)

def compute_balanced_class_weights(y):
    """
    学習データに存在するクラスに対してクラス重みを計算します。
    """
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {cls: weight for cls, weight in zip(classes, weights)}

def train_with_subject_validation(subjects_data, train_subjects, val_subject, max_epochs=20, batch_size=64):
    """
    1つの内側foldを学習し、検証被験者上の性能と最適epochを返します。
    """
    X_train, y_train = stack_subjects(subjects_data, train_subjects)
    X_val = subjects_data[val_subject]['X']
    y_val = subjects_data[val_subject]['y']

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    model = build_1d_cnn_binary((WINDOW_SIZE, NUM_CHANNELS))
    model.compile(optimizer=Adam(5e-4), loss='binary_crossentropy', metrics=['accuracy'])

    class_weights = compute_balanced_class_weights(y_train)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=0
    )

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    y_val_probs = model.predict(X_val, verbose=0)
    y_val_pred = (y_val_probs > 0.5).astype(int).ravel()
    val_f1 = f1_score(y_val, y_val_pred)
    best_epoch = int(np.argmin(history.history['val_loss']) + 1)

    fold_result = {
        'val_subject': val_subject,
        'train_subjects': train_subjects,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1,
        'best_epoch': best_epoch,
        'history': history.history
    }

    tf.keras.backend.clear_session()
    return fold_result

def run_inner_loso_cv(subjects_data, inner_subjects, max_epochs=20, batch_size=64):
    """
    test被験者を除いた被験者群で内側LOSOを実行し、最終学習に使うepoch数を決めます。
    """
    inner_results = []

    for val_subject in inner_subjects:
        train_subjects = [subj for subj in inner_subjects if subj != val_subject]
        print(f"  [Inner CV] Train on: {train_subjects}")
        print(f"  [Inner CV] Validate on: {val_subject}")

        fold_result = train_with_subject_validation(
            subjects_data,
            train_subjects,
            val_subject,
            max_epochs=max_epochs,
            batch_size=batch_size
        )
        inner_results.append(fold_result)
        print(
            f"  [Inner CV] {val_subject} - "
            f"Val Loss: {fold_result['val_loss']:.4f}, "
            f"Val Acc: {fold_result['val_accuracy']:.4f}, "
            f"Val F1: {fold_result['val_f1']:.4f}, "
            f"Best Epoch: {fold_result['best_epoch']}"
        )

    selected_epochs = int(np.clip(
        round(np.mean([res['best_epoch'] for res in inner_results])),
        1,
        max_epochs
    ))

    inner_summary = {
        'selected_epochs': selected_epochs,
        'avg_val_loss': float(np.mean([res['val_loss'] for res in inner_results])),
        'avg_val_accuracy': float(np.mean([res['val_accuracy'] for res in inner_results])),
        'avg_val_f1': float(np.mean([res['val_f1'] for res in inner_results])),
        'folds': inner_results
    }
    return inner_summary

def train_final_model(subjects_data, train_subjects, epochs, batch_size=64):
    """
    内側CVで決めたepoch数を用いて、test被験者以外の全データで最終学習します。
    """
    X_train, y_train = stack_subjects(subjects_data, train_subjects)
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    model = build_1d_cnn_binary((WINDOW_SIZE, NUM_CHANNELS))
    model.compile(optimizer=Adam(5e-4), loss='binary_crossentropy', metrics=['accuracy'])

    class_weights = compute_balanced_class_weights(y_train)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=0
    )
    return model, history

# --- メイン実行処理 (LOSO Cross-Validation) ---

if __name__ == "__main__":
    MAX_EPOCHS = 20
    BATCH_SIZE = 64

    # 1. 全被験者データの読み込み
    print("Loading data per subject for LOSO-CV...")
    subjects_data = load_data_per_subject()
    
    if not subjects_data:
        print("No data loaded. Exiting.")
        exit(1)

    subject_ids = list(subjects_data.keys())
    print(f"Loaded {len(subject_ids)} subjects: {subject_ids}")

    # LOSOの結果を保存するリスト
    loso_results = []
    
    # 全被験者に対してループ (Leave-One-Subject-Out)
    for test_subj in subject_ids:
        print(f"\n=== LOSO Iteration: Testing on {test_subj} ===")
        
        # 1. テストデータ (1人: 最終評価用)
        X_test = subjects_data[test_subj]['X']
        y_test = subjects_data[test_subj]['y']

        available_train_subjs = [s for s in subject_ids if s != test_subj]
        if len(available_train_subjs) < 2:
            print("Need at least 3 subjects for nested LOSO (train/val/test). Exiting.")
            exit(1)

        print(f"Outer train subjects: {available_train_subjs}")
        print(f"Test on: {test_subj}")
        print("Running inner LOSO-CV for validation subject rotation...")
        inner_summary = run_inner_loso_cv(
            subjects_data,
            available_train_subjs,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE
        )

        selected_epochs = inner_summary['selected_epochs']
        print(
            f"Inner CV summary - Avg Val Loss: {inner_summary['avg_val_loss']:.4f}, "
            f"Avg Val Acc: {inner_summary['avg_val_accuracy']:.4f}, "
            f"Avg Val F1: {inner_summary['avg_val_f1']:.4f}, "
            f"Selected Epochs: {selected_epochs}"
        )

        X_train_all, y_train_all = stack_subjects(subjects_data, available_train_subjs)
        print(
            f"Final train data: {X_train_all.shape}, "
            f"Test data: {X_test.shape}"
        )

        print(f"Training final model for {test_subj} with {selected_epochs} epochs...")
        model, history = train_final_model(
            subjects_data,
            available_train_subjs,
            epochs=selected_epochs,
            batch_size=BATCH_SIZE
        )
        
        # 評価
        print(f"Evaluating model on {test_subj}...")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # 予測とメトリクス算出
        y_probs = model.predict(X_test, verbose=0)
        y_pred = (y_probs > 0.5).astype(int) # 閾値は0.5固定 (または検証データで調整も可)
        
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"Subject {test_subj} - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        
        loso_results.append({
            'subject': test_subj,
            'accuracy': accuracy,
            'f1_score': f1,
            'selected_epochs': selected_epochs,
            'inner_cv_avg_val_f1': inner_summary['avg_val_f1'],
            'report': report,
            'inner_cv_summary': inner_summary,
            'history': history.history
        })
        
        # メモリ解放のためにKerasセッションをクリア
        tf.keras.backend.clear_session()

    # --- LOSO全体の集計結果表示 ---
    print("\n\n=== LOSO Cross-Validation Summary ===")
    accuracies = [res['accuracy'] for res in loso_results]
    f1_scores = [res['f1_score'] for res in loso_results]
    
    print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Average F1-Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
    
    print("\nDetailed Results per Subject:")
    for res in loso_results:
        print(f"  {res['subject']}: Acc={res['accuracy']:.4f}, F1={res['f1_score']:.4f}")

    # 結果をCSVに保存
    df_results = pd.DataFrame(loso_results)
    # reportカラムは辞書なので、主要なメトリクスだけ抽出して展開すると良いでしょう
    df_simple = pd.DataFrame({
        'Subject': [res['subject'] for res in loso_results],
        'Accuracy': [res['accuracy'] for res in loso_results],
        'F1_Score': [res['f1_score'] for res in loso_results]
    })
    df_simple.to_csv('loso_results.csv', index=False)
    print("LOSO results saved to loso_results.csv")
