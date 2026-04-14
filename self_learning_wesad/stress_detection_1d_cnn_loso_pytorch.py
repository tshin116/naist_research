import copy
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset

"""
WESAD の胸部センサーデータを用いたストレス二値分類を、PyTorch で実装したスクリプトです。

元の TensorFlow 版と同じ考え方で、以下の流れを実行します。
1. 被験者ごとに生データを 5 秒窓へ分割する
2. ラベルを「ストレス / 非ストレス」の 2 値へ変換する
3. 被験者ごとに最初の 5 分間を基準に標準化する
4. 外側 LOSO と内側 LOSO を使って学習・評価する
5. 内側 CV で決めた epoch 数で最終学習し、テスト被験者を評価する

PyTorch では Conv1d の入力形状が (batch, channels, time) なので、
前処理後に (window, time, channel) から軸を入れ替えて学習へ渡します。
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "WESAD")
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

WINDOW_SECONDS = 5
FS = 700
WINDOW_SIZE = WINDOW_SECONDS * FS
NUM_CHANNELS = 8

MAX_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EARLY_STOPPING_PATIENCE = 3
CALIBRATION_WINDOWS = 60
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """乱数シードを固定する。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """GPU が使える場合は CUDA、使えない場合は CPU を返す。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_window(chest, label_arr, start, end):
    """
    1 つの時間窓を切り出して 8 チャンネルの行列を作る。

    戻り値の形状は (WINDOW_SIZE, NUM_CHANNELS)。
    ラベルは窓内の最頻値を採用する。
    Windowサイズを3500に揃える。
    """
    def get_signal(key):
        signal = np.array(chest.get(key, np.zeros(WINDOW_SIZE))[start:end]).ravel()
        if len(signal) < WINDOW_SIZE:
            # np.pad(signal, (0, 増やす長さ))...signalに0を増やす長さ分伸ばす
            signal = np.pad(signal, (0, WINDOW_SIZE - len(signal)))
        return signal

    try:
        ecg = get_signal("ECG")
        eda = get_signal("EDA")
        emg = get_signal("EMG")
        resp = get_signal("Resp")
        temp = get_signal("Temp")

        # 無ければ(3500, 3)のゼロ配列
        acc = np.array(chest.get("ACC", np.zeros((WINDOW_SIZE, 3)))[start:end])
        # 2次元かつ3列なら
        if acc.ndim == 2 and acc.shape[1] == 3:
            acc_x, acc_y, acc_z = acc[:, 0], acc[:, 1], acc[:, 2]
        else:
            acc_x = acc_y = acc_z = np.zeros(WINDOW_SIZE)

        signals = [ecg, eda, emg, resp, temp, acc_x, acc_y, acc_z]
        # (8,3500)-> (3500, 8)へ転置？
        window_data = np.stack(signals, axis=-1)

        # 区間内のラベルを取り出す
        segment = label_arr[start:end].astype(int)
        # 代表ラベルを選ぶ
        mode_label = np.bincount(segment).argmax()
        return window_data, mode_label
    except Exception as exc:
        print(f"Error at window {start}: {exc}")
        return None, None


def segment_data_raw(data_dict):
    """被験者 1 名分の連続信号を固定長ウィンドウへ分割する。"""
    x_data, y_data = [], []
    if "signal" not in data_dict or "label" not in data_dict:
        return np.array([]), np.array([])

    chest = data_dict["signal"]["chest"]
    total_samples = len(chest["ECG"].flatten())
    num_windows = total_samples // WINDOW_SIZE

    for i in range(num_windows):
        start = i * WINDOW_SIZE
        end = start + WINDOW_SIZE
        window_data, label = get_window(chest, data_dict["label"], start, end)
        if window_data is not None:
            x_data.append(window_data)
            y_data.append(label)

    return np.array(x_data), np.array(y_data)


def filter_map_labels_binary(x_data, y_data):
    """
    WESAD のラベルから必要なものだけを残し、二値分類用へ変換する。

    1, 3, 4 -> 0 (非ストレス)
    2       -> 1 (ストレス)
    """
    valid = np.isin(y_data, [1, 2, 3, 4])
    x_data, y_data = x_data[valid], y_data[valid]

    mapping = {1: 0, 2: 1, 3: 0, 4: 0}
    y_mapped = np.array([mapping[val] for val in y_data], dtype=np.int64)
    return x_data[: len(y_mapped)], y_mapped


def load_subject_data(folder):
    """被験者フォルダ内の pkl を読み込み、ウィンドウ分割まで実行する。"""
    files = [name for name in os.listdir(folder) if name.endswith(".pkl")]
    if not files:
        return np.array([]), np.array([])

    try:
        with open(os.path.join(folder, files[0]), "rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        return segment_data_raw(data)
    except Exception as exc:
        print(f"Skip {folder}: {exc}")
        return np.array([]), np.array([])


def load_data_per_subject():
    """
    全被験者のデータを読み込む。

    標準化は「被験者ごと」に行う。
    さらに、最初の 60 ウィンドウだけで scaler を fit しているため、
    学習時に未来の情報を使わず、被験者冒頭の安静区間を基準にそろえる設計になっている。
    """
    subjects_data = {}
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory not found at {DATASET_DIR}")
        return {}

    for subject_id in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, subject_id)
        if not os.path.isdir(path):
            continue

        print(f"Loading data for: {subject_id}")
        x_sub, y_sub = load_subject_data(path)
        if x_sub.size == 0:
            continue

        x_sub, y_sub = filter_map_labels_binary(x_sub, y_sub)

        flat = x_sub.reshape(x_sub.shape[0], -1)
        scaler = StandardScaler()
        if len(flat) > CALIBRATION_WINDOWS:
            scaler.fit(flat[:CALIBRATION_WINDOWS])
        else:
            scaler.fit(flat)

        scaled = scaler.transform(flat).astype(np.float32)
        x_sub_scaled = scaled.reshape(-1, WINDOW_SIZE, NUM_CHANNELS)

        subjects_data[subject_id] = {"X": x_sub_scaled, "y": y_sub}

    return subjects_data


class StressCNN1D(nn.Module):
    """
    TensorFlow 版の 1D CNN を PyTorch で書き直したモデル。

    構成:
    Conv1d -> ReLU -> BatchNorm -> MaxPool を 3 回
    -> Global Average Pooling
    -> 全結合層
    -> 二値分類 logits 出力
    """
    def __init__(self, num_channels=NUM_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x_input):
        x_input = self.features(x_input)
        return self.classifier(x_input).squeeze(-1)


def stack_subjects(subjects_data, subject_list):
    """複数被験者のデータをまとめて学習用の 1 配列へ結合する。"""
    x_list = [subjects_data[subj]["X"] for subj in subject_list]
    y_list = [subjects_data[subj]["y"] for subj in subject_list]
    return np.vstack(x_list), np.concatenate(y_list)


def prepare_tensors(x_data, y_data):
    """
    NumPy 配列を PyTorch テンソルへ変換する。

    元データは (batch, time, channel) だが、Conv1d 用に
    (batch, channel, time) へ並べ替える。
    """
    x_tensor = torch.from_numpy(np.transpose(x_data, (0, 2, 1))).float()
    y_tensor = torch.from_numpy(y_data.astype(np.float32))
    return x_tensor, y_tensor


def make_loader(x_data, y_data, batch_size, shuffle_data):
    """学習・検証で共通利用する DataLoader を作る。"""
    x_tensor, y_tensor = prepare_tensors(x_data, y_data)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)


def compute_pos_weight(y_data):
    """
    クラス不均衡に対応するため、正例クラスの重みを計算する。

    BCEWithLogitsLoss の pos_weight に渡し、
    ストレス例が少ないときも学習が偏りにくいようにする。
    """
    positive_count = int(np.sum(y_data == 1))
    negative_count = int(np.sum(y_data == 0))
    if positive_count == 0:
        return None
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)


def evaluate_model(model, data_loader, criterion, device):
    """
    モデルを評価し、loss・accuracy・F1 と予測結果をまとめて返す。

    scikit-learn の f1_score は、二値分類で average を指定しない場合、
    デフォルトで label=1 を正例とした F1 だけを返す。
    ここでは多数派クラスに評価が偏らないように、label=0 と label=1 の F1 を個別に計算し、
    さらに両者の平均である mean_f1 と混同行列も返す。
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    probabilities = []
    labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    y_pred = (y_prob > 0.5).astype(int)
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = float(np.mean(y_pred == y_true))
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    f1_non_stress = float(f1_per_class[0])
    f1_stress = float(f1_per_class[1])
    mean_f1 = float(np.mean(f1_per_class))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1_stress,
        "f1_non_stress": f1_non_stress,
        "f1_stress": f1_stress,
        "mean_f1": mean_f1,
        "confusion_matrix": conf_matrix,
        "y_true": y_true.astype(int),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def format_confusion_matrix(conf_matrix):
    """
    混同行列を読みやすい文字列に整形する。

    labels=[0, 1] で計算しているため、行が正解ラベル、列が予測ラベル。
    [[TN, FP],
     [FN, TP]]
    """
    tn, fp, fn, tp = conf_matrix.ravel()
    return (
        "Confusion Matrix (rows=true, cols=pred)\n"
        "              Pred 0   Pred 1\n"
        f"True 0        {tn:6d}   {fp:6d}\n"
        f"True 1        {fn:6d}   {tp:6d}"
    )


def save_loso_checkpoint(model, test_subject, train_subjects, selected_epochs, test_metrics, inner_summary):
    """
    外側 LOSO の各 fold で得られた最終モデルを保存する。

    state_dict に加えて、あとで解析しやすいように被験者IDや評価指標も一緒に保存する。
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(CHECKPOINT_DIR, f"loso_test_{test_subject}.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "selected_epochs": selected_epochs,
            "window_seconds": WINDOW_SECONDS,
            "sampling_rate": FS,
            "window_size": WINDOW_SIZE,
            "num_channels": NUM_CHANNELS,
            "metrics": {
                "accuracy": test_metrics["accuracy"],
                "f1_non_stress": test_metrics["f1_non_stress"],
                "f1_stress": test_metrics["f1_stress"],
                "mean_f1": test_metrics["mean_f1"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            },
            "inner_cv_summary": inner_summary,
        },
        save_path,
    )
    return save_path


def fit_model(model, train_loader, val_loader, epochs, device, pos_weight=None, patience=None):
    """
    学習ループ本体。

    val_loader がある場合は各 epoch 後に検証し、検証 loss が最も良かった重みを保持する。
    patience を指定すると、改善が止まった時点で早期終了する。
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"loss": [], "accuracy": []}
    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []

    best_state = None
    best_val_loss = float("inf")
    best_epoch = epochs
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            # 出力は logits なので、予測ラベル化の前に sigmoid を通す。
            predictions = (torch.sigmoid(logits) > 0.5).float()
            total_correct += int((predictions == y_batch).sum().item())
            total_samples += batch_size

        history["loss"].append(total_loss / max(total_samples, 1))
        history["accuracy"].append(total_correct / max(total_samples, 1))

        if val_loader is None:
            continue

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_epoch, criterion


def train_with_subject_validation(subjects_data, train_subjects, val_subject, device, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    """
    内側 LOSO の 1 fold を学習する。

    ある 1 名を検証用被験者に固定し、それ以外で学習して
    検証性能と最適 epoch を返す。
    """
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_val = subjects_data[val_subject]["X"]
    y_val = subjects_data[val_subject]["y"]

    x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle_data=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle_data=False)

    model = StressCNN1D().to(device)
    pos_weight = compute_pos_weight(y_train)
    history, best_epoch, criterion = fit_model(
        model,
        train_loader,
        val_loader,
        epochs=max_epochs,
        device=device,
        pos_weight=pos_weight,
        patience=EARLY_STOPPING_PATIENCE,
    )

    val_metrics = evaluate_model(model, val_loader, criterion, device)
    fold_result = {
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
        "val_f1_non_stress": val_metrics["f1_non_stress"],
        "val_f1_stress": val_metrics["f1_stress"],
        "val_mean_f1": val_metrics["mean_f1"],
        "best_epoch": int(best_epoch),
        "history": history,
    }
    return fold_result


def run_inner_loso_cv(subjects_data, inner_subjects, device, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE):
    """
    テスト被験者を除いた集合に対して内側 LOSO を回す。

    各 fold の best epoch を平均し、外側 fold の最終学習 epoch として使う。
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
            device=device,
            max_epochs=max_epochs,
            batch_size=batch_size,
        )
        inner_results.append(fold_result)
        print(
            f"  [Inner CV] {val_subject} - "
            f"Val Loss: {fold_result['val_loss']:.4f}, "
            f"Val Acc: {fold_result['val_accuracy']:.4f}, "
            f"Val F1(0): {fold_result['val_f1_non_stress']:.4f}, "
            f"Val F1(1): {fold_result['val_f1_stress']:.4f}, "
            f"Val Mean F1: {fold_result['val_mean_f1']:.4f}, "
            f"Best Epoch: {fold_result['best_epoch']}"
        )

    selected_epochs = int(
        np.clip(
            round(np.mean([result["best_epoch"] for result in inner_results])),
            1,
            max_epochs,
        )
    )

    return {
        "selected_epochs": selected_epochs,
        "avg_val_loss": float(np.mean([result["val_loss"] for result in inner_results])),
        "avg_val_accuracy": float(np.mean([result["val_accuracy"] for result in inner_results])),
        "avg_val_f1": float(np.mean([result["val_f1"] for result in inner_results])),
        "avg_val_f1_non_stress": float(np.mean([result["val_f1_non_stress"] for result in inner_results])),
        "avg_val_f1_stress": float(np.mean([result["val_f1_stress"] for result in inner_results])),
        "avg_val_mean_f1": float(np.mean([result["val_mean_f1"] for result in inner_results])),
        "folds": inner_results,
    }


def train_final_model(subjects_data, train_subjects, device, epochs, batch_size=BATCH_SIZE):
    """内側 CV で決めた epoch 数を使い、訓練被験者全体で最終学習する。"""
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_SEED)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle_data=True)
    model = StressCNN1D().to(device)
    pos_weight = compute_pos_weight(y_train)
    history, _, criterion = fit_model(
        model,
        train_loader,
        val_loader=None,
        epochs=epochs,
        device=device,
        pos_weight=pos_weight,
        patience=None,
    )
    return model, history, criterion


if __name__ == "__main__":
    # 実行全体の流れ:
    # 1. 全被験者データを読み込む
    # 2. 外側 LOSO で 1 名ずつテスト被験者にする
    # 3. 残り被験者で内側 LOSO を回して epoch 数を決める
    # 4. 残り被験者全体で最終学習し、テスト被験者を評価する
    # 5. 全被験者の結果を平均し、CSV に保存する
    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    print("Loading data per subject for LOSO-CV...")
    subjects_data = load_data_per_subject()

    if not subjects_data:
        print("No data loaded. Exiting.")
        raise SystemExit(1)

    subject_ids = list(subjects_data.keys())
    print(f"Loaded {len(subject_ids)} subjects: {subject_ids}")

    loso_results = []

    for test_subj in subject_ids:
        print(f"\n=== LOSO Iteration: Testing on {test_subj} ===")

        # 外側 fold では test_subj を完全に未知の被験者として扱う。
        x_test = subjects_data[test_subj]["X"]
        y_test = subjects_data[test_subj]["y"]

        available_train_subjs = [subject_id for subject_id in subject_ids if subject_id != test_subj]
        if len(available_train_subjs) < 2:
            print("Need at least 3 subjects for nested LOSO (train/val/test). Exiting.")
            raise SystemExit(1)

        print(f"Outer train subjects: {available_train_subjs}")
        print(f"Test on: {test_subj}")
        print("Running inner LOSO-CV for validation subject rotation...")

        # 内側 CV の目的は「最終学習を何 epoch で止めるか」を決めること。
        inner_summary = run_inner_loso_cv(
            subjects_data,
            available_train_subjs,
            device=device,
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
        )

        selected_epochs = inner_summary["selected_epochs"]
        print(
            f"Inner CV summary - Avg Val Loss: {inner_summary['avg_val_loss']:.4f}, "
            f"Avg Val Acc: {inner_summary['avg_val_accuracy']:.4f}, "
            f"Avg Val F1(0): {inner_summary['avg_val_f1_non_stress']:.4f}, "
            f"Avg Val F1(1): {inner_summary['avg_val_f1_stress']:.4f}, "
            f"Avg Val Mean F1: {inner_summary['avg_val_mean_f1']:.4f}, "
            f"Selected Epochs: {selected_epochs}"
        )

        x_train_all, _ = stack_subjects(subjects_data, available_train_subjs)
        print(f"Final train data: {x_train_all.shape}, Test data: {x_test.shape}")

        print(f"Training final model for {test_subj} with {selected_epochs} epochs...")
        model, history, criterion = train_final_model(
            subjects_data,
            available_train_subjs,
            device=device,
            epochs=selected_epochs,
            batch_size=BATCH_SIZE,
        )

        print(f"Evaluating model on {test_subj}...")
        test_loader = make_loader(x_test, y_test, batch_size=BATCH_SIZE, shuffle_data=False)
        test_metrics = evaluate_model(model, test_loader, criterion, device)

        report = classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            output_dict=True,
            zero_division=0,
        )

        print(
            f"Subject {test_subj} - "
            f"Accuracy: {test_metrics['accuracy']:.4f}, "
            f"F1(Non-Stress=0): {test_metrics['f1_non_stress']:.4f}, "
            f"F1(Stress=1): {test_metrics['f1_stress']:.4f}, "
            f"Mean F1: {test_metrics['mean_f1']:.4f}"
        )
        print(format_confusion_matrix(test_metrics["confusion_matrix"]))

        checkpoint_path = save_loso_checkpoint(
            model=model,
            test_subject=test_subj,
            train_subjects=available_train_subjs,
            selected_epochs=selected_epochs,
            test_metrics=test_metrics,
            inner_summary=inner_summary,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

        loso_results.append(
            {
                "subject": test_subj,
                "accuracy": test_metrics["accuracy"],
                "f1_score": test_metrics["f1"],
                "f1_non_stress": test_metrics["f1_non_stress"],
                "f1_stress": test_metrics["f1_stress"],
                "mean_f1": test_metrics["mean_f1"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "selected_epochs": selected_epochs,
                "inner_cv_avg_val_f1": inner_summary["avg_val_f1"],
                "inner_cv_avg_val_f1_non_stress": inner_summary["avg_val_f1_non_stress"],
                "inner_cv_avg_val_f1_stress": inner_summary["avg_val_f1_stress"],
                "inner_cv_avg_val_mean_f1": inner_summary["avg_val_mean_f1"],
                "checkpoint_path": checkpoint_path,
                "report": report,
                "inner_cv_summary": inner_summary,
                "history": history,
            }
        )

    print("\n\n=== LOSO Cross-Validation Summary ===")
    accuracies = [result["accuracy"] for result in loso_results]
    f1_non_stress_scores = [result["f1_non_stress"] for result in loso_results]
    f1_stress_scores = [result["f1_stress"] for result in loso_results]
    mean_f1_scores = [result["mean_f1"] for result in loso_results]
    total_confusion_matrix = np.sum([result["confusion_matrix"] for result in loso_results], axis=0)

    print(f"Average Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    print(f"Average F1 Non-Stress(0): {np.mean(f1_non_stress_scores):.4f} (+/- {np.std(f1_non_stress_scores):.4f})")
    print(f"Average F1 Stress(1): {np.mean(f1_stress_scores):.4f} (+/- {np.std(f1_stress_scores):.4f})")
    print(f"Average Mean F1: {np.mean(mean_f1_scores):.4f} (+/- {np.std(mean_f1_scores):.4f})")
    print("\nTotal Confusion Matrix over all LOSO folds:")
    print(format_confusion_matrix(total_confusion_matrix))

    print("\nDetailed Results per Subject:")
    for result in loso_results:
        print(
            f"  {result['subject']}: "
            f"Acc={result['accuracy']:.4f}, "
            f"F1(0)={result['f1_non_stress']:.4f}, "
            f"F1(1)={result['f1_stress']:.4f}, "
            f"MeanF1={result['mean_f1']:.4f}"
        )
        print(format_confusion_matrix(result["confusion_matrix"]))

    tn_list = [int(result["confusion_matrix"][0, 0]) for result in loso_results]
    fp_list = [int(result["confusion_matrix"][0, 1]) for result in loso_results]
    fn_list = [int(result["confusion_matrix"][1, 0]) for result in loso_results]
    tp_list = [int(result["confusion_matrix"][1, 1]) for result in loso_results]
    df_simple = pd.DataFrame(
        {
            "Subject": [result["subject"] for result in loso_results],
            "Accuracy": [result["accuracy"] for result in loso_results],
            "F1_Non_Stress_0": [result["f1_non_stress"] for result in loso_results],
            "F1_Stress_1": [result["f1_stress"] for result in loso_results],
            "Mean_F1": [result["mean_f1"] for result in loso_results],
            "TN": tn_list,
            "FP": fp_list,
            "FN": fn_list,
            "TP": tp_list,
            "Selected_Epochs": [result["selected_epochs"] for result in loso_results],
            "Inner_CV_Avg_F1_Non_Stress_0": [result["inner_cv_avg_val_f1_non_stress"] for result in loso_results],
            "Inner_CV_Avg_F1_Stress_1": [result["inner_cv_avg_val_f1_stress"] for result in loso_results],
            "Inner_CV_Avg_Mean_F1": [result["inner_cv_avg_val_mean_f1"] for result in loso_results],
        }
    )
    output_path = os.path.join(SCRIPT_DIR, "loso_results_pytorch.csv")
    df_simple.to_csv(output_path, index=False)
    print(f"LOSO results saved to {output_path}")
