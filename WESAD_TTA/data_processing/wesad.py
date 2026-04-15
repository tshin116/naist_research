"""WESAD データセットの読み込みと前処理。

WESAD の胸部センサ信号を被験者ごとに読み込み、固定長の時系列窓へ分割する。
このファイルでは、元ラベルを 3 クラス（中性・ストレス・楽しさ）へ写像し、
被験者ごとの冒頭窓を基準に標準化したうえで PyTorch の `DataLoader` を作成する。
"""

import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


LABEL_MAPPING = {1: 0, 2: 1, 3: 2}


def get_window(chest, label_arr, start, end, window_size):
    """1 つの時間窓を切り出し、8 チャンネルの行列と代表ラベルを返す。

    胸部センサから ECG、EDA、EMG、Resp、Temp、ACC x/y/z を取り出す。
    ラベルは窓内の最頻値を採用する。信号長が足りない場合は 0 padding する。
    """
    def get_signal(key):
        """単一センサ信号を指定区間で取り出し、窓長へそろえる。"""
        signal = np.array(chest.get(key, np.zeros(window_size))[start:end]).ravel()
        if len(signal) < window_size:
            signal = np.pad(signal, (0, window_size - len(signal)))
        return signal

    ecg = get_signal("ECG")
    eda = get_signal("EDA")
    emg = get_signal("EMG")
    resp = get_signal("Resp")
    temp = get_signal("Temp")

    acc = np.array(chest.get("ACC", np.zeros((window_size, 3)))[start:end])
    if acc.ndim == 2 and acc.shape[1] == 3:
        acc_x, acc_y, acc_z = acc[:, 0], acc[:, 1], acc[:, 2]
    else:
        acc_x = acc_y = acc_z = np.zeros(window_size)

    window_data = np.stack([ecg, eda, emg, resp, temp, acc_x, acc_y, acc_z], axis=-1)
    segment = label_arr[start:end].astype(int)
    mode_label = np.bincount(segment).argmax()
    return window_data, mode_label


def segment_data_raw(data_dict, window_size):
    """被験者 1 名分の連続信号を固定長窓へ分割する。"""
    if "signal" not in data_dict or "label" not in data_dict:
        return np.array([]), np.array([])

    chest = data_dict["signal"]["chest"]
    total_samples = len(chest["ECG"].flatten())
    num_windows = total_samples // window_size
    x_data, y_data = [], []

    for idx in range(num_windows):
        start = idx * window_size
        end = start + window_size
        window_data, label = get_window(chest, data_dict["label"], start, end, window_size)
        x_data.append(window_data)
        y_data.append(label)

    return np.array(x_data), np.array(y_data)


def filter_map_labels(x_data, y_data):
    """WESAD の元ラベルを 3 クラスへ変換する。

    使用するラベルは 1 (baseline→0), 2 (stress→1), 3 (amusement→2) のみ。
    ラベル 0, 4, 6, 7 は除外する。
    """
    valid = np.isin(y_data, list(LABEL_MAPPING.keys()))
    x_data, y_data = x_data[valid], y_data[valid]
    y_mapped = np.array([LABEL_MAPPING[val] for val in y_data], dtype=np.int64)
    return x_data[: len(y_mapped)], y_mapped


def load_subject_data(folder, window_size):
    """被験者フォルダから `.pkl` を読み込み、窓分割まで実行する。"""
    files = [name for name in os.listdir(folder) if name.endswith(".pkl")]
    if not files:
        return np.array([]), np.array([])

    with open(os.path.join(folder, files[0]), "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    return segment_data_raw(data, window_size)


def processed_subject_path(args, subject_id):
    """前処理済み `.npz` の保存パスを返す。

    窓長、チャンネル数、標準化に使う calibration 窓数をファイル名に含めることで、
    設定を変えたときに古いキャッシュと混ざりにくくしている。
    """
    filename = (
        f"{subject_id}_ws{args.window_size}_"
        f"ch{args.num_channels}_cal{args.calibration_windows}.npz"
    )
    return os.path.join(os.path.abspath(args.processed_dir), filename)


def save_processed_subject(args, subject_id, x_data, y_data):
    """被験者 1 名分の前処理済み X/y を `.npz` として保存する。"""
    save_path = processed_subject_path(args, subject_id)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        X=x_data.astype(np.float32),
        y=y_data.astype(np.int64),
        subject_id=subject_id,
        window_size=args.window_size,
        num_channels=args.num_channels,
        calibration_windows=args.calibration_windows,
    )
    return save_path


def load_processed_subject(args, subject_id):
    """前処理済み `.npz` から被験者 1 名分の X/y を読み込む。"""
    load_path = processed_subject_path(args, subject_id)
    if not os.path.exists(load_path):
        return None

    with np.load(load_path, allow_pickle=False) as data:
        x_data = data["X"].astype(np.float32)
        y_data = data["y"].astype(np.int64)
    return {"X": x_data, "y": y_data}


def preprocess_subject(args, subject_id):
    """元の WESAD `.pkl` から被験者 1 名分を前処理する。

    この関数は cache を読まず、必ず raw data から窓分割、二値ラベル化、
    被験者内標準化までを実行する。
    """
    dataset_dir = os.path.abspath(args.dataset_dir)
    path = os.path.join(dataset_dir, subject_id)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Subject folder not found: {path}")

    x_sub, y_sub = load_subject_data(path, args.window_size)
    if x_sub.size == 0:
        raise ValueError(f"No WESAD windows were loaded for subject: {subject_id}")

    x_sub, y_sub = filter_map_labels(x_sub, y_sub)
    # StandardScaler は 2 次元入力を受け取るため、窓とチャンネルを一度 flatten する。
    flat = x_sub.reshape(x_sub.shape[0], -1)
    scaler = StandardScaler()
    if len(flat) > args.calibration_windows:
        scaler.fit(flat[: args.calibration_windows])
    else:
        scaler.fit(flat)

    scaled = scaler.transform(flat).astype(np.float32)
    x_sub_scaled = scaled.reshape(-1, args.window_size, args.num_channels)
    return {"X": x_sub_scaled, "y": y_sub}


def load_or_preprocess_subject(args, subject_id):
    """前処理済みデータがあれば読み込み、なければ raw data から作って保存する。"""
    if getattr(args, "use_processed", True):
        cached = load_processed_subject(args, subject_id)
        if cached is not None:
            print(f"Loading processed data for: {subject_id}")
            return cached

    print(f"Preprocessing raw data for: {subject_id}")
    subject_data = preprocess_subject(args, subject_id)
    if getattr(args, "use_processed", True):
        save_path = save_processed_subject(args, subject_id, subject_data["X"], subject_data["y"])
        print(f"Processed data saved to: {save_path}")
    return subject_data


def discover_subjects(dataset_dir):
    """データセットディレクトリから `S2` のような被験者フォルダを列挙する。"""
    return [
        name
        for name in sorted(os.listdir(dataset_dir))
        if os.path.isdir(os.path.join(dataset_dir, name)) and name.startswith("S")
    ]


def load_data_per_subject(args):
    """全被験者のデータを読み込み、被験者単位で標準化して返す。

    標準化は被験者ごとに行う。各被験者の最初の `calibration_windows` 窓だけで
    scaler を fit することで、将来のテスト区間全体の統計を使いすぎない設計にしている。
    """
    subjects_data = {}
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    subject_ids = args.subjects or discover_subjects(dataset_dir)
    for subject_id in subject_ids:
        subjects_data[subject_id] = load_or_preprocess_subject(args, subject_id)

    return subjects_data


def stack_subjects(subjects_data, subject_list):
    """複数被験者の配列を結合し、学習用の X/y にまとめる。"""
    x_list = [subjects_data[subj]["X"] for subj in subject_list]
    y_list = [subjects_data[subj]["y"] for subj in subject_list]
    return np.vstack(x_list), np.concatenate(y_list)


def prepare_tensors(x_data, y_data):
    """NumPy 配列を Conv1d 用の PyTorch テンソルへ変換する。

    前処理後の形状は `(batch, time, channel)` だが、PyTorch の `Conv1d` は
    `(batch, channel, time)` を要求するため、ここで軸を入れ替える。
    """
    x_tensor = torch.from_numpy(np.transpose(x_data, (0, 2, 1))).float()
    y_tensor = torch.from_numpy(y_data.astype(np.int64))
    return x_tensor, y_tensor


def make_loader(x_data, y_data, batch_size, shuffle_data=False, num_workers=0):
    """X/y 配列から PyTorch `DataLoader` を作る。"""
    x_tensor, y_tensor = prepare_tensors(x_data, y_data)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_workers)


def get_loso_loaders(args, subjects_data=None):
    """指定した `target_domain` をテスト被験者とする LOSO 用 loader を返す。

    戻り値は source loader、target loader、学習に使った被験者 ID のリスト。
    `adapt.py` では target loader だけを評価に使う。
    """
    if subjects_data is None:
        subjects_data = load_data_per_subject(args)

    subject_ids = list(subjects_data.keys())
    if args.target_domain not in subject_ids:
        raise ValueError(f"target_domain {args.target_domain} not found in loaded subjects: {subject_ids}")

    train_subjects = [subject_id for subject_id in subject_ids if subject_id != args.target_domain]
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_test = subjects_data[args.target_domain]["X"]
    y_test = subjects_data[args.target_domain]["y"]

    source_loader = make_loader(x_train, y_train, args.batch_size, shuffle_data=True, num_workers=args.num_workers)
    target_loader = make_loader(x_test, y_test, args.batch_size, shuffle_data=False, num_workers=args.num_workers)
    return source_loader, target_loader, train_subjects


def get_target_loader(args):
    """評価専用 loader を作成する。

    学習では target 以外の被験者が必要だが、評価では target 被験者だけで十分。
    `adapt.py` ではこの関数を使うことで、S2 評価時に S2 以外のデータを読まない。
    """
    subject_data = load_or_preprocess_subject(args, args.target_domain)
    target_shuffle = getattr(args, "target_shuffle", False)
    target_loader = make_loader(
        subject_data["X"],
        subject_data["y"],
        args.batch_size,
        shuffle_data=target_shuffle,
        num_workers=args.num_workers,
    )
    return target_loader


def preprocess_all_subjects(args):
    """全被験者の前処理済み `.npz` を作成する。"""
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    subject_ids = args.subjects or discover_subjects(dataset_dir)
    saved_paths = []
    for subject_id in subject_ids:
        print(f"Preprocessing raw data for: {subject_id}")
        subject_data = preprocess_subject(args, subject_id)
        save_path = save_processed_subject(args, subject_id, subject_data["X"], subject_data["y"])
        saved_paths.append(save_path)
        print(f"Processed data saved to: {save_path}")
    return saved_paths
