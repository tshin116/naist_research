"""EmoWear dataset loading and preprocessing.

This module uses the cleaned/synchronized CSV package. Each participant folder
contains trial-level SAM ratings in ``surveys.csv`` and trial timing markers in
``markers-phase2.csv``. We extract video-viewing intervals and assign the trial
SAM valence/arousal label to all windows in that interval.

The first implementation intentionally uses only sensors available for all
participants: Empatica E4 and BioHarness 3. SensorTile ACC/GYRO files are large,
partly missing, and have irregular timestamps, so they are left for a later
explicit extension.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SIGNAL_SPECS = [
    ("signals-e4-bvp.csv", "value", "e4_bvp"),
    ("signals-e4-eda.csv", "value", "e4_eda"),
    ("signals-e4-skt.csv", "value", "e4_skt"),
    ("signals-e4-acc.csv", "x", "e4_acc_x"),
    ("signals-e4-acc.csv", "y", "e4_acc_y"),
    ("signals-e4-acc.csv", "z", "e4_acc_z"),
    ("signals-bh3-ecg.csv", "value", "bh3_ecg"),
    ("signals-bh3-rsp.csv", "value", "bh3_rsp"),
]

LABEL_TYPES = {"VALENCE": "valence", "AROUSAL": "arousal"}


def discover_subjects(dataset_dir):
    """Return EmoWear participant folder names in numeric order."""
    if not os.path.isdir(dataset_dir):
        return []
    subjects = []
    for name in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, name)
        prefix = name.split("-", maxsplit=1)[0]
        if os.path.isdir(path) and prefix.isdigit():
            subjects.append(name)
    return sorted(subjects, key=lambda subject: int(subject.split("-", maxsplit=1)[0]))


def _subject_dir(args, subject_id):
    return os.path.join(os.path.abspath(args.dataset_dir), subject_id)


def _label_threshold(args, trials, label_column):
    threshold_mode = getattr(args, "label_threshold_mode", "global_median")
    configured = getattr(args, "label_threshold", None)
    if configured is not None:
        return float(configured)
    if threshold_mode == "subject_mean":
        return float(trials[label_column].mean())
    if threshold_mode == "subject_median":
        return float(trials[label_column].median())
    if threshold_mode == "scale_midpoint":
        return 5.0
    # Defaults to global medians observed in the current CSV package.
    if label_column == "valence":
        return 5.1
    if label_column == "arousal":
        return 5.0
    raise ValueError(f"Unknown EmoWear label column: {label_column}")


def _read_signal_columns(subject_path, filename, columns):
    path = os.path.join(subject_path, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"EmoWear signal file not found: {path}")
    usecols = ["timestamp", *sorted(set(columns))]
    frame = pd.read_csv(path, usecols=usecols)
    timestamps = frame["timestamp"].to_numpy(dtype=np.float64)
    order = np.argsort(timestamps, kind="mergesort")
    timestamps = timestamps[order]
    unique_timestamps, unique_index = np.unique(timestamps, return_index=True)

    values = {}
    for column in columns:
        arr = frame[column].to_numpy(dtype=np.float32)[order][unique_index]
        values[column] = arr
    return unique_timestamps, values


def _load_subject_signals(subject_path):
    grouped = {}
    for filename, column, _ in SIGNAL_SPECS:
        grouped.setdefault(filename, []).append(column)

    loaded = {}
    for filename, columns in grouped.items():
        timestamps, values = _read_signal_columns(subject_path, filename, columns)
        for column, signal in values.items():
            loaded[(filename, column)] = (timestamps, signal)
    return loaded


def _load_trials(subject_path):
    surveys_path = os.path.join(subject_path, "surveys.csv")
    markers_path = os.path.join(subject_path, "markers-phase2.csv")
    if not os.path.exists(surveys_path):
        raise FileNotFoundError(f"EmoWear surveys.csv not found: {surveys_path}")
    if not os.path.exists(markers_path):
        raise FileNotFoundError(f"EmoWear markers-phase2.csv not found: {markers_path}")
    surveys = pd.read_csv(surveys_path)
    markers = pd.read_csv(markers_path)
    trials = surveys.merge(markers[["seq", "exp", "vidB", "postB"]], on=["seq", "exp"], how="inner")
    trials = trials.dropna(subset=["vidB", "postB"]).copy()
    trials = trials[trials["postB"] > trials["vidB"]].reset_index(drop=True)
    if trials.empty:
        raise ValueError(f"No valid EmoWear trials found under {subject_path}")
    return trials


def processed_subject_path(args, subject_id):
    label_type = getattr(args, "label_type", "VALENCE").lower()
    threshold_mode = getattr(args, "label_threshold_mode", "global_median")
    filename = (
        f"{subject_id}_{label_type}_{threshold_mode}_"
        f"ws{args.window_size}_stride{getattr(args, 'stride_size', args.window_size)}_"
        f"sr{args.sampling_rate}_ch{args.num_channels}.npz"
    )
    return os.path.join(os.path.abspath(args.processed_dir), filename)


def save_processed_subject(args, subject_id, x_data, y_data, metadata):
    save_path = processed_subject_path(args, subject_id)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        X=x_data.astype(np.float32),
        y=y_data.astype(np.int64),
        subject_id=subject_id,
        label_type=getattr(args, "label_type", "VALENCE"),
        label_threshold=float(metadata["label_threshold"]),
        signal_names=np.asarray(metadata["signal_names"]),
        window_size=args.window_size,
        stride_size=getattr(args, "stride_size", args.window_size),
        sampling_rate=args.sampling_rate,
        num_channels=args.num_channels,
    )
    return save_path


def load_processed_subject(args, subject_id):
    load_path = processed_subject_path(args, subject_id)
    if not os.path.exists(load_path):
        return None
    with np.load(load_path, allow_pickle=False) as data:
        return {"X": data["X"].astype(np.float32), "y": data["y"].astype(np.int64)}


def preprocess_subject(args, subject_id):
    label_type = getattr(args, "label_type", "VALENCE").upper()
    if label_type not in LABEL_TYPES:
        raise ValueError(f"Unknown EmoWear label_type: {label_type}. Choose AROUSAL or VALENCE.")
    if args.num_channels != len(SIGNAL_SPECS):
        raise ValueError(f"EmoWear E4/BH3 preprocessing expects num_channels={len(SIGNAL_SPECS)}")

    subject_path = _subject_dir(args, subject_id)
    if not os.path.isdir(subject_path):
        raise FileNotFoundError(f"EmoWear subject folder not found: {subject_path}")

    trials = _load_trials(subject_path)
    label_column = LABEL_TYPES[label_type]
    threshold = _label_threshold(args, trials, label_column)
    signals = _load_subject_signals(subject_path)

    sampling_rate = int(args.sampling_rate)
    window_size = int(args.window_size)
    stride_size = int(getattr(args, "stride_size", window_size))
    signal_names = [name for _, _, name in SIGNAL_SPECS]
    windows = []
    labels = []

    for _, trial in trials.iterrows():
        start_time = float(trial["vidB"])
        end_time = float(trial["postB"])
        if end_time <= start_time:
            continue
        grid = np.arange(start_time, end_time, 1.0 / sampling_rate, dtype=np.float64)
        if len(grid) < window_size:
            continue

        channels = []
        valid_trial = True
        for filename, column, _ in SIGNAL_SPECS:
            timestamps, values = signals[(filename, column)]
            if grid[0] < timestamps[0] or grid[-1] > timestamps[-1]:
                valid_trial = False
                break
            channels.append(np.interp(grid, timestamps, values).astype(np.float32))
        if not valid_trial:
            continue

        trial_signal = np.stack(channels, axis=-1)
        trial_label = 1 if float(trial[label_column]) >= threshold else 0
        for start in range(0, len(trial_signal) - window_size + 1, stride_size):
            windows.append(trial_signal[start : start + window_size])
            labels.append(trial_label)

    if not windows:
        raise ValueError(f"No EmoWear windows created for subject {subject_id}")

    x_data = np.asarray(windows, dtype=np.float32)
    y_data = np.asarray(labels, dtype=np.int64)

    scaler = StandardScaler()
    flat = x_data.reshape(-1, x_data.shape[-1])
    scaled = scaler.fit_transform(flat).astype(np.float32)
    x_data = scaled.reshape(x_data.shape)

    return {
        "X": x_data,
        "y": y_data,
        "metadata": {"label_threshold": threshold, "signal_names": signal_names},
    }


def load_or_preprocess_subject(args, subject_id):
    if getattr(args, "use_processed", True):
        cached = load_processed_subject(args, subject_id)
        if cached is not None:
            print(f"Loading processed EmoWear data for: {subject_id}")
            return cached

    print(f"Preprocessing EmoWear raw data for: {subject_id}")
    subject_data = preprocess_subject(args, subject_id)
    if getattr(args, "use_processed", True):
        save_path = save_processed_subject(
            args,
            subject_id,
            subject_data["X"],
            subject_data["y"],
            subject_data["metadata"],
        )
        print(f"Processed EmoWear data saved to: {save_path}")
    return {"X": subject_data["X"], "y": subject_data["y"]}


def load_data_per_subject(args):
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"EmoWear dataset directory not found: {dataset_dir}")
    subject_ids = args.subjects or discover_subjects(dataset_dir)
    return {subject_id: load_or_preprocess_subject(args, subject_id) for subject_id in subject_ids}


def stack_subjects(subjects_data, subject_list):
    x_list = [subjects_data[subj]["X"] for subj in subject_list]
    y_list = [subjects_data[subj]["y"] for subj in subject_list]
    return np.vstack(x_list), np.concatenate(y_list)


def prepare_tensors(x_data, y_data, label_mode="binary"):
    x_tensor = torch.from_numpy(np.transpose(x_data, (0, 2, 1))).float()
    y_tensor = torch.from_numpy(y_data.astype(np.float32))
    return x_tensor, y_tensor


def make_loader(x_data, y_data, batch_size, shuffle_data=False, num_workers=0, label_mode="binary"):
    x_tensor, y_tensor = prepare_tensors(x_data, y_data, label_mode=label_mode)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_workers)


def make_target_loader(
    x_data,
    y_data,
    batch_size,
    shuffle_data=False,
    class_balanced=False,
    num_workers=0,
    label_mode="binary",
    num_classes=None,
    seed=42,
):
    if class_balanced:
        raise NotImplementedError("EmoWear class-balanced target loader is not implemented yet.")
    return make_loader(x_data, y_data, batch_size, shuffle_data=shuffle_data, num_workers=num_workers)


def get_loso_loaders(args, subjects_data=None):
    if subjects_data is None:
        subjects_data = load_data_per_subject(args)
    subject_ids = list(subjects_data.keys())
    if args.target_domain not in subject_ids:
        raise ValueError(f"target_domain {args.target_domain} not found in loaded EmoWear subjects: {subject_ids}")
    train_subjects = [subject_id for subject_id in subject_ids if subject_id != args.target_domain]
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_test = subjects_data[args.target_domain]["X"]
    y_test = subjects_data[args.target_domain]["y"]
    source_loader = make_loader(x_train, y_train, args.batch_size, shuffle_data=True, num_workers=args.num_workers)
    target_loader = make_loader(x_test, y_test, args.batch_size, shuffle_data=False, num_workers=args.num_workers)
    return source_loader, target_loader, train_subjects


def get_target_loader(args):
    subject_data = load_or_preprocess_subject(args, args.target_domain)
    return make_target_loader(
        subject_data["X"],
        subject_data["y"],
        args.batch_size,
        shuffle_data=getattr(args, "target_shuffle", False),
        num_workers=args.num_workers,
    )
