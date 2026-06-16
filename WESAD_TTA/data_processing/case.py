"""CASE dataset loading and preprocessing for WESAD_TTA.

This module reads CASE raw files arranged as:

    dataset_dir/
      annotations/sub1_joystick.txt
      physiological/sub1_DAQ.txt

It follows the preprocessing used in Personalized_Affective_Computing:
winsorization, low-pass filtering, downsampling to 50 Hz, per-subject min-max
normalization, and 10 s windows with 5 s stride. Labels are binarized per
subject by comparing each window's mean arousal/valence annotation to that
subject's mean annotation.
"""

import os

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


SIGNAL_COLUMNS = ["ecg", "bvp", "gsr", "rsp", "temp", "emg_zygo", "emg_coru", "emg_trap"]
LABEL_COLUMNS = {"AROUSAL": "axis1", "VALENCE": "axis2"}


def _is_lfs_pointer(path):
    with open(path, "rb") as handle:
        head = handle.read(80)
    return head.startswith(b"version https://git-lfs.github.com/spec")


def discover_subjects(dataset_dir):
    annotation_dir = os.path.join(dataset_dir, "annotations")
    if not os.path.isdir(annotation_dir):
        return []
    subjects = []
    for name in os.listdir(annotation_dir):
        if name.startswith("sub") and name.endswith("_joystick.txt"):
            subject_num = name.removeprefix("sub").removesuffix("_joystick.txt")
            if subject_num.isdigit():
                subjects.append(f"S{int(subject_num)}")
    return sorted(subjects, key=lambda subject: int(subject[1:]))


def _subject_number(subject_id):
    if isinstance(subject_id, str) and subject_id.startswith("S"):
        return int(subject_id[1:])
    return int(subject_id)


def _raw_paths(args, subject_id):
    subject_num = _subject_number(subject_id)
    dataset_dir = os.path.abspath(args.dataset_dir)
    annotation_path = os.path.join(dataset_dir, "annotations", f"sub{subject_num}_joystick.txt")
    physiological_path = os.path.join(dataset_dir, "physiological", f"sub{subject_num}_DAQ.txt")
    return annotation_path, physiological_path


def _validate_raw_paths(annotation_path, physiological_path):
    for path in [annotation_path, physiological_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CASE raw file not found: {path}")
        if _is_lfs_pointer(path):
            raise RuntimeError(
                "CASE raw file is a Git LFS pointer, not the actual data. "
                f"Fetch LFS files first: {path}"
            )


def _active_annotation_ranges(annotation_df):
    ranges = []
    start = None
    end = None
    for index, row in annotation_df.iterrows():
        is_active = not (row["axis1"] == 0 and row["axis2"] == 0)
        if is_active:
            if start is None:
                start = index
            end = index
        elif start is not None:
            ranges.append((start, end))
            start = None
            end = None
    if start is not None:
        ranges.append((start, end))
    return ranges


def _select_active_segments(annotation_df, physiological_df):
    ranges = _active_annotation_ranges(annotation_df)
    annotation_segments = []
    physiological_segments = []
    for start, end in ranges:
        annotation_segments.append(annotation_df.iloc[start:end])
        start_time = annotation_df.loc[start, "time"]
        end_time = annotation_df.loc[end, "time"]
        phy_start = (physiological_df["time"] - start_time).abs().idxmin()
        phy_end = (physiological_df["time"] - end_time).abs().idxmin()
        physiological_segments.append(physiological_df.iloc[phy_start:phy_end])

    if not annotation_segments or not physiological_segments:
        raise ValueError("No active CASE annotation/physiological segments found.")
    return pd.concat(annotation_segments, ignore_index=True), pd.concat(physiological_segments, ignore_index=True)


def _lowpass_filter(values, cutoff=10, sampling_rate=1000, order=3):
    nyquist = 0.5 * sampling_rate
    b, a = scipy.signal.butter(order, cutoff / nyquist, btype="low", analog=False)
    return scipy.signal.lfilter(b, a, values)


def _filter_resample_normalize(values, original_rate=1000, target_rate=50):
    values = scipy.stats.mstats.winsorize(values, limits=[0.03, 0.03])
    if original_rate / 2 > 10:
        values = _lowpass_filter(values, cutoff=10, sampling_rate=original_rate)
    step = int(original_rate / target_rate)
    values = np.asarray(values)[::step].reshape(-1, 1)
    scaler = MinMaxScaler()
    return scaler.fit_transform(values).reshape(-1)


def processed_subject_path(args, subject_id):
    label_type = getattr(args, "label_type", "VALENCE").lower()
    filename = (
        f"{subject_id}_{label_type}_ws{args.window_size}_"
        f"stride{getattr(args, 'stride_size', args.window_size)}_ch{args.num_channels}.npz"
    )
    return os.path.join(os.path.abspath(args.processed_dir), filename)


def save_processed_subject(args, subject_id, x_data, y_data):
    save_path = processed_subject_path(args, subject_id)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        X=x_data.astype(np.float32),
        y=y_data.astype(np.int64),
        subject_id=subject_id,
        label_type=getattr(args, "label_type", "VALENCE"),
        window_size=args.window_size,
        stride_size=getattr(args, "stride_size", args.window_size),
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
    if label_type not in LABEL_COLUMNS:
        raise ValueError(f"Unknown CASE label_type: {label_type}. Choose AROUSAL or VALENCE.")

    annotation_path, physiological_path = _raw_paths(args, subject_id)
    _validate_raw_paths(annotation_path, physiological_path)

    annotation_df = pd.read_csv(annotation_path, sep="\t", header=None)
    annotation_df.columns = ["time", "axis1", "axis2"]
    physiological_df = pd.read_csv(physiological_path, sep="\t", header=None)
    physiological_df.columns = ["time", *SIGNAL_COLUMNS]

    annotation_df, physiological_df = _select_active_segments(annotation_df, physiological_df)
    label_values = annotation_df[LABEL_COLUMNS[label_type]].to_numpy(dtype=np.float32)
    personalized_threshold = float(np.mean(label_values))

    signals = []
    for column in SIGNAL_COLUMNS:
        signals.append(
            _filter_resample_normalize(
                physiological_df[column].to_numpy(dtype=np.float32),
                original_rate=1000,
                target_rate=args.sampling_rate,
            )
        )

    min_length = min(len(signal) for signal in signals)
    signals = [signal[:min_length] for signal in signals]
    label_sampling_rate = getattr(args, "label_sampling_rate", 20)
    window_size = args.window_size
    stride_size = getattr(args, "stride_size", window_size)
    label_window_size = int(args.window_seconds * label_sampling_rate)

    x_data = []
    y_data = []
    for start in range(0, min_length - window_size, stride_size):
        label_start = int((start * label_sampling_rate) // args.sampling_rate)
        label_end = label_start + label_window_size
        label_window = label_values[label_start:label_end]
        if len(label_window) == 0:
            continue
        window = np.stack([signal[start : start + window_size] for signal in signals], axis=-1)
        x_data.append(window)
        y_data.append(1 if float(np.mean(label_window)) > personalized_threshold else 0)

    if not x_data:
        raise ValueError(f"No CASE windows created for subject {subject_id}")
    return {"X": np.asarray(x_data, dtype=np.float32), "y": np.asarray(y_data, dtype=np.int64)}


def load_or_preprocess_subject(args, subject_id):
    if getattr(args, "use_processed", True):
        cached = load_processed_subject(args, subject_id)
        if cached is not None:
            print(f"Loading processed CASE data for: {subject_id}")
            return cached

    print(f"Preprocessing CASE raw data for: {subject_id}")
    subject_data = preprocess_subject(args, subject_id)
    if getattr(args, "use_processed", True):
        save_path = save_processed_subject(args, subject_id, subject_data["X"], subject_data["y"])
        print(f"Processed CASE data saved to: {save_path}")
    return subject_data


def load_data_per_subject(args):
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"CASE dataset directory not found: {dataset_dir}")
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
        raise NotImplementedError("CASE class-balanced target loader is not implemented yet.")
    return make_loader(x_data, y_data, batch_size, shuffle_data=shuffle_data, num_workers=num_workers)


def get_loso_loaders(args, subjects_data=None):
    if subjects_data is None:
        subjects_data = load_data_per_subject(args)
    subject_ids = list(subjects_data.keys())
    if args.target_domain not in subject_ids:
        raise ValueError(f"target_domain {args.target_domain} not found in loaded CASE subjects: {subject_ids}")
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
