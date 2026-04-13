"""内側 LOSO を使わず、固定 epoch で WESAD 1D-CNN を学習する。

このスクリプトは `train.py` の nested LOSO 版とは別に、外側 LOSO だけを行う。
各 fold では target 被験者を完全に除外し、残り全員で `max_epochs` だけ固定学習する。
検証被験者を置かないため、epoch 数は `cfg/algorithm/source_fixed.yaml` の値をそのまま使う。
"""

import os
from datetime import datetime

import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from config import parse_args
from data_processing.wesad import load_data_per_subject, make_loader, stack_subjects
from metrics import compute_pos_weight, evaluate_model, format_confusion_matrix
from train import fit_model
from utils import get_device, get_model, set_seed


def checkpoint_path(args, test_subject):
    """固定 epoch 学習用 checkpoint の保存パスを返す。"""
    return os.path.join(args.resume, args.dataset, args.model, test_subject, f"{args.dataset}_{test_subject}_checkpoint.pt")


def make_output_dir(args):
    """ログ出力用の日時付きディレクトリを作成し、実行設定を保存する。"""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_path, args.dataset, "train_fixed_loso", current_time)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, default_flow_style=False)
    return out_dir


def train_fixed_model(subjects_data, train_subjects, args, device):
    """指定された train subjects 全体で固定 epoch 学習する。"""
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_train, y_train = shuffle(x_train, y_train, random_state=args.seed)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle_data=True, num_workers=args.num_workers)

    model = get_model(args).to(device)
    history, _, criterion = fit_model(
        model,
        train_loader,
        val_loader=None,
        args=args,
        device=device,
        pos_weight=compute_pos_weight(y_train),
        patience=None,
    )
    return model, history, criterion


def save_fixed_checkpoint(args, model, test_subject, train_subjects, test_metrics):
    """固定 epoch 学習済みモデルとメタ情報を保存する。"""
    save_path = checkpoint_path(args, test_subject)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "selected_epochs": args.max_epochs,
            "training_mode": "fixed_epoch_no_inner_loso",
            "window_seconds": args.window_seconds,
            "sampling_rate": args.sampling_rate,
            "window_size": args.window_size,
            "num_channels": args.num_channels,
            "metrics": {
                "accuracy": test_metrics["accuracy"],
                "f1_non_stress": test_metrics["f1_non_stress"],
                "f1_stress": test_metrics["f1_stress"],
                "mean_f1": test_metrics["mean_f1"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            },
            "config": vars(args),
        },
        save_path,
    )
    return save_path


def main():
    """設定読み込みから固定 epoch LOSO 学習、結果保存までを実行する。"""
    args = parse_args("Train WESAD 1D-CNN source models with fixed-epoch outer LOSO.")
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = make_output_dir(args)

    print(f"Using device: {device}")
    print(f"Training mode: fixed_epoch_no_inner_loso, epochs={args.max_epochs}")
    subjects_data = load_data_per_subject(args)
    if not subjects_data:
        raise SystemExit("No data loaded.")

    subject_ids = list(subjects_data.keys())
    print(f"Loaded {len(subject_ids)} subjects: {subject_ids}")
    loso_results = []

    for test_subject in subject_ids:
        print(f"\n=== Fixed LOSO Iteration: Testing on {test_subject} ===")
        train_subjects = [subject_id for subject_id in subject_ids if subject_id != test_subject]
        print(f"Train subjects: {train_subjects}")
        print(f"Test subject: {test_subject}")

        model, history, criterion = train_fixed_model(subjects_data, train_subjects, args, device)
        x_test = subjects_data[test_subject]["X"]
        y_test = subjects_data[test_subject]["y"]
        test_loader = make_loader(x_test, y_test, args.batch_size, shuffle_data=False, num_workers=args.num_workers)
        test_metrics = evaluate_model(model, test_loader, device, criterion=criterion)
        report = classification_report(test_metrics["y_true"], test_metrics["y_pred"], output_dict=True, zero_division=0)

        print(
            f"Subject {test_subject} - "
            f"Accuracy: {test_metrics['accuracy']:.4f}, "
            f"F1(0): {test_metrics['f1_non_stress']:.4f}, "
            f"F1(1): {test_metrics['f1_stress']:.4f}, "
            f"Mean F1: {test_metrics['mean_f1']:.4f}"
        )
        print(format_confusion_matrix(test_metrics["confusion_matrix"]))

        saved_path = save_fixed_checkpoint(args, model, test_subject, train_subjects, test_metrics)
        print(f"Checkpoint saved to {saved_path}")

        loso_results.append(
            {
                "Subject": test_subject,
                "Accuracy": test_metrics["accuracy"],
                "F1_Non_Stress_0": test_metrics["f1_non_stress"],
                "F1_Stress_1": test_metrics["f1_stress"],
                "Mean_F1": test_metrics["mean_f1"],
                "TN": int(test_metrics["confusion_matrix"][0, 0]),
                "FP": int(test_metrics["confusion_matrix"][0, 1]),
                "FN": int(test_metrics["confusion_matrix"][1, 0]),
                "TP": int(test_metrics["confusion_matrix"][1, 1]),
                "Fixed_Epochs": args.max_epochs,
                "Checkpoint": saved_path,
                "Report": report,
                "History": history,
            }
        )

    csv_path = os.path.join(out_dir, "loso_fixed_epoch_results.csv")
    pd.DataFrame(loso_results).to_csv(csv_path, index=False)
    print(f"Fixed-epoch LOSO results saved to {csv_path}")


if __name__ == "__main__":
    main()
