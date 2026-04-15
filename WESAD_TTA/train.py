"""WESAD の 1D-CNN を LOSO で学習するメインプログラム。

外側 LOSO では 1 名の被験者を完全なテスト対象として残し、残りの被験者で
学習する。さらに内側 LOSO を使って最終学習に使う epoch 数を決めることで、
テスト被験者の情報を使わずに early stopping 相当の判断を行う。
"""

import copy
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from config import parse_args
from data_processing.wesad import load_data_per_subject, make_loader, stack_subjects
from metrics import evaluate_model, format_confusion_matrix
from utils import get_device, get_model, set_seed


def fit_model(model, train_loader, val_loader, args, device, patience=None):
    """1 つのモデルを指定 epoch 数だけ学習する。

    `val_loader` がある場合は検証 loss が最も小さい重みを保持する。
    `patience` が指定されていれば、検証 loss の改善が止まった時点で早期終了する。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"loss": [], "accuracy": []}
    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []

    best_state = None
    best_val_loss = float("inf")
    best_epoch = args.max_epochs
    epochs_without_improvement = 0

    for epoch in range(args.max_epochs):
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
            predictions = logits.argmax(dim=1)
            total_correct += int((predictions == y_batch).sum().item())
            total_samples += batch_size

        history["loss"].append(total_loss / max(total_samples, 1))
        history["accuracy"].append(total_correct / max(total_samples, 1))

        # 最終学習では検証被験者を置かないため、ここで epoch を継続する。
        if val_loader is None:
            continue

        val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)
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


def train_with_subject_validation(subjects_data, train_subjects, val_subject, args, device):
    """内側 LOSO の 1 fold を学習し、検証被験者の性能を返す。"""
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_val = subjects_data[val_subject]["X"]
    y_val = subjects_data[val_subject]["y"]

    x_train, y_train = shuffle(x_train, y_train, random_state=args.seed)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle_data=True, num_workers=args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle_data=False, num_workers=args.num_workers)

    model = get_model(args).to(device)
    history, best_epoch, criterion = fit_model(
        model,
        train_loader,
        val_loader,
        args,
        device,
        patience=args.early_stopping_patience,
    )
    val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)
    return {
        "val_subject": val_subject,
        "train_subjects": train_subjects,
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_f1_neutral": val_metrics["f1_neutral"],
        "val_f1_stress": val_metrics["f1_stress"],
        "val_f1_amusement": val_metrics["f1_amusement"],
        "val_mean_f1": val_metrics["mean_f1"],
        "best_epoch": int(best_epoch),
        "history": history,
    }


def run_inner_loso_cv(subjects_data, inner_subjects, args, device):
    """外側テスト被験者を除いた集合で内側 LOSO を回す。

    各 fold で得られた best epoch の平均を、外側 fold の最終学習 epoch として使う。
    """
    inner_results = []
    for val_subject in inner_subjects:
        train_subjects = [subj for subj in inner_subjects if subj != val_subject]
        print(f"  [Inner CV] Train on: {train_subjects}")
        print(f"  [Inner CV] Validate on: {val_subject}")
        fold_result = train_with_subject_validation(subjects_data, train_subjects, val_subject, args, device)
        inner_results.append(fold_result)
        print(
            f"  [Inner CV] {val_subject} - "
            f"Val Loss: {fold_result['val_loss']:.4f}, "
            f"Val Acc: {fold_result['val_accuracy']:.4f}, "
            f"Val F1(0): {fold_result['val_f1_neutral']:.4f}, "
            f"Val F1(1): {fold_result['val_f1_stress']:.4f}, "
            f"Val F1(2): {fold_result['val_f1_amusement']:.4f}, "
            f"Val Mean F1: {fold_result['val_mean_f1']:.4f}, "
            f"Best Epoch: {fold_result['best_epoch']}"
        )

    selected_epochs = int(
        np.clip(
            round(np.mean([result["best_epoch"] for result in inner_results])),
            1,
            args.max_epochs,
        )
    )
    return {
        "selected_epochs": selected_epochs,
        "avg_val_loss": float(np.mean([result["val_loss"] for result in inner_results])),
        "avg_val_accuracy": float(np.mean([result["val_accuracy"] for result in inner_results])),
        "avg_val_f1_neutral": float(np.mean([result["val_f1_neutral"] for result in inner_results])),
        "avg_val_f1_stress": float(np.mean([result["val_f1_stress"] for result in inner_results])),
        "avg_val_f1_amusement": float(np.mean([result["val_f1_amusement"] for result in inner_results])),
        "avg_val_mean_f1": float(np.mean([result["val_mean_f1"] for result in inner_results])),
        "folds": inner_results,
    }


def train_final_model(subjects_data, train_subjects, selected_epochs, args, device):
    """内側 LOSO で決めた epoch 数を使い、訓練被験者全体で最終学習する。"""
    x_train, y_train = stack_subjects(subjects_data, train_subjects)
    x_train, y_train = shuffle(x_train, y_train, random_state=args.seed)
    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle_data=True, num_workers=args.num_workers)

    model = get_model(args).to(device)
    run_args = copy.copy(args)
    run_args.max_epochs = selected_epochs
    history, _, criterion = fit_model(
        model,
        train_loader,
        val_loader=None,
        args=run_args,
        device=device,
        patience=None,
    )
    return model, history, criterion


def checkpoint_path(args, test_subject):
    """外側 LOSO fold のチェックポイント保存パスを返す。"""
    return os.path.join(args.resume, args.dataset, args.model, test_subject, f"{args.dataset}_{test_subject}_checkpoint.pt")


def save_loso_checkpoint(args, model, test_subject, train_subjects, selected_epochs, test_metrics, inner_summary):
    """学習済みモデルと fold のメタ情報を保存する。

    `adapt.py` で再利用できるよう、`model_state_dict` だけでなく、テスト被験者、
    学習被験者、評価指標、内側 LOSO の概要も同じファイルに入れる。
    """
    save_path = checkpoint_path(args, test_subject)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "test_subject": test_subject,
            "train_subjects": train_subjects,
            "selected_epochs": selected_epochs,
            "window_seconds": args.window_seconds,
            "sampling_rate": args.sampling_rate,
            "window_size": args.window_size,
            "num_channels": args.num_channels,
            "metrics": {
                "accuracy": test_metrics["accuracy"],
                "f1_neutral": test_metrics["f1_neutral"],
                "f1_stress": test_metrics["f1_stress"],
                "f1_amusement": test_metrics["f1_amusement"],
                "mean_f1": test_metrics["mean_f1"],
                "confusion_matrix": test_metrics["confusion_matrix"],
            },
            "inner_cv_summary": inner_summary,
            "config": vars(args),
        },
        save_path,
    )
    return save_path


def make_output_dir(args):
    """ログ出力用の日時付きディレクトリを作成し、実行設定を保存する。"""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_path, args.dataset, "train_loso", current_time)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, default_flow_style=False)
    return out_dir


def main():
    """設定読み込みから LOSO 学習、結果保存までを実行する。"""
    args = parse_args("Train WESAD 1D-CNN source models with nested LOSO.")
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = make_output_dir(args)

    print(f"Using device: {device}")
    subjects_data = load_data_per_subject(args)
    if not subjects_data:
        raise SystemExit("No data loaded.")

    subject_ids = list(subjects_data.keys())
    print(f"Loaded {len(subject_ids)} subjects: {subject_ids}")
    loso_results = []

    for test_subject in subject_ids:
        print(f"\n=== LOSO Iteration: Testing on {test_subject} ===")
        train_subjects = [subject_id for subject_id in subject_ids if subject_id != test_subject]
        if len(train_subjects) < 2:
            raise SystemExit("Need at least 3 subjects for nested LOSO.")

        inner_summary = run_inner_loso_cv(subjects_data, train_subjects, args, device)
        selected_epochs = inner_summary["selected_epochs"]
        print(f"Training final model for {test_subject} with {selected_epochs} epochs.")

        model, history, criterion = train_final_model(subjects_data, train_subjects, selected_epochs, args, device)
        x_test = subjects_data[test_subject]["X"]
        y_test = subjects_data[test_subject]["y"]
        test_loader = make_loader(x_test, y_test, args.batch_size, shuffle_data=False, num_workers=args.num_workers)
        test_metrics = evaluate_model(model, test_loader, device, criterion=criterion)
        report = classification_report(test_metrics["y_true"], test_metrics["y_pred"], output_dict=True, zero_division=0)

        print(
            f"Subject {test_subject} - "
            f"Accuracy: {test_metrics['accuracy']:.4f}, "
            f"F1(0): {test_metrics['f1_neutral']:.4f}, "
            f"F1(1): {test_metrics['f1_stress']:.4f}, "
            f"F1(2): {test_metrics['f1_amusement']:.4f}, "
            f"Mean F1: {test_metrics['mean_f1']:.4f}"
        )
        print(format_confusion_matrix(test_metrics["confusion_matrix"]))

        saved_path = save_loso_checkpoint(
            args,
            model,
            test_subject,
            train_subjects,
            selected_epochs,
            test_metrics,
            inner_summary,
        )
        print(f"Checkpoint saved to {saved_path}")

        cm = test_metrics["confusion_matrix"]
        loso_results.append(
            {
                "Subject": test_subject,
                "Accuracy": test_metrics["accuracy"],
                "F1_Neutral_0": test_metrics["f1_neutral"],
                "F1_Stress_1": test_metrics["f1_stress"],
                "F1_Amusement_2": test_metrics["f1_amusement"],
                "Mean_F1": test_metrics["mean_f1"],
                "Selected_Epochs": selected_epochs,
                "Inner_CV_Avg_Mean_F1": inner_summary["avg_val_mean_f1"],
                "Checkpoint": saved_path,
                "Report": report,
                "History": history,
            }
        )

    csv_path = os.path.join(out_dir, "loso_results.csv")
    pd.DataFrame(loso_results).to_csv(csv_path, index=False)
    print(f"LOSO results saved to {csv_path}")


if __name__ == "__main__":
    main()
