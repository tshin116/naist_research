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
from metrics import compute_pos_weight, evaluate_model, format_confusion_matrix
from utils import get_device, get_model, set_seed


def fit_model(model, train_loader, val_loader, args, device, pos_weight=None, patience=None):
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
            predictions = (torch.sigmoid(logits) > 0.5).float()
            total_correct += int((predictions == y_batch).sum().item())
            total_samples += batch_size

        history["loss"].append(total_loss / max(total_samples, 1))
        history["accuracy"].append(total_correct / max(total_samples, 1))

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
        pos_weight=compute_pos_weight(y_train),
        patience=args.early_stopping_patience,
    )
    val_metrics = evaluate_model(model, val_loader, device, criterion=criterion)
    return {
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


def run_inner_loso_cv(subjects_data, inner_subjects, args, device):
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
            f"Val F1(0): {fold_result['val_f1_non_stress']:.4f}, "
            f"Val F1(1): {fold_result['val_f1_stress']:.4f}, "
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
        "avg_val_f1": float(np.mean([result["val_f1"] for result in inner_results])),
        "avg_val_f1_non_stress": float(np.mean([result["val_f1_non_stress"] for result in inner_results])),
        "avg_val_f1_stress": float(np.mean([result["val_f1_stress"] for result in inner_results])),
        "avg_val_mean_f1": float(np.mean([result["val_mean_f1"] for result in inner_results])),
        "folds": inner_results,
    }


def train_final_model(subjects_data, train_subjects, selected_epochs, args, device):
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
        pos_weight=compute_pos_weight(y_train),
        patience=None,
    )
    return model, history, criterion


def checkpoint_path(args, test_subject):
    return os.path.join(args.resume, args.dataset, args.model, test_subject, f"{args.dataset}_{test_subject}_checkpoint.pt")


def save_loso_checkpoint(args, model, test_subject, train_subjects, selected_epochs, test_metrics, inner_summary):
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
                "f1_non_stress": test_metrics["f1_non_stress"],
                "f1_stress": test_metrics["f1_stress"],
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
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_path, args.dataset, "train_loso", current_time)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, default_flow_style=False)
    return out_dir


def main():
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
            f"F1(0): {test_metrics['f1_non_stress']:.4f}, "
            f"F1(1): {test_metrics['f1_stress']:.4f}, "
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
