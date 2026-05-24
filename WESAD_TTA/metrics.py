"""WESAD 分類実験で使う損失関数と評価指標をまとめたモジュール。

二値分類では既存実験と互換性を保つため single-logit の `BCEWithLogitsLoss` を使う。
3分類では multi-class logits に対して `CrossEntropyLoss` を使い、各クラス F1 と
macro F1 を記録する。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score


CLASS_NAMES = {
    "binary": ["Non-Stress", "Stress"],
    "3class": ["Baseline", "Stress", "Amusement"],
}


def get_label_mode(args=None):
    """設定から分類モードを取り出す。未指定なら従来通り二値分類として扱う。"""
    if args is None:
        return "binary"
    return getattr(args, "label_mode", "binary")


def compute_pos_weight(y_data):
    """二値分類の `BCEWithLogitsLoss` に渡す正例クラス重みを計算する。"""
    positive_count = int(np.sum(y_data == 1))
    negative_count = int(np.sum(y_data == 0))
    if positive_count == 0:
        return None
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)


def compute_class_weights(y_data, num_classes):
    """多クラス `CrossEntropyLoss` に渡すクラス重みを計算する。

    各クラスの出現数に反比例する重みを使う。存在しないクラスには 0 を置き、
    その fold で学習データにないクラスを無理に学習対象にしない。
    """
    counts = np.bincount(y_data.astype(int), minlength=num_classes).astype(np.float32)
    total = float(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = total / (num_classes * counts[present])
    return torch.tensor(weights, dtype=torch.float32)


def make_criterion(args, y_train=None):
    """分類モードに合った loss 関数を作成する。"""
    label_mode = get_label_mode(args)
    if label_mode == "binary":
        pos_weight = compute_pos_weight(y_train) if y_train is not None else None
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    class_weights = None
    if y_train is not None:
        class_weights = compute_class_weights(y_train, getattr(args, "num_classes", 3))
    return nn.CrossEntropyLoss(weight=class_weights)


def move_criterion_to_device(criterion, device):
    """loss 関数内の class weight を device に移す。"""
    return criterion.to(device)


def _is_binary_logits(logits):
    """logits の形状から single-logit 二値分類かどうかを判定する。"""
    return logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1)


def _prediction_arrays(logits):
    """logits から評価用の確率と予測ラベルを作る。"""
    if _is_binary_logits(logits):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        return probs, preds.reshape(-1)

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1).astype(int)
    return probs, preds


def _label_indices(y_true, y_pred, num_classes=None):
    """混同行列と per-class F1 に使うラベル一覧を返す。"""
    if num_classes is not None:
        return list(range(num_classes))
    max_label = int(max(np.max(y_true), np.max(y_pred))) if len(y_true) else 0
    return list(range(max_label + 1))


def evaluate_model(model, data_loader, device, criterion=None, adapt=False, num_classes=None):
    """モデルを評価し、loss と分類指標を辞書で返す。

    `adapt=False` の通常評価では `torch.no_grad()` を使う。Tent 評価では
    forward 中に backward が必要なため、`adapt=True` として勾配計算を有効にする。
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if not adapt:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    probabilities = []
    predictions = []
    labels = []

    grad_context = torch.enable_grad() if adapt else torch.no_grad()
    with grad_context:
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            if hasattr(model, "record_batch_labels"):
                model.record_batch_labels(y_batch)
            loss = criterion(logits, y_batch)
            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            batch_prob, batch_pred = _prediction_arrays(logits)
            probabilities.append(batch_prob)
            predictions.append(batch_pred)
            labels.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(labels).astype(int)
    y_pred = np.concatenate(predictions).astype(int)
    y_prob = np.concatenate(probabilities)
    labels_for_metrics = _label_indices(y_true, y_pred, num_classes=num_classes)
    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=labels_for_metrics,
        average=None,
        zero_division=0,
    )
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels_for_metrics)

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(np.mean(y_pred == y_true)),
        "f1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "mean_f1": float(np.mean(f1_per_class)),
        "macro_f1": float(np.mean(f1_per_class)),
        "f1_per_class": f1_per_class,
        "confusion_matrix": conf_matrix,
        "labels": labels_for_metrics,
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    for class_index, class_f1 in zip(labels_for_metrics, f1_per_class):
        metrics[f"f1_class_{class_index}"] = float(class_f1)

    # 従来の二値分類ログとの互換性を保つ。
    metrics["f1_non_stress"] = metrics.get("f1_class_0", 0.0)
    metrics["f1_stress"] = metrics.get("f1_class_1", 0.0)
    return metrics


def format_confusion_matrix(conf_matrix, class_names=None):
    """混同行列をログで読みやすい文字列に整形する。"""
    num_classes = conf_matrix.shape[0]
    if class_names is None:
        class_names = [f"Class {idx}" for idx in range(num_classes)]

    header = " " * 14 + "".join(f"Pred {idx:>6}" for idx in range(num_classes))
    rows = ["Confusion Matrix (rows=true, cols=pred)", header]
    for idx, row in enumerate(conf_matrix):
        name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
        rows.append(f"True {idx} {name[:7]:>7} " + "".join(f"{int(value):8d}" for value in row))
    return "\n".join(rows)
