"""WESAD 3 クラス分類で使う評価指標をまとめたモジュール。

中性 (0)・ストレス (1)・楽しさ (2) の 3 クラス分類に対応し、
accuracy、クラスごとの F1、macro F1、混同行列を記録する。
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

CLASS_NAMES = ["neutral", "stress", "amusement"]


def evaluate_model(model, data_loader, device, criterion=None, adapt=False):
    """モデルを評価し、loss と分類指標を辞書で返す。

    `adapt=False` の通常評価では `torch.no_grad()` を使う。Tent 評価では
    forward 中に backward が必要なため、`adapt=True` として勾配計算を有効にする。
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if not adapt:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    labels = []

    grad_context = torch.enable_grad() if adapt else torch.no_grad()
    with grad_context:
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_logits.append(logits.detach().cpu().numpy())
            labels.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(labels)
    logits_np = np.concatenate(all_logits)
    y_pred = logits_np.argmax(axis=1)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(np.mean(y_pred == y_true)),
        "f1_neutral": float(f1_per_class[0]),
        "f1_stress": float(f1_per_class[1]),
        "f1_amusement": float(f1_per_class[2]),
        "mean_f1": float(np.mean(f1_per_class)),
        "confusion_matrix": conf_matrix,
        "y_true": y_true.astype(int),
        "y_pred": y_pred,
    }


def format_confusion_matrix(conf_matrix):
    """3 クラス混同行列をログで読みやすい文字列に整形する。"""
    lines = [
        "Confusion Matrix (rows=true, cols=pred)",
        "                 Pred 0   Pred 1   Pred 2",
    ]
    for i, name in enumerate(CLASS_NAMES):
        row = conf_matrix[i]
        lines.append(f"True {i} ({name:>9})  {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")
    return "\n".join(lines)
