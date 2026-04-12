import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score


def compute_pos_weight(y_data):
    positive_count = int(np.sum(y_data == 1))
    negative_count = int(np.sum(y_data == 0))
    if positive_count == 0:
        return None
    return torch.tensor([negative_count / positive_count], dtype=torch.float32)


def evaluate_model(model, data_loader, device, criterion=None, adapt=False):
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if not adapt:
        model.eval()

    total_loss = 0.0
    total_samples = 0
    probabilities = []
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

            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(y_batch.detach().cpu().numpy())

    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probabilities)
    y_pred = (y_prob > 0.5).astype(int)
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(np.mean(y_pred == y_true)),
        "f1": float(f1_per_class[1]),
        "f1_non_stress": float(f1_per_class[0]),
        "f1_stress": float(f1_per_class[1]),
        "mean_f1": float(np.mean(f1_per_class)),
        "confusion_matrix": conf_matrix,
        "y_true": y_true.astype(int),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def format_confusion_matrix(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return (
        "Confusion Matrix (rows=true, cols=pred)\n"
        "              Pred 0   Pred 1\n"
        f"True 0        {tn:6d}   {fp:6d}\n"
        f"True 1        {fn:6d}   {tp:6d}"
    )
