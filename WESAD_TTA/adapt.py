import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import yaml

from config import parse_args
from metrics import evaluate_model, format_confusion_matrix
from TTA.setup import get_adaptation
from utils import get_dataset, get_device, get_model, set_seed


def checkpoint_path(args):
    return os.path.join(
        args.resume,
        args.dataset,
        args.model,
        args.target_domain,
        f"{args.dataset}_{args.target_domain}_checkpoint.pt",
    )


def make_output_dir(args):
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_path, args.dataset, args.adaption, args.target_domain, current_time)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(vars(args), handle, default_flow_style=False)
    return out_dir


def main():
    args = parse_args("Evaluate WESAD source or test-time adaptation.")
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = make_output_dir(args)

    _, target_loader, train_subjects = get_dataset(args)
    base_model = get_model(args).to(device)
    ckpt_path = checkpoint_path(args)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    criterion = nn.BCEWithLogitsLoss()
    source_metrics = evaluate_model(base_model, target_loader, device, criterion=criterion)

    model = get_adaptation(args, base_model)
    adapt_metrics = evaluate_model(model, target_loader, device, criterion=criterion, adapt=args.adaption != "source")

    record = {
        "target_domain": args.target_domain,
        "train_subjects": train_subjects,
        "adaption": args.adaption,
        "source_accuracy": source_metrics["accuracy"],
        "source_f1_non_stress": source_metrics["f1_non_stress"],
        "source_f1_stress": source_metrics["f1_stress"],
        "source_mean_f1": source_metrics["mean_f1"],
        "adapt_accuracy": adapt_metrics["accuracy"],
        "adapt_f1_non_stress": adapt_metrics["f1_non_stress"],
        "adapt_f1_stress": adapt_metrics["f1_stress"],
        "adapt_mean_f1": adapt_metrics["mean_f1"],
    }

    print(
        f"Source Accuracy: {source_metrics['accuracy']:.4f}, "
        f"Source Mean F1: {source_metrics['mean_f1']:.4f}, "
        f"Adapt Accuracy: {adapt_metrics['accuracy']:.4f}, "
        f"Adapt Mean F1: {adapt_metrics['mean_f1']:.4f}"
    )
    print("Source")
    print(format_confusion_matrix(source_metrics["confusion_matrix"]))
    print("Adapt")
    print(format_confusion_matrix(adapt_metrics["confusion_matrix"]))

    with open(os.path.join(out_dir, "log.txt"), "w", encoding="utf-8") as handle:
        for key, value in record.items():
            handle.write(f"{key}: {value}\n")

    pd.DataFrame([record]).to_csv(os.path.join(out_dir, "result.csv"), index=False)


if __name__ == "__main__":
    main()
