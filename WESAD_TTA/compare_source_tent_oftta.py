"""Source、Tent、OFTTA を被験者ごとに比較するスクリプト。

`adapt.py` を手法ごと・被験者ごとに起動すると、結果の集計が手作業になる。
このスクリプトでは 1 プロセス内で各被験者の checkpoint を読み込み、Source、
Tent、OFTTA を評価して、横並びの CSV と Markdown 表を保存する。
"""

import argparse
import os
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from config import load_yaml
from data_processing.wesad import discover_subjects
from metrics import evaluate_model, make_criterion, move_criterion_to_device
from TTA.setup import get_adaptation
from utils import get_device, get_model, get_target_dataset, set_seed


METHODS = ["source", "tent", "oftta"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Source, Tent, and OFTTA on WESAD LOSO checkpoints.")
    parser.add_argument("--default_cfg", type=str, default="./cfg/default.yaml")
    parser.add_argument("--dataset_cfg", type=str, default="./cfg/dataset/wesad.yaml")
    parser.add_argument("--out_path", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_args(cli_args, method, subject):
    """共通設定、データセット設定、手法設定を統合する。"""
    cfg = {}
    cfg.update(load_yaml(cli_args.default_cfg))
    cfg.update(load_yaml(cli_args.dataset_cfg))
    cfg.update(load_yaml(f"./cfg/algorithm/{method}.yaml"))
    cfg["target_domain"] = subject
    cfg["out_path"] = cli_args.out_path
    if cli_args.resume is not None:
        cfg["resume"] = cli_args.resume
    if cli_args.device is not None:
        cfg["device"] = cli_args.device
    if cli_args.seed is not None:
        cfg["seed"] = cli_args.seed
    return SimpleNamespace(**cfg)


def checkpoint_path(args):
    return os.path.join(
        args.resume,
        args.dataset,
        args.model,
        args.target_domain,
        f"{args.dataset}_{args.target_domain}_checkpoint.pt",
    )


def load_checkpoint_model(args, device):
    """指定被験者の LOSO checkpoint を読み込んだモデルを返す。"""
    model = get_model(args).to(device)
    path = checkpoint_path(args)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model


def evaluate_method(args, target_loader, device, criterion):
    """1 手法を評価し、表に入れる指標だけを返す。"""
    model = load_checkpoint_model(args, device)
    if args.adaption == "source":
        metrics = evaluate_model(
            model,
            target_loader,
            device,
            criterion=criterion,
            num_classes=getattr(args, "num_classes", None),
        )
    else:
        adapted_model = get_adaptation(args, model)
        metrics = evaluate_model(
            adapted_model,
            target_loader,
            device,
            criterion=criterion,
            adapt=True,
            num_classes=getattr(args, "num_classes", None),
        )
    return {
        "accuracy": metrics["accuracy"],
        "f1_non_stress": metrics["f1_non_stress"],
        "f1_stress": metrics["f1_stress"],
        "f1_class_2": metrics.get("f1_class_2", None),
        "mean_f1": metrics["mean_f1"],
    }


def resolve_subjects(cli_args):
    """CLI 指定があればそれを使い、なければ WESAD ディレクトリから被験者を列挙する。"""
    if cli_args.subjects:
        return cli_args.subjects
    dataset_cfg = load_yaml(cli_args.dataset_cfg)
    subjects = dataset_cfg.get("subjects")
    if subjects:
        return subjects
    dataset_dir = os.path.abspath(dataset_cfg["dataset_dir"])
    return sorted(discover_subjects(dataset_dir), key=lambda subject: int(subject[1:]))


def format_float(value):
    return f"{value:.4f}"


def dataframe_to_markdown(df):
    """追加依存なしで DataFrame を Markdown table に変換する。"""
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + row_lines)


def save_metric_barplot(df, metric_suffix, ylabel, title, output_path):
    """被験者ごとの Source/Tent/OFTTA 比較棒グラフを保存する。"""
    subject_df = df[~df["Subject"].isin(["Average", "Std"])].copy()
    subjects = subject_df["Subject"].tolist()
    x = np.arange(len(subjects))
    width = 0.26
    series = [
        ("Source", f"Source_{metric_suffix}", "#4C78A8"),
        ("Tent", f"Tent_{metric_suffix}", "#F58518"),
        ("OFTTA", f"OFTTA_{metric_suffix}", "#54A24B"),
    ]

    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.7), 5.5))
    for idx, (label, column, color) in enumerate(series):
        ax.bar(x + (idx - 1) * width, subject_df[column].to_numpy(), width, label=label, color=color)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Subject")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_average_barplot(df, output_path):
    """Average 行だけを使い、3 手法の主要指標を保存する。"""
    avg = df[df["Subject"] == "Average"].iloc[0]
    metrics = [
        ("Accuracy", "Acc"),
        ("Mean F1", "MeanF1"),
        ("F1 Stress", "F1_1"),
    ]
    if "Source_F1_2" in df.columns:
        metrics.append(("F1 Amusement", "F1_2"))
    methods = ["Source", "Tent", "OFTTA"]
    x = np.arange(len(metrics))
    width = 0.26
    colors = {"Source": "#4C78A8", "Tent": "#F58518", "OFTTA": "#54A24B"}

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for idx, method in enumerate(methods):
        values = [avg[f"{method}_{suffix}"] for _, suffix in metrics]
        ax.bar(x + (idx - 1) * width, values, width, label=method, color=colors[method])

    ax.set_title("Average Performance Across Subjects")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in metrics])
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_output_dir(args):
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_path, "wesad", "compare_source_tent_oftta", current_time)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    cli_args = parse_args()
    base_args = build_args(cli_args, "source", "S2")
    set_seed(base_args.seed)
    device = get_device(base_args.device)
    criterion = move_criterion_to_device(make_criterion(base_args), device)
    subjects = resolve_subjects(cli_args)
    out_dir = make_output_dir(base_args)

    rows = []
    for subject in subjects:
        print(f"=== {subject} ===")
        target_args = build_args(cli_args, "source", subject)
        target_loader = get_target_dataset(target_args)
        row = {"Subject": subject}

        for method in METHODS:
            method_args = build_args(cli_args, method, subject)
            # target_shuffle=True の比較では DataLoader の反復ごとに順序が変わる。
            # 各手法が同じ shuffled batch 列を見るよう、評価直前に seed を戻す。
            set_seed(method_args.seed)
            metrics = evaluate_method(method_args, target_loader, device, criterion)
            prefix = method.capitalize() if method != "oftta" else "OFTTA"
            row[f"{prefix}_Acc"] = metrics["accuracy"]
            row[f"{prefix}_F1_0"] = metrics["f1_non_stress"]
            row[f"{prefix}_F1_1"] = metrics["f1_stress"]
            if metrics["f1_class_2"] is not None:
                row[f"{prefix}_F1_2"] = metrics["f1_class_2"]
            row[f"{prefix}_MeanF1"] = metrics["mean_f1"]
            print(
                f"  {prefix}: Acc={metrics['accuracy']:.4f}, "
                f"F1(0)={metrics['f1_non_stress']:.4f}, "
                f"F1(1)={metrics['f1_stress']:.4f}, "
                f"F1(2)={(metrics['f1_class_2'] or 0.0):.4f}, "
                f"MeanF1={metrics['mean_f1']:.4f}"
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    numeric_cols = [column for column in df.columns if column != "Subject"]
    avg_row = {"Subject": "Average"}
    std_row = {"Subject": "Std"}
    for column in numeric_cols:
        avg_row[column] = float(df[column].mean())
        std_row[column] = float(df[column].std(ddof=0))
    summary_df = pd.concat([df, pd.DataFrame([avg_row, std_row])], ignore_index=True)

    csv_path = os.path.join(out_dir, "source_tent_oftta_comparison.csv")
    md_path = os.path.join(out_dir, "source_tent_oftta_comparison.md")
    mean_f1_plot_path = os.path.join(out_dir, "source_tent_oftta_mean_f1.png")
    stress_f1_plot_path = os.path.join(out_dir, "source_tent_oftta_stress_f1.png")
    amusement_f1_plot_path = os.path.join(out_dir, "source_tent_oftta_amusement_f1.png")
    accuracy_plot_path = os.path.join(out_dir, "source_tent_oftta_accuracy.png")
    average_plot_path = os.path.join(out_dir, "source_tent_oftta_average.png")
    summary_df.to_csv(csv_path, index=False)

    display_df = summary_df.copy()
    for column in numeric_cols:
        display_df[column] = display_df[column].map(format_float)
    markdown = dataframe_to_markdown(display_df)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Source / Tent / OFTTA Comparison\n\n")
        handle.write(markdown)
        handle.write("\n")

    save_metric_barplot(
        summary_df,
        "MeanF1",
        "Mean F1",
        "Source vs Tent vs OFTTA: Mean F1 by Subject",
        mean_f1_plot_path,
    )
    save_metric_barplot(
        summary_df,
        "F1_1",
        "F1 Stress",
        "Source vs Tent vs OFTTA: Stress-class F1 by Subject",
        stress_f1_plot_path,
    )
    if "Source_F1_2" in summary_df.columns:
        save_metric_barplot(
            summary_df,
            "F1_2",
            "F1 Amusement",
            "Source vs Tent vs OFTTA: Amusement-class F1 by Subject",
            amusement_f1_plot_path,
        )
    save_metric_barplot(
        summary_df,
        "Acc",
        "Accuracy",
        "Source vs Tent vs OFTTA: Accuracy by Subject",
        accuracy_plot_path,
    )
    save_average_barplot(summary_df, average_plot_path)

    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "subjects": subjects,
                "methods": METHODS,
                "default_cfg": cli_args.default_cfg,
                "dataset_cfg": cli_args.dataset_cfg,
                "resume": base_args.resume,
            },
            handle,
            default_flow_style=False,
        )

    print(f"\nCSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")
    print(f"Mean F1 plot saved to: {mean_f1_plot_path}")
    print(f"Stress F1 plot saved to: {stress_f1_plot_path}")
    if "Source_F1_2" in summary_df.columns:
        print(f"Amusement F1 plot saved to: {amusement_f1_plot_path}")
    print(f"Accuracy plot saved to: {accuracy_plot_path}")
    print(f"Average plot saved to: {average_plot_path}")


if __name__ == "__main__":
    main()
