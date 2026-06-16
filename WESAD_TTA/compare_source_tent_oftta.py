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
from metrics import evaluate_model, make_criterion, move_criterion_to_device
from TTA.setup import get_adaptation
from utils import get_data_module, get_device, get_model, get_target_dataset, set_seed


METHODS = ["source", "tent", "oftta"]
METHOD_LABELS = {
    "source": "Source",
    "tent": "Tent",
    "tent_bn_only": "TentBnOnly",
    "tent_lr1e3": "TentLR1e3",
    "tent_bn_only_lr1e3": "TentBnOnlyLR1e3",
    "ema_tent": "EMATent",
    "mi_dynamic_ema_tent": "MIDynamicEMATent",
    "mi_dynamic_ema_tent_relaxed": "MIDynamicRelaxed",
    "mi_dynamic_ema_tent_relaxed_anchor": "MIDynamicRelaxedAnchor",
    "mi_dynamic_ema_tent_fast_ema": "MIDynamicFastEMA",
    "mi_dynamic_ema_tent_fixed_ema_relaxed": "MIDynamicFixedEMARelaxed",
    "dynamix_ema_tent": "DynaMixEMATent",
    "realistic_tta": "RealisticTTA",
    "tema": "TEMA",
    "dua": "DUA",
    "note": "NOTE",
    "rotta": "RoTTA",
    "delta": "DELTA",
    "mi_ema_tent_mi_gate_only": "MIEMATentGateOnly",
    "mi_ema_tent_gate_aware": "MIEMATentGateAware",
    "mi_ema_tent_dynamic": "MIDynamicEMATent",
    "mi_ema_tent_mi_gate_only_relaxed": "MIGateOnlyRelaxed",
    "mi_ema_tent_gate_aware_relaxed": "MIGateAwareRelaxed",
    "ema_tent_probe025": "EMATentProbe025",
    "ema_tent_probe050": "EMATentProbe050",
    "ema_tent_probe100": "EMATentProbe100",
    "ema_tent_mingate025": "EMATentMinGate025",
    "ema_tent_mingate050": "EMATentMinGate050",
    "ema_tent_safe": "EMATentSafe",
    "ema_tent_adaptive": "EMATentAdaptive",
    "ema_tent_grid_s20_e50_b30_m080_lr1e3": "EmaTentGridS20E50B30M080LR1e3",
    "ema_tent_grid_s20_e50_b30_m080_bn_only": "EmaTentGridS20E50B30M080BnOnly",
    "ema_tent_grid_s20_e50_b30_m080_bn_only_lr1e3": "EmaTentGridS20E50B30M080BnOnlyLR1e3",
    "oftta": "OFTTA",
    "mem_oftta": "MemOFTTA",
    "norm": "Norm",
    "norm_bn_only": "NormBnOnly",
}
PLOT_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#B279A2",
    "#72B7B2",
    "#E45756",
    "#FF9DA6",
    "#9D755D",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Source, Tent, and OFTTA on WESAD LOSO checkpoints.")
    parser.add_argument("--default_cfg", type=str, default="./cfg/default.yaml")
    parser.add_argument("--dataset_cfg", type=str, default="./cfg/dataset/wesad.yaml")
    parser.add_argument("--out_path", type=str, default="./logs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--stream_mode", type=str, default=None, choices=["original", "shuffle", "class_block", "synthetic_block"])
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def method_label(method):
    return METHOD_LABELS.get(method, method.replace("_", " ").title().replace(" ", ""))


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
    if cli_args.batch_size is not None:
        cfg["batch_size"] = cli_args.batch_size
    if cli_args.stream_mode is not None:
        cfg["stream_mode"] = cli_args.stream_mode
        cfg["target_shuffle"] = cli_args.stream_mode == "shuffle"
    if cli_args.block_length is not None:
        cfg["block_length"] = cli_args.block_length
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
        if hasattr(adapted_model, "get_diagnostics"):
            metrics["_tta_diagnostics"] = adapted_model.get_diagnostics()
    return {
        "accuracy": metrics["accuracy"],
        "f1_non_stress": metrics["f1_non_stress"],
        "f1_stress": metrics["f1_stress"],
        "f1_class_2": metrics.get("f1_class_2", None),
        "mean_f1": metrics["mean_f1"],
        "_tta_diagnostics": metrics.get("_tta_diagnostics", []),
    }


def resolve_subjects(cli_args):
    """CLI 指定があればそれを使い、なければ WESAD ディレクトリから被験者を列挙する。"""
    if cli_args.subjects:
        return cli_args.subjects
    dataset_cfg = load_yaml(cli_args.dataset_cfg)
    subjects = dataset_cfg.get("subjects")
    if subjects:
        return subjects
    cfg = {}
    cfg.update(load_yaml(cli_args.default_cfg))
    cfg.update(dataset_cfg)
    cfg.update(load_yaml("./cfg/algorithm/source.yaml"))
    args = SimpleNamespace(**cfg)
    dataset_dir = os.path.abspath(dataset_cfg["dataset_dir"])
    return get_data_module(args).discover_subjects(dataset_dir)


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


def write_gate_summary(diagnostics_df, output_path):
    rows = []
    group_cols = ["Method"]
    if "Method" not in diagnostics_df.columns:
        return
    for method, group in diagnostics_df.groupby(group_cols):
        if isinstance(method, tuple):
            method = method[0]
        row = {"Method": method, "Batches": int(len(group))}
        for column in [
            "g_mi",
            "diversity_norm",
            "sample_entropy_norm",
            "w_source",
            "w_ema",
            "w_batch",
            "alpha",
            "rho",
            "n_eff",
            "entropy_updated",
            "loss_entropy",
            "pred_max_class_ratio",
            "pred_num_present_classes",
            "raw_gate",
            "effective_gate",
            "effective_batch_weight",
            "effective_source_weight",
            "effective_ema_weight",
            "true_max_class_ratio",
            "true_num_present_classes",
            "true_imbalance",
            "updated",
        ]:
            if column in group.columns:
                row[f"{column}_mean"] = float(group[column].mean())
                row[f"{column}_min"] = float(group[column].min())
                row[f"{column}_max"] = float(group[column].max())
        rows.append(row)

    summary = pd.DataFrame(rows)
    display = summary.copy()
    for column in display.columns:
        if column not in {"Method", "Batches"}:
            display[column] = display[column].map(format_float)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# EMA-Tent Gate Diagnostics\n\n")
        handle.write(dataframe_to_markdown(display))
        handle.write("\n")


def write_stream_summary(summary_df, diagnostics_df, output_path, methods, stream_mode, block_length):
    """stream mode 実験用に性能と batch 偏り指標を横断表へまとめる。"""
    if diagnostics_df is None or diagnostics_df.empty:
        return

    subject_rows = summary_df[~summary_df["Subject"].isin(["Average", "Std"])].copy()
    rows = []
    for _, perf_row in subject_rows.iterrows():
        subject = perf_row["Subject"]
        for method in methods:
            label = method_label(method)
            row = {
                "Subject": subject,
                "Method": label,
                "stream_mode": stream_mode,
                "block_length": block_length,
                "accuracy": perf_row.get(f"{label}_Acc"),
                "macro_f1": perf_row.get(f"{label}_MacroF1"),
            }
            diag = diagnostics_df[
                (diagnostics_df["Subject"] == subject)
                & (diagnostics_df["Method"] == label)
            ]
            if not diag.empty:
                if "true_max_class_ratio" in diag.columns:
                    row["single_class_batch_ratio"] = float((diag["true_max_class_ratio"] >= 0.999999).mean())
                    row["max_class_ratio"] = float(diag["true_max_class_ratio"].mean())
                if "true_imbalance" in diag.columns:
                    row["imbalance"] = float(diag["true_imbalance"].mean())
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_metric_barplot(df, metric_suffix, ylabel, title, output_path, methods):
    """被験者ごとの Source/Tent/OFTTA 比較棒グラフを保存する。"""
    subject_df = df[~df["Subject"].isin(["Average", "Std"])].copy()
    subjects = subject_df["Subject"].tolist()
    x = np.arange(len(subjects))
    labels = [method_label(method) for method in methods]
    width = min(0.8 / max(len(labels), 1), 0.26)
    offset_center = (len(labels) - 1) / 2

    fig, ax = plt.subplots(figsize=(max(12, len(subjects) * 0.7), 5.5))
    for idx, label in enumerate(labels):
        column = f"{label}_{metric_suffix}"
        if column not in subject_df.columns:
            continue
        ax.bar(
            x + (idx - offset_center) * width,
            subject_df[column].to_numpy(),
            width,
            label=label,
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
        )

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


def save_average_barplot(df, output_path, methods):
    """Average 行だけを使い、3 手法の主要指標を保存する。"""
    avg = df[df["Subject"] == "Average"].iloc[0]
    metrics = [
        ("Accuracy", "Acc"),
        ("Macro-F1", "MacroF1"),
        ("F1 Stress", "F1_1"),
    ]
    if "Source_F1_2" in df.columns:
        metrics.append(("F1 Amusement", "F1_2"))
    labels = [method_label(method) for method in methods]
    x = np.arange(len(metrics))
    width = min(0.8 / max(len(labels), 1), 0.26)
    offset_center = (len(labels) - 1) / 2

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for idx, label in enumerate(labels):
        values = [avg[f"{label}_{suffix}"] for _, suffix in metrics]
        ax.bar(
            x + (idx - offset_center) * width,
            values,
            width,
            label=label,
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
        )

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
    if getattr(args, "run_name", None):
        current_time = f"{current_time}_{args.run_name}"
    out_dir = os.path.join(args.out_path, args.dataset, "compare_source_tent_oftta", current_time)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main():
    cli_args = parse_args()
    methods = cli_args.methods if cli_args.methods is not None else METHODS
    base_args = build_args(cli_args, "source", "S2")
    base_args.run_name = cli_args.run_name
    set_seed(base_args.seed)
    device = get_device(base_args.device)
    criterion = move_criterion_to_device(make_criterion(base_args), device)
    subjects = resolve_subjects(cli_args)
    out_dir = make_output_dir(base_args)

    rows = []
    diagnostic_rows = []
    for subject in subjects:
        print(f"=== {subject} ===")
        target_args = build_args(cli_args, "source", subject)
        target_loader = get_target_dataset(target_args)
        row = {"Subject": subject}

        for method in methods:
            method_args = build_args(cli_args, method, subject)
            # target_shuffle=True の比較では DataLoader の反復ごとに順序が変わる。
            # 各手法が同じ shuffled batch 列を見るよう、評価直前に seed を戻す。
            set_seed(method_args.seed)
            metrics = evaluate_method(method_args, target_loader, device, criterion)
            prefix = method_label(method)
            row[f"{prefix}_Acc"] = metrics["accuracy"]
            row[f"{prefix}_F1_0"] = metrics["f1_non_stress"]
            row[f"{prefix}_F1_1"] = metrics["f1_stress"]
            if metrics["f1_class_2"] is not None:
                row[f"{prefix}_F1_2"] = metrics["f1_class_2"]
            row[f"{prefix}_MacroF1"] = metrics["mean_f1"]
            for diagnostic in metrics.get("_tta_diagnostics", []):
                diagnostic_rows.append({"Subject": subject, "Method": prefix, **diagnostic})
            print(
                f"  {prefix}: Acc={metrics['accuracy']:.4f}, "
                f"F1(0)={metrics['f1_non_stress']:.4f}, "
                f"F1(1)={metrics['f1_stress']:.4f}, "
                f"F1(2)={(metrics['f1_class_2'] or 0.0):.4f}, "
                f"MacroF1={metrics['mean_f1']:.4f}"
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
    macro_f1_plot_path = os.path.join(out_dir, "source_tent_oftta_macro_f1.png")
    stress_f1_plot_path = os.path.join(out_dir, "source_tent_oftta_stress_f1.png")
    amusement_f1_plot_path = os.path.join(out_dir, "source_tent_oftta_amusement_f1.png")
    accuracy_plot_path = os.path.join(out_dir, "source_tent_oftta_accuracy.png")
    average_plot_path = os.path.join(out_dir, "source_tent_oftta_average.png")
    diagnostics_path = os.path.join(out_dir, "ema_tent_batch_diagnostics.csv")
    diagnostics_summary_path = os.path.join(out_dir, "ema_tent_gate_summary.md")
    stream_summary_path = os.path.join(out_dir, "tta_stream_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    if diagnostic_rows:
        diagnostics_df = pd.DataFrame(diagnostic_rows)
        diagnostics_df.to_csv(diagnostics_path, index=False)
        write_gate_summary(diagnostics_df, diagnostics_summary_path)
        write_stream_summary(
            summary_df,
            diagnostics_df,
            stream_summary_path,
            methods,
            getattr(base_args, "stream_mode", "original"),
            getattr(base_args, "block_length", None),
        )

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
        "MacroF1",
        "Macro-F1",
        "Source vs Tent vs OFTTA: Macro-F1 by Subject",
        macro_f1_plot_path,
        methods,
    )
    save_metric_barplot(
        summary_df,
        "F1_1",
        "F1 Stress",
        "Source vs Tent vs OFTTA: Stress-class F1 by Subject",
        stress_f1_plot_path,
        methods,
    )
    if "Source_F1_2" in summary_df.columns:
        save_metric_barplot(
            summary_df,
            "F1_2",
            "F1 Amusement",
            "Source vs Tent vs OFTTA: Amusement-class F1 by Subject",
            amusement_f1_plot_path,
            methods,
        )
    save_metric_barplot(
        summary_df,
        "Acc",
        "Accuracy",
        "Source vs Tent vs OFTTA: Accuracy by Subject",
        accuracy_plot_path,
        methods,
    )
    save_average_barplot(summary_df, average_plot_path, methods)

    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "subjects": subjects,
                "methods": methods,
                "default_cfg": cli_args.default_cfg,
                "dataset_cfg": cli_args.dataset_cfg,
                "resume": base_args.resume,
                "batch_size": base_args.batch_size,
                "stream_mode": getattr(base_args, "stream_mode", "original"),
                "block_length": getattr(base_args, "block_length", None),
                "run_name": cli_args.run_name,
            },
            handle,
            default_flow_style=False,
        )

    print(f"\nCSV saved to: {csv_path}")
    print(f"Markdown saved to: {md_path}")
    print(f"Macro-F1 plot saved to: {macro_f1_plot_path}")
    print(f"Stress F1 plot saved to: {stress_f1_plot_path}")
    if "Source_F1_2" in summary_df.columns:
        print(f"Amusement F1 plot saved to: {amusement_f1_plot_path}")
    print(f"Accuracy plot saved to: {accuracy_plot_path}")
    print(f"Average plot saved to: {average_plot_path}")
    if diagnostic_rows:
        print(f"EMA-Tent diagnostics saved to: {diagnostics_path}")
        print(f"EMA-Tent gate summary saved to: {diagnostics_summary_path}")
        print(f"TTA stream summary saved to: {stream_summary_path}")


if __name__ == "__main__":
    main()
