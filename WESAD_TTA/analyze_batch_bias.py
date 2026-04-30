"""Analyze target-batch class composition and its relation to TTA performance."""

import argparse
import csv
import math
import os
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from config import load_yaml
from data_processing.wesad import discover_subjects
from utils import get_target_dataset, set_seed


DEFAULT_CLASS_NAMES = {
    0: "Baseline",
    1: "Stress",
    2: "Amusement",
}
DEFAULT_METHODS = ["Tent", "OFTTA"]
BIAS_METRICS = [
    ("mean_max_class_ratio", "Mean max class ratio"),
    ("mean_imbalance", "Mean batch imbalance"),
    ("single_class_batch_ratio", "Single-class batch ratio"),
    ("mean_missing_classes", "Mean missing classes"),
]
CLASS_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2", "#E45756"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize WESAD target batch class bias and correlate it with TTA performance."
    )
    parser.add_argument("--default_cfg", type=str, default="./cfg/default.yaml")
    parser.add_argument("--dataset_cfg", type=str, default="./cfg/dataset/wesad_3class.yaml")
    parser.add_argument(
        "--comparison_csv",
        type=str,
        required=True,
        help="Sequential-order Source/Tent/OFTTA comparison CSV.",
    )
    parser.add_argument(
        "--shuffle_comparison_csv",
        type=str,
        default=None,
        help="Optional shuffled-batch comparison CSV for shuffle-gain correlation.",
    )
    parser.add_argument("--out_path", type=str, default="./logs")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method labels in comparison CSV to correlate. If omitted, all non-Source *_MacroF1/*_MeanF1 columns are used.",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def build_base_args(cli_args, subject):
    cfg = {}
    cfg.update(load_yaml(cli_args.default_cfg))
    cfg.update(load_yaml(cli_args.dataset_cfg))
    cfg["target_domain"] = subject
    if cli_args.seed is not None:
        cfg["seed"] = cli_args.seed
    # This analysis should expose the natural block structure unless the user
    # intentionally passes a shuffled dataset config.
    return SimpleNamespace(**cfg)


def resolve_subjects(cli_args):
    if cli_args.subjects:
        return cli_args.subjects
    dataset_cfg = load_yaml(cli_args.dataset_cfg)
    subjects = dataset_cfg.get("subjects")
    if subjects:
        return subjects
    dataset_dir = os.path.abspath(dataset_cfg["dataset_dir"])
    return sorted(discover_subjects(dataset_dir), key=lambda subject: int(subject[1:]))


def make_output_dir(cli_args):
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(cli_args.out_path, "wesad", "batch_bias_analysis", current_time)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def read_comparison_csv(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject = row["Subject"]
            if subject in {"Average", "Std"}:
                continue
            parsed = {"Subject": subject}
            for key, value in row.items():
                if key == "Subject":
                    continue
                parsed[key] = float(value)
            rows[subject] = parsed
    return rows


def infer_methods(comparison_rows, cli_methods=None):
    if cli_methods:
        return cli_methods
    first = next(iter(comparison_rows.values()))
    methods = []
    for key in first:
        if key.endswith("_MacroF1"):
            method = key[: -len("_MacroF1")]
        elif key.endswith("_MeanF1"):
            method = key[: -len("_MeanF1")]
        else:
            continue
        if method != "Source":
            methods.append(method)
    return methods or DEFAULT_METHODS


def macro_f1_key(row, method):
    """Return the Macro-F1 column key, accepting old MeanF1 logs for compatibility."""
    macro_key = f"{method}_MacroF1"
    legacy_key = f"{method}_MeanF1"
    if macro_key in row:
        return macro_key
    if legacy_key in row:
        return legacy_key
    return None


def batch_counts_from_loader(loader, num_classes):
    batch_rows = []
    for batch_index, (_, y_batch) in enumerate(loader):
        labels = y_batch.detach().cpu().numpy().astype(int).reshape(-1)
        counts = np.bincount(labels, minlength=num_classes).astype(float)
        size = int(counts.sum())
        proportions = counts / max(size, 1)
        max_ratio = float(proportions.max()) if size else 0.0
        entropy = normalized_entropy(proportions)
        batch_rows.append(
            {
                "batch_index": batch_index,
                "batch_size": size,
                "counts": counts,
                "proportions": proportions,
                "majority_class": int(proportions.argmax()) if size else -1,
                "max_class_ratio": max_ratio,
                "normalized_entropy": entropy,
                "imbalance": 1.0 - entropy,
                "missing_classes": int(np.sum(counts == 0)),
                "is_single_class": bool(np.isclose(max_ratio, 1.0)),
            }
        )
    return batch_rows


def normalized_entropy(proportions):
    nonzero = proportions[proportions > 0]
    if len(proportions) <= 1:
        return 1.0
    entropy = -float(np.sum(nonzero * np.log(nonzero)))
    return entropy / math.log(len(proportions))


def summarize_subject(subject, batch_rows, num_classes):
    if not batch_rows:
        return {
            "Subject": subject,
            "num_batches": 0,
            "num_samples": 0,
            "mean_max_class_ratio": 0.0,
            "std_max_class_ratio": 0.0,
            "mean_imbalance": 0.0,
            "std_imbalance": 0.0,
            "single_class_batch_ratio": 0.0,
            "mean_missing_classes": 0.0,
            **{f"overall_class_{idx}_ratio": 0.0 for idx in range(num_classes)},
        }

    max_ratios = np.array([row["max_class_ratio"] for row in batch_rows], dtype=float)
    imbalances = np.array([row["imbalance"] for row in batch_rows], dtype=float)
    missing = np.array([row["missing_classes"] for row in batch_rows], dtype=float)
    single = np.array([row["is_single_class"] for row in batch_rows], dtype=float)
    total_counts = np.sum([row["counts"] for row in batch_rows], axis=0)
    total_samples = int(total_counts.sum())
    overall = total_counts / max(total_samples, 1)

    summary = {
        "Subject": subject,
        "num_batches": len(batch_rows),
        "num_samples": total_samples,
        "mean_max_class_ratio": float(max_ratios.mean()),
        "std_max_class_ratio": float(max_ratios.std()),
        "mean_imbalance": float(imbalances.mean()),
        "std_imbalance": float(imbalances.std()),
        "single_class_batch_ratio": float(single.mean()),
        "mean_missing_classes": float(missing.mean()),
    }
    for idx in range(num_classes):
        summary[f"overall_class_{idx}_ratio"] = float(overall[idx])
    return summary


def attach_performance(summary_rows, comparison_rows, methods, shuffle_rows=None):
    for row in summary_rows:
        subject = row["Subject"]
        perf = comparison_rows.get(subject)
        if perf is None:
            continue
        source_key = macro_f1_key(perf, "Source")
        if source_key is None:
            continue
        row["Source_MacroF1"] = perf[source_key]
        for method in methods:
            metric_key = macro_f1_key(perf, method)
            if metric_key is None:
                continue
            row[f"{method}_MacroF1"] = perf[metric_key]
            row[f"{method}_delta_vs_source"] = perf[metric_key] - perf[source_key]

        if shuffle_rows is not None and subject in shuffle_rows:
            shuffle_perf = shuffle_rows[subject]
            for method in methods:
                metric_key = macro_f1_key(perf, method)
                shuffle_metric_key = macro_f1_key(shuffle_perf, method)
                if metric_key is None or shuffle_metric_key is None:
                    continue
                row[f"{method}_shuffle_gain"] = shuffle_perf[shuffle_metric_key] - perf[metric_key]


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(path, rows, title):
    if not rows:
        return
    headers = list(rows[0].keys())
    formatted_rows = [[format_value(row.get(header, "")) for header in headers] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in formatted_rows))
        for idx, header in enumerate(headers)
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(str(h).ljust(widths[idx]) for idx, h in enumerate(headers)) + " |\n")
        handle.write("| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |\n")
        for row in formatted_rows:
            handle.write("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |\n")


def class_names(num_classes):
    return [DEFAULT_CLASS_NAMES.get(idx, f"Class {idx}") for idx in range(num_classes)]


def save_batch_distribution_plot(subject_batches, num_classes, output_path):
    subjects = list(subject_batches.keys())
    fig_height = max(6.0, 0.55 * len(subjects))
    fig, axes = plt.subplots(
        len(subjects),
        1,
        figsize=(14, fig_height),
        sharex=True,
        squeeze=False,
    )
    names = class_names(num_classes)

    for row_index, subject in enumerate(subjects):
        ax = axes[row_index][0]
        batches = subject_batches[subject]
        x = np.arange(len(batches))
        bottom = np.zeros(len(batches))
        for class_idx in range(num_classes):
            values = np.array([batch["proportions"][class_idx] for batch in batches])
            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.92,
                color=CLASS_COLORS[class_idx % len(CLASS_COLORS)],
                linewidth=0,
                label=names[class_idx] if row_index == 0 else None,
            )
            bottom += values
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(subject, rotation=0, ha="right", va="center")
        ax.set_yticks([0.0, 1.0])
        ax.grid(axis="y", linestyle=":", alpha=0.25)

    axes[-1][0].set_xlabel("Batch index in target loader")
    fig.suptitle("Class Composition of Each Target Batch")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(num_classes, 4), bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_batch_imbalance_heatmap(subject_batches, output_path):
    subjects = list(subject_batches.keys())
    max_batches = max(len(batches) for batches in subject_batches.values())
    matrix = np.full((len(subjects), max_batches), np.nan, dtype=float)
    for row_idx, subject in enumerate(subjects):
        for batch in subject_batches[subject]:
            matrix[row_idx, batch["batch_index"]] = batch["max_class_ratio"]

    cmap = plt.cm.magma.copy()
    cmap.set_bad("#E5E5E5")
    fig, ax = plt.subplots(figsize=(14, max(5, 0.42 * len(subjects))))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=1 / 3, vmax=1.0)
    ax.set_title("Batch Bias Heatmap: Maximum Class Ratio per Batch")
    ax.set_xlabel("Batch index in target loader")
    ax.set_ylabel("Subject")
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Max class ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def rankdata(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return np.array(ranks, dtype=float)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.isclose(x.std(), 0) or np.isclose(y.std(), 0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(list(x)), rankdata(list(y)))


def fit_line(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.isclose(x.std(), 0):
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def save_correlation_scatter(summary_rows, target_suffix, output_path, methods):
    y_columns = [f"{method}_{target_suffix}" for method in methods if f"{method}_{target_suffix}" in summary_rows[0]]
    if not y_columns:
        return
    fig, axes = plt.subplots(
        len(BIAS_METRICS),
        len(y_columns),
        figsize=(max(6, 5.2 * len(y_columns)), 3.4 * len(BIAS_METRICS)),
        squeeze=False,
    )

    for row_idx, (metric, metric_label) in enumerate(BIAS_METRICS):
        x = np.array([row[metric] for row in summary_rows], dtype=float)
        for col_idx, y_col in enumerate(y_columns):
            ax = axes[row_idx][col_idx]
            y = np.array([row[y_col] for row in summary_rows], dtype=float)
            ax.scatter(x, y, color="#4C78A8", edgecolor="white", linewidth=0.7, s=54)
            for row, x_value, y_value in zip(summary_rows, x, y):
                ax.annotate(row["Subject"], (x_value, y_value), fontsize=8, xytext=(3, 3), textcoords="offset points")

            line = fit_line(x, y)
            if line is not None:
                slope, intercept = line
                xs = np.linspace(float(x.min()), float(x.max()), 100)
                ax.plot(xs, slope * xs + intercept, color="#E45756", linewidth=1.6)

            r = pearson(x, y)
            rho = spearman(x, y)
            method = y_col[: -len(f"_{target_suffix}")]
            title = f"{method}: r={r:.3f}, rho={rho:.3f}"
            ax.set_title(title)
            ax.set_xlabel(metric_label)
            ax.set_ylabel(y_col)
            ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.grid(True, linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_source_performance_scatter(summary_rows, output_path, methods):
    if "Source_MacroF1" not in summary_rows[0]:
        return
    y_columns = []
    for method in methods:
        delta_col = f"{method}_delta_vs_source"
        if delta_col in summary_rows[0]:
            y_columns.append(delta_col)
    if not y_columns:
        return

    fig, axes = plt.subplots(
        1,
        len(y_columns),
        figsize=(max(6, 5.2 * len(y_columns)), 4.4),
        squeeze=False,
    )
    x = np.array([row["Source_MacroF1"] for row in summary_rows], dtype=float)
    for col_idx, y_col in enumerate(y_columns):
        ax = axes[0][col_idx]
        y = np.array([row[y_col] for row in summary_rows], dtype=float)
        ax.scatter(x, y, color="#4C78A8", edgecolor="white", linewidth=0.7, s=64)
        for row, x_value, y_value in zip(summary_rows, x, y):
            ax.annotate(row["Subject"], (x_value, y_value), fontsize=8, xytext=(3, 3), textcoords="offset points")

        line = fit_line(x, y)
        if line is not None:
            slope, intercept = line
            xs = np.linspace(float(x.min()), float(x.max()), 100)
            ax.plot(xs, slope * xs + intercept, color="#E45756", linewidth=1.8)

        r = pearson(x, y)
        rho = spearman(x, y)
        method = y_col[: -len("_delta_vs_source")]
        ax.set_title(f"{method}: r={r:.3f}, rho={rho:.3f}")
        ax.set_xlabel("Source Macro-F1")
        ax.set_ylabel(f"{method} - Source Macro-F1")
        ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle("Source performance vs TTA improvement")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def source_performance_correlation_rows(summary_rows, methods):
    rows = []
    if "Source_MacroF1" not in summary_rows[0]:
        return rows
    x = [row["Source_MacroF1"] for row in summary_rows]
    for method in methods:
        y_col = f"{method}_delta_vs_source"
        if y_col not in summary_rows[0]:
            continue
        y = [row[y_col] for row in summary_rows]
        rows.append(
            {
                "target": y_col,
                "x_metric": "Source_MacroF1",
                "n": len(summary_rows),
                "pearson_r": pearson(x, y),
                "spearman_rho": spearman(x, y),
            }
        )
    return rows


def correlation_rows(summary_rows, suffixes, methods):
    rows = []
    for suffix in suffixes:
        for method in methods:
            y_col = f"{method}_{suffix}"
            if y_col not in summary_rows[0]:
                continue
            y = [row[y_col] for row in summary_rows]
            for metric, _ in BIAS_METRICS:
                x = [row[metric] for row in summary_rows]
                rows.append(
                    {
                        "target": y_col,
                        "bias_metric": metric,
                        "n": len(summary_rows),
                        "pearson_r": pearson(x, y),
                        "spearman_rho": spearman(x, y),
                    }
                )
    return rows


def save_summary_boxplot(summary_rows, output_path):
    metrics = ["mean_max_class_ratio", "mean_imbalance", "single_class_batch_ratio", "mean_missing_classes"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4), squeeze=False)
    for idx, metric in enumerate(metrics):
        ax = axes[0][idx]
        values = [row[metric] for row in summary_rows]
        ax.boxplot(values, vert=True, widths=0.45)
        ax.scatter(np.ones(len(values)), values, color="#4C78A8", edgecolor="white", zorder=3)
        for row, value in zip(summary_rows, values):
            ax.annotate(row["Subject"], (1.0, value), fontsize=8, xytext=(4, 0), textcoords="offset points")
        ax.set_title(metric)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.suptitle("Distribution of Subject-Level Batch Bias Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    cli_args = parse_args()
    subjects = resolve_subjects(cli_args)
    base_cfg = load_yaml(cli_args.dataset_cfg)
    num_classes = int(base_cfg.get("num_classes", 3))
    seed = cli_args.seed if cli_args.seed is not None else load_yaml(cli_args.default_cfg).get("seed", 42)
    set_seed(seed)

    out_dir = make_output_dir(cli_args)
    subject_batches = {}
    summary_rows = []
    batch_detail_rows = []

    for subject in subjects:
        args = build_base_args(cli_args, subject)
        set_seed(seed)
        loader = get_target_dataset(args)
        batches = batch_counts_from_loader(loader, num_classes)
        subject_batches[subject] = batches
        summary_rows.append(summarize_subject(subject, batches, num_classes))

        for batch in batches:
            detail = {
                "Subject": subject,
                "batch_index": batch["batch_index"],
                "batch_size": batch["batch_size"],
                "majority_class": batch["majority_class"],
                "max_class_ratio": batch["max_class_ratio"],
                "normalized_entropy": batch["normalized_entropy"],
                "imbalance": batch["imbalance"],
                "missing_classes": batch["missing_classes"],
                "is_single_class": int(batch["is_single_class"]),
            }
            for class_idx in range(num_classes):
                detail[f"class_{class_idx}_count"] = int(batch["counts"][class_idx])
                detail[f"class_{class_idx}_ratio"] = float(batch["proportions"][class_idx])
            batch_detail_rows.append(detail)

    comparison_rows = read_comparison_csv(cli_args.comparison_csv)
    shuffle_rows = read_comparison_csv(cli_args.shuffle_comparison_csv) if cli_args.shuffle_comparison_csv else None
    methods = infer_methods(comparison_rows, cli_args.methods)
    attach_performance(summary_rows, comparison_rows, methods, shuffle_rows=shuffle_rows)

    summary_csv = os.path.join(out_dir, "batch_bias_summary.csv")
    detail_csv = os.path.join(out_dir, "batch_bias_by_batch.csv")
    corr_csv = os.path.join(out_dir, "batch_bias_correlations.csv")
    source_corr_csv = os.path.join(out_dir, "source_performance_correlations.csv")
    summary_md = os.path.join(out_dir, "batch_bias_summary.md")
    corr_md = os.path.join(out_dir, "batch_bias_correlations.md")
    source_corr_md = os.path.join(out_dir, "source_performance_correlations.md")

    write_csv(summary_csv, summary_rows)
    write_csv(detail_csv, batch_detail_rows)
    write_markdown_table(summary_md, summary_rows, "Batch Bias Summary")

    suffixes = ["delta_vs_source"]
    if shuffle_rows is not None:
        suffixes.append("shuffle_gain")
    corr = correlation_rows(summary_rows, suffixes, methods)
    write_csv(corr_csv, corr)
    write_markdown_table(corr_md, corr, "Batch Bias Correlations")
    source_corr = source_performance_correlation_rows(summary_rows, methods)
    write_csv(source_corr_csv, source_corr)
    write_markdown_table(source_corr_md, source_corr, "Source Performance Correlations")

    save_batch_distribution_plot(
        subject_batches,
        num_classes,
        os.path.join(out_dir, "batch_class_distribution_by_subject.png"),
    )
    save_batch_imbalance_heatmap(
        subject_batches,
        os.path.join(out_dir, "batch_max_class_ratio_heatmap.png"),
    )
    save_summary_boxplot(
        summary_rows,
        os.path.join(out_dir, "batch_bias_metric_boxplots.png"),
    )
    save_correlation_scatter(
        summary_rows,
        "delta_vs_source",
        os.path.join(out_dir, "batch_bias_vs_tta_delta_scatter.png"),
        methods,
    )
    save_source_performance_scatter(
        summary_rows,
        os.path.join(out_dir, "source_macro_f1_vs_tta_delta_scatter.png"),
        methods,
    )
    if shuffle_rows is not None:
        save_correlation_scatter(
            summary_rows,
            "shuffle_gain",
            os.path.join(out_dir, "batch_bias_vs_shuffle_gain_scatter.png"),
            methods,
        )

    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "default_cfg": cli_args.default_cfg,
                "dataset_cfg": cli_args.dataset_cfg,
                "comparison_csv": cli_args.comparison_csv,
                "shuffle_comparison_csv": cli_args.shuffle_comparison_csv,
                "subjects": subjects,
                "num_classes": num_classes,
                "methods": methods,
                "seed": seed,
            },
            handle,
            allow_unicode=True,
            default_flow_style=False,
        )

    print(f"Output directory: {out_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Batch detail CSV: {detail_csv}")
    print(f"Correlation CSV: {corr_csv}")
    print(f"Source performance correlation CSV: {source_corr_csv}")
    print("Figures:")
    print(f"  {os.path.join(out_dir, 'batch_class_distribution_by_subject.png')}")
    print(f"  {os.path.join(out_dir, 'batch_max_class_ratio_heatmap.png')}")
    print(f"  {os.path.join(out_dir, 'batch_bias_metric_boxplots.png')}")
    print(f"  {os.path.join(out_dir, 'batch_bias_vs_tta_delta_scatter.png')}")
    print(f"  {os.path.join(out_dir, 'source_macro_f1_vs_tta_delta_scatter.png')}")
    if shuffle_rows is not None:
        print(f"  {os.path.join(out_dir, 'batch_bias_vs_shuffle_gain_scatter.png')}")


if __name__ == "__main__":
    main()
