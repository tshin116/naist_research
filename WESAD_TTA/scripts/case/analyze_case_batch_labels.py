#!/usr/bin/env python3
"""Summarize label composition of CASE target batches.

The CASE preprocessing binarizes each subject by that subject's annotation
mean. Therefore label 0/1 means below/above the subject-specific threshold.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def discover_processed_files(processed_dir: Path, label_type: str) -> list[Path]:
    files = sorted(
        processed_dir.glob(f"S*_{label_type.lower()}_ws*_stride*_ch*.npz"),
        key=lambda p: int(p.name.split("_", 1)[0][1:]),
    )
    if not files:
        raise FileNotFoundError(f"No processed CASE files found in {processed_dir}")
    return files


def summarize_batches(processed_dir: Path, label_type: str, batch_size: int) -> pd.DataFrame:
    rows = []
    for path in discover_processed_files(processed_dir, label_type):
        subject = path.name.split("_", 1)[0]
        with np.load(path, allow_pickle=False) as data:
            labels = data["y"].astype(int)

        num_batches = int(np.ceil(len(labels) / batch_size))
        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(labels))
            batch_labels = labels[start:end]
            count_0 = int((batch_labels == 0).sum())
            count_1 = int((batch_labels == 1).sum())
            n = int(len(batch_labels))
            majority = max(count_0, count_1)
            rows.append(
                {
                    "label_type": label_type.upper(),
                    "subject": subject,
                    "batch_index": batch_index,
                    "start_window": start,
                    "end_window_exclusive": end,
                    "num_windows": n,
                    "label0_count": count_0,
                    "label1_count": count_1,
                    "label1_ratio": count_1 / n if n else 0.0,
                    "majority_ratio": majority / n if n else 0.0,
                    "is_single_class": int(count_0 == 0 or count_1 == 0),
                }
            )
    return pd.DataFrame(rows)


def summarize_subjects(batch_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in batch_df.groupby("subject", sort=False):
        rows.append(
            {
                "subject": subject,
                "num_batches": int(len(group)),
                "num_windows": int(group["num_windows"].sum()),
                "label0_windows": int(group["label0_count"].sum()),
                "label1_windows": int(group["label1_count"].sum()),
                "single_class_batches": int(group["is_single_class"].sum()),
                "mean_majority_ratio": float(group["majority_ratio"].mean()),
                "max_majority_ratio": float(group["majority_ratio"].max()),
                "batch_sizes": ", ".join(str(int(v)) for v in group["num_windows"].tolist()),
                "batch_label_counts": " | ".join(
                    f"B{int(row.batch_index)}:0={int(row.label0_count)},1={int(row.label1_count)}"
                    for row in group.itertuples()
                ),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *row_lines])


def write_markdown(label_type: str, batch_df: pd.DataFrame, subject_df: pd.DataFrame, path: Path) -> None:
    total_batches = int(len(batch_df))
    single_batches = int(batch_df["is_single_class"].sum())
    mean_majority = float(batch_df["majority_ratio"].mean())
    max_majority = float(batch_df["majority_ratio"].max())

    display = subject_df.copy()
    for column in ["mean_majority_ratio", "max_majority_ratio"]:
        display[column] = display[column].map(lambda x: f"{x:.3f}")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# CASE {label_type.upper()} Batch Label Composition\n\n")
        handle.write(f"- batch_size: `{int(batch_df['num_windows'].max())}`\n")
        handle.write(f"- total batches: `{total_batches}`\n")
        handle.write(f"- single-class batches: `{single_batches}`\n")
        handle.write(f"- mean majority ratio: `{mean_majority:.3f}`\n")
        handle.write(f"- max majority ratio: `{max_majority:.3f}`\n\n")
        handle.write("Label meaning:\n\n")
        handle.write("- `0`: below subject-specific annotation mean\n")
        handle.write("- `1`: above subject-specific annotation mean\n\n")
        handle.write(dataframe_to_markdown(display))
        handle.write("\n")


def plot_subject_bars(label_type: str, batch_df: pd.DataFrame, path: Path) -> None:
    subjects = list(batch_df["subject"].drop_duplicates())
    fig_height = max(5.0, len(subjects) * 0.32)
    fig, axes = plt.subplots(len(subjects), 1, figsize=(14, fig_height), sharex=True)
    if len(subjects) == 1:
        axes = [axes]

    colors = {0: "#4C78A8", 1: "#E45756"}
    for ax, subject in zip(axes, subjects):
        group = batch_df[batch_df["subject"] == subject]
        left = 0
        for row in group.itertuples():
            ax.barh(0, row.label0_count, left=left, color=colors[0], height=0.7)
            left += row.label0_count
            ax.barh(0, row.label1_count, left=left, color=colors[1], height=0.7)
            left += row.label1_count
            ax.axvline(left, color="white", linewidth=1)
        ax.set_yticks([0])
        ax.set_yticklabels([subject], fontsize=10)
        ax.set_ylim(-0.7, 0.7)

    axes[0].legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=colors[0], label="0: below mean"),
            plt.Rectangle((0, 0), 1, 1, color=colors[1], label="1: above mean"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.8),
    )
    axes[-1].set_xlabel("Window index in target subject")
    fig.suptitle(f"CASE {label_type.upper()} target-batch label composition", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valence_cfg", default="./cfg/dataset/case_valence.yaml")
    parser.add_argument("--arousal_cfg", default="./cfg/dataset/case_arousal.yaml")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or f"./logs/case/batch_label_analysis/{timestamp}_CASE_batch_labels")
    out_dir.mkdir(parents=True, exist_ok=True)

    for cfg_path in [args.valence_cfg, args.arousal_cfg]:
        cfg = load_yaml(cfg_path)
        label_type = cfg["label_type"].lower()
        batch_size = int(cfg["batch_size"])
        processed_dir = Path(cfg["processed_dir"]).resolve()

        batch_df = summarize_batches(processed_dir, label_type, batch_size)
        subject_df = summarize_subjects(batch_df)

        batch_csv = out_dir / f"case_{label_type}_batch_label_counts.csv"
        subject_csv = out_dir / f"case_{label_type}_subject_batch_summary.csv"
        md_path = out_dir / f"case_{label_type}_batch_label_summary.md"
        png_path = out_dir / f"case_{label_type}_batch_label_composition.png"

        batch_df.to_csv(batch_csv, index=False)
        subject_df.to_csv(subject_csv, index=False)
        write_markdown(label_type, batch_df, subject_df, md_path)
        plot_subject_bars(label_type, batch_df, png_path)

        print(f"{label_type.upper()} batch CSV: {batch_csv}")
        print(f"{label_type.upper()} subject CSV: {subject_csv}")
        print(f"{label_type.upper()} summary MD: {md_path}")
        print(f"{label_type.upper()} plot PNG: {png_path}")


if __name__ == "__main__":
    main()
