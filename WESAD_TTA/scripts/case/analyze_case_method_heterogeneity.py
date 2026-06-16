#!/usr/bin/env python3
"""Analyze why different TTA methods win for different CASE subjects."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "logs/case/method_heterogeneity_analysis/260616_CASE_method_heterogeneity"

TASKS = {
    "valence": {
        "basic": ROOT
        / "logs/case/compare_source_tent_oftta/260616_053356_CASE_valence_SourceNormTentOFTTAEMATent",
        "focused": ROOT
        / "logs/case/compare_source_tent_oftta/260616_061218_CASE_valence_EMATentFocusedGrid",
        "batch": ROOT
        / "logs/case/batch_label_analysis/260616_063111_CASE_batch_labels/case_valence_batch_label_counts.csv",
    },
    "arousal": {
        "basic": ROOT
        / "logs/case/compare_source_tent_oftta/260616_053358_CASE_arousal_SourceNormTentOFTTAEMATent",
        "focused": ROOT
        / "logs/case/compare_source_tent_oftta/260616_061238_CASE_arousal_EMATentFocusedGrid",
        "batch": ROOT
        / "logs/case/batch_label_analysis/260616_063111_CASE_batch_labels/case_arousal_batch_label_counts.csv",
    },
}

BASE_METHODS = ["Source", "Norm", "Tent", "OFTTA"]
DEFAULT_EMA = "EMATent"
BEST_EMA = "EmaTentCaseS00E50B50M070"


def subject_sort_key(subject: str) -> int:
    return int(str(subject).removeprefix("S"))


def non_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["Subject"].isin(["Average", "Std"])].copy()


def load_performance(task_cfg: dict) -> pd.DataFrame:
    basic = non_summary(pd.read_csv(task_cfg["basic"] / "source_tent_oftta_comparison.csv"))
    focused = non_summary(pd.read_csv(task_cfg["focused"] / "source_tent_oftta_comparison.csv"))

    keep_cols = ["Subject"]
    for suffix in ["Acc", "F1_0", "F1_1", "MacroF1"]:
        col = f"{BEST_EMA}_{suffix}"
        if col in focused.columns:
            keep_cols.append(col)
    perf = basic.merge(focused[keep_cols], on="Subject", how="left")

    method_cols = [f"{method}_MacroF1" for method in BASE_METHODS + [DEFAULT_EMA, BEST_EMA]]
    available_methods = [col.removesuffix("_MacroF1") for col in method_cols if col in perf.columns]
    perf["BestMethod"] = perf[[f"{m}_MacroF1" for m in available_methods]].idxmax(axis=1).str.removesuffix("_MacroF1")
    perf["BestBaseline"] = perf[[f"{m}_MacroF1" for m in BASE_METHODS]].idxmax(axis=1).str.removesuffix("_MacroF1")
    perf["BestBaseline_MacroF1"] = perf[[f"{m}_MacroF1" for m in BASE_METHODS]].max(axis=1)

    for method in [DEFAULT_EMA, BEST_EMA, "Norm", "Tent", "OFTTA"]:
        if f"{method}_MacroF1" in perf.columns:
            perf[f"{method}_gain_vs_source"] = perf[f"{method}_MacroF1"] - perf["Source_MacroF1"]
            perf[f"{method}_gain_vs_best_baseline"] = perf[f"{method}_MacroF1"] - perf["BestBaseline_MacroF1"]

    perf["DefaultEMA_beats_all_baselines"] = (perf[f"{DEFAULT_EMA}_gain_vs_best_baseline"] > 0).astype(int)
    perf["BestEMA_beats_all_baselines"] = (perf[f"{BEST_EMA}_gain_vs_best_baseline"] > 0).astype(int)
    perf["DefaultEMA_beats_source"] = (perf[f"{DEFAULT_EMA}_gain_vs_source"] > 0).astype(int)
    perf["BestEMA_beats_source"] = (perf[f"{BEST_EMA}_gain_vs_source"] > 0).astype(int)
    return perf


def load_batch_features(batch_path: Path) -> pd.DataFrame:
    batch = pd.read_csv(batch_path)
    rows = []
    for subject, group in batch.groupby("subject", sort=False):
        label1 = group["label1_ratio"].to_numpy()
        majority = group["majority_ratio"].to_numpy()
        rows.append(
            {
                "Subject": subject,
                "num_windows": int(group["num_windows"].sum()),
                "num_batches": int(len(group)),
                "mean_majority_ratio": float(majority.mean()),
                "max_majority_ratio": float(majority.max()),
                "std_majority_ratio": float(majority.std(ddof=0)),
                "single_class_batches": int(group["is_single_class"].sum()),
                "full_single_class_batches": int(((group["is_single_class"] == 1) & (group["num_windows"] == 64)).sum()),
                "first_batch_majority_ratio": float(majority[0]),
                "last_batch_majority_ratio": float(majority[-1]),
                "first_batch_label1_ratio": float(label1[0]),
                "last_batch_label1_ratio": float(label1[-1]),
                "batch_label1_drift": float(abs(label1[-1] - label1[0])),
                "mean_abs_batch_label1_change": float(np.mean(np.abs(np.diff(label1)))) if len(label1) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_gate_features(diagnostics_path: Path, method: str, prefix: str) -> pd.DataFrame:
    if not diagnostics_path.exists():
        return pd.DataFrame(columns=["Subject"])
    diag = pd.read_csv(diagnostics_path)
    diag = diag[diag["Method"] == method].copy()
    if diag.empty:
        return pd.DataFrame(columns=["Subject"])
    rows = []
    for subject, group in diag.groupby("Subject", sort=False):
        rows.append(
            {
                "Subject": subject,
                f"{prefix}_mean_raw_gate": float(group["raw_gate"].mean()),
                f"{prefix}_min_raw_gate": float(group["raw_gate"].min()),
                f"{prefix}_mean_effective_batch_weight": float(group["effective_batch_weight"].mean()),
                f"{prefix}_mean_effective_source_weight": float(group["effective_source_weight"].mean()),
                f"{prefix}_mean_confidence": float(group["confidence"].mean()),
                f"{prefix}_updated_rate": float(group["updated"].mean()),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame, target_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for target in target_cols:
        for feature in feature_cols:
            valid = df[[target, feature]].dropna()
            if len(valid) < 4 or valid[target].nunique() < 2 or valid[feature].nunique() < 2:
                continue
            pearson_r, pearson_p = pearsonr(valid[feature], valid[target])
            spearman_r, spearman_p = spearmanr(valid[feature], valid[target])
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "spearman_r"], ascending=[True, False])


def group_summary(df: pd.DataFrame, group_col: str, feature_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_col)[feature_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(group_col)
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: f"{x:.4f}")
    headers = [str(c) for c in display.columns]
    rows = [[str(v) for v in row] for row in display.to_numpy()]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def save_scatter(df: pd.DataFrame, x: str, y: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(df[x], df[y], s=46, color="#4C78A8", alpha=0.85)
    for row in df.itertuples():
        ax.annotate(row.Subject, (getattr(row, x), getattr(row, y)), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def analyze_task(task: str, cfg: dict) -> pd.DataFrame:
    perf = load_performance(cfg)
    batch = load_batch_features(cfg["batch"])
    default_gate = load_gate_features(cfg["basic"] / "ema_tent_batch_diagnostics.csv", DEFAULT_EMA, "default_ema")
    best_gate = load_gate_features(cfg["focused"] / "ema_tent_batch_diagnostics.csv", BEST_EMA, "best_ema")

    df = perf.merge(batch, on="Subject", how="left").merge(default_gate, on="Subject", how="left").merge(best_gate, on="Subject", how="left")
    df = df.sort_values("Subject", key=lambda s: s.map(subject_sort_key))
    df.insert(0, "Task", task)

    out_prefix = OUT_DIR / task
    df.to_csv(out_prefix.with_suffix(".per_subject_features.csv"), index=False)

    feature_cols = [
        "Source_MacroF1",
        "Norm_gain_vs_source",
        "Tent_gain_vs_source",
        "OFTTA_gain_vs_source",
        "num_windows",
        "num_batches",
        "mean_majority_ratio",
        "max_majority_ratio",
        "std_majority_ratio",
        "single_class_batches",
        "full_single_class_batches",
        "batch_label1_drift",
        "mean_abs_batch_label1_change",
        "default_ema_mean_raw_gate",
        "default_ema_mean_effective_batch_weight",
        "default_ema_mean_confidence",
        "best_ema_mean_raw_gate",
        "best_ema_mean_effective_batch_weight",
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]
    target_cols = [
        f"{DEFAULT_EMA}_gain_vs_best_baseline",
        f"{BEST_EMA}_gain_vs_best_baseline",
        f"{DEFAULT_EMA}_gain_vs_source",
        f"{BEST_EMA}_gain_vs_source",
    ]
    corr = correlation_table(df, target_cols, feature_cols)
    corr.to_csv(out_prefix.with_suffix(".correlations.csv"), index=False)

    method_cols = ["Subject", "BestMethod", "BestBaseline", "Source_MacroF1", "Norm_MacroF1", "Tent_MacroF1", "OFTTA_MacroF1", f"{DEFAULT_EMA}_MacroF1", f"{BEST_EMA}_MacroF1", f"{DEFAULT_EMA}_gain_vs_best_baseline", f"{BEST_EMA}_gain_vs_best_baseline", "mean_majority_ratio", "single_class_batches", "default_ema_mean_raw_gate", "default_ema_mean_effective_batch_weight"]
    method_cols = [col for col in method_cols if col in df.columns]
    df[method_cols].to_csv(out_prefix.with_suffix(".method_summary.csv"), index=False)

    group_cols = [
        "Source_MacroF1",
        "mean_majority_ratio",
        "single_class_batches",
        "mean_abs_batch_label1_change",
        "default_ema_mean_raw_gate",
        "default_ema_mean_effective_batch_weight",
        f"{DEFAULT_EMA}_gain_vs_best_baseline",
        f"{BEST_EMA}_gain_vs_best_baseline",
    ]
    group_cols = [col for col in group_cols if col in df.columns]
    group_summary(df, "DefaultEMA_beats_all_baselines", group_cols).to_csv(out_prefix.with_suffix(".default_ema_group_summary.csv"), index=False)
    group_summary(df, "BestEMA_beats_all_baselines", group_cols).to_csv(out_prefix.with_suffix(".best_ema_group_summary.csv"), index=False)

    save_scatter(
        df,
        "Source_MacroF1",
        f"{DEFAULT_EMA}_gain_vs_best_baseline",
        out_prefix.with_suffix(".default_ema_gain_vs_source.png"),
        f"CASE {task}: default EMA-Tent gain vs source strength",
    )
    save_scatter(
        df,
        "mean_majority_ratio",
        f"{DEFAULT_EMA}_gain_vs_best_baseline",
        out_prefix.with_suffix(".default_ema_gain_vs_batch_bias.png"),
        f"CASE {task}: default EMA-Tent gain vs batch bias",
    )
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = []
    md_parts = ["# CASE Method Heterogeneity Analysis\n"]
    for task, cfg in TASKS.items():
        df = analyze_task(task, cfg)
        all_frames.append(df)
        summary_cols = [
            "Subject",
            "BestMethod",
            "BestBaseline",
            "Source_MacroF1",
            "Norm_MacroF1",
            "Tent_MacroF1",
            "OFTTA_MacroF1",
            f"{DEFAULT_EMA}_MacroF1",
            f"{BEST_EMA}_MacroF1",
            f"{DEFAULT_EMA}_gain_vs_best_baseline",
            f"{BEST_EMA}_gain_vs_best_baseline",
            "mean_majority_ratio",
            "single_class_batches",
            "default_ema_mean_raw_gate",
        ]
        summary_cols = [col for col in summary_cols if col in df.columns]
        top_corr = pd.read_csv(OUT_DIR / f"{task}.correlations.csv")
        top_corr = top_corr[top_corr["target"] == f"{DEFAULT_EMA}_gain_vs_best_baseline"].copy()
        top_corr["abs_spearman"] = top_corr["spearman_r"].abs()
        top_corr = top_corr.sort_values("abs_spearman", ascending=False).head(8)

        md_parts.append(f"\n## {task.title()}\n")
        md_parts.append("### Per-subject method summary\n")
        md_parts.append(dataframe_to_markdown(df[summary_cols]))
        md_parts.append("\n### Strongest correlations with default EMA-Tent gain vs best baseline\n")
        md_parts.append(dataframe_to_markdown(top_corr.drop(columns=["abs_spearman"])))

    all_df = pd.concat(all_frames, ignore_index=True)
    all_df.to_csv(OUT_DIR / "case_method_heterogeneity_all_subjects.csv", index=False)

    with open(OUT_DIR / "case_method_heterogeneity_analysis.md", "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(md_parts))
        handle.write("\n")

    print(f"Analysis saved to: {OUT_DIR}")
    print(f"Markdown: {OUT_DIR / 'case_method_heterogeneity_analysis.md'}")


if __name__ == "__main__":
    main()
