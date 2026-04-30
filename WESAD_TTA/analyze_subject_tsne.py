"""Visualize inter-subject variability in WESAD feature space with t-SNE."""

import argparse
import os
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import load_yaml
from data_processing.wesad import discover_subjects, load_or_preprocess_subject, prepare_tensors
from utils import get_device, get_model, set_seed


LABEL_NAMES = {
    0: "baseline",
    1: "stress",
    2: "amusement",
}


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization of WESAD 1D-CNN features.")
    parser.add_argument("--default_cfg", type=str, default="./cfg/default.yaml")
    parser.add_argument("--dataset_cfg", type=str, default="./cfg/dataset/wesad_3class.yaml")
    parser.add_argument("--resume", type=str, default="./ckpt_3class")
    parser.add_argument(
        "--checkpoint_subject",
        type=str,
        default="S2",
        help="LOSO checkpoint fold used as the fixed feature extractor.",
    )
    parser.add_argument("--out_path", type=str, default="./logs")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--max_per_subject_label", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--point_size", type=float, default=22.0)
    parser.add_argument("--font_size", type=float, default=16.0)
    parser.add_argument("--legend_font_size", type=float, default=11.0)
    parser.add_argument("--title_font_size", type=float, default=18.0)
    parser.add_argument(
        "--figure_format",
        type=str,
        default="pdf",
        choices=["pdf", "png"],
        help="Output format for figures.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def build_args(cli_args):
    cfg = {}
    cfg.update(load_yaml(cli_args.default_cfg))
    cfg.update(load_yaml(cli_args.dataset_cfg))
    cfg["target_domain"] = cli_args.checkpoint_subject
    cfg["resume"] = cli_args.resume
    if cli_args.device is not None:
        cfg["device"] = cli_args.device
    return SimpleNamespace(**cfg)


def checkpoint_path(args):
    return os.path.join(
        args.resume,
        args.dataset,
        args.model,
        args.target_domain,
        f"{args.dataset}_{args.target_domain}_checkpoint.pt",
    )


def load_feature_model(args, device):
    model = get_model(args).to(device)
    path = checkpoint_path(args)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, path


def resolve_subjects(args, cli_args):
    if cli_args.subjects:
        return cli_args.subjects
    subjects = getattr(args, "subjects", None)
    if subjects:
        return subjects
    dataset_dir = os.path.abspath(args.dataset_dir)
    return sorted(discover_subjects(dataset_dir), key=lambda subject: int(subject[1:]))


def sample_subject_windows(subject_data, subject, max_per_label, rng):
    x_data = subject_data["X"]
    y_data = subject_data["y"]
    indices = []
    for label in sorted(np.unique(y_data).astype(int)):
        label_indices = np.flatnonzero(y_data == label)
        if max_per_label > 0 and len(label_indices) > max_per_label:
            label_indices = rng.choice(label_indices, size=max_per_label, replace=False)
        indices.extend(label_indices.tolist())
    indices = np.array(sorted(indices), dtype=int)
    subjects = np.array([subject] * len(indices))
    return x_data[indices], y_data[indices], subjects


@torch.no_grad()
def extract_features(model, x_data, y_data, subjects, batch_size, label_mode, device):
    x_tensor, y_tensor = prepare_tensors(x_data, y_data, label_mode=label_mode)
    features = []
    labels = []
    subj = []
    for start in range(0, len(x_tensor), batch_size):
        end = start + batch_size
        x_batch = x_tensor[start:end].to(device)
        _, feat = model(x_batch, return_feature=True)
        features.append(feat.detach().cpu().numpy())
        labels.append(y_tensor[start:end].detach().cpu().numpy().astype(int))
        subj.append(subjects[start:end])
    return np.vstack(features), np.concatenate(labels), np.concatenate(subj)


def compute_embedding(features, seed, perplexity):
    if features.shape[1] > 50 and features.shape[0] > 50:
        features = PCA(n_components=50, random_state=seed).fit_transform(features)
    perplexity = min(perplexity, max(5.0, (features.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(features), perplexity


def subject_color_map(subjects):
    unique = sorted(np.unique(subjects), key=lambda subject: int(subject[1:]))
    cmap = plt.get_cmap("tab20")
    return {subject: cmap(idx % 20) for idx, subject in enumerate(unique)}


def apply_axis_font(ax, font_size, title_font_size):
    ax.title.set_fontsize(title_font_size)
    ax.xaxis.label.set_fontsize(font_size)
    ax.yaxis.label.set_fontsize(font_size)
    ax.tick_params(axis="both", labelsize=max(font_size - 2, 8))


def save_by_label_subject_plot(
    embedding,
    labels,
    subjects,
    output_path,
    point_size,
    font_size,
    legend_font_size,
    title_font_size,
):
    colors = subject_color_map(subjects)
    unique_subjects = sorted(colors, key=lambda subject: int(subject[1:]))
    unique_labels = sorted(np.unique(labels).astype(int))

    fig, axes = plt.subplots(1, len(unique_labels), figsize=(6.4 * len(unique_labels), 5.8), sharex=True, sharey=True)
    if len(unique_labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, unique_labels):
        mask_label = labels == label
        for subject in unique_subjects:
            mask = mask_label & (subjects == subject)
            if not np.any(mask):
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=point_size,
                alpha=0.78,
                color=colors[subject],
                label=subject,
                linewidths=0,
            )
        ax.set_title(f"{label}: {LABEL_NAMES.get(label, str(label))}")
        ax.set_xlabel("t-SNE 1")
        ax.grid(alpha=0.2)
        apply_axis_font(ax, font_size, title_font_size)
    axes[0].set_ylabel("t-SNE 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=subject, markerfacecolor=colors[subject], markersize=6)
        for subject in unique_subjects
    ]
    axes[-1].legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.88,
        fontsize=legend_font_size,
        borderpad=0.4,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    fig.suptitle("WESAD 1D-CNN features by label, colored by subject", fontsize=title_font_size + 2)
    fig.tight_layout(rect=[0, 0, 1.0, 0.94])
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_all_label_plot(embedding, labels, output_path, point_size, font_size, legend_font_size, title_font_size):
    colors = {0: "#4C78A8", 1: "#E45756", 2: "#54A24B"}
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for label in sorted(np.unique(labels).astype(int)):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=0.7,
            color=colors.get(label, None),
            label=f"{label}: {LABEL_NAMES.get(label, str(label))}",
            linewidths=0,
        )
    ax.set_title("WESAD 1D-CNN features colored by emotion label")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    apply_axis_font(ax, font_size, title_font_size)
    ax.legend(frameon=False, fontsize=legend_font_size)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def save_all_subject_plot(embedding, subjects, output_path, point_size, font_size, legend_font_size, title_font_size):
    colors = subject_color_map(subjects)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for subject in sorted(colors, key=lambda value: int(value[1:])):
        mask = subjects == subject
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=point_size,
            alpha=0.7,
            color=colors[subject],
            label=subject,
            linewidths=0,
        )
    ax.set_title("WESAD 1D-CNN features colored by subject")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    apply_axis_font(ax, font_size, title_font_size)
    ax.legend(frameon=False, ncol=3, fontsize=legend_font_size)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary(
    path,
    args,
    checkpoint,
    subjects,
    labels,
    sample_counts,
    perplexity,
    max_per_subject_label,
    point_size,
    font_size,
    legend_font_size,
    title_font_size,
    outputs,
):
    lines = [
        "# WESAD Subject t-SNE Summary",
        "",
        "## Setup",
        "",
        f"- Feature extractor checkpoint: `{checkpoint}`",
        f"- Checkpoint fold: `{args.target_domain}`",
        f"- Number of subjects: {len(subjects)}",
        f"- Number of sampled windows: {len(labels)}",
        f"- t-SNE perplexity: {perplexity:.2f}",
        f"- Max samples per subject and label: {max_per_subject_label}",
        f"- Scatter point size: {point_size}",
        f"- Font size: {font_size}",
        f"- Legend font size: {legend_font_size}",
        f"- Title font size: {title_font_size}",
        "",
        "This visualization uses one fixed LOSO-trained 1D-CNN as the feature extractor, so all windows are embedded in the same 64-dimensional feature space before t-SNE.",
        "",
        "## Sample Counts",
        "",
        "| Subject | baseline | stress | amusement | total |",
        "|---|---:|---:|---:|---:|",
    ]
    for subject in sorted(sample_counts, key=lambda value: int(value[1:])):
        counts = sample_counts[subject]
        total = sum(counts.values())
        lines.append(
            f"| {subject} | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {total} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- Label-wise subject-colored t-SNE: `{outputs['by_label']}`",
            f"- All-window subject-colored t-SNE: `{outputs['by_subject']}`",
            f"- All-window label-colored t-SNE: `{outputs['by_label_all']}`",
            "",
            "## Interpretation for Paper",
            "",
            "`t-SNE 1` and `t-SNE 2` are the two coordinates of the t-SNE embedding. They do not correspond to physical sensor axes; only local neighborhood structure in the projected feature space should be interpreted qualitatively.",
            "",
            "If subject-colored clusters remain separated within each emotion-label panel, the visualization supports the claim that physiological features contain strong subject-specific variation even under the same emotion label. This should be treated as qualitative evidence motivating personalized adaptation, not as a standalone quantitative proof.",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    cli_args = parse_args()
    set_seed(cli_args.seed)
    args = build_args(cli_args)
    device = get_device(getattr(args, "device", "auto"))
    model, checkpoint = load_feature_model(args, device)
    subjects = resolve_subjects(args, cli_args)
    rng = np.random.default_rng(cli_args.seed)

    x_parts = []
    y_parts = []
    subject_parts = []
    sample_counts = {}
    for subject in subjects:
        subject_data = load_or_preprocess_subject(args, subject)
        x_sub, y_sub, subj_sub = sample_subject_windows(
            subject_data,
            subject,
            cli_args.max_per_subject_label,
            rng,
        )
        x_parts.append(x_sub)
        y_parts.append(y_sub)
        subject_parts.append(subj_sub)
        sample_counts[subject] = {
            int(label): int(np.sum(y_sub == label))
            for label in sorted(np.unique(y_sub).astype(int))
        }

    x_data = np.vstack(x_parts)
    y_data = np.concatenate(y_parts)
    subject_data = np.concatenate(subject_parts)
    features, labels, subjects_arr = extract_features(
        model,
        x_data,
        y_data,
        subject_data,
        cli_args.batch_size,
        getattr(args, "label_mode", "binary"),
        device,
    )
    embedding, used_perplexity = compute_embedding(features, cli_args.seed, cli_args.perplexity)

    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = os.path.join(cli_args.out_path, "wesad", "subject_tsne", current_time)
    os.makedirs(out_dir, exist_ok=True)
    ext = cli_args.figure_format
    outputs = {
        "by_label": os.path.join(out_dir, f"tsne_by_label_subject_color.{ext}"),
        "by_subject": os.path.join(out_dir, f"tsne_all_subject_color.{ext}"),
        "by_label_all": os.path.join(out_dir, f"tsne_all_label_color.{ext}"),
    }
    save_by_label_subject_plot(
        embedding,
        labels,
        subjects_arr,
        outputs["by_label"],
        cli_args.point_size,
        cli_args.font_size,
        cli_args.legend_font_size,
        cli_args.title_font_size,
    )
    save_all_subject_plot(
        embedding,
        subjects_arr,
        outputs["by_subject"],
        cli_args.point_size,
        cli_args.font_size,
        cli_args.legend_font_size,
        cli_args.title_font_size,
    )
    save_all_label_plot(
        embedding,
        labels,
        outputs["by_label_all"],
        cli_args.point_size,
        cli_args.font_size,
        cli_args.legend_font_size,
        cli_args.title_font_size,
    )

    summary_path = os.path.join(out_dir, "subject_tsne_summary.md")
    write_summary(
        summary_path,
        args,
        checkpoint,
        subjects,
        labels,
        sample_counts,
        used_perplexity,
        cli_args.max_per_subject_label,
        cli_args.point_size,
        cli_args.font_size,
        cli_args.legend_font_size,
        cli_args.title_font_size,
        outputs,
    )

    print(f"Output directory: {out_dir}")
    print(f"Summary: {summary_path}")
    for output in outputs.values():
        print(f"Figure: {output}")


if __name__ == "__main__":
    main()
