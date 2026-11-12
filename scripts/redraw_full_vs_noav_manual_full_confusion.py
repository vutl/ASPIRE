#!/usr/bin/env python3
"""Redraw confusion and per-class F1 plots using a manually provided Full confusion matrix."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LABEL_NAMES = ["Happy", "Sad", "Neutral", "Angry"]

# User-provided best Full confusion matrix:
# rows=true, cols=pred, label order = [Happy, Sad, Neutral, Angry]
FULL_CM = np.array(
    [
        [1229, 87, 220, 100],
        [45, 850, 145, 44],
        [210, 180, 1213, 105],
        [51, 21, 127, 904],
    ],
    dtype=np.int64,
)


def save_heatmap(matrix, out_path, title, fmt, center=None, cmap="Blues"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap=cmap,
        center=center,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_side_by_side_heatmaps(left_matrix, right_matrix, out_path, left_title, right_title, fmt, cmap="Blues"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    vmax = max(float(np.max(left_matrix)), float(np.max(right_matrix)))
    for ax, matrix, title in zip(axes, [left_matrix, right_matrix], [left_title, right_title]):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            cmap=cmap,
            vmin=0.0 if fmt != "d" else None,
            vmax=vmax if fmt != "d" else None,
            ax=ax,
            cbar=False,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def per_class_f1_from_confusion(cm: np.ndarray) -> np.ndarray:
    f1s = []
    for idx in range(cm.shape[0]):
        tp = float(cm[idx, idx])
        fp = float(cm[:, idx].sum() - tp)
        fn = float(cm[idx, :].sum() - tp)
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return np.asarray(f1s, dtype=np.float64)


def main():
    base = Path("results/full_vs_noav_analysis")
    plots = base / "plots"
    tables = base / "tables"
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    noav_path = tables / "confusion_noav_counts.npy"
    if not noav_path.exists():
        raise FileNotFoundError(f"Missing No-AV confusion matrix at {noav_path}")
    cm_noav = np.load(noav_path)
    cm_full = FULL_CM
    cm_delta = cm_full - cm_noav

    np.save(tables / "confusion_full_counts.npy", cm_full)
    np.save(tables / "confusion_delta_counts.npy", cm_delta)

    row_norm_full = cm_full.astype(float) / (cm_full.sum(axis=1, keepdims=True) + 1e-8)
    row_norm_noav = cm_noav.astype(float) / (cm_noav.sum(axis=1, keepdims=True) + 1e-8)
    row_norm_delta = row_norm_full - row_norm_noav

    save_heatmap(cm_full, plots / "confusion_full_counts.pdf", "Full ASPIRE", "d")
    save_heatmap(cm_noav, plots / "confusion_noav_counts.pdf", "No-AV-Sup", "d")
    save_heatmap(cm_delta, plots / "confusion_delta_counts.pdf", "Delta (Full - No-AV)", "d", center=0, cmap="coolwarm")
    save_heatmap(row_norm_full, plots / "confusion_full_row_norm.pdf", "Full ASPIRE (row-normalized)", ".3f")
    save_heatmap(row_norm_noav, plots / "confusion_noav_row_norm.pdf", "No-AV-Sup (row-normalized)", ".3f")
    save_side_by_side_heatmaps(
        cm_full,
        cm_noav,
        plots / "confusion_side_by_side_counts.pdf",
        "Full ASPIRE",
        "No-AV-Sup",
        "d",
    )
    save_side_by_side_heatmaps(
        row_norm_full,
        row_norm_noav,
        plots / "confusion_side_by_side_row_norm.pdf",
        "Full ASPIRE (row-normalized)",
        "No-AV-Sup (row-normalized)",
        ".3f",
    )
    save_heatmap(
        row_norm_delta,
        plots / "confusion_delta_row_norm.pdf",
        "Delta row-normalized (Full - No-AV)",
        ".3f",
        center=0,
        cmap="coolwarm",
    )

    f1_full = per_class_f1_from_confusion(cm_full)
    f1_noav = per_class_f1_from_confusion(cm_noav)
    delta = f1_full - f1_noav

    df = pd.DataFrame(
        {
            "label": LABEL_NAMES,
            "f1_noav": f1_noav,
            "f1_full": f1_full,
            "delta_full_minus_noav": delta,
        }
    )
    df.to_csv(tables / "per_class_f1_delta.csv", index=False)

    x = np.arange(len(LABEL_NAMES))
    plt.figure(figsize=(7, 5))
    plt.bar(x - 0.18, f1_noav, width=0.36, label="No-AV-Sup")
    plt.bar(x + 0.18, f1_full, width=0.36, label="Full ASPIRE")
    plt.xticks(x, LABEL_NAMES)
    plt.ylabel("Per-class F1")
    plt.title("Per-class F1: Full vs No-AV")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(plots / "per_class_f1_comparison.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in delta]
    plt.bar(LABEL_NAMES, delta, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Delta F1 (Full - No-AV)")
    plt.title("Per-class F1 Delta")
    plt.tight_layout()
    plt.savefig(plots / "per_class_f1_delta.pdf")
    plt.close()

    print("Saved:")
    print(plots / "confusion_full_counts.pdf")
    print(plots / "confusion_full_row_norm.pdf")
    print(plots / "confusion_side_by_side_counts.pdf")
    print(plots / "confusion_side_by_side_row_norm.pdf")
    print(plots / "confusion_delta_counts.pdf")
    print(plots / "confusion_delta_row_norm.pdf")
    print(plots / "per_class_f1_comparison.pdf")
    print(plots / "per_class_f1_delta.pdf")


if __name__ == "__main__":
    main()
