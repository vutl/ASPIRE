#!/usr/bin/env python3
"""Plot 5-fold aggregated arousal/valence spaces for Full ASPIRE."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    LABEL_NAMES,
    add_shared_args,
    aggregate_cached,
    compute_ccc,
    ensure_cache_for_models,
    ensure_output_dirs,
)


def label_names_from_ids(label_ids: np.ndarray):
    return np.array([LABEL_NAMES[int(i)] for i in label_ids])


def save_av_scatter(
    arousal,
    valence,
    labels,
    out_path,
    title,
    xlabel,
    ylabel,
):
    plt.figure(figsize=(6.5, 6))
    sns.scatterplot(
        x=arousal,
        y=valence,
        hue=labels,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=14,
        linewidth=0,
        alpha=0.75,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_regression_scatter(y_true, y_pred, out_path, title, axis_label):
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.figure(figsize=(5.8, 5.6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.35, edgecolors="none")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
    plt.xlabel(f"True {axis_label}")
    plt.ylabel(f"Predicted {axis_label}")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_regression_side_by_side(arousal_true, arousal_pred, valence_true, valence_pred, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, y_true, y_pred, axis_label in [
        (axes[0], arousal_true, arousal_pred, "Arousal"),
        (axes[1], valence_true, valence_pred, "Valence"),
    ]:
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.scatter(y_true, y_pred, s=10, alpha=0.35, edgecolors="none")
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel(f"True {axis_label}")
        ax.set_ylabel(f"Predicted {axis_label}")
        ax.set_title(axis_label)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_av_side_by_side(
    arousal_left,
    valence_left,
    labels_left,
    title_left,
    arousal_right,
    valence_right,
    labels_right,
    title_right,
    out_path,
    xlabel,
    ylabel,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8))
    for ax, arousal, valence, labels, title, show_legend in [
        (axes[0], arousal_left, valence_left, labels_left, title_left, False),
        (axes[1], arousal_right, valence_right, labels_right, title_right, True),
    ]:
        sns.scatterplot(
            x=arousal,
            y=valence,
            hue=labels,
            hue_order=LABEL_NAMES,
            palette="tab10",
            s=14,
            linewidth=0,
            alpha=0.75,
            ax=ax,
            legend=show_legend,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    full = aggregate_cached(dirs["cache"], "full", args.folds)
    noav = aggregate_cached(dirs["cache"], "noav", args.folds)

    if full["arousal_pred"].size == 0 or full["valence_pred"].size == 0:
        raise RuntimeError("Full model cache is missing arousal/valence predictions.")

    labels_true = label_names_from_ids(full["y_true"])
    labels_pred = label_names_from_ids(full["y_pred"])
    labels_pred_noav = label_names_from_ids(noav["y_pred"])

    save_av_scatter(
        full["arousal_true"],
        full["valence_true"],
        labels_true,
        dirs["plots"] / "av_true_by_true_emotion.pdf",
        "Ground-Truth Arousal/Valence by Emotion",
        "True Arousal",
        "True Valence",
    )
    save_av_scatter(
        full["arousal_pred"],
        full["valence_pred"],
        labels_true,
        dirs["plots"] / "av_pred_by_true_emotion.pdf",
        "Predicted Arousal/Valence colored by True Emotion",
        "Predicted Arousal",
        "Predicted Valence",
    )
    save_av_scatter(
        full["arousal_pred"],
        full["valence_pred"],
        labels_pred,
        dirs["plots"] / "av_pred_by_pred_emotion.pdf",
        "Predicted Arousal/Valence colored by Predicted Emotion",
        "Predicted Arousal",
        "Predicted Valence",
    )
    save_av_scatter(
        full["arousal_true"],
        full["valence_true"],
        labels_pred,
        dirs["plots"] / "av_true_by_pred_emotion_full.pdf",
        "Ground-Truth Arousal/Valence colored by Predicted Emotion (Full)",
        "True Arousal",
        "True Valence",
    )
    save_av_scatter(
        noav["arousal_true"],
        noav["valence_true"],
        labels_pred_noav,
        dirs["plots"] / "av_true_by_pred_emotion_noav.pdf",
        "Ground-Truth Arousal/Valence colored by Predicted Emotion (No-AV-Sup)",
        "True Arousal",
        "True Valence",
    )
    save_av_side_by_side(
        full["arousal_true"],
        full["valence_true"],
        labels_pred,
        "Full ASPIRE",
        noav["arousal_true"],
        noav["valence_true"],
        labels_pred_noav,
        "No-AV-Sup",
        dirs["plots"] / "av_true_by_pred_emotion_side_by_side.pdf",
        "True Arousal",
        "True Valence",
    )

    save_regression_scatter(
        full["arousal_true"],
        full["arousal_pred"],
        dirs["plots"] / "arousal_true_vs_pred.pdf",
        "Arousal: True vs Predicted",
        "Arousal",
    )
    save_regression_scatter(
        full["valence_true"],
        full["valence_pred"],
        dirs["plots"] / "valence_true_vs_pred.pdf",
        "Valence: True vs Predicted",
        "Valence",
    )
    save_regression_side_by_side(
        full["arousal_true"],
        full["arousal_pred"],
        full["valence_true"],
        full["valence_pred"],
        dirs["plots"] / "av_true_vs_pred_side_by_side.pdf",
    )

    summary_df = pd.DataFrame(
        [
            {
                "num_samples": int(full["y_true"].shape[0]),
                "arousal_ccc": compute_ccc(full["arousal_true"], full["arousal_pred"]),
                "valence_ccc": compute_ccc(full["valence_true"], full["valence_pred"]),
                "arousal_true_min": float(np.min(full["arousal_true"])),
                "arousal_true_max": float(np.max(full["arousal_true"])),
                "valence_true_min": float(np.min(full["valence_true"])),
                "valence_true_max": float(np.max(full["valence_true"])),
                "arousal_pred_min": float(np.min(full["arousal_pred"])),
                "arousal_pred_max": float(np.max(full["arousal_pred"])),
                "valence_pred_min": float(np.min(full["valence_pred"])),
                "valence_pred_max": float(np.max(full["valence_pred"])),
            }
        ]
    )
    summary_df.to_csv(dirs["tables"] / "full_av_space_summary.csv", index=False)

    print("Saved:")
    print(dirs["plots"] / "av_true_by_true_emotion.pdf")
    print(dirs["plots"] / "av_pred_by_true_emotion.pdf")
    print(dirs["plots"] / "av_pred_by_pred_emotion.pdf")
    print(dirs["plots"] / "av_true_by_pred_emotion_full.pdf")
    print(dirs["plots"] / "av_true_by_pred_emotion_noav.pdf")
    print(dirs["plots"] / "av_true_by_pred_emotion_side_by_side.pdf")
    print(dirs["plots"] / "av_true_vs_pred_side_by_side.pdf")
    print(dirs["tables"] / "full_av_space_summary.csv")


if __name__ == "__main__":
    main()
