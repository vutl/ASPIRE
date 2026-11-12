#!/usr/bin/env python3
"""Plot Full vs No-AV confusion matrices and row-normalized delta."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    LABEL_IDS,
    LABEL_NAMES,
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
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


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    full = aggregate_cached(dirs["cache"], "full", args.folds)
    noav = aggregate_cached(dirs["cache"], "noav", args.folds)

    cm_full = confusion_matrix(full["y_true"], full["y_pred"], labels=LABEL_IDS)
    cm_noav = confusion_matrix(noav["y_true"], noav["y_pred"], labels=LABEL_IDS)
    cm_delta = cm_full - cm_noav

    np.save(dirs["tables"] / "confusion_full_counts.npy", cm_full)
    np.save(dirs["tables"] / "confusion_noav_counts.npy", cm_noav)
    np.save(dirs["tables"] / "confusion_delta_counts.npy", cm_delta)

    row_norm_full = cm_full.astype(float) / (cm_full.sum(axis=1, keepdims=True) + 1e-8)
    row_norm_noav = cm_noav.astype(float) / (cm_noav.sum(axis=1, keepdims=True) + 1e-8)
    row_norm_delta = row_norm_full - row_norm_noav

    save_heatmap(cm_full, dirs["plots"] / "confusion_full_counts.pdf", "Full ASPIRE", "d")
    save_heatmap(cm_noav, dirs["plots"] / "confusion_noav_counts.pdf", "No-AV-Sup", "d")
    save_heatmap(cm_delta, dirs["plots"] / "confusion_delta_counts.pdf", "Delta (Full - No-AV)", "d", center=0, cmap="coolwarm")
    save_heatmap(row_norm_full, dirs["plots"] / "confusion_full_row_norm.pdf", "Full ASPIRE (row-normalized)", ".3f")
    save_heatmap(row_norm_noav, dirs["plots"] / "confusion_noav_row_norm.pdf", "No-AV-Sup (row-normalized)", ".3f")
    save_heatmap(
        row_norm_delta,
        dirs["plots"] / "confusion_delta_row_norm.pdf",
        "Delta row-normalized (Full - No-AV)",
        ".3f",
        center=0,
        cmap="coolwarm",
    )

    print("Saved confusion matrices to:")
    print(dirs["plots"] / "confusion_full_counts.pdf")
    print(dirs["plots"] / "confusion_delta_row_norm.pdf")


if __name__ == "__main__":
    main()
