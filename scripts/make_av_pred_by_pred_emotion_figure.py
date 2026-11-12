#!/usr/bin/env python3
"""Make a paper-ready AV scatter colored by predicted emotion on predicted AV space."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import LABEL_NAMES, add_shared_args, aggregate_cached, ensure_cache_for_models, ensure_output_dirs


def label_names_from_ids(label_ids):
    return [LABEL_NAMES[int(i)] for i in label_ids]


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    parser.add_argument("--figures_dir", default="figures")
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ensure_cache_for_models(args)

    full = aggregate_cached(dirs["cache"], "full", args.folds)
    if full["arousal_pred"].size == 0 or full["valence_pred"].size == 0:
        raise RuntimeError("Full model cache is missing arousal/valence predictions.")

    labels_pred = label_names_from_ids(full["y_pred"])

    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    sns.scatterplot(
        x=full["arousal_pred"],
        y=full["valence_pred"],
        hue=labels_pred,
        hue_order=LABEL_NAMES,
        palette="tab10",
        s=14,
        linewidth=0,
        alpha=0.75,
        ax=ax,
    )
    lim = float(max(abs(full["arousal_pred"]).max(), abs(full["valence_pred"]).max()))
    lim = max(1.0, round(lim + 0.05, 2))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Arousal")
    ax.set_ylabel("Predicted Valence")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=True, fancybox=True)
    plt.tight_layout(pad=0.15)
    out_path = figures_dir / "av_pred_by_pred_emotion.pdf"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print(out_path)


if __name__ == "__main__":
    main()
