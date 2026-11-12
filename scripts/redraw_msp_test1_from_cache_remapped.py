#!/usr/bin/env python3
"""Redraw MSP test1 AV scatter figures from cache with old->new label remap."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import LABEL_NAMES, aggregate_cached, ensure_output_dirs


LABEL_REMAP_OLD_TO_NEW = {
    0: 3,  # old ang -> new angry
    1: 0,  # old hap -> new happy
    2: 2,  # old neu -> new neutral
    3: 1,  # old sad -> new sad
}


def label_names_from_ids(label_ids):
    return [LABEL_NAMES[int(i)] for i in label_ids]


def remap_labels(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.asarray([LABEL_REMAP_OLD_TO_NEW[int(x)] for x in y_pred], dtype=np.int64)


def map_to_minus1_1(x, source_min=None, source_max=None, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    if source_min is None or source_max is None or np.isclose(source_min, source_max):
        mx = np.max(np.abs(x)) if x.size else 1.0
        if mx < 1e-12:
            return np.zeros_like(x)
        return np.clip(x / mx, -1.0 + eps, 1.0 - eps)
    mapped = -1.0 + 2.0 * (x - source_min) / (source_max - source_min + 1e-12)
    return np.clip(mapped, -1.0 + eps, 1.0 - eps)


def compute_symmetric_limit(arousal, valence, min_lim: float = 1.0, pad: float = 0.05) -> float:
    lim = float(max(np.max(np.abs(arousal)), np.max(np.abs(valence))))
    return max(min_lim, round(lim + pad, 2))


def compute_msp_plot_limit(arousal, valence, base_lim: float = 1.0, pad: float = 0.04) -> float:
    """Keep MSP plots visually in [-1, 1] space but avoid points touching the frame."""
    lim = float(max(np.max(np.abs(arousal)), np.max(np.abs(valence))))
    return max(base_lim, round(lim + pad, 2))


def plot_single(arousal, valence, labels, title, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
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
    )
    lim = compute_msp_plot_limit(arousal, valence)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Arousal")
    ax.set_ylabel("Predicted Valence")
    ax.set_title(title, fontsize=13)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    plt.tight_layout(pad=0.15, rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_pair(left, right, left_title: str, right_title: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))
    legend_handles = legend_labels = None
    for ax, item, title, show_legend in [
        (axes[0], left, left_title, False),
        (axes[1], right, right_title, True),
    ]:
        sns.scatterplot(
            x=item["arousal_pred"],
            y=item["valence_pred"],
            hue=label_names_from_ids(item["y_pred"]),
            hue_order=LABEL_NAMES,
            palette="tab10",
            s=14,
            linewidth=0,
            alpha=0.75,
            ax=ax,
            legend=show_legend,
        )
        if title.startswith("MSP"):
            lim = compute_msp_plot_limit(item["arousal_pred"], item["valence_pred"])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        else:
            lim = compute_symmetric_limit(item["arousal_pred"], item["valence_pred"])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted Arousal")
        ax.set_ylabel("Predicted Valence")
        ax.set_title(title, fontsize=13)
        if ax.legend_ is not None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend_.remove()
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            frameon=False,
            fontsize=10,
            handletextpad=0.4,
            columnspacing=0.9,
        )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.11, wspace=0.22)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_pair_vertical(left, right, top_title: str, bottom_title: str, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 12.2))
    legend_handles = legend_labels = None
    for ax, item, title, show_legend in [
        (axes[0], left, top_title, False),
        (axes[1], right, bottom_title, False),
    ]:
        sns.scatterplot(
            x=item["arousal_pred"],
            y=item["valence_pred"],
            hue=label_names_from_ids(item["y_pred"]),
            hue_order=LABEL_NAMES,
            palette="tab10",
            s=14,
            linewidth=0,
            alpha=0.75,
            ax=ax,
            legend=show_legend,
        )
        if title.startswith("MSP"):
            lim = compute_msp_plot_limit(item["arousal_pred"], item["valence_pred"])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        else:
            lim = compute_symmetric_limit(item["arousal_pred"], item["valence_pred"])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Predicted Arousal")
        ax.set_ylabel("Predicted Valence")
        ax.set_title(title, fontsize=13)

    legend_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=sns.color_palette("tab10", n_colors=4)[i], markersize=6)
        for i in range(len(LABEL_NAMES))
    ]
    legend_labels = LABEL_NAMES
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        frameon=False,
        fontsize=10,
        handletextpad=0.4,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.07, hspace=0.28)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def main():
    root = Path("/home4/quanpn/ICME2026/AVD_Test")
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dirs = ensure_output_dirs(root / "results" / "full_vs_noav_analysis")
    full = aggregate_cached(dirs["cache"], "full", [1, 2, 3, 4, 5])

    cache_path = root / "results" / "full_vs_noav_analysis" / "tables" / "msp_av_pred_test1.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing MSP cache: {cache_path}")
    d = np.load(cache_path)
    msp = {k: d[k] for k in d.files}
    msp["y_pred"] = remap_labels(msp["y_pred"])
    msp["arousal_pred"] = map_to_minus1_1(msp["arousal_pred"])
    msp["valence_pred"] = map_to_minus1_1(msp["valence_pred"])

    plot_single(
        msp["arousal_pred"],
        msp["valence_pred"],
        label_names_from_ids(msp["y_pred"]),
        "MSP Test1",
        figures_dir / "msp_av_pred_by_pred_emotion_test1.pdf",
    )
    plot_pair(
        full,
        msp,
        "IEMOCAP",
        "MSP Test1",
        figures_dir / "av_pred_by_pred_emotion_iemocap_msp_test1_side_by_side.pdf",
    )
    plot_pair_vertical(
        full,
        msp,
        "IEMOCAP",
        "MSP Test1",
        figures_dir / "av_pred_by_pred_emotion_iemocap_msp_test1_vertical.pdf",
    )
    print(figures_dir / "msp_av_pred_by_pred_emotion_test1.pdf")
    print(figures_dir / "av_pred_by_pred_emotion_iemocap_msp_test1_side_by_side.pdf")
    print(figures_dir / "av_pred_by_pred_emotion_iemocap_msp_test1_vertical.pdf")


if __name__ == "__main__":
    main()
