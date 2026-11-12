#!/usr/bin/env python3
"""Plot per-class F1 comparison and delta for Full ASPIRE vs No-AV-Sup."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    LABEL_IDS,
    LABEL_NAMES,
    add_shared_args,
    aggregate_cached,
    ensure_cache_for_models,
    ensure_output_dirs,
)


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    full = aggregate_cached(dirs["cache"], "full", args.folds)
    noav = aggregate_cached(dirs["cache"], "noav", args.folds)

    f1_full = f1_score(full["y_true"], full["y_pred"], labels=LABEL_IDS, average=None)
    f1_noav = f1_score(noav["y_true"], noav["y_pred"], labels=LABEL_IDS, average=None)
    delta = f1_full - f1_noav

    df = pd.DataFrame(
        {
            "label": LABEL_NAMES,
            "f1_noav": f1_noav,
            "f1_full": f1_full,
            "delta_full_minus_noav": delta,
        }
    )
    df.to_csv(dirs["tables"] / "per_class_f1_delta.csv", index=False)

    x = np.arange(len(LABEL_NAMES))
    plt.figure(figsize=(7, 5))
    plt.bar(x - 0.18, f1_noav, width=0.36, label="No-AV-Sup")
    plt.bar(x + 0.18, f1_full, width=0.36, label="Full ASPIRE")
    plt.xticks(x, LABEL_NAMES)
    plt.ylabel("Per-class F1")
    plt.title("Per-class F1: Full vs No-AV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "per_class_f1_comparison.pdf")
    plt.close()

    plt.figure(figsize=(7, 5))
    colors = ["#2e7d32" if v >= 0 else "#c62828" for v in delta]
    plt.bar(LABEL_NAMES, delta, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Delta F1 (Full - No-AV)")
    plt.title("Per-class F1 Delta")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "per_class_f1_delta.pdf")
    plt.close()

    print("Saved:")
    print(dirs["tables"] / "per_class_f1_delta.csv")
    print(dirs["plots"] / "per_class_f1_delta.pdf")


if __name__ == "__main__":
    main()
