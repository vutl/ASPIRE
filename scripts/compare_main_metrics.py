#!/usr/bin/env python3
"""Compare 5-fold main metrics for Full ASPIRE vs No-AV-Sup."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.full_vs_noav_analysis_utils import (
    add_shared_args,
    aggregate_cached,
    compute_ccc,
    ensure_cache_for_models,
    ensure_output_dirs,
    load_cached_fold,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    add_shared_args(parser)
    args = parser.parse_args()

    dirs = ensure_output_dirs(args.output)
    ensure_cache_for_models(args)

    fold_rows = []
    for fold_id in args.folds:
        full = load_cached_fold(dirs["cache"], "full", fold_id)
        noav = load_cached_fold(dirs["cache"], "noav", fold_id)

        fold_rows.append(
            {
                "fold": fold_id,
                "acc_full": accuracy_score(full["y_true"], full["y_pred"]),
                "f1_macro_full": f1_score(full["y_true"], full["y_pred"], average="macro"),
                "f1_weighted_full": f1_score(full["y_true"], full["y_pred"], average="weighted"),
                "acc_noav": accuracy_score(noav["y_true"], noav["y_pred"]),
                "f1_macro_noav": f1_score(noav["y_true"], noav["y_pred"], average="macro"),
                "f1_weighted_noav": f1_score(noav["y_true"], noav["y_pred"], average="weighted"),
                "ccc_arousal_full": compute_ccc(full["arousal_true"], full["arousal_pred"]),
                "ccc_valence_full": compute_ccc(full["valence_true"], full["valence_pred"]),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(dirs["tables"] / "main_metrics_by_fold.csv", index=False)

    summary_rows = []
    for name, prefix in [("No-AV-Sup", "noav"), ("Full ASPIRE", "full")]:
        summary_rows.append(
            {
                "model": name,
                "acc_mean": fold_df[f"acc_{prefix}"].mean(),
                "acc_std": fold_df[f"acc_{prefix}"].std(ddof=1),
                "f1_macro_mean": fold_df[f"f1_macro_{prefix}"].mean(),
                "f1_macro_std": fold_df[f"f1_macro_{prefix}"].std(ddof=1),
                "f1_weighted_mean": fold_df[f"f1_weighted_{prefix}"].mean(),
                "f1_weighted_std": fold_df[f"f1_weighted_{prefix}"].std(ddof=1),
            }
        )

    delta_row = {
        "model": "Delta (Full - No-AV)",
        "acc_mean": fold_df["acc_full"].mean() - fold_df["acc_noav"].mean(),
        "acc_std": np.nan,
        "f1_macro_mean": fold_df["f1_macro_full"].mean() - fold_df["f1_macro_noav"].mean(),
        "f1_macro_std": np.nan,
        "f1_weighted_mean": fold_df["f1_weighted_full"].mean() - fold_df["f1_weighted_noav"].mean(),
        "f1_weighted_std": np.nan,
    }

    summary_df = pd.DataFrame(summary_rows + [delta_row])
    summary_df.to_csv(dirs["tables"] / "main_metrics_summary.csv", index=False)

    ccc_summary = pd.DataFrame(
        [
            {
                "model": "Full ASPIRE",
                "ccca_mean": fold_df["ccc_arousal_full"].mean(),
                "ccca_std": fold_df["ccc_arousal_full"].std(ddof=1),
                "cccv_mean": fold_df["ccc_valence_full"].mean(),
                "cccv_std": fold_df["ccc_valence_full"].std(ddof=1),
            }
        ]
    )
    ccc_summary.to_csv(dirs["tables"] / "full_ccc_summary.csv", index=False)

    agg_full = aggregate_cached(dirs["cache"], "full", args.folds)
    agg_noav = aggregate_cached(dirs["cache"], "noav", args.folds)
    aggregate_metrics = {
        "full": {
            "acc": accuracy_score(agg_full["y_true"], agg_full["y_pred"]),
            "f1_macro": f1_score(agg_full["y_true"], agg_full["y_pred"], average="macro"),
            "f1_weighted": f1_score(agg_full["y_true"], agg_full["y_pred"], average="weighted"),
            "ccca": compute_ccc(agg_full["arousal_true"], agg_full["arousal_pred"]),
            "cccv": compute_ccc(agg_full["valence_true"], agg_full["valence_pred"]),
        },
        "noav": {
            "acc": accuracy_score(agg_noav["y_true"], agg_noav["y_pred"]),
            "f1_macro": f1_score(agg_noav["y_true"], agg_noav["y_pred"], average="macro"),
            "f1_weighted": f1_score(agg_noav["y_true"], agg_noav["y_pred"], average="weighted"),
        },
    }
    write_json(dirs["tables"] / "aggregate_metrics.json", aggregate_metrics)

    print("Saved:")
    print(dirs["tables"] / "main_metrics_by_fold.csv")
    print(dirs["tables"] / "main_metrics_summary.csv")
    print(dirs["tables"] / "full_ccc_summary.csv")


if __name__ == "__main__":
    main()
