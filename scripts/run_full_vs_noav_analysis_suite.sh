#!/usr/bin/env bash
set -euo pipefail

FULL_DIR="${FULL_DIR:-checkpoints/method2_kfold_wpro}"
NOAV_DIR="${NOAV_DIR:-checkpoints/method2_kfold_wpro_ablation_no_avsup}"
CSV_PATH="${CSV_PATH:-datasets/IEMOCAP_KFOLD/tenfolds/all.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-results/full_vs_noav_analysis}"
DEVICE="${DEVICE:-cuda:2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WHICH="${WHICH:-best}"

COMMON_ARGS=(
  --full_dir "$FULL_DIR"
  --noav_dir "$NOAV_DIR"
  --csv "$CSV_PATH"
  --output "$OUTPUT_DIR"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --which "$WHICH"
  --folds 1 2 3 4 5
)

echo "[1/4] Compare main metrics"
python -u scripts/compare_main_metrics.py "${COMMON_ARGS[@]}"

echo "[2/4] Plot confusion matrices"
python -u scripts/plot_confusion_full_vs_noav.py "${COMMON_ARGS[@]}"

echo "[3/4] Plot per-class F1 delta"
python -u scripts/plot_per_class_f1_delta.py "${COMMON_ARGS[@]}"

echo "[4/4] Plot UMAP/t-SNE sweeps"
python -u scripts/plot_umap_fused_features.py \
  "${COMMON_ARGS[@]}" \
  --run_sweep \
  --random_state 42 \
  --umap_neighbors 15 \
  --umap_min_dist 0.1 \
  --umap_neighbors_list 15 30 \
  --umap_min_dist_list 0.1 0.3 \
  --tsne_perplexity 30 \
  --tsne_perplexity_list 10 30 50 \
  --tsne_pca_dim 50 \
  --tsne_max_iter 1000

echo "Done. Outputs are under: $OUTPUT_DIR"
