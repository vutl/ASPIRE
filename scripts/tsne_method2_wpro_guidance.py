#!/usr/bin/env python3
"""
Run Method2 W-PRO inference in two modes (with/without AV guidance) and plot t-SNE
trajectories of pre-guidance vs post-guidance representations.

- Loads a single checkpoint (per-fold) and IEMOCAP test split.
- Computes metrics for both modes to quantify dependence on AV refinement.
- Extracts paired feature vectors for t-SNE: Vector A (pre-guidance, AV masked)
  and Vector B (post-guidance, full model).
- Saves numpy dumps and a PNG scatter+arrow plot.

Example:
  python3 scripts/tsne_method2_wpro_guidance.py \
    --checkpoints_dir checkpoints/method2_kfold_wpro \
    --config configs/method2_wpro_config.yaml \
    --csv datasets/IEMOCAP_KFOLD/IEMOCAP_all.csv \
    --fold 5 --which best --device cuda:0 \
    --classes 0 3 --max_samples 150 --perplexity 30 \
    --out_prefix logs/tsne_method2_wpro_guidance_fold5
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_folds(csv_path):
    df = pd.read_csv(csv_path)
    max_fold = df["fold"].max()
    if max_fold == 10:
        fold_to_session = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}
        df["session"] = df["fold"].map(fold_to_session)
    else:
        df["session"] = df["fold"]

    folds = []
    for s in range(1, 6):
        folds.append({"session": s, "test": df[df["session"] == s], "train_val": df[df["session"] != s]})
    return folds


def compute_ccc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def load_checkpoint(model, ckpt_path, device):
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)


def evaluate_with_and_without_guidance(model, loader, device, max_samples=None, collect_features=True):
    model.eval()
    all_labels = []
    all_arousal_true = []
    all_valence_true = []
    logits_full = []
    logits_masked = []
    feats_post = []
    feats_pre = []
    arousal_full = []
    valence_full = []
    arousal_masked = []
    valence_masked = []

    with torch.no_grad():
        seen = 0
        for batch in tqdm(loader, desc="Inference (full vs masked)"):
            audio = batch["audio"].to(device)
            text_input_ids = batch["text_input_ids"].to(device)
            text_attention_mask = batch["text_attention_mask"].to(device)
            labels = batch["emotion_label"].to(device)
            arousal_true = batch["arousal_value"].to(device)
            valence_true = batch["valence_value"].to(device)

            out_full = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask,
                use_guidance=True,
                return_intermediate=collect_features,
            )
            out_masked = model(
                audio_input=audio,
                text_input=text_input_ids,
                text_attention_mask=text_attention_mask,
                use_guidance=False,
                return_intermediate=collect_features,
            )

            all_labels.append(labels.cpu())
            logits_full.append(out_full["emotion_logits"].cpu())
            logits_masked.append(out_masked["emotion_logits"].cpu())
            arousal_full.append(out_full["arousal_value"].cpu())
            valence_full.append(out_full["valence_value"].cpu())
            arousal_masked.append(out_masked["arousal_value"].cpu())
            valence_masked.append(out_masked["valence_value"].cpu())
            all_arousal_true.append(arousal_true.cpu())
            all_valence_true.append(valence_true.cpu())

            if collect_features:
                # Use penultimate fused features (decision space) for t-SNE
                feats_post.append(out_full["fused_features"].cpu())
                feats_pre.append(out_masked["fused_features"].cpu())
                
                # Debug: check if guidance masking actually works
                if len(feats_post) == 1:  # First batch
                    diff = torch.mean(torch.abs(out_full["fused_features"] - out_masked["fused_features"]))
                    print(f"DEBUG: Mean abs diff between full/masked features: {diff.item():.6f}")

            seen += labels.size(0)
            if max_samples is not None and max_samples > 0 and seen >= max_samples:
                break

    labels = torch.cat(all_labels)
    logits_full = torch.cat(logits_full)
    logits_masked = torch.cat(logits_masked)
    arousal_full = torch.cat(arousal_full).view(-1).numpy()
    valence_full = torch.cat(valence_full).view(-1).numpy()
    arousal_masked = torch.cat(arousal_masked).view(-1).numpy()
    valence_masked = torch.cat(valence_masked).view(-1).numpy()
    arousal_true = torch.cat(all_arousal_true).view(-1).numpy()
    valence_true = torch.cat(all_valence_true).view(-1).numpy()

    preds_full = torch.argmax(logits_full, dim=1)
    preds_masked = torch.argmax(logits_masked, dim=1)

    def _metrics(preds, labels_tensor, arousal_pred, valence_pred):
        labels_np = labels_tensor.numpy()
        accuracy = (preds == labels_tensor).float().mean().item()
        precision_macro = precision_score(labels_np, preds.numpy(), average="macro", zero_division=0)
        recall_macro = recall_score(labels_np, preds.numpy(), average="macro", zero_division=0)
        f1_macro = f1_score(labels_np, preds.numpy(), average="macro")
        f1_weighted = f1_score(labels_np, preds.numpy(), average="weighted")
        arousal_ccc = compute_ccc(arousal_true, arousal_pred)
        valence_ccc = compute_ccc(valence_true, valence_pred)
        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "arousal_ccc": arousal_ccc,
            "valence_ccc": valence_ccc,
            "avg_ccc": 0.5 * (arousal_ccc + valence_ccc),
        }

    metrics_full = _metrics(preds_full, labels, arousal_full, valence_full)
    metrics_masked = _metrics(preds_masked, labels, arousal_masked, valence_masked)

    features_pre = torch.cat(feats_pre).numpy() if feats_pre else None
    features_post = torch.cat(feats_post).numpy() if feats_post else None

    return labels.numpy(), features_pre, features_post, metrics_full, metrics_masked


def _filter_by_classes(features_pre, features_post, labels, classes):
    idx = [i for i, lbl in enumerate(labels) if (classes is None or lbl in classes)]
    if len(idx) == 0:
        return None, None, None
    return features_pre[idx], features_post[idx], labels[idx]


def _fit_tsne(combined, perplexity):
    try:
        return TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42).fit_transform(combined)
    except TypeError:
        return TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42).fit_transform(combined)


def run_tsne_and_plot(features_pre, features_post, labels, classes, perplexity, out_png):
    filtered = _filter_by_classes(features_pre, features_post, labels, classes)
    if filtered[0] is None:
        print("No samples matched the requested classes; skipping plot.")
        return

    features_pre, features_post, labels = filtered

    # Fit TSNE on combined set to keep spaces aligned
    combined = np.concatenate([features_pre, features_post], axis=0)
    emb = _fit_tsne(combined, perplexity)
    emb_pre = emb[: len(features_pre)]
    emb_post = emb[len(features_pre) :]

    colors = {0: "red", 1: "purple", 2: "gray", 3: "blue"}
    plt.figure(figsize=(8, 6))

    # Plot pre-guidance points
    for c in sorted(set(labels)):
        mask = labels == c
        plt.scatter(
            emb_pre[mask, 0],
            emb_pre[mask, 1],
            marker="o",
            facecolors="none",
            edgecolors=colors.get(int(c), "black"),
            label=f"Class {c} (pre)",
            alpha=0.7,
        )

    # Plot post-guidance points
    for c in sorted(set(labels)):
        mask = labels == c
        plt.scatter(
            emb_post[mask, 0],
            emb_post[mask, 1],
            marker="*",
            color=colors.get(int(c), "black"),
            label=f"Class {c} (post)",
            alpha=0.85,
        )

    # Arrows showing displacement per sample
    for i in range(len(labels)):
        plt.arrow(
            emb_pre[i, 0],
            emb_pre[i, 1],
            emb_post[i, 0] - emb_pre[i, 0],
            emb_post[i, 1] - emb_pre[i, 1],
            color=colors.get(int(labels[i]), "black"),
            alpha=0.25,
            length_includes_head=True,
            head_width=0.5,
        )

    plt.title("Method2 W-PRO: Pre-guidance vs Post-guidance")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {out_png}")


def run_side_by_side(features_pre, features_post, labels, classes, perplexity, out_png):
    filtered = _filter_by_classes(features_pre, features_post, labels, classes)
    if filtered[0] is None:
        print("No samples matched the requested classes; skipping side-by-side plot.")
        return
    features_pre, features_post, labels = filtered

    combined = np.concatenate([features_pre, features_post], axis=0)
    emb = _fit_tsne(combined, perplexity)
    emb_pre = emb[: len(features_pre)]
    emb_post = emb[len(features_pre) :]

    colors = {0: "red", 1: "purple", 2: "gray", 3: "blue"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for c in sorted(set(labels)):
        mask = labels == c
        axes[0].scatter(
            emb_pre[mask, 0],
            emb_pre[mask, 1],
            marker="o",
            facecolors="none",
            edgecolors=colors.get(int(c), "black"),
            label=f"Class {c} (pre)",
            alpha=0.8,
        )
    axes[0].set_title("Masked (no guidance)")

    for c in sorted(set(labels)):
        mask = labels == c
        axes[1].scatter(
            emb_post[mask, 0],
            emb_post[mask, 1],
            marker="*",
            color=colors.get(int(c), "black"),
            label=f"Class {c} (post)",
            alpha=0.8,
        )
    axes[1].set_title("Full (with guidance)")

    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved side-by-side plot to {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--fold", type=int, nargs="*", default=[5], help="Fold(s) to evaluate (1-5)")
    parser.add_argument("--which", type=str, choices=["best", "last"], default="best")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=200, help="Limit samples for plotting")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Class IDs to visualize (default: all classes)")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--out_prefix", type=str, default="logs/tsne_method2_wpro_guidance")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    folds = prepare_folds(args.csv)

    all_rows = []
    all_labels = []
    all_pre = []
    all_post = []

    for fold_idx in args.fold:
        if fold_idx < 1 or fold_idx > 5:
            print(f"Skipping invalid fold {fold_idx}")
            continue

        fold_data = folds[fold_idx - 1]
        test_df = fold_data["test"]

        test_dataset = IEMOCAPDataset(
            test_df,
            audio_max_length=cfg.get("data", {}).get("max_audio_length", 160000),
            tokenizer_name=cfg.get("model", {}).get("roberta_model_name", "roberta-base"),
            max_text_length=cfg.get("data", {}).get("max_text_length", 128),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        model = Method2CrossAttentionAV(
            num_emotions=cfg["model"].get("num_emotions", 4),
            wavlm_model_name=cfg["model"].get("wavlm_model_name"),
            wavlm_prosody_model_name=cfg["model"].get("wavlm_prosody_model_name"),
            roberta_model_name=cfg["model"].get("roberta_model_name", "roberta-base"),
            hidden_dim=cfg["model"].get("hidden_dim", 256),
            av_refinement_layers=cfg["model"].get("av_refinement_layers", 2),
            freeze_wavlm=cfg["model"].get("freeze_wavlm", False),
            freeze_roberta=cfg["model"].get("freeze_roberta", False),
            num_cross_attn_heads=cfg["model"].get("num_cross_attn_heads", 8),
            post_attention_layers=cfg["model"].get("post_attention_layers", 2),
            av_prediction_layers=cfg["model"].get("av_prediction_layers", 3),
        )

        ckpt_path = Path(args.checkpoints_dir) / f"fold_{fold_idx}" / ("best_model.pt" if args.which == "best" else "last_model.pt")
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path}, skip fold {fold_idx}")
            continue

        if args.dry_run:
            print(f"[DRY RUN] Fold {fold_idx}: would load {ckpt_path}, test samples {len(test_dataset)}")
            continue

        load_checkpoint(model, ckpt_path, device)
        model.to(device)

        labels, feats_pre, feats_post, metrics_full, metrics_masked = evaluate_with_and_without_guidance(
            model,
            test_loader,
            device,
            max_samples=args.max_samples,
            collect_features=True,
        )

        out_prefix = Path(args.out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Per-fold saves
        np.savez_compressed(
            f"{out_prefix}_fold{fold_idx}.npz",
            labels=labels,
            features_pre=feats_pre,
            features_post=feats_post,
        )
        import json

        with open(f"{out_prefix}_fold{fold_idx}_metrics.json", "w") as f:
            json.dump({
                "full": {k: float(v) for k, v in metrics_full.items()},
                "masked": {k: float(v) for k, v in metrics_masked.items()}
            }, f, indent=2)

        run_tsne_and_plot(
            feats_pre,
            feats_post,
            labels,
            classes=set(args.classes) if args.classes else None,
            perplexity=args.perplexity,
            out_png=f"{out_prefix}_fold{fold_idx}_tsne.png",
        )

        run_side_by_side(
            feats_pre,
            feats_post,
            labels,
            classes=set(args.classes) if args.classes else None,
            perplexity=args.perplexity,
            out_png=f"{out_prefix}_fold{fold_idx}_sidebyside.png",
        )

        row_full = {"fold": fold_idx, "mode": "full"}
        row_full.update(metrics_full)
        row_masked = {"fold": fold_idx, "mode": "masked"}
        row_masked.update(metrics_masked)
        all_rows.extend([row_full, row_masked])

        # Accumulate for aggregate plots
        all_labels.append(labels)
        all_pre.append(feats_pre)
        all_post.append(feats_post)

    if args.dry_run:
        return

    if not all_rows:
        print("No folds processed; exiting.")
        return

    # Save metrics CSV
    metrics_df = pd.DataFrame(all_rows)
    metrics_csv = f"{args.out_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics CSV to {metrics_csv}")

    # Aggregate plots across folds
    agg_labels = np.concatenate(all_labels)
    agg_pre = np.concatenate(all_pre)
    agg_post = np.concatenate(all_post)

    run_tsne_and_plot(
        agg_pre,
        agg_post,
        agg_labels,
        classes=set(args.classes) if args.classes else None,
        perplexity=args.perplexity,
        out_png=f"{args.out_prefix}_allfolds_tsne.png",
    )

    run_side_by_side(
        agg_pre,
        agg_post,
        agg_labels,
        classes=set(args.classes) if args.classes else None,
        perplexity=args.perplexity,
        out_png=f"{args.out_prefix}_allfolds_sidebyside.png",
    )


if __name__ == "__main__":
    main()
