#!/usr/bin/env python3
"""Shared utilities for Full ASPIRE vs No-AV-Supervision analysis."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.iemocap_dataset import IEMOCAPDataset, collate_fn
from models.method2_crossattn_av import Method2CrossAttentionAV
from models.method2_crossattn_av_ablation_no_avsup import (
    Method2CrossAttentionAV_AblationNoAVSup,
)

LABEL_IDS = [0, 1, 2, 3]
LABEL_NAMES = ["Happy", "Sad", "Neutral", "Angry"]


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--full_dir", required=True, help="Checkpoint dir for Full ASPIRE")
    parser.add_argument("--noav_dir", required=True, help="Checkpoint dir for No-AV-Sup ablation")
    parser.add_argument("--csv", required=True, help="IEMOCAP CSV with fold/session info")
    parser.add_argument("--output", default="results/full_vs_noav_analysis")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_audio_length", type=int, default=160000)
    parser.add_argument("--max_text_length", type=int, default=128)
    parser.add_argument("--roberta_model_name", default="roberta-base")
    parser.add_argument("--wavlm_model_name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--wavlm_prosody_model_name", default="Orange/Speaker-wavLM-pro")
    parser.add_argument("--num_emotions", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_cross_attn_heads", type=int, default=8)
    parser.add_argument("--post_attention_layers", type=int, default=2)
    parser.add_argument("--av_prediction_layers", type=int, default=3)
    parser.add_argument("--av_refinement_layers", type=int, default=2)
    parser.add_argument("--freeze_wavlm", action="store_true")
    parser.add_argument("--freeze_roberta", action="store_true")
    parser.add_argument(
        "--which",
        choices=["best", "last"],
        default="best",
        help="Prefer best_model.pt or last_model.pt when both exist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run inference even if per-fold cache already exists",
    )
    return parser


def ensure_output_dirs(output_dir: str | Path) -> Dict[str, Path]:
    out = Path(output_dir)
    cache = out / "cache"
    plots = out / "plots"
    tables = out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return {"root": out, "cache": cache, "plots": plots, "tables": tables}


def get_fold_test_df(csv_path: str, fold_id: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df["fold"].max() == 10:
        fold_to_session = {
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 3,
            7: 4,
            8: 4,
            9: 5,
            10: 5,
        }
        df["session"] = df["fold"].map(fold_to_session)
        return df[df["session"] == fold_id].reset_index(drop=True)
    return df[df["fold"] == fold_id].reset_index(drop=True)


def build_loader(args, test_df: pd.DataFrame):
    dataset = IEMOCAPDataset(
        test_df,
        audio_max_length=args.max_audio_length,
        sample_rate=16000,
        tokenizer_name=args.roberta_model_name,
        max_text_length=args.max_text_length,
    )
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def build_full_model(args):
    return Method2CrossAttentionAV(
        num_emotions=args.num_emotions,
        wavlm_model_name=args.wavlm_model_name,
        wavlm_prosody_model_name=args.wavlm_prosody_model_name,
        roberta_model_name=args.roberta_model_name,
        hidden_dim=args.hidden_dim,
        av_refinement_layers=args.av_refinement_layers,
        freeze_wavlm=args.freeze_wavlm,
        freeze_roberta=args.freeze_roberta,
        num_cross_attn_heads=args.num_cross_attn_heads,
        post_attention_layers=args.post_attention_layers,
        av_prediction_layers=args.av_prediction_layers,
    )


def build_noav_model(args):
    return Method2CrossAttentionAV_AblationNoAVSup(
        num_emotions=args.num_emotions,
        wavlm_model_name=args.wavlm_model_name,
        wavlm_prosody_model_name=args.wavlm_prosody_model_name,
        roberta_model_name=args.roberta_model_name,
        hidden_dim=args.hidden_dim,
        av_refinement_layers=args.av_refinement_layers,
        freeze_wavlm=args.freeze_wavlm,
        freeze_roberta=args.freeze_roberta,
        num_cross_attn_heads=args.num_cross_attn_heads,
        post_attention_layers=args.post_attention_layers,
        av_prediction_layers=args.av_prediction_layers,
    )


def safe_load_and_filter_state(model, checkpoint_path: str, device: str = "cpu"):
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state

    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and isinstance(value, torch.Tensor) and value.shape == model_state[key].shape:
            filtered[key] = value
        else:
            skipped.append(key)
    model.load_state_dict(filtered, strict=False)
    return {"loaded": len(filtered), "skipped": skipped}


def find_checkpoint(model_dir: str | Path, fold_id: int, which: str = "best") -> Path:
    fold_dir = Path(model_dir) / f"fold_{fold_id}"
    preferred = fold_dir / f"{which}_model.pt"
    fallback = fold_dir / ("last_model.pt" if which == "best" else "best_model.pt")
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No checkpoint found in {fold_dir}")


def compute_ccc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size < 2 or y_pred.size != y_true.size:
        return float("nan")
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float((2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8))


def infer_collect(model, loader, device: str):
    model_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(model_device)
    model.eval()

    result = {
        "y_true": [],
        "y_pred": [],
        "arousal_true": [],
        "valence_true": [],
        "arousal_pred": [],
        "valence_pred": [],
        "fused_features": [],
        "final_representation": [],
        "arousal_embedding": [],
        "valence_embedding": [],
    }

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(model_device)
            text_ids = batch["text_input_ids"].to(model_device)
            text_mask = batch["text_attention_mask"].to(model_device)
            out = model(
                audio_input=audio,
                text_input=text_ids,
                audio_attention_mask=None,
                text_attention_mask=text_mask,
                return_intermediate=True,
            )

            logits = out["emotion_logits"]
            preds = torch.argmax(logits, dim=1)

            result["y_true"].append(batch["emotion_label"].cpu().numpy())
            result["y_pred"].append(preds.cpu().numpy())
            result["arousal_true"].append(batch["arousal_value"].cpu().numpy().reshape(-1))
            result["valence_true"].append(batch["valence_value"].cpu().numpy().reshape(-1))

            if "arousal_value" in out:
                result["arousal_pred"].append(out["arousal_value"].detach().cpu().numpy().reshape(-1))
            if "valence_value" in out:
                result["valence_pred"].append(out["valence_value"].detach().cpu().numpy().reshape(-1))
            if "fused_features" in out:
                result["fused_features"].append(out["fused_features"].detach().cpu().numpy())
            if "final_representation" in out:
                result["final_representation"].append(out["final_representation"].detach().cpu().numpy())
            if "arousal_embedding" in out:
                result["arousal_embedding"].append(out["arousal_embedding"].detach().cpu().numpy())
            if "valence_embedding" in out:
                result["valence_embedding"].append(out["valence_embedding"].detach().cpu().numpy())

    packed = {}
    for key, arrays in result.items():
        if arrays:
            packed[key] = np.concatenate(arrays, axis=0)
        else:
            packed[key] = np.array([])
    return packed


def cache_file(cache_dir: Path, model_key: str, fold_id: int) -> Path:
    return cache_dir / f"{model_key}_fold{fold_id}.npz"


def run_fold_cached(args, model_key: str, fold_id: int, force: bool = False) -> Path:
    dirs = ensure_output_dirs(args.output)
    cache_path = cache_file(dirs["cache"], model_key, fold_id)
    if cache_path.exists() and not force:
        return cache_path

    test_df = get_fold_test_df(args.csv, fold_id)
    loader = build_loader(args, test_df)
    if model_key == "full":
        model = build_full_model(args)
        ckpt = find_checkpoint(args.full_dir, fold_id, which=args.which)
    elif model_key == "noav":
        model = build_noav_model(args)
        ckpt = find_checkpoint(args.noav_dir, fold_id, which=args.which)
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    info = safe_load_and_filter_state(model, str(ckpt), device="cpu")
    result = infer_collect(model, loader, device=args.device)
    result["fold"] = np.array([fold_id], dtype=np.int64)
    result["loaded_count"] = np.array([info["loaded"]], dtype=np.int64)
    np.savez_compressed(cache_path, **result)
    return cache_path


def ensure_cache_for_models(args, folds: List[int] | None = None, force: bool = False):
    folds = folds or args.folds
    for fold_id in folds:
        run_fold_cached(args, "full", fold_id, force=force or args.force)
        run_fold_cached(args, "noav", fold_id, force=force or args.force)


def load_cached_fold(cache_dir: Path, model_key: str, fold_id: int) -> Dict[str, np.ndarray]:
    path = cache_file(cache_dir, model_key, fold_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def aggregate_cached(cache_dir: Path, model_key: str, folds: List[int]) -> Dict[str, np.ndarray]:
    rows = [load_cached_fold(cache_dir, model_key, fold_id) for fold_id in folds]
    keys = rows[0].keys()
    merged = {}
    for key in keys:
        if key in {"fold", "loaded_count"}:
            merged[key] = np.concatenate([row[key].reshape(-1) for row in rows], axis=0)
            continue
        arrays = [row[key] for row in rows if row[key].size > 0]
        if arrays:
            merged[key] = np.concatenate(arrays, axis=0)
        else:
            merged[key] = np.array([])
    return merged


def write_json(path: str | Path, payload: dict):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
