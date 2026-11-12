#!/usr/bin/env python3
"""
Extract embeddings for IEMOCAP (4-class) using pre-trained models to avoid heavy encoder forward during training.
Extracts WavLM-base-plus, W-PRO (Orange/Speaker-wavLM-pro), and RoBERTa CLS embeddings.
Matches logging/print style of MSP extraction script.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def extract_wavlm_embeddings(audio_paths, model_name, device, batch_size=8):
    """Extract WavLM-family embeddings with mean pooling."""
    print(f"\nLoading {model_name}...")
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    max_len = 160000  # 10 seconds at 16kHz

    with torch.no_grad():
        for i in tqdm(range(0, len(audio_paths), batch_size), desc=f"Extracting {model_name}"):
            batch_paths = audio_paths[i:i + batch_size]
            batch_audios = []

            for path in batch_paths:
                try:
                    waveform, sr = torchaudio.load(path)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    waveform = waveform.squeeze(0)
                    if len(waveform) > max_len:
                        waveform = waveform[:max_len]
                    else:
                        waveform = torch.nn.functional.pad(waveform, (0, max_len - len(waveform)))
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    waveform = torch.zeros(max_len)
                batch_audios.append(waveform)

            batch_tensor = torch.stack(batch_audios).to(device)
            outputs = model(batch_tensor)
            pooled = torch.mean(outputs.last_hidden_state, dim=1)
            embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"  ✓ Extracted {embeddings.shape[0]} embeddings, shape: {embeddings.shape}")

    del model
    torch.cuda.empty_cache()
    return embeddings


def extract_roberta_embeddings(transcripts, tokenizer_name, device, batch_size=32):
    """Extract RoBERTa CLS embeddings for transcripts."""
    print(f"\nLoading {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(tokenizer_name).to(device)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(transcripts), batch_size), desc="Extracting RoBERTa"):
            batch_texts = transcripts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    print(f"  ✓ Extracted {embeddings.shape[0]} embeddings, shape: {embeddings.shape}")

    del model, tokenizer
    torch.cuda.empty_cache()
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='datasets/IEMOCAP_KFOLD/all.csv', help='Path to IEMOCAP CSV (with fold column).')
    parser.add_argument('--output_dir', default='datasets/IEMOCAP_KFOLD/embeddings', help='Output directory to save embeddings and metadata.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size_audio', type=int, default=8)
    parser.add_argument('--batch_size_text', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("IEMOCAP EMBEDDING EXTRACTION")
    print("=" * 80)

    # Load CSV
    print(f"\nLoading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Filter to 4 emotions
    keep_emotions = ['ang', 'sad', 'hap', 'neu']
    df = df[df['emotion'].isin(keep_emotions)].reset_index(drop=True)
    print(f"Total samples (4-class): {len(df)}")
    print(f"Emotion distribution: {df['emotion'].value_counts().to_dict()}")

    # Audio paths and transcripts
    audio_paths = df['path'].tolist()
    transcripts = df['transcript'].fillna('').astype(str).tolist()

    # Map fold → session (same logic as training script)
    max_fold = df['fold'].max()
    if max_fold == 10:
        fold_to_session = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5}
        sessions = df['fold'].map(fold_to_session).tolist()
    else:
        sessions = df['fold'].tolist()

    # Extract WavLM-base-plus embeddings
    print("\n" + "=" * 80)
    print("1. Extracting WavLM-base-plus embeddings")
    print("=" * 80)
    wavlm_embeddings = extract_wavlm_embeddings(
        audio_paths,
        "microsoft/wavlm-base-plus",
        args.device,
        args.batch_size_audio
    )
    wavlm_file = os.path.join(args.output_dir, 'wavlm_base_plus_embeddings.npy')
    np.save(wavlm_file, wavlm_embeddings)
    print(f"  ✓ Saved to {wavlm_file}")

    # Extract W-PRO embeddings
    print("\n" + "=" * 80)
    print("2. Extracting W-PRO (WavLM-prosody) embeddings")
    print("=" * 80)
    wpro_embeddings = extract_wavlm_embeddings(
        audio_paths,
        "Orange/Speaker-wavLM-pro",
        args.device,
        args.batch_size_audio
    )
    wpro_file = os.path.join(args.output_dir, 'wpro_embeddings.npy')
    np.save(wpro_file, wpro_embeddings)
    print(f"  ✓ Saved to {wpro_file}")

    # Extract RoBERTa embeddings
    print("\n" + "=" * 80)
    print("3. Extracting RoBERTa embeddings")
    print("=" * 80)
    roberta_embeddings = extract_roberta_embeddings(
        transcripts,
        "roberta-base",
        args.device,
        args.batch_size_text
    )
    roberta_file = os.path.join(args.output_dir, 'roberta_embeddings.npy')
    np.save(roberta_file, roberta_embeddings)
    print(f"  ✓ Saved to {roberta_file}")

    # Save metadata
    print("\n" + "=" * 80)
    print("4. Saving metadata")
    print("=" * 80)
    metadata = {
        'sample_names': df['sample_name'].tolist(),
        'emotion': df['emotion'].tolist(),
        'fold': df['fold'].tolist(),
        'session': sessions,
        'arousal': df['arousal'].tolist(),
        'valence': df['valence'].tolist(),
        'dominance': df['dominance'].tolist() if 'dominance' in df.columns else [3.0] * len(df),
        'paths': audio_paths,
        'transcripts': transcripts
    }
    metadata_file = os.path.join(args.output_dir, 'metadata.npz')
    np.savez(metadata_file, **metadata)
    print(f"  ✓ Saved to {metadata_file}")

    print("\n" + "=" * 80)
    print("✓ EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Embeddings saved to: {args.output_dir}")
    print(f"  - WavLM-base-plus: {wavlm_embeddings.shape}")
    print(f"  - W-PRO: {wpro_embeddings.shape}")
    print(f"  - RoBERTa: {roberta_embeddings.shape}")
    print(f"  - Metadata: {len(metadata['sample_names'])} samples")


if __name__ == '__main__':
    main()
