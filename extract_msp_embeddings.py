#!/usr/bin/env python3
"""
Extract embeddings for MSP-Podcast dataset
Extracts: WavLM-base-plus, W-PRO (WavLM-prosody), RoBERTa
Saves embeddings to disk for faster evaluation
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
import torchaudio
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

def extract_wavlm_embeddings(audio_paths, model_name, device, batch_size=8):
    """Extract WavLM embeddings"""
    print(f"\nLoading {model_name}...")
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(audio_paths), batch_size), desc=f"Extracting {model_name}"):
            batch_paths = audio_paths[i:i+batch_size]
            batch_audios = []
            
            # Load batch
            for path in batch_paths:
                try:
                    waveform, sr = torchaudio.load(path)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    waveform = waveform.squeeze(0)
                    
                    # Pad/truncate to 10 seconds
                    max_len = 160000
                    if len(waveform) > max_len:
                        waveform = waveform[:max_len]
                    else:
                        waveform = torch.nn.functional.pad(waveform, (0, max_len - len(waveform)))
                    
                    batch_audios.append(waveform)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    batch_audios.append(torch.zeros(160000))
            
            # Stack and process
            batch_tensor = torch.stack(batch_audios).to(device)
            outputs = model(batch_tensor)
            
            # Get last hidden state and mean pool
            hidden_states = outputs.last_hidden_state  # [B, T, D]
            pooled = torch.mean(hidden_states, dim=1)  # [B, D]
            
            embeddings.append(pooled.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    print(f"  ✓ Extracted {embeddings.shape[0]} embeddings, shape: {embeddings.shape}")
    
    del model
    torch.cuda.empty_cache()
    
    return embeddings

def extract_roberta_embeddings(transcripts, tokenizer_name, device, batch_size=32):
    """Extract RoBERTa embeddings"""
    print(f"\nLoading {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(tokenizer_name).to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(transcripts), batch_size), desc="Extracting RoBERTa"):
            batch_texts = transcripts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Forward
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get [CLS] token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, D]
            
            embeddings.append(cls_embeddings.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    print(f"  ✓ Extracted {embeddings.shape[0]} embeddings, shape: {embeddings.shape}")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return embeddings

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', default='datasets/MSP-Podcast/msp_podcast.tsv')
    parser.add_argument('--output_dir', default='datasets/MSP-Podcast/embeddings')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size_audio', type=int, default=8)
    parser.add_argument('--batch_size_text', type=int, default=32)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("MSP-PODCAST EMBEDDING EXTRACTION")
    print("="*80)
    
    # Load TSV
    print(f"\nLoading {args.tsv}...")
    df = pd.read_csv(args.tsv, sep='\t')
    
    # Filter to 4 emotions
    df = df[df['label'].isin(['ang', 'sad', 'hap', 'neu'])].reset_index(drop=True)
    print(f"Total samples (4-class): {len(df)}")
    print(f"Emotion distribution: {df['label'].value_counts().to_dict()}")
    
    # Get audio paths
    audio_paths = df['path'].tolist()
    
    # Load transcripts
    print("\nLoading transcripts...")
    base_dir = os.path.dirname(args.tsv)
    transcript_dir = os.path.join(base_dir, 'Transcripts')
    transcripts = []
    
    for utt_id in tqdm(df['utt_id'], desc="Loading transcripts"):
        transcript_file = os.path.join(transcript_dir, f'{utt_id}.txt')
        if os.path.exists(transcript_file):
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
            except:
                transcript = ""
        else:
            transcript = ""
        transcripts.append(transcript)
    
    print(f"  ✓ Loaded {len(transcripts)} transcripts")
    
    # Extract WavLM-base-plus embeddings
    print("\n" + "="*80)
    print("1. Extracting WavLM-base-plus embeddings")
    print("="*80)
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
    print("\n" + "="*80)
    print("2. Extracting W-PRO (WavLM-prosody) embeddings")
    print("="*80)
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
    print("\n" + "="*80)
    print("3. Extracting RoBERTa embeddings")
    print("="*80)
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
    print("\n" + "="*80)
    print("4. Saving metadata")
    print("="*80)
    
    # Load A-V values from labels_consensus.csv for alignment
    base_dir = os.path.dirname(args.tsv)
    labels_csv = os.path.join(base_dir, 'Labels', 'labels_consensus.csv')
    labels_df = pd.read_csv(labels_csv)
    
    # Create dict: filename -> {arousal, valence, dominance}
    av_dict = {}
    for _, row in labels_df.iterrows():
        av_dict[row['FileName']] = {
            'arousal': row['EmoAct'],
            'valence': row['EmoVal'],
            'dominance': row['EmoDom'] if 'EmoDom' in labels_df.columns else 3.0,
        }
    
    # Extract A-V values in same order as embeddings (by audio path basename)
    arousal_vals = []
    valence_vals = []
    dominance_vals = []
    for path in audio_paths:
        fname = os.path.basename(path)
        av = av_dict.get(fname, {'arousal': 3.0, 'valence': 3.0, 'dominance': 3.0})
        arousal_vals.append(av['arousal'])
        valence_vals.append(av['valence'])
        dominance_vals.append(av['dominance'])
    
    metadata = {
        'utt_ids': df['utt_id'].values,
        'labels': df['label'].values,
        'splits': df['split'].values,
        'audio_paths': audio_paths,
        'transcripts': transcripts,
        'arousal': np.array(arousal_vals, dtype=np.float32),
        'valence': np.array(valence_vals, dtype=np.float32),
        'dominance': np.array(dominance_vals, dtype=np.float32),
    }
    
    metadata_file = os.path.join(args.output_dir, 'metadata.npz')
    np.savez(metadata_file, **metadata)
    print(f"  ✓ Saved to {metadata_file}")
    
    print("\n" + "="*80)
    print("✓ EXTRACTION COMPLETE!")
    print("="*80)
    print(f"Embeddings saved to: {args.output_dir}")
    print(f"  - WavLM-base-plus: {wavlm_embeddings.shape}")
    print(f"  - W-PRO: {wpro_embeddings.shape}")
    print(f"  - RoBERTa: {roberta_embeddings.shape}")
    print(f"  - Metadata: {len(metadata['utt_ids'])} samples")

if __name__ == '__main__':
    main()
