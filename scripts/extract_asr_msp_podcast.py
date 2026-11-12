#!/usr/bin/env python3
"""
Extract ASR text from MSP-Podcast audio using Whisper
Saves to msp_podcast_asr.tsv with 'asr_transcript' column
"""
import os
import sys
import argparse
import pandas as pd
import whisper
from tqdm import tqdm
from pathlib import Path


def extract_asr_text_msp(
    input_tsv,
    output_tsv,
    model_name="base",
    device="cuda",
    save_every=100,
    batch_size=16
):
    """
    Extract ASR text from MSP-Podcast audio files
    
    Args:
        input_tsv: Path to msp_podcast.tsv
        output_tsv: Path to output TSV (will add 'asr_transcript' column)
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to use
        save_every: Save checkpoint every N samples
        batch_size: Number of files to process in parallel
    """
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=device)
    print(f"Batch size: {batch_size}")
    
    print(f"Loading input TSV: {input_tsv}")
    df = pd.read_csv(input_tsv, sep='\t')
    
    print(f"Total samples: {len(df)}")
    
    # Check if 'path' column exists
    if 'path' not in df.columns:
        raise ValueError("TSV must have 'path' column with audio file paths")
    
    # Check if partial results exist (for resume)
    checkpoint_file = output_tsv + ".checkpoint"
    start_idx = 0
    asr_transcripts = []
    
    if os.path.exists(checkpoint_file):
        print(f"\nFound checkpoint file, resuming from previous run...")
        checkpoint_df = pd.read_csv(checkpoint_file, sep='\t')
        start_idx = len(checkpoint_df)
        asr_transcripts = checkpoint_df['asr_transcript'].tolist()
        print(f"Resuming from sample {start_idx}")
    
    # Process audio files in batches
    for batch_start in tqdm(range(start_idx, len(df), batch_size), desc="Extracting ASR text"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_rows = df.iloc[batch_start:batch_end]
        
        # Process batch
        for idx, row in enumerate(batch_rows.itertuples(index=False), start=batch_start):
            audio_path = row.path
            
            # Check if audio exists
            if not os.path.exists(audio_path):
                asr_transcripts.append("")
                continue
            
            try:
                # Transcribe audio
                result = model.transcribe(audio_path, language="en", fp16=(device=="cuda"))
                transcript = result['text'].strip()
                asr_transcripts.append(transcript)
                
            except Exception as e:
                print(f"\nError processing {audio_path}: {e}")
                asr_transcripts.append("")
        
        # Save checkpoint after each batch
        if (batch_end % save_every) <= batch_size or batch_end == len(df):
            temp_df = df.iloc[:len(asr_transcripts)].copy()
            temp_df['asr_transcript'] = asr_transcripts
            temp_df.to_csv(checkpoint_file, sep='\t', index=False)
            print(f"\nCheckpoint saved at sample {len(asr_transcripts)}")
    
    # Add to dataframe
    df['asr_transcript'] = asr_transcripts
    
    # Save output
    print(f"\nSaving results to: {output_tsv}")
    df.to_csv(output_tsv, sep='\t', index=False)
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Successfully transcribed: {sum(1 for t in asr_transcripts if t)}")
    print(f"  Empty transcripts: {sum(1 for t in asr_transcripts if not t)}")
    
    # Average length
    lengths = [len(t.split()) for t in asr_transcripts if t]
    if lengths:
        print(f"  Average transcript length: {sum(lengths)/len(lengths):.1f} words")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ASR text from MSP-Podcast using Whisper")
    parser.add_argument("--input_tsv", type=str, 
                        default="datasets/MSP-Podcast/msp_podcast.tsv",
                        help="Input TSV with audio paths")
    parser.add_argument("--output_tsv", type=str, 
                        default="datasets/MSP-Podcast/msp_podcast_asr.tsv",
                        help="Output TSV with ASR transcripts")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N samples")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing")
    
    args = parser.parse_args()
    
    extract_asr_text_msp(
        input_tsv=args.input_tsv,
        output_tsv=args.output_tsv,
        model_name=args.model,
        device=args.device,
        save_every=args.save_every,
        batch_size=args.batch_size
    )
