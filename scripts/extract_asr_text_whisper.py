"""
Extract text transcripts from audio using Whisper ASR
Saves results to CSV for later use in training
"""
import os
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
import argparse
import whisper


def extract_asr_text(
    input_csv,
    output_csv,
    model_name="base",  # tiny, base, small, medium, large
    device="cuda",
    batch_size=1,  # Whisper processes one at a time
    save_every=100  # Save checkpoint every N samples
):
    """
    Extract ASR text from audio files using Whisper
    
    Args:
        input_csv: Path to input CSV with 'path' column (audio paths)
        output_csv: Path to output CSV (will add 'asr_transcript' column)
        model_name: Whisper model size (tiny, base, small, medium, large)
        device: Device to use
        batch_size: Not used (Whisper processes one sample at a time)
        save_every: Save checkpoint every N samples
    """
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=device)
    
    print(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Total samples: {len(df)}")
    
    # Check if 'path' column exists
    if 'path' not in df.columns:
        raise ValueError("Input CSV must have 'path' column with audio file paths")
    
    # Check if partial results exist (for resume)
    checkpoint_file = output_csv + ".checkpoint"
    start_idx = 0
    asr_transcripts = []
    
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint: {checkpoint_file}")
        checkpoint_df = pd.read_csv(checkpoint_file)
        start_idx = len(checkpoint_df)
        asr_transcripts = checkpoint_df['asr_transcript'].tolist()
        print(f"Resuming from index {start_idx}/{len(df)}")
    
    # Process each audio file
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ASR text", initial=start_idx):
        # Skip already processed
        if idx < start_idx:
            continue
            
        audio_path = row['path']
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"Warning: File not found: {audio_path}")
            asr_transcripts.append("")
            continue
        
        try:
            # Transcribe with Whisper
            # Whisper expects audio at 16kHz
            result = model.transcribe(
                audio_path,
                language="en",  # IEMOCAP is English
                fp16=(device == "cuda")  # Use FP16 on GPU
            )
            
            # Get text
            text = result["text"].strip()
            asr_transcripts.append(text)
            
            # Print first few samples
            if idx < 5:
                print(f"\nSample {idx}:")
                print(f"  Audio: {audio_path}")
                print(f"  ASR: {text}")
                if 'transcript' in row:
                    print(f"  GT:  {row['transcript']}")
            
            # Save checkpoint periodically
            if (idx + 1) % save_every == 0:
                temp_df = df.iloc[:idx+1].copy()
                temp_df['asr_transcript'] = asr_transcripts
                temp_df.to_csv(checkpoint_file, index=False)
                print(f"\nCheckpoint saved at {idx+1}/{len(df)}")
                    
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            asr_transcripts.append("")
    
    # Add to dataframe
    df['asr_transcript'] = asr_transcripts
    
    # Save output
    print(f"\nSaving results to: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    # Remove checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed checkpoint file: {checkpoint_file}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Successfully transcribed: {sum(1 for t in asr_transcripts if t)}")
    print(f"  Empty transcripts: {sum(1 for t in asr_transcripts if not t)}")
    
    # Average length
    lengths = [len(t.split()) for t in asr_transcripts if t]
    if lengths:
        print(f"  Average words per transcript: {sum(lengths) / len(lengths):.1f}")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ASR text using Whisper")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input CSV with audio paths")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV with ASR transcripts")
    parser.add_argument("--model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    extract_asr_text(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_name=args.model,
        device=args.device
    )
