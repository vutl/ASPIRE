"""
Visualize architecture and dimensions for both methods
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from models.method1_triple_wavlm import Method1TripleWavLM
from models.method2_crossattn_av import Method2CrossAttentionAV


def print_tensor_info(name, tensor, indent=0):
    """Pretty print tensor information"""
    prefix = "  " * indent
    if isinstance(tensor, torch.Tensor):
        print(f"{prefix}{name}: {list(tensor.shape)}")
    else:
        print(f"{prefix}{name}: {tensor}")


def visualize_method1():
    print("="*80)
    print("METHOD 1: TRIPLE WAVLM - Detailed Architecture & Dimensions")
    print("="*80)
    
    model = Method1TripleWavLM(num_emotions=4, hidden_dim=256)
    
    # Dummy input
    batch_size = 2
    audio_length = 16000 * 3
    audio = torch.randn(batch_size, audio_length)
    
    print("\n📥 INPUT")
    print_tensor_info("audio_input", audio, indent=1)
    
    # Hook to capture intermediate outputs
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0]
            else:
                activations[name] = output
        return hook
    
    # Register hooks
    model.wavlm.register_forward_hook(get_activation('wavlm'))
    model.prosody_encoder.register_forward_hook(get_activation('prosody_feat'))
    model.text_encoder.register_forward_hook(get_activation('text_feat'))
    model.fusion_arousal.register_forward_hook(get_activation('arousal_embedding'))
    model.fusion_valence.register_forward_hook(get_activation('valence_embedding'))
    model.av_refinement.register_forward_hook(get_activation('av_refined'))
    model.final_fusion.register_forward_hook(get_activation('fused_features'))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio)
    
    print("\n🔹 STEP 1: WavLM Feature Extraction")
    if 'wavlm' in activations:
        print_tensor_info("wavlm_features", activations['wavlm'], indent=1)
    
    print("\n🔹 STEP 2: Branch 1 - Arousal (W + Prosody)")
    if 'prosody_feat' in activations:
        print_tensor_info("prosody_feat", activations['prosody_feat'], indent=1)
    print("  fusion_arousal:")
    print(f"    Input: wavlm_pooled [B, 768] + prosody_feat [B, 256]")
    if 'arousal_embedding' in activations:
        print_tensor_info("arousal_embedding", activations['arousal_embedding'], indent=2)
    print_tensor_info("arousal_value", outputs['arousal_value'], indent=1)
    
    print("\n🔹 STEP 3: Branch 2 - Valence (W + Text)")
    if 'text_feat' in activations:
        print_tensor_info("text_feat", activations['text_feat'], indent=1)
    print("  fusion_valence:")
    print(f"    Input: wavlm_pooled [B, 768] + text_feat [B, 256]")
    if 'valence_embedding' in activations:
        print_tensor_info("valence_embedding", activations['valence_embedding'], indent=2)
    print_tensor_info("valence_value", outputs['valence_value'], indent=1)
    
    print("\n🔹 STEP 4: A-V Refinement")
    print(f"  av_values: [arousal_value + valence_value] → [B, 2]")
    print("  av_refinement (MLP):")
    print("    ├─ Linear(2 → 256) → LayerNorm → ReLU → Dropout")
    print("    ├─ Linear(256 → 256) → LayerNorm → ReLU → Dropout")
    print("    └─ + residual")
    if 'av_refined' in activations:
        print_tensor_info("av_refined", activations['av_refined'], indent=2)
    
    print("\n🔹 STEP 5: Final Fusion (W + A + V + AV_refined)")
    print("  Concatenation:")
    print("    ├─ wavlm_pooled: [B, 768]")
    print("    ├─ arousal_embedding: [B, 256]")
    print("    ├─ valence_embedding: [B, 256]")
    print("    └─ av_refined: [B, 256]")
    print("    Total: [B, 1536]")
    print("\n  final_fusion layers:")
    print("    ├─ Linear(1536 → 512) → LayerNorm → ReLU → Dropout(0.2)")
    print("    └─ Linear(512 → 256) → LayerNorm → ReLU → Dropout(0.2)")
    if 'fused_features' in activations:
        print_tensor_info("fused_features", activations['fused_features'], indent=2)
    
    print("\n  emotion_classifier:")
    print("    └─ Linear(256 → 4)")
    print_tensor_info("emotion_logits", outputs['emotion_logits'], indent=2)
    
    print("\n📤 OUTPUTS")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print_tensor_info(key, value, indent=1)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")
    print("="*80 + "\n")


def visualize_method2():
    print("="*80)
    print("METHOD 2: CROSS-ATTENTION A-V - Detailed Architecture & Dimensions")
    print("="*80)
    
    model = Method2CrossAttentionAV(num_emotions=4, hidden_dim=256)
    
    # Dummy input
    batch_size = 2
    audio_length = 16000 * 3
    audio = torch.randn(batch_size, audio_length)
    
    print("\n📥 INPUT")
    print_tensor_info("audio_input", audio, indent=1)
    
    # Hook to capture intermediate outputs
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0]
            else:
                activations[name] = output
        return hook
    
    # Register hooks
    model.wavlm.register_forward_hook(get_activation('wavlm'))
    model.prosody_encoder_temporal.register_forward_hook(get_activation('prosody_temporal'))
    model.text_encoder_temporal.register_forward_hook(get_activation('text_temporal'))
    model.cross_attention.register_forward_hook(get_activation('cross_attention'))
    model.av_refinement.register_forward_hook(get_activation('av_refined'))
    model.final_fusion.register_forward_hook(get_activation('fused_features'))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio)
    
    print("\n🔹 STEP 1: WavLM Feature Extraction (FIRST use)")
    if 'wavlm' in activations:
        print_tensor_info("wavlm_features", activations['wavlm'], indent=1)
    
    print("\n🔹 STEP 2: Prosody & Text Encoders")
    print("  Prosody branch (for Arousal):")
    print("    ├─ TransformerEncoder(2 layers): [B, T, 768] → [B, T, 768]")
    print("    └─ Linear projection: [B, T, 768] → [B, T, 256]")
    if 'prosody_temporal' in activations:
        print_tensor_info("prosody_temporal", activations['prosody_temporal'], indent=2)
    
    print("\n  Text branch (for Valence):")
    print("    ├─ TransformerEncoder(2 layers): [B, T, 768] → [B, T, 768]")
    print("    └─ Linear projection: [B, T, 768] → [B, T, 256]")
    if 'text_temporal' in activations:
        print_tensor_info("text_temporal", activations['text_temporal'], indent=2)
    
    print("\n🔹 STEP 3: Bidirectional Cross-Attention")
    print("  cross_attention(prosody, text):")
    print("    ├─ prosody attends to text → prosody_attended")
    print("    └─ text attends to prosody → text_attended")
    print_tensor_info("prosody_attended", outputs['prosody_attended'], indent=2)
    print_tensor_info("text_attended", outputs['text_attended'], indent=2)
    
    print("\n🔹 STEP 4: Pooling & Embeddings")
    print("  arousal_pooling (Self-Attention Pooling):")
    print("    prosody_attended [B, T, 256] → arousal_embedding [B, 256]")
    print_tensor_info("arousal_embedding", outputs['arousal_embedding'], indent=2)
    
    print("\n  valence_pooling (Self-Attention Pooling):")
    print("    text_attended [B, T, 256] → valence_embedding [B, 256]")
    print_tensor_info("valence_embedding", outputs['valence_embedding'], indent=2)
    
    print("\n🔹 STEP 5: A-V Value Prediction")
    print("  arousal_head:")
    print("    └─ Linear(256 → 128) → ReLU → Dropout → Linear(128 → 1)")
    print_tensor_info("arousal_value", outputs['arousal_value'], indent=2)
    
    print("\n  valence_head:")
    print("    └─ Linear(256 → 128) → ReLU → Dropout → Linear(128 → 1)")
    print_tensor_info("valence_value", outputs['valence_value'], indent=2)
    
    print("\n🔹 STEP 6: A-V Refinement")
    print("  av_values: [arousal_value + valence_value] → [B, 2]")
    print("  av_refinement (MLP):")
    print("    ├─ Linear(2 → 256) → LayerNorm → ReLU → Dropout")
    print("    ├─ Linear(256 → 256) → LayerNorm → ReLU → Dropout")
    print("    └─ + residual")
    if 'av_refined' in activations:
        print_tensor_info("av_refined", activations['av_refined'], indent=2)
    
    print("\n🔹 STEP 7: Final Fusion (SECOND use of WavLM)")
    print("  ⚠️  CURRENT IMPLEMENTATION:")
    print("  Concatenation:")
    print("    ├─ wavlm_pooled: [B, 768]")
    print("    └─ av_refined: [B, 256]")
    print("    Total: [B, 1024]")
    print("\n  ❌ NOT USING: arousal_embedding [B, 256], valence_embedding [B, 256]")
    print("\n  final_fusion layers:")
    print("    ├─ Linear(1024 → 512) → LayerNorm → ReLU → Dropout(0.2)")
    print("    └─ Linear(512 → 256) → LayerNorm → ReLU → Dropout(0.2)")
    if 'fused_features' in activations:
        print_tensor_info("fused_features", activations['fused_features'], indent=2)
    
    print("\n  emotion_classifier:")
    print("    └─ Linear(256 → 4)")
    print_tensor_info("emotion_logits", outputs['emotion_logits'], indent=2)
    
    print("\n  💡 SUGGESTED FIX:")
    print("  Should concatenate: [wavlm_pooled, arousal_embedding, valence_embedding, av_refined]")
    print("  Total dimension: [B, 768 + 256 + 256 + 256] = [B, 1536]")
    
    print("\n📤 OUTPUTS")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print_tensor_info(key, value, indent=1)
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,}")
    print("="*80 + "\n")


def compare_final_fusion():
    print("="*80)
    print("COMPARISON: Final Fusion Inputs")
    print("="*80)
    
    print("\n📊 Method 1:")
    print("  Concatenates: [wavlm_pooled, arousal_emb, valence_emb, av_refined]")
    print("  Dimensions: [B, 768 + 256 + 256 + 256] = [B, 1536]")
    print("  ✅ Uses ALL information from prosody and text encoders")
    
    print("\n📊 Method 2 (Current):")
    print("  Concatenates: [wavlm_pooled, av_refined]")
    print("  Dimensions: [B, 768 + 256] = [B, 1024]")
    print("  ❌ Discards arousal_embedding and valence_embedding")
    print("  ⚠️  Information from prosody/text encoders only flows through A-V values")
    
    print("\n📊 Method 2 (Suggested):")
    print("  Concatenates: [wavlm_pooled, arousal_emb, valence_emb, av_refined]")
    print("  Dimensions: [B, 768 + 256 + 256 + 256] = [B, 1536]")
    print("  ✅ Uses ALL information like Method 1")
    print("  ✅ PLUS has cross-attention benefit")
    print("  💡 Best of both worlds!")
    
    print("\n" + "="*80 + "\n")


def show_loss_details():
    print("="*80)
    print("LOSS FUNCTIONS")
    print("="*80)
    
    print("\n📊 Multi-Task Loss (default: MSE)")
    print("-" * 40)
    print("L_total = α * L_emotion + β * L_arousal + γ * L_valence")
    print("\nwhere:")
    print("  L_emotion = CrossEntropyLoss(emotion_logits, emotion_label)")
    print("             Shape: [B, 4] vs [B] → scalar")
    print("\n  L_arousal = MSE(arousal_value, arousal_target)")
    print("             Shape: [B, 1] vs [B, 1] → scalar")
    print("\n  L_valence = MSE(valence_value, valence_target)")
    print("             Shape: [B, 1] vs [B, 1] → scalar")
    print("\nDefault weights:")
    print("  α (emotion_weight) = 1.0")
    print("  β (arousal_weight) = 0.5")
    print("  γ (valence_weight) = 0.5")
    
    print("\n\n📊 Alternative: CCC Loss")
    print("-" * 40)
    print("L_total = α * L_emotion + β * CCC_loss(arousal) + γ * CCC_loss(valence)")
    print("\nwhere CCC (Concordance Correlation Coefficient):")
    print("  CCC_loss = 1 - CCC")
    print("  CCC = (2 * covariance) / (σ_pred² + σ_target² + (μ_pred - μ_target)²)")
    print("\n💡 CCC is better than MSE for continuous emotion because:")
    print("   - Measures agreement (not just difference)")
    print("   - Considers both precision and accuracy")
    print("   - More robust to outliers")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n🔍 DETAILED ARCHITECTURE VISUALIZATION\n")
    
    visualize_method1()
    print("\n" * 2)
    visualize_method2()
    print("\n" * 2)
    compare_final_fusion()
    show_loss_details()
    
    print("\n✅ Visualization complete!")
    print("\n💡 Key Takeaway:")
    print("   Method 2 should add arousal/valence embeddings to final fusion")
    print("   to fully utilize the cross-attention benefits.\n")
