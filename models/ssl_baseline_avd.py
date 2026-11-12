"""
SSL Baseline for A-V-D Emotion Recognition

Architecture:
1. Audio -> WavLM (SSL) -> Extract features [B, T, 768]
2. Split into 2 branches:
   - Branch 1: Pool -> Predict A-V-D values -> Project to AV_refined [B, 256]
   - Branch 2: Pool -> Keep as embedding [B, 768]
3. Final fusion: [Branch2_embedding + AV_refined] -> Emotion classifier

This is a simple baseline using only SSL (WavLM) without text/prosody branches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

from models.base_components import (
    SelfAttentionPooling,
    AVRefinementModule
)


class SSLBaselineAVD(nn.Module):
    """
    SSL Baseline for A-V-D Emotion Recognition
    
    Flow:
    Audio -> WavLM -> 2 branches
      Branch 1: Pooling -> A-V-D prediction -> AV_refined
      Branch 2: Pooling -> Embedding
    Final: [Branch2_embedding + AV_refined] -> Emotion
    """
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=False,
        av_prediction_layers=3
    ):
        super().__init__()
        
        # === WavLM SSL model ===
        print(f"Loading WavLM: {wavlm_model_name}")
        self.wavlm = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_dim = self.wavlm.config.hidden_size  # 768 for WavLM-base-plus
        
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            print(f"WavLM frozen: {freeze_wavlm}")
        
        # === Branch 1: A-V-D prediction branch ===
        self.avd_pooling = SelfAttentionPooling(self.wavlm_dim)
        
        # A-V-D prediction layers
        av_layers = []
        for i in range(av_prediction_layers):
            if i == 0:
                av_layers.extend([
                    nn.Linear(self.wavlm_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            else:
                av_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
        
        self.av_prediction_layers = nn.Sequential(*av_layers)
        
        # A-V-D heads
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)
        self.dominance_head = nn.Linear(hidden_dim, 1)
        
        # A-V-D refinement module
        self.av_refinement = AVRefinementModule(
            av_dim=3,  # [arousal, valence, dominance]
            hidden_dim=hidden_dim,
            num_layers=av_refinement_layers
        )
        
        # === Branch 2: Emotion embedding branch ===
        self.emotion_pooling = SelfAttentionPooling(self.wavlm_dim)
        
        # === Final fusion: [emotion_embedding + av_refined] -> Emotion ===
        # Input: [wavlm_pooled (768) + av_refined (256)] = 1024
        input_fusion_dim = self.wavlm_dim + hidden_dim  # 768 + 256 = 1024
        
        self.final_fusion = nn.Sequential(
            nn.Linear(input_fusion_dim, hidden_dim * 2),  # 1024 → 512
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 → 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)  # 128 → num_emotions
        )
    
    def forward(self, audio_input, text_input=None, audio_attention_mask=None, text_attention_mask=None):
        """
        Args:
            audio_input: [B, T_audio] - raw audio waveform
            text_input: Not used (kept for compatibility)
            audio_attention_mask: [B, T_audio]
            text_attention_mask: Not used
        Returns:
            dict with emotion_logits, arousal_value, valence_value, dominance_value
        """
        # === Extract WavLM features ===
        wavlm_output = self.wavlm(audio_input, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_output.last_hidden_state  # [B, T, 768]
        
        # === Branch 1: A-V-D prediction branch ===
        avd_pooled, _ = self.avd_pooling(wavlm_features)  # [B, 768]
        av_features = self.av_prediction_layers(avd_pooled)  # [B, hidden_dim]
        
        # Predict A-V-D values
        arousal_value = self.arousal_head(av_features)  # [B, 1]
        valence_value = self.valence_head(av_features)  # [B, 1]
        dominance_value = self.dominance_head(av_features)  # [B, 1]
        
        # Refine A-V-D values
        avd_values = torch.cat([arousal_value, valence_value, dominance_value], dim=-1)  # [B, 3]
        av_refined = self.av_refinement(avd_values)  # [B, hidden_dim]
        
        # === Branch 2: Emotion embedding branch ===
        emotion_pooled, _ = self.emotion_pooling(wavlm_features)  # [B, 768]
        
        # === Final fusion: [emotion_embedding + av_refined] -> Emotion ===
        final_representation = torch.cat([
            emotion_pooled,  # [B, 768]
            av_refined       # [B, 256]
        ], dim=-1)  # [B, 1024]
        
        # Final fusion and classification
        fused_features = self.final_fusion(final_representation)  # [B, 256]
        emotion_logits = self.emotion_classifier(fused_features)  # [B, num_emotions]
        
        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'dominance_value': dominance_value,
            'av_features': av_features,
            'av_refined': av_refined,
            'emotion_embedding': emotion_pooled,
            'avd_embedding': avd_pooled
        }


if __name__ == "__main__":
    print("Testing SSLBaselineAVD...")
    model = SSLBaselineAVD(num_emotions=4, hidden_dim=256)
    
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds
    audio_input = torch.randn(batch_size, audio_length)
    
    with torch.no_grad():
        outputs = model(audio_input)
    
    print(f"Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"Arousal: {outputs['arousal_value'].shape}")
    print(f"Valence: {outputs['valence_value'].shape}")
    print(f"Dominance: {outputs['dominance_value'].shape}")
    print(f"AV refined: {outputs['av_refined'].shape}")
    print(f"Emotion embedding: {outputs['emotion_embedding'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\n=== SSL Baseline ===")
    print("Simple architecture using only WavLM (no prosody/text branches)")
    print("2 branches from WavLM: AVD prediction + Emotion embedding")
