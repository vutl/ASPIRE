"""
HuBERT A-V Model
Simple single-branch architecture:
1. HuBERT extracts acoustic features (frozen)
2. Self-attention pooling to aggregate temporal features
3. Task-specific layers (embedding + refinement + classifier)
4. Outputs: emotion logits, arousal value, valence value
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel
from thop import profile, clever_format


class SelfAttentionPooling(nn.Module):
    """Self-attention pooling to aggregate temporal features"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, x):
        # x: [B, T, D]
        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
        return pooled, attn_weights


class HuBERTAV(nn.Module):
    """
    HuBERT for A-V emotion recognition
    
    Architecture:
    Audio → HuBERT-base (frozen) [B, T, 768]
         → SelfAttentionPooling [B, 768]
         → Embedding layer → [B, hidden_dim]
         → A-V refinement layers → [B, hidden_dim]
         → Arousal head → [B, 1]
         → Valence head → [B, 1]
         → Final fusion → [B, hidden_dim]
         → Emotion classifier → [B, num_emotions]
    """
    def __init__(
        self,
        num_emotions=4,
            hubert_model_name="facebook/hubert-base-ls960",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_hubert=True
    ):
        super().__init__()
        
        # === HuBERT backbone (frozen) ===
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        self.hubert_dim = self.hubert.config.hidden_size  # 768
        
        if freeze_hubert:
            for param in self.hubert.parameters():
                param.requires_grad = False
            print(f"HuBERT frozen: {freeze_hubert}")
        
        # === Task-specific layers (trainable) ===
        
        # Self-attention pooling
        self.pooling = SelfAttentionPooling(self.hubert_dim)
        
        # Embedding layer: 768 → hidden_dim
        self.embedding = nn.Sequential(
            nn.Linear(self.hubert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # A-V prediction layers
        av_layers = []
        for i in range(av_refinement_layers):
            av_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        self.av_prediction = nn.Sequential(*av_layers)
        
        # A-V heads
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)
        
        # A-V refinement: [2] → hidden_dim
        self.av_refinement = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Final fusion: embedding + av_refined → emotion
        fusion_input_dim = hidden_dim + hidden_dim  # 512
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
    
    def forward(self, audio_input, audio_attention_mask=None):
        """
        Args:
            audio_input: [B, T_audio] - raw audio waveform
            audio_attention_mask: [B, T_audio] - attention mask for audio
        
        Returns:
            dict with:
                - emotion_logits: [B, num_emotions]
                - arousal_value: [B, 1]
                - valence_value: [B, 1]
        """
        # Extract features from HuBERT (frozen)
        hubert_output = self.hubert(audio_input, attention_mask=audio_attention_mask)
        hubert_features = hubert_output.last_hidden_state  # [B, T, 768]
        
        # Self-attention pooling
        pooled_features, _ = self.pooling(hubert_features)  # [B, 768]
        
        # Embedding
        embedded = self.embedding(pooled_features)  # [B, hidden_dim]
        
        # A-V prediction
        av_features = self.av_prediction(embedded)  # [B, hidden_dim]
        arousal_value = self.arousal_head(av_features)  # [B, 1]
        valence_value = self.valence_head(av_features)  # [B, 1]
        
        # A-V refinement
        av_values = torch.cat([arousal_value, valence_value], dim=-1)  # [B, 2]
        av_refined = self.av_refinement(av_values)  # [B, hidden_dim]
        
        # Final fusion
        final_input = torch.cat([embedded, av_refined], dim=-1)  # [B, 2*hidden_dim]
        fused_features = self.final_fusion(final_input)  # [B, hidden_dim]
        
        # Emotion classification
        emotion_logits = self.emotion_classifier(fused_features)  # [B, num_emotions]
        
        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value
        }


if __name__ == "__main__":
    # Test the model
    model = HuBERTAV(num_emotions=4, hidden_dim=256)
    
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"HuBERT A-V Model")
    print(f"{'='*60}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters: {total - trainable:,}")
    
    # Dummy input
    batch_size = 2
    audio_length = 16000 * 3
    audio_input = torch.randn(batch_size, audio_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio_input)
    
    print(f"\nOutput shapes:")
    print(f"  Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"  Arousal value: {outputs['arousal_value'].shape}")
    print(f"  Valence value: {outputs['valence_value'].shape}")
    
    # Compute FLOPs
    try:
        flops, params = profile(
            model,
            inputs=(audio_input,),
            verbose=False
        )
        flops_str, params_str = clever_format([flops, params], "%.3f")
        print(f"\nFLOPs: {flops_str}, Params: {params_str}")
    except:
        print("FLOPs computation skipped")
