"""
Method 1: WavLM + Prosody/Text Cross-Attention Architecture
Architecture:
1. W (WavLM) extracts features from audio
2. Prosody Encoder processes WavLM features -> cross-attention with W -> Fusion -> Arousal branch
3. Text Encoder processes WavLM features -> cross-attention with W -> Fusion -> Valence branch
4. Each branch predicts A and V separately
5. A-V values go through refinement module (replaces graph network)
6. Final fusion: WavLM + fusion(W+Prosody) + fusion(W+Text) + AV_refined
7. Final layers -> Emotion classifier

Flow:
    Audio ──> WavLM ──┬──> Prosody Encoder ──> Cross-Attn ──> Fusion ──> Arousal
                      │
                      ├──> Text Encoder ────> Cross-Attn ──> Fusion ──> Valence
                      │
                      │    [Arousal, Valence] ──> Refinement ──> AV_refined
                      │
                      └──> WavLM + fusion(W+Prosody) + fusion(W+Text) + AV_refined ──> Final Fusion ──> Layers ──> Emotion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoModel

from models.base_components import (
    SelfAttentionPooling,
    FusionLayer,
    AVRefinementModule,
    ProsodyEncoder,
    TextSemanticEncoder
)


class Method1TripleWavLM(nn.Module):
    """
    Method 1: WavLM base appears 3 times in the architecture
    - Branch 1: W + Prosody -> A
    - Branch 2: W + Text -> V
    - Branch 3: W + A + V + AV_refined -> Emotion
    """
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=False
    ):
        super().__init__()
        
        # WavLM base model (shared across all branches)
        self.wavlm = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_dim = self.wavlm.config.hidden_size  # 768
        
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        
        # Branch 1: Prosody encoder for Arousal
        self.prosody_encoder = ProsodyEncoder(
            output_dim=hidden_dim,
            use_attention=True
        )
        
        # Branch 2: Text/Semantic encoder for Valence
        self.text_encoder = TextSemanticEncoder(
            output_dim=hidden_dim,
            use_attention=True
        )
        
        # Fusion layers for W + Prosody -> A and W + Text -> V
        self.fusion_arousal = FusionLayer(
            dim1=self.wavlm_dim,
            dim2=hidden_dim,
            output_dim=hidden_dim,
            fusion_type='gated'
        )
        
        self.fusion_valence = FusionLayer(
            dim1=self.wavlm_dim,
            dim2=hidden_dim,
            output_dim=hidden_dim,
            fusion_type='gated'
        )
        
        # Arousal and Valence prediction heads
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Single value for arousal
        )
        
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Single value for valence
        )
        
        # A-V refinement module (replaces GNN/GAT)
        self.av_refinement = AVRefinementModule(
            av_dim=2,  # [arousal, valence]
            hidden_dim=hidden_dim,
            num_layers=av_refinement_layers
        )
        
        # Global pooling for WavLM features (used in final fusion)
        self.wavlm_pooling = SelfAttentionPooling(self.wavlm_dim)
        
        # Final fusion: WavLM + fusion(W+Prosody) + fusion(W+Text) + AV_refined
        # Note: arousal_embedding is already fusion(W+Prosody), valence_embedding is fusion(W+Text)
        # Input dimension: 768 + 256*3 = 1536
        input_fusion_dim = self.wavlm_dim + hidden_dim * 3
        
        self.final_fusion = nn.Sequential(
            nn.Linear(input_fusion_dim, hidden_dim * 2),  # 1536 → 512
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 → 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Final classification layers (Standard Deep Classifier)
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)  # 128 → 4
        )
    
    def forward(self, audio_input, attention_mask=None):
        """
        Args:
            audio_input: [B, T_audio] - raw audio waveform
            attention_mask: [B, T_audio] - attention mask
        Returns:
            dict with:
                - emotion_logits: [B, num_emotions]
                - arousal_value: [B, 1]
                - valence_value: [B, 1]
                - arousal_embedding: [B, hidden_dim]
                - valence_embedding: [B, hidden_dim]
        """
        # Extract WavLM features (used 3 times in this method)
        wavlm_output = self.wavlm(audio_input, attention_mask=attention_mask)
        wavlm_features = wavlm_output.last_hidden_state  # [B, T, 768]
        
        # === Branch 1: W + Prosody -> Arousal ===
        # Step 1.1: Extract prosody features
        prosody_feat = self.prosody_encoder(wavlm_features)  # [B, hidden_dim]
        
        # Step 1.2: Pool WavLM for fusion with prosody
        wavlm_pooled_for_arousal, _ = self.wavlm_pooling(wavlm_features)  # [B, 768]
        
        # Step 1.3: Fusion W + Prosody
        arousal_embedding = self.fusion_arousal(
            wavlm_pooled_for_arousal,
            prosody_feat
        )  # [B, hidden_dim]
        
        # Step 1.4: Predict arousal value
        arousal_value = self.arousal_head(arousal_embedding)  # [B, 1]
        
        # === Branch 2: W + Text -> Valence ===
        # Step 2.1: Extract text/semantic features
        text_feat = self.text_encoder(wavlm_features)  # [B, hidden_dim]
        
        # Step 2.2: Pool WavLM for fusion with text (reuse same pooling)
        wavlm_pooled_for_valence = wavlm_pooled_for_arousal  # [B, 768]
        
        # Step 2.3: Fusion W + Text
        valence_embedding = self.fusion_valence(
            wavlm_pooled_for_valence,
            text_feat
        )  # [B, hidden_dim]
        
        # Step 2.4: Predict valence value
        valence_value = self.valence_head(valence_embedding)  # [B, 1]
        
        # === Refine A-V values ===
        av_values = torch.cat([arousal_value, valence_value], dim=-1)  # [B, 2]
        av_refined = self.av_refinement(av_values)  # [B, hidden_dim]
        
        # === Branch 3: Fusion(W + Prosody) and Fusion(W + Text) ===
        # These are the branch embeddings after fusion
        wpro_fusion = arousal_embedding  # [B, hidden_dim] - already fused W + Prosody
        wtext_fusion = valence_embedding  # [B, hidden_dim] - already fused W + Text
        
        # === Final fusion: WavLM + fusion(W+Prosody) + fusion(W+Text) + AV_refined -> Emotion ===
        # Concatenate all representations
        final_representation = torch.cat([
            wavlm_pooled_for_arousal,  # WavLM: [B, 768]
            wpro_fusion,                # fusion(W+Prosody): [B, hidden_dim]
            wtext_fusion,               # fusion(W+Text): [B, hidden_dim]
            av_refined                  # AV_refined: [B, hidden_dim]
        ], dim=-1)  # [B, 768 + hidden_dim + hidden_dim + hidden_dim]
        
        # Final fusion layers
        fused_features = self.final_fusion(final_representation)  # [B, hidden_dim]
        
        # Final classification layers (thêm layers trước classifier)
        emotion_logits = self.emotion_classifier(fused_features)  # [B, num_emotions]
        
        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'arousal_embedding': wpro_fusion,     # fusion(W+Prosody)
            'valence_embedding': wtext_fusion,    # fusion(W+Text)
            'av_refined': av_refined
        }


if __name__ == "__main__":
    # Test the model
    model = Method1TripleWavLM(num_emotions=4, hidden_dim=256)
    
    # Dummy input
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    audio_input = torch.randn(batch_size, audio_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio_input)
    
    print("Method 1 (Triple WavLM) Test:")
    print(f"Emotion logits shape: {outputs['emotion_logits'].shape}")
    print(f"Arousal value shape: {outputs['arousal_value'].shape}")
    print(f"Valence value shape: {outputs['valence_value'].shape}")
    print(f"Arousal embedding shape: {outputs['arousal_embedding'].shape}")
    print(f"Valence embedding shape: {outputs['valence_embedding'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
