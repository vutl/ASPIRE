"""
Method 2 Variant: Cross-Attention A-V Architecture with Fusion Concat
Architecture:
1. WavLM_prosody (Wpro) extracts acoustic/prosody features from audio
2. RoBERTa extracts semantic/linguistic features from transcript text
3. Wpro and RoBERTa cross-attention with each other
4. Each branch goes through some layers after cross-attention
5. Fusion of the two branches -> branch_fused
6. Fusion goes through some layers -> outputs A-V values
7. A-V values go through refinement module (replaces graph network)
8. Separate WavLM branch extracts global acoustic features
9. Final fusion: WavLM + branch_fused + AV_refined -> Emotion labels

Difference from original Method 2:
- Original: Concatenates [WavLM, Arousal_Emb, Valence_Emb, AV_Refined]
- This Variant: Concatenates [WavLM, Branch_Fused, AV_Refined]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer

from models.base_components import (
    SelfAttentionPooling,
    BiDirectionalCrossAttention,
    FusionLayer,
    AVRefinementModule,
    ProsodyEncoder,
    TextSemanticEncoder
)


class Method2CrossAttentionAV_FusionConcat(nn.Module):
    """
    Method 2 Variant: Cross-Attention A-V Architecture with Fusion Concat
    
    Flow:
    1. Wpro (WavLM) extracts acoustic features from audio
    2. RoBERTa extracts semantic features from text transcript
    3. Wpro and RoBERTa cross-attention
    4. Each branch goes through post-attention layers
    5. Fusion of both branches -> branch_fused
    6. branch_fused -> layers -> A-V values
    7. A-V values -> refinement module -> AV_refined
    8. Separate WavLM extracts global acoustic features
    9. Final fusion: WavLM + branch_fused + AV_refined -> Emotion
    """
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name=None,  # Optional: Use W-PRO for prosody (Orange/Speaker-wavLM-pro)
        roberta_model_name="roberta-base",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=False,
        freeze_roberta=False,
        num_cross_attn_heads=8,
        post_attention_layers=2,
        av_prediction_layers=3
    ):
        super().__init__()
        
        # === Branch 1: WavLM for prosody (Wpro) - extracts acoustic/prosody features ===
        # Use W-PRO (Orange/Speaker-wavLM-pro) if specified, otherwise use standard WavLM
        prosody_model_name = wavlm_prosody_model_name if wavlm_prosody_model_name else wavlm_model_name
        self.use_wpro = (wavlm_prosody_model_name is not None)
        
        if self.use_wpro:
            # W-PRO model - specialized for non-timbral prosody features
            print(f"Loading W-PRO model: {prosody_model_name}")
            from transformers import WavLMModel
            self.wavlm_prosody = WavLMModel.from_pretrained(prosody_model_name)
            # W-PRO outputs embeddings, need to get hidden size from config
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size
        else:
            # Standard WavLM-base-plus
            print(f"Loading standard WavLM for prosody: {prosody_model_name}")
            self.wavlm_prosody = Wav2Vec2Model.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size  # 768 for WavLM-base-plus
        
        if freeze_wavlm:
            for param in self.wavlm_prosody.parameters():
                param.requires_grad = False
            print(f"WavLM prosody frozen: {freeze_wavlm}")
        
        # Project prosody features to hidden_dim
        self.prosody_projection = nn.Linear(self.wavlm_dim, hidden_dim)
        
        # === Branch 2: RoBERTa for semantic/text - processes transcript ===
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size  # 768 for roberta-base
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        # Project text features
        self.text_projection = nn.Linear(self.roberta_dim, hidden_dim)
        
        # === Cross-Attention between Wpro and RoBERTa ===
        self.cross_attention = BiDirectionalCrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attn_heads,
            dropout=0.1
        )
        
        # === Post-attention layers for each branch ===
        # Prosody branch post-attention layers
        self.prosody_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=post_attention_layers
        )
        
        # Text branch post-attention layers
        self.text_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=post_attention_layers
        )
        
        # Pooling for both branches
        self.prosody_pooling = SelfAttentionPooling(hidden_dim)
        self.text_pooling = SelfAttentionPooling(hidden_dim)
        
        # === Fusion of the two branches ===
        self.branch_fusion = FusionLayer(
            dim1=hidden_dim,
            dim2=hidden_dim,
            output_dim=hidden_dim,
            fusion_type='gated'
        )
        
        # === Fusion goes through layers to predict A-V values ===
        av_layers = []
        for i in range(av_prediction_layers):
            if i == 0:
                av_layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
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
        
        # A-V prediction heads
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)
        
        # === A-V refinement module (replaces graph network) ===
        self.av_refinement = AVRefinementModule(
            av_dim=2,  # [arousal, valence]
            hidden_dim=hidden_dim,
            num_layers=av_refinement_layers
        )
        
        # === Separate WavLM branch for final fusion ===
        self.wavlm_final = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_final_dim = self.wavlm_final.config.hidden_size  # Always 768 for WavLM-base-plus
        
        if freeze_wavlm:
            for param in self.wavlm_final.parameters():
                param.requires_grad = False
        
        self.wavlm_pooling = SelfAttentionPooling(self.wavlm_final_dim)
        
        # === Final fusion: WavLM + branch_fused + AV_refined -> Emotion ===
        # CHANGED: Using branch_fused (256) instead of arousal_emb (256) + valence_emb (256)
        # Input dim: 768 (WavLM) + 256 (Branch Fused) + 256 (AV Refined) = 1280
        input_fusion_dim = self.wavlm_final_dim + hidden_dim * 2
        
        self.final_fusion = nn.Sequential(
            nn.Linear(input_fusion_dim, hidden_dim * 2),  # 1280 → 512
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),  # 512 → 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Emotion classifier with deep architecture
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # 256 → 128
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)  # 128 → 4
        )
    
    def forward(self, audio_input, text_input=None, audio_attention_mask=None, text_attention_mask=None):
        """
        Args:
            audio_input: [B, T_audio] - raw audio waveform
            text_input: [B, T_text] - tokenized text input_ids from transcript
            audio_attention_mask: [B, T_audio] - attention mask for audio
            text_attention_mask: [B, T_text] - attention mask for text
        Returns:
            dict with:
                - emotion_logits: [B, num_emotions]
                - arousal_value: [B, 1]
                - valence_value: [B, 1]
        """
        # === Step 1: Extract prosody features from audio (Wpro) ===
        wpro_output = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_features = wpro_output.last_hidden_state  # [B, T_audio, 768]
        wpro_projected = self.prosody_projection(wpro_features)  # [B, T_audio, hidden_dim]
        
        # === Step 2: Extract semantic features from text (RoBERTa) ===
        if text_input is None:
            # Fallback: if no text provided, create dummy text features
            batch_size = audio_input.size(0)
            text_projected = torch.zeros(batch_size, wpro_projected.size(1), wpro_projected.size(2), 
                                        device=audio_input.device)
        else:
            roberta_output = self.roberta(text_input, attention_mask=text_attention_mask)
            text_features = roberta_output.last_hidden_state  # [B, T_text, 768]
            text_projected = self.text_projection(text_features)  # [B, T_text, hidden_dim]
        
        # === Step 3: Cross-Attention between Wpro and RoBERTa ===
        wpro_attended, text_attended = self.cross_attention(
            wpro_projected,
            text_projected
        )  # Both [B, T, hidden_dim]
        
        # === Step 4: Each branch goes through post-attention layers ===
        wpro_refined = self.prosody_post_attention(wpro_attended)  # [B, T, hidden_dim]
        text_refined = self.text_post_attention(text_attended)  # [B, T, hidden_dim]
        
        # Pool both branches
        wpro_pooled, _ = self.prosody_pooling(wpro_refined)  # [B, hidden_dim]
        text_pooled, _ = self.text_pooling(text_refined)  # [B, hidden_dim]
        
        # Store for analysis (optional)
        arousal_embedding = wpro_pooled
        valence_embedding = text_pooled
        
        # === Step 5: Fusion of the two branches ===
        branch_fused = self.branch_fusion(wpro_pooled, text_pooled)  # [B, hidden_dim]
        
        # === Step 6: Fusion goes through layers -> outputs A-V values ===
        av_features = self.av_prediction_layers(branch_fused)  # [B, hidden_dim]
        
        arousal_value = self.arousal_head(av_features)  # [B, 1]
        valence_value = self.valence_head(av_features)  # [B, 1]
        
        # === Step 7: A-V values go through refinement module ===
        av_values = torch.cat([arousal_value, valence_value], dim=-1)  # [B, 2]
        av_refined = self.av_refinement(av_values)  # [B, hidden_dim]
        
        # === Step 8: Separate WavLM branch for final fusion ===
        wavlm_output = self.wavlm_final(audio_input, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_output.last_hidden_state  # [B, T, 768]
        wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)  # [B, 768]
        
        # === Step 9: Final fusion: WavLM + branch_fused + AV_refined -> Emotion ===
        # CHANGED: Using branch_fused instead of separate arousal/valence embeddings
        final_representation = torch.cat([
            wavlm_pooled,         # WavLM: [B, 768]
            branch_fused,         # Fused Prosody+Text: [B, hidden_dim]
            av_refined            # AV_refined: [B, hidden_dim]
        ], dim=-1)  # [B, 768 + 256 + 256] = [B, 1280]
        
        # Final fusion and classification
        fused_features = self.final_fusion(final_representation)  # [B, hidden_dim]
        emotion_logits = self.emotion_classifier(fused_features)  # [B, num_emotions]
        
        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'arousal_embedding': arousal_embedding,
            'valence_embedding': valence_embedding,
            'av_features': av_features,
            'branch_fused': branch_fused,
            'av_refined': av_refined,
            'wpro_attended': wpro_attended,
            'text_attended': text_attended
        }


if __name__ == "__main__":
    # Test the model
    model = Method2CrossAttentionAV_FusionConcat(num_emotions=4, hidden_dim=256)
    
    # Dummy input
    batch_size = 2
    audio_length = 16000 * 3  # 3 seconds at 16kHz
    audio_input = torch.randn(batch_size, audio_length)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio_input)
    
    print("Method 2 Variant (Fusion Concat) Test:")
    print(f"Emotion logits shape: {outputs['emotion_logits'].shape}")
    print(f"Arousal value shape: {outputs['arousal_value'].shape}")
    print(f"Valence value shape: {outputs['valence_value'].shape}")
    print(f"Branch fused shape: {outputs['branch_fused'].shape}")
    print(f"AV refined shape: {outputs['av_refined'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
