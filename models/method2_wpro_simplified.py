"""
Method 2 W-PRO Simplified: Cross-Attention A-V-D Architecture with Simplified Final Fusion
Architecture:
1. WavLM_prosody (Wpro) extracts acoustic/prosody features from audio
2. RoBERTa extracts semantic/linguistic features from transcript text
3. Wpro and RoBERTa cross-attention with each other
4. Each branch goes through some layers after cross-attention
5. Fusion of the two branches
6. Fusion goes through some layers -> outputs A-V-D values (Arousal, Valence, Dominance)
7. A-V-D values go through refinement module (replaces graph network)
8. Separate WavLM branch extracts global acoustic features
9. **SIMPLIFIED** Final fusion: WavLM + AVD_refined ONLY -> Emotion labels

Difference from full Method2:
- Final fusion only takes [WavLM_pooled (768) + AV_refined (256)] = 1024 dim
- Does NOT include arousal_embedding, valence_embedding, branch_fused
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


class Method2WPROSimplified(nn.Module):
    """
    Method 2 W-PRO Simplified: Cross-Attention A-V-D with simplified final fusion
    
    Same as Method2CrossAttentionAVD but final fusion only takes:
    [WavLM_pooled + AV_refined] instead of [WavLM + arousal_emb + valence_emb + AV_refined]
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
        prosody_model_name = wavlm_prosody_model_name if wavlm_prosody_model_name else wavlm_model_name
        self.use_wpro = (wavlm_prosody_model_name is not None)
        
        if self.use_wpro:
            print(f"Loading W-PRO model: {prosody_model_name}")
            from transformers import WavLMModel
            self.wavlm_prosody = WavLMModel.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size
        else:
            print(f"Loading standard WavLM for prosody: {prosody_model_name}")
            self.wavlm_prosody = Wav2Vec2Model.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size
        
        if freeze_wavlm:
            for param in self.wavlm_prosody.parameters():
                param.requires_grad = False
            print(f"WavLM prosody frozen: {freeze_wavlm}")
        
        self.prosody_projection = nn.Linear(self.wavlm_dim, hidden_dim)
        
        # === Branch 2: RoBERTa for semantic/text ===
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        self.text_projection = nn.Linear(self.roberta_dim, hidden_dim)
        
        # === Cross-Attention ===
        self.cross_attention = BiDirectionalCrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attn_heads,
            dropout=0.1
        )
        
        # === Post-attention layers ===
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
        
        self.prosody_pooling = SelfAttentionPooling(hidden_dim)
        self.text_pooling = SelfAttentionPooling(hidden_dim)
        
        # === Branch fusion ===
        self.branch_fusion = FusionLayer(
            dim1=hidden_dim,
            dim2=hidden_dim,
            output_dim=hidden_dim,
            fusion_type='gated'
        )
        
        # === A-V-D prediction layers ===
        av_layers = []
        for i in range(av_prediction_layers):
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
        
        # === A-V-D refinement ===
        self.av_refinement = AVRefinementModule(
            av_dim=3,
            hidden_dim=hidden_dim,
            num_layers=av_refinement_layers
        )
        
        # === Separate WavLM branch ===
        self.wavlm_final = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_final_dim = self.wavlm_final.config.hidden_size
        
        if freeze_wavlm:
            for param in self.wavlm_final.parameters():
                param.requires_grad = False
        
        self.wavlm_pooling = SelfAttentionPooling(self.wavlm_final_dim)
        
        # === SIMPLIFIED Final fusion: WavLM + AV_refined ONLY ===
        # Input: [wavlm_pooled (768) + av_refined (256)] = 1024
        input_fusion_dim = self.wavlm_final_dim + hidden_dim  # 768 + 256 = 1024
        
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
            text_input: [B, T_text] - tokenized text input_ids
            audio_attention_mask: [B, T_audio]
            text_attention_mask: [B, T_text]
        Returns:
            dict with emotion_logits, arousal_value, valence_value, dominance_value
        """
        # Extract prosody features
        wpro_output = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_features = wpro_output.last_hidden_state
        wpro_projected = self.prosody_projection(wpro_features)
        
        # Extract text features
        if text_input is None:
            batch_size = audio_input.size(0)
            text_projected = torch.zeros(batch_size, wpro_projected.size(1), wpro_projected.size(2),
                                        device=audio_input.device)
        else:
            roberta_output = self.roberta(text_input, attention_mask=text_attention_mask)
            text_features = roberta_output.last_hidden_state
            text_projected = self.text_projection(text_features)
        
        # Cross-attention
        wpro_attended, text_attended = self.cross_attention(wpro_projected, text_projected)
        
        # Post-attention processing
        wpro_refined = self.prosody_post_attention(wpro_attended)
        text_refined = self.text_post_attention(text_attended)
        
        # Pool
        wpro_pooled, _ = self.prosody_pooling(wpro_refined)
        text_pooled, _ = self.text_pooling(text_refined)
        
        # Store embeddings (not used in final fusion, but kept for compatibility)
        arousal_embedding = wpro_pooled
        valence_embedding = text_pooled
        
        # Branch fusion
        branch_fused = self.branch_fusion(wpro_pooled, text_pooled)
        
        # A-V-D prediction
        av_features = self.av_prediction_layers(branch_fused)
        arousal_value = self.arousal_head(av_features)
        valence_value = self.valence_head(av_features)
        dominance_value = self.dominance_head(av_features)
        
        # A-V-D refinement
        avd_values = torch.cat([arousal_value, valence_value, dominance_value], dim=-1)
        av_refined = self.av_refinement(avd_values)
        
        # Separate WavLM branch
        wavlm_output = self.wavlm_final(audio_input, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_output.last_hidden_state
        wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)
        
        # === SIMPLIFIED Final fusion: ONLY [WavLM + AV_refined] ===
        final_representation = torch.cat([
            wavlm_pooled,    # [B, 768]
            av_refined       # [B, 256]
        ], dim=-1)  # [B, 1024]
        
        # Final fusion and classification
        fused_features = self.final_fusion(final_representation)
        emotion_logits = self.emotion_classifier(fused_features)
        
        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'dominance_value': dominance_value,
            'arousal_embedding': arousal_embedding,
            'valence_embedding': valence_embedding,
            'av_features': av_features,
            'branch_fused': branch_fused,
            'av_refined': av_refined,
            'wpro_attended': wpro_attended,
            'text_attended': text_attended
        }


if __name__ == "__main__":
    print("Testing Method2WPROSimplified...")
    model = Method2WPROSimplified(num_emotions=4, hidden_dim=256)
    
    batch_size = 2
    audio_length = 16000 * 3
    audio_input = torch.randn(batch_size, audio_length)
    
    with torch.no_grad():
        outputs = model(audio_input)
    
    print(f"Emotion logits: {outputs['emotion_logits'].shape}")
    print(f"Arousal: {outputs['arousal_value'].shape}")
    print(f"Valence: {outputs['valence_value'].shape}")
    print(f"Dominance: {outputs['dominance_value'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("\n=== Simplified Final Fusion ===")
    print("Final fusion input: [WavLM_pooled + AV_refined] = 1024 dim")
    print("Removed: arousal_embedding, valence_embedding, branch_fused")
