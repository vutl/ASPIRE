"""
Variant of Method2CrossAttentionAV that accepts precomputed emotion2vec embeddings
for the final acoustic branch while preserving W‑PRO for the prosody branch.

This file is new and purposely does not modify `models/method2_crossattn_av.py`.
When `use_emotion2vec=True`, the model expects batches to include ``audio_embedding``
of shape [B, emotion2vec_dim] and will project it to the final pooled audio representation.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer

from models.base_components import (
    SelfAttentionPooling,
    BiDirectionalCrossAttention,
    FusionLayer,
    AVRefinementModule,
    ProsodyEncoder,
    TextSemanticEncoder
)


class Method2CrossAttentionAV_Emotion2Vec(nn.Module):
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name=None,  # keep W‑PRO for prosody branch when provided
        roberta_model_name="roberta-base",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=False,
        freeze_roberta=False,
        num_cross_attn_heads=8,
        post_attention_layers=2,
        av_prediction_layers=3,
        use_emotion2vec=False,
        emotion2vec_dim=768
    ):
        super().__init__()

        self.use_emotion2vec = use_emotion2vec
        self.emotion2vec_dim = emotion2vec_dim

        # === Prosody branch (W‑PRO if provided) ===
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
            for p in self.wavlm_prosody.parameters():
                p.requires_grad = False

        self.prosody_projection = nn.Linear(self.wavlm_dim, hidden_dim)

        # === Text branch (RoBERTa) ===
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

        if freeze_roberta:
            for p in self.roberta.parameters():
                p.requires_grad = False

        self.text_projection = nn.Linear(self.roberta_dim, hidden_dim)

        # === Cross-attention and post-attention layers ===
        self.cross_attention = BiDirectionalCrossAttention(dim=hidden_dim, num_heads=num_cross_attn_heads, dropout=0.1)

        self.prosody_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True),
            num_layers=post_attention_layers
        )
        self.text_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True),
            num_layers=post_attention_layers
        )

        self.prosody_pooling = SelfAttentionPooling(hidden_dim)
        self.text_pooling = SelfAttentionPooling(hidden_dim)

        self.branch_fusion = FusionLayer(dim1=hidden_dim, dim2=hidden_dim, output_dim=hidden_dim, fusion_type='gated')

        # === A-V prediction layers ===
        av_layers = []
        for i in range(av_prediction_layers):
            av_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.av_prediction_layers = nn.Sequential(*av_layers)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

        self.av_refinement = AVRefinementModule(av_dim=2, hidden_dim=hidden_dim, num_layers=av_refinement_layers)

        # === Final acoustic branch: either WavLM (default) or projected emotion2vec ===
        if not self.use_emotion2vec:
            self.wavlm_final = Wav2Vec2Model.from_pretrained(wavlm_model_name)
            self.wavlm_final_dim = self.wavlm_final.config.hidden_size
            if freeze_wavlm:
                for p in self.wavlm_final.parameters():
                    p.requires_grad = False
            self.wavlm_pooling = SelfAttentionPooling(self.wavlm_final_dim)
        else:
            # Do not load WavLM for final branch; project precomputed embedding to a pooled dim
            # Use a default target dim of 768 to match expected concatenation size
            self.wavlm_final_dim = 768
            self.emotion2vec_proj = nn.Sequential(
                nn.LayerNorm(self.emotion2vec_dim),
                nn.Linear(self.emotion2vec_dim, self.wavlm_final_dim),
                nn.GELU(),
                nn.LayerNorm(self.wavlm_final_dim)
            )

        # === Final fusion and classifier (same as original) ===
        input_fusion_dim = self.wavlm_final_dim + hidden_dim * 3
        self.final_fusion = nn.Sequential(
            nn.Linear(input_fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)
        )

    def forward(self, audio_input=None, text_input=None, audio_attention_mask=None, text_attention_mask=None, audio_embedding=None):
        # Prosody W‑PRO branch (unchanged)
        # audio_input is required for prosody branch; if missing, create zero dummy tensor
        if audio_input is None:
            raise ValueError("audio_input is required for prosody (W-PRO) branch")

        wpro_output = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_features = wpro_output.last_hidden_state
        wpro_projected = self.prosody_projection(wpro_features)

        # Text branch
        if text_input is None:
            batch_size = audio_input.size(0)
            text_projected = torch.zeros(batch_size, wpro_projected.size(1), wpro_projected.size(2), device=audio_input.device)
        else:
            roberta_output = self.roberta(text_input, attention_mask=text_attention_mask)
            text_features = roberta_output.last_hidden_state
            text_projected = self.text_projection(text_features)

        # Cross-attention
        wpro_attended, text_attended = self.cross_attention(wpro_projected, text_projected)

        wpro_refined = self.prosody_post_attention(wpro_attended)
        text_refined = self.text_post_attention(text_attended)

        wpro_pooled, _ = self.prosody_pooling(wpro_refined)
        text_pooled, _ = self.text_pooling(text_refined)

        arousal_embedding = wpro_pooled
        valence_embedding = text_pooled

        branch_fused = self.branch_fusion(wpro_pooled, text_pooled)

        av_features = self.av_prediction_layers(branch_fused)
        arousal_value = self.arousal_head(av_features)
        valence_value = self.valence_head(av_features)

        av_values = torch.cat([arousal_value, valence_value], dim=-1)
        av_refined = self.av_refinement(av_values)

        # Final acoustic pooled representation: either from WavLM or projected emotion2vec
        if self.use_emotion2vec:
            if audio_embedding is None:
                raise ValueError("audio_embedding must be provided when use_emotion2vec=True")
            # audio_embedding: [B, D]
            wavlm_pooled = self.emotion2vec_proj(audio_embedding)
        else:
            wavlm_output = self.wavlm_final(audio_input, attention_mask=audio_attention_mask)
            wavlm_features = wavlm_output.last_hidden_state
            wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)

        final_representation = torch.cat([
            wavlm_pooled,
            arousal_embedding,
            valence_embedding,
            av_refined
        ], dim=-1)

        fused_features = self.final_fusion(final_representation)
        emotion_logits = self.emotion_classifier(fused_features)

        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'arousal_embedding': arousal_embedding,
            'valence_embedding': valence_embedding,
            'av_refined': av_refined,
            'branch_fused': branch_fused
        }


if __name__ == "__main__":
    print("This module provides Method2CrossAttentionAV_Emotion2Vec model. Import and use it from training scripts.")
