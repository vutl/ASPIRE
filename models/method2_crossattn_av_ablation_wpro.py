import torch
import torch.nn as nn
from models.method2_crossattn_av import Method2CrossAttentionAV


class Method2CrossAttentionAV_AblationWPRO(Method2CrossAttentionAV):
    """Ablation: final fusion = wavlm + av_refined + wpro (prosody pooled)
    Inherits all components from `Method2CrossAttentionAV` and replaces
    the final fusion + classifier to accept the smaller concatenation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Recreate final_fusion to match new input dim:
        # wavlm_final_dim + av_refined (hidden_dim) + wpro pooled (hidden_dim)
        input_fusion_dim = self.wavlm_final_dim + self.prosody_projection.out_features + self.prosody_projection.out_features
        hidden_dim = self.prosody_projection.out_features

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

        # Recreate emotion classifier keeping same pattern
        out_classes = 4
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, out_classes)
        )

    def forward(self, audio_input, text_input=None, audio_attention_mask=None, text_attention_mask=None):
        # === Prosody (Wpro) ===
        wpro_output = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_features = wpro_output.last_hidden_state
        wpro_projected = self.prosody_projection(wpro_features)

        # === Text ===
        if text_input is None:
            text_projected = torch.zeros_like(wpro_projected)
        else:
            text_output = self.roberta(text_input, attention_mask=text_attention_mask)
            text_features = text_output.last_hidden_state
            text_projected = self.text_projection(text_features)

        # === Cross-attention ===
        wpro_attended, text_attended = self.cross_attention(
            wpro_projected,
            text_projected
        )

        # === Post-attention ===
        wpro_refined = self.prosody_post_attention(wpro_attended)
        text_refined = self.text_post_attention(text_attended)

        # === Pooling to get embeddings ===
        arousal_embedding, _ = self.prosody_pooling(wpro_refined)
        valence_embedding, _ = self.text_pooling(text_refined)

        # === Branch fusion -> A-V prediction ===
        branch_fused = self.branch_fusion(arousal_embedding, valence_embedding)
        av_feat = self.av_prediction_layers(branch_fused)
        arousal_value = self.arousal_head(av_feat)
        valence_value = self.valence_head(av_feat)

        # === A-V refinement ===
        av_values = torch.cat([arousal_value, valence_value], dim=1)
        av_refined = self.av_refinement(av_values)

        # === WavLM final pooling ===
        wavlm_out = self.wavlm_final(audio_input, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_out.last_hidden_state
        wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)

        # === Final fusion: wavlm + av_refined + wpro (arousal_embedding used as prosody pooled) ===
        final_input = torch.cat([wavlm_pooled, av_refined, arousal_embedding], dim=1)
        final_feat = self.final_fusion(final_input)
        emotion_logits = self.emotion_classifier(final_feat)

        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'arousal_embedding': arousal_embedding,
            'valence_embedding': valence_embedding,
            'av_refined': av_refined,
            'wavlm_pooled': wavlm_pooled
        }
