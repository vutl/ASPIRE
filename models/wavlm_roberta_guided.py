"""Two-stream WavLM+RoBERTa reference with predicted AV guidance.

This is a reconstruction of the two-stream guided reference in Table 7; the
original paper's exact reference implementation/checkpoints are unavailable.
"""
import torch
import torch.nn as nn

from models.base_components import AVRefinementModule, FusionLayer
from models.method2_crossattn_av_ablation_no_avsup_no_global import (
    Method2CrossAttentionAV_AblationNoAVSupNoGlobal,
)


class WavLMRobertaGuided(Method2CrossAttentionAV_AblationNoAVSupNoGlobal):
    def __init__(self, hidden_dim=256, av_refinement_layers=2, av_prediction_layers=3, **kwargs):
        super().__init__(hidden_dim=hidden_dim, wavlm_prosody_model_name=None, **kwargs)
        self.branch_fusion = FusionLayer(hidden_dim, hidden_dim, hidden_dim, fusion_type='gated')
        self.av_prediction_layers = nn.Sequential(*[
            layer for _ in range(av_prediction_layers)
            for layer in (nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                          nn.ReLU(), nn.Dropout(0.1))
        ])
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)
        self.av_refinement = AVRefinementModule(2, hidden_dim, av_refinement_layers)
        self.final_fusion = nn.Sequential(
            # Same late-fusion MLP as Full ASPIRE, with eGlobal omitted.
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(0.2),
        )

    def forward(self, audio_input, text_input=None, audio_attention_mask=None, text_attention_mask=None):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_wavlm):
            audio_features = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask).last_hidden_state
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_roberta):
            text_features = self.roberta(text_input, attention_mask=text_attention_mask).last_hidden_state
        audio = self.prosody_projection(audio_features)
        text = self.text_projection(text_features)
        audio, text = self.cross_attention(audio, text)
        audio, _ = self.prosody_pooling(self.prosody_post_attention(audio))
        text, _ = self.text_pooling(self.text_post_attention(text))
        features = self.av_prediction_layers(self.branch_fusion(audio, text))
        arousal, valence = self.arousal_head(features), self.valence_head(features)
        av = self.av_refinement(torch.cat([arousal, valence], dim=-1))
        logits = self.emotion_classifier(self.final_fusion(torch.cat([audio, text, av], dim=-1)))
        return {'emotion_logits': logits, 'arousal_value': arousal, 'valence_value': valence}
