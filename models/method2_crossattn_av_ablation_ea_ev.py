"""
Ablation of Method2 Cross-Attention A-V:
Final fusion uses three branches: AV_refined + eA + eV

This module mirrors `Method2CrossAttentionAV` but produces two learned
embeddings `eA` and `eV` (projected from `av_features`) and concatenates
them with `av_refined` before the final classifier.
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
)


class Method2CrossAttentionAV_Ablation_eA_eV(nn.Module):
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name=None,
        roberta_model_name="roberta-base",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=False,
        freeze_roberta=False,
        num_cross_attn_heads=8,
        post_attention_layers=2,
        av_prediction_layers=3,
    ):
        super().__init__()

        prosody_model_name = wavlm_prosody_model_name if wavlm_prosody_model_name else wavlm_model_name
        self.use_wpro = (wavlm_prosody_model_name is not None)

        if self.use_wpro:
            from transformers import WavLMModel
            self.wavlm_prosody = WavLMModel.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size
        else:
            self.wavlm_prosody = Wav2Vec2Model.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size

        if freeze_wavlm:
            for p in self.wavlm_prosody.parameters():
                p.requires_grad = False

        self.prosody_projection = nn.Linear(self.wavlm_dim, hidden_dim)

        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        if freeze_roberta:
            for p in self.roberta.parameters():
                p.requires_grad = False
        self.text_projection = nn.Linear(self.roberta_dim, hidden_dim)

        self.cross_attention = BiDirectionalCrossAttention(dim=hidden_dim, num_heads=num_cross_attn_heads, dropout=0.1)

        self.prosody_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True),
            num_layers=post_attention_layers,
        )
        self.text_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True),
            num_layers=post_attention_layers,
        )

        self.prosody_pooling = SelfAttentionPooling(hidden_dim)
        self.text_pooling = SelfAttentionPooling(hidden_dim)

        self.branch_fusion = FusionLayer(dim1=hidden_dim, dim2=hidden_dim, output_dim=hidden_dim, fusion_type='gated')

        av_layers = []
        for i in range(av_prediction_layers):
            av_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        self.av_prediction_layers = nn.Sequential(*av_layers)

        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

        self.av_refinement = AVRefinementModule(av_dim=2, hidden_dim=hidden_dim, num_layers=av_refinement_layers)

        # final wavlm branch
        self.wavlm_final = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_final_dim = self.wavlm_final.config.hidden_size
        if freeze_wavlm:
            for p in self.wavlm_final.parameters():
                p.requires_grad = False
        self.wavlm_pooling = SelfAttentionPooling(self.wavlm_final_dim)

        # eA and eV projection heads (produce embeddings from av_features)
        self.ea_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.ev_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # final fusion input: av_refined + eA + eV  (three branches)
        input_fusion_dim = hidden_dim * 3
        self.final_fusion = nn.Sequential(
            nn.Linear(input_fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

    def forward(self, audio_input, text_input=None, audio_attention_mask=None, text_attention_mask=None):
        # prosody
        wpro_out = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_feats = wpro_out.last_hidden_state
        wpro_proj = self.prosody_projection(wpro_feats)

        # text
        if text_input is None:
            batch_size = audio_input.size(0)
            text_proj = torch.zeros(batch_size, wpro_proj.size(1), wpro_proj.size(2), device=audio_input.device)
        else:
            r_out = self.roberta(text_input, attention_mask=text_attention_mask)
            text_feats = r_out.last_hidden_state
            text_proj = self.text_projection(text_feats)

        # cross-attention
        w_att, t_att = self.cross_attention(wpro_proj, text_proj)

        w_ref = self.prosody_post_attention(w_att)
        t_ref = self.text_post_attention(t_att)

        w_pooled, _ = self.prosody_pooling(w_ref)
        t_pooled, _ = self.text_pooling(t_ref)

        arousal_embedding = w_pooled
        valence_embedding = t_pooled

        branch_fused = self.branch_fusion(w_pooled, t_pooled)

        av_feat = self.av_prediction_layers(branch_fused)
        arousal_value = self.arousal_head(av_feat)
        valence_value = self.valence_head(av_feat)

        av_values = torch.cat([arousal_value, valence_value], dim=-1)
        av_refined = self.av_refinement(av_values)

        # eA / eV embeddings from av features
        eA = self.ea_proj(av_feat)
        eV = self.ev_proj(av_feat)

        # final fusion: use av_refined + eA + eV (three branches)
        final_in = torch.cat([av_refined, eA, eV], dim=-1)
        fused = self.final_fusion(final_in)
        logits = self.emotion_classifier(fused)

        return {
            'emotion_logits': logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'eA': eA,
            'eV': eV,
            'av_refined': av_refined,
            'branch_fused': branch_fused,
        }


if __name__ == '__main__':
    m = Method2CrossAttentionAV_Ablation_eA_eV()
    a = torch.randn(2, 16000 * 3)
    t = torch.randint(0, 2000, (2, 32))
    tm = torch.ones_like(t)
    out = m(a, t, tm)
    print({k: v.shape for k, v in out.items()})
