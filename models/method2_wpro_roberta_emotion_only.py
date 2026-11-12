"""
WPRO (Orange/Speaker-wavLM-pro) + RoBERTa fusion model that outputs only emotion logits.

Drop-in replacement for trainer's model API: forward(audio_input, text_input, text_attention_mask)
returns {'emotion_logits': logits}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoModel


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x, mask=None):
        attn = self.attention(x)
        attn = F.softmax(attn, dim=1)
        pooled = torch.sum(x * attn, dim=1)
        return pooled


class Method2WPRORobertaEmotionOnly(nn.Module):
    """Prosody WavLM (WPRO) + RoBERTa fusion → emotion logits."""
    def __init__(
        self,
        num_emotions=4,
        wavlm_prosody_model_name="Orange/Speaker-wavLM-pro",
        roberta_model_name="roberta-base",
        hidden_dim=256,
        freeze_wpro=True,
        freeze_roberta=True,
        **kwargs
    ):
        super().__init__()

        # Prosody WavLM backbone
        self.wpro = Wav2Vec2Model.from_pretrained(wavlm_prosody_model_name)
        self.wpro_dim = self.wpro.config.hidden_size
        if freeze_wpro:
            for p in self.wpro.parameters():
                p.requires_grad = False

        # RoBERTa backbone
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size
        if freeze_roberta:
            for p in self.roberta.parameters():
                p.requires_grad = False

        # Pooling / projections
        self.prosody_pool = SelfAttentionPooling(self.wpro_dim)
        self.prosody_proj = nn.Sequential(
            nn.Linear(self.wpro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(self.roberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Fusion & classifier
        fusion_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)
        )

    def forward(self, audio_input, text_input=None, text_attention_mask=None):
        # Prosody pathway
        wout = self.wpro(audio_input)
        w_feats = wout.last_hidden_state
        w_pooled = self.prosody_pool(w_feats)
        w_emb = self.prosody_proj(w_pooled)

        # Text pathway
        if text_input is not None:
            tout = self.roberta(input_ids=text_input, attention_mask=text_attention_mask)
            t_feats = tout.last_hidden_state
            if text_attention_mask is not None:
                mask = text_attention_mask.unsqueeze(-1).float()
                t_sum = (t_feats * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                t_pooled = t_sum / denom
            else:
                t_pooled = t_feats.mean(dim=1)
            t_emb = self.text_proj(t_pooled)
        else:
            t_emb = torch.zeros_like(w_emb)

        fused = torch.cat([w_emb, t_emb], dim=-1)
        fused = self.fusion(fused)
        logits = self.classifier(fused)

        return {'emotion_logits': logits}


if __name__ == '__main__':
    m = Method2WPRORobertaEmotionOnly()
    a = torch.randn(2, 16000 * 3)
    t = torch.randint(0, 2000, (2, 32))
    tm = torch.ones_like(t)
    out = m(a, t, tm)
    print({k: v.shape for k, v in out.items()})
