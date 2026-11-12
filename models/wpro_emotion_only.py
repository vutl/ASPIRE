"""
W-PRO Emotion-only model
Simple single-branch WavLM prosody model that outputs only emotion logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel

from models.base_components import SelfAttentionPooling


class WProEmotionOnly(nn.Module):
    def __init__(self, num_emotions=4, wpro_model_name="Orange/Speaker-wavLM-pro", hidden_dim=256, freeze_wpro=True, dropout=0.1):
        super().__init__()
        self.wpro = WavLMModel.from_pretrained(wpro_model_name)
        self.wpro_dim = self.wpro.config.hidden_size

        if freeze_wpro:
            for p in self.wpro.parameters():
                p.requires_grad = False

        self.pooling = SelfAttentionPooling(self.wpro_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.wpro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions)
        )

    def forward(self, audio_input, audio_attention_mask=None):
        out = self.wpro(audio_input, attention_mask=audio_attention_mask)
        feats = out.last_hidden_state  # [B, T, D]
        pooled, _ = self.pooling(feats)
        emb = self.proj(pooled)
        logits = self.classifier(emb)
        return { 'emotion_logits': logits }


if __name__ == '__main__':
    m = WProEmotionOnly()
    a = torch.randn(2, 16000 * 3)
    out = m(a)
    print({k: v.shape for k, v in out.items()})
