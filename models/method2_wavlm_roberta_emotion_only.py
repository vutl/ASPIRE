"""
WavLM (base-plus) + RoBERTa fusion model that outputs only emotion logits.

Designed as a drop-in replacement for `Method2CrossAttentionAV` in the k-fold
trainer: constructor accepts the common config args (extra args ignored) and
`forward(audio_input, text_input, text_attention_mask)` returns a dict with
`'emotion_logits'` key.
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
        # x: [B, T, D]
        attn = self.attention(x)  # [B, T, 1]
        attn = F.softmax(attn, dim=1)
        pooled = torch.sum(x * attn, dim=1)  # [B, D]
        return pooled


class Method2WavLMRobertaEmotionOnly(nn.Module):
    """Simple fusion: audio (WavLM) + text (RoBERTa) → emotion logits."""
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        roberta_model_name="roberta-base",
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=True,
        freeze_roberta=True,
        **kwargs
    ):
        super().__init__()

        # Backbones
        self.wavlm = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_dim = self.wavlm.config.hidden_size
        if freeze_wavlm:
            for p in self.wavlm.parameters():
                p.requires_grad = False

        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size
        if freeze_roberta:
            for p in self.roberta.parameters():
                p.requires_grad = False

        # Pooling / projection
        self.audio_pool = SelfAttentionPooling(self.wavlm_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(self.roberta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(self.wavlm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Fusion and classifier
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
        """
        Args as used by the k-fold trainer:
            audio_input: [B, T_audio]
            text_input: [B, T_text]
            text_attention_mask: [B, T_text]

        Returns: {'emotion_logits': [B, num_emotions]}
        """
        # Audio pathway
        wavlm_out = self.wavlm(audio_input)
        audio_feats = wavlm_out.last_hidden_state  # [B, T_a, D]
        audio_pooled = self.audio_pool(audio_feats)  # [B, D]
        audio_emb = self.audio_proj(audio_pooled)  # [B, hidden]

        # Text pathway (if provided)
        if text_input is not None:
            text_out = self.roberta(input_ids=text_input, attention_mask=text_attention_mask)
            text_feats = text_out.last_hidden_state  # [B, T_t, D]
            # mean pool with mask if provided
            if text_attention_mask is not None:
                mask = text_attention_mask.unsqueeze(-1).float()
                text_sum = (text_feats * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                text_pooled = text_sum / denom
            else:
                text_pooled = text_feats.mean(dim=1)
            text_emb = self.text_proj(text_pooled)
        else:
            # If no text, use zero vector
            text_emb = torch.zeros_like(audio_emb)

        # Fusion
        fused = torch.cat([audio_emb, text_emb], dim=-1)
        fused = self.fusion(fused)

        logits = self.classifier(fused)

        return {'emotion_logits': logits}


if __name__ == '__main__':
    # smoke test
    model = Method2WavLMRobertaEmotionOnly()
    a = torch.randn(2, 16000 * 3)
    t = torch.randint(0, 2000, (2, 32))
    m = torch.ones_like(t)
    out = model(a, t, m)
    print({k: v.shape for k, v in out.items()})
