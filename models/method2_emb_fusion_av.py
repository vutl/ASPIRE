"""
Lightweight fusion model for precomputed embeddings (wavlm, wpro, roberta).
Outputs: emotion logits + arousal/valence scalars (AV only).
"""
import torch
import torch.nn as nn


class ProjectionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class Method2EmbeddingFusionAV(nn.Module):
    def __init__(
        self,
        num_emotions=4,
        wavlm_dim=768,
        wpro_dim=1024,
        roberta_dim=768,
        hidden_dim=256,
        dropout=0.1,
        av_hidden=128,
    ):
        super().__init__()
        self.wavlm_proj = ProjectionBlock(wavlm_dim, hidden_dim, dropout)
        self.wpro_proj = ProjectionBlock(wpro_dim, hidden_dim, dropout)
        self.roberta_proj = ProjectionBlock(roberta_dim, hidden_dim, dropout)

        fusion_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_emotions),
        )

        self.av_trunk = nn.Sequential(
            nn.Linear(fusion_dim, av_hidden),
            nn.LayerNorm(av_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.arousal_head = nn.Linear(av_hidden, 1)
        self.valence_head = nn.Linear(av_hidden, 1)

    def forward(self, wavlm_emb, wpro_emb, roberta_emb):
        wavlm_h = self.wavlm_proj(wavlm_emb)
        wpro_h = self.wpro_proj(wpro_emb)
        roberta_h = self.roberta_proj(roberta_emb)

        fused = torch.cat([wavlm_h, wpro_h, roberta_h], dim=-1)
        fused = self.fusion(fused)

        emotion_logits = self.emotion_head(fused)
        av_feat = self.av_trunk(fused)
        arousal = self.arousal_head(av_feat)
        valence = self.valence_head(av_feat)

        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal,
            'valence_value': valence,
        }
