"""
Emotion2Vec embedding model for MSP-Podcast.
Takes precomputed emotion2vec embeddings (768-d) and predicts emotion class + A/V.
"""
import torch
import torch.nn as nn


class Emotion2VecAV(nn.Module):
    def __init__(
        self,
        num_emotions=4,
        input_dim=768,
        hidden_dim=256,
        av_refinement_layers=2
    ):
        super().__init__()

        self.embedding_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        av_layers = []
        for _ in range(av_refinement_layers):
            av_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
        self.av_prediction = nn.Sequential(*av_layers)

        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

        self.av_refinement = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        fusion_input_dim = hidden_dim * 2
        self.final_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
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

    def forward(self, embedding):
        x = self.embedding_proj(embedding)
        av_features = self.av_prediction(x)
        arousal_value = self.arousal_head(av_features)
        valence_value = self.valence_head(av_features)

        av_values = torch.cat([arousal_value, valence_value], dim=-1)
        av_refined = self.av_refinement(av_values)

        fused = self.final_fusion(torch.cat([x, av_refined], dim=-1))
        emotion_logits = self.emotion_classifier(fused)

        return {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
        }


if __name__ == "__main__":
    model = Emotion2VecAV()
    dummy = torch.randn(2, 768)
    out = model(dummy)
    print({k: v.shape for k, v in out.items()})
