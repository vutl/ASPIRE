"""
Method 2 (W-PRO) ablation: NO A/V supervision.

Architecture keeps the same three backbones and cross-attention pipeline as the
original Method2CrossAttentionAV, but removes:
- Arousal / Valence prediction heads
- A/V refinement module
- A/V supervision

Final fusion uses only:
- global WavLM pooled representation
- W-PRO branch pooled representation (former eA branch)
- RoBERTa branch pooled representation (former eV branch)
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel, AutoTokenizer

from models.base_components import (
    SelfAttentionPooling,
    BiDirectionalCrossAttention,
)


class Method2CrossAttentionAV_AblationNoAVSup(nn.Module):
    """Classification-only ablation of Method 2.

    This model preserves the three-backbone architecture:
      1) W-PRO / WavLM prosody branch
      2) RoBERTa text branch
      3) global WavLM branch

    but removes all A/V prediction and refinement components.
    """

    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        wavlm_prosody_model_name=None,
        roberta_model_name="roberta-base",
        hidden_dim=256,
        av_refinement_layers=2,   # kept for config compatibility (unused)
        freeze_wavlm=False,
        freeze_roberta=False,
        num_cross_attn_heads=8,
        post_attention_layers=2,
        av_prediction_layers=3,   # kept for config compatibility (unused)
    ):
        super().__init__()
        self.freeze_wavlm = freeze_wavlm
        self.freeze_roberta = freeze_roberta

        # === Branch 1: WavLM / W-PRO prosody branch ===
        prosody_model_name = wavlm_prosody_model_name if wavlm_prosody_model_name else wavlm_model_name
        self.use_wpro = wavlm_prosody_model_name is not None

        if self.use_wpro:
            print(f"Loading W-PRO model: {prosody_model_name}")
            from transformers import WavLMModel
            self.wavlm_prosody = WavLMModel.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size
        else:
            print(f"Loading standard WavLM for prosody: {prosody_model_name}")
            self.wavlm_prosody = AutoModel.from_pretrained(prosody_model_name)
            self.wavlm_dim = self.wavlm_prosody.config.hidden_size

        if freeze_wavlm:
            for param in self.wavlm_prosody.parameters():
                param.requires_grad = False
            print(f"WavLM prosody frozen: {freeze_wavlm}")

        self.prosody_projection = nn.Linear(self.wavlm_dim, hidden_dim)

        # === Branch 2: RoBERTa semantic/text branch ===
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.text_projection = nn.Linear(self.roberta_dim, hidden_dim)

        # === Cross-attention + post-attention refinement ===
        self.cross_attention = BiDirectionalCrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attn_heads,
            dropout=0.1,
        )

        self.prosody_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=post_attention_layers,
        )

        self.text_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=post_attention_layers,
        )

        self.prosody_pooling = SelfAttentionPooling(hidden_dim)
        self.text_pooling = SelfAttentionPooling(hidden_dim)

        # === Separate WavLM branch for final fusion ===
        self.wavlm_final = AutoModel.from_pretrained(wavlm_model_name)
        self.wavlm_final_dim = self.wavlm_final.config.hidden_size

        if freeze_wavlm:
            for param in self.wavlm_final.parameters():
                param.requires_grad = False

        self.wavlm_pooling = SelfAttentionPooling(self.wavlm_final_dim)

        # === Final fusion: WavLM + prosody/text pooled embeddings -> Emotion ===
        input_fusion_dim = self.wavlm_final_dim + hidden_dim * 2

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

    def forward(
        self,
        audio_input,
        text_input=None,
        audio_attention_mask=None,
        text_attention_mask=None,
        use_guidance=True,          # kept for compatibility (unused)
        return_intermediate=False,
    ):
        # === Step 1: Prosody branch ===
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_wavlm):
            wpro_output = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_features = wpro_output.last_hidden_state
        wpro_projected = self.prosody_projection(wpro_features)

        # === Step 2: Text branch ===
        if text_input is None:
            batch_size = audio_input.size(0)
            text_projected = torch.zeros(
                batch_size,
                wpro_projected.size(1),
                wpro_projected.size(2),
                device=audio_input.device,
            )
        else:
            with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_roberta):
                roberta_output = self.roberta(text_input, attention_mask=text_attention_mask)
            text_features = roberta_output.last_hidden_state
            text_projected = self.text_projection(text_features)

        # === Step 3: Cross-attention ===
        wpro_attended, text_attended = self.cross_attention(wpro_projected, text_projected)

        # === Step 4: Post-attention refinement ===
        wpro_refined = self.prosody_post_attention(wpro_attended)
        text_refined = self.text_post_attention(text_attended)

        # === Step 5: Pool branch features ===
        wpro_pooled, _ = self.prosody_pooling(wpro_refined)
        text_pooled, _ = self.text_pooling(text_refined)

        # Keep naming for compatibility / analysis
        arousal_embedding = wpro_pooled
        valence_embedding = text_pooled

        # === Step 6: Global WavLM branch ===
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_wavlm):
            wavlm_output = self.wavlm_final(audio_input, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_output.last_hidden_state
        wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)

        # === Step 7: Final fusion without any A/V supervision/refinement ===
        final_representation = torch.cat(
            [
                wavlm_pooled,
                arousal_embedding,
                valence_embedding,
            ],
            dim=-1,
        )

        fused_features = self.final_fusion(final_representation)
        emotion_logits = self.emotion_classifier(fused_features)

        output = {
            "emotion_logits": emotion_logits,
            "arousal_embedding": arousal_embedding,
            "valence_embedding": valence_embedding,
            "wpro_attended": wpro_attended,
            "text_attended": text_attended,
        }

        if return_intermediate:
            output.update(
                {
                    "final_representation": final_representation,
                    "fused_features": fused_features,
                    "guidance_used": False,
                }
            )

        return output


if __name__ == "__main__":
    model = Method2CrossAttentionAV_AblationNoAVSup(num_emotions=4, hidden_dim=256)
    audio = torch.randn(2, 16000 * 3)
    text = torch.randint(0, 1000, (2, 128))
    mask = torch.ones(2, 128, dtype=torch.long)
    with torch.no_grad():
        out = model(audio, text_input=text, text_attention_mask=mask)
    print({k: v.shape for k, v in out.items() if hasattr(v, "shape")})
