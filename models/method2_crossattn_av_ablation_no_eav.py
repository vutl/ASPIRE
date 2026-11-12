import torch
import torch.nn as nn

from models.method2_crossattn_av import Method2CrossAttentionAV


class Method2CrossAttentionAV_AblationNoEAV(Method2CrossAttentionAV):
    """Ablation: final fusion excludes eAB (AV-refined).

    Final fusion uses only:
    - `wavlm_pooled`
    - `arousal_embedding` (eA from WPRO branch)
    - `valence_embedding` (eV from RoBERTa branch)

    This removes AV-refined embedding (eAB) from the final classification input.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_dim = self.text_projection.out_features
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
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(
        self,
        audio_input,
        text_input=None,
        audio_attention_mask=None,
        text_attention_mask=None,
        use_guidance=True,
        return_intermediate=False,
    ):
        # === Step 1: Extract prosody features from audio (Wpro) ===
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_wavlm):
            wpro_output = self.wavlm_prosody(audio_input, attention_mask=audio_attention_mask)
        wpro_features = wpro_output.last_hidden_state
        wpro_projected = self.prosody_projection(wpro_features)

        # === Step 2: Extract semantic features from text (RoBERTa) ===
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

        # === Step 3: Cross-Attention between Wpro and RoBERTa ===
        wpro_attended, text_attended = self.cross_attention(wpro_projected, text_projected)

        # === Step 4: Post-attention refinement ===
        wpro_refined = self.prosody_post_attention(wpro_attended)
        text_refined = self.text_post_attention(text_attended)

        # Pool both branches to get embeddings
        wpro_pooled, _ = self.prosody_pooling(wpro_refined)
        text_pooled, _ = self.text_pooling(text_refined)
        arousal_embedding = wpro_pooled
        valence_embedding = text_pooled

        # === Step 5: Branch fusion ===
        branch_fused = self.branch_fusion(wpro_pooled, text_pooled)

        # === Step 6: Predict A/V ===
        av_features = self.av_prediction_layers(branch_fused)
        arousal_value = self.arousal_head(av_features)
        valence_value = self.valence_head(av_features)

        # === Step 7: Refine A/V ===
        av_values = torch.cat([arousal_value, valence_value], dim=-1)
        av_refined = self.av_refinement(av_values)

        # === Step 8: Global WavLM branch ===
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_wavlm):
            wavlm_output = self.wavlm_final(audio_input, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_output.last_hidden_state
        wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)

        # === Step 9: Final fusion WITHOUT eAB (AV-refined) ===
        # Keep eA/eV and exclude av_refined from final classifier input.
        pre_guidance_representation = torch.cat(
            [wavlm_pooled, arousal_embedding, valence_embedding], dim=-1
        )

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
            "arousal_value": arousal_value,
            "valence_value": valence_value,
            "arousal_embedding": arousal_embedding,
            "valence_embedding": valence_embedding,
            "av_features": av_features,
            "branch_fused": branch_fused,
            "av_refined": av_refined,
            "wpro_attended": wpro_attended,
            "text_attended": text_attended,
        }

        if return_intermediate:
            output.update(
                {
                    "final_representation": final_representation,
                    "pre_guidance_representation": pre_guidance_representation,
                    "fused_features": fused_features,
                    "guidance_used": use_guidance,
                }
            )

        return output


if __name__ == "__main__":
    model = Method2CrossAttentionAV_AblationNoEAV(num_emotions=4, hidden_dim=256)
    audio = torch.randn(2, 16000 * 3)
    with torch.no_grad():
        out = model(audio)
    print({k: v.shape for k, v in out.items() if hasattr(v, "shape")})
