
"""ASPIRE-Lite base class.

Lightens Method-2 by replacing:
- W-PRO -> eGeMAPS
- WavLM global encoder -> tiny mel-based audio encoder
- RoBERTa-base -> small text encoder (configurable via roberta_model_name arg)

The A/V guidance logic remains unchanged.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

from models.base_components import (
    SelfAttentionPooling,
    BiDirectionalCrossAttention,
    FusionLayer,
    AVRefinementModule,
)
from models.lite_components import OpenSmileEGeMAPSExtractor, TinyMelAudioEncoder


class Method2CrossAttentionAVLiteBase(nn.Module):
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name='tiny-mel-audio',
        wavlm_prosody_model_name=None,
        roberta_model_name='sentence-transformers/paraphrase-MiniLM-L3-v2',
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=False,
        freeze_roberta=False,
        num_cross_attn_heads=8,
        post_attention_layers=2,
        av_prediction_layers=3,
        egemaps_dim=88,
        sample_rate=16000,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.audio_encoder_name = wavlm_model_name
        self.text_model_name = roberta_model_name
        self.egemaps_extractor = OpenSmileEGeMAPSExtractor(sample_rate=sample_rate, use_opensmile=True)

        self.prosody_projection = nn.Sequential(
            nn.Linear(egemaps_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.text_model = AutoModel.from_pretrained(roberta_model_name)
        self.text_dim = self.text_model.config.hidden_size
        if freeze_roberta:
            for p in self.text_model.parameters():
                p.requires_grad = False
        self.text_projection = nn.Linear(self.text_dim, hidden_dim)

        self.cross_attention = BiDirectionalCrossAttention(dim=hidden_dim, num_heads=num_cross_attn_heads, dropout=0.1)
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
        self.branch_fusion = FusionLayer(hidden_dim, hidden_dim, hidden_dim, fusion_type='gated')

        av_layers = []
        for _ in range(av_prediction_layers):
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

        self.audio_encoder = TinyMelAudioEncoder(sample_rate=sample_rate, n_mels=64, output_dim=hidden_dim, dropout=0.1)
        if freeze_wavlm:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        self.global_audio_dim = hidden_dim

        input_fusion_dim = self.global_audio_dim + hidden_dim * 3
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

        print(f'[{self.__class__.__name__}]')
        print(f'  Prosody: eGeMAPS(88) -> {hidden_dim}')
        print(f'  Text: {roberta_model_name} ({self.text_dim}) -> {hidden_dim}')
        print(f'  Global audio: TinyMelAudioEncoder -> {hidden_dim}')

    def forward(
        self,
        audio_input,
        text_input=None,
        audio_attention_mask=None,
        text_attention_mask=None,
        use_guidance=True,
        return_intermediate=False,
    ):
        with torch.no_grad():
            egemaps = self.egemaps_extractor(audio_input)
        prosody_seq = self.prosody_projection(egemaps).unsqueeze(1)

        text_out = self.text_model(input_ids=text_input.long(), attention_mask=text_attention_mask)
        text_features = self.text_projection(text_out.last_hidden_state)

        prosody_attended, text_attended = self.cross_attention(prosody_seq, text_features)
        prosody_refined = self.prosody_post_attention(prosody_attended)
        text_refined = self.text_post_attention(text_attended)

        arousal_embedding, _ = self.prosody_pooling(prosody_refined)
        valence_embedding, _ = self.text_pooling(text_refined)

        branch_fused = self.branch_fusion(arousal_embedding, valence_embedding)
        av_features = self.av_prediction_layers(branch_fused)
        arousal_value = self.arousal_head(av_features)
        valence_value = self.valence_head(av_features)
        av_values = torch.cat([arousal_value, valence_value], dim=-1)
        av_refined = self.av_refinement(av_values)
        if not use_guidance:
            av_refined = torch.zeros_like(av_refined)

        wavlm_pooled = self.audio_encoder(audio_input)

        final_representation = torch.cat([
            wavlm_pooled,
            arousal_embedding,
            valence_embedding,
            av_refined,
        ], dim=-1)
        fused_features = self.final_fusion(final_representation)
        emotion_logits = self.emotion_classifier(fused_features)

        outputs = {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
        }
        if return_intermediate:
            outputs.update({
                'arousal_embedding': arousal_embedding,
                'valence_embedding': valence_embedding,
                'branch_fused': branch_fused,
                'av_refined': av_refined,
                'final_representation': final_representation,
                'fused_features': fused_features,
            })
        return outputs
