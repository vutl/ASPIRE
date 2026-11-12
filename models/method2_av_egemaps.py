#!/usr/bin/env python3
"""
Method 2: Cross-Attention A-V Architecture with eGeMAPS Prosody
Architecture:
1. WavLM extracts acoustic/prosody features from audio
2. RoBERTa extracts semantic/linguistic features from transcript text
3. eGeMAPS extracts traditional prosody features from audio
4. WavLM and RoBERTa cross-attention with each other
5. Each branch goes through layers after cross-attention
6. Fusion of the two branches
7. Fusion goes through layers -> outputs A-V values
8. A-V values go through refinement module
9. Separate WavLM branch extracts global acoustic features
10. Final fusion: WavLM + branch_fusion + AV_refined -> Emotion labels + A-V values

Flow:
    Audio ──> WavLM (prosody) ──┐
                                ├─> Cross-Attention ─> Layers ─┐
    Text ──> RoBERTa ───────────┘                              ├─> Fusion ─> Layers ─> A-V values ─> Refinement ─> AV_refined
                                                               │                                                       │
    Audio ──> eGeMAPS ──────────────────────────────────────────┤
                                                               │
    Audio ──> WavLM (final) ────────────────────────────────────┴──────────────────────────────────────────────> Final Fusion ─> Emotion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, AutoModel

from models.base_components import (
    SelfAttentionPooling,
    BiDirectionalCrossAttention,
    FusionLayer,
    AVRefinementModule,
)
from models.lite_components import OpenSmileEGeMAPSExtractor


class Method2CrossAttentionAVEGeMAPS(nn.Module):
    """
    Method 2: Cross-Attention A-V Architecture with eGeMAPS Prosody
    
    Replaces W-PRO with eGeMAPS features while keeping WavLM + RoBERTa cross-attention.
    
    Flow:
    1. WavLM extracts acoustic features from audio
    2. RoBERTa extracts semantic features from text transcript
    3. eGeMAPS extracts traditional prosody features from audio
    4. WavLM and RoBERTa cross-attention
    5. Each branch goes through post-attention layers
    6. Fusion of both branches -> goes through layers -> A-V values
    7. A-V values -> refinement module -> AV_refined
    8. Separate WavLM extracts global acoustic features
    9. Final fusion: WavLM + fusion(step 6) + AV_refined -> Emotion + A-V
    """
    def __init__(
        self,
        num_emotions=4,
        wavlm_model_name="microsoft/wavlm-base-plus",
        roberta_model_name="roberta-base",
        egemaps_dim=88,
        hidden_dim=256,
        av_refinement_layers=2,
        freeze_wavlm=True,
        freeze_roberta=True,
        num_cross_attn_heads=8,
        num_heads=4,
        post_attention_layers=2,
        av_prediction_layers=3,
        num_transformer_layers=4,
        ff_dim=512,
        dropout=0.1,
        sample_rate=16000,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim

        # === Branch 1: eGeMAPS prosody branch (replaces WavLM prosody) ===
        self.egemaps_dim = egemaps_dim
        self.egemaps_extractor = OpenSmileEGeMAPSExtractor(
            sample_rate=sample_rate,
            use_opensmile=True,
        )
        self.prosody_proj = nn.Linear(egemaps_dim, hidden_dim)
        self.prosody_norm = nn.LayerNorm(hidden_dim)
        self.prosody_dropout = nn.Dropout(dropout)

        # === Branch 2: RoBERTa for semantic/text - processes transcript ===
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.roberta_dim = self.roberta.config.hidden_size  # 768 for roberta-base
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        self.semantic_projection = nn.Linear(self.roberta_dim, hidden_dim)
        self.semantic_pooling = SelfAttentionPooling(hidden_dim)

        # === Cross-Attention between eGeMAPS (prosody) and RoBERTa ===
        self.cross_attention = BiDirectionalCrossAttention(
            dim=hidden_dim,
            num_heads=num_cross_attn_heads,
            dropout=dropout,
        )

        # === Post-attention layers for each branch ===
        self.prosody_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=post_attention_layers
        )
        self.semantic_post_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=post_attention_layers
        )
        self.prosody_pooling = SelfAttentionPooling(hidden_dim)
        self.text_pooling = SelfAttentionPooling(hidden_dim)

        # === Fusion of the two branches ===
        self.branch_fusion = FusionLayer(
            dim1=hidden_dim,
            dim2=hidden_dim,
            output_dim=hidden_dim,
            fusion_type='gated'
        )

        # === Fusion goes through layers to predict A-V values ===
        av_layers = []
        for i in range(av_prediction_layers):
            av_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.av_prediction_layers = nn.Sequential(*av_layers)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

        # === A-V refinement module ===
        self.av_refinement = AVRefinementModule(
            av_dim=2,
            hidden_dim=hidden_dim,
            num_layers=av_refinement_layers
        )

        # === Final WavLM branch for emotion (global acoustic features) ===
        self.wavlm_final = Wav2Vec2Model.from_pretrained(wavlm_model_name)
        self.wavlm_final_dim = self.wavlm_final.config.hidden_size
        if freeze_wavlm:
            for param in self.wavlm_final.parameters():
                param.requires_grad = False
        self.wavlm_pooling = SelfAttentionPooling(self.wavlm_final_dim)

        # === Final fusion: WavLM + arousal_embedding + valence_embedding + AV_refined -> Emotion ===
        input_fusion_dim = self.wavlm_final_dim + hidden_dim * 3
        self.final_fusion = nn.Sequential(
            nn.Linear(input_fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)
        )

        print(f"[Method2CrossAttentionAVEGeMAPS]")
        print(f"  eGeMAPS: {egemaps_dim} → {hidden_dim} (prosody branch)")
        print(f"  RoBERTa semantic: {self.roberta_dim} → {hidden_dim}")
        print(f"  Final fusion: {input_fusion_dim} → {hidden_dim}")
        print(f"  Output: Emotion ({num_emotions} classes) + Arousal + Valence")
    
    def forward(
        self,
        audio=None,
        text_input_ids=None,
        text_attention_mask=None,
        egemaps=None,
        audio_input=None,
        text_input=None,
        audio_attention_mask=None,
        use_guidance=True,
        return_intermediate=False,
    ):
        """
        Args:
            audio/audio_input: (batch_size, time_steps) raw audio waveform
            text_input_ids/text_input: tokenized text ids
            text_attention_mask: text attention mask
            egemaps: optional precomputed eGeMAPS features (batch_size, 88)
        Returns:
            dict with keys:
              - 'emotion_logits': (batch_size, num_emotions)
              - 'arousal_value': (batch_size, 1)
              - 'valence_value': (batch_size, 1)
        """
        # Support both the legacy signature and the Method-2 trainer signature.
        if audio is None:
            audio = audio_input
        if text_input_ids is None:
            text_input_ids = text_input
        if text_input_ids is None or text_attention_mask is None or audio is None:
            raise ValueError("audio, text_input_ids/text_input, and text_attention_mask are required")

        if egemaps is None:
            with torch.no_grad():
                egemaps = self.egemaps_extractor(audio)

        # === Step 1: eGeMAPS prosody branch ===
        prosody_feat = self.prosody_proj(egemaps)  # (batch_size, hidden_dim)
        prosody_feat = self.prosody_norm(prosody_feat)
        prosody_feat = F.gelu(prosody_feat)
        prosody_feat = self.prosody_dropout(prosody_feat)
        # Add sequence dim for transformer: [B, 1, D]
        prosody_feat_seq = prosody_feat.unsqueeze(1)

        # === Step 2: RoBERTa semantic branch ===
        roberta_output = self.roberta(input_ids=text_input_ids.long(), attention_mask=text_attention_mask)
        semantic_feat = roberta_output.last_hidden_state  # (batch_size, seq_len, roberta_dim)
        semantic_feat = self.semantic_projection(semantic_feat)  # (batch_size, seq_len, hidden_dim)

        # === Step 3: Cross-Attention between eGeMAPS and RoBERTa ===
        # eGeMAPS attends to text, text attends to eGeMAPS
        prosody_attended, text_attended = self.cross_attention(prosody_feat_seq, semantic_feat)

        # === Step 4: Post-attention layers ===
        prosody_refined = self.prosody_post_attention(prosody_attended)  # [B, 1, hidden_dim]
        text_refined = self.semantic_post_attention(text_attended)  # [B, seq_len, hidden_dim]

        # Pool both branches
        prosody_pooled, _ = self.prosody_pooling(prosody_refined)  # [B, hidden_dim]
        text_pooled, _ = self.text_pooling(text_refined)  # [B, hidden_dim]

        # === Step 5: Fusion of the two branches ===
        branch_fused = self.branch_fusion(prosody_pooled, text_pooled)  # [B, hidden_dim]

        # === Step 6: Fusion goes through layers -> outputs A-V values ===
        av_features = self.av_prediction_layers(branch_fused)  # [B, hidden_dim]
        arousal_value = self.arousal_head(av_features)  # [B, 1]
        valence_value = self.valence_head(av_features)  # [B, 1]

        # === Step 7: A-V values go through refinement module ===
        av_values = torch.cat([arousal_value, valence_value], dim=-1)  # [B, 2]
        av_refined = self.av_refinement(av_values)  # [B, hidden_dim]
        if not use_guidance:
            av_refined = torch.zeros_like(av_refined)

        # === Step 8: Separate WavLM branch for final fusion ===
        wavlm_output = self.wavlm_final(audio, attention_mask=audio_attention_mask)
        wavlm_features = wavlm_output.last_hidden_state  # [B, T, wavlm_final_dim]
        wavlm_pooled, _ = self.wavlm_pooling(wavlm_features)  # [B, wavlm_final_dim]

        # === Step 9: Final fusion: WavLM + arousal_embedding + valence_embedding + AV_refined -> Emotion ===
        final_representation = torch.cat([
            wavlm_pooled,         # WavLM: [B, wavlm_final_dim]
            prosody_pooled,       # eGeMAPS-attended features: [B, hidden_dim]
            text_pooled,          # Text-attended features: [B, hidden_dim]
            av_refined            # AV_refined: [B, hidden_dim]
        ], dim=-1)  # [B, wavlm_final_dim + hidden_dim*3]
        fused_features = self.final_fusion(final_representation)  # [B, hidden_dim]
        emotion_logits = self.emotion_classifier(fused_features)  # [B, num_emotions]

        outputs = {
            'emotion_logits': emotion_logits,
            'arousal_value': arousal_value,
            'valence_value': valence_value,
            'arousal': arousal_value,
            'valence': valence_value,
        }
        if return_intermediate:
            outputs.update({
                'arousal_embedding': prosody_pooled,
                'valence_embedding': text_pooled,
                'branch_fused': branch_fused,
                'av_refined': av_refined,
                'final_representation': final_representation,
                'fused_features': fused_features,
            })
        return outputs
