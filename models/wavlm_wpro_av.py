import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class SimpleAudioBackbone(nn.Module):
    """Lightweight backbone used when `use_pretrained=False` to avoid HF downloads."""
    def __init__(self, in_dim=1, out_dim=768, kernel=16):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=kernel//2, padding=kernel//4)

    def forward(self, x):
        # x: [B, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out = self.conv(x)
        out = out.permute(0, 2, 1)  # [B, T', D]
        return out


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Tanh(), nn.Linear(input_dim, 1))

    def forward(self, x):
        # x: [B, T, D]
        w = self.att(x)  # [B, T, 1]
        w = F.softmax(w, dim=1)
        pooled = torch.sum(x * w, dim=1)
        return pooled


class WavLMWProAV(nn.Module):
    """Combined WavLM (global) + W-PRO (prosody) audio model with A-V + emotion outputs.

    If `use_pretrained=False`, small convolutional backbones are used to avoid
    downloading large HF models; this mode is intended for quick parameter/FLOPs
    estimation on the local machine.
    """
    def __init__(self,
                 num_emotions=4,
                 wavlm_model_name="microsoft/wavlm-base-plus",
                 wpro_model_name="Orange/Speaker-wavLM-pro",
                 hidden_dim=256,
                 av_refinement_layers=2,
                 freeze_wavlm=True,
                 freeze_wpro=True,
                 use_pretrained=False):
        super().__init__()

        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.wavlm = Wav2Vec2Model.from_pretrained(wavlm_model_name)
            self.wavlm_dim = self.wavlm.config.hidden_size
            self.wpro = Wav2Vec2Model.from_pretrained(wpro_model_name)
            self.wpro_dim = self.wpro.config.hidden_size
        else:
            # small backbones
            self.wavlm = SimpleAudioBackbone(in_dim=1, out_dim=768)
            self.wavlm_dim = 768
            self.wpro = SimpleAudioBackbone(in_dim=1, out_dim=256)
            self.wpro_dim = 256

        if freeze_wavlm and use_pretrained:
            for p in self.wavlm.parameters():
                p.requires_grad = False
        if freeze_wpro and use_pretrained:
            for p in self.wpro.parameters():
                p.requires_grad = False

        # Projections
        self.wavlm_proj = nn.Linear(self.wavlm_dim, hidden_dim)
        self.wpro_proj = nn.Linear(self.wpro_dim, hidden_dim)

        # Fusion
        fusion_dim = hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # A-V prediction
        av_layers = []
        for _ in range(av_refinement_layers):
            av_layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1)]
        self.av_prediction = nn.Sequential(*av_layers)
        self.arousal_head = nn.Linear(hidden_dim, 1)
        self.valence_head = nn.Linear(hidden_dim, 1)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_emotions)
        )

        # pooling
        self.wavlm_pool = SelfAttentionPooling(self.wavlm_dim)
        self.wpro_pool = SelfAttentionPooling(self.wpro_dim)

    def forward(self, audio_input, audio_attention_mask=None, text_input=None, text_attention_mask=None):
        # audio_input: [B, T]
        if self.use_pretrained:
            wav_out = self.wavlm(audio_input, attention_mask=audio_attention_mask)
            wav_feats = wav_out.last_hidden_state
            wpro_out = self.wpro(audio_input, attention_mask=audio_attention_mask)
            wpro_feats = wpro_out.last_hidden_state
        else:
            wav_feats = self.wavlm(audio_input)
            wpro_feats = self.wpro(audio_input)

        wav_pooled = self.wavlm_pool(wav_feats)
        wpro_pooled = self.wpro_pool(wpro_feats)

        wav_emb = self.wavlm_proj(wav_pooled)
        wpro_emb = self.wpro_proj(wpro_pooled)

        fused = torch.cat([wav_emb, wpro_emb], dim=-1)
        fused = self.fusion(fused)

        av_feat = self.av_prediction(fused)
        arousal = self.arousal_head(av_feat)
        valence = self.valence_head(av_feat)

        # final emotion classifier uses both fused and AV-refined
        final_in = torch.cat([fused, av_feat], dim=-1)
        logits = self.classifier(final_in)

        return {
            'emotion_logits': logits,
            'arousal_value': arousal,
            'valence_value': valence
        }


if __name__ == '__main__':
    m = WavLMWProAV(use_pretrained=False)
    a = torch.randn(2, 16000 * 3)
    out = m(a)
    print({k: v.shape for k, v in out.items()})
