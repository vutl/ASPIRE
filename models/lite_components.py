
"""Lightweight components for ASPIRE variants.

Includes:
- Exact eGeMAPS extraction via openSMILE when available
- 88-dim proxy handcrafted prosody features as a fallback
- Tiny mel-spectrogram based global audio encoder
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class TemporalSelfAttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, T, D]
        w = torch.softmax(self.attn(x), dim=1)
        pooled = torch.sum(x * w, dim=1)
        return pooled, w


class OpenSmileEGeMAPSExtractor(nn.Module):
    """Return 88-dim eGeMAPS functionals.

    If `opensmile` is installed, this uses the exact eGeMAPSv02 functionals.
    Otherwise, it falls back to a deterministic 88-dim handcrafted proxy:
    40 log-mel mean stats + 40 log-mel std stats + 8 waveform statistics.
    """

    def __init__(self, sample_rate: int = 16000, use_opensmile: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.use_opensmile = False
        self.smile = None
        if use_opensmile:
            try:
                import opensmile  # type: ignore
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals,
                )
                self.use_opensmile = True
                print('[OpenSmileEGeMAPSExtractor] Using exact openSMILE eGeMAPSv02 (88 dims).')
            except Exception as e:
                print(f'[OpenSmileEGeMAPSExtractor] openSMILE unavailable, using 88-dim handcrafted proxy. Reason: {e}')

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=40,
            f_min=20,
            f_max=7600,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def _ensure_transforms_device(self, device: torch.device):
        self.mel = self.mel.to(device)
        self.to_db = self.to_db.to(device)

    def _proxy_features(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: [B, T] -> [B, 88]"""
        device = audio.device
        self._ensure_transforms_device(device)
        mel = self.to_db(self.mel(audio) + 1e-8)  # [B, 40, Tm]
        mel_mean = mel.mean(dim=-1)
        mel_std = mel.std(dim=-1, unbiased=False)

        # 8 waveform-level stats
        rms = torch.sqrt(torch.clamp((audio ** 2).mean(dim=-1, keepdim=True), min=1e-8))
        audio_mean = audio.mean(dim=-1, keepdim=True)
        audio_std = audio.std(dim=-1, unbiased=False, keepdim=True)
        audio_max = audio.max(dim=-1, keepdim=True).values
        audio_min = audio.min(dim=-1, keepdim=True).values
        zcr = (audio[:, 1:] * audio[:, :-1] < 0).float().mean(dim=-1, keepdim=True)
        abs_mean = audio.abs().mean(dim=-1, keepdim=True)
        abs_std = audio.abs().std(dim=-1, unbiased=False, keepdim=True)

        extra = torch.cat([
            rms, audio_mean, audio_std, audio_max,
            audio_min, zcr, abs_mean, abs_std,
        ], dim=-1)
        return torch.cat([mel_mean, mel_std, extra], dim=-1)

    def _exact_features(self, audio: torch.Tensor) -> torch.Tensor:
        # CPU loop, exact openSMILE extraction.
        feats = []
        for wav in audio.detach().cpu():
            x = wav.numpy()
            # trim padded tail if possible
            nz = (abs(x) > 1e-8).nonzero()[0] if (abs(x) > 1e-8).any() else None
            if nz is not None and len(nz) > 0:
                x = x[: nz[-1] + 1]
            df = self.smile.process_signal(x, self.sample_rate)
            feats.append(torch.tensor(df.values[0], dtype=torch.float32))
        return torch.stack(feats, dim=0).to(audio.device)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.use_opensmile and self.smile is not None:
            return self._exact_features(audio)
        return self._proxy_features(audio)


class TinyMelAudioEncoder(nn.Module):
    """Small global audio encoder based on log-mel + Conv1D + 1 Transformer layer."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        output_dim: int = 256,
        dropout: float = 0.1,
        num_transformer_layers: int = 1,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.output_dim = output_dim

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            win_length=400,
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, output_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=4,
                dim_feedforward=output_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_transformer_layers,
        )
        self.pool = TemporalSelfAttentionPooling(output_dim)

    def _ensure_transforms_device(self, device: torch.device):
        self.mel = self.mel.to(device)
        self.to_db = self.to_db.to(device)

    def forward(self, audio: torch.Tensor, return_sequence: bool = False):
        device = audio.device
        self._ensure_transforms_device(device)
        mel = self.to_db(self.mel(audio) + 1e-8)  # [B, M, T]
        x = self.conv(mel)  # [B, D, T']
        x = x.transpose(1, 2)  # [B, T', D]
        x = self.temporal_encoder(x)
        pooled, attn = self.pool(x)
        if return_sequence:
            return pooled, x, attn
        return pooled
