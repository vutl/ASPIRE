"""
Base components for emotion recognition models
Includes: Attention modules, Fusion layers, Prosody encoder wrapper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2Model
from huggingface_hub import hf_hub_download


class SelfAttentionPooling(nn.Module):
    """Self-attention pooling to aggregate temporal features"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, x):
        # x: [B, T, D]
        attn_weights = self.attention(x)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
        return pooled, attn_weights


class CrossAttention(nn.Module):
    """Bidirectional cross-attention between two sequences"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # For Q from first sequence, K,V from second
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x1, x2):
        """
        Args:
            x1: [B, T1, D] - query sequence
            x2: [B, T2, D] - key, value sequence
        Returns:
            x1_attended: [B, T1, D] - x1 attended to x2
        """
        B, T1, D = x1.shape
        T2 = x2.shape[1]
        
        # Project
        Q = self.q_proj(x1).view(B, T1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T1, d]
        K = self.k_proj(x2).view(B, T2, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T2, d]
        V = self.v_proj(x2).view(B, T2, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T2, d]
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [B, H, T1, T2]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ V  # [B, H, T1, d]
        out = out.transpose(1, 2).contiguous().view(B, T1, D)  # [B, T1, D]
        out = self.out_proj(out)
        
        return out


class BiDirectionalCrossAttention(nn.Module):
    """Bidirectional cross-attention: A attends to V, V attends to A"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn_a_to_v = CrossAttention(dim, num_heads, dropout)
        self.cross_attn_v_to_a = CrossAttention(dim, num_heads, dropout)
        
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        
    def forward(self, arousal_feat, valence_feat):
        """
        Args:
            arousal_feat: [B, T, D]
            valence_feat: [B, T, D]
        Returns:
            a_attended: [B, T, D] - arousal features attended to valence
            v_attended: [B, T, D] - valence features attended to arousal
        """
        # A attends to V
        a_cross = self.cross_attn_a_to_v(arousal_feat, valence_feat)
        a_attended = self.norm_a(arousal_feat + a_cross)  # Residual connection
        
        # V attends to A
        v_cross = self.cross_attn_v_to_a(valence_feat, arousal_feat)
        v_attended = self.norm_v(valence_feat + v_cross)  # Residual connection
        
        return a_attended, v_attended


class FusionLayer(nn.Module):
    """Fusion layer with gating mechanism"""
    def __init__(self, dim1, dim2, output_dim, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(dim1 + dim2, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(dim1 + dim2, output_dim),
                nn.Sigmoid()
            )
            self.transform1 = nn.Linear(dim1, output_dim)
            self.transform2 = nn.Linear(dim2, output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def forward(self, x1, x2):
        if self.fusion_type == 'concat':
            return self.fusion(torch.cat([x1, x2], dim=-1))
        elif self.fusion_type == 'gated':
            gate = self.gate(torch.cat([x1, x2], dim=-1))
            t1 = self.transform1(x1)
            t2 = self.transform2(x2)
            return gate * t1 + (1 - gate) * t2


class AVRefinementModule(nn.Module):
    """Module to refine A-V values without using GNN/GAT"""
    def __init__(self, av_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(av_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Residual projection if needed
        self.residual_proj = nn.Linear(av_dim, hidden_dim) if av_dim != hidden_dim else nn.Identity()
        
    def forward(self, av_values):
        """
        Args:
            av_values: [B, av_dim] - concatenated [arousal_value, valence_value]
        Returns:
            refined_av: [B, hidden_dim] - refined A-V representations
        """
        residual = self.residual_proj(av_values)
        out = self.mlp(av_values)
        return out + residual  # Residual connection


class ProsodyEncoder(nn.Module):
    """Prosody encoder using WavLM-based prosody extractor"""
    def __init__(self, output_dim=256, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Load WavLM prosody model from HuggingFace
        # This is the Orange/Speaker-wavLM-pro model
        try:
            from transformers import WavLMModel
            self.prosody_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            prosody_dim = self.prosody_model.config.hidden_size  # 768
        except:
            print("Warning: Could not load prosody model, using placeholder")
            prosody_dim = 768
            self.prosody_model = None
        
        if use_attention:
            self.attention = SelfAttentionPooling(prosody_dim)
        
        self.projection = nn.Sequential(
            nn.Linear(prosody_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, wavlm_features):
        """
        Args:
            wavlm_features: [B, T, 768] - WavLM features
        Returns:
            prosody_feat: [B, output_dim] - prosody features for arousal
        """
        if self.prosody_model is not None:
            # Process through prosody-specific layers
            # For now, we use WavLM features directly
            # In full implementation, would use Orange/Speaker-wavLM-pro
            prosody_feat = wavlm_features
        else:
            prosody_feat = wavlm_features
        
        if self.use_attention:
            prosody_feat, _ = self.attention(prosody_feat)  # [B, 768]
        else:
            prosody_feat = torch.mean(prosody_feat, dim=1)  # [B, 768]
        
        prosody_feat = self.projection(prosody_feat)  # [B, output_dim]
        return prosody_feat


class TextSemanticEncoder(nn.Module):
    """Text/Semantic encoder using RoBERTa"""
    def __init__(self, output_dim=256, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # For audio-only emotion recognition, we extract semantic info from acoustic features
        # This is a projection layer that learns semantic-like representations
        # In a multimodal setting, this would be replaced with actual RoBERTa
        
        self.semantic_extractor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        if use_attention:
            self.attention = SelfAttentionPooling(768)
        
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, wavlm_features):
        """
        Args:
            wavlm_features: [B, T, 768] - WavLM features
        Returns:
            semantic_feat: [B, output_dim] - semantic features for valence
        """
        # Extract semantic information
        semantic_feat = self.semantic_extractor(wavlm_features)  # [B, T, 768]
        
        if self.use_attention:
            semantic_feat, _ = self.attention(semantic_feat)  # [B, 768]
        else:
            semantic_feat = torch.mean(semantic_feat, dim=1)  # [B, 768]
        
        semantic_feat = self.projection(semantic_feat)  # [B, output_dim]
        return semantic_feat
