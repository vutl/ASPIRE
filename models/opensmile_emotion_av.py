"""
OpenSMILE-based Emotion + A-V Classifier
Architecture: eGeMAPS features (88-d) → Transformer → Emotion (4-class) + Arousal + Valence
This is a single-branch model that learns both emotion and 2D A-V from traditional acoustic features.
"""
import torch
import torch.nn as nn


class OpenSMILEEmotionAVClassifier(nn.Module):
    """
    Simple baseline: OpenSMILE features only
    Predicts: Emotion (4-class) + Arousal (continuous) + Valence (continuous)
    """
    def __init__(
        self,
        input_dim=88,
        hidden_dim=256,
        num_emotions=4,
        num_transformer_layers=4,
        num_heads=4,
        ff_dim=512,
        dropout=0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions
        
        # ============ Embedding ============
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ============ Transformer Encoder ============
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        
        # ============ Emotion Classifier ============
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_emotions)
        )
        
        # ============ Arousal Predictor ============
        self.arousal_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # ============ Valence Predictor ============
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Tanh()  # Normalize to [-1, 1]
        )
    
    def forward(self, opensmile):
        """
        Args:
            opensmile: [B, 88] - OpenSMILE eGeMAPS features
        
        Returns:
            dict with keys:
                - emotion_logits: [B, num_emotions]
                - arousal: [B, 1]
                - valence: [B, 1]
        """
        # [B, 88] -> [B, 1, 256] (add time dimension for transformer)
        x = opensmile.unsqueeze(1)  # [B, 1, 88]
        x = self.embedding(x)  # [B, 1, 256]
        
        # Transformer
        x = self.transformer(x)  # [B, 1, 256]
        
        # Remove time dimension
        x = x.squeeze(1)  # [B, 256]
        
        # Predictions
        emotion_logits = self.emotion_classifier(x)
        arousal = self.arousal_head(x)
        valence = self.valence_head(x)
        
        return {
            'emotion_logits': emotion_logits,
            'arousal': arousal,
            'valence': valence,
        }
