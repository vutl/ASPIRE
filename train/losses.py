"""
Loss functions for multi-task emotion recognition
Includes: Emotion classification loss, A-V regression loss, combined loss

Recommended configuration:
- Emotion labels: CrossEntropyLoss (classification)
- A-V values: CCC Loss (continuous emotion regression)
- Combined: L_total = α * L_emotion + β * L_arousal_CCC + γ * L_valence_CCC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropyLoss(nn.Module):
    """Multi-class focal loss for single-label classification.

    Args:
        alpha: balancing factor; float or sequence of per-class weights.
        gamma: focusing parameter; higher => down-weight easy examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B]
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        # Gather the log prob for the true class
        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Alpha handling (scalar or per-class)
        if isinstance(self.alpha, torch.Tensor):
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
        else:
            alpha_t = self.alpha

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt
        return loss.mean()


class CombinedEmotionLoss(nn.Module):
    """
    Combined loss for multi-task learning:
    1. Emotion classification loss (Cross-Entropy)
    2. Arousal regression loss (MSE or Smooth L1)
    3. Valence regression loss (MSE or Smooth L1)
    """
    def __init__(
        self,
        emotion_weight=1.0,
        arousal_weight=0.5,
        valence_weight=0.5,
        regression_loss='mse'
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.arousal_weight = arousal_weight
        self.valence_weight = valence_weight
        
        # Classification loss
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        
        # Regression loss
        if regression_loss == 'mse':
            self.regression_loss_fn = nn.MSELoss()
        elif regression_loss == 'smooth_l1':
            self.regression_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown regression_loss: {regression_loss}")
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with keys:
                - emotion_logits: [B, num_emotions]
                - arousal_value: [B, 1]
                - valence_value: [B, 1]
            targets: dict with keys:
                - emotion_label: [B] - integer labels
                - arousal_value: [B, 1] - float values
                - valence_value: [B, 1] - float values
        Returns:
            dict with:
                - total_loss
                - emotion_loss
                - arousal_loss
                - valence_loss
        """
        # Emotion classification loss
        emotion_loss = self.emotion_loss_fn(
            outputs['emotion_logits'],
            targets['emotion_label']
        )
        
        # Arousal regression loss
        arousal_loss = self.regression_loss_fn(
            outputs['arousal_value'],
            targets['arousal_value']
        )
        
        # Valence regression loss
        valence_loss = self.regression_loss_fn(
            outputs['valence_value'],
            targets['valence_value']
        )
        
        # Combined loss
        total_loss = (
            self.emotion_weight * emotion_loss +
            self.arousal_weight * arousal_loss +
            self.valence_weight * valence_loss
        )
        
        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'arousal_loss': arousal_loss,
            'valence_loss': valence_loss
        }


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) Loss
    Often used for continuous emotion recognition
    CCC measures the agreement between predicted and ground truth
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1] - predicted values
            target: [B, 1] - ground truth values
        Returns:
            ccc_loss: scalar (1 - CCC, so minimizing this maximizes CCC)
        """
        pred = pred.squeeze()
        target = target.squeeze()
        
        # Mean
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        # Variance (unbiased=False to handle batch_size=1)
        pred_var = torch.var(pred, unbiased=False)
        target_var = torch.var(target, unbiased=False)
        
        # Covariance
        covariance = torch.mean((pred - pred_mean) * (target - target_mean))
        
        # CCC
        ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
        
        # Return 1 - CCC as loss (to minimize)
        return 1 - ccc


class CombinedEmotionLossCCC(nn.Module):
    """
    Combined loss using CCC for A-V regression (RECOMMENDED)
    
    Loss formula:
        L_total = α * L_emotion + β * L_arousal_CCC + γ * L_valence_CCC
    
    where:
        - L_emotion: CrossEntropyLoss for emotion classification (labels)
        - L_arousal_CCC: 1 - CCC for arousal regression (continuous)
        - L_valence_CCC: 1 - CCC for valence regression (continuous)
    
    CCC (Concordance Correlation Coefficient) is preferred over MSE because:
        - Measures agreement between predicted and ground truth
        - Considers both precision (correlation) and accuracy (bias)
        - Better for continuous emotion modeling
        - Robust to outliers
    
    Default weights:
        α (emotion_weight) = 1.0  - Emotion classification
        β (arousal_weight) = 1.0  - Arousal regression  
        γ (valence_weight) = 1.0  - Valence regression
    """
    def __init__(
        self,
        emotion_weight=1.0,
        arousal_weight=1.0,
        valence_weight=1.0
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.arousal_weight = arousal_weight
        self.valence_weight = valence_weight
        
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        self.ccc_loss_fn = CCCLoss()
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with emotion_logits, arousal_value, valence_value
            targets: dict with emotion_label, arousal_value, valence_value
        Returns:
            dict with losses
        """
        # Emotion classification loss
        emotion_loss = self.emotion_loss_fn(
            outputs['emotion_logits'],
            targets['emotion_label']
        )
        
        # Arousal CCC loss
        arousal_loss = self.ccc_loss_fn(
            outputs['arousal_value'],
            targets['arousal_value']
        )
        
        # Valence CCC loss
        valence_loss = self.ccc_loss_fn(
            outputs['valence_value'],
            targets['valence_value']
        )
        
        # Combined loss
        total_loss = (
            self.emotion_weight * emotion_loss +
            self.arousal_weight * arousal_loss +
            self.valence_weight * valence_loss
        )
        
        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'arousal_loss': arousal_loss,
            'valence_loss': valence_loss
        }


class CombinedEmotionLossCCCFocal(nn.Module):
    """Same as CombinedEmotionLossCCC but using focal loss for emotion branch."""
    def __init__(
        self,
        emotion_weight=1.0,
        arousal_weight=1.0,
        valence_weight=1.0,
        focal_alpha=1.0,
        focal_gamma=2.0
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.arousal_weight = arousal_weight
        self.valence_weight = valence_weight

        self.emotion_loss_fn = FocalCrossEntropyLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ccc_loss_fn = CCCLoss()

    def forward(self, outputs, targets):
        emotion_loss = self.emotion_loss_fn(outputs['emotion_logits'], targets['emotion_label'])
        arousal_loss = self.ccc_loss_fn(outputs['arousal_value'], targets['arousal_value'])
        valence_loss = self.ccc_loss_fn(outputs['valence_value'], targets['valence_value'])

        total_loss = (
            self.emotion_weight * emotion_loss +
            self.arousal_weight * arousal_loss +
            self.valence_weight * valence_loss
        )

        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'arousal_loss': arousal_loss,
            'valence_loss': valence_loss
        }


class CombinedEmotionLossAVDCCC(nn.Module):
    """
    Combined loss using CCC for AVD regression (Arousal, Valence, Dominance)
    
    Loss formula:
        L_total = α * L_emotion + β * L_arousal_CCC + γ * L_valence_CCC + δ * L_dominance_CCC
    """
    def __init__(
        self,
        emotion_weight=1.0,
        arousal_weight=1.0,
        valence_weight=1.0,
        dominance_weight=1.0
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.arousal_weight = arousal_weight
        self.valence_weight = valence_weight
        self.dominance_weight = dominance_weight
        
        self.emotion_loss_fn = nn.CrossEntropyLoss()
        self.ccc_loss_fn = CCCLoss()
    
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with emotion_logits, arousal_value, valence_value, dominance_value
            targets: dict with emotion_label, arousal_value, valence_value, dominance_value
        Returns:
            dict with losses
        """
        # Emotion classification loss
        emotion_loss = self.emotion_loss_fn(
            outputs['emotion_logits'],
            targets['emotion_label']
        )
        
        # Arousal CCC loss
        arousal_loss = self.ccc_loss_fn(
            outputs['arousal_value'],
            targets['arousal_value']
        )
        
        # Valence CCC loss
        valence_loss = self.ccc_loss_fn(
            outputs['valence_value'],
            targets['valence_value']
        )
        
        # Dominance CCC loss
        dominance_loss = self.ccc_loss_fn(
            outputs['dominance_value'],
            targets['dominance_value']
        )
        
        # Combined loss
        total_loss = (
            self.emotion_weight * emotion_loss +
            self.arousal_weight * arousal_loss +
            self.valence_weight * valence_loss +
            self.dominance_weight * dominance_loss
        )
        
        return {
            'total_loss': total_loss,
            'emotion_loss': emotion_loss,
            'arousal_loss': arousal_loss,
            'valence_loss': valence_loss,
            'dominance_loss': dominance_loss
        }
