"""
Focal Loss for Center Heatmap Detection

Standalone focal loss implementation with testing utilities
to debug center heatmap learning issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for center heatmap prediction.
    
    Addresses extreme class imbalance in center heatmap where only
    a few pixels are positive (object centers) while most are negative.
    
    Loss = -α * (1-pt)^γ * log(pt)
    where pt = p if target=1, else pt = 1-p
    """
    
    def __init__(self, alpha=2.0, beta=4.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for negative examples (default: 2.0)
            beta: Focusing parameter for positive examples (default: 4.0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred_logits, target):
        """
        Args:
            pred_logits: Predicted heatmap LOGITS [B, C, H, W] (before sigmoid)
            target: Ground truth heatmap [B, C, H, W] (0-1, gaussian peaks)
            
        Returns:
            loss: Focal loss value
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred_logits)
        
        # Ensure predictions are in valid range
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # Compute focal weights
        # For positive pixels: (1 - pred)^β
        # For negative pixels: (1 - target)^α * pred^β
        # Use threshold for positive mask (Gaussian peaks may be 0.999 instead of exactly 1.0)
        pos_mask = (target > 0.99).float()  # Gaussian peaks > 0.99 are considered positive
        neg_mask = (target <= 0.99).float()  # Everything else is negative
        
        pos_loss = -pos_mask * torch.pow(1 - pred, self.beta) * torch.log(pred)
        neg_loss = -neg_mask * torch.pow(1 - target, self.alpha) * torch.pow(pred, self.beta) * torch.log(1 - pred)
        
        loss = pos_loss + neg_loss
        
        if self.reduction == 'mean':
            # Normalize by number of positive examples
            num_pos = pos_mask.sum().clamp(min=1.0)
            return loss.sum() / num_pos
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def test_focal_loss():
    """
    Test focal loss with various scenarios to verify correctness.
    """
    print("=" * 80)
    print("TESTING FOCAL LOSS")
    print("=" * 80)
    
    focal_loss = FocalLoss(alpha=2.0, beta=4.0)
    
    # Test 1: Perfect prediction
    print("\n[Test 1] Perfect prediction (pred = target)")
    pred_logits = torch.zeros(1, 1, 5, 5)  # logits = 0 → sigmoid = 0.5
    target = torch.zeros(1, 1, 5, 5)
    target[0, 0, 2, 2] = 1.0  # One positive center
    
    # Set prediction to match target
    pred_logits[0, 0, 2, 2] = 10.0  # High logit → sigmoid ≈ 1.0
    
    loss = focal_loss(pred_logits, target)
    print(f"  Target has 1 positive at (2,2)")
    print(f"  Pred logit at (2,2): {pred_logits[0, 0, 2, 2]:.4f} → sigmoid: {torch.sigmoid(pred_logits[0, 0, 2, 2]):.4f}")
    print(f"  Loss: {loss.item():.6f} (should be very small)")
    
    # Test 2: Complete mismatch
    print("\n[Test 2] Complete mismatch (pred opposite of target)")
    pred_logits = torch.zeros(1, 1, 5, 5)
    target = torch.zeros(1, 1, 5, 5)
    target[0, 0, 2, 2] = 1.0
    
    # Predict low probability where target is high
    pred_logits[0, 0, 2, 2] = -10.0  # Low logit → sigmoid ≈ 0.0
    
    loss = focal_loss(pred_logits, target)
    print(f"  Target has 1 positive at (2,2)")
    print(f"  Pred logit at (2,2): {pred_logits[0, 0, 2, 2]:.4f} → sigmoid: {torch.sigmoid(pred_logits[0, 0, 2, 2]):.4f}")
    print(f"  Loss: {loss.item():.6f} (should be large)")
    
    # Test 3: Realistic Gaussian target (peak at 0.995)
    print("\n[Test 3] Realistic Gaussian target (peak = 0.995)")
    pred_logits = torch.randn(1, 1, 10, 10) * 0.1  # Random predictions
    target = torch.zeros(1, 1, 10, 10)
    
    # Create Gaussian-like target (peak not exactly 1.0)
    cx, cy = 5, 5
    sigma = 2.0
    for i in range(10):
        for j in range(10):
            dist_sq = (i - cx)**2 + (j - cy)**2
            target[0, 0, i, j] = torch.exp(torch.tensor(-dist_sq / (2 * sigma**2)))
    
    print(f"  Target max value: {target.max().item():.6f}")
    print(f"  Target min value: {target.min().item():.6f}")
    print(f"  Number of pixels > 0.99: {(target > 0.99).sum().item()}")
    print(f"  Number of pixels > 0.5: {(target > 0.5).sum().item()}")
    
    loss = focal_loss(pred_logits, target)
    print(f"  Loss with random predictions: {loss.item():.6f}")
    
    # Test 4: Check gradient flow
    print("\n[Test 4] Gradient flow test")
    pred_logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros(2, 1, 8, 8)
    target[0, 0, 3, 3] = 1.0
    target[1, 0, 5, 5] = 0.998  # Realistic Gaussian peak
    
    loss = focal_loss(pred_logits, target)
    loss.backward()
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradients computed: {pred_logits.grad is not None}")
    print(f"  Gradient mean: {pred_logits.grad.abs().mean().item():.6f}")
    print(f"  Gradient max: {pred_logits.grad.abs().max().item():.6f}")
    
    # Test 5: Multiple objects
    print("\n[Test 5] Multiple objects in same image")
    pred_logits = torch.zeros(1, 1, 20, 20)
    target = torch.zeros(1, 1, 20, 20)
    
    # Add 3 Gaussian peaks
    centers = [(5, 5), (10, 15), (15, 8)]
    for cx, cy in centers:
        for i in range(20):
            for j in range(20):
                dist_sq = (i - cx)**2 + (j - cy)**2
                val = torch.exp(torch.tensor(-dist_sq / (2 * 2.0**2)))
                target[0, 0, i, j] = max(target[0, 0, i, j].item(), val.item())
    
    num_pos = (target > 0.99).sum().item()
    print(f"  Number of centers: {len(centers)}")
    print(f"  Number of positive pixels (>0.99): {num_pos}")
    print(f"  Target max: {target.max().item():.6f}")
    
    loss = focal_loss(pred_logits, target)
    print(f"  Loss: {loss.item():.6f}")
    
    # Test 6: Empty target (no objects)
    print("\n[Test 6] Empty target (no objects)")
    pred_logits = torch.randn(1, 1, 10, 10) * 0.5
    target = torch.zeros(1, 1, 10, 10)
    
    loss = focal_loss(pred_logits, target)
    print(f"  Target all zeros")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Note: Loss should penalize high predictions on background")
    
    print("\n" + "=" * 80)
    print("FOCAL LOSS TESTS COMPLETED")
    print("=" * 80)


def debug_focal_loss_on_batch(pred_logits, target, name="Batch"):
    """
    Debug focal loss computation on a batch.
    
    Args:
        pred_logits: [B, 1, H, W] prediction logits
        target: [B, 1, H, W] target heatmap
        name: Name for this batch
    """
    print(f"\n{'=' * 80}")
    print(f"FOCAL LOSS DEBUG: {name}")
    print(f"{'=' * 80}")
    
    B, C, H, W = pred_logits.shape
    pred = torch.sigmoid(pred_logits)
    
    print(f"\nInput shapes:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  target: {target.shape}")
    
    print(f"\nPrediction statistics:")
    print(f"  Logits  - min: {pred_logits.min():.4f}, max: {pred_logits.max():.4f}, mean: {pred_logits.mean():.4f}")
    print(f"  Sigmoid - min: {pred.min():.4f}, max: {pred.max():.4f}, mean: {pred.mean():.4f}")
    
    print(f"\nTarget statistics:")
    print(f"  min: {target.min():.4f}, max: {target.max():.4f}, mean: {target.mean():.4f}")
    
    # Analyze positive/negative masks
    pos_mask = (target > 0.99).float()
    neg_mask = (target <= 0.99).float()
    
    print(f"\nMask statistics:")
    print(f"  Positive pixels (target > 0.99): {pos_mask.sum().item():.0f} / {B*C*H*W}")
    print(f"  Negative pixels (target <= 0.99): {neg_mask.sum().item():.0f} / {B*C*H*W}")
    
    if pos_mask.sum() > 0:
        pos_pred = pred[pos_mask.bool()]
        pos_target = target[pos_mask.bool()]
        print(f"\nPositive pixels analysis:")
        print(f"  Target values: min={pos_target.min():.4f}, max={pos_target.max():.4f}, mean={pos_target.mean():.4f}")
        print(f"  Pred values: min={pos_pred.min():.4f}, max={pos_pred.max():.4f}, mean={pos_pred.mean():.4f}")
    else:
        print(f"\n⚠️  WARNING: No positive pixels found!")
    
    # Compute loss
    focal_loss = FocalLoss(alpha=2.0, beta=4.0)
    loss = focal_loss(pred_logits, target)
    
    print(f"\nFocal Loss: {loss.item():.6f}")
    
    # Check if loss is changing
    if pos_mask.sum() > 0:
        # Compute what loss would be with perfect prediction
        perfect_logits = pred_logits.clone()
        perfect_logits[pos_mask.bool()] = 10.0  # High confidence on positives
        perfect_logits[neg_mask.bool()] = -10.0  # Low confidence on negatives
        perfect_loss = focal_loss(perfect_logits, target)
        print(f"Loss with perfect prediction: {perfect_loss.item():.6f}")
        print(f"Improvement potential: {(loss.item() - perfect_loss.item()) / loss.item() * 100:.1f}%")
    
    print(f"{'=' * 80}\n")
    
    return loss


if __name__ == "__main__":
    test_focal_loss()
