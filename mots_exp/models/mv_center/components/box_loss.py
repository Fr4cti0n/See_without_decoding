"""
Box Regression Losses (L1 + GIoU)

Standalone box loss implementations with testing utilities
to debug bounding box regression issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1BoxLoss(nn.Module):
    """
    L1 loss for box regression.
    Simple and effective for box coordinate prediction.
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred_boxes, target_boxes, mask=None):
        """
        Args:
            pred_boxes: [B, 4, H, W] or [N, 4] predicted boxes
            target_boxes: [B, 4, H, W] or [N, 4] target boxes
            mask: [B, 1, H, W] or [N, 1] mask for valid boxes (optional)
            
        Returns:
            loss: L1 loss value
        """
        if mask is not None:
            # Extract only valid boxes
            if mask.dim() == 4:  # [B, 1, H, W]
                mask = mask.expand(-1, 4, -1, -1)  # [B, 4, H, W]
                pred_boxes = pred_boxes[mask.bool()].view(-1, 4)
                target_boxes = target_boxes[mask.bool()].view(-1, 4)
            else:  # [N, 1]
                mask = mask.expand(-1, 4)  # [N, 4]
                pred_boxes = pred_boxes[mask.bool()].view(-1, 4)
                target_boxes = target_boxes[mask.bool()].view(-1, 4)
        
        if pred_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        return F.l1_loss(pred_boxes, target_boxes, reduction=self.reduction)


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for box regression.
    
    GIoU addresses limitations of standard IoU for non-overlapping boxes
    and provides better gradients for optimization.
    """
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes, format='cxcywh'):
        """
        Args:
            pred_boxes: [N, 4] in format specified by 'format' parameter
            target_boxes: [N, 4] in format specified by 'format' parameter
            format: 'cxcywh' (center x, center y, width, height) or
                    'xyxy' (x1, y1, x2, y2)
            
        Returns:
            giou_loss: 1 - GIoU (mean over all boxes)
        """
        if pred_boxes.numel() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # Convert to corner format if needed
        if format == 'cxcywh':
            pred_corners = self._center_to_corners(pred_boxes)
            target_corners = self._center_to_corners(target_boxes)
        else:
            pred_corners = pred_boxes
            target_corners = target_boxes
        
        # Calculate intersection
        lt = torch.max(pred_corners[:, :2], target_corners[:, :2])  # Left-top
        rb = torch.min(pred_corners[:, 2:], target_corners[:, 2:])  # Right-bottom
        
        intersection_wh = torch.clamp(rb - lt, min=0)
        intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]
        
        # Calculate areas
        if format == 'cxcywh':
            pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
            target_area = target_boxes[:, 2] * target_boxes[:, 3]
        else:
            pred_wh = pred_corners[:, 2:] - pred_corners[:, :2]
            target_wh = target_corners[:, 2:] - target_corners[:, :2]
            pred_area = pred_wh[:, 0] * pred_wh[:, 1]
            target_area = target_wh[:, 0] * target_wh[:, 1]
        
        union_area = pred_area + target_area - intersection_area
        
        # IoU
        iou = intersection_area / (union_area + self.eps)
        
        # Calculate enclosing box for GIoU
        enclosing_lt = torch.min(pred_corners[:, :2], target_corners[:, :2])
        enclosing_rb = torch.max(pred_corners[:, 2:], target_corners[:, 2:])
        enclosing_wh = torch.clamp(enclosing_rb - enclosing_lt, min=0)
        enclosing_area = enclosing_wh[:, 0] * enclosing_wh[:, 1]
        
        # GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + self.eps)
        
        # Return loss (1 - GIoU)
        return (1.0 - giou).mean()
    
    def _center_to_corners(self, boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=-1)


def test_box_losses():
    """
    Test box losses with various scenarios.
    """
    print("=" * 80)
    print("TESTING BOX LOSSES")
    print("=" * 80)
    
    l1_loss = L1BoxLoss()
    giou_loss = GIoULoss()
    
    # Test 1: Perfect prediction
    print("\n[Test 1] Perfect prediction (pred = target)")
    pred = torch.tensor([[5.0, 5.0, 2.0, 2.0], [10.0, 10.0, 3.0, 3.0]])  # [cx, cy, w, h]
    target = pred.clone()
    
    l1 = l1_loss(pred, target)
    giou = giou_loss(pred, target, format='cxcywh')
    
    print(f"  L1 Loss: {l1.item():.6f} (should be 0)")
    print(f"  GIoU Loss: {giou.item():.6f} (should be 0)")
    
    # Test 2: Slight offset
    print("\n[Test 2] Slight offset (0.1 units in each direction)")
    pred = torch.tensor([[5.1, 5.1, 2.0, 2.0], [10.1, 10.1, 3.0, 3.0]])
    target = torch.tensor([[5.0, 5.0, 2.0, 2.0], [10.0, 10.0, 3.0, 3.0]])
    
    l1 = l1_loss(pred, target)
    giou = giou_loss(pred, target, format='cxcywh')
    
    print(f"  L1 Loss: {l1.item():.6f}")
    print(f"  GIoU Loss: {giou.item():.6f}")
    
    # Test 3: Size mismatch
    print("\n[Test 3] Size mismatch (predicted box too large)")
    pred = torch.tensor([[5.0, 5.0, 4.0, 4.0]])  # Double size
    target = torch.tensor([[5.0, 5.0, 2.0, 2.0]])
    
    l1 = l1_loss(pred, target)
    giou = giou_loss(pred, target, format='cxcywh')
    
    print(f"  L1 Loss: {l1.item():.6f}")
    print(f"  GIoU Loss: {giou.item():.6f}")
    
    # Test 4: Non-overlapping boxes
    print("\n[Test 4] Non-overlapping boxes")
    pred = torch.tensor([[0.0, 0.0, 1.0, 1.0]])  # At origin
    target = torch.tensor([[10.0, 10.0, 1.0, 1.0]])  # Far away
    
    l1 = l1_loss(pred, target)
    giou = giou_loss(pred, target, format='cxcywh')
    
    print(f"  L1 Loss: {l1.item():.6f}")
    print(f"  GIoU Loss: {giou.item():.6f} (should be close to 2.0)")
    
    # Test 5: With mask (some boxes invalid)
    print("\n[Test 5] With mask (filtering valid boxes)")
    pred = torch.randn(2, 4, 5, 5)
    target = torch.randn(2, 4, 5, 5)
    mask = torch.zeros(2, 1, 5, 5)
    mask[0, 0, 2, 2] = 1  # Only one valid box
    mask[1, 0, 3, 3] = 1  # And another
    
    l1 = l1_loss(pred, target, mask)
    
    print(f"  Total pixels: {2 * 5 * 5}")
    print(f"  Valid pixels: {mask.sum().item():.0f}")
    print(f"  L1 Loss: {l1.item():.6f}")
    
    # Test 6: Gradient flow
    print("\n[Test 6] Gradient flow test")
    pred = torch.tensor([[5.0, 5.0, 2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[5.5, 5.5, 2.5, 2.5]])
    
    giou = giou_loss(pred, target, format='cxcywh')
    giou.backward()
    
    print(f"  GIoU Loss: {giou.item():.6f}")
    print(f"  Gradients: {pred.grad}")
    print(f"  Gradient magnitude: {pred.grad.abs().mean().item():.6f}")
    
    # Test 7: Batch processing
    print("\n[Test 7] Batch of boxes")
    pred = torch.randn(10, 4).abs() + 1.0  # Ensure positive sizes
    target = pred + torch.randn(10, 4) * 0.2  # Add noise
    
    l1 = l1_loss(pred, target)
    giou = giou_loss(pred, target, format='cxcywh')
    
    print(f"  Number of boxes: 10")
    print(f"  L1 Loss: {l1.item():.6f}")
    print(f"  GIoU Loss: {giou.item():.6f}")
    
    print("\n" + "=" * 80)
    print("BOX LOSS TESTS COMPLETED")
    print("=" * 80)


def debug_box_loss_on_batch(pred_boxes, target_boxes, mask, name="Batch"):
    """
    Debug box loss computation on a batch.
    
    Args:
        pred_boxes: [B, 4, H, W] predicted boxes
        target_boxes: [B, 4, H, W] target boxes  
        mask: [B, 1, H, W] valid box mask
        name: Name for this batch
    """
    print(f"\n{'=' * 80}")
    print(f"BOX LOSS DEBUG: {name}")
    print(f"{'=' * 80}")
    
    B, C, H, W = pred_boxes.shape
    
    print(f"\nInput shapes:")
    print(f"  pred_boxes: {pred_boxes.shape}")
    print(f"  target_boxes: {target_boxes.shape}")
    print(f"  mask: {mask.shape}")
    
    num_valid = mask.sum().item()
    print(f"\nValid boxes:")
    print(f"  Number of valid boxes: {num_valid:.0f}")
    print(f"  Percentage: {num_valid / (B*H*W) * 100:.2f}%")
    
    if num_valid > 0:
        # Extract valid boxes
        mask_expanded = mask.expand(-1, 4, -1, -1)
        valid_pred = pred_boxes[mask_expanded.bool()].view(-1, 4)
        valid_target = target_boxes[mask_expanded.bool()].view(-1, 4)
        
        print(f"\nPredicted box statistics:")
        for i, name in enumerate(['cx', 'cy', 'w', 'h']):
            print(f"  {name}: min={valid_pred[:, i].min():.4f}, max={valid_pred[:, i].max():.4f}, mean={valid_pred[:, i].mean():.4f}")
        
        print(f"\nTarget box statistics:")
        for i, name in enumerate(['cx', 'cy', 'w', 'h']):
            print(f"  {name}: min={valid_target[:, i].min():.4f}, max={valid_target[:, i].max():.4f}, mean={valid_target[:, i].mean():.4f}")
        
        # Compute losses
        l1_loss = L1BoxLoss()
        giou_loss = GIoULoss()
        
        l1 = l1_loss(pred_boxes, target_boxes, mask)
        giou = giou_loss(valid_pred, valid_target, format='cxcywh')
        
        print(f"\nLosses:")
        print(f"  L1 Loss: {l1.item():.6f}")
        print(f"  GIoU Loss: {giou.item():.6f}")
        
        # Analyze differences
        diff = (valid_pred - valid_target).abs()
        print(f"\nPrediction errors:")
        for i, name in enumerate(['cx', 'cy', 'w', 'h']):
            print(f"  {name} error: mean={diff[:, i].mean():.4f}, max={diff[:, i].max():.4f}")
        
    else:
        print(f"\n⚠️  WARNING: No valid boxes found in mask!")
    
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    test_box_losses()
