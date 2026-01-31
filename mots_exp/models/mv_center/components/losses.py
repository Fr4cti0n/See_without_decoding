"""
MV-Center Loss Functions

Implements the complete loss function for MV-Center training:
1. Focal Loss: For center heatmap (handles class imbalance)
2. L1 + GIoU Loss: For bounding box regression
3. Optional Confidence Loss: For detection confidence

Based on CenterNet and modern object detection losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
            alpha: Weighting factor for positive examples
            beta: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmap [B, C, H, W] (after sigmoid, 0-1)
            target: Ground truth heatmap [B, C, H, W] (0-1, gaussian peaks)
            
        Returns:
            loss: Focal loss value
        """
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


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for box regression.
    
    GIoU addresses limitations of standard IoU for non-overlapping boxes
    and provides better gradients for optimization.
    """
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] in format [cx, cy, w, h]
            target_boxes: [N, 4] in format [cx, cy, w, h]
            
        Returns:
            giou_loss: 1 - GIoU
        """
        # Convert center format to corner format [x1, y1, x2, y2]
        pred_corners = self._center_to_corners(pred_boxes)
        target_corners = self._center_to_corners(target_boxes)
        
        # Calculate intersection
        lt = torch.max(pred_corners[:, :2], target_corners[:, :2])  # Left-top
        rb = torch.min(pred_corners[:, 2:], target_corners[:, 2:])  # Right-bottom
        
        intersection_wh = torch.clamp(rb - lt, min=0)
        intersection_area = intersection_wh[:, 0] * intersection_wh[:, 1]
        
        # Calculate areas
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]
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
        return 1.0 - giou
    
    def _center_to_corners(self, boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        return torch.stack([x1, y1, x2, y2], dim=-1)


class MVCenterLoss(nn.Module):
    """
    Complete MV-Center loss combining:
    1. Focal loss for center heatmap
    2. L1 + GIoU loss for box regression  
    3. Optional confidence loss
    """
    
    def __init__(self, 
                 center_weight=1.0,
                 box_weight=1.0, 
                 giou_weight=2.0,
                 conf_weight=0.5,
                 use_confidence=False,
                 focal_alpha=2.0,
                 focal_beta=4.0):
        """
        Args:
            center_weight: Weight for focal loss
            box_weight: Weight for L1 box loss
            giou_weight: Weight for GIoU loss
            conf_weight: Weight for confidence loss
            use_confidence: Whether to use confidence loss
            focal_alpha: Focal loss alpha parameter
            focal_beta: Focal loss beta parameter
        """
        super().__init__()
        
        self.center_weight = center_weight
        self.box_weight = box_weight
        self.giou_weight = giou_weight
        self.conf_weight = conf_weight
        self.use_confidence = use_confidence
        
        # Loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, beta=focal_beta)
        self.giou_loss = GIoULoss()
        
        if use_confidence:
            self.conf_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Compute MV-Center loss.
        
        Args:
            predictions: Model predictions
                - Dict with levels ('P3', 'P4')
                - Each level: {'center': [B,1,H,W], 'box': [B,4,H,W], 'conf': [B,1,H,W] (optional)}
            targets: Ground truth targets
                - Dict with levels ('P3', 'P4') 
                - Each level: {'center': [B,1,H,W], 'box': [B,4,H,W], 'mask': [B,1,H,W]}
                
        Returns:
            loss_dict: Dictionary of individual losses
            total_loss: Combined total loss
        """
        total_center_loss = 0.0
        total_box_loss = 0.0 
        total_giou_loss = 0.0
        total_conf_loss = 0.0
        
        num_levels = len(predictions)
        num_positive = 0
        
        for level in predictions.keys():
            if level not in targets:
                continue
                
            pred = predictions[level]
            target = targets[level]
            
            # Center heatmap loss (focal loss)
            center_loss = self.focal_loss(pred['center'], target['center'])
            total_center_loss += center_loss
            
            # Box regression loss (only on positive pixels)
            if 'mask' in target:
                pos_mask = target['mask'].bool()  # [B, 1, H, W]
                num_pos_level = pos_mask.sum().float()
                
                if num_pos_level > 0:
                    # Extract positive predictions and targets
                    pred_boxes_pos = pred['box'][pos_mask.expand(-1, 4, -1, -1)].view(-1, 4)
                    target_boxes_pos = target['box'][pos_mask.expand(-1, 4, -1, -1)].view(-1, 4)
                    
                    # L1 loss for box regression
                    box_l1_loss = F.l1_loss(pred_boxes_pos, target_boxes_pos, reduction='mean')
                    total_box_loss += box_l1_loss
                    
                    # Convert box regression to absolute coordinates for GIoU
                    pred_boxes_abs = self._regression_to_boxes(pred_boxes_pos, pos_mask)
                    target_boxes_abs = self._regression_to_boxes(target_boxes_pos, pos_mask)
                    
                    # GIoU loss
                    giou_loss = self.giou_loss(pred_boxes_abs, target_boxes_abs).mean()
                    total_giou_loss += giou_loss
                    
                    num_positive += num_pos_level
            
            # Optional confidence loss
            if self.use_confidence and 'conf' in pred and 'conf_target' in target:
                conf_loss = self.conf_loss(pred['conf'], target['conf_target'])
                total_conf_loss += conf_loss
        
        # Average across levels
        total_center_loss /= num_levels
        if num_positive > 0:
            total_box_loss /= num_levels
            total_giou_loss /= num_levels
        if self.use_confidence:
            total_conf_loss /= num_levels
        
        # Combine losses
        total_loss = (
            self.center_weight * total_center_loss +
            self.box_weight * total_box_loss +
            self.giou_weight * total_giou_loss +
            self.conf_weight * total_conf_loss
        )
        
        loss_dict = {
            'center_loss': total_center_loss,
            'box_loss': total_box_loss,
            'giou_loss': total_giou_loss,
            'total_loss': total_loss
        }
        
        if self.use_confidence:
            loss_dict['conf_loss'] = total_conf_loss
        
        return loss_dict, total_loss
    
    def _regression_to_boxes(self, box_regression, pos_mask):
        """
        Convert box regression [dx, dy, log(w), log(h)] to absolute coordinates.
        
        This is a simplified version - in practice you'd need the actual
        pixel coordinates where the positive predictions occurred.
        """
        dx, dy, log_w, log_h = box_regression.unbind(-1)
        
        # For simplicity, assume unit grid coordinates
        # In practice, you'd compute actual pixel coordinates
        w = torch.exp(log_w).clamp(min=0.1, max=10.0)  # Clamp for stability
        h = torch.exp(log_h).clamp(min=0.1, max=10.0)
        
        # Create dummy center coordinates (this should be real coordinates)
        cx = dx  # In practice: grid_x + dx  
        cy = dy  # In practice: grid_y + dy
        
        return torch.stack([cx, cy, w, h], dim=-1)


def create_mv_center_loss(center_weight=1.0, box_weight=1.0, giou_weight=2.0, 
                         conf_weight=0.5, use_confidence=False):
    """
    Factory function to create MV-Center loss.
    
    Args:
        center_weight: Weight for center heatmap loss  
        box_weight: Weight for box regression L1 loss
        giou_weight: Weight for GIoU loss
        conf_weight: Weight for confidence loss
        use_confidence: Whether to use confidence prediction
        
    Returns:
        loss_fn: MVCenterLoss instance
    """
    return MVCenterLoss(
        center_weight=center_weight,
        box_weight=box_weight,
        giou_weight=giou_weight,
        conf_weight=conf_weight,
        use_confidence=use_confidence
    )


if __name__ == "__main__":
    # Test the loss functions
    print("Testing MV-Center Loss Functions...")
    
    # Test focal loss
    print("\n1. Testing Focal Loss:")
    focal_loss = FocalLoss(alpha=2.0, beta=4.0)
    
    # Create dummy heatmaps
    pred_heatmap = torch.sigmoid(torch.randn(2, 1, 8, 8))  # [B, C, H, W]
    target_heatmap = torch.zeros(2, 1, 8, 8)
    target_heatmap[0, 0, 3, 3] = 1.0  # One positive pixel per batch
    target_heatmap[1, 0, 5, 2] = 1.0
    
    focal_loss_val = focal_loss(pred_heatmap, target_heatmap)
    print(f"  Focal loss: {focal_loss_val.item():.4f}")
    
    # Test GIoU loss
    print("\n2. Testing GIoU Loss:")
    giou_loss = GIoULoss()
    
    # Create dummy boxes [cx, cy, w, h]
    pred_boxes = torch.tensor([[5.0, 5.0, 2.0, 2.0], [3.0, 3.0, 1.5, 1.5]])
    target_boxes = torch.tensor([[5.1, 4.9, 2.1, 1.9], [3.2, 2.8, 1.4, 1.6]])
    
    giou_loss_val = giou_loss(pred_boxes, target_boxes).mean()
    print(f"  GIoU loss: {giou_loss_val.item():.4f}")
    
    # Test complete MV-Center loss
    print("\n3. Testing Complete MV-Center Loss:")
    mv_loss = create_mv_center_loss(
        center_weight=1.0, 
        box_weight=1.0, 
        giou_weight=2.0,
        use_confidence=False
    )
    
    # Create dummy predictions and targets
    predictions = {
        'P3': {
            'center': torch.sigmoid(torch.randn(2, 1, 15, 15)),
            'box': torch.randn(2, 4, 15, 15)
        },
        'P4': {
            'center': torch.sigmoid(torch.randn(2, 1, 8, 8)), 
            'box': torch.randn(2, 4, 8, 8)
        }
    }
    
    targets = {
        'P3': {
            'center': torch.zeros(2, 1, 15, 15),
            'box': torch.randn(2, 4, 15, 15),
            'mask': torch.zeros(2, 1, 15, 15)
        },
        'P4': {
            'center': torch.zeros(2, 1, 8, 8),
            'box': torch.randn(2, 4, 8, 8), 
            'mask': torch.zeros(2, 1, 8, 8)
        }
    }
    
    # Add some positive samples
    targets['P3']['center'][0, 0, 7, 7] = 1.0
    targets['P3']['mask'][0, 0, 7, 7] = 1.0
    targets['P4']['center'][1, 0, 3, 3] = 1.0
    targets['P4']['mask'][1, 0, 3, 3] = 1.0
    
    loss_dict, total_loss = mv_loss(predictions, targets)
    
    print(f"  Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value.item():.4f}")
    
    # Test with confidence
    print("\n4. Testing with Confidence Loss:")
    mv_loss_conf = create_mv_center_loss(use_confidence=True)
    
    # Add confidence predictions and targets
    for level in predictions:
        predictions[level]['conf'] = torch.randn_like(predictions[level]['center'])
        targets[level]['conf_target'] = torch.rand_like(targets[level]['center'])
    
    loss_dict_conf, total_loss_conf = mv_loss_conf(predictions, targets)
    
    print(f"  Loss breakdown (with confidence):")
    for key, value in loss_dict_conf.items():
        print(f"    {key}: {value.item():.4f}")
    
    print(f"\nAll loss tests completed successfully!")
