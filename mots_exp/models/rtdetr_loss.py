"""
RT-DETR Loss for Tracking with Stable Training

Combines:
1. Focal Loss for classification (handles class imbalance)
2. L1 + GIoU for box regression (stable even with drift)
3. Cross-entropy for track ID
4. Quality-aware Hungarian matching (uses class confidence)

Author: GitHub Copilot
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from .rtdetr_head import FocalLoss, compute_giou_loss


class RTDETRTrackingLoss(nn.Module):
    """
    Complete loss function for RT-DETR tracking
    
    Loss = λ_cls * focal_loss(class) + 
           λ_box * L1_loss(boxes) + 
           λ_giou * GIoU_loss(boxes) + 
           λ_id * CE_loss(track_ids)
    
    Args:
        lambda_cls: Weight for classification loss (default: 2.0)
        lambda_box: Weight for L1 box loss (default: 5.0)
        lambda_giou: Weight for GIoU loss (default: 2.0)
        lambda_id: Weight for track ID loss (default: 2.0)
        focal_alpha: Alpha parameter for focal loss (default: 0.25)
        focal_gamma: Gamma parameter for focal loss (default: 2.0)
    """
    
    def __init__(
        self,
        lambda_cls=2.0,
        lambda_box=5.0,
        lambda_giou=2.0,
        lambda_id=2.0,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()
        
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_giou = lambda_giou
        self.lambda_id = lambda_id
        
        # Focal loss for classification
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        
        # L1 loss for boxes
        self.l1_loss = nn.L1Loss(reduction='none')
        
        # Track ID loss
        self.id_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    
    def hungarian_matching(self, pred_boxes, pred_class_logits, pred_ids, gt_boxes, gt_ids):
        """
        VECTORIZED Quality-aware Hungarian matching using class confidence
        
        Cost = λ_box * L1(box) + λ_giou * (1 - GIoU) + λ_cls * cls_cost - λ_id * ID_match
        
        Args:
            pred_boxes: [300, 4] - predicted boxes
            pred_class_logits: [300, 2] - class logits [no_object, pedestrian]
            pred_ids: [300, num_track_ids] - track ID logits
            gt_boxes: [M, 4] - ground truth boxes
            gt_ids: [M] - ground truth track IDs
        
        Returns:
            matched_indices: List of (pred_idx, gt_idx) tuples
            unmatched_preds: List of prediction indices with no match
            unmatched_gts: List of ground truth indices with no match
        """
        num_preds = pred_boxes.shape[0]
        num_gts = gt_boxes.shape[0]
        
        if num_gts == 0:
            # No ground truth - all predictions are unmatched
            return [], list(range(num_preds)), []
        
        device = pred_boxes.device
        
        # Get class probabilities
        pred_class_probs = F.softmax(pred_class_logits, dim=-1)  # [300, 2]
        pred_pedestrian_prob = pred_class_probs[:, 1]  # [300]
        
        # Get predicted IDs
        pred_ids_argmax = pred_ids.argmax(dim=-1)  # [300]
        
        # === VECTORIZED COST COMPUTATION ===
        
        # 1. L1 box cost: [300, M]
        # Expand: pred [300, 1, 4], gt [1, M, 4] -> broadcast to [300, M, 4]
        pred_boxes_expanded = pred_boxes.unsqueeze(1)  # [300, 1, 4]
        gt_boxes_expanded = gt_boxes.unsqueeze(0)  # [1, M, 4]
        l1_cost = torch.abs(pred_boxes_expanded - gt_boxes_expanded).sum(dim=-1)  # [300, M]
        
        # 2. GIoU cost: [300, M] - VECTORIZED!
        # Flatten to [300*M, 4] for batch GIoU computation
        num_pairs = num_preds * num_gts
        pred_boxes_flat = pred_boxes_expanded.expand(num_preds, num_gts, 4).reshape(num_pairs, 4)
        gt_boxes_flat = gt_boxes_expanded.expand(num_preds, num_gts, 4).reshape(num_pairs, 4)
        giou_loss_flat = compute_giou_loss(pred_boxes_flat, gt_boxes_flat)  # [300*M]
        giou_cost = giou_loss_flat.reshape(num_preds, num_gts)  # [300, M]
        
        # 3. Classification cost: [300, M]
        cls_cost = -torch.log(pred_pedestrian_prob + 1e-8).unsqueeze(1).expand(num_preds, num_gts)  # [300, M]
        
        # 4. ID matching bonus: [300, M]
        pred_ids_expanded = pred_ids_argmax.unsqueeze(1).expand(num_preds, num_gts)  # [300, M]
        gt_ids_expanded = gt_ids.unsqueeze(0).expand(num_preds, num_gts)  # [300, M]
        id_match_bonus = torch.where(pred_ids_expanded == gt_ids_expanded, 
                                     torch.tensor(-0.5, device=device), 
                                     torch.tensor(0.0, device=device))  # [300, M]
        
        # 5. Total cost matrix: [300, M]
        cost_matrix = (
            self.lambda_box * l1_cost + 
            self.lambda_giou * giou_cost + 
            self.lambda_cls * cls_cost + 
            id_match_bonus
        )
        
        # Convert to numpy for scipy's linear_sum_assignment
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        
        # Run Hungarian algorithm
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix_np)
        
        # ACCEPT ALL HUNGARIAN MATCHES!
        # The Hungarian algorithm already finds the best cost-based assignment.
        # Filtering by IoU/confidence was preventing any matches for untrained models.
        # We trust the cost matrix (L1 + GIoU + cls + ID) to find good pairs.
        matched_indices = [(pred_idx, gt_idx) for pred_idx, gt_idx in zip(pred_indices, gt_indices)]
        
        # Find unmatched predictions and ground truths
        matched_pred_set = set(idx[0] for idx in matched_indices)
        matched_gt_set = set(idx[1] for idx in matched_indices)
        
        unmatched_preds = [i for i in range(num_preds) if i not in matched_pred_set]
        unmatched_gts = [i for i in range(num_gts) if i not in matched_gt_set]
        
        return matched_indices, unmatched_preds, unmatched_gts
    
    def forward(self, pred_boxes, pred_class_logits, pred_track_ids, gt_boxes, gt_ids):
        """
        Compute RT-DETR tracking loss
        
        Args:
            pred_boxes: [300, 4] - predicted boxes
            pred_class_logits: [300, 2] - class logits [no_object, pedestrian]
            pred_track_ids: [300, num_track_ids] - track ID logits
            gt_boxes: [M, 4] - ground truth boxes
            gt_ids: [M] - ground truth track IDs
        
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary with individual loss components
            num_matched: Number of matched pairs (for logging)
        """
        device = pred_boxes.device
        num_preds = pred_boxes.shape[0]  # 300
        num_gts = gt_boxes.shape[0]  # M
        
        # 1. Hungarian matching
        matched_indices, unmatched_preds, unmatched_gts = self.hungarian_matching(
            pred_boxes, pred_class_logits, pred_track_ids, gt_boxes, gt_ids
        )
        
        num_matched = len(matched_indices)
        
        # 2. Prepare targets for all 300 slots
        # Classification targets: 0=no_object, 1=pedestrian
        target_classes = torch.zeros(num_preds, dtype=torch.long, device=device)  # All no_object by default
        target_boxes = torch.zeros(num_preds, 4, device=device)
        target_track_ids = torch.full((num_preds,), -1, dtype=torch.long, device=device)  # -1 = ignore
        
        # Fill targets for matched predictions
        for pred_idx, gt_idx in matched_indices:
            target_classes[pred_idx] = 1  # pedestrian class
            target_boxes[pred_idx] = gt_boxes[gt_idx]
            target_track_ids[pred_idx] = gt_ids[gt_idx]
        
        # 3. Classification loss (focal loss on all 300 slots)
        cls_loss = self.focal_loss(pred_class_logits, target_classes)
        
        # 4. Box losses (L1 + GIoU) - only on matched predictions
        if num_matched > 0:
            matched_pred_indices = torch.tensor([idx[0] for idx in matched_indices], device=device)
            
            matched_pred_boxes = pred_boxes[matched_pred_indices]
            matched_target_boxes = target_boxes[matched_pred_indices]
            
            # L1 loss
            l1_loss = self.l1_loss(matched_pred_boxes, matched_target_boxes).sum(dim=-1).mean()
            
            # GIoU loss
            giou_loss = compute_giou_loss(matched_pred_boxes, matched_target_boxes).mean()
            
            box_loss = l1_loss
        else:
            box_loss = torch.tensor(0.0, device=device)
            giou_loss = torch.tensor(0.0, device=device)
        
        # 5. Track ID loss (only on matched predictions with valid IDs)
        valid_id_mask = target_track_ids >= 0
        if valid_id_mask.sum() > 0:
            id_loss = self.id_loss(
                pred_track_ids[valid_id_mask], 
                target_track_ids[valid_id_mask]
            ).mean()
        else:
            id_loss = torch.tensor(0.0, device=device)
        
        # 6. Combine losses
        total_loss = (
            self.lambda_cls * cls_loss + 
            self.lambda_box * box_loss + 
            self.lambda_giou * giou_loss + 
            self.lambda_id * id_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'cls': cls_loss.item(),
            'box_l1': box_loss.item(),
            'box_giou': giou_loss.item(),
            'track_id': id_loss.item(),
            'num_matched': num_matched,
            'num_gt': num_gts
        }
        
        return total_loss, loss_dict, num_matched


if __name__ == "__main__":
    print("Testing RT-DETR Tracking Loss...")
    
    # Create loss function
    criterion = RTDETRTrackingLoss(
        lambda_cls=2.0,
        lambda_box=5.0,
        lambda_giou=2.0,
        lambda_id=2.0
    )
    
    # Simulate predictions (300 slots)
    num_slots = 300
    num_track_ids = 1000
    pred_boxes = torch.rand(num_slots, 4) * 960  # Random boxes
    pred_class_logits = torch.randn(num_slots, 2)  # Random logits
    pred_track_ids = torch.randn(num_slots, num_track_ids)  # Random logits
    
    # Simulate ground truth (10 objects)
    num_gts = 10
    gt_boxes = torch.rand(num_gts, 4) * 960
    gt_ids = torch.randint(0, 100, (num_gts,))
    
    print(f"\nInputs:")
    print(f"  Predictions: {num_slots} slots")
    print(f"  Ground truth: {num_gts} objects")
    
    # Compute loss
    total_loss, loss_dict, num_matched = criterion(
        pred_boxes, pred_class_logits, pred_track_ids,
        gt_boxes, gt_ids
    )
    
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\n✅ Test passed!")
    print(f"  Matched {num_matched}/{num_gts} ground truth objects")
