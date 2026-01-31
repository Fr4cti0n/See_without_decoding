"""
Enhanced Detection Loss with No-Object Classification

Implements DETR-style loss with:
1. Hungarian matching for optimal object assignment
2. Binary classification (object vs no-object)
3. Focal loss for class imbalance
4. No-object penalty for false positives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for optimal bipartite matching between predictions and targets.
    
    Similar to DETR's matcher but adapted for tracking scenario.
    """
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(self, pred_boxes, pred_logits, target_boxes):
        """
        Compute optimal matching between predictions and targets.
        
        Args:
            pred_boxes: [N, 4] predicted boxes in [cx, cy, w, h]
            pred_logits: [N, 1] or [N] objectness logits
            target_boxes: [M, 4] target boxes in [cx, cy, w, h]
        
        Returns:
            pred_indices: [K] indices of matched predictions
            target_indices: [K] indices of matched targets
        """
        N = pred_boxes.shape[0]
        M = target_boxes.shape[0]
        
        if N == 0 or M == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        
        # Ensure pred_logits is 1D
        if pred_logits.ndim == 2:
            pred_logits = pred_logits.squeeze(-1)
        
        # Classification cost (probability of being an object)
        prob = pred_logits.sigmoid()
        cost_class = -prob.unsqueeze(1).expand(N, M)  # [N, M] - we want high prob for objects
        
        # L1 cost on boxes
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)  # [N, M]
        
        # GIoU cost
        cost_giou = -self.compute_giou(pred_boxes, target_boxes)  # [N, M]
        
        # Final cost matrix
        C = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )
        
        # Hungarian algorithm
        C_np = C.cpu().numpy()
        pred_idx, target_idx = linear_sum_assignment(C_np)
        
        return torch.as_tensor(pred_idx, dtype=torch.long), torch.as_tensor(target_idx, dtype=torch.long)
    
    def compute_giou(self, boxes1, boxes2):
        """
        Compute pairwise GIoU between two sets of boxes.
        
        Args:
            boxes1: [N, 4] boxes in [cx, cy, w, h]
            boxes2: [M, 4] boxes in [cx, cy, w, h]
        
        Returns:
            giou: [N, M] pairwise GIoU
        """
        # Convert to [x1, y1, x2, y2]
        boxes1_xyxy = self.cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self.cxcywh_to_xyxy(boxes2)
        
        N, M = boxes1.shape[0], boxes2.shape[0]
        
        # Expand for pairwise computation
        boxes1_exp = boxes1_xyxy.unsqueeze(1).expand(N, M, 4)  # [N, M, 4]
        boxes2_exp = boxes2_xyxy.unsqueeze(0).expand(N, M, 4)  # [N, M, 4]
        
        # Intersection
        x1_inter = torch.max(boxes1_exp[:, :, 0], boxes2_exp[:, :, 0])
        y1_inter = torch.max(boxes1_exp[:, :, 1], boxes2_exp[:, :, 1])
        x2_inter = torch.min(boxes1_exp[:, :, 2], boxes2_exp[:, :, 2])
        y2_inter = torch.min(boxes1_exp[:, :, 3], boxes2_exp[:, :, 3])
        
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
        
        # Areas
        area1 = (boxes1_exp[:, :, 2] - boxes1_exp[:, :, 0]) * (boxes1_exp[:, :, 3] - boxes1_exp[:, :, 1])
        area2 = (boxes2_exp[:, :, 2] - boxes2_exp[:, :, 0]) * (boxes2_exp[:, :, 3] - boxes2_exp[:, :, 1])
        
        # Union
        union = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union + 1e-6)
        
        # Enclosing box
        x1_enc = torch.min(boxes1_exp[:, :, 0], boxes2_exp[:, :, 0])
        y1_enc = torch.min(boxes1_exp[:, :, 1], boxes2_exp[:, :, 1])
        x2_enc = torch.max(boxes1_exp[:, :, 2], boxes2_exp[:, :, 2])
        y2_enc = torch.max(boxes1_exp[:, :, 3], boxes2_exp[:, :, 3])
        
        enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
        
        # GIoU
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        
        return giou
    
    @staticmethod
    def cxcywh_to_xyxy(boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where p_t is the model's estimated probability for the class with label y.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [N] or [N, 1] predicted logits
            targets: [N] binary targets (0 or 1)
        
        Returns:
            loss: Scalar focal loss
        """
        # Ensure 1D
        if logits.ndim == 2:
            logits = logits.squeeze(-1)
        
        # Compute probability
        p = torch.sigmoid(logits)
        
        # Focal loss formula
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)  # p if y=1, 1-p if y=0
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        return loss.mean()


class DetectionLossWithNoObject(nn.Module):
    """
    Enhanced detection loss with no-object classification.
    
    Combines:
    1. Box regression loss (L1 + GIoU) for matched objects
    2. Classification loss (Focal) for object vs no-object
    3. Velocity consistency loss (optional)
    """
    def __init__(
        self,
        box_weight=5.0,
        class_weight=2.0,
        giou_weight=2.0,
        no_object_weight=0.1,
        velocity_weight=0.0,
        use_focal_loss=True,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()
        self.box_weight = box_weight
        self.class_weight = class_weight
        self.giou_weight = giou_weight
        self.no_object_weight = no_object_weight
        self.velocity_weight = velocity_weight
        
        # Matcher for Hungarian assignment
        self.matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_bbox=box_weight,
            cost_giou=giou_weight
        )
        
        # Classification loss
        if use_focal_loss:
            self.class_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.class_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets, pred_logits, confidences=None):
        """
        Compute detection loss with no-object classification.
        
        Args:
            predictions: List[Tensor[N_i, 4]] predicted boxes per frame
            targets: List[Tensor[M_i, 4]] target boxes per frame
            pred_logits: List[Tensor[N_i]] or List[Tensor[N_i, 1]] objectness logits
            confidences: Optional, not used (kept for compatibility)
        
        Returns:
            loss_dict: Dictionary with loss components
        """
        device = predictions[0].device if len(predictions) > 0 else torch.device('cpu')
        
        # ✅ CRITICAL: Initialize as None and accumulate losses properly
        # Using torch.tensor(0.0) creates leaf tensors that break gradient flow
        total_box_loss = None
        total_giou_loss = None
        total_class_loss = None
        total_no_obj_loss = None
        num_frames = 0
        num_matched = 0
        
        for pred, tgt, logits in zip(predictions, targets, pred_logits):
            if pred.shape[0] == 0:
                continue
            
            num_frames += 1
            
            # Ensure logits is 1D
            if logits.ndim == 2:
                logits = logits.squeeze(-1)
            
            if tgt.shape[0] == 0:
                # No ground truth - all predictions should be no-object
                no_obj_targets = torch.zeros_like(logits)
                no_obj_loss = self.class_loss_fn(logits, no_obj_targets)
                total_no_obj_loss = no_obj_loss if total_no_obj_loss is None else total_no_obj_loss + no_obj_loss
                continue
            
            # Hungarian matching
            pred_idx, tgt_idx = self.matcher(pred, logits, tgt)
            
            if len(pred_idx) > 0:
                # Matched predictions - object loss
                matched_pred = pred[pred_idx]
                matched_tgt = tgt[tgt_idx]
                matched_logits = logits[pred_idx]
                
                # Box regression loss (L1)
                box_loss = F.l1_loss(matched_pred, matched_tgt)
                total_box_loss = box_loss if total_box_loss is None else total_box_loss + box_loss
                
                # GIoU loss
                giou_loss = self.compute_giou_loss(matched_pred, matched_tgt)
                total_giou_loss = giou_loss if total_giou_loss is None else total_giou_loss + giou_loss
                
                # Classification loss for matched (should be object = 1)
                obj_targets = torch.ones_like(matched_logits)
                class_loss = self.class_loss_fn(matched_logits, obj_targets)
                total_class_loss = class_loss if total_class_loss is None else total_class_loss + class_loss
                
                num_matched += len(pred_idx)
            
            # Unmatched predictions - no-object loss
            if len(pred_idx) < pred.shape[0]:
                # Create mask for unmatched predictions
                all_idx = torch.arange(pred.shape[0], device=device)
                matched_mask = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
                matched_mask[pred_idx] = True
                unmatched_idx = all_idx[~matched_mask]
                
                if len(unmatched_idx) > 0:
                    unmatched_logits = logits[unmatched_idx]
                    no_obj_targets = torch.zeros_like(unmatched_logits)
                    no_obj_loss = self.class_loss_fn(unmatched_logits, no_obj_targets)
                    total_no_obj_loss = no_obj_loss if total_no_obj_loss is None else total_no_obj_loss + no_obj_loss
        
        # Average over frames and handle None values
        if num_frames > 0:
            total_box_loss = (total_box_loss / num_frames) if total_box_loss is not None else torch.tensor(0.0, device=device, requires_grad=True)
            total_giou_loss = (total_giou_loss / num_frames) if total_giou_loss is not None else torch.tensor(0.0, device=device, requires_grad=True)
            total_class_loss = (total_class_loss / num_frames) if total_class_loss is not None else torch.tensor(0.0, device=device, requires_grad=True)
            total_no_obj_loss = (total_no_obj_loss / num_frames) if total_no_obj_loss is not None else torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # No frames processed - return zero losses
            total_box_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_giou_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_class_loss = torch.tensor(0.0, device=device, requires_grad=True)
            total_no_obj_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Weighted total loss
        total_loss = (
            self.box_weight * total_box_loss +
            self.giou_weight * total_giou_loss +
            self.class_weight * total_class_loss +
            self.no_object_weight * total_no_obj_loss
        )
        
        loss_dict = {
            'box': total_box_loss,
            'giou': total_giou_loss,
            'classification': total_class_loss,
            'no_object': total_no_obj_loss,
            'velocity': torch.tensor(0.0, device=device),  # Placeholder for compatibility
            'confidence': total_class_loss  # Alias for compatibility
        }
        
        # Return tuple (loss, loss_dict) as expected by training code
        return total_loss, loss_dict
    
    def compute_giou_loss(self, boxes1, boxes2):
        """Compute GIoU loss between matched boxes."""
        # Convert to [x1, y1, x2, y2]
        boxes1_xyxy = self.cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self.cxcywh_to_xyxy(boxes2)
        
        # Intersection
        x1_inter = torch.max(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
        y1_inter = torch.max(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
        x2_inter = torch.min(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
        y2_inter = torch.min(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
        
        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)
        
        # Areas
        area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
        area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
        
        # Union
        union = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / (union + 1e-6)
        
        # Enclosing box
        x1_enc = torch.min(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
        y1_enc = torch.min(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
        x2_enc = torch.max(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
        y2_enc = torch.max(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
        
        enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
        
        # GIoU
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        
        # Loss (1 - GIoU)
        return (1 - giou).mean()
    
    @staticmethod
    def cxcywh_to_xyxy(boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]"""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
