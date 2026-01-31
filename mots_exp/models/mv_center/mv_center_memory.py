"""
Memory-Based Motion Vector Tracker

Main model for GOP-level object tracking using motion vectors and LSTM.
"""

import torch
import torch.nn as nn
from .components.lstm_tracker import MotionGuidedTracker


class MVCenterMemoryTracker(nn.Module):
    """
    Memory-based tracker for motion vector-only object detection.
    
    Architecture:
    1. I-frame: Initialize object memory from ground truth boxes
    2. P-frames: Update positions using MV + LSTM
    3. Training: GOP-level sequences with box regression loss
    
    Input: Motion vector sequences [T, 2, H, W]
    Output: Tracked bounding boxes per frame
    
    Args:
        feature_dim: Dimension of motion features
        hidden_dim: Dimension of LSTM hidden state
        max_objects: Maximum number of objects
        grid_size: Size of motion vector grid
        image_size: Image size in pixels
        use_roi_align: If True, use ROI Align for spatially-aligned motion features
        roi_size: Output size for ROI Align (e.g., (7, 7))
    """
    
    def __init__(self, feature_dim=128, hidden_dim=256, max_objects=100, 
                 grid_size=40, image_size=640, use_roi_align=False, roi_size=(7, 7)):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.grid_size = grid_size
        self.image_size = image_size
        self.use_roi_align = use_roi_align
        self.roi_size = roi_size
        
        # Core tracker
        self.tracker = MotionGuidedTracker(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_objects=max_objects,
            use_roi_align=use_roi_align,
            roi_size=roi_size
        )
    
    def forward_gop(self, motion_sequences, iframe_boxes, iframe_ids=None):
        """
        Forward pass for a complete GOP sequence.
        
        Args:
            motion_sequences: [T, 2, H, W] motion vectors for T P-frames
            iframe_boxes: [N, 4] initial bounding boxes from I-frame
            iframe_ids: [N] object IDs (optional)
            
        Returns:
            predictions: List of [N_t, 4] bounding boxes per frame
            confidences: List of [N_t] confidences per frame
        """
        # Initialize from I-frame
        self.tracker.init_from_iframe(iframe_boxes, iframe_ids)
        
        num_frames = len(motion_sequences)
        predictions = []
        confidences = []
        
        # Process each P-frame sequentially
        for t in range(num_frames):
            mv_t = motion_sequences[t]  # [2, H, W]
            
            # Update tracking
            boxes_t, conf_t = self.tracker.forward_pframe(
                mv_t, self.grid_size, self.image_size
            )
            
            predictions.append(boxes_t)
            confidences.append(conf_t)
        
        return predictions, confidences
    
    def forward(self, motion_vectors, iframe_boxes=None, mode='single_frame'):
        """
        Flexible forward pass.
        
        Args:
            motion_vectors: [2, H, W] or [T, 2, H, W]
            iframe_boxes: [N, 4] initial boxes (required for GOP mode)
            mode: 'single_frame' or 'gop_sequence'
            
        Returns:
            Depends on mode:
            - single_frame: (boxes, confidences)
            - gop_sequence: (predictions_list, confidences_list)
        """
        if mode == 'gop_sequence':
            if iframe_boxes is None:
                raise ValueError("iframe_boxes required for GOP sequence mode")
            return self.forward_gop(motion_vectors, iframe_boxes)
        
        elif mode == 'single_frame':
            # Single P-frame update
            boxes, conf = self.tracker.forward_pframe(
                motion_vectors, self.grid_size, self.image_size
            )
            return boxes, conf
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def reset(self):
        """Reset tracker for new GOP."""
        self.tracker.reset()
    
    def get_state(self):
        """Get current tracker state."""
        return self.tracker.get_current_state()
    
    def get_model_info(self):
        """Get model configuration info for checkpointing."""
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'max_objects': self.max_objects,
            'grid_size': self.grid_size,
            'image_size': self.image_size,
            'use_roi_align': self.use_roi_align,
            'roi_size': self.roi_size
        }


class MVCenterMemoryLoss(nn.Module):
    """
    Loss function for memory-based tracker.
    
    Components:
    1. Box regression loss (L1 + IoU)
    2. Velocity consistency loss
    3. Confidence loss
    
    Supports dynamic loss balancing via optional balancer.
    """
    
    def __init__(self, box_weight=1.0, velocity_weight=0.5, conf_weight=0.1, 
                 use_dynamic_balancing=False):
        super().__init__()
        self.box_weight = box_weight
        self.velocity_weight = velocity_weight
        self.conf_weight = conf_weight
        self.use_dynamic_balancing = use_dynamic_balancing
        
        # Dynamic loss balancer (optional)
        self.balancer = None
        if use_dynamic_balancing:
            from .components.dynamic_loss_balancer import create_memory_tracker_balancer
            self.balancer = create_memory_tracker_balancer({
                'box': box_weight,
                'velocity': velocity_weight,
                'confidence': conf_weight
            })
    
    def box_loss(self, pred_boxes, target_boxes):
        """
        Box regression loss (L1 + GIoU).
        
        Handles case where #predictions != #targets by matching closest boxes.
        
        Args:
            pred_boxes: [N, 4] predicted boxes
            target_boxes: [M, 4] target boxes (M may != N)
            
        Returns:
            loss: Scalar loss
        """
        # Handle different number of predictions vs targets
        n_pred = len(pred_boxes)
        n_tgt = len(target_boxes)
        
        if n_pred == 0 or n_tgt == 0:
            return torch.tensor(0.0, device=pred_boxes.device if n_pred > 0 else target_boxes.device)
        
        if n_pred == n_tgt:
            # Same number - direct matching
            matched_pred = pred_boxes
            matched_tgt = target_boxes
        else:
            # Different numbers - use simple matching (take min)
            # TODO: Implement Hungarian matching for better association
            n_match = min(n_pred, n_tgt)
            matched_pred = pred_boxes[:n_match]
            matched_tgt = target_boxes[:n_match]
        
        # L1 loss on box coordinates
        l1_loss = nn.functional.l1_loss(matched_pred, matched_tgt)
        
        # GIoU loss
        giou_loss = self.giou_loss(matched_pred, matched_tgt)
        
        return l1_loss + giou_loss
    
    def giou_loss(self, boxes1, boxes2):
        """
        Generalized IoU loss.
        
        Args:
            boxes1: [N, 4] boxes in [cx, cy, w, h] format
            boxes2: [N, 4] boxes in [cx, cy, w, h] format
            
        Returns:
            loss: Scalar GIoU loss
        """
        # Convert to [x1, y1, x2, y2]
        boxes1_xyxy = self.cxcywh_to_xyxy(boxes1)
        boxes2_xyxy = self.cxcywh_to_xyxy(boxes2)
        
        # Calculate IoU
        iou = self.box_iou(boxes1_xyxy, boxes2_xyxy)
        
        # Calculate enclosing box
        x1_enc = torch.min(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
        y1_enc = torch.min(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
        x2_enc = torch.max(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
        y2_enc = torch.max(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
        
        enc_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)
        
        # Union area
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        union = area1 + area2 - iou * torch.min(area1, area2)
        
        # GIoU
        giou = iou - (enc_area - union) / (enc_area + 1e-6)
        
        # Loss
        loss = (1 - giou).mean()
        
        return loss
    
    def box_iou(self, boxes1, boxes2):
        """
        Calculate IoU between boxes.
        
        Args:
            boxes1: [N, 4] in xyxy format
            boxes2: [N, 4] in xyxy format
            
        Returns:
            iou: [N] IoU values
        """
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        union_area = area1 + area2 - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    def cxcywh_to_xyxy(self, boxes):
        """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def velocity_loss(self, pred_boxes_seq, target_boxes_seq):
        """
        Velocity consistency loss.
        
        Ensures predicted motion is consistent with ground truth motion.
        Handles variable number of objects per frame.
        
        Args:
            pred_boxes_seq: List of [N_t, 4] predicted boxes over time
            target_boxes_seq: List of [M_t, 4] target boxes over time
            
        Returns:
            loss: Scalar loss
        """
        if len(pred_boxes_seq) < 2:
            return torch.tensor(0.0, device=pred_boxes_seq[0].device if len(pred_boxes_seq) > 0 else 'cpu')
        
        total_loss = 0.0
        num_pairs = 0
        
        for t in range(len(pred_boxes_seq) - 1):
            n_pred_t = len(pred_boxes_seq[t])
            n_pred_t1 = len(pred_boxes_seq[t+1])
            n_tgt_t = len(target_boxes_seq[t])
            n_tgt_t1 = len(target_boxes_seq[t+1])
            
            if n_pred_t == 0 or n_pred_t1 == 0 or n_tgt_t == 0 or n_tgt_t1 == 0:
                continue
            
            # Match objects (take minimum to avoid mismatch)
            n_match = min(n_pred_t, n_pred_t1, n_tgt_t, n_tgt_t1)
            
            # Predicted velocity
            pred_vel = pred_boxes_seq[t+1][:n_match, :2] - pred_boxes_seq[t][:n_match, :2]
            
            # Target velocity
            target_vel = target_boxes_seq[t+1][:n_match, :2] - target_boxes_seq[t][:n_match, :2]
            
            # L2 loss on velocity
            loss = nn.functional.mse_loss(pred_vel, target_vel)
            total_loss += loss
            num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0)
    
    def forward(self, predictions, targets, confidences=None):
        """
        Compute total loss.
        
        Args:
            predictions: List of [N, 4] predicted boxes per frame
            targets: List of [N, 4] target boxes per frame
            confidences: List of [N] confidence scores (optional)
            
        Returns:
            total_loss: Scalar total loss
            loss_dict: Dictionary of individual losses
        """
        # Box regression loss per frame
        box_losses = []
        for pred, tgt in zip(predictions, targets):
            if len(pred) > 0 and len(tgt) > 0:
                box_losses.append(self.box_loss(pred, tgt))
        
        box_loss = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0)
        
        # Velocity consistency loss
        vel_loss = self.velocity_loss(predictions, targets)
        
        # Confidence loss (if provided)
        if confidences is not None:
            # High confidence should correspond to accurate predictions
            conf_loss = 0.0
            num_conf_frames = 0
            for pred, tgt, conf in zip(predictions, targets, confidences):
                n_pred = len(pred)
                n_tgt = len(tgt)
                
                if n_pred > 0 and n_tgt > 0:
                    # Match objects (take minimum)
                    n_match = min(n_pred, n_tgt)
                    pred_matched = pred[:n_match]
                    tgt_matched = tgt[:n_match]
                    conf_matched = conf[:n_match]
                    
                    # IoU as proxy for accuracy
                    pred_xyxy = self.cxcywh_to_xyxy(pred_matched)
                    tgt_xyxy = self.cxcywh_to_xyxy(tgt_matched)
                    iou = self.box_iou(pred_xyxy, tgt_xyxy)
                    
                    # Confidence should match IoU
                    conf_loss += nn.functional.mse_loss(conf_matched, iou)
                    num_conf_frames += 1
            
            conf_loss = conf_loss / num_conf_frames if num_conf_frames > 0 else 0.0
        else:
            conf_loss = 0.0
        
        # Apply dynamic balancing if enabled
        if self.use_dynamic_balancing and self.balancer is not None:
            # Update balancer with current raw losses
            raw_loss_dict = {
                'box': box_loss,
                'velocity': vel_loss,
                'confidence': conf_loss if isinstance(conf_loss, torch.Tensor) else torch.tensor(conf_loss)
            }
            weights = self.balancer.update(raw_loss_dict)
            
            # Use dynamically adjusted weights
            box_w = weights['box']
            vel_w = weights['velocity']
            conf_w = weights['confidence']
        else:
            # Use fixed weights
            box_w = self.box_weight
            vel_w = self.velocity_weight
            conf_w = self.conf_weight
        
        # Total loss with weights
        total_loss = (
            box_w * box_loss + 
            vel_w * vel_loss +
            conf_w * conf_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'box': box_loss,
            'velocity': vel_loss,
            'confidence': conf_loss,
            # Also return weights for monitoring
            'box_weight': box_w if not isinstance(box_w, torch.Tensor) else box_w.item() if box_w.numel() == 1 else box_w,
            'velocity_weight': vel_w if not isinstance(vel_w, torch.Tensor) else vel_w.item() if vel_w.numel() == 1 else vel_w,
            'conf_weight': conf_w if not isinstance(conf_w, torch.Tensor) else conf_w.item() if conf_w.numel() == 1 else conf_w
        }
        
        return total_loss, loss_dict


def create_mv_center_memory_tracker(feature_dim=128, hidden_dim=256, 
                                     max_objects=100, grid_size=40, image_size=640,
                                     use_roi_align=False, roi_size=(7, 7)):
    """
    Factory function to create memory-based tracker.
    
    Args:
        feature_dim: Feature dimension for MV encoder
        hidden_dim: Hidden dimension for LSTM
        max_objects: Maximum number of trackable objects
        grid_size: Size of motion vector grid
        image_size: Image size in pixels
        use_roi_align: If True, use ROI Align for spatially-aligned motion features
        roi_size: Output size for ROI Align
        
    Returns:
        model: MVCenterMemoryTracker instance
    """
    model = MVCenterMemoryTracker(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        max_objects=max_objects,
        grid_size=grid_size,
        image_size=image_size,
        use_roi_align=use_roi_align,
        roi_size=roi_size
    )
    return model
