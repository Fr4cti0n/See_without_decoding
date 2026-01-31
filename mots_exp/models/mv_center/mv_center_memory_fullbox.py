"""
Enhanced Memory-Based Motion Vector Tracker with Full BBox Prediction

Improvements:
1. Predicts FULL bounding boxes [cx, cy, w, h] directly (not just center deltas)
2. Uses CNN backbone to process full motion vector field
3. Combines global MV context with local box features
"""

import torch
import torch.nn as nn
from .components.lstm_tracker_enhanced import EnhancedLSTMObjectTracker
from .components.mv_encoder import MotionVectorEncoder


class MVCenterMemoryTrackerFullBox(nn.Module):
    """
    Enhanced memory-based tracker with:
    - Full bbox prediction (can adapt to scale changes)
    - CNN backbone for global MV context
    - Combined local + global features
    
    Args:
        feature_dim: Dimension of motion features
        hidden_dim: Dimension of LSTM hidden state
        max_objects: Maximum number of objects
        grid_size: Size of motion vector grid
        image_size: Image size in pixels
        use_backbone: If True, use CNN backbone for MV processing
    """
    
    def __init__(self, feature_dim=64, hidden_dim=128, max_objects=100, 
                 grid_size=40, image_size=640, use_backbone=True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.grid_size = grid_size
        self.image_size = image_size
        self.use_backbone = use_backbone
        
        # Simple feature encoder (used as fallback if backbone disabled)
        self.feature_encoder = MotionVectorEncoder(
            input_channels=2,
            feature_dim=feature_dim,
            grid_size=grid_size
        )
        
        # Enhanced tracker with backbone and full bbox prediction
        self.tracker = EnhancedLSTMObjectTracker(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_backbone=use_backbone
        )
        
        # Store tracking state
        self.active_boxes = None
        self.active_ids = None
        self.lstm_states = None
        self.active_velocities = None
    
    def init_from_iframe(self, boxes, ids=None):
        """
        Initialize tracker from I-frame ground truth.
        
        Args:
            boxes: [N, 4] ground truth boxes
            ids: [N] object IDs (optional)
        """
        self.active_boxes = boxes.clone()
        
        if ids is None:
            self.active_ids = torch.arange(len(boxes), device=boxes.device)
        else:
            self.active_ids = ids.clone()
        
        # Initialize velocities as zero
        self.active_velocities = torch.zeros(len(boxes), 2, device=boxes.device)
        
        # Initialize LSTM states
        self.lstm_states = None
    
    def forward_gop(self, motion_sequences, iframe_boxes, iframe_ids=None):
        """
        Forward pass for a complete GOP sequence.
        
        Args:
            motion_sequences: [T, 2, H, W] motion vectors for T P-frames
            iframe_boxes: [N, 4] initial bounding boxes from I-frame
            iframe_ids: [N] object IDs (optional)
            
        Returns:
            predictions: List of [N, 4] bounding boxes per frame
            confidences: List of [N] confidences per frame
        """
        # Debug input
        print(f"\n🧠 DEBUG Neural Network Input (forward_gop):")
        print(f"   - motion_sequences.shape = {motion_sequences.shape}")
        print(f"   - motion_sequences dtype = {motion_sequences.dtype}")
        print(f"   - motion_sequences min/max = {motion_sequences.min():.4f} / {motion_sequences.max():.4f}")
        print(f"   - motion_sequences mean/std = {motion_sequences.mean():.4f} / {motion_sequences.std():.4f}")
        print(f"   - Unique values (frame 0): {torch.unique(motion_sequences[0]).numel()}")
        print(f"   - Sample motion values (frame 0, channel 0): {motion_sequences[0, 0].flatten()[:10].tolist()}")
        print(f"   - iframe_boxes.shape = {iframe_boxes.shape}")
        print(f"   - Number of initial objects = {len(iframe_boxes)}")
        
        # Initialize from I-frame
        self.init_from_iframe(iframe_boxes, iframe_ids)
        
        num_frames = len(motion_sequences)
        all_predictions = []
        all_confidences = []
        
        # Process each P-frame
        for frame_idx in range(num_frames):
            mv_field = motion_sequences[frame_idx]  # [2, H, W]
            
            # Encode motion features (only used if backbone disabled)
            mv_features = self.feature_encoder(mv_field.unsqueeze(0)).squeeze(0)  # [C, H, W]
            
            # Update positions using enhanced tracker
            # This will use backbone if enabled, otherwise use mv_features
            updated_boxes, updated_vels, confidences, new_states = self.tracker(
                positions=self.active_boxes,
                velocities=self.active_velocities,
                mv_field=mv_field,
                mv_features=mv_features,
                lstm_states=self.lstm_states,
                grid_size=self.grid_size,
                image_size=self.image_size
            )
            
            # Debug print
            if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                print(f"   📍 Processing P-frame: {len(self.active_boxes)} active objects")
                print(f"      - motion_vectors.shape = {mv_field.shape}")
                print(f"      - MV unique values = {torch.unique(mv_field).numel()}")
                print(f"      - MV min/max = {mv_field.min():.4f} / {mv_field.max():.4f}")
            
            # Update state
            self.active_boxes = updated_boxes
            self.active_velocities = updated_vels
            self.lstm_states = new_states
            
            # Store predictions
            all_predictions.append(updated_boxes)
            all_confidences.append(confidences)
        
        return all_predictions, all_confidences
    
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
            # Check if initialized either via wrapper or tracker
            if self.active_boxes is None:
                # Check if initialized via tracker.init_from_iframe()
                if hasattr(self.tracker, 'current_boxes') and self.tracker.current_boxes is not None:
                    # Sync state from tracker
                    self.active_boxes = self.tracker.current_boxes
                    self.active_ids = getattr(self.tracker, 'current_ids', None)
                    self.active_velocities = torch.zeros(len(self.active_boxes), 2, device=self.active_boxes.device)
                    self.lstm_states = None
                else:
                    raise ValueError("Must initialize tracker with init_from_iframe before single_frame mode")
            
            # Encode motion features
            mv_features = self.feature_encoder(motion_vectors.unsqueeze(0)).squeeze(0)  # [C, H, W]
            
            # Update positions
            updated_boxes, updated_vels, confidences, new_states = self.tracker(
                positions=self.active_boxes,
                velocities=self.active_velocities,
                mv_field=motion_vectors,
                mv_features=mv_features,
                lstm_states=self.lstm_states,
                grid_size=self.grid_size,
                image_size=self.image_size
            )
            
            # Update state
            self.active_boxes = updated_boxes
            self.active_velocities = updated_vels
            self.lstm_states = new_states
            
            return updated_boxes, confidences
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def reset(self):
        """Reset tracker for new GOP."""
        self.active_boxes = None
        self.active_ids = None
        self.lstm_states = None
        self.active_velocities = None
    
    def get_state(self):
        """Get current tracker state."""
        return {
            'boxes': self.active_boxes,
            'ids': self.active_ids,
            'lstm_states': self.lstm_states,
            'velocities': self.active_velocities
        }
    
    def get_model_info(self):
        """Get model configuration info for checkpointing."""
        return {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'max_objects': self.max_objects,
            'grid_size': self.grid_size,
            'image_size': self.image_size,
            'use_backbone': self.use_backbone
        }


class MVCenterMemoryLossFullBox(nn.Module):
    """
    Loss function for full bbox prediction model.
    
    Differences from center-only:
    - Computes loss on all 4 bbox components
    - Can weight w,h differently than cx,cy if needed
    """
    
    def __init__(self, box_weight=5.0, conf_weight=0.0, id_weight=0.5,
                 center_weight=1.0, size_weight=1.0):
        super().__init__()
        self.box_weight = box_weight
        self.conf_weight = conf_weight
        self.id_weight = id_weight
        self.center_weight = center_weight  # Weight for cx, cy loss
        self.size_weight = size_weight      # Weight for w, h loss
        
        print(f"📊 Loss weights:")
        print(f"   - Box: {box_weight} (center={center_weight}, size={size_weight})")
        print(f"   - Conf: {conf_weight}")
        print(f"   - ID: {id_weight}")
    
    def forward(self, predictions, targets, confidences=None):
        """
        Compute loss for full bbox predictions.
        
        Args:
            predictions: List of [N, 4] predicted boxes per frame
            targets: List of [N, 4] ground truth boxes per frame
            confidences: List of [N] confidence scores per frame (optional)
            
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual loss components
        """
        num_frames = len(predictions)
        
        box_losses = []
        conf_losses = []
        
        for frame_idx in range(num_frames):
            pred_boxes = predictions[frame_idx]
            gt_boxes = targets[frame_idx]
            
            # Skip if no predictions or targets
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # Handle case where number of predictions != number of targets
            n_match = min(len(pred_boxes), len(gt_boxes))
            pred_valid = pred_boxes[:n_match]
            gt_valid = gt_boxes[:n_match]
            
            # 🆕 Separate loss for center and size
            # Center loss (cx, cy)
            center_loss = torch.nn.functional.l1_loss(
                pred_valid[:, :2], gt_valid[:, :2], reduction='mean'
            )
            
            # Size loss (w, h)
            size_loss = torch.nn.functional.l1_loss(
                pred_valid[:, 2:], gt_valid[:, 2:], reduction='mean'
            )
            
            # Combined box loss with different weights
            box_loss = self.center_weight * center_loss + self.size_weight * size_loss
            box_losses.append(box_loss)
            
            # Confidence loss (always 1.0 for valid objects)
            if self.conf_weight > 0 and confidences is not None:
                conf = confidences[frame_idx]
                conf_valid = conf[:n_match]
                target_conf = torch.ones_like(conf_valid)
                conf_loss = torch.nn.functional.binary_cross_entropy(
                    conf_valid, target_conf, reduction='mean'
                )
                conf_losses.append(conf_loss)
        
        # Average losses across frames
        total_box_loss = torch.stack(box_losses).mean() if box_losses else torch.tensor(0.0).to(predictions[0].device)
        total_conf_loss = torch.stack(conf_losses).mean() if conf_losses else torch.tensor(0.0).to(predictions[0].device)
        
        # Weighted total loss
        total_loss = (
            self.box_weight * total_box_loss +
            self.conf_weight * total_conf_loss
        )
        
        # Return loss dict for logging
        loss_dict = {
            'box': total_box_loss.item(),
            'confidence': total_conf_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict
