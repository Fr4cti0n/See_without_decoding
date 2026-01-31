"""
Motion-Aligned Memory Tracker with ROI Align

Enhanced version of MVCenterMemoryTracker that spatially aligns
motion vectors with bounding boxes using ROI Align.

Key improvement: Instead of sampling motion vectors globally, this model
extracts per-box motion features that are spatially aligned with each object.
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_roi_align import MotionAlignedBoxEncoder
from mv_center.components.memory_bank import ObjectMemoryBank
from mv_center.components.mv_encoder import MotionVectorEncoder
from mv_center.mv_center_memory import MVCenterMemoryLoss


class MotionAlignedLSTMTracker(nn.Module):
    """
    LSTM tracker with motion-aligned features from ROI Align.
    
    Instead of sampling global motion features, this directly uses
    per-box motion-aligned features from ROI Align.
    """
    
    def __init__(self, motion_feature_dim=64, hidden_dim=256, roi_size=(7, 7)):
        super().__init__()
        self.motion_feature_dim = motion_feature_dim
        self.hidden_dim = hidden_dim
        self.roi_size = roi_size
        
        # Motion ROI Align encoder
        self.motion_encoder = MotionAlignedBoxEncoder(
            box_dim=4,
            motion_feature_dim=motion_feature_dim,
            hidden_dim=128,
            roi_size=roi_size
        )
        
        # Input: [encoded_features (128), velocity (2)]
        input_dim = 128 + 2  # = 130
        
        # LSTM cell for temporal tracking
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        
        # Position delta predictor
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # [dx, dy, dw, dh]
        )
        
        # Velocity predictor
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # [vx, vy]
        )
        
        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def init_hidden(self, num_objects, device):
        """Initialize LSTM hidden states."""
        h = torch.zeros(num_objects, self.hidden_dim, device=device)
        c = torch.zeros(num_objects, self.hidden_dim, device=device)
        return h, c
    
    def forward(self, positions, velocities, motion_vectors, lstm_states=None):
        """
        Update object positions using motion-aligned features.
        
        Args:
            positions: [N, 4] current positions [cx, cy, w, h] normalized
            velocities: [N, 2] current velocities [vx, vy]
            motion_vectors: [H, W, 2] motion vector field
            lstm_states: Tuple of (h, c) LSTM states, or None to initialize
            
        Returns:
            updated_positions: [N, 4] updated positions
            updated_velocities: [N, 2] updated velocities
            confidences: [N] update confidences
            new_lstm_states: Tuple of (h, c) updated LSTM states
        """
        num_objects = len(positions)
        device = positions.device
        
        # Initialize LSTM states if needed
        if lstm_states is None:
            h, c = self.init_hidden(num_objects, device)
        else:
            h, c = lstm_states
        
        # Extract motion-aligned features for each box
        # This is the KEY difference: per-box ROI Align on motion vectors
        motion_aligned_features = self.motion_encoder(positions, motion_vectors)  # [N, 128]
        
        # Construct LSTM input
        lstm_input = torch.cat([
            motion_aligned_features,  # [N, 128]
            velocities,               # [N, 2]
        ], dim=1)  # [N, 130]
        
        # LSTM update
        h_new, c_new = self.lstm(lstm_input, (h, c))
        
        # Predict position delta
        position_delta = self.position_head(h_new)  # [N, 4]
        
        # Predict new velocity
        new_velocities = self.velocity_head(h_new)  # [N, 2]
        
        # Predict confidence
        confidences = self.confidence_head(h_new).squeeze(1)  # [N]
        
        # Update positions
        updated_positions = positions + position_delta
        
        # Ensure valid width/height (normalized coordinates)
        updated_positions = torch.cat([
            updated_positions[:, :2],  # cx, cy
            torch.clamp(updated_positions[:, 2:4], min=0.01, max=1.0)  # w, h
        ], dim=1)
        
        return updated_positions, new_velocities, confidences, (h_new, c_new)


class MotionAlignedTracker(nn.Module):
    """
    Complete motion-aligned tracking module with ROI Align.
    
    Combines:
    - Motion ROI Align (spatially aligned per-box features)
    - Object memory bank
    - LSTM tracker with aligned features
    """
    
    def __init__(self, motion_feature_dim=64, hidden_dim=256, max_objects=100, roi_size=(7, 7)):
        super().__init__()
        
        self.motion_feature_dim = motion_feature_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.roi_size = roi_size
        
        # Memory bank for object states
        self.memory_bank = ObjectMemoryBank(
            max_objects=max_objects,
            feature_dim=hidden_dim
        )
        
        # LSTM tracker with ROI Align
        self.lstm_tracker = MotionAlignedLSTMTracker(
            motion_feature_dim=motion_feature_dim,
            hidden_dim=hidden_dim,
            roi_size=roi_size
        )
    
    def init_from_iframe(self, boxes, object_ids=None):
        """
        Initialize tracker from I-frame boxes.
        
        Args:
            boxes: [N, 4] ground truth boxes from I-frame
            object_ids: [N] object IDs (optional)
        """
        self.memory_bank.initialize(boxes, object_ids)
    
    def forward_pframe(self, motion_vectors, grid_size=40, image_size=640):
        """
        Forward pass for a single P-frame.
        
        Args:
            motion_vectors: [2, H, W] or [H, W, 2] motion vectors
            grid_size: Size of motion grid (unused, kept for API compatibility)
            image_size: Image size (unused, kept for API compatibility)
            
        Returns:
            boxes: [N, 4] predicted boxes
            confidences: [N] confidences
        """
        # Get current state from memory bank
        positions, velocities, lstm_states = self.memory_bank.get_current_state()
        
        if len(positions) == 0:
            # No objects to track
            return torch.zeros(0, 4, device=motion_vectors.device), \
                   torch.zeros(0, device=motion_vectors.device)
        
        # Ensure motion vectors are in [H, W, 2] format
        if motion_vectors.shape[0] == 2:  # [2, H, W]
            motion_vectors = motion_vectors.permute(1, 2, 0)  # → [H, W, 2]
        
        # Update positions using motion-aligned LSTM
        updated_positions, updated_velocities, confidences, new_lstm_states = \
            self.lstm_tracker(positions, velocities, motion_vectors, lstm_states)
        
        # Update memory bank
        self.memory_bank.update(updated_positions, updated_velocities, new_lstm_states)
        
        return updated_positions, confidences
    
    def reset(self):
        """Reset tracker for new GOP."""
        self.memory_bank.reset()
    
    def get_current_state(self):
        """Get current tracker state."""
        return self.memory_bank.get_current_state()


class MVCenterMotionAlignedTracker(nn.Module):
    """
    Memory-based tracker with motion-aligned features (ROI Align).
    
    This is the main model that replaces MVCenterMemoryTracker.
    
    Key difference: Uses ROI Align to extract per-box motion features
    that are spatially aligned with each object, rather than global
    motion feature sampling.
    """
    
    def __init__(self, motion_feature_dim=64, hidden_dim=256, max_objects=100,
                 grid_size=40, image_size=640, roi_size=(7, 7)):
        super().__init__()
        
        self.motion_feature_dim = motion_feature_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.grid_size = grid_size
        self.image_size = image_size
        self.roi_size = roi_size
        
        # Core tracker with ROI Align
        self.tracker = MotionAlignedTracker(
            motion_feature_dim=motion_feature_dim,
            hidden_dim=hidden_dim,
            max_objects=max_objects,
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
            
            # Update tracking with ROI Align
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
            'motion_feature_dim': self.motion_feature_dim,
            'hidden_dim': self.hidden_dim,
            'max_objects': self.max_objects,
            'grid_size': self.grid_size,
            'image_size': self.image_size,
            'roi_size': self.roi_size
        }


def test_motion_aligned_tracker():
    """Test the motion-aligned tracker."""
    print("🧪 Testing Motion-Aligned Tracker\n")
    
    # Parameters
    batch_size = 1
    num_frames = 5
    num_objects = 3
    H, W = 40, 40
    
    # Create dummy data
    motion_sequences = torch.randn(num_frames, 2, H, W)
    iframe_boxes = torch.tensor([
        [0.3, 0.4, 0.2, 0.3],
        [0.7, 0.5, 0.25, 0.35],
        [0.5, 0.7, 0.15, 0.2],
    ])
    
    print(f"Motion sequences shape: {motion_sequences.shape}")
    print(f"I-frame boxes shape: {iframe_boxes.shape}")
    print(f"I-frame boxes:\n{iframe_boxes}\n")
    
    # Create tracker
    tracker = MVCenterMotionAlignedTracker(
        motion_feature_dim=64,
        hidden_dim=256,
        max_objects=100,
        roi_size=(7, 7)
    )
    
    print("Model created successfully!")
    print(f"Parameters: {sum(p.numel() for p in tracker.parameters()):,}\n")
    
    # Forward pass
    predictions, confidences = tracker.forward_gop(motion_sequences, iframe_boxes)
    
    print(f"Predictions:")
    for t, (boxes, conf) in enumerate(zip(predictions, confidences)):
        print(f"  Frame {t}: {boxes.shape[0]} objects, conf range [{conf.min():.3f}, {conf.max():.3f}]")
    
    print(f"\n✅ Motion-Aligned Tracker test passed!")
    
    # Compare parameter counts
    print(f"\n📊 Architecture Comparison:")
    print(f"   Motion-Aligned Model: {sum(p.numel() for p in tracker.parameters()):,} parameters")
    print(f"   Key improvement: Per-box ROI Align on motion vectors")
    print(f"   ROI size: {tracker.roi_size}")
    print(f"   Motion feature dim: {tracker.motion_feature_dim}")


if __name__ == "__main__":
    test_motion_aligned_tracker()
