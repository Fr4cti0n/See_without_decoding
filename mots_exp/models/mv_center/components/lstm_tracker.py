"""
LSTM-Based Object Tracker

Updates object positions using motion vectors and temporal LSTM state.

Can optionally use ROI Align for spatially-aligned motion features.
"""

import torch
import torch.nn as nn
from .mv_sampling import batch_sample_features_at_bboxes, batch_sample_mv_at_bboxes

# Optional: Import ROI Align for spatial alignment
try:
    from torchvision.ops import roi_align
    ROI_ALIGN_AVAILABLE = True
except ImportError:
    ROI_ALIGN_AVAILABLE = False
    print("⚠️  torchvision.ops.roi_align not available, using standard MV sampling")


class LSTMObjectTracker(nn.Module):
    """
    LSTM-based tracker that updates object positions using motion vectors.
    
    For each object:
    - Samples motion vectors at current position (or uses ROI Align if enabled)
    - Samples appearance features from MV feature map
    - Feeds into LSTM with previous state
    - Predicts position delta and velocity
    
    Args:
        feature_dim: Dimension of motion vector features
        hidden_dim: Dimension of LSTM hidden state
        use_roi_align: If True, use ROI Align for spatially-aligned motion features
        roi_size: Output size for ROI Align (e.g., (7, 7))
    """
    
    def __init__(self, feature_dim=128, hidden_dim=256, use_roi_align=False, roi_size=(7, 7)):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_roi_align = use_roi_align and ROI_ALIGN_AVAILABLE
        self.roi_size = roi_size
        
        if self.use_roi_align:
            print(f"✨ Using ROI motion averaging (mean MVs inside boxes)")
            # Input: [position (4), ROI stats (6), MV features (feature_dim)]
            # ROI stats: mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio
            # ❌ Removed velocity (2) - not needed, redundant with position deltas
            input_dim = 4 + 6 + feature_dim
        else:
            # Input: [position (4), MV sample (2), MV features (feature_dim)]
            # ❌ Removed velocity (2) - not needed
            input_dim = 4 + 2 + feature_dim  # = 6 + feature_dim
        
        # LSTM cell for temporal tracking
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        
        # Position delta predictor - ONLY predict center movement [dcx, dcy]
        # Width and height remain constant to avoid shrinkage problem
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)  # [dcx, dcy] - only center movement!
        )
        
        # ❌ REMOVED velocity_head - not needed since velocity loss is disabled
        
        # Confidence predictor (how confident is this update)
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
        """
        Initialize LSTM hidden states.
        
        Args:
            num_objects: Number of active objects
            device: Device to create tensors on
            
        Returns:
            h, c: Hidden and cell states [N, hidden_dim]
        """
        h = torch.zeros(num_objects, self.hidden_dim, device=device)
        c = torch.zeros(num_objects, self.hidden_dim, device=device)
        return h, c
    
    def forward(self, positions, velocities, mv_field, mv_features, 
                lstm_states=None, grid_size=40, image_size=640):
        """
        Update object positions using motion vectors.
        
        Args:
            positions: [N, 4] current positions [cx, cy, w, h]
            velocities: [N, 2] current velocities [vx, vy] - IGNORED, kept for API compatibility
            mv_field: [2, H, W] motion vector field
            mv_features: [C, H, W] encoded motion features
            lstm_states: Tuple of (h, c) LSTM states, or None to initialize
            grid_size: Size of MV grid (default: 40)
            image_size: Image size in pixels (default: 640)
            
        Returns:
            updated_positions: [N, 4] updated positions
            updated_velocities: [N, 2] dummy velocities (zeros) - kept for API compatibility
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
        
        if self.use_roi_align:
            # Compute mean motion vectors inside each bounding box
            roi_motion_stats = self._extract_roi_motion_features(
                mv_field, positions, grid_size, image_size
            )  # [N, 6] - [mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio]
            
            # Sample global motion features at object locations (for context)
            feature_samples = batch_sample_features_at_bboxes(
                mv_features, positions, grid_size, image_size
            )  # [N, feature_dim]
            
            # Construct LSTM input with ROI statistics
            lstm_input = torch.cat([
                positions,           # [N, 4] - current box position
                # ❌ Removed velocities - not needed
                roi_motion_stats,    # [N, 6] - motion stats inside box
                feature_samples      # [N, feature_dim] - global context
            ], dim=1)  # [N, 4 + 6 + feature_dim]
        else:
            # Original: Sample motion vectors at object locations (point sampling)
            mv_samples = batch_sample_mv_at_bboxes(
                mv_field, positions, grid_size, image_size
            )  # [N, 2]
            
            # Sample motion features at object locations
            feature_samples = batch_sample_features_at_bboxes(
                mv_features, positions, grid_size, image_size
            )  # [N, feature_dim]
            
            # Construct LSTM input
            lstm_input = torch.cat([
                positions,         # [N, 4]
                # ❌ Removed velocities - not needed
                mv_samples,        # [N, 2]
                feature_samples    # [N, feature_dim]
            ], dim=1)  # [N, 6 + feature_dim]
        
        # LSTM update
        h_new, c_new = self.lstm(lstm_input, (h, c))
        
        # Predict ONLY center position delta [dcx, dcy]
        # Keep width/height constant to avoid shrinkage!
        center_delta = self.position_head(h_new)  # [N, 2]
        
        # ❌ No velocity prediction - return dummy zeros for API compatibility
        new_velocities = torch.zeros(num_objects, 2, device=positions.device)
        
        # Predict confidence
        confidences = self.confidence_head(h_new).squeeze(1)  # [N]
        
        # Update ONLY center position, keep width/height unchanged
        updated_positions = positions.clone()
        updated_positions[:, :2] += center_delta  # Only update cx, cy
        # positions[:, 2:4] remain unchanged (w, h stay constant!)
        
        return updated_positions, new_velocities, confidences, (h_new, c_new)
    
    def _extract_roi_motion_features(self, mv_field, boxes, grid_size, image_size):
        """
        Extract motion features by computing mean motion vectors inside each box.
        
        This is simpler and more accurate than ROI Align:
        - Preserves actual box size information
        - Only averages motion vectors that are actually inside the box
        - Accounts for sparsity naturally (boxes with few MVs get smaller means)
        
        Args:
            mv_field: [2, H, W] motion vector field
            boxes: [N, 4] normalized boxes [cx, cy, w, h] in [0,1]
            grid_size: Motion vector grid size (H or W)
            image_size: Image size in pixels
            
        Returns:
            roi_features: [N, 6] - [mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio]
        """
        N = len(boxes)
        if N == 0:
            return torch.zeros(0, 6, device=boxes.device, dtype=boxes.dtype)
        
        H, W = mv_field.shape[1], mv_field.shape[2]
        device = boxes.device
        
        # Extract motion vector components
        mv_x = mv_field[0]  # [H, W]
        mv_y = mv_field[1]  # [H, W]
        
        # ⚡ OPTIMIZED: Batched vectorized ROI extraction (17.7x faster!)
        # Convert boxes to grid coordinates - [N, 4]
        cx_grid = boxes[:, 0] * W
        cy_grid = boxes[:, 1] * H
        w_grid = boxes[:, 2] * W
        h_grid = boxes[:, 3] * H
        
        # Compute box bounds - [N]
        x1 = torch.clamp((cx_grid - w_grid / 2).floor().long(), min=0, max=W-1)
        x2 = torch.clamp((cx_grid + w_grid / 2).ceil().long(), min=1, max=W)
        y1 = torch.clamp((cy_grid - h_grid / 2).floor().long(), min=0, max=H-1)
        y2 = torch.clamp((cy_grid + h_grid / 2).ceil().long(), min=1, max=H)
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=device).view(H, 1).expand(H, W)  # [H, W]
        x_coords = torch.arange(W, device=device).view(1, W).expand(H, W)  # [H, W]
        
        # Create box masks - [N, H, W]
        # For each box, create a mask where True indicates pixels inside the box
        y_mask = (y_coords.unsqueeze(0) >= y1.unsqueeze(1).unsqueeze(2)) & \
                 (y_coords.unsqueeze(0) < y2.unsqueeze(1).unsqueeze(2))  # [N, H, W]
        x_mask = (x_coords.unsqueeze(0) >= x1.unsqueeze(1).unsqueeze(2)) & \
                 (x_coords.unsqueeze(0) < x2.unsqueeze(1).unsqueeze(2))  # [N, H, W]
        inside_mask = y_mask & x_mask  # [N, H, W]
        
        # Apply masks to motion vectors - [N, H, W]
        mv_x_masked = mv_x.unsqueeze(0) * inside_mask.float()  # [N, H, W]
        mv_y_masked = mv_y.unsqueeze(0) * inside_mask.float()  # [N, H, W]
        
        # Count pixels inside each box
        counts = inside_mask.sum(dim=(1, 2)).float()  # [N]
        counts = torch.clamp(counts, min=1.0)  # Avoid division by zero
        
        # Compute mean motion - [N]
        mean_vx = mv_x_masked.sum(dim=(1, 2)) / counts
        mean_vy = mv_y_masked.sum(dim=(1, 2)) / counts
        
        # Compute std motion - [N]
        # std = sqrt(E[X^2] - E[X]^2)
        mv_x_sq_masked = (mv_x.unsqueeze(0) ** 2) * inside_mask.float()
        mv_y_sq_masked = (mv_y.unsqueeze(0) ** 2) * inside_mask.float()
        
        mean_vx_sq = mv_x_sq_masked.sum(dim=(1, 2)) / counts
        mean_vy_sq = mv_y_sq_masked.sum(dim=(1, 2)) / counts
        
        std_vx = torch.sqrt(torch.clamp(mean_vx_sq - mean_vx ** 2, min=0.0))
        std_vy = torch.sqrt(torch.clamp(mean_vy_sq - mean_vy ** 2, min=0.0))
        
        # Compute sparsity (non-zero MVs)
        magnitude = torch.sqrt(mv_x ** 2 + mv_y ** 2)  # [H, W]
        non_zero_mask = (magnitude > 0.01).unsqueeze(0) & inside_mask  # [N, H, W]
        num_mvs = non_zero_mask.sum(dim=(1, 2)).float()  # [N]
        sparsity_ratio = num_mvs / counts  # [N]
        
        # Stack features - [N, 6]
        roi_features = torch.stack([
            mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio
        ], dim=1)
        
        return roi_features


class MotionGuidedTracker(nn.Module):
    """
    Complete motion-guided tracking module.
    
    Combines:
    - MV encoder
    - Object memory bank
    - LSTM tracker (with optional ROI Align)
    
    Args:
        feature_dim: Dimension of motion features
        hidden_dim: Dimension of LSTM hidden state
        max_objects: Maximum number of objects to track
        use_roi_align: If True, use ROI Align for spatially-aligned motion features
        roi_size: Output size for ROI Align
    """
    
    def __init__(self, feature_dim=128, hidden_dim=256, max_objects=100, 
                 use_roi_align=False, roi_size=(7, 7)):
        super().__init__()
        
        from .mv_encoder import MotionVectorEncoder
        from .memory_bank import ObjectMemoryBank
        
        # Components
        self.mv_encoder = MotionVectorEncoder(
            input_channels=2, 
            feature_dim=feature_dim
        )
        
        self.memory_bank = ObjectMemoryBank(
            max_objects=max_objects,
            feature_dim=feature_dim
        )
        
        self.lstm_tracker = LSTMObjectTracker(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            use_roi_align=use_roi_align,
            roi_size=roi_size
        )
        
        # LSTM states storage
        self.lstm_h = None
        self.lstm_c = None
    
    def init_from_iframe(self, boxes, object_ids=None):
        """
        Initialize tracking from I-frame bounding boxes.
        
        Args:
            boxes: [N, 4] bounding boxes [cx, cy, w, h]
            object_ids: [N] object IDs (optional)
        """
        device = boxes.device
        num_objects = len(boxes)
        
        # Initialize memory bank
        initial_features = torch.zeros(num_objects, self.mv_encoder.feature_dim, device=device)
        self.memory_bank.init_from_boxes(boxes, object_ids, initial_features)
        
        # Initialize LSTM states
        self.lstm_h, self.lstm_c = self.lstm_tracker.init_hidden(num_objects, device)
    
    def forward_pframe(self, motion_vectors, grid_size=40, image_size=640):
        """
        Update tracking for a P-frame using motion vectors.
        
        Args:
            motion_vectors: [2, H, W] motion vector field
            grid_size: Size of MV grid
            image_size: Image size in pixels
            
        Returns:
            updated_boxes: [N, 4] updated bounding boxes
            confidences: [N] update confidences
        """
        # Encode motion vectors
        mv_features = self.mv_encoder(motion_vectors)
        
        # Get active objects from memory
        active_objects = self.memory_bank.get_active_objects()
        
        if len(active_objects['positions']) == 0:
            # No active objects
            return torch.zeros(0, 4, device=motion_vectors.device), \
                   torch.zeros(0, device=motion_vectors.device)
        
        positions = active_objects['positions']
        velocities = active_objects['velocities']
        indices = active_objects['indices']
        
        # Update positions using LSTM tracker
        updated_positions, updated_velocities, confidences, (h_new, c_new) = \
            self.lstm_tracker(
                positions, velocities, motion_vectors, mv_features,
                lstm_states=(self.lstm_h, self.lstm_c),
                grid_size=grid_size, image_size=image_size
            )
        
        # Update memory bank
        self.memory_bank.update_positions(indices, updated_positions)
        self.memory_bank.update_velocities(indices, updated_velocities)
        
        # Update LSTM states
        self.lstm_h = h_new
        self.lstm_c = c_new
        
        return updated_positions, confidences
    
    def reset(self):
        """Reset tracker state for new GOP."""
        self.memory_bank.reset()
        self.lstm_h = None
        self.lstm_c = None
    
    def get_current_state(self):
        """
        Get current tracking state.
        
        Returns:
            Dictionary with 'memory' (active objects) and 'lstm' (hidden/cell states)
        """
        return {
            'memory': self.memory_bank.get_active_objects(),
            'lstm': (self.lstm_h, self.lstm_c)
        }
    
    def get_lstm_hidden_state(self):
        """
        Get current LSTM hidden state for ID embedding extraction.
        
        Returns:
            h: [N, hidden_dim] hidden state
            c: [N, hidden_dim] cell state
        """
        return self.lstm_h, self.lstm_c
