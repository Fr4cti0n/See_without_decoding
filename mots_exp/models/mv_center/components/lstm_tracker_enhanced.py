"""
Enhanced LSTM-Based Object Tracker with Backbone

Improvements over original:
1. Predicts FULL bounding boxes [cx, cy, w, h] directly (not just deltas)
2. Adds CNN backbone to process full motion vector field
3. Combines global MV context with local box features

This allows the model to:
- Adapt to scale changes (predict w, h)
- Use full MV field context (not just local sampling)
- Make more informed predictions with global scene understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mv_sampling import batch_sample_features_at_bboxes


class MotionVectorBackbone(nn.Module):
    """
    CNN backbone to process full motion vector field.
    
    Extracts hierarchical features from MV field to provide
    global context about scene motion and camera movement.
    
    Architecture:
    - Input: [2, H, W] motion vector field
    - Output: [feature_dim, H', W'] processed features
    """
    
    def __init__(self, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Hierarchical feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, mv_field):
        """
        Process motion vector field.
        
        Args:
            mv_field: [B, 2, H, W] motion vectors
            
        Returns:
            features: [B, feature_dim, H, W] processed features
        """
        # Add batch dimension if needed
        if mv_field.dim() == 3:
            mv_field = mv_field.unsqueeze(0)  # [1, 2, H, W]
        
        x = self.conv1(mv_field)  # [B, 32, H, W]
        x = self.conv2(x)         # [B, 64, H, W]
        x = self.conv3(x)         # [B, feature_dim, H, W]
        
        return x


class EnhancedLSTMObjectTracker(nn.Module):
    """
    Enhanced LSTM-based tracker with:
    1. Full bbox prediction [cx, cy, w, h]
    2. CNN backbone for global MV context
    3. Combined local + global features
    
    Args:
        feature_dim: Dimension of motion vector features
        hidden_dim: Dimension of LSTM hidden state
        use_backbone: If True, use CNN backbone for MV processing
    """
    
    def __init__(self, feature_dim=64, hidden_dim=128, use_backbone=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.use_backbone = use_backbone
        
        # Motion vector backbone (optional)
        if use_backbone:
            print(f"✨ Using CNN backbone for motion vector processing (feature_dim={feature_dim})")
            self.mv_backbone = MotionVectorBackbone(feature_dim=feature_dim)
        else:
            print(f"⚠️  No backbone - using simple feature encoder")
            self.mv_backbone = None
        
        # 🔧 FIX: Input ONLY motion features (NO position!)
        # Input: [ROI stats (6), MV features (feature_dim)]
        # ROI stats: mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio
        input_dim = 6 + feature_dim  # ← Changed from 4 + 6 + feature_dim
        
        # LSTM cell for temporal tracking
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        
        # 🆕 FULL BBOX PREDICTOR - predicts absolute [cx, cy, w, h]
        # This allows model to adapt to scale changes
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)  # [cx, cy, w, h] - FULL prediction!
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
    
    def init_from_iframe(self, boxes, ids=None):
        """
        Initialize tracker from I-frame.
        
        This method is called by the wrapper to store initial boxes and IDs.
        The actual initialization is handled by the wrapper, but we need
        this method for API compatibility with the training script.
        
        Args:
            boxes: [N, 4] initial bounding boxes
            ids: [N] object IDs (optional)
        """
        # Store for reference (wrapper handles actual state)
        self.current_boxes = boxes
        self.current_ids = ids
    
    def reset(self):
        """
        Reset tracker state for new GOP.
        
        API compatibility method - actual state managed by wrapper.
        """
        self.current_boxes = None
        self.current_ids = None
    
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
            velocities: [N, 2] current velocities [vx, vy] - IGNORED
            mv_field: [2, H, W] motion vector field
            mv_features: [C, H, W] encoded motion features (ignored if using backbone)
            lstm_states: Tuple of (h, c) LSTM states, or None to initialize
            grid_size: Size of MV grid (default: 40)
            image_size: Image size in pixels (default: 640)
            
        Returns:
            updated_positions: [N, 4] predicted FULL bounding boxes
            updated_velocities: [N, 2] dummy velocities (zeros)
            confidences: [N] prediction confidences
            new_lstm_states: Tuple of (h, c) updated LSTM states
        """
        num_objects = len(positions)
        device = positions.device
        
        # Initialize LSTM states if needed
        if lstm_states is None:
            h, c = self.init_hidden(num_objects, device)
        else:
            h, c = lstm_states
        
        # 🆕 Process MV field with backbone if enabled
        if self.use_backbone:
            # Use CNN backbone to extract rich features from full MV field
            processed_features = self.mv_backbone(mv_field)  # [1, feature_dim, H, W]
            processed_features = processed_features.squeeze(0)  # [feature_dim, H, W]
        else:
            # Use provided pre-encoded features
            processed_features = mv_features
        
        # Extract local motion statistics inside each bounding box
        roi_motion_stats = self._extract_roi_motion_features(
            mv_field, positions, grid_size, image_size
        )  # [N, 6] - [mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio]
        
        # Sample global features at object locations
        feature_samples = batch_sample_features_at_bboxes(
            processed_features, positions, grid_size, image_size
        )  # [N, feature_dim]
        
        # 🔧 FIX: DO NOT include current position in LSTM input!
        # Force model to predict from motion features alone
        # Otherwise it learns identity mapping: output ≈ input
        lstm_input = torch.cat([
            # positions,         # ❌ REMOVED - was causing identity learning!
            roi_motion_stats,    # [N, 6] - local motion stats
            feature_samples      # [N, feature_dim] - global context from backbone
        ], dim=1)  # [N, 6 + feature_dim]
        
        # LSTM update
        h_new, c_new = self.lstm(lstm_input, (h, c))
        
        # 🆕 Predict FULL bounding box [cx, cy, w, h]
        # Model can now adapt to scale changes!
        predicted_bbox = self.bbox_head(h_new)  # [N, 4]
        
        # Apply sigmoid to ensure normalized coordinates [0, 1]
        predicted_bbox = torch.sigmoid(predicted_bbox)
        
        # Dummy velocities for API compatibility
        new_velocities = torch.zeros(num_objects, 2, device=device)
        
        # Predict confidence
        confidences = self.confidence_head(h_new).squeeze(1)  # [N]
        
        # Return predicted bounding boxes directly
        updated_positions = predicted_bbox
        
        return updated_positions, new_velocities, confidences, (h_new, c_new)
    
    def _extract_roi_motion_features(self, mv_field, boxes, grid_size, image_size):
        """
        Extract motion features by computing mean motion vectors inside each box.
        Includes ALL motion vectors (zeros + non-zeros) for accurate representation.
        
        Args:
            mv_field: [2, H, W] motion vector field
            boxes: [N, 4] bounding boxes in normalized coords [cx, cy, w, h]
            grid_size: MV grid size
            image_size: Image size
            
        Returns:
            roi_features: [N, 6] motion statistics
                - mean_vx, mean_vy: Mean motion (includes zeros!)
                - std_vx, std_vy: Standard deviation
                - num_mvs: Number of MVs in box
                - sparsity_ratio: Ratio of zero MVs
        """
        device = mv_field.device
        num_boxes = len(boxes)
        
        roi_features = []
        
        for box in boxes:
            cx, cy, w, h = box
            
            # Convert normalized coords to grid indices
            x_min = int(max(0, (cx - w/2) * grid_size))
            x_max = int(min(grid_size, (cx + w/2) * grid_size))
            y_min = int(max(0, (cy - h/2) * grid_size))
            y_max = int(min(grid_size, (cy + h/2) * grid_size))
            
            # Extract MVs inside box
            box_mvs_x = mv_field[0, y_min:y_max, x_min:x_max]
            box_mvs_y = mv_field[1, y_min:y_max, x_min:x_max]
            
            # Compute statistics
            total_cells = box_mvs_x.numel()
            
            if total_cells > 0:
                # ✅ FIXED: Compute mean/std of ALL MVs (including zeros)
                # This correctly represents stationary objects (zero motion)
                mean_vx = box_mvs_x.mean()
                mean_vy = box_mvs_y.mean()
                std_vx = box_mvs_x.std() if total_cells > 1 else torch.tensor(0.0, device=device)
                std_vy = box_mvs_y.std() if total_cells > 1 else torch.tensor(0.0, device=device)
                
                # Count non-zero MVs for sparsity
                non_zero_mask_x = box_mvs_x != 0
                non_zero_mask_y = box_mvs_y != 0
                non_zero_mask = non_zero_mask_x | non_zero_mask_y
                num_mvs = non_zero_mask.sum().float()
                sparsity_ratio = 1.0 - (num_mvs / total_cells)
            else:
                mean_vx = torch.tensor(0.0, device=device)
                mean_vy = torch.tensor(0.0, device=device)
                std_vx = torch.tensor(0.0, device=device)
                std_vy = torch.tensor(0.0, device=device)
                num_mvs = torch.tensor(0.0, device=device)
                sparsity_ratio = torch.tensor(1.0, device=device)
            
            # Stack features: [mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio]
            roi_features.append(torch.stack([
                mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio
            ]))
        
        return torch.stack(roi_features)  # [N, 6]
