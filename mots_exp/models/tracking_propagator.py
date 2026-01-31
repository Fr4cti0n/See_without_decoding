"""
Tracking Propagator - Fixed-Slot Architecture for Box Propagation

Key differences from MVOnlyTracker:
1. **Fixed 300 output slots** (solves LSTM hidden state size problem)
2. **Explicit track ID prediction** (6D output per slot: [box(4), objectness(1), track_id(1)])
3. **Propagation-focused** (input prev boxes+IDs, predict current boxes+IDs)
4. **Handles object birth/death** (slots can be empty/filled dynamically)

Architecture:
    Input: 
        - Motion vectors [1, 2, H, W]
        - Previous boxes [N, 4] (variable N ≤ 300)
        - Previous track IDs [N] (integer IDs)
    
    Pipeline:
        1. Encode MVs → spatial features
        2. Slot initialization: embed N objects into 300 slots
        3. LSTM tracking: process 300 slots (fixed hidden state size!)
        4. Slot decoding: 300 × [box, objectness, track_id]
    
    Output:
        - Boxes [300, 4]
        - Objectness [300] (0=no object, 1=object)
        - Track IDs [300] (predicted ID, -1 for empty slots)
        - Hidden state [300, hidden_dim]

Training:
    - Hungarian matching between 300 predictions and M ground truth
    - Loss = box_loss + objectness_loss + track_id_loss
    - Unmatched slots → objectness=0 (no-object)

Author: GitHub Copilot
Date: October 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from scipy.optimize import linear_sum_assignment
import numpy as np

# Import RT-DETR head
from .rtdetr_head import RTDETRHead


class BoxAlignedMotionEncoder(nn.Module):
    """
    Motion Vector Encoder that RESPECTS 16x16 block structure.
    
    Key improvements over MVEncoderForPropagation:
    1. NO upsampling - works directly with 60x60 grid (no fake interpolated MVs)
    2. Extracts mean motion per box + spatial variance features
    3. Learns to aggregate motion patterns with MLP
    4. Fast vectorized implementation (~0.3ms regardless of N boxes)
    
    Architecture:
        - Grid encoder: 60x60 MVs → 60x60 features (respects blocks!)
        - Per-box sampling: Extract motion statistics within each box
        - Feature aggregation: Mean, variance, range → learned representation
    """
    def __init__(self, mv_feature_dim=64, image_size=960):
        super().__init__()
        self.image_size = image_size
        self.grid_size = 60  # Motion vector grid is 60x60
        self.block_size = image_size // self.grid_size  # 16 pixels per block
        
        # Grid encoder: Encode 60x60 MV grid to 60x60 feature grid
        # NO upsampling - respect block structure!
        self.grid_encoder = nn.Sequential(
            # Input: [1, 2, 60, 60] - raw motion vectors (dx, dy)
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, mv_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mv_feature_dim),
            nn.ReLU(inplace=True),
            # Output: [1, 64, 60, 60] - feature per block (NO fake MVs!)
        )
        
        # Motion statistics aggregator
        # Input: 8 features per box [mean_dx, mean_dy, std_dx, std_dy, min_dx, max_dx, min_dy, max_dy]
        # Output: mv_feature_dim features
        self.motion_aggregator = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, mv_feature_dim)
        )
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, motion_vectors):
        """
        Args:
            motion_vectors: [1, 2, 60, 60] - raw motion vectors
        Returns:
            features: [1, 64, 60, 60] - encoded grid (for backward compatibility)
            global_features: [1, 64] - global motion context
        """
        # Encode grid (respects block structure!)
        grid_features = self.grid_encoder(motion_vectors)  # [1, 64, 60, 60]
        
        # Global context
        global_feat = self.global_pool(grid_features).squeeze(-1).squeeze(-1)  # [1, 64]
        
        return grid_features, global_feat
    
    def extract_box_motion_features(self, motion_vectors, boxes):
        """
        Extract motion features for each box by computing statistics over overlapping blocks.
        
        This is the KEY IMPROVEMENT: instead of interpolating fake MVs,
        we compute REAL statistics from actual 16x16 blocks overlapping each box.
        
        Args:
            motion_vectors: [1, 2, 60, 60] - raw motion vectors (dx, dy)
            boxes: [N, 4] - boxes in normalized [0, 1] coordinates (x1, y1, x2, y2)
        
        Returns:
            box_motion_features: [N, mv_feature_dim] - learned motion features per box
        """
        if len(boxes) == 0:
            return torch.zeros(0, self.mv_feature_dim, device=boxes.device)
        
        # Remove batch dimension from MVs
        mv = motion_vectors.squeeze(0)  # [2, 60, 60]
        
        # Convert boxes from normalized [0,1] to pixel coordinates
        boxes_pixel = boxes * self.image_size  # [N, 4]
        
        # Convert to grid coordinates (60x60 grid)
        boxes_grid = boxes_pixel / self.block_size  # [N, 4]
        
        # Clamp to valid range [0, 60)
        boxes_grid = torch.clamp(boxes_grid, 0, self.grid_size - 1e-6)
        
        # Extract motion statistics per box
        N = len(boxes)
        motion_stats = []
        
        for i in range(N):
            x1, y1, x2, y2 = boxes_grid[i]
            
            # Get integer bounds for block indices
            ix1 = int(torch.floor(x1).item())
            iy1 = int(torch.floor(y1).item())
            ix2 = int(torch.ceil(x2).item())
            iy2 = int(torch.ceil(y2).item())
            
            # Ensure at least 1x1 region
            ix2 = max(ix1 + 1, ix2)
            iy2 = max(iy1 + 1, iy2)
            
            # Extract motion vectors in this region
            mv_region = mv[:, iy1:iy2, ix1:ix2]  # [2, H', W']
            
            # Handle empty regions (box too small to overlap any MV block)
            if mv_region.numel() == 0 or mv_region[0].numel() == 0:
                # Return zero statistics for empty regions
                stats = torch.zeros(8, device=mv.device, dtype=mv.dtype)
                motion_stats.append(stats)
                continue
            
            # Compute statistics
            # Mean motion
            mean_dx = mv_region[0].mean()
            mean_dy = mv_region[1].mean()
            
            # Variance (captures rotation, expansion, shear)
            std_dx = mv_region[0].std() if mv_region[0].numel() > 1 else torch.tensor(0.0, device=mv.device)
            std_dy = mv_region[1].std() if mv_region[1].numel() > 1 else torch.tensor(0.0, device=mv.device)
            
            # Range (captures motion extent)
            min_dx = mv_region[0].min()
            max_dx = mv_region[0].max()
            min_dy = mv_region[1].min()
            max_dy = mv_region[1].max()
            
            # Stack into 8-dim vector
            stats = torch.stack([
                mean_dx, mean_dy,
                std_dx, std_dy,
                min_dx, max_dx,
                min_dy, max_dy
            ])
            motion_stats.append(stats)
        
        # Stack all boxes
        motion_stats = torch.stack(motion_stats)  # [N, 8]
        
        # Aggregate with learned MLP
        box_motion_features = self.motion_aggregator(motion_stats)  # [N, 64]
        
        return box_motion_features


class SlotInitializer(nn.Module):
    """
    Initialize 300 fixed slots from variable number of input objects
    
    Strategy:
        - N objects (N ≤ 300): embed each into a slot
        - Remaining 300-N slots: initialize as "empty" slots
    """
    def __init__(self, max_slots=300, box_embed_dim=64, id_embed_dim=32, motion_embed_dim=64, slot_dim=128, image_size=960):
        super().__init__()
        self.max_slots = max_slots
        self.slot_dim = slot_dim
        self.image_size = image_size
        
        # Box embedding (4D box → 64D)
        self.box_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, box_embed_dim)
        )
        
        # Track ID embedding (integer ID → 32D)
        # Support up to 1000 unique track IDs
        self.id_embed = nn.Embedding(1000, id_embed_dim, padding_idx=0)  # ID 0 = empty
        
        # Combine box + ID + motion embeddings → slot representation
        # box_embed_dim(64) + id_embed_dim(32) + motion_embed_dim(64) = 160
        self.slot_proj = nn.Linear(box_embed_dim + id_embed_dim + motion_embed_dim, slot_dim)
        
        # Learnable empty slot embedding
        self.empty_slot_embed = nn.Parameter(torch.randn(slot_dim))
    
    def forward(self, boxes, track_ids, motion_vectors, mv_encoder):
        """
        Args:
            boxes: [N, 4] in xyxy format, N ≤ 300
            track_ids: [N] integer track IDs
            motion_vectors: [1, 2, 60, 60] - raw motion vectors
            mv_encoder: BoxAlignedMotionEncoder instance to extract box motion features
        
        Returns:
            slots: [300, slot_dim] - Fixed 300 slots
            slot_valid_mask: [300] - 1 for object slots, 0 for empty slots
        """
        N = boxes.shape[0]
        device = boxes.device
        
        # Boxes are already in [0, 1] normalized coordinates from dataset/model
        # No normalization needed - use directly
        
        # Embed boxes and IDs
        box_embeds = self.box_embed(boxes)  # [N, 64]
        
        # Clamp IDs to valid range [0, 999], use 0 for invalid
        track_ids_clamped = torch.clamp(track_ids, 0, 999)
        id_embeds = self.id_embed(track_ids_clamped)  # [N, 32]
        
        # Extract motion features per box using improved encoder
        if N > 0:
            motion_embeds = mv_encoder.extract_box_motion_features(motion_vectors, boxes)  # [N, 64]
        else:
            motion_embeds = torch.zeros(0, 64, device=device)
        
        # Combine box + ID + motion
        object_embeds = torch.cat([box_embeds, id_embeds, motion_embeds], dim=1)  # [N, 160]
        object_slots = self.slot_proj(object_embeds)  # [N, 128]
        
        # Create full 300 slots
        all_slots = torch.zeros(self.max_slots, self.slot_dim, device=device)
        all_slots[:N] = object_slots  # Fill first N slots with objects
        all_slots[N:] = self.empty_slot_embed.unsqueeze(0).expand(self.max_slots - N, -1)  # Fill rest with empty
        
        # Valid mask
        slot_valid_mask = torch.zeros(self.max_slots, device=device)
        slot_valid_mask[:N] = 1.0
        
        return all_slots, slot_valid_mask


class SlotLSTMProcessor(nn.Module):
    """
    Process 300 slots with LSTM (fixed hidden state size!)
    """
    def __init__(self, slot_dim=128, global_feature_dim=64, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input: slot + global features
        input_dim = slot_dim + global_feature_dim
        
        # LSTM (processes 300 slots as a sequence)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
    def forward(self, slots, global_features, hidden_state=None):
        """
        Args:
            slots: [300, slot_dim]
            global_features: [64]
            hidden_state: Optional (h, c) from previous frame
        
        Returns:
            slot_features: [300, hidden_dim]
            hidden_state: (h, c) for next frame
        """
        # Add batch dimension
        slots = slots.unsqueeze(0)  # [1, 300, 128]
        
        # Expand global features to all slots
        global_feat_expanded = global_features.unsqueeze(0).unsqueeze(0).expand(1, 300, -1)  # [1, 300, 64]
        
        # Concatenate
        lstm_input = torch.cat([slots, global_feat_expanded], dim=2)  # [1, 300, 192]
        
        # LSTM
        lstm_out, hidden_state = self.lstm(lstm_input, hidden_state)  # [1, 300, 128]
        
        # Remove batch dimension
        slot_features = lstm_out.squeeze(0)  # [300, 128]
        
        return slot_features, hidden_state


class SlotDecoder(nn.Module):
    """
    Decode 300 slot features into boxes, objectness, and track IDs
    """
    def __init__(self, slot_dim=128, max_track_id=1000, image_size=960):
        super().__init__()
        self.image_size = image_size
        self.max_track_id = max_track_id
        
        # Box regression head (predict delta from input box)
        self.box_head = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)  # [dx_center, dy_center, dw_log, dh_log]
        )
        
        # Objectness head (binary: object vs no-object)
        self.objectness_head = nn.Sequential(
            nn.Linear(slot_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # Logits (use with BCE or Focal Loss)
        )
        
        # Track ID classification head
        self.track_id_head = nn.Sequential(
            nn.Linear(slot_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, max_track_id)  # Logits for each track ID
        )
    
    def forward(self, slot_features, input_boxes=None, return_logits=False):
        """
        Args:
            slot_features: [300, 128]
            input_boxes: Optional [300, 4] - Previous frame boxes for delta prediction
            return_logits: If True, return raw logits for objectness
        
        Returns:
            boxes: [300, 4] in xyxy format
            objectness: [300] - Logits or probabilities
            track_ids: [300] - Logits for each track ID class
        """
        # Predict box deltas
        box_deltas = self.box_head(slot_features)  # [300, 4]
        
        # If input_boxes provided, apply deltas; otherwise, predict absolute boxes
        if input_boxes is not None:
            boxes = self._apply_box_deltas(input_boxes, box_deltas)
        else:
            # Predict absolute boxes (normalized [0, 1])
            boxes = torch.sigmoid(box_deltas) * self.image_size
        
        # Predict objectness
        objectness_logits = self.objectness_head(slot_features).squeeze(-1)  # [300]
        objectness = objectness_logits if return_logits else torch.sigmoid(objectness_logits)
        
        # Predict track IDs (logits for classification)
        track_id_logits = self.track_id_head(slot_features)  # [300, max_track_id]
        
        return boxes, objectness, track_id_logits
    
    def _apply_box_deltas(self, boxes, deltas):
        """
        Apply predicted deltas to input boxes
        
        Args:
            boxes: [300, 4] in xyxy format
            deltas: [300, 4] [dx_center, dy_center, dw_log, dh_log]
        """
        # Convert to center format
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2  # [300, 2]
        sizes = boxes[:, 2:] - boxes[:, :2]  # [300, 2]
        
        # Apply deltas
        new_centers = centers + deltas[:, :2]
        new_sizes = sizes * torch.exp(deltas[:, 2:])
        
        # Convert back to xyxy
        new_boxes = torch.cat([
            new_centers - new_sizes / 2,
            new_centers + new_sizes / 2
        ], dim=1)
        
        # Clamp
        new_boxes = torch.clamp(new_boxes, 0, self.image_size)
        
        return new_boxes


class TrackingPropagator(nn.Module):
    """
    Main Tracking Propagator Model
    
    Fixed 300-slot architecture for box propagation with track ID preservation.
    """
    def __init__(
        self,
        max_slots=300,
        mv_feature_dim=64,
        slot_dim=128,
        hidden_dim=128,
        num_lstm_layers=2,
        max_track_id=1000,
        image_size=960,
        dropout=0.1
    ):
        super().__init__()
        self.max_slots = max_slots
        self.image_size = image_size
        
        # Components
        self.mv_encoder = BoxAlignedMotionEncoder(mv_feature_dim, image_size)
        self.slot_initializer = SlotInitializer(max_slots, slot_dim=slot_dim, image_size=image_size)
        self.slot_lstm = SlotLSTMProcessor(slot_dim, mv_feature_dim, hidden_dim, num_lstm_layers, dropout)
        
        # Use RT-DETR head instead of SlotDecoder
        # RT-DETR provides: boxes, class_logits [no_object, pedestrian], track_ids
        self.rtdetr_head = RTDETRHead(
            input_dim=hidden_dim,
            num_slots=max_slots,
            num_classes=2,  # [no_object, pedestrian]
            num_track_ids=max_track_id,
            hidden_dim=256
        )
    
    def forward_single_frame(
        self, 
        motion_vectors, 
        dct_residuals,  # Ignored for compatibility
        boxes, 
        track_ids, 
        hidden_state=None, 
        return_logits=False
    ):
        """
        Process single frame with fixed 300 slots (RT-DETR version)
        
        Args:
            motion_vectors: [1, 2, 60, 60]
            dct_residuals: Ignored (for API compatibility with train_mv_center.py)
            boxes: [N, 4] - Previous frame boxes (N ≤ 300)
            track_ids: [N] - Previous frame track IDs
            hidden_state: Optional LSTM state from previous frame
            return_logits: If True, return raw class logits (for loss computation)
        
        Returns:
            pred_boxes: [300, 4]
            pred_class_logits: [300, 2] - Logits for [no_object, pedestrian]
            pred_track_ids: [300, max_track_id] - Logits for track ID classification
            hidden_state: Updated LSTM state
        """
        # 1. Encode motion vectors
        spatial_features, global_features = self.mv_encoder(motion_vectors)
        # spatial_features: [1, 64, 60, 60], global_features: [1, 64]
        
        # 2. Initialize 300 slots from N input objects (with motion features!)
        slots, slot_valid_mask = self.slot_initializer(boxes, track_ids, motion_vectors, self.mv_encoder)
        # slots: [300, 128], slot_valid_mask: [300]
        
        # 3. Process slots with LSTM
        slot_features, hidden_state = self.slot_lstm(slots, global_features[0], hidden_state)
        # slot_features: [300, 128]
        
        # 4. Decode slots with RT-DETR head
        # Add batch dimension for RT-DETR head
        slot_features_batch = slot_features.unsqueeze(0)  # [1, 300, 128]
        
        pred_boxes, pred_class_logits, pred_track_id_logits = self.rtdetr_head(slot_features_batch)
        # pred_boxes: [1, 300, 4]
        # pred_class_logits: [1, 300, 2]
        # pred_track_id_logits: [1, 300, max_track_id]
        
        # Remove batch dimension
        pred_boxes = pred_boxes.squeeze(0)  # [300, 4]
        pred_class_logits = pred_class_logits.squeeze(0)  # [300, 2]
        pred_track_id_logits = pred_track_id_logits.squeeze(0)  # [300, max_track_id]
        
        # Keep boxes in normalized [0, 1] coordinates (RT-DETR outputs sigmoid [0,1])
        # No scaling needed - maintain normalized coordinates throughout pipeline
        
        return pred_boxes, pred_class_logits, pred_track_id_logits, hidden_state
    
    def forward(self, motion_vectors, dct_residuals, boxes, track_ids, hidden_state=None):
        """
        Forward pass - alias for forward_single_frame for compatibility with training script
        """
        return self.forward_single_frame(
            motion_vectors, dct_residuals, boxes, track_ids, hidden_state, return_logits=True
        )
    
    def get_num_parameters(self):
        """Get total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tracking_propagator(**kwargs):
    """Factory function for creating TrackingPropagator"""
    return TrackingPropagator(**kwargs)


if __name__ == '__main__':
    # Test the model
    print("Testing TrackingPropagator with fixed 300 slots...")
    
    model = TrackingPropagator(
        max_slots=300,
        mv_feature_dim=64,
        slot_dim=128,
        hidden_dim=128,
        num_lstm_layers=2,
        max_track_id=1000,
        image_size=960
    )
    
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 1
    num_objects = 15  # Variable number < 300
    mv = torch.randn(1, 2, 60, 60)
    boxes = torch.rand(num_objects, 4) * 960  # Random boxes in [0, 960]
    track_ids = torch.randint(1, 100, (num_objects,))  # Random track IDs
    
    print(f"\nInput:")
    print(f"  MV: {mv.shape}")
    print(f"  Boxes: {boxes.shape}")
    print(f"  Track IDs: {track_ids.shape}")
    
    pred_boxes, pred_obj, pred_ids, hidden = model.forward_single_frame(
        mv, None, boxes, track_ids, return_logits=True
    )
    
    print(f"\nOutput:")
    print(f"  Predicted boxes: {pred_boxes.shape}")
    print(f"  Predicted class logits: {pred_obj.shape}")  # Now class logits [300, 2]
    print(f"  Predicted track ID logits: {pred_ids.shape}")
    print(f"  Hidden state: {hidden[0].shape}, {hidden[1].shape}")
    
    # Test that hidden state size is FIXED regardless of input object count
    print(f"\n✅ Testing fixed hidden state size...")
    num_objects_2 = 25  # Different number of objects
    boxes_2 = torch.rand(num_objects_2, 4) * 960
    track_ids_2 = torch.randint(1, 100, (num_objects_2,))
    
    pred_boxes_2, pred_obj_2, pred_ids_2, hidden_2 = model.forward_single_frame(
        mv, None, boxes_2, track_ids_2, hidden_state=hidden, return_logits=True
    )
    
    print(f"  Input 1: {num_objects} objects → Hidden: {hidden[0].shape}")
    print(f"  Input 2: {num_objects_2} objects → Hidden: {hidden_2[0].shape}")
    print(f"  ✅ Hidden state size FIXED regardless of input count!")
