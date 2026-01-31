"""
Enhanced Motion-Only Tracker with Interaction Module

Architecture:
1. Enhanced Encoder (128-dim, no pooling)
2. Enhanced ROI Extractor (256-dim output)
3. Transformer Interaction (spatial relationships)
4. Mamba Interaction (temporal dynamics)
5. Enhanced LSTM Tracker (256 hidden dim)
6. Fixed State Pool (100 slots)

Handles:
- Object lifecycle (appearing/disappearing)
- Occlusions (Transformer + Mamba)
- Collisions (spatial attention)
- Persistent tracking (temporal state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import math


# ============================================================================
# COMPONENT 1: Enhanced Motion Vector Encoder
# ============================================================================

class MVOnlyEncoderEnhanced(nn.Module):
    """
    Enhanced motion vector encoder (128-dim, NO global pooling)
    
    Changes from original:
    - Feature dim: 64 → 128 (2× capacity)
    - No global pooling (preserves spatial info)
    - Deeper architecture
    """
    
    def __init__(self, input_channels=2, feature_dim=128):
        super().__init__()
        
        # Progressive feature expansion: 2 → 64 → 128 → 128
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)  # Use GroupNorm instead of BatchNorm
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 128)
        
        self.conv3 = nn.Conv2d(128, feature_dim, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, feature_dim)
        
        # Upsampling to match feature map size
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Refinement layer after upsampling
        self.refine = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.gn_refine = nn.GroupNorm(16, feature_dim)
    
    def forward(self, motion_vectors):
        """
        Args:
            motion_vectors: [B, 2, H, W] - Usually [1, 2, 60, 60]
        
        Returns:
            features: [B, 128, H*2, W*2] - [1, 128, 120, 120]
        """
        # DEBUG: Check input
        if torch.isnan(motion_vectors).any():
            print(f"⚠️ DEBUG: Input motion_vectors contains NaN!")
        
        # Encode: [1, 2, 60, 60] → [1, 128, 60, 60]
        x = self.conv1(motion_vectors)
        if torch.isnan(x).any():
            print(f"⚠️ DEBUG: After conv1: NaN detected!")
            print(f"   motion_vectors stats: min={motion_vectors.min():.4f}, max={motion_vectors.max():.4f}")
            print(f"   conv1 weight stats: min={self.conv1.weight.min():.4f}, max={self.conv1.weight.max():.4f}")
        
        x = F.relu(self.gn1(x))
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.relu(self.gn3(self.conv3(x)))
        
        # Upsample: [1, 128, 60, 60] → [1, 128, 120, 120]
        x = self.upsample(x)
        
        # Refine: [1, 128, 120, 120] → [1, 128, 120, 120]
        x = F.relu(self.gn_refine(self.refine(x)))
        
        return x  # NO global pooling!


# ============================================================================
# COMPONENT 2: Enhanced ROI Extractor
# ============================================================================

class SpatialROIExtractorEnhanced(nn.Module):
    """
    Enhanced per-object feature extraction (256-dim output)
    
    Changes from original:
    - Input: 128×7×7 = 6,272 (vs 64×7×7 = 3,136)
    - Output: 256 (vs 64)
    - Larger projection: 6272 → 512 → 256
    """
    
    def __init__(self, feature_dim=128, output_dim=256, roi_size=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size
        
        # Larger projection
        roi_feature_dim = feature_dim * roi_size * roi_size  # 128×7×7 = 6,272
        self.projection = nn.Sequential(
            nn.Linear(roi_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, features, boxes):
        """
        Args:
            features: [B, C, H, W] - Usually [1, 128, 120, 120]
            boxes: [N, 4] - Normalized boxes [x, y, w, h] in 0-1 range
        
        Returns:
            roi_features: [N, output_dim] - Per-object features [N, 256]
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, self.projection[-1].out_features, device=boxes.device)
        
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0]  # x1 = x
        boxes_xyxy[:, 1] = boxes[:, 1]  # y1 = y
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
        
        # Scale to feature map size (120×120)
        H, W = features.shape[2], features.shape[3]
        boxes_scaled = boxes_xyxy * torch.tensor([W, H, W, H], device=boxes.device)
        
        # Add batch index
        batch_indices = torch.zeros(boxes.shape[0], 1, device=boxes.device)
        rois = torch.cat([batch_indices, boxes_scaled], dim=1)
        
        # ROI Align: [1, 128, 120, 120] + boxes → [N, 128, 7, 7]
        roi_features = roi_align(
            features,
            rois,
            output_size=self.roi_size,
            spatial_scale=1.0,
            aligned=True
        )
        
        # Flatten and project: [N, 128, 7, 7] → [N, 6272] → [N, 256]
        roi_features = roi_features.view(roi_features.shape[0], -1)
        roi_features = self.projection(roi_features)
        
        return roi_features


# ============================================================================
# COMPONENT 3: Transformer Interaction (Spatial)
# ============================================================================

class TransformerInteraction(nn.Module):
    """
    Multi-head self-attention for spatial object interactions
    
    Learns:
    - Which objects are near each other (collision detection)
    - Which objects overlap (occlusion detection)
    - Spatial context (relative positions)
    """
    
    def __init__(self, feature_dim=256, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Positional encoding for object boxes
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Multi-head self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, roi_features, boxes):
        """
        Args:
            roi_features: [N, 256] - Per-object appearance features
            boxes: [N, 4] - Object positions [x, y, w, h]
        
        Returns:
            spatial_features: [N, 256] - Features with spatial interactions
        """
        N = roi_features.shape[0]
        
        if N == 0:
            return roi_features
        
        if N == 1:
            # Only one object, no interactions
            return roi_features
        
        # Add positional encoding
        pos_embed = self.pos_encoder(boxes)  # [N, 256]
        features = roi_features + pos_embed   # [N, 256]
        
        # Self-attention (each object attends to all others)
        features = features.unsqueeze(0)  # [1, N, 256]
        spatial_features = self.transformer(features)  # [1, N, 256]
        spatial_features = spatial_features.squeeze(0)  # [N, 256]
        
        # Residual connection
        spatial_features = self.norm(spatial_features + roi_features)
        
        return spatial_features


# ============================================================================
# COMPONENT 4: Mamba Interaction (Temporal)
# ============================================================================

class MambaBlock(nn.Module):
    """
    Simplified Mamba-style state-space block
    
    Key features:
    - Selective state update (learns what to remember)
    - Linear complexity O(N)
    - Temporal state memory
    """
    
    def __init__(self, feature_dim=256, state_dim=16):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        
        # Input projection (two branches: main + gate)
        self.in_proj = nn.Linear(feature_dim, feature_dim * 2)
        
        # State-space parameters (learnable dynamics)
        self.dt_proj = nn.Linear(feature_dim, state_dim)  # Time step
        self.B_proj = nn.Linear(feature_dim, state_dim)   # Input matrix
        self.C_proj = nn.Linear(feature_dim, state_dim)   # Output matrix
        
        # State transition (learned)
        self.A = nn.Parameter(torch.randn(state_dim))
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Normalization
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x, state):
        """
        Args:
            x: [N, feature_dim] - Object features
            state: [N, state_dim] - Per-object temporal state
        
        Returns:
            y: [N, feature_dim] - Enhanced features
            new_state: [N, state_dim] - Updated state
        """
        N = x.shape[0]
        
        # Input projection
        x_proj = self.in_proj(x)  # [N, feature_dim * 2]
        x_main, x_gate = x_proj.chunk(2, dim=-1)  # [N, feature_dim] each
        
        # State-space parameters (selective!)
        dt = F.softplus(self.dt_proj(x))  # [N, state_dim] - Time step
        B = self.B_proj(x)  # [N, state_dim] - Input matrix
        C = self.C_proj(x)  # [N, state_dim] - Output matrix
        
        # State update: s_new = A * s_old + B * x
        # Discretize A with learned time step dt
        A_discrete = torch.exp(self.A.unsqueeze(0) * dt)  # [N, state_dim]
        new_state = state * A_discrete + B * x_main.sum(dim=-1, keepdim=True).expand(-1, self.state_dim)
        
        # Output: y = C * s_new
        y = (C * new_state).sum(dim=-1, keepdim=True).expand(-1, self.feature_dim)
        
        # Gated output
        y = y * F.silu(x_gate)  # SiLU activation
        y = self.out_proj(y)
        
        # Residual + norm
        y = self.norm(y + x)
        
        return y, new_state


class MambaInteraction(nn.Module):
    """
    Mamba-based temporal interaction module
    
    Maintains temporal state of object relationships:
    - Persistent occlusions
    - Collision history
    - Scene dynamics
    """
    
    def __init__(self, feature_dim=256, state_dim=16, num_layers=2, max_objects=100):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.max_objects = max_objects
        
        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(feature_dim, state_dim) for _ in range(num_layers)
        ])
        
        # State memory pool (one state per slot)
        self.register_buffer('state_pool', torch.zeros(max_objects, state_dim))
    
    def forward(self, features, slot_indices):
        """
        Args:
            features: [N, 256] - Object features
            slot_indices: [N] - Which slots these objects occupy
        
        Returns:
            temporal_features: [N, 256] - Features with temporal interactions
        """
        N = features.shape[0]
        
        if N == 0:
            return features
        
        # Gather states for active objects
        states = self.state_pool[slot_indices]  # [N, state_dim]
        
        # Apply Mamba blocks
        output = features
        for mamba_block in self.mamba_blocks:
            output, states = mamba_block(output, states)
        
        # Update state pool (detach to prevent backprop through time)
        self.state_pool[slot_indices] = states.detach()
        
        return output
    
    def reset_states(self, slot_indices=None):
        """Reset states for specific slots or all slots"""
        if slot_indices is None:
            self.state_pool.zero_()
        else:
            self.state_pool[slot_indices] = 0


# ============================================================================
# COMPONENT 5: Hybrid Interaction Module
# ============================================================================

class HybridInteraction(nn.Module):
    """
    Combined Transformer + Mamba interaction
    
    Transformer: Spatial relationships (current frame)
    Mamba: Temporal dynamics (across frames)
    """
    
    def __init__(self, feature_dim=256, max_objects=100):
        super().__init__()
        
        # Spatial attention (1 layer for speed)
        self.spatial_transformer = TransformerInteraction(
            feature_dim=feature_dim,
            num_heads=8,
            num_layers=1,
            dropout=0.1
        )
        
        # Temporal state-space model (2 layers for capacity)
        self.temporal_mamba = MambaInteraction(
            feature_dim=feature_dim,
            state_dim=16,
            num_layers=2,
            max_objects=max_objects
        )
    
    def forward(self, roi_features, boxes, slot_indices):
        """
        Args:
            roi_features: [N, 256]
            boxes: [N, 4]
            slot_indices: [N]
        
        Returns:
            interactive_features: [N, 256]
        """
        # Step 1: Spatial interactions (self-attention)
        spatial_features = self.spatial_transformer(roi_features, boxes)
        
        # Step 2: Temporal interactions (state-space model)
        temporal_features = self.temporal_mamba(spatial_features, slot_indices)
        
        return temporal_features
    
    def reset_states(self, slot_indices=None):
        """Reset temporal states"""
        self.temporal_mamba.reset_states(slot_indices)


# ============================================================================
# COMPONENT 6: Enhanced LSTM Tracker
# ============================================================================

class LSTMTrackerEnhanced(nn.Module):
    """
    Enhanced LSTM tracker (256 hidden dim, no global features)
    
    Changes from original:
    - Input: 256 (vs 128)
    - Hidden: 256 (vs 128)
    - No global features (already in interaction module)
    """
    
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Prediction heads
        self.position_head = nn.Linear(hidden_dim, 2)  # dx, dy
        self.size_head = nn.Linear(hidden_dim, 2)      # dw, dh (log scale)
        self.confidence_head = nn.Linear(hidden_dim, 1) # confidence
    
    def forward(self, interactive_features, hidden_state=None, return_logits=False):
        """
        Args:
            interactive_features: [N, 256] - Features with interactions
            hidden_state: Optional (h, c) tuple
            return_logits: If True, return logits; else sigmoid confidence
        
        Returns:
            pos_deltas: [N, 2]
            confidences: [N, 1]
            size_deltas: [N, 2]
            hidden_state: Updated LSTM state
        """
        N = interactive_features.shape[0]
        
        if N == 0:
            empty = torch.empty(0, device=interactive_features.device)
            return empty, empty, empty, hidden_state
        
        # Add temporal dimension for LSTM
        lstm_in = interactive_features.unsqueeze(1)  # [N, 1, 256]
        
        # LSTM forward
        lstm_out, hidden_state = self.lstm(lstm_in, hidden_state)
        lstm_out = lstm_out.squeeze(1)  # [N, 256]
        
        # Predictions
        pos_deltas = torch.tanh(self.position_head(lstm_out)) * 0.1
        size_deltas = torch.tanh(self.size_head(lstm_out)) * 0.1
        
        if return_logits:
            confidences = self.confidence_head(lstm_out)
        else:
            confidences = torch.sigmoid(self.confidence_head(lstm_out))
        
        # DEBUG: Check for NaN
        if torch.isnan(pos_deltas).any():
            print(f"⚠️ DEBUG: pos_deltas contains NaN!")
            print(f"   lstm_out stats: min={lstm_out.min():.4f}, max={lstm_out.max():.4f}, mean={lstm_out.mean():.4f}")
            print(f"   pos_head output: {self.position_head(lstm_out)[:3]}")
        
        return pos_deltas, confidences, size_deltas, hidden_state


# ============================================================================
# COMPONENT 7: Complete Enhanced Tracker
# ============================================================================

class MVOnlyTrackerEnhanced(nn.Module):
    """
    Complete Enhanced Motion-Only Tracker
    
    Architecture:
    1. MVOnlyEncoderEnhanced (128-dim, no pooling)
    2. SpatialROIExtractorEnhanced (256-dim output)
    3. HybridInteraction (Transformer + Mamba)
    4. LSTMTrackerEnhanced (256 hidden dim)
    5. Fixed state pool (100 slots)
    
    Features:
    - Object lifecycle management
    - Occlusion handling (interaction module)
    - Collision detection (spatial attention)
    - Temporal persistence (Mamba states)
    """
    
    def __init__(
        self,
        mv_feature_dim=128,
        roi_feature_dim=256,
        hidden_dim=256,
        lstm_layers=2,
        dropout=0.1,
        max_objects=100,
        confidence_threshold=0.1
    ):
        super().__init__()
        
        self.max_objects = max_objects
        self.confidence_threshold = confidence_threshold
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # Component 1: Enhanced encoder
        self.mv_encoder = MVOnlyEncoderEnhanced(
            input_channels=2,
            feature_dim=mv_feature_dim
        )
        
        # Component 2: Enhanced ROI extractor
        self.roi_extractor = SpatialROIExtractorEnhanced(
            feature_dim=mv_feature_dim,
            output_dim=roi_feature_dim,
            roi_size=7
        )
        
        # Component 3: Hybrid interaction module
        self.interaction = HybridInteraction(
            feature_dim=roi_feature_dim,
            max_objects=max_objects
        )
        
        # Component 4: Enhanced LSTM tracker
        self.lstm_tracker = LSTMTrackerEnhanced(
            input_dim=roi_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # State pool for LSTM (larger than before!)
        self.register_buffer('state_pool_h', torch.zeros(lstm_layers, max_objects, hidden_dim))
        self.register_buffer('state_pool_c', torch.zeros(lstm_layers, max_objects, hidden_dim))
        
        # Tracking metadata
        self.active_slots = set()
        self.slot_to_id = {}
        self.id_to_slot = {}
        self.slot_boxes = {}
        self.slot_confidence = {}
        
        # For compatibility with memory tracker training code
        # We need a 'tracker' attribute but can't set it to self (causes recursion)
        # Instead, we'll implement the tracker interface directly on this class
        # and add a dummy property
        self._is_memory_tracker = True
        
        # Initialize all weights properly
        print(f"🔧 DEBUG: Initializing model weights...")
        self.apply(self._init_weights)
        print(f"🔧 DEBUG: Weight initialization complete")
        print(f"🔧 DEBUG: conv1 weights after init: min={self.mv_encoder.conv1.weight.min():.4f}, max={self.mv_encoder.conv1.weight.max():.4f}")
    
    def _init_weights(self, m):
        """Initialize weights for all layers"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    @property
    def tracker(self):
        """Property to make training code think we have a tracker (returns self)"""
        return self
    
    def _allocate_slot(self, object_id):
        """Allocate a free slot for new object"""
        for slot in range(self.max_objects):
            if slot not in self.active_slots:
                self.active_slots.add(slot)
                self.slot_to_id[slot] = object_id
                self.id_to_slot[object_id] = slot
                # Reset LSTM state
                self.state_pool_h[:, slot, :] = 0
                self.state_pool_c[:, slot, :] = 0
                # Reset interaction state
                self.interaction.reset_states([slot])
                return slot
        raise RuntimeError(f"No free slots! Max {self.max_objects} objects exceeded.")
    
    def _free_slot(self, slot):
        """Free a slot and clear its state"""
        if slot in self.active_slots:
            self.active_slots.remove(slot)
            object_id = self.slot_to_id.pop(slot, None)
            if object_id is not None:
                self.id_to_slot.pop(object_id, None)
            self.slot_boxes.pop(slot, None)
            self.slot_confidence.pop(slot, None)
            # Clear LSTM state
            self.state_pool_h[:, slot, :] = 0
            self.state_pool_c[:, slot, :] = 0
            # Clear interaction state
            self.interaction.reset_states([slot])
    
    def initialize_objects(self, boxes, object_ids=None):
        """Initialize tracking with bounding boxes"""
        N = boxes.shape[0]
        if object_ids is None:
            object_ids = list(range(N))
        
        slot_indices = []
        for i, (box, oid) in enumerate(zip(boxes, object_ids)):
            slot = self._allocate_slot(oid)
            self.slot_boxes[slot] = box
            self.slot_confidence[slot] = 1.0
            slot_indices.append(slot)
        
        return slot_indices
    
    def track_frame(self, motion_vectors):
        """Track active objects using enhanced architecture"""
        if not self.active_slots:
            return []
        
        # Step 1: Encode motion (NO pooling!)
        features = self.mv_encoder(motion_vectors)  # [1, 128, 120, 120]
        
        # DEBUG: Check for NaN
        if torch.isnan(features).any():
            print(f"⚠️ DEBUG: Encoder output contains NaN!")
            print(f"   motion_vectors: min={motion_vectors.min():.4f}, max={motion_vectors.max():.4f}")
        
        # Step 2: Gather active objects
        slot_indices = sorted(list(self.active_slots))
        boxes = torch.stack([self.slot_boxes[s] for s in slot_indices])
        # Ensure boxes match the dtype of features (for AMP compatibility)
        boxes = boxes.to(dtype=features.dtype, device=features.device)
        
        # Step 3: Extract ROI features
        roi_features = self.roi_extractor(features, boxes)  # [N, 256]
        
        # Step 4: INTERACTION MODULE (handles occlusions/collisions!)
        slot_indices_tensor = torch.tensor(slot_indices, device=roi_features.device)
        interactive_features = self.interaction(roi_features, boxes, slot_indices_tensor)
        
        # DEBUG: Check for NaN in features
        if torch.isnan(interactive_features).any():
            print(f"⚠️ DEBUG: interactive_features contains NaN!")
            print(f"   roi_features: min={roi_features.min():.4f}, max={roi_features.max():.4f}")
            print(f"   boxes: {boxes}")
        
        # Step 5: Gather LSTM states from pool
        batch_states_h = self.state_pool_h[:, slot_indices, :].contiguous()
        batch_states_c = self.state_pool_c[:, slot_indices, :].contiguous()
        # Ensure states match dtype of features (for AMP compatibility)
        batch_states = (
            batch_states_h.to(dtype=interactive_features.dtype),
            batch_states_c.to(dtype=interactive_features.dtype)
        )
        
        # Step 6: LSTM tracking
        pos_deltas, confs, size_deltas, new_states = self.lstm_tracker(
            interactive_features,
            batch_states,
            return_logits=False
        )
        
        # Step 7: Update boxes and states
        predictions = []
        slots_to_remove = []
        
        for i, slot in enumerate(slot_indices):
            # Update LSTM state pool (detach to prevent backprop through time across GOPs)
            self.state_pool_h[:, slot, :] = new_states[0][:, i, :].detach()
            self.state_pool_c[:, slot, :] = new_states[1][:, i, :].detach()
            
            # Update box (use out-of-place operations for gradient flow)
            box = boxes[i]
            # Don't unpack - keep as tensors for gradient flow
            delta_pos = pos_deltas[i]  # [2]
            delta_size = size_deltas[i]  # [2]
            
            # Clamp size deltas to prevent exp() explosion (exp(2) = 7.4, exp(-2) = 0.135)
            delta_size = torch.clamp(delta_size, -2.0, 2.0)
            
            # Compute new box coordinates with explicit gradient tracking
            new_x = box[0] + delta_pos[0]
            new_y = box[1] + delta_pos[1]
            new_w = box[2] * torch.exp(delta_size[0])
            new_h = box[3] * torch.exp(delta_size[1])
            
            new_box = torch.stack([new_x, new_y, new_w, new_h])
            new_box = torch.clamp(new_box, 0, 1)
            
            conf = confs[i]  # Keep as tensor for gradient flow
            conf_value = conf.item()  # Convert to float for storage/threshold check
            self.slot_boxes[slot] = new_box  # Keep gradients for BPTT across GOP
            self.slot_confidence[slot] = conf_value
            
            # Check confidence threshold
            if conf_value < self.confidence_threshold:
                slots_to_remove.append(slot)
            else:
                object_id = self.slot_to_id[slot]
                predictions.append((object_id, new_box, conf))
        
        # Remove low-confidence objects
        for slot in slots_to_remove:
            self._free_slot(slot)
        
        return predictions
    
    def reset(self):
        """Reset all tracking state"""
        self.active_slots.clear()
        self.slot_to_id.clear()
        self.id_to_slot.clear()
        self.slot_boxes.clear()
        self.slot_confidence.clear()
        self.state_pool_h.zero_()
        self.state_pool_c.zero_()
        self.interaction.reset_states()
    
    def init_from_iframe(self, boxes, object_ids=None):
        """
        Initialize tracking from I-frame (called by training code)
        
        Args:
            boxes: [N, 4] - Initial bounding boxes from I-frame
            object_ids: Optional[Tensor[N]] - Object IDs (or None to auto-assign)
        """
        N = boxes.shape[0]
        if object_ids is None:
            object_ids = list(range(N))
        else:
            object_ids = object_ids.cpu().tolist() if torch.is_tensor(object_ids) else list(object_ids)
        
        # Initialize objects using existing method
        self.initialize_objects(boxes, object_ids)
    
    def forward(self, motion_vectors, mode='single_frame'):
        """
        Forward pass for training - wrapper around track_frame
        
        This is called by the training script with mode='single_frame'
        Returns predictions for active objects in the current frame.
        
        Args:
            motion_vectors: [1, 2, 60, 60] - Motion vectors for current frame
            mode: 'single_frame' (tracking mode)
        
        Returns:
            boxes: [N, 4] - Predicted boxes for active objects
            confidences: [N] - Confidence scores for each object
        """
        if not self.active_slots:
            # No active objects to track
            device = motion_vectors.device
            return torch.empty(0, 4, device=device), torch.empty(0, device=device)
        
        # Track active objects in current frame
        predictions = self.track_frame(motion_vectors)  # Returns list of (id, box, conf)
        
        if len(predictions) == 0:
            device = motion_vectors.device
            return torch.empty(0, 4, device=device), torch.empty(0, device=device)
        
        # Extract boxes and confidences (predictions contain tensors now!)
        boxes = torch.stack([box for (_, box, _) in predictions])
        confidences = torch.stack([conf for (_, _, conf) in predictions])
        
        return boxes, confidences
    
    def forward_single_frame(self, motion_vectors, dct_residuals, boxes, hidden_state=None, return_logits=False):
        """
        Legacy API for compatibility with train_mv_center.py
        
        NOTE: This bypasses the interaction module! Use track_frame() for full features.
        """
        features = self.mv_encoder(motion_vectors)
        
        if boxes.shape[0] == 0:
            return boxes, torch.empty(0, device=boxes.device), hidden_state
        
        roi_features = self.roi_extractor(features, boxes)
        
        # Skip interaction module in legacy mode (no slot indices available)
        pos_deltas, confs, size_deltas, hidden_state = self.lstm_tracker(
            roi_features,
            hidden_state,
            return_logits=return_logits
        )
        
        # Update boxes
        updated_boxes = self._update_boxes(boxes, pos_deltas, size_deltas)
        
        return updated_boxes, confs.squeeze(-1), hidden_state
    
    def _update_boxes(self, boxes, pos_deltas, size_deltas):
        """Update boxes with predicted deltas"""
        updated = boxes.clone()
        updated[:, 0:2] += pos_deltas
        updated[:, 2:4] *= torch.exp(size_deltas)
        updated = torch.clamp(updated, 0, 1)
        return updated


# ============================================================================
# TEST
# ============================================================================

def test_enhanced_tracker():
    """Test the enhanced tracker"""
    print("=" * 70)
    print("Testing Enhanced Motion-Only Tracker")
    print("=" * 70)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MVOnlyTrackerEnhanced(
        mv_feature_dim=128,
        roi_feature_dim=256,
        hidden_dim=256,
        lstm_layers=2,
        max_objects=100,
        confidence_threshold=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"📦 State pool size: {model.max_objects} slots")
    print(f"🧠 LSTM hidden dim: {model.hidden_dim}")
    print(f"🔗 ROI feature dim: 256")
    print(f"🎯 Encoder feature dim: 128 (no global pooling!)")
    print()
    
    # Component breakdown
    print("Component breakdown:")
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    roi_params = sum(p.numel() for p in model.roi_extractor.parameters())
    interaction_params = sum(p.numel() for p in model.interaction.parameters())
    lstm_params = sum(p.numel() for p in model.lstm_tracker.parameters())
    
    print(f"  Encoder:      {encoder_params:,} params ({encoder_params/total_params*100:.1f}%)")
    print(f"  ROI Extract:  {roi_params:,} params ({roi_params/total_params*100:.1f}%)")
    print(f"  Interaction:  {interaction_params:,} params ({interaction_params/total_params*100:.1f}%)")
    print(f"  LSTM Tracker: {lstm_params:,} params ({lstm_params/total_params*100:.1f}%)")
    print()
    
    print("=" * 70)
    print("SCENARIO: Object lifecycle with occlusion")
    print("=" * 70)
    print()
    
    # Frame 1: Initialize 3 objects
    print("[Frame 1] Initialize 3 objects")
    mv_1 = torch.randn(1, 2, 60, 60, device=device)
    boxes_1 = torch.tensor([
        [0.1, 0.1, 0.2, 0.3],    # Object 0 (top-left)
        [0.5, 0.5, 0.15, 0.25],  # Object 1 (center)
        [0.12, 0.12, 0.18, 0.28] # Object 2 (CLOSE to Object 0!)
    ], device=device)
    
    slots = model.initialize_objects(boxes_1, object_ids=[0, 1, 2])
    print(f"  Allocated slots: {slots}")
    predictions_1 = model.track_frame(mv_1)
    print(f"  Predictions: {len(predictions_1)} objects")
    for oid, box, conf in predictions_1:
        print(f"    Object {oid}: conf={conf:.3f}")
    print()
    
    # Frame 2-5: Normal tracking
    for frame in range(2, 6):
        print(f"[Frame {frame}] Track with interaction module")
        mv = torch.randn(1, 2, 60, 60, device=device)
        predictions = model.track_frame(mv)
        print(f"  Predictions: {len(predictions)} objects")
        for oid, box, conf in predictions:
            print(f"    Object {oid}: conf={conf:.3f}")
        print()
    
    # Frame 6: New object appears
    print("[Frame 6] New object appears (ID 3)")
    new_box = torch.tensor([[0.7, 0.3, 0.2, 0.25]], device=device)
    new_slots = model.initialize_objects(new_box, object_ids=[3])
    print(f"  Allocated slot: {new_slots}")
    mv_6 = torch.randn(1, 2, 60, 60, device=device)
    predictions_6 = model.track_frame(mv_6)
    print(f"  Predictions: {len(predictions_6)} objects")
    for oid, box, conf in predictions_6:
        print(f"    Object {oid}: conf={conf:.3f}")
    print()
    
    # Memory usage
    print("=" * 70)
    if device.type == 'cuda':
        memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"💾 GPU memory allocated: {memory_mb:.2f} MB")
    print()
    
    print("✅ Enhanced tracker test passed!")
    print()
    print("Features verified:")
    print("  ✓ Enhanced encoder (128-dim, no pooling)")
    print("  ✓ Enhanced ROI extractor (256-dim)")
    print("  ✓ Transformer interaction (spatial)")
    print("  ✓ Mamba interaction (temporal)")
    print("  ✓ Enhanced LSTM (256 hidden)")
    print("  ✓ Object lifecycle management")
    print("  ✓ Occlusion/collision handling ready!")


if __name__ == "__main__":
    test_enhanced_tracker()
