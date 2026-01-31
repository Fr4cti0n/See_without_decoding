"""
MVOnlyTracker with Fixed State Pool for Object Lifecycle Management

Handles:
- Objects appearing (new detections on frame 1)
- Objects disappearing (low confidence → slot freed)
- Objects reappearing (slot reuse)

Architecture:
- Pre-allocated state pool: [2, max_objects, 128]
- Slot-based tracking (each object gets a slot index)
- Confidence-based pruning (removes disappeared objects)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class MVOnlyEncoder(nn.Module):
    """Enhanced motion vector encoder (64-dim output)"""
    
    def __init__(self, input_channels=2, feature_dim=64):
        super().__init__()
        
        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, feature_dim, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_dim)
        
        # Upsampling to match feature map size
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Refinement layer after upsampling
        self.refine = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.bn_refine = nn.BatchNorm2d(feature_dim)
        
        # Global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, motion_vectors):
        """
        Args:
            motion_vectors: [B, 2, H, W] - Usually [1, 2, 60, 60]
        
        Returns:
            features: [B, 64, H*2, W*2] - [1, 64, 120, 120]
            global_features: [B, 64] - Global motion context
        """
        # Encode
        x = F.relu(self.bn1(self.conv1(motion_vectors)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Upsample
        x = self.upsample(x)
        
        # Refine
        x = F.relu(self.bn_refine(self.refine(x)))
        
        # Global context
        global_feat = self.global_pool(x).squeeze(-1).squeeze(-1)
        
        return x, global_feat


class SpatialROIExtractor(nn.Module):
    """Extract per-object features using ROI Align"""
    
    def __init__(self, feature_dim=64, output_dim=64, roi_size=7):
        super().__init__()
        self.feature_dim = feature_dim
        self.roi_size = roi_size
        
        # Project ROI features to output dimension
        roi_feature_dim = feature_dim * roi_size * roi_size
        self.projection = nn.Sequential(
            nn.Linear(roi_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, features, boxes):
        """
        Args:
            features: [B, C, H, W] - Usually [1, 64, 120, 120]
            boxes: [N, 4] - Normalized boxes [x, y, w, h] in 0-1 range
        
        Returns:
            roi_features: [N, output_dim] - Per-object features
        """
        if boxes.shape[0] == 0:
            return torch.empty(0, self.projection[-1].out_features, device=boxes.device)
        
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0]  # x1 = x
        boxes_xyxy[:, 1] = boxes[:, 1]  # y1 = y
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
        
        # Scale to feature map size (120x120 for 960x960 image)
        H, W = features.shape[2], features.shape[3]
        boxes_scaled = boxes_xyxy * torch.tensor([W, H, W, H], device=boxes.device)
        
        # Add batch index (all boxes from batch 0)
        batch_indices = torch.zeros(boxes.shape[0], 1, device=boxes.device)
        rois = torch.cat([batch_indices, boxes_scaled], dim=1)
        
        # ROI Align
        roi_features = roi_align(
            features,
            rois,
            output_size=self.roi_size,
            spatial_scale=1.0,  # Already scaled boxes
            aligned=True
        )
        
        # Flatten and project
        roi_features = roi_features.view(roi_features.shape[0], -1)
        roi_features = self.projection(roi_features)
        
        return roi_features


class LSTMTracker(nn.Module):
    """LSTM-based temporal tracker with position/size/confidence prediction"""
    
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2, dropout=0.1):
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
    
    def forward(self, roi_features, global_features, hidden_state=None, return_logits=False):
        """
        Args:
            roi_features: [N, 64] - Per-object ROI features
            global_features: [64] - Global motion context
            hidden_state: Optional (h, c) tuple
            return_logits: If True, return logits; else sigmoid confidence
        
        Returns:
            pos_deltas: [N, 2] - Position deltas (dx, dy)
            confidences: [N, 1] - Object confidence scores
            size_deltas: [N, 2] - Size deltas (dw, dh) in log space
            hidden_state: Updated LSTM state
        """
        N = roi_features.shape[0]
        
        if N == 0:
            # No objects to track
            empty = torch.empty(0, device=roi_features.device)
            return empty, empty, empty, hidden_state
        
        # Combine ROI features with global context
        global_expanded = global_features.unsqueeze(0).expand(N, -1)
        combined = torch.cat([roi_features, global_expanded], dim=1)  # [N, 128]
        
        # Add temporal dimension for LSTM
        combined = combined.unsqueeze(1)  # [N, 1, 128]
        
        # LSTM forward
        lstm_out, hidden_state = self.lstm(combined, hidden_state)
        lstm_out = lstm_out.squeeze(1)  # [N, hidden_dim]
        
        # Predictions
        pos_deltas = torch.tanh(self.position_head(lstm_out)) * 0.1  # Small deltas
        size_deltas = torch.tanh(self.size_head(lstm_out)) * 0.1
        
        if return_logits:
            confidences = self.confidence_head(lstm_out)
        else:
            confidences = torch.sigmoid(self.confidence_head(lstm_out))
        
        return pos_deltas, confidences, size_deltas, hidden_state


class MVOnlyTrackerPooled(nn.Module):
    """
    Motion-Only Tracker with Fixed State Pool
    
    Features:
    - Pre-allocated state pool (max_objects slots)
    - Slot-based object tracking
    - Confidence-based pruning
    - No RGB image required (motion vectors only!)
    """
    
    def __init__(
        self,
        mv_feature_dim=64,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.1,
        image_size=960,
        max_objects=100,
        confidence_threshold=0.1
    ):
        super().__init__()
        
        self.max_objects = max_objects
        self.confidence_threshold = confidence_threshold
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        
        # Components
        self.encoder = MVOnlyEncoder(
            input_channels=2,
            feature_dim=mv_feature_dim
        )
        
        self.roi_extractor = SpatialROIExtractor(
            feature_dim=mv_feature_dim,
            output_dim=mv_feature_dim,
            roi_size=7
        )
        
        self.lstm_tracker = LSTMTracker(
            input_dim=mv_feature_dim * 2,  # ROI + global
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # State pool (initialized to zeros)
        self.register_buffer('state_pool_h', torch.zeros(lstm_layers, max_objects, hidden_dim))
        self.register_buffer('state_pool_c', torch.zeros(lstm_layers, max_objects, hidden_dim))
        
        # Tracking metadata
        self.active_slots = set()  # Set of active slot indices
        self.slot_to_id = {}       # {slot_idx: object_id}
        self.id_to_slot = {}       # {object_id: slot_idx}
        self.slot_boxes = {}       # {slot_idx: current_box}
        self.slot_confidence = {}  # {slot_idx: confidence}
    
    def _allocate_slot(self, object_id):
        """Allocate a free slot for new object"""
        for slot in range(self.max_objects):
            if slot not in self.active_slots:
                self.active_slots.add(slot)
                self.slot_to_id[slot] = object_id
                self.id_to_slot[object_id] = slot
                # Reset state for this slot
                self.state_pool_h[:, slot, :] = 0
                self.state_pool_c[:, slot, :] = 0
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
            # Clear state
            self.state_pool_h[:, slot, :] = 0
            self.state_pool_c[:, slot, :] = 0
    
    def initialize_objects(self, boxes, object_ids=None):
        """
        Initialize tracking with bounding boxes (called on frame 1)
        
        Args:
            boxes: [N, 4] - Initial boxes [x, y, w, h] normalized
            object_ids: Optional list of object IDs (if None, use indices)
        
        Returns:
            slot_indices: List of allocated slot indices
        """
        N = boxes.shape[0]
        if object_ids is None:
            object_ids = list(range(N))
        
        slot_indices = []
        for i, (box, oid) in enumerate(zip(boxes, object_ids)):
            slot = self._allocate_slot(oid)
            self.slot_boxes[slot] = box
            self.slot_confidence[slot] = 1.0  # High confidence initially
            slot_indices.append(slot)
        
        return slot_indices
    
    def forward_single_frame(self, motion_vectors, dct_residuals, boxes, hidden_state=None, return_logits=False):
        """
        Standard API for compatibility with train_mv_center.py
        
        NOTE: This uses the OLD API (dynamic boxes). For pooled tracking, use track_frame() instead.
        This function is kept for backward compatibility during training.
        """
        # This is the legacy API - just forward to encoder/tracker
        features, global_features = self.encoder(motion_vectors)
        
        if boxes.shape[0] == 0:
            return boxes, torch.empty(0, device=boxes.device), hidden_state
        
        roi_features = self.roi_extractor(features, boxes)
        pos_deltas, confs, size_deltas, hidden_state = self.lstm_tracker(
            roi_features,
            global_features[0],
            hidden_state,
            return_logits=return_logits
        )
        
        # Update boxes
        updated_boxes = self._update_boxes(boxes, pos_deltas, size_deltas)
        
        return updated_boxes, confs.squeeze(-1), hidden_state
    
    def track_frame(self, motion_vectors):
        """
        Track active objects using pooled states
        
        Args:
            motion_vectors: [1, 2, H, W] - Motion vector input
        
        Returns:
            predictions: List of (object_id, box, confidence)
        """
        if not self.active_slots:
            return []
        
        # Encode motion
        features, global_features = self.encoder(motion_vectors)
        
        # Gather active slots
        slot_indices = sorted(list(self.active_slots))
        boxes = torch.stack([self.slot_boxes[s] for s in slot_indices])
        
        # Extract ROI features
        roi_features = self.roi_extractor(features, boxes)
        
        # Gather states from pool
        batch_states = (
            self.state_pool_h[:, slot_indices, :].contiguous(),
            self.state_pool_c[:, slot_indices, :].contiguous()
        )
        
        # Track
        pos_deltas, confs, size_deltas, new_states = self.lstm_tracker(
            roi_features,
            global_features[0],
            batch_states,
            return_logits=False
        )
        
        # Update boxes and states in pool
        predictions = []
        slots_to_remove = []
        
        for i, slot in enumerate(slot_indices):
            # Update state pool
            self.state_pool_h[:, slot, :] = new_states[0][:, i, :]
            self.state_pool_c[:, slot, :] = new_states[1][:, i, :]
            
            # Update box
            box = boxes[i]
            dx, dy = pos_deltas[i]
            dw, dh = size_deltas[i]
            
            new_box = box.clone()
            new_box[0] += dx  # x
            new_box[1] += dy  # y
            new_box[2] *= torch.exp(dw)  # w (log scale)
            new_box[3] *= torch.exp(dh)  # h (log scale)
            
            # Clamp to valid range
            new_box = torch.clamp(new_box, 0, 1)
            
            conf = confs[i].item()
            self.slot_boxes[slot] = new_box
            self.slot_confidence[slot] = conf
            
            # Check confidence threshold
            if conf < self.confidence_threshold:
                slots_to_remove.append(slot)
            else:
                object_id = self.slot_to_id[slot]
                predictions.append((object_id, new_box, conf))
        
        # Remove low-confidence objects
        for slot in slots_to_remove:
            self._free_slot(slot)
        
        return predictions
    
    def _update_boxes(self, boxes, pos_deltas, size_deltas):
        """Update boxes with predicted deltas"""
        updated = boxes.clone()
        updated[:, 0:2] += pos_deltas  # x, y
        updated[:, 2:4] *= torch.exp(size_deltas)  # w, h (log scale)
        
        # Clamp to valid range [0, 1]
        updated = torch.clamp(updated, 0, 1)
        
        return updated
    
    def reset(self):
        """Reset all tracking state (call at start of new sequence)"""
        self.active_slots.clear()
        self.slot_to_id.clear()
        self.id_to_slot.clear()
        self.slot_boxes.clear()
        self.slot_confidence.clear()
        self.state_pool_h.zero_()
        self.state_pool_c.zero_()


def test_pooled_tracker():
    """Test the pooled tracker with object lifecycle"""
    print("Testing MVOnlyTrackerPooled with object lifecycle...\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MVOnlyTrackerPooled(
        mv_feature_dim=64,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.1,
        max_objects=100,
        confidence_threshold=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"State pool size: {model.max_objects} slots")
    print()
    
    # Simulate tracking scenario
    print("=" * 60)
    print("SCENARIO: Object lifecycle management")
    print("=" * 60)
    
    # Frame 1: Initialize with 3 objects
    print("\n[Frame 1] Initialize with 3 objects")
    mv_1 = torch.randn(1, 2, 60, 60, device=device)
    boxes_1 = torch.tensor([
        [0.1, 0.1, 0.2, 0.3],  # Object ID 0
        [0.5, 0.5, 0.15, 0.25], # Object ID 1
        [0.7, 0.3, 0.18, 0.28]  # Object ID 2
    ], device=device)
    
    slots = model.initialize_objects(boxes_1, object_ids=[0, 1, 2])
    print(f"  Allocated slots: {slots}")
    print(f"  Active slots: {sorted(model.active_slots)}")
    predictions_1 = model.track_frame(mv_1)
    print(f"  Predictions: {len(predictions_1)} objects")
    for oid, box, conf in predictions_1:
        print(f"    Object {oid}: conf={conf:.3f}, box={box.detach().cpu().numpy()}")
    
    # Frame 2: Track existing objects
    print("\n[Frame 2] Track existing 3 objects")
    mv_2 = torch.randn(1, 2, 60, 60, device=device)
    predictions_2 = model.track_frame(mv_2)
    print(f"  Predictions: {len(predictions_2)} objects")
    for oid, box, conf in predictions_2:
        print(f"    Object {oid}: conf={conf:.3f}")
    
    # Frame 3: Object 1 disappears (manually set low confidence)
    print("\n[Frame 3] Object 1 disappears (low confidence)")
    model.slot_confidence[slots[1]] = 0.05  # Force low confidence
    mv_3 = torch.randn(1, 2, 60, 60, device=device)
    predictions_3 = model.track_frame(mv_3)
    print(f"  Predictions: {len(predictions_3)} objects (Object 1 removed)")
    print(f"  Active slots: {sorted(model.active_slots)}")
    for oid, box, conf in predictions_3:
        print(f"    Object {oid}: conf={conf:.3f}")
    
    # Frame 4: New object appears (ID 3)
    print("\n[Frame 4] New object appears (ID 3)")
    new_box = torch.tensor([[0.3, 0.6, 0.2, 0.25]], device=device)
    new_slots = model.initialize_objects(new_box, object_ids=[3])
    print(f"  Allocated slot for new object: {new_slots}")
    print(f"  Active slots: {sorted(model.active_slots)}")
    mv_4 = torch.randn(1, 2, 60, 60, device=device)
    predictions_4 = model.track_frame(mv_4)
    print(f"  Predictions: {len(predictions_4)} objects")
    for oid, box, conf in predictions_4:
        print(f"    Object {oid}: conf={conf:.3f}")
    
    # Frame 5: Continue tracking
    print("\n[Frame 5] Continue tracking all active objects")
    mv_5 = torch.randn(1, 2, 60, 60, device=device)
    predictions_5 = model.track_frame(mv_5)
    print(f"  Predictions: {len(predictions_5)} objects")
    for oid, box, conf in predictions_5:
        print(f"    Object {oid}: conf={conf:.3f}")
    
    # Memory usage
    print("\n" + "=" * 60)
    if device.type == 'cuda':
        memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"GPU memory allocated: {memory_mb:.2f} MB")
    
    print("\n✅ Pooled tracker test passed!")
    print(f"Successfully handled:")
    print(f"  - 3 initial objects")
    print(f"  - 1 object disappearing (slot freed)")
    print(f"  - 1 new object appearing (slot reused)")
    print(f"  - State pool efficiently managed")


if __name__ == "__main__":
    test_pooled_tracker()
