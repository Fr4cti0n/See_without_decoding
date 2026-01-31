"""
Improved Fast DCT-MV Tracker - Box-Aligned Motion Features

Key improvements over original FastDCTMVTracker:
1. ✅ Uses BoxAlignedMotionEncoder instead of global pooling
2. ✅ Extracts box-specific motion features (mean, std, range per box)
3. ✅ Respects 16×16 macroblock structure (no fake interpolation)
4. ✅ Each box gets DIFFERENT motion features (not shared global features)

This should perform BETTER than Mean-VC baseline because:
- Learned aggregation of motion statistics
- Temporal modeling with LSTM
- Non-linear motion pattern learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import BoxAlignedMotionEncoder from parent models directory
from ..tracking_propagator import BoxAlignedMotionEncoder


class ImprovedFastDCTMVTracker(nn.Module):
    """
    Improved fast tracker with box-aligned motion features.
    
    Architecture changes from original FastDCTMVTracker:
    
    OLD (BROKEN):
    ┌─────────────────────────────────────────────────────────┐
    │ MVs [60×60×2] → CNN → Global Pool → [64]               │
    │ All boxes share SAME global feature ❌                  │
    └─────────────────────────────────────────────────────────┘
    
    NEW (FIXED):
    ┌─────────────────────────────────────────────────────────┐
    │ MVs [60×60×2] → BoxAlignedEncoder                       │
    │   ├→ For Box 1: Extract region MVs → Stats → [64]      │
    │   ├→ For Box 2: Extract region MVs → Stats → [64]      │
    │   └→ For Box N: Extract region MVs → Stats → [64]      │
    │ Each box gets DIFFERENT features ✅                      │
    └─────────────────────────────────────────────────────────┘
    
    Key modifications:
    1. Replace global pooling with BoxAlignedMotionEncoder.extract_box_motion_features()
    2. Add per-box motion feature extraction
    3. Keep rest of architecture identical (LSTM, heads)
    """
    
    def __init__(
        self,
        num_dct_coeffs=16,
        mv_channels=2,
        dct_channels=0,  # For MV-only ablation
        mv_feature_dim=64,  # Match BoxAlignedMotionEncoder output
        dct_feature_dim=32,
        fused_feature_dim=64,
        hidden_dim=128,
        lstm_layers=1,
        dropout=0.1,
        image_size=960,
        box_embedding_dim=32
    ):
        super().__init__()
        
        self.image_size = image_size
        self.mv_feature_dim = mv_feature_dim
        self.mv_channels = mv_channels
        self.dct_channels = dct_channels
        self.use_mv = mv_channels > 0
        self.use_dct = dct_channels > 0
        self.box_embedding_dim = box_embedding_dim
        
        print("\n" + "="*80)
        print("🚀 IMPROVED FAST TRACKER - Box-Aligned Motion Features")
        print("="*80)
        print(f"✅ Using BoxAlignedMotionEncoder (no global pooling!)")
        print(f"✅ Per-box motion features: mean, std, range")
        print(f"✅ Respects 16×16 macroblock structure")
        print(f"   - MV channels: {mv_channels}")
        print(f"   - MV feature dim: {mv_feature_dim}")
        print(f"   - DCT channels: {dct_channels}")
        print(f"   - Hidden dim: {hidden_dim}")
        print("="*80 + "\n")
        
        # ✅ MODIFICATION 1: Replace global pooling with box-aligned encoder
        if self.use_mv:
            self.mv_encoder = BoxAlignedMotionEncoder(
                mv_feature_dim=mv_feature_dim,
                image_size=image_size
            )
            # Add 'encoder' attribute for training code compatibility
            self.encoder = self.mv_encoder  # ✅ Makes is_dct_mv_tracker = True
            print(f"✅ BoxAlignedMotionEncoder initialized (feature_dim={mv_feature_dim})")
        else:
            self.mv_encoder = None
            self.encoder = None
            print(f"⚠️  No motion vectors (MV disabled)")
        
        # DCT encoder (if needed - for MV-only we set dct_channels=0)
        if self.use_dct:
            # Simple DCT encoder (placeholder - not used in MV-only ablation)
            self.dct_encoder = nn.Sequential(
                nn.Conv2d(dct_channels, dct_feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dct_feature_dim),
                nn.ReLU(inplace=True)
            )
            feature_dim = mv_feature_dim + dct_feature_dim if self.use_mv else dct_feature_dim
        else:
            self.dct_encoder = None
            feature_dim = mv_feature_dim
        
        # ✅ MODIFICATION 2: Box encoder (coordinates only - no global pooling)
        self.box_encoder = nn.Sequential(
            nn.Linear(4, box_embedding_dim),
            nn.LayerNorm(box_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(box_embedding_dim, box_embedding_dim),
            nn.LayerNorm(box_embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # ✅ MODIFICATION 3: LSTM input is now box-specific features + box embedding
        lstm_input_dim = feature_dim + box_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )
        
        print(f"✅ LSTM input dim: {lstm_input_dim} (motion_feat={feature_dim} + box_embed={box_embedding_dim})")
        
        # Output heads (unchanged)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # (dx, dy)
        )
        
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # (dw, dh)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # confidence score
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward_single_frame(self, motion_vectors, dct_residuals, boxes, hidden_state=None, return_logits=False):
        """
        Process a single frame with BOX-ALIGNED features
        
        Args:
            motion_vectors: [1, 2, 60, 60] or None
            dct_residuals: [1, 120, 120, C] or None
            boxes: [N, 4] - Bounding boxes in PIXEL coordinates [0, image_size] (x1, y1, x2, y2)
            hidden_state: Optional LSTM state tuple (h, c)
            return_logits: If True, return raw logits; if False, return probabilities
        
        Returns:
            pred_boxes: [N, 4] - Updated bounding boxes in PIXEL coordinates
            confidences: [N] - Confidence scores
            hidden_state: Updated LSTM state
        """
        N = boxes.shape[0]
        device = boxes.device
        
        if N == 0:
            # Handle empty box case
            return (
                torch.zeros(0, 4, device=device),
                torch.zeros(0, device=device),
                hidden_state
            )
        
        # ✅ CRITICAL CHANGE: Extract box-specific motion features
        # OLD: global_features = F.adaptive_avg_pool2d(encoded, 1)  ❌
        # NEW: box_motion_features = extract per-box statistics  ✅
        
        # ✅ Normalize boxes to [0,1] for BoxAlignedMotionEncoder
        # Boxes come in as PIXELS [0, 960], need to normalize for encoder
        boxes_norm = boxes.clone() / self.image_size
        boxes_norm = torch.clamp(boxes_norm, 0, 1)  # Ensure valid range
        
        # 🔍 DIAGNOSTIC: Print input shapes and values
        if N > 0 and torch.rand(1).item() < 0.01:  # Print 1% of batches
            print(f"\n🔍 [ImprovedFastTracker.forward_single_frame]")
            print(f"   Boxes (pixel):  shape={boxes.shape}, range=[{boxes.min():.2f}, {boxes.max():.2f}]")
            print(f"   Boxes (norm):   shape={boxes_norm.shape}, range=[{boxes_norm.min():.4f}, {boxes_norm.max():.4f}]")
            if motion_vectors is not None:
                print(f"   Motion vectors: shape={motion_vectors.shape}, range=[{motion_vectors.min():.2f}, {motion_vectors.max():.2f}]")
            if dct_residuals is not None:
                print(f"   DCT residuals:  shape={dct_residuals.shape}, range=[{dct_residuals.min():.2f}, {dct_residuals.max():.2f}]")
        
        # Extract motion features (different for each box!)
        if self.use_mv and motion_vectors is not None:
            # This is the KEY: extract_box_motion_features computes
            # mean, std, range PER BOX (not global pooling!)
            box_motion_features = self.mv_encoder.extract_box_motion_features(
                motion_vectors,  # [1, 2, 60, 60]
                boxes_norm       # [N, 4] in [0, 1]
            )  # Returns: [N, mv_feature_dim]
            
            # 🔍 DIAGNOSTIC: Verify box-specific features
            if N > 0 and torch.rand(1).item() < 0.01:
                print(f"   Motion features: shape={box_motion_features.shape}, range=[{box_motion_features.min():.4f}, {box_motion_features.max():.4f}]")
                if N > 1:
                    # Check that different boxes have different features (not global)
                    feat_diff = (box_motion_features[0] - box_motion_features[1]).abs().mean()
                    print(f"   Feature difference (box 0 vs 1): {feat_diff:.6f} {'✅ DIFFERENT' if feat_diff > 0.01 else '⚠️ SAME (global pooling?)'}")
        else:
            box_motion_features = torch.zeros(N, self.mv_feature_dim, device=device)
        
        # DCT features (if used)
        if self.use_dct and dct_residuals is not None:
            # Similar box-aligned extraction for DCT (not implemented in MV-only)
            # For now, use global pooling (since we're doing MV-only ablation)
            dct_feat = torch.zeros(N, 0, device=device)  # Empty for MV-only
        else:
            dct_feat = torch.zeros(N, 0, device=device)
        
        # Concatenate motion features (if DCT is used)
        if dct_feat.shape[1] > 0:
            visual_features = torch.cat([box_motion_features, dct_feat], dim=-1)
        else:
            visual_features = box_motion_features  # [N, mv_feature_dim]
        
        # Encode box coordinates
        box_coord_features = self.box_encoder(boxes_norm)  # [N, box_embedding_dim]
        
        # Combine: box-specific motion + box coordinates
        combined_features = torch.cat([visual_features, box_coord_features], dim=-1)
        # [N, mv_feature_dim + box_embedding_dim]
        
        # LSTM temporal modeling
        combined_features = combined_features.unsqueeze(1)  # [N, 1, input_dim]
        lstm_out, hidden_state = self.lstm(combined_features, hidden_state)
        lstm_out = lstm_out.squeeze(1)  # [N, hidden_dim]
        
        # Predict outputs
        pos_deltas = self.position_head(lstm_out)  # [N, 2]
        size_deltas = self.size_head(lstm_out)     # [N, 2]
        conf_logits = self.confidence_head(lstm_out)  # [N, 1]
        
        # 🔍 DIAGNOSTIC: Print predictions
        if N > 0 and torch.rand(1).item() < 0.01:
            print(f"   Position deltas: range=[{pos_deltas.min():.2f}, {pos_deltas.max():.2f}], mean={pos_deltas.mean():.4f}")
            print(f"   Size deltas:     range=[{size_deltas.min():.2f}, {size_deltas.max():.2f}], mean={size_deltas.mean():.4f}")
            print(f"   Confidence logits: range=[{conf_logits.min():.2f}, {conf_logits.max():.2f}], mean={conf_logits.mean():.4f}")
        
        # Apply sigmoid to confidence
        if return_logits:
            confidences = conf_logits.squeeze(-1)  # [N]
        else:
            confidences = torch.sigmoid(conf_logits).squeeze(-1)  # [N]
        
        # Update boxes
        pred_boxes = self._update_boxes(boxes, pos_deltas, size_deltas)
        
        # 🔍 DIAGNOSTIC: Print final boxes
        if N > 0 and torch.rand(1).item() < 0.01:
            print(f"   Output boxes: shape={pred_boxes.shape}, range=[{pred_boxes.min():.2f}, {pred_boxes.max():.2f}]")
            if N > 0:
                # Check if boxes can move/resize
                box_movement = (pred_boxes - boxes).abs().mean()
                print(f"   Box movement: {box_movement:.4f} pixels/box\n")
        
        return pred_boxes, confidences, hidden_state
    
    def _update_boxes(self, boxes, pos_deltas, size_deltas):
        """
        Update box coordinates using predicted deltas
        
        Args:
            boxes: [N, 4] - Current boxes [x1, y1, x2, y2]
            pos_deltas: [N, 2] - Position deltas [dx, dy]
            size_deltas: [N, 2] - Size deltas [dw, dh]
        
        Returns:
            updated_boxes: [N, 4] - Updated boxes
        """
        # Convert to center format
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        
        # Apply deltas
        new_cx = cx + pos_deltas[:, 0]
        new_cy = cy + pos_deltas[:, 1]
        new_w = w * torch.exp(size_deltas[:, 0])  # Exponential for scale changes
        new_h = h * torch.exp(size_deltas[:, 1])
        
        # Convert back to corner format
        new_x1 = new_cx - new_w / 2
        new_y1 = new_cy - new_h / 2
        new_x2 = new_cx + new_w / 2
        new_y2 = new_cy + new_h / 2
        
        # Clamp to valid pixel range [0, image_size]
        new_x1 = torch.clamp(new_x1, 0, self.image_size)
        new_y1 = torch.clamp(new_y1, 0, self.image_size)
        new_x2 = torch.clamp(new_x2, 0, self.image_size)
        new_y2 = torch.clamp(new_y2, 0, self.image_size)
        
        updated_boxes = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
        return updated_boxes
    
    def forward(self, motion_vectors_seq, dct_residuals_seq, initial_boxes):
        """
        Track objects across a sequence of frames
        
        Args:
            motion_vectors_seq: [T, 2, 60, 60] or None
            dct_residuals_seq: [T, 120, 120, C] or None
            initial_boxes: [N, 4] - Initial bounding boxes
        
        Returns:
            tracked_boxes: [T, N, 4] - Tracked boxes for all frames
            confidences: [T, N] - Confidence scores
            position_deltas: [T, N, 2] - Position updates (for analysis)
        """
        T = motion_vectors_seq.shape[0] if motion_vectors_seq is not None else dct_residuals_seq.shape[0]
        N = initial_boxes.shape[0]
        device = initial_boxes.device
        
        tracked_boxes = []
        confidences_seq = []
        position_deltas_seq = []
        
        hidden_state = None
        current_boxes = initial_boxes
        
        for t in range(T):
            # Get current frame data
            mvs = motion_vectors_seq[t:t+1] if motion_vectors_seq is not None else None
            dct = dct_residuals_seq[t:t+1] if dct_residuals_seq is not None else None
            
            # Track
            current_boxes, confs, hidden_state = self.forward_single_frame(
                mvs, dct, current_boxes, hidden_state
            )
            
            # Store results
            tracked_boxes.append(current_boxes)
            confidences_seq.append(confs)
        
        # Stack results
        tracked_boxes = torch.stack(tracked_boxes)  # [T, N, 4]
        confidences_seq = torch.stack(confidences_seq)  # [T, N]
        
        return tracked_boxes, confidences_seq, None  # position_deltas not tracked
