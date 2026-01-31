"""
Fast DCT-MV Tracker - Simplified architecture without ROI extraction or attention
Focus: Maximum speed with minimal accuracy loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components.dct_mv_encoder import SpatiallyAlignedDCTMVEncoder


class FastDCTMVTracker(nn.Module):
    """
    Simplified tracker without ROI extraction or attention mechanisms.
    
    Key simplifications:
    1. No ROI extraction - use global average pooling instead
    2. No attention mechanisms
    3. Simpler LSTM with fewer layers
    4. Direct box coordinate regression
    
    Architecture:
    1. Encode MV + DCT features → [B, C, H, W]
    2. Global pooling → [B, C]
    3. Simple MLP to process box inputs
    4. Concat features + box embeddings
    5. LSTM for temporal modeling
    6. Direct regression of position/size/confidence
    """
    
    def __init__(
        self,
        num_dct_coeffs=16,
        mv_channels=2,
        dct_channels=64,
        mv_feature_dim=32,
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
        self.fused_feature_dim = fused_feature_dim
        self.mv_channels = mv_channels
        self.dct_channels = dct_channels
        self.use_mv = mv_channels > 0
        self.use_dct = dct_channels > 0
        self.box_embedding_dim = box_embedding_dim
        
        # Component 1: Spatially aligned encoder (reuse existing)
        self.encoder = SpatiallyAlignedDCTMVEncoder(
            num_dct_coeffs=num_dct_coeffs,
            mv_channels=mv_channels,
            dct_channels=dct_channels,
            mv_feature_dim=mv_feature_dim,
            dct_feature_dim=dct_feature_dim,
            fused_feature_dim=fused_feature_dim
        )
        
        # Component 2: Simple box embedding (replace ROI extraction)
        # Encode box coordinates [x1, y1, x2, y2] → embedding
        self.box_encoder = nn.Sequential(
            nn.Linear(4, box_embedding_dim),
            nn.LayerNorm(box_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(box_embedding_dim, box_embedding_dim),
            nn.LayerNorm(box_embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Component 3: Simplified LSTM tracker
        # Input: global features + box embedding
        lstm_input_dim = fused_feature_dim + box_embedding_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Component 4: Output heads (simpler than before)
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
        Process a single frame (main inference method)
        
        Args:
            motion_vectors: [1, 2, 60, 60] or None
            dct_residuals: [1, 120, 120, C] or None (C = num coefficients)
            boxes: [N, 4] - Bounding boxes in image coordinates [0, image_size]
            hidden_state: Optional LSTM state tuple (h, c)
            return_logits: If True, return raw logits; if False, return probabilities
        
        Returns:
            pred_boxes: [N, 4] - Updated bounding boxes (SAME FORMAT AS STANDARD MODEL)
            confidences: [N] - Confidence scores (flattened to match standard model)
            hidden_state: Updated LSTM state
        """
        N = boxes.shape[0]
        device = boxes.device
        
        # Step 1: Encode spatial features
        fused_features, _ = self.encoder(motion_vectors, dct_residuals)
        # fused_features: [1, C, H, W]
        
        # Step 2: Global pooling (replace ROI extraction)
        # Use adaptive avg pooling to get global context
        global_features = F.adaptive_avg_pool2d(fused_features, 1)  # [1, C, 1, 1]
        global_features = global_features.view(1, -1)  # [1, C]
        
        # Expand global features for each object
        global_features = global_features.expand(N, -1)  # [N, C]
        
        # Step 3: Encode box coordinates
        # Normalize boxes to [0, 1]
        boxes_norm = boxes.clone()
        boxes_norm = boxes_norm / self.image_size
        box_features = self.box_encoder(boxes_norm)  # [N, box_embedding_dim]
        
        # Step 4: Concatenate features
        combined_features = torch.cat([global_features, box_features], dim=-1)
        # [N, fused_feature_dim + box_embedding_dim]
        
        # Step 5: LSTM temporal modeling
        combined_features = combined_features.unsqueeze(1)  # [N, 1, input_dim]
        lstm_out, hidden_state = self.lstm(combined_features, hidden_state)
        lstm_out = lstm_out.squeeze(1)  # [N, hidden_dim]
        
        # Step 6: Predict outputs
        pos_deltas = self.position_head(lstm_out)  # [N, 2]
        size_deltas = self.size_head(lstm_out)     # [N, 2]
        conf_logits = self.confidence_head(lstm_out)  # [N, 1]
        
        # Apply sigmoid to confidence
        if return_logits:
            confidences = conf_logits.squeeze(-1)  # [N]
        else:
            confidences = torch.sigmoid(conf_logits).squeeze(-1)  # [N]
        
        # Step 7: Update boxes (same as standard model)
        pred_boxes = self._update_boxes(boxes, pos_deltas, size_deltas)
        
        # Return same format as standard model: (pred_boxes, confidences, hidden_state)
        return pred_boxes, confidences, hidden_state
    
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
            position_deltas: [T, N, 2] - Position updates
        """
        T = motion_vectors_seq.shape[0] if motion_vectors_seq is not None else dct_residuals_seq.shape[0]
        N = initial_boxes.shape[0]
        
        # Initialize tracking state
        current_boxes = initial_boxes.clone()
        hidden_state = None
        
        tracked_boxes_list = []
        confidences_list = []
        deltas_list = []
        
        # Process each frame
        for t in range(T):
            # Get current frame data
            motion_vectors = motion_vectors_seq[t:t+1] if motion_vectors_seq is not None else None
            dct_residuals = dct_residuals_seq[t:t+1] if dct_residuals_seq is not None else None
            
            # Forward pass
            pos_deltas, confs, size_deltas, hidden_state = self.forward_single_frame(
                motion_vectors,
                dct_residuals,
                current_boxes,
                hidden_state,
                return_logits=False
            )
            
            # Update box positions
            updated_boxes = self._update_boxes(current_boxes, pos_deltas, size_deltas)
            
            # Store results
            tracked_boxes_list.append(updated_boxes)
            confidences_list.append(confs.squeeze(-1))
            deltas_list.append(pos_deltas)
            
            # Update state for next frame
            current_boxes = updated_boxes
        
        # Stack results
        tracked_boxes = torch.stack(tracked_boxes_list, dim=0)
        confidences = torch.stack(confidences_list, dim=0)
        position_deltas = torch.stack(deltas_list, dim=0)
        
        return tracked_boxes, confidences, position_deltas
    
    def _update_boxes(self, boxes, pos_deltas, size_deltas):
        """
        Update bounding boxes with predicted deltas
        
        Args:
            boxes: [N, 4] - Current boxes (x1, y1, x2, y2)
            pos_deltas: [N, 2] - Position updates (dx, dy)
            size_deltas: [N, 2] - Size updates (dw, dh)
        
        Returns:
            updated_boxes: [N, 4] - Updated boxes
        """
        # Extract box properties
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Apply position deltas (scaled by box size)
        cx_new = cx + pos_deltas[:, 0] * w
        cy_new = cy + pos_deltas[:, 1] * h
        
        # Apply size deltas (exponential scaling)
        w_new = w * torch.exp(size_deltas[:, 0].clamp(-1, 1))
        h_new = h * torch.exp(size_deltas[:, 1].clamp(-1, 1))
        
        # Convert back to corner format
        x1_new = cx_new - w_new / 2
        y1_new = cy_new - h_new / 2
        x2_new = cx_new + w_new / 2
        y2_new = cy_new + h_new / 2
        
        # Clamp to image boundaries
        x1_new = torch.clamp(x1_new, 0, self.image_size)
        y1_new = torch.clamp(y1_new, 0, self.image_size)
        x2_new = torch.clamp(x2_new, 0, self.image_size)
        y2_new = torch.clamp(y2_new, 0, self.image_size)
        
        updated_boxes = torch.stack([x1_new, y1_new, x2_new, y2_new], dim=-1)
        return updated_boxes


class UltraFastDCTMVTracker(nn.Module):
    """
    Even more simplified version without LSTM for maximum speed.
    Uses only feedforward network for tracking.
    """
    
    def __init__(
        self,
        num_dct_coeffs=16,
        mv_channels=2,
        dct_channels=64,
        mv_feature_dim=32,
        dct_feature_dim=32,
        fused_feature_dim=64,
        hidden_dim=128,
        dropout=0.1,
        image_size=960,
        box_embedding_dim=32
    ):
        super().__init__()
        
        self.image_size = image_size
        self.fused_feature_dim = fused_feature_dim
        self.box_embedding_dim = box_embedding_dim
        
        # Encoder
        self.encoder = SpatiallyAlignedDCTMVEncoder(
            num_dct_coeffs=num_dct_coeffs,
            mv_channels=mv_channels,
            dct_channels=dct_channels,
            mv_feature_dim=mv_feature_dim,
            dct_feature_dim=dct_feature_dim,
            fused_feature_dim=fused_feature_dim
        )
        
        # Box encoder
        self.box_encoder = nn.Sequential(
            nn.Linear(4, box_embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        # Simple feedforward tracker (no LSTM)
        combined_dim = fused_feature_dim + box_embedding_dim
        self.tracker = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Output heads
        self.position_head = nn.Linear(hidden_dim // 2, 2)
        self.size_head = nn.Linear(hidden_dim // 2, 2)
        self.confidence_head = nn.Linear(hidden_dim // 2, 1)
    
    def forward_single_frame(self, motion_vectors, dct_residuals, boxes, hidden_state=None, return_logits=False):
        """Process single frame without temporal modeling"""
        N = boxes.shape[0]
        
        # Encode features
        fused_features, _ = self.encoder(motion_vectors, dct_residuals)
        global_features = F.adaptive_avg_pool2d(fused_features, 1).view(1, -1)
        global_features = global_features.expand(N, -1)
        
        # Encode boxes
        boxes_norm = boxes / self.image_size
        box_features = self.box_encoder(boxes_norm)
        
        # Combine and track
        combined = torch.cat([global_features, box_features], dim=-1)
        features = self.tracker(combined)
        
        # Predictions
        pos_deltas = self.position_head(features)
        size_deltas = self.size_head(features)
        conf_logits = self.confidence_head(features)
        
        confidences = conf_logits if return_logits else torch.sigmoid(conf_logits)
        
        return pos_deltas, confidences, size_deltas, None  # No hidden state
