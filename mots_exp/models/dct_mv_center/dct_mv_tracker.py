"""
DCT-MV Center-based Tracker
Main model combining all components for object tracking with DCT residuals
"""

import torch
import torch.nn as nn
from .components.dct_mv_encoder import SpatiallyAlignedDCTMVEncoder
from .components.spatial_roi_extractor import SpatialROIExtractor
from .components.multiscale_roi_extractor import MultiScaleROIExtractor, EfficientMultiScaleROIExtractor
from .components.parallel_multiscale_roi_extractor import ParallelMultiScaleROIExtractor, ParallelMultiScaleROIExtractorV2
from .components.parallel_heads_with_attention import (
    ParallelHeadsWithAttention,
    LightweightParallelHeadsWithAttention
)
from .components.lstm_tracker import LSTMTracker


class DCTMVCenterTracker(nn.Module):
    """
    Complete tracking model using DCT residuals and motion vectors
    
    Architecture:
    1. Encode MV + DCT with spatial alignment
    2. Extract ROI features for each tracked object
    3. LSTM-based temporal tracking
    4. Predict position updates and confidences
    """
    def __init__(
        self,
        num_dct_coeffs=16,
        mv_channels=2,                # Number of MV input channels (0 to disable)
        dct_channels=64,              # Number of DCT input channels (0 to disable)
        mv_feature_dim=32,
        dct_feature_dim=32,
        fused_feature_dim=64,
        roi_feature_dim=64,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.1,
        image_size=960,
        use_multiscale_roi=False,    # Enable multi-scale ROI
        roi_sizes=[3, 7, 11],         # ROI scales to use
        use_parallel_heads=False,     # Use parallel heads (no compression!)
        use_attention=False,          # NEW: Add attention to parallel heads
        attention_heads=4             # Number of attention heads
    ):
        super().__init__()
        
        self.image_size = image_size
        self.fused_feature_dim = fused_feature_dim
        self.mv_channels = mv_channels
        self.dct_channels = dct_channels
        self.use_mv = mv_channels > 0
        self.use_dct = dct_channels > 0
        self.use_multiscale_roi = use_multiscale_roi
        self.use_parallel_heads = use_parallel_heads
        self.use_attention = use_attention
        
        # Component 1: Spatially aligned encoder
        self.encoder = SpatiallyAlignedDCTMVEncoder(
            num_dct_coeffs=num_dct_coeffs,
            mv_channels=mv_channels,
            dct_channels=dct_channels,
            mv_feature_dim=mv_feature_dim,
            dct_feature_dim=dct_feature_dim,
            fused_feature_dim=fused_feature_dim
        )
        
        # Component 2: ROI extractor (single-scale, multi-scale, parallel heads, or attention)
        if use_parallel_heads and use_attention:
            print(f"   🎯 Using PARALLEL HEADS + ATTENTION with scales {roi_sizes}")
            self.roi_extractor = ParallelHeadsWithAttention(
                feature_dim=fused_feature_dim,
                roi_sizes=roi_sizes,
                output_dim=roi_feature_dim,
                image_size=image_size,
                num_attention_heads=attention_heads,
                use_scale_attention=True,
                use_cross_scale_attention=True,
                dropout=dropout
            )
        elif use_parallel_heads:
            print(f"   🚀 Using PARALLEL HEADS Multi-Scale ROI with scales {roi_sizes}")
            self.roi_extractor = ParallelMultiScaleROIExtractor(
                feature_dim=fused_feature_dim,
                roi_sizes=roi_sizes,
                output_dim=roi_feature_dim,
                image_size=image_size,
                use_global_context=True,
                learnable_weights=True
            )
        elif use_multiscale_roi:
            print(f"   🔍 Using Multi-Scale ROI Extractor with scales {roi_sizes}")
            self.roi_extractor = EfficientMultiScaleROIExtractor(
                feature_dim=fused_feature_dim,
                roi_sizes=roi_sizes,
                output_dim=roi_feature_dim,
                image_size=image_size,
                use_global_context=True
            )
        else:
            self.roi_extractor = SpatialROIExtractor(
                feature_dim=fused_feature_dim,
                roi_size=7,
                output_dim=roi_feature_dim,
                image_size=image_size  # ✅ Pass image size for proper normalization
            )
        
        # Component 3: LSTM tracker
        self.lstm_tracker = LSTMTracker(
            roi_feature_dim=roi_feature_dim,
            global_feature_dim=fused_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout
        )
    
    def forward(self, motion_vectors_seq, dct_residuals_seq, initial_boxes):
        """
        Track objects across a sequence of frames
        
        Args:
            motion_vectors_seq: [T, 2, 60, 60] - Motion vector sequence
            dct_residuals_seq: [T, 120, 120, 64] - DCT residual sequence
            initial_boxes: [N, 4] - Initial bounding boxes (x1, y1, x2, y2)
        
        Returns:
            tracked_boxes: [T, N, 4] - Tracked boxes for all frames
            confidences: [T, N] - Confidence scores
            position_deltas: [T, N, 2] - Position updates
        """
        T = motion_vectors_seq.shape[0]
        N = initial_boxes.shape[0]
        device = motion_vectors_seq.device
        
        # Initialize tracking state
        current_boxes = initial_boxes.clone()
        hidden_state = None
        
        tracked_boxes_list = []
        confidences_list = []
        deltas_list = []
        
        # Process each frame
        for t in range(T):
            # Get current frame data
            motion_vectors = motion_vectors_seq[t:t+1]  # [1, 2, 60, 60]
            dct_residuals = dct_residuals_seq[t:t+1]    # [1, 120, 120, 64]
            
            # Encode features
            fused_features, global_features = self.encoder(motion_vectors, dct_residuals)
            # fused_features: [1, 64, 120, 120]
            # global_features: [1, 64]
            
            # Extract ROI features
            if self.use_attention:
                roi_features, attention_info = self.roi_extractor(fused_features, current_boxes)
            else:
                roi_features = self.roi_extractor(fused_features, current_boxes)
            # roi_features: [N, 64]
            
            # Track objects
            pos_deltas, confs, size_deltas, hidden_state = self.lstm_tracker(
                roi_features,
                global_features[0],  # [64]
                hidden_state
            )
            # pos_deltas: [N, 2], confs: [N, 1], size_deltas: [N, 2]
            
            # Update box positions
            updated_boxes = self._update_boxes(current_boxes, pos_deltas, size_deltas)
            
            # Store results
            tracked_boxes_list.append(updated_boxes)
            confidences_list.append(confs.squeeze(-1))
            deltas_list.append(pos_deltas)
            
            # Update state for next frame
            current_boxes = updated_boxes
        
        # Stack results
        tracked_boxes = torch.stack(tracked_boxes_list, dim=0)  # [T, N, 4]
        confidences = torch.stack(confidences_list, dim=0)      # [T, N]
        position_deltas = torch.stack(deltas_list, dim=0)       # [T, N, 2]
        
        return tracked_boxes, confidences, position_deltas
    
    def forward_single_frame(self, motion_vectors, dct_residuals, boxes, hidden_state=None, return_logits=False):
        """
        Process a single frame (useful for inference and training)
        
        Args:
            motion_vectors: [1, 2, 60, 60]
            dct_residuals: [1, 120, 120, 64]
            boxes: [N, 4] - Bounding boxes in image coordinates [0, image_size]
            hidden_state: Optional LSTM state
            return_logits: If True, return raw logits for training; if False, return probabilities
        
        Returns:
            updated_boxes: [N, 4] - In image coordinates [0, image_size]
            confidences: [N] - logits or probabilities depending on return_logits
            hidden_state: Updated LSTM state
        """
        # Encode
        fused_features, global_features = self.encoder(motion_vectors, dct_residuals)
        
        # Extract ROI features (ROI extractor handles coordinate normalization)
        if self.use_attention:
            roi_features, attention_info = self.roi_extractor(fused_features, boxes)
        else:
            roi_features = self.roi_extractor(fused_features, boxes)
        
        # Track (with logits option)
        pos_deltas, confs, size_deltas, hidden_state = self.lstm_tracker(
            roi_features,
            global_features[0],
            hidden_state,
            return_logits=return_logits
        )
        
        # Update boxes
        updated_boxes = self._update_boxes(boxes, pos_deltas, size_deltas)
        
        return updated_boxes, confs.squeeze(-1), hidden_state
    
    def _update_boxes(self, boxes, pos_deltas, size_deltas):
        """
        Update boxes with predicted deltas
        
        Args:
            boxes: [N, 4] - x1, y1, x2, y2 in image coordinates
            pos_deltas: [N, 2] - dx_center, dy_center
            size_deltas: [N, 2] - dw, dh (as log-space deltas)
        """
        # Get center and size
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        sizes = boxes[:, 2:] - boxes[:, :2]
        
        # Update
        new_centers = centers + pos_deltas
        new_sizes = sizes * torch.exp(size_deltas)
        
        # Convert back to corners
        new_boxes = torch.cat([
            new_centers - new_sizes / 2,
            new_centers + new_sizes / 2
        ], dim=1)
        
        # Clamp to valid range
        new_boxes = torch.clamp(new_boxes, 0, self.image_size)
        
        return new_boxes
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def load_mv_encoder_weights(self, state_dict, strict=False):
        """
        Load motion vector encoder weights from pre-trained MV-only model
        
        Args:
            state_dict: State dict from MV-only model
            strict: Whether to strictly enforce matching keys
        """
        # Extract MV encoder weights
        mv_encoder_state = {}
        for key, value in state_dict.items():
            if 'encoder.mv' in key:
                # Map keys to new architecture
                new_key = key.replace('encoder.', 'encoder.mv_encoder.')
                mv_encoder_state[new_key] = value
        
        # Load with partial matching
        self.load_state_dict(mv_encoder_state, strict=strict)
        print(f"Loaded {len(mv_encoder_state)} MV encoder parameters from pre-trained model")
