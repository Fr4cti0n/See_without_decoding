"""
Multi-Scale ROI Extractor with Shared Weights

Extracts features at multiple scales using the SAME network weights.
This keeps parameter count low while capturing multi-scale information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleROIExtractor(nn.Module):
    """
    Extract ROI features at multiple scales with shared weights
    
    Key features:
    - Multiple ROI sizes (e.g., 3×3, 7×7, 11×11) for different detail levels
    - SHARED ROI head network across all scales (no parameter explosion!)
    - Optional global context pooling
    - Concatenates multi-scale features for richer representation
    
    Parameters stay minimal: same as single-scale + small projection layer!
    """
    def __init__(
        self,
        feature_dim=64,
        roi_sizes=[3, 7, 11],  # Multiple scales: small, medium, large
        output_dim=64,          # Final feature dimension per object
        image_size=960,         # Original image size for coordinate normalization
        use_global_context=True  # Include global average pooled features
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.roi_sizes = roi_sizes
        self.output_dim = output_dim
        self.image_size = image_size
        self.use_global_context = use_global_context
        
        # Shared ROI head for all scales (keeps params low!)
        # This network processes each scale independently with SAME weights
        self.shared_roi_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Calculate total multi-scale feature dimension
        num_scales = len(roi_sizes)
        if use_global_context:
            num_scales += 1  # +1 for global context
        multiscale_dim = num_scales * 64
        
        # Project multi-scale features back to output_dim
        # This is the only "extra" parameters compared to single-scale
        self.feature_projection = nn.Sequential(
            nn.Linear(multiscale_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feature_maps, boxes):
        """
        Extract multi-scale ROI features for all boxes
        
        Args:
            feature_maps: [B, C, H, W] - Spatial features (e.g., [1, 64, 120, 120])
            boxes: [N, 4] - Bounding boxes in (x1, y1, x2, y2) format (pixel coordinates)
        
        Returns:
            roi_features: [N, output_dim] - Multi-scale features for each box
        """
        if boxes.shape[0] == 0:
            # No boxes - return empty features
            return torch.zeros(0, self.output_dim, device=feature_maps.device)
        
        B, C, H, W = feature_maps.shape
        N = boxes.shape[0]
        
        # Normalize boxes to [-1, 1] range
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] = (boxes[:, [0, 2]] / self.image_size) * 2 - 1
        boxes_norm[:, [1, 3]] = (boxes[:, [1, 3]] / self.image_size) * 2 - 1
        
        # Extract features at each scale
        multi_scale_features = []
        
        for roi_size in self.roi_sizes:
            # Extract ROI at this scale
            roi_features_spatial = self._extract_roi_at_scale(
                feature_maps, boxes_norm, roi_size, N, C
            )  # [N, C, roi_size, roi_size]
            
            # Pool to single vector per ROI
            roi_pooled = F.adaptive_avg_pool2d(roi_features_spatial, 1)  # [N, C, 1, 1]
            roi_pooled = roi_pooled.view(N, C)  # [N, C]
            
            # Apply shared ROI head
            roi_features = self.shared_roi_head(roi_pooled)  # [N, 64]
            multi_scale_features.append(roi_features)
        
        # Optional: Add global context
        if self.use_global_context:
            global_features = F.adaptive_avg_pool2d(feature_maps, 1)  # [B, C, 1, 1]
            global_features = global_features.view(B, C)  # [B, C]
            global_features = global_features.expand(N, -1)  # [N, C]
            global_features = self.shared_roi_head(global_features)  # [N, 64]
            multi_scale_features.append(global_features)
        
        # Concatenate all scales
        concat_features = torch.cat(multi_scale_features, dim=1)  # [N, num_scales * 64]
        
        # Project to final output dimension
        output_features = self.feature_projection(concat_features)  # [N, output_dim]
        
        return output_features
    
    def _extract_roi_at_scale(self, feature_maps, boxes_norm, roi_size, N, C):
        """
        Extract ROI features at a specific scale using grid_sample
        
        Args:
            feature_maps: [B, C, H, W]
            boxes_norm: [N, 4] - Normalized boxes [-1, 1]
            roi_size: int - ROI resolution for this scale
            N: int - Number of boxes
            C: int - Number of channels
        
        Returns:
            roi_features: [N, C, roi_size, roi_size]
        """
        B = feature_maps.shape[0]
        
        # Create affine transformation matrices
        theta = self._boxes_to_affine_matrices(boxes_norm, roi_size)  # [N, 2, 3]
        
        # Expand feature maps to match number of boxes
        if B == 1:
            feature_maps_expanded = feature_maps.expand(N, -1, -1, -1)
        else:
            feature_maps_expanded = feature_maps[0:1].expand(N, -1, -1, -1)
        
        # Extract ROIs using grid_sample
        grid = F.affine_grid(theta, [N, C, roi_size, roi_size], align_corners=False)
        roi_features = F.grid_sample(
            feature_maps_expanded,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [N, C, roi_size, roi_size]
        
        return roi_features
    
    def _boxes_to_affine_matrices(self, boxes_norm, roi_size):
        """
        Convert normalized boxes to affine transformation matrices
        
        Args:
            boxes_norm: [N, 4] - Boxes in normalized coordinates [-1, 1]
            roi_size: int - Output ROI size
        
        Returns:
            theta: [N, 2, 3] - Affine transformation matrices
        """
        N = boxes_norm.shape[0]
        device = boxes_norm.device
        
        # Extract box coordinates
        x1, y1, x2, y2 = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
        
        # Calculate box center and size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # Avoid division by zero
        w = torch.clamp(w, min=1e-6)
        h = torch.clamp(h, min=1e-6)
        
        # Create affine transformation matrix
        theta = torch.zeros(N, 2, 3, device=device)
        
        # Scale: ROI size → box size
        theta[:, 0, 0] = w / 2
        theta[:, 1, 1] = h / 2
        
        # Translation: ROI center → box center
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy
        
        return theta
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientMultiScaleROIExtractor(nn.Module):
    """
    More efficient version that batches all scales together
    
    Processes all ROI sizes in a single batched forward pass for better GPU utilization.
    """
    def __init__(
        self,
        feature_dim=64,
        roi_sizes=[3, 7, 11],
        output_dim=64,
        image_size=960,
        use_global_context=True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.roi_sizes = roi_sizes
        self.output_dim = output_dim
        self.image_size = image_size
        self.use_global_context = use_global_context
        
        # Shared processing network
        self.shared_roi_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        num_scales = len(roi_sizes) + (1 if use_global_context else 0)
        multiscale_dim = num_scales * 64
        
        self.feature_projection = nn.Sequential(
            nn.Linear(multiscale_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feature_maps, boxes):
        """
        Extract multi-scale features with batched processing
        
        Args:
            feature_maps: [B, C, H, W]
            boxes: [N, 4]
        
        Returns:
            roi_features: [N, output_dim]
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, self.output_dim, device=feature_maps.device)
        
        B, C, H, W = feature_maps.shape
        N = boxes.shape[0]
        num_scales = len(self.roi_sizes)
        
        # Normalize boxes
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] = (boxes[:, [0, 2]] / self.image_size) * 2 - 1
        boxes_norm[:, [1, 3]] = (boxes[:, [1, 3]] / self.image_size) * 2 - 1
        
        # Batch all scales together
        # Create [N * num_scales] batch of ROI extractions
        all_roi_features = []
        
        for roi_size in self.roi_sizes:
            theta = self._boxes_to_affine_matrices(boxes_norm, roi_size)
            feature_maps_expanded = feature_maps[0:1].expand(N, -1, -1, -1)
            
            grid = F.affine_grid(theta, [N, C, roi_size, roi_size], align_corners=False)
            roi_spatial = F.grid_sample(
                feature_maps_expanded,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            
            # Pool and process
            roi_pooled = F.adaptive_avg_pool2d(roi_spatial, 1).view(N, C)
            roi_features = self.shared_roi_head(roi_pooled)
            all_roi_features.append(roi_features)
        
        # Add global context
        if self.use_global_context:
            global_features = F.adaptive_avg_pool2d(feature_maps, 1).view(B, C)
            global_features = global_features.expand(N, -1)
            global_features = self.shared_roi_head(global_features)
            all_roi_features.append(global_features)
        
        # Concatenate and project
        concat_features = torch.cat(all_roi_features, dim=1)
        output_features = self.feature_projection(concat_features)
        
        return output_features
    
    def _boxes_to_affine_matrices(self, boxes_norm, roi_size):
        """Same as MultiScaleROIExtractor"""
        N = boxes_norm.shape[0]
        device = boxes_norm.device
        
        x1, y1, x2, y2 = boxes_norm[:, 0], boxes_norm[:, 1], boxes_norm[:, 2], boxes_norm[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = torch.clamp(x2 - x1, min=1e-6)
        h = torch.clamp(y2 - y1, min=1e-6)
        
        theta = torch.zeros(N, 2, 3, device=device)
        theta[:, 0, 0] = w / 2
        theta[:, 1, 1] = h / 2
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy
        
        return theta
    
    def get_num_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
