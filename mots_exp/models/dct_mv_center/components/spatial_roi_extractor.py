"""
Spatial ROI Extractor with Batched Processing
Extracts region features for tracked objects with no for-loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialROIExtractor(nn.Module):
    """
    Extract ROI features from spatial feature maps
    
    Optimized for speed:
    - Batched processing (no for-loops)
    - Uses grid_sample for efficient ROI extraction
    - Works with bounding boxes of any size
    """
    def __init__(
        self,
        feature_dim=64,
        roi_size=7,         # ROI pooling resolution
        output_dim=64,      # Final feature dimension per object
        image_size=960      # Original image size for coordinate normalization
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.roi_size = roi_size
        self.output_dim = output_dim
        self.image_size = image_size  # ✅ Store for proper normalization
        
        # Feature refinement after ROI pooling
        self.roi_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(roi_size),
            nn.Flatten(),
            nn.Linear(feature_dim * roi_size * roi_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, feature_maps, boxes):
        """
        Extract ROI features for all boxes
        
        Args:
            feature_maps: [B, C, H, W] - Spatial features (e.g., [1, 64, 120, 120])
            boxes: [N, 4] - Bounding boxes in (x1, y1, x2, y2) format (pixel coordinates)
        
        Returns:
            roi_features: [N, output_dim] - Features for each box
        """
        if boxes.shape[0] == 0:
            # No boxes - return empty features
            return torch.zeros(0, self.output_dim, device=feature_maps.device)
        
        B, C, H, W = feature_maps.shape
        N = boxes.shape[0]
        
        # ============================================================
        # METHOD 1: Grid Sample (Differentiable, Fast)
        # ============================================================
        # Convert boxes to normalized coordinates [-1, 1]
        # ✅ CRITICAL FIX: Normalize by image_size (960) not feature_size (H,W=120)
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] = (boxes[:, [0, 2]] / self.image_size) * 2 - 1  # x coordinates
        boxes_norm[:, [1, 3]] = (boxes[:, [1, 3]] / self.image_size) * 2 - 1  # y coordinates
        
        # Create sampling grid for each box
        # Grid shape: [N, roi_size, roi_size, 2]
        theta = self._boxes_to_affine_matrices(boxes_norm, self.roi_size)
        
        # Expand feature maps to match number of boxes
        # [B, C, H, W] → [N, C, H, W] (assuming B=1 for now)
        if B == 1:
            feature_maps_expanded = feature_maps.expand(N, -1, -1, -1)
        else:
            # If batched, need to handle box-to-batch mapping
            # For now, assume all boxes belong to batch 0
            feature_maps_expanded = feature_maps[0:1].expand(N, -1, -1, -1)
        
        # Extract ROIs using grid_sample
        grid = F.affine_grid(theta, [N, C, self.roi_size, self.roi_size], align_corners=False)
        roi_features_spatial = F.grid_sample(
            feature_maps_expanded,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [N, C, roi_size, roi_size]
        
        # Apply ROI head
        roi_features = self.roi_head(roi_features_spatial)  # [N, output_dim]
        
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
        # Maps from ROI grid to feature map coordinates
        theta = torch.zeros(N, 2, 3, device=device)
        
        # Scale: ROI size → box size
        theta[:, 0, 0] = w / 2
        theta[:, 1, 1] = h / 2
        
        # Translation: ROI center → box center
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy
        
        return theta
    
    def forward_with_masks(self, feature_maps, boxes):
        """
        Alternative method using binary masks (simpler but slightly slower)
        
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
        
        # Create binary masks for each box
        masks = self._create_box_masks(boxes, H, W)  # [N, H, W]
        
        # Extract features
        roi_features_list = []
        for i in range(N):
            # Apply mask to feature maps
            mask = masks[i:i+1, None, :, :]  # [1, 1, H, W]
            masked_features = feature_maps[0:1] * mask  # [1, C, H, W]
            
            # Get bounding box region
            x1, y1, x2, y2 = boxes[i].long()
            x1, x2 = torch.clamp(x1, 0, W-1), torch.clamp(x2, 1, W)
            y1, y2 = torch.clamp(y1, 0, H-1), torch.clamp(y2, 1, H)
            
            # Extract ROI patch
            roi_patch = masked_features[:, :, y1:y2, x1:x2]  # [1, C, h, w]
            
            # Pool to fixed size
            roi_pooled = F.adaptive_avg_pool2d(roi_patch, self.roi_size)  # [1, C, roi_size, roi_size]
            
            # Apply ROI head
            roi_feature = self.roi_head(roi_pooled)  # [1, output_dim]
            roi_features_list.append(roi_feature)
        
        roi_features = torch.cat(roi_features_list, dim=0)  # [N, output_dim]
        
        return roi_features
    
    def _create_box_masks(self, boxes, H, W):
        """
        Create binary masks for bounding boxes
        
        Args:
            boxes: [N, 4] - Boxes in (x1, y1, x2, y2) format
            H, W: int - Feature map height and width
        
        Returns:
            masks: [N, H, W] - Binary masks
        """
        N = boxes.shape[0]
        device = boxes.device
        
        # Create coordinate grids
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        y_grid = y_grid[None, :, :].expand(N, -1, -1)  # [N, H, W]
        x_grid = x_grid[None, :, :].expand(N, -1, -1)  # [N, H, W]
        
        # Extract box coordinates
        x1 = boxes[:, 0:1, None]  # [N, 1, 1]
        y1 = boxes[:, 1:2, None]
        x2 = boxes[:, 2:3, None]
        y2 = boxes[:, 3:4, None]
        
        # Create masks
        masks = (
            (x_grid >= x1) & (x_grid < x2) &
            (y_grid >= y1) & (y_grid < y2)
        ).float()  # [N, H, W]
        
        return masks
