"""
Motion Vector Sampling and Feature Extraction

Utilities for extracting motion vectors and features at object locations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_mv_at_bbox(mv_field, bbox, grid_size=40, image_size=640):
    """
    Sample motion vectors within a bounding box region.
    
    Args:
        mv_field: [2, H, W] motion vector field (typically 40×40)
        bbox: [4] bounding box [cx, cy, w, h] in pixel coordinates
        grid_size: Size of MV grid (default: 40)
        image_size: Size of original image (default: 640)
        
    Returns:
        mv_sample: [2] averaged motion vector for the box
        mv_region: [2, H', W'] motion vectors in box region
    """
    cx, cy, w, h = bbox
    
    # Convert pixel coordinates to MV grid coordinates
    # Each MV block is 16×16 pixels (640/40 = 16)
    block_size = image_size / grid_size
    
    cx_mv = cx / block_size
    cy_mv = cy / block_size
    w_mv = w / block_size
    h_mv = h / block_size
    
    # Define sampling region (with bounds checking)
    x1 = max(0, int(cx_mv - w_mv / 2))
    x2 = min(grid_size, int(cx_mv + w_mv / 2) + 1)
    y1 = max(0, int(cy_mv - h_mv / 2))
    y2 = min(grid_size, int(cy_mv + h_mv / 2) + 1)
    
    # Ensure valid region
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    
    # Extract MV blocks in this region
    mv_region = mv_field[:, y1:y2, x1:x2]  # [2, H', W']
    
    # Compute average motion in this region
    mv_avg = mv_region.mean(dim=[-2, -1])  # [2]
    
    return mv_avg, mv_region


def sample_features_at_bbox(feature_map, bbox, grid_size=40, image_size=640):
    """
    Sample features within a bounding box region using bilinear interpolation.
    
    Args:
        feature_map: [C, H, W] feature map
        bbox: [4] bounding box [cx, cy, w, h] in pixel coordinates
        grid_size: Size of feature grid (default: 40)
        image_size: Size of original image (default: 640)
        
    Returns:
        features: [C] averaged features for the box
    """
    cx, cy, w, h = bbox
    
    # Convert to grid coordinates
    block_size = image_size / grid_size
    cx_grid = cx / block_size
    cy_grid = cy / block_size
    w_grid = w / block_size
    h_grid = h / block_size
    
    # Define sampling region
    x1 = max(0, int(cx_grid - w_grid / 2))
    x2 = min(grid_size, int(cx_grid + w_grid / 2) + 1)
    y1 = max(0, int(cy_grid - h_grid / 2))
    y2 = min(grid_size, int(cy_grid + h_grid / 2) + 1)
    
    # Ensure valid region
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    
    # Extract features in this region
    feature_region = feature_map[:, y1:y2, x1:x2]  # [C, H', W']
    
    # Average pool
    features = feature_region.mean(dim=[-2, -1])  # [C]
    
    return features


def batch_sample_mv_at_bboxes(mv_field, bboxes, grid_size=40, image_size=640):
    """
    Sample motion vectors for multiple bounding boxes.
    
    Args:
        mv_field: [2, H, W] motion vector field
        bboxes: [N, 4] bounding boxes
        grid_size: Size of MV grid
        image_size: Size of original image
        
    Returns:
        mv_samples: [N, 2] averaged motion vectors
    """
    num_boxes = len(bboxes)
    mv_samples = torch.zeros(num_boxes, 2, device=mv_field.device, dtype=mv_field.dtype)
    
    for i, bbox in enumerate(bboxes):
        mv_avg, _ = sample_mv_at_bbox(mv_field, bbox, grid_size, image_size)
        mv_samples[i] = mv_avg
    
    return mv_samples


def batch_sample_features_at_bboxes(feature_map, bboxes, grid_size=40, image_size=640):
    """
    Sample features for multiple bounding boxes.
    
    Args:
        feature_map: [C, H, W] feature map
        bboxes: [N, 4] bounding boxes
        grid_size: Size of feature grid
        image_size: Size of original image
        
    Returns:
        features: [N, C] sampled features
    """
    num_boxes = len(bboxes)
    num_channels = feature_map.size(0)
    features = torch.zeros(num_boxes, num_channels, device=feature_map.device, dtype=feature_map.dtype)
    
    for i, bbox in enumerate(bboxes):
        feat = sample_features_at_bbox(feature_map, bbox, grid_size, image_size)
        features[i] = feat
    
    return features


class RoIFeatureExtractor(nn.Module):
    """
    Extract features from regions of interest using RoI pooling.
    More efficient than sequential sampling.
    """
    
    def __init__(self, output_size=7):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, feature_map, bboxes, grid_size=40, image_size=640):
        """
        Extract features using RoI pooling.
        
        Args:
            feature_map: [C, H, W] feature map
            bboxes: [N, 4] bounding boxes in [cx, cy, w, h] format
            grid_size: Size of feature grid
            image_size: Size of original image
            
        Returns:
            roi_features: [N, C] pooled features
        """
        if len(bboxes) == 0:
            return torch.zeros(0, feature_map.size(0), device=feature_map.device)
        
        # Convert bboxes to [x1, y1, x2, y2] format in grid coordinates
        block_size = image_size / grid_size
        
        # [cx, cy, w, h] -> [x1, y1, x2, y2]
        x1 = (bboxes[:, 0] - bboxes[:, 2] / 2) / block_size
        y1 = (bboxes[:, 1] - bboxes[:, 3] / 2) / block_size
        x2 = (bboxes[:, 0] + bboxes[:, 2] / 2) / block_size
        y2 = (bboxes[:, 1] + bboxes[:, 3] / 2) / block_size
        
        # Normalize to [0, 1]
        x1 = x1 / grid_size
        y1 = y1 / grid_size
        x2 = x2 / grid_size
        y2 = y2 / grid_size
        
        # Clamp to valid range
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)
        
        # Create grid for sampling
        theta = torch.zeros(len(bboxes), 2, 3, device=feature_map.device)
        for i in range(len(bboxes)):
            # Scale and translate to RoI
            w = x2[i] - x1[i]
            h = y2[i] - y1[i]
            cx = (x1[i] + x2[i]) / 2
            cy = (y1[i] + y2[i]) / 2
            
            # Affine transformation matrix
            theta[i, 0, 0] = w  # scale x
            theta[i, 1, 1] = h  # scale y
            theta[i, 0, 2] = 2 * cx - 1  # translate x
            theta[i, 1, 2] = 2 * cy - 1  # translate y
        
        # Apply grid sampling
        grid = F.affine_grid(theta, (len(bboxes), feature_map.size(0), self.output_size, self.output_size), 
                             align_corners=False)
        
        # Expand feature map for batch sampling
        feature_map_batch = feature_map.unsqueeze(0).expand(len(bboxes), -1, -1, -1)
        
        # Sample features
        sampled = F.grid_sample(feature_map_batch, grid, align_corners=False, mode='bilinear')
        
        # Average pool to get single feature vector per RoI
        roi_features = sampled.mean(dim=[-2, -1])  # [N, C]
        
        return roi_features
