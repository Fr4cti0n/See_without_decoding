"""
Motion ROI Align Module

Spatially aligns motion vectors with bounding boxes using ROI Align.
This allows the model to extract per-box motion features that are
directly aligned with each object's spatial region.

Key idea: Instead of processing motion vectors globally, extract
motion features from the specific region of each bounding box.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


class MotionROIAlign(nn.Module):
    """
    Extract spatially-aligned motion features for each bounding box.
    
    This module applies ROI Align to motion vectors, treating them as
    a 2-channel feature map. For each bounding box, it extracts a fixed-size
    (e.g., 7×7) grid of motion vectors from that box's spatial region.
    
    Args:
        roi_size (tuple): Output spatial size, e.g., (7, 7)
        motion_feature_dim (int): Dimension of output motion features
        spatial_scale (float): Scale factor to convert normalized coords to motion grid coords
    """
    
    def __init__(self, roi_size=(7, 7), motion_feature_dim=64):
        super().__init__()
        self.roi_size = roi_size
        self.motion_feature_dim = motion_feature_dim
        
        # Process aligned motion features with CNN
        # Input: [N, 2, roi_size[0], roi_size[1]]
        # Output: [N, motion_feature_dim]
        self.motion_processor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, motion_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(motion_feature_dim),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)  # Global pooling → [N, motion_feature_dim, 1, 1]
        )
        
    def forward(self, boxes, motion_vectors):
        """
        Extract motion features aligned with each box.
        
        Args:
            boxes: Tensor [N, 4] in normalized [0,1] format [cx, cy, w, h]
            motion_vectors: Tensor [H, W, 2] motion vector grid
            
        Returns:
            motion_features: Tensor [N, motion_feature_dim]
        """
        if boxes.shape[0] == 0:
            # No boxes, return empty features
            return torch.zeros(0, self.motion_feature_dim, device=boxes.device, dtype=boxes.dtype)
        
        H, W = motion_vectors.shape[:2]
        device = boxes.device
        dtype = boxes.dtype
        
        # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] in absolute coords
        boxes_abs = boxes.clone()
        
        # Convert to absolute pixel coordinates
        boxes_abs[:, [0, 2]] *= W  # cx, w → pixel space
        boxes_abs[:, [1, 3]] *= H  # cy, h → pixel space
        
        # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = torch.zeros_like(boxes_abs)
        boxes_xyxy[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2  # y2 = cy + h/2
        
        # Clamp to valid range
        boxes_xyxy[:, [0, 2]] = torch.clamp(boxes_xyxy[:, [0, 2]], 0, W)
        boxes_xyxy[:, [1, 3]] = torch.clamp(boxes_xyxy[:, [1, 3]], 0, H)
        
        # Prepare motion vectors for ROI Align: [H, W, 2] → [1, 2, H, W]
        mv_tensor = motion_vectors.permute(2, 0, 1).unsqueeze(0)  # [1, 2, H, W]
        
        # ROI Align expects boxes as list of [M, 4] tensors (one per batch element)
        # Since we have batch size 1, wrap boxes in a list
        boxes_list = [boxes_xyxy]
        
        # Apply ROI Align
        # Output: [N, 2, roi_size[0], roi_size[1]]
        aligned_mvs = roi_align(
            mv_tensor,
            boxes_list,
            output_size=self.roi_size,
            spatial_scale=1.0,  # Already in absolute coords
            sampling_ratio=-1,  # Adaptive sampling
            aligned=True  # Use aligned version (more accurate)
        )
        
        # Process aligned motion vectors through CNN
        # [N, 2, 7, 7] → [N, motion_feature_dim]
        motion_features = self.motion_processor(aligned_mvs)  # [N, motion_feature_dim, 1, 1]
        motion_features = motion_features.squeeze(-1).squeeze(-1)  # [N, motion_feature_dim]
        
        return motion_features


class MotionAlignedBoxEncoder(nn.Module):
    """
    Encode bounding boxes with spatially-aligned motion features.
    
    Combines:
    1. Box geometric features (position, size)
    2. Spatially-aligned motion features (from ROI Align)
    
    Args:
        box_dim (int): Dimension of box features (typically 4: cx, cy, w, h)
        motion_feature_dim (int): Dimension of motion features from ROI Align
        hidden_dim (int): Dimension of output features
        roi_size (tuple): ROI Align output size
    """
    
    def __init__(self, box_dim=4, motion_feature_dim=64, hidden_dim=128, roi_size=(7, 7)):
        super().__init__()
        self.box_dim = box_dim
        self.motion_feature_dim = motion_feature_dim
        self.hidden_dim = hidden_dim
        
        # ROI Align module for motion vectors
        self.motion_roi_align = MotionROIAlign(
            roi_size=roi_size,
            motion_feature_dim=motion_feature_dim
        )
        
        # Box embedding
        self.box_embedding = nn.Sequential(
            nn.Linear(box_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer: combine box and motion features
        self.fusion = nn.Sequential(
            nn.Linear(64 + motion_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, boxes, motion_vectors):
        """
        Encode boxes with aligned motion features.
        
        Args:
            boxes: Tensor [N, box_dim] normalized bounding boxes
            motion_vectors: Tensor [H, W, 2] motion vector grid
            
        Returns:
            encoded_features: Tensor [N, hidden_dim]
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, self.hidden_dim, device=boxes.device, dtype=boxes.dtype)
        
        # Extract motion features aligned with each box
        motion_features = self.motion_roi_align(boxes, motion_vectors)  # [N, motion_feature_dim]
        
        # Embed box geometric features
        box_features = self.box_embedding(boxes)  # [N, 64]
        
        # Concatenate box and motion features
        combined = torch.cat([box_features, motion_features], dim=1)  # [N, 64 + motion_feature_dim]
        
        # Fuse features
        encoded = self.fusion(combined)  # [N, hidden_dim]
        
        return encoded


def test_motion_roi_align():
    """Test the Motion ROI Align module."""
    print("🧪 Testing Motion ROI Align Module\n")
    
    # Create dummy data
    batch_size = 1
    num_boxes = 3
    H, W = 40, 40  # Motion vector grid size
    
    # Random motion vectors
    motion_vectors = torch.randn(H, W, 2)
    print(f"Motion vectors shape: {motion_vectors.shape}")
    
    # Random boxes [cx, cy, w, h] normalized [0, 1]
    boxes = torch.tensor([
        [0.3, 0.4, 0.2, 0.3],  # Box 1
        [0.7, 0.5, 0.25, 0.35],  # Box 2
        [0.5, 0.7, 0.15, 0.2],  # Box 3
    ])
    print(f"Boxes shape: {boxes.shape}")
    print(f"Boxes:\n{boxes}\n")
    
    # Test MotionROIAlign
    roi_align_module = MotionROIAlign(roi_size=(7, 7), motion_feature_dim=64)
    motion_features = roi_align_module(boxes, motion_vectors)
    
    print(f"Output motion features shape: {motion_features.shape}")
    print(f"Expected: [{num_boxes}, 64]")
    assert motion_features.shape == (num_boxes, 64), "Shape mismatch!"
    print("✅ MotionROIAlign test passed!\n")
    
    # Test MotionAlignedBoxEncoder
    encoder = MotionAlignedBoxEncoder(
        box_dim=4,
        motion_feature_dim=64,
        hidden_dim=128,
        roi_size=(7, 7)
    )
    encoded = encoder(boxes, motion_vectors)
    
    print(f"Encoded features shape: {encoded.shape}")
    print(f"Expected: [{num_boxes}, 128]")
    assert encoded.shape == (num_boxes, 128), "Shape mismatch!"
    print("✅ MotionAlignedBoxEncoder test passed!\n")
    
    # Test with empty boxes
    empty_boxes = torch.zeros(0, 4)
    empty_encoded = encoder(empty_boxes, motion_vectors)
    print(f"Empty boxes encoded shape: {empty_encoded.shape}")
    assert empty_encoded.shape == (0, 128), "Empty boxes shape mismatch!"
    print("✅ Empty boxes test passed!\n")
    
    print("🎉 All tests passed!")


if __name__ == "__main__":
    test_motion_roi_align()
