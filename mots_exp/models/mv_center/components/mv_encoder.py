"""
Motion Vector Feature Encoder

CNN-based encoder for processing motion vector fields (40×40) into feature maps.
"""

import torch
import torch.nn as nn


class MotionVectorEncoder(nn.Module):
    """
    Encodes motion vector fields into dense feature representations.
    
    Input: [2, H, W] motion vectors (u, v components)
    Output: [feature_dim, H, W] encoded features
    """
    
    def __init__(self, input_channels=2, feature_dim=128, grid_size=40):
        super().__init__()
        self.grid_size = grid_size
        self.feature_dim = feature_dim
        
        # Convolutional encoder
        self.encoder = nn.Sequential(
            # Initial embedding: 2 -> 64
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Feature extraction: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Deep features: 128 -> feature_dim
            nn.Conv2d(128, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Optional: Residual connection for motion magnitude
        self.motion_magnitude_branch = nn.Sequential(
            nn.Conv2d(input_channels, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, motion_vectors):
        """
        Encode motion vectors into features.
        
        Args:
            motion_vectors: [2, H, W] or [B, 2, H, W]
                u, v components of motion field
        
        Returns:
            features: [feature_dim, H, W] or [B, feature_dim, H, W]
        """
        # Handle both batched and unbatched inputs
        if motion_vectors.ndim == 3:
            motion_vectors = motion_vectors.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Encode through main branch
        features = self.encoder(motion_vectors)
        
        # Add residual connection from magnitude
        magnitude_features = self.motion_magnitude_branch(motion_vectors)
        features = features + magnitude_features
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features


class TemporalMotionEncoder(nn.Module):
    """
    Encode multiple temporal motion vector fields.
    
    Useful when processing sequences of motion vectors (e.g., from multiple P-frames).
    """
    
    def __init__(self, input_channels=2, feature_dim=128, num_frames=2):
        super().__init__()
        self.num_frames = num_frames
        self.feature_dim = feature_dim
        
        # Per-frame encoder
        self.frame_encoder = MotionVectorEncoder(
            input_channels=input_channels,
            feature_dim=feature_dim // 2  # Half features per frame
        )
        
        # Temporal aggregation
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(feature_dim // 2 * num_frames, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, motion_sequences):
        """
        Encode temporal motion sequences.
        
        Args:
            motion_sequences: [T, 2, H, W] sequence of motion fields
                T = number of frames (typically 2 for bi-directional)
        
        Returns:
            features: [feature_dim, H, W] temporally aggregated features
        """
        frame_features = []
        
        for t in range(len(motion_sequences)):
            feat = self.frame_encoder(motion_sequences[t])
            frame_features.append(feat)
        
        # Concatenate along channel dimension
        temporal_features = torch.cat(frame_features, dim=0)  # [T * (feature_dim//2), H, W]
        
        # Fuse temporal information
        fused = self.temporal_fusion(temporal_features.unsqueeze(0)).squeeze(0)
        
        return fused


class LightweightMVEncoder(nn.Module):
    """
    Lightweight motion vector encoder for faster inference.
    
    Uses depthwise separable convolutions.
    """
    
    def __init__(self, input_channels=2, feature_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Depthwise separable conv 1
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, 
                      groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Depthwise separable conv 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, motion_vectors):
        """Encode motion vectors."""
        if motion_vectors.ndim == 3:
            motion_vectors = motion_vectors.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        features = self.encoder(motion_vectors)
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features
