"""
Spatially Aligned DCT-MV Encoder
Combines Motion Vectors (60×60) and DCT Residuals (120×120) with spatial alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiallyAlignedDCTMVEncoder(nn.Module):
    """
    Encoder that preserves spatial relationship between MVs and DCT blocks
    
    Strategy:
    1. Encode DCT at 120×120 resolution
    2. Encode MV at 60×60 resolution
    3. Upsample MV to 120×120 to match DCT
    4. Fuse at same spatial resolution
    """
    def __init__(
        self,
        num_dct_coeffs=16,      # Use top 16 DCT coefficients
        mv_channels=2,          # Number of MV input channels (0 to disable)
        dct_channels=64,        # Number of DCT input channels (0 to disable)
        mv_feature_dim=32,      # MV encoder output channels
        dct_feature_dim=32,     # DCT encoder output channels
        fused_feature_dim=64    # Final fused features
    ):
        super().__init__()
        
        self.num_dct_coeffs = num_dct_coeffs
        self.mv_channels = mv_channels
        self.dct_channels = dct_channels
        self.use_mv = mv_channels > 0
        self.use_dct = dct_channels > 0
        
        # Ensure at least one modality is enabled
        if not self.use_mv and not self.use_dct:
            raise ValueError("At least one of mv_channels or dct_channels must be > 0")
        
        # ============================================================
        # MOTION VECTOR ENCODER (works at 60×60 resolution)
        # ============================================================
        if self.use_mv:
            self.mv_encoder = nn.Sequential(
                # Input: [mv_channels, 60, 60]
                nn.Conv2d(mv_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                # Output: [32, 60, 60]
            )
            
            # Upsample MV features to match DCT resolution (60→120)
            # Using nearest neighbor to preserve sharp motion boundaries
            self.mv_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            
            # Additional conv after upsampling to refine features
            self.mv_refine = nn.Sequential(
                nn.Conv2d(32, mv_feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(mv_feature_dim),
                nn.ReLU(inplace=True),
                # Output: [mv_feature_dim, 120, 120]
            )
        else:
            self.mv_encoder = None
            self.mv_upsample = None
            self.mv_refine = None
        
        # ============================================================
        # DCT RESIDUAL ENCODER (works at 120×120 resolution)
        # ============================================================
        if self.use_dct:
            # First, reduce dct_channels coefficients to top K=num_dct_coeffs
            # Only needed if dct_channels > num_dct_coeffs
            if dct_channels > num_dct_coeffs:
                self.dct_coefficient_selector = nn.Conv2d(
                    dct_channels, num_dct_coeffs, 
                    kernel_size=1,  # 1×1 conv to select/mix coefficients
                    bias=False
                )
                # Initialize to select top-K coefficients in zig-zag order
                self._init_coefficient_selector()
            else:
                # Use all provided coefficients directly
                self.dct_coefficient_selector = None
                self.num_dct_coeffs = dct_channels
            
            # Encode selected DCT coefficients
            self.dct_encoder = nn.Sequential(
                # Input: [num_dct_coeffs, 120, 120]
                nn.Conv2d(self.num_dct_coeffs, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, dct_feature_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(dct_feature_dim),
                nn.ReLU(inplace=True),
                # Output: [dct_feature_dim, 120, 120]
            )
        else:
            self.dct_coefficient_selector = None
            self.dct_encoder = None
        
        # ============================================================
        # SPATIAL FUSION (at 120×120 resolution)
        # ============================================================
        # Fuse MV and DCT features at same spatial resolution
        # Input dimensions depend on which modalities are enabled
        fusion_input_dim = (mv_feature_dim if self.use_mv else 0) + (dct_feature_dim if self.use_dct else 0)
        
        self.fusion = nn.Sequential(
            # Input: [fusion_input_dim, 120, 120] (32, 64, or 32 channels)
            nn.Conv2d(fusion_input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, fused_feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_feature_dim),
            nn.ReLU(inplace=True),
            # Output: [fused_feature_dim, 120, 120]
        )
        
        # Global feature pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def _init_coefficient_selector(self):
        """
        Initialize coefficient selector to pick top-K in zig-zag order
        
        Zig-zag order (JPEG/H.264 standard):
        DC, AC1, AC2, AC3, AC8, AC9, AC10, AC16, ...
        """
        # Zig-zag order for 8×8 DCT block
        zig_zag_order = [
            0,  1,  8, 16,  9,  2,  3, 10,  # First 8 (DC + low-freq)
           17, 24, 32, 25, 18, 11,  4,  5,  # Next 8 (low-to-mid freq)
           12, 19, 26, 33, 40, 48, 41, 34,  # Coefficients 16-23
           27, 20, 13,  6,  7, 14, 21, 28,  # Coefficients 24-31
           35, 42, 49, 56, 57, 50, 43, 36,
           29, 22, 15, 23, 30, 37, 44, 51,
           58, 59, 52, 45, 38, 31, 39, 46,
           53, 60, 61, 54, 47, 55, 62, 63
        ]
        
        with torch.no_grad():
            # Initialize weights to select top-K coefficients
            weight = self.dct_coefficient_selector.weight  # [K, 64, 1, 1]
            weight.zero_()
            
            # Set to identity for top-K coefficients
            for i in range(self.num_dct_coeffs):
                coeff_idx = zig_zag_order[i]
                weight[i, coeff_idx, 0, 0] = 1.0
    
    def forward(self, motion_vectors=None, dct_residuals=None):
        """
        Forward pass with spatial alignment
        
        Args:
            motion_vectors: [B, mv_channels, 60, 60] or None if MV disabled
            dct_residuals: [B, 120, 120, dct_channels] or None if DCT disabled
                          NOTE: spatial dims first!
        
        Returns:
            fused_features: [B, 64, 120, 120] - Spatially aligned features
            global_features: [B, 64] - Global context
        """
        features_to_fuse = []
        
        # ============================================================
        # MOTION VECTOR BRANCH (if enabled)
        # ============================================================
        if self.use_mv:
            if motion_vectors is None:
                raise ValueError("motion_vectors is required when use_mv=True")
            mv_features = self.mv_encoder(motion_vectors)  # [B, 32, 60, 60]
            mv_features = self.mv_upsample(mv_features)    # [B, 32, 120, 120]
            mv_features = self.mv_refine(mv_features)      # [B, mv_feature_dim, 120, 120]
            features_to_fuse.append(mv_features)
        
        # ============================================================
        # DCT RESIDUAL BRANCH (if enabled)
        # ============================================================
        if self.use_dct:
            if dct_residuals is None:
                raise ValueError("dct_residuals is required when use_dct=True")
            # Rearrange DCT input to [B, C, H, W] format
            # Input: [B, 120, 120, dct_channels] → Output: [B, dct_channels, 120, 120]
            dct_residuals = dct_residuals.permute(0, 3, 1, 2)
            
            # Select top-K coefficients (if coefficient selector is used)
            if self.dct_coefficient_selector is not None:
                dct_selected = self.dct_coefficient_selector(dct_residuals)  # [B, num_dct_coeffs, 120, 120]
            else:
                dct_selected = dct_residuals  # Use all provided coefficients
            
            dct_features = self.dct_encoder(dct_selected)  # [B, dct_feature_dim, 120, 120]
            features_to_fuse.append(dct_features)
        
        # ============================================================
        # SPATIAL FUSION
        # ============================================================
        # Concatenate enabled branches along channel dim
        combined = torch.cat(features_to_fuse, dim=1)  # [B, fusion_input_dim, 120, 120]
        fused_features = self.fusion(combined)         # [B, fused_feature_dim, 120, 120]
        
        # Global features for object-level context
        global_features = self.global_pool(fused_features)  # [B, fused_feature_dim, 1, 1]
        global_features = global_features.squeeze(-1).squeeze(-1)  # [B, fused_feature_dim]
        
        return fused_features, global_features
