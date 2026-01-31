#!/usr/bin/env python3
"""
MV-Center Training Script

Modular training pipeline for MV-Center architecture supporting:
- Version 1: Motion-only baseline (u,v motion vectors)
- Version 2: Motion + residuals (future expansion)
- Version 3: Motion + residuals + SSM tracking (future expansion)

Author: GitHub Copilot
Date: October 2024
"""

import argparse
import os
import sys
import json
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Add project paths
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import MV-Center components (only what we need)
try:
    from mots_exp.models.mv_center import (
        create_mv_center_v1_motion_only,
        create_mv_center_v1_with_magnitude,
        create_mv_center_v1_with_embeddings,
        create_model_from_config
    )
except ImportError:
    # These may not exist - we'll just use memory tracker
    pass

# Import memory-based tracker
from mots_exp.models.mv_center.mv_center_memory import (
    create_mv_center_memory_tracker,
    MVCenterMemoryLoss
)

# Import DCT-MV model
from mots_exp.models.dct_mv_center import DCTMVCenterTracker

# Import Fast DCT-MV model (without ROI and attention for maximum speed)
from mots_exp.models.dct_mv_center.fast_dct_mv_tracker import FastDCTMVTracker, UltraFastDCTMVTracker

# ✅ Import Improved Fast DCT-MV model (with box-aligned motion features)
from mots_exp.models.dct_mv_center.improved_fast_tracker import ImprovedFastDCTMVTracker

# Import MV-Only model (lightweight, motion-vector-only)
from mots_exp.models.mv_only_tracker import MVOnlyTracker

# Import Enhanced MV-Only model (with Transformer + Mamba interaction)
from mots_exp.models.mv_only_tracker_enhanced import MVOnlyTrackerEnhanced

# Import metrics
from mots_exp.metrics import simple_map

# Import dataset utilities
try:
    # Add the dataset directory to Python path
    dataset_dir = os.path.join(project_root, 'dataset')
    if dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)
    
    # Import the dataset factory using the same approach as the test script
    from dataset.factory.dataset_factory import create_mots_dataset
    print("✅ Dataset factory imported successfully")
    HAS_REAL_DATASET = True
except ImportError as e:
    print(f"⚠️ Dataset factory import failed: {e}")
    HAS_REAL_DATASET = False
    create_mots_dataset = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MV-Center Modular Training")
    
    # Model selection
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2', 'v3', 'memory'],
                       help='MV-Center version (v1: motion-only, v2: +residuals, v3: +SSM, memory: LSTM tracker)')
    parser.add_argument('--config', type=str, default='standard', choices=['tiny', 'standard', 'tracking'],
                       help='Model configuration preset')
    parser.add_argument('--use-magnitude', action='store_true',
                       help='Use 3-channel motion vectors (u,v,magnitude) instead of 2-channel (u,v)')
    parser.add_argument('--use-embeddings', action='store_true',
                       help='Enable embedding prediction for tracking')
    parser.add_argument('--use-memory', action='store_true',
                       help='Use memory-based LSTM tracker instead of dense heatmap')
    
    # DCT-MV model parameters
    parser.add_argument('--use-dct', action='store_true',
                       help='✨ Use DCT-MV model with DCT residuals + motion vectors')
    parser.add_argument('--use-mv-only', action='store_true',
                       help='🚀 Use MV-Only model (motion vectors only, 73x less memory than DCT-MV)')
    parser.add_argument('--use-mv-enhanced', action='store_true',
                       help='🔥 Use Enhanced MV-Only model (128-dim encoder, 256-dim ROI, Transformer+Mamba interaction, occlusion handling!)')
    parser.add_argument('--num-dct-coeffs', type=int, default=16,
                       help='Number of DCT coefficients to use (default: 16 for top frequencies)')
    
    # Ablation study parameters
    parser.add_argument('--dct-coeffs', type=int, default=64, choices=[0, 8, 16, 32, 64],
                       help='🔬 Number of DCT coefficients to load from dataset (0=disable DCT, 8/16/32/64 for ablation study)')
    parser.add_argument('--no-mv', action='store_true',
                       help='🔬 Disable motion vector encoder (for ablation study: DCT-only model)')
    
    # Detection loss with no-object classification (DETR-style)
    parser.add_argument('--use-detection-loss', action='store_true',
                       help='✨ Use DETR-style detection loss with Hungarian matching and no-object classification')
    parser.add_argument('--class-weight', type=float, default=2.0,
                       help='Classification loss weight (object vs no-object)')
    parser.add_argument('--no-object-weight', type=float, default=0.1,
                       help='No-object classification weight (for unmatched predictions)')
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                       help='Use Focal Loss instead of BCE for classification (handles class imbalance)')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                       help='Focal loss alpha parameter (weight for positive class)')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                       help='Focal loss gamma parameter (focusing on hard examples)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,  # ⬆️ Increased 2e-4 → 1e-3 (5x faster learning)
                       help='Learning rate (default: 1e-3 for faster convergence)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                       help='Weight decay (default: 0.05 as per MV-Center paper)')
    
    # Loss weights (MV-Center V1 specification)
    parser.add_argument('--center-weight', type=float, default=4.0,
                       help='Center heatmap loss weight')
    parser.add_argument('--box-weight', type=float, default=5.0,  # ⬆️ YOLO-style: bbox loss dominates
                       help='Box regression L1 loss weight')
    parser.add_argument('--giou-weight', type=float, default=2.0,  # ✅ Enabled for detection loss
                       help='GIoU loss weight (geometric IoU for box regression)')
    parser.add_argument('--conf-weight', type=float, default=0.0,  # ❌ Disabled - not helping
                       help='Confidence loss weight')
    
    # Memory tracker specific loss weights
    # velocity loss removed to simplify training pipeline
    parser.add_argument('--id-weight', type=float, default=0.5,  # 🆔 ID loss for object identity
                       help='ID classification loss weight (max 1.0)')
    parser.add_argument('--use-dynamic-balancing', action='store_true',
                       help='Use dynamic loss balancing to automatically adjust weights')
    
    # Memory tracker parameters
    parser.add_argument('--feature-dim', type=int, default=64,  # 🔻 Reduced 128→64 (smaller, faster)
                       help='Feature dimension for memory tracker')
    parser.add_argument('--hidden-dim', type=int, default=128,  # 🔻 Reduced 256→128 (smaller LSTM)
                       help='LSTM hidden dimension for memory tracker')
    parser.add_argument('--max-objects', type=int, default=100,
                       help='Maximum trackable objects for memory tracker')
    parser.add_argument('--gop-length', type=int, default=10,
                       help='GOP sequence length for memory tracker (number of P-frames)')
    parser.add_argument('--max-gops', type=int, default=70,
                       help='Maximum number of GOPs to load in RAM for training (default: 70, balanced across videos)')
    parser.add_argument('--max-val-gops', type=int, default=30,
                       help='Maximum number of GOPs to load for validation (default: 30, balanced across videos)')
    parser.add_argument('--use-roi-align', action='store_true',
                       help='Use ROI Align for spatially-aligned motion features (KEY IMPROVEMENT)')
    parser.add_argument('--roi-size', type=int, default=7,
                       help='ROI Align output size (e.g., 7 for 7x7 grid)')
    
    # Multi-scale ROI parameters
    parser.add_argument('--use-multiscale-roi', action='store_true',
                       help='🔍 Enable multi-scale ROI extractor (3x3, 7x7, 11x11) for better feature extraction')
    parser.add_argument('--roi-sizes', type=int, nargs='+', default=[3, 7, 11],
                       help='ROI sizes for multi-scale extractor (default: [3, 7, 11])')
    parser.add_argument('--use-global-context', action='store_true', default=True,
                       help='Include global context in multi-scale ROI (default: True)')
    parser.add_argument('--use-parallel-heads', action='store_true',
                       help='🚀 Use parallel heads multi-scale (NO compression, higher performance!)')
    parser.add_argument('--use-attention', action='store_true',
                       help='🎯 Add multi-head attention to parallel heads (best performance!)')
    parser.add_argument('--attention-heads', type=int, default=4,
                       help='Number of attention heads for multi-head attention')
    
    # Fast architecture (without ROI and attention)
    parser.add_argument('--use-fast', action='store_true',
                       help='⚡ Use fast architecture WITHOUT ROI extraction and attention (maximum speed)')
    parser.add_argument('--use-ultra-fast', action='store_true',
                       help='⚡⚡ Use ultra-fast architecture WITHOUT ROI, attention, AND LSTM (feedforward only)')
    parser.add_argument('--box-embedding-dim', type=int, default=32,
                       help='Box embedding dimension for fast architecture (default: 32)')
    
    # Enhanced architecture parameters (NEW!)
    parser.add_argument('--use-id-loss', action='store_true',
                       help='✨ Enable ID embedding loss for identity-aware tracking')
    parser.add_argument('--use-negative-sampling', action='store_true',
                       help='✨ Enable negative sampling for background penalization')
    # ❌ Removed duplicate --id-weight (already defined above at line 104)
    parser.add_argument('--negative-weight', type=float, default=0.5,
                       help='Weight for negative sampling loss')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Dimension of ID embeddings')
    parser.add_argument('--n-negative-samples', type=int, default=20,
                       help='Number of background samples per frame for negative sampling')
    parser.add_argument('--remove-velocity-loss', action='store_true',
                       help='Remove redundant velocity loss (set velocity_weight=0)')
    parser.add_argument('--use-enhanced-model', action='store_true',
                       help='Use enhanced model with ID embeddings (automatically enabled with --use-id-loss)')
    parser.add_argument('--use-full-bbox', action='store_true',
                       help='🆕 Use full bbox prediction with CNN backbone (predicts [cx,cy,w,h] directly)')
    parser.add_argument('--use-backbone', action='store_true', default=True,
                       help='Use CNN backbone for MV processing (default: True)')
    parser.add_argument('--center-weight-bbox', type=float, default=1.0,
                       help='Weight for center loss in full bbox prediction')
    parser.add_argument('--size-weight-bbox', type=float, default=1.0,
                       help='Weight for size loss in full bbox prediction')
    
    # Data parameters
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum training samples (for testing)')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum sequences to process')
    parser.add_argument('--train-split', type=float, default=0.5,
                       help='Fraction of sequences to use for training (default 0.5)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Fraction of sequences to use for validation (default 0.2)')
    parser.add_argument('--resolution', type=int, default=960,
                       help='Video resolution (640 or 960)')
    
    # Training options
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loader workers')
    parser.add_argument('--amp', action='store_true',
                       help='Use Automatic Mixed Precision training')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    
    # Output and logging
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log-freq', type=int, default=10,
                       help='Log training metrics every N steps')
    
    # Video generation
    parser.add_argument('--generate-videos', action='store_true',
                       help='Generate tracking videos at the end of training')
    parser.add_argument('--video-output-dir', type=str, default=None,
                       help='Output directory for videos (default: {output_dir}/videos)')
    parser.add_argument('--max-video-gops', type=int, default=2,
                       help='Maximum number of GOPs to generate videos for (default: 2)')
    
    # Debugging and testing
    parser.add_argument('--dry-run', action='store_true',
                       help='Test configuration without training')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize training samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    return device


def create_model(args, device):
    """Create MV-Center model based on arguments."""
    print(f"🧠 Creating MV-Center {args.version.upper()} model...")
    
    # 🚀 MV-Only model (motion vectors only, ultra-lightweight)
    if args.use_mv_only:
        print("   🚀 Using MV-Only lightweight model")
        print("      - Motion vectors ONLY (no DCT residuals)")
        print(f"      - Feature dim: {args.feature_dim}, Hidden dim: {args.hidden_dim}")
        print(f"      - Memory: ~7 MB per 3 GOPs (vs 512 MB for DCT-MV)")
        print(f"      - Expected speedup: 73x less memory, same or better FPS")
        
        # Multi-scale configuration (same as DCT-MV for consistency)
        if args.use_parallel_heads:
            print(f"      🚀 PARALLEL HEADS Multi-Scale: scales={args.roi_sizes}")
            print(f"      ✨ NO COMPRESSION: Each scale processed independently!")
            if args.use_attention:
                print(f"      🎯 WITH ATTENTION: {args.attention_heads} heads for scale fusion")
        elif args.use_multiscale_roi:
            print(f"      🔍 Multi-scale ROI: scales={args.roi_sizes}, global_context={args.use_global_context}")
        
        model = MVOnlyTracker(
            mv_feature_dim=64,  # Enhanced from DCT-MV's 32 (more capacity for MV-only)
            hidden_dim=args.hidden_dim,
            lstm_layers=2,
            dropout=0.1,
            image_size=args.resolution,
            use_multiscale_roi=args.use_multiscale_roi,
            roi_sizes=args.roi_sizes,
            use_parallel_heads=args.use_parallel_heads,
            use_attention=args.use_attention,
            attention_heads=args.attention_heads,
            # Pass unused params for compatibility
            num_dct_coeffs=None,
            dct_feature_dim=None,
            fused_feature_dim=args.feature_dim,
            roi_feature_dim=args.feature_dim,
        )
        model = model.to(device)
        
        # Use same loss function selection as DCT-MV
        if args.use_detection_loss:
            print("   🎯 Using DETR-style detection loss:")
            print(f"      - Hungarian matching for optimal object assignment")
            print(f"      - Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
            print(f"      - Box weight: {args.box_weight}, GIoU weight: {args.giou_weight}")
            print(f"      - Class weight: {args.class_weight}, No-object weight: {args.no_object_weight}")
            
            from mots_exp.models.mv_center.components.detection_loss import DetectionLossWithNoObject
            criterion = DetectionLossWithNoObject(
                box_weight=args.box_weight,
                class_weight=args.class_weight,
                giou_weight=args.giou_weight,
                no_object_weight=args.no_object_weight,
                velocity_weight=0.0,  # Not used in detection loss
                use_focal_loss=args.use_focal_loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
        else:
            print("   📝 Using standard MVCenterMemoryLoss")
            from mots_exp.models.mv_center.mv_center_memory import MVCenterMemoryLoss
            criterion = MVCenterMemoryLoss(
                box_weight=args.box_weight,
                velocity_weight=args.velocity_weight,
                conf_weight=args.conf_weight,
                use_dynamic_balancing=args.use_dynamic_balancing
            )
        
        return model, criterion
    
    # 🔥 Enhanced MV-Only model (with Transformer + Mamba interaction!)
    if args.use_mv_enhanced:
        print("   🔥 Using Enhanced MV-Only model with Interaction Module")
        print("      - Enhanced Encoder: 128-dim features (2× original, NO pooling!)")
        print("      - Enhanced ROI: 256-dim output (4× original!)")
        print("      - Transformer Interaction: Spatial attention for occlusion/collision")
        print("      - Mamba Interaction: Temporal state-space for persistent tracking")
        print("      - Enhanced LSTM: 256 hidden dim (2× original)")
        print(f"      - Memory: ~25 MB per 3 GOPs (vs 512 MB for DCT-MV, still 20× less!)")
        print(f"      - Model size: ~23 MB (vs 126 MB for RT-DETR, 5× smaller!)")
        print(f"      - Handles: Occlusions, Collisions, Object lifecycle")
        
        model = MVOnlyTrackerEnhanced(
            mv_feature_dim=128,      # 2× original (64 → 128)
            roi_feature_dim=256,     # 4× original (64 → 256)
            hidden_dim=256,          # 2× original (128 → 256)
            lstm_layers=2,
            dropout=0.1,
            max_objects=100,         # State pool size
            confidence_threshold=0.1  # Auto-remove low confidence objects
        )
        print(f"🔧 DEBUG: Before .to(device): conv1 weights = min={model.mv_encoder.conv1.weight.min():.4f}, max={model.mv_encoder.conv1.weight.max():.4f}")
        model = model.to(device)
        print(f"🔧 DEBUG: After .to(device): conv1 weights = min={model.mv_encoder.conv1.weight.min():.4f}, max={model.mv_encoder.conv1.weight.max():.4f}")
        
        # Use detection loss (recommended for enhanced model)
        if args.use_detection_loss:
            print("   🎯 Using DETR-style detection loss:")
            print(f"      - Hungarian matching for optimal object assignment")
            print(f"      - Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
            print(f"      - Box weight: {args.box_weight}, GIoU weight: {args.giou_weight}")
            print(f"      - Class weight: {args.class_weight}, No-object weight: {args.no_object_weight}")
            
            from mots_exp.models.mv_center.components.detection_loss import DetectionLossWithNoObject
            criterion = DetectionLossWithNoObject(
                box_weight=args.box_weight,
                class_weight=args.class_weight,
                giou_weight=args.giou_weight,
                no_object_weight=args.no_object_weight,
                velocity_weight=0.0,
                use_focal_loss=args.use_focal_loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
        else:
            print("   📝 Using standard MVCenterMemoryLoss")
            from mots_exp.models.mv_center.mv_center_memory import MVCenterMemoryLoss
            criterion = MVCenterMemoryLoss(
                box_weight=args.box_weight,
                velocity_weight=getattr(args, 'velocity_weight', 1.0),  # Default to 1.0 if not set
                conf_weight=args.conf_weight,
                use_dynamic_balancing=getattr(args, 'use_dynamic_balancing', False)
            )
        
        return model, criterion
    
    # 🆕 DCT-MV model (spatially-aligned DCT + MV)
    if args.use_dct:
        # ✅ ABLATION FIX: Calculate actual DCT channels for display
        dct_display = getattr(args, 'dct_coeffs', 64) if not args.use_mv_only else 0
        
        # Determine model input channels for ablation study
        mv_channels = 0 if getattr(args, 'no_mv', False) else 2  # 0 if --no-mv, else 2
        dct_coeffs_val = getattr(args, 'dct_coeffs', 64)         # Number of DCT coeffs from dataset
        dct_channels = dct_coeffs_val if (args.use_dct and not args.use_mv_only) else 0
        
        # ⚡ Check if using fast architecture (no ROI, no attention)
        if args.use_ultra_fast:
            print("   ⚡⚡ Using ULTRA-FAST DCT-MV model (no ROI, no attention, no LSTM)")
            print(f"      - DCT residuals: {dct_display} coefficients")
            print(f"      - Motion vectors: {'disabled' if args.no_mv else 'enabled'}")
            print(f"      - Architecture: Global pooling + Feedforward network")
            print(f"      - Expected speedup: ~3-5x faster than standard model")
            print(f"      🔬 Ablation config: mv_channels={mv_channels}, dct_channels={dct_channels}")
            
            model = UltraFastDCTMVTracker(
                num_dct_coeffs=dct_channels,
                mv_channels=mv_channels,
                dct_channels=dct_channels,
                mv_feature_dim=32,
                dct_feature_dim=32,
                fused_feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                dropout=0.1,
                image_size=args.resolution,
                box_embedding_dim=getattr(args, 'box_embedding_dim', 32)
            )
        elif args.use_fast:
            print("   ⚡ Using IMPROVED FAST DCT-MV model (box-aligned motion features)")
            print(f"      - DCT residuals: {dct_display} coefficients")
            print(f"      - Motion vectors: {'disabled' if args.no_mv else 'enabled with BOX-ALIGNED extraction'}")
            print(f"      - Architecture: BoxAlignedMotionEncoder + Simple LSTM")
            print(f"      - ✅ FIX: No global pooling! Each box gets different motion features")
            print(f"      - Expected speedup: ~2-3x faster than standard model")
            print(f"      🔬 Ablation config: mv_channels={mv_channels}, dct_channels={dct_channels}")
            
            model = ImprovedFastDCTMVTracker(
                num_dct_coeffs=dct_channels,
                mv_channels=mv_channels,
                dct_channels=dct_channels,
                mv_feature_dim=64,  # ✅ Match BoxAlignedMotionEncoder output
                dct_feature_dim=32,
                fused_feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                lstm_layers=1,  # Simpler LSTM
                dropout=0.1,
                image_size=args.resolution,
                box_embedding_dim=getattr(args, 'box_embedding_dim', 32)
            )
        else:
            print("   ✨ Using DCT-MV spatially-aligned model")
            print(f"      - DCT residuals: {dct_display} coefficients (ablation study)")
            print(f"      - Motion vectors: {'disabled' if getattr(args, 'no_mv', False) else 'upsampled to match DCT resolution'}")
            print(f"      - ROI feature dim: {args.feature_dim}, Hidden dim: {args.hidden_dim}")
            
            # Multi-scale configuration
            if args.use_parallel_heads:
                print(f"      🚀 PARALLEL HEADS Multi-Scale: scales={args.roi_sizes}")
                print(f"      ✨ NO COMPRESSION: Each scale processed independently!")
                print(f"      📊 Expected params: ~485K (+4%), Target mAP: ~42%")
                if args.use_attention:
                    print(f"      🎯 WITH ATTENTION: {args.attention_heads} heads for scale fusion")
                    print(f"      📈 Expected mAP boost: +3-5% (target: ~45-47%)")
            elif args.use_multiscale_roi:
                print(f"      🔍 Multi-scale ROI: scales={args.roi_sizes}, global_context={args.use_global_context}")
                print(f"      📊 Expected param increase: ~7% (+48K), Speed: 2.5-3 GOP/s (still 50x faster than RGB)")
            
            print(f"      🔬 Ablation config: mv_channels={mv_channels}, dct_channels={dct_channels}")
            
            model = DCTMVCenterTracker(
                num_dct_coeffs=dct_channels,        # ✅ ABLATION FIX: Use dct_channels (not old args.num_dct_coeffs)
                mv_channels=mv_channels,            # ✨ NEW: 0 for DCT-only, 2 for MV+DCT
                dct_channels=dct_channels,          # ✨ NEW: 0 for MV-only, 8/16/32/64 for DCT variants
                mv_feature_dim=32,
                dct_feature_dim=32,
                fused_feature_dim=args.feature_dim,
                roi_feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                lstm_layers=2,
                dropout=0.1,
                image_size=args.resolution,
                use_multiscale_roi=args.use_multiscale_roi,
                roi_sizes=args.roi_sizes,
                use_parallel_heads=args.use_parallel_heads,  # Parallel heads support
                use_attention=args.use_attention,            # NEW: Attention support
                attention_heads=args.attention_heads         # Number of attention heads
            )
        model = model.to(device)
        
        # Choose loss function based on args
        if args.use_detection_loss:
            print("   🎯 Using DETR-style detection loss:")
            print(f"      - Hungarian matching for optimal object assignment")
            print(f"      - Focal Loss (α={args.focal_alpha}, γ={args.focal_gamma})")
            print(f"      - Box weight: {args.box_weight}, GIoU weight: {args.giou_weight}")
            print(f"      - Class weight: {args.class_weight}, No-object weight: {args.no_object_weight}")
            
            from mots_exp.models.mv_center.components.detection_loss import DetectionLossWithNoObject
            criterion = DetectionLossWithNoObject(
                box_weight=args.box_weight,
                class_weight=args.class_weight,
                giou_weight=args.giou_weight,
                no_object_weight=args.no_object_weight,
                velocity_weight=0.0,  # Not used in detection loss
                use_focal_loss=args.use_focal_loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma
            )
        else:
            print("   📝 Using standard MVCenterMemoryLoss")
            from mots_exp.models.mv_center.mv_center_memory import MVCenterMemoryLoss
            criterion = MVCenterMemoryLoss(
                box_weight=args.box_weight,
                velocity_weight=getattr(args, 'velocity_weight', 1.0),
                conf_weight=args.conf_weight,
                use_dynamic_balancing=getattr(args, 'use_dynamic_balancing', False)
            )
        
        return model, criterion
    
    # Auto-enable enhanced model if ID loss or negative sampling requested
    use_enhanced = args.use_enhanced_model or args.use_id_loss or args.use_negative_sampling
    
    # velocity loss removed; --remove-velocity-loss flag kept for backward compatibility (no-op)
    
    # Memory-based tracker (new architecture)
    if args.use_memory or args.version == 'memory':
        # 🆕 NEW: Full bbox prediction model with CNN backbone
        if args.use_full_bbox:
            print("   ✨ Using FULL BBOX prediction model with CNN backbone")
            print(f"      - Predicts [cx, cy, w, h] directly (not just deltas)")
            print(f"      - CNN backbone for global MV context")
            print(f"      - Feature dim: {args.feature_dim}, Hidden dim: {args.hidden_dim}")
            print(f"      - Backbone enabled: {args.use_backbone}")
            
            from mots_exp.models.mv_center.mv_center_memory_fullbox import (
                MVCenterMemoryTrackerFullBox, MVCenterMemoryLossFullBox
            )
            
            model = MVCenterMemoryTrackerFullBox(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                max_objects=args.max_objects,
                grid_size=40,
                image_size=args.resolution,
                use_backbone=args.use_backbone
            )
            model = model.to(device)
            
            # Create full bbox loss function
            criterion = MVCenterMemoryLossFullBox(
                box_weight=args.box_weight,
                conf_weight=args.conf_weight,
                id_weight=args.id_weight,
                center_weight=args.center_weight_bbox,
                size_weight=args.size_weight_bbox
            )
            
            # Return both model and criterion
            return model, criterion
        
        # Choose standard or enhanced model
        if use_enhanced:
            print("   ✨ Using ENHANCED Memory-based LSTM tracker")
            from mots_exp.models.mv_center.mv_center_memory_enhanced import (
                MVCenterMemoryTrackerEnhanced, MVCenterMemoryLossEnhanced
            )
            
            if args.use_roi_align:
                print(f"      📝 With ROI Align (size: {args.roi_size}x{args.roi_size})")
            if args.use_id_loss:
                print(f"      📝 With ID embedding loss (dim: {args.embedding_dim})")
            if args.use_negative_sampling:
                print(f"      📝 With negative sampling ({args.n_negative_samples} samples)")
            
            model = MVCenterMemoryTrackerEnhanced(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                max_objects=args.max_objects,
                grid_size=40,  # Fixed for motion vector grid
                image_size=args.resolution,
                use_roi_align=args.use_roi_align,
                roi_size=(args.roi_size, args.roi_size),
                use_id_embedding=args.use_id_loss,
                embedding_dim=args.embedding_dim
            )
            model = model.to(device)
            
            # Create enhanced loss function
            criterion = MVCenterMemoryLossEnhanced(
                box_weight=args.box_weight,
                velocity_weight=args.velocity_weight,
                conf_weight=args.conf_weight,
                id_weight=args.id_weight,
                negative_weight=args.negative_weight,
                use_id_loss=args.use_id_loss,
                use_negative_sampling=args.use_negative_sampling,
                n_negative_samples=args.n_negative_samples,
                use_dynamic_balancing=args.use_dynamic_balancing
            )
        else:
            # Standard model (original)
            if args.use_roi_align:
                print("   📝 Memory-based LSTM tracker with ROI Align ✨")
                print(f"      ROI size: {args.roi_size}x{args.roi_size}")
            else:
                print("   📝 Memory-based LSTM tracker (standard)")
            model = create_mv_center_memory_tracker(
                feature_dim=args.feature_dim,
                hidden_dim=args.hidden_dim,
                max_objects=args.max_objects,
                grid_size=40,  # Fixed for 640x640 images with 16x16 blocks
                image_size=args.resolution,
                use_roi_align=args.use_roi_align,
                roi_size=(args.roi_size, args.roi_size)
            )
            model = model.to(device)
            
            # Create memory-specific loss function
            criterion = MVCenterMemoryLoss(
                box_weight=args.box_weight,
                velocity_weight=args.velocity_weight,
                conf_weight=args.conf_weight,
                use_dynamic_balancing=args.use_dynamic_balancing
            )
        
        # Return both model and criterion for memory tracker
        return model, criterion
    
    # Determine input channels
    input_channels = 3 if args.use_magnitude else 2
    
    # Override embeddings for V1 if explicitly requested
    use_embeddings = args.use_embeddings
    
    if args.version == 'v1':
        # MV-Center V1: Motion-only
        if args.use_magnitude:
            if use_embeddings:
                print("   📝 V1 with magnitude and embeddings")
                model = create_mv_center_v1_with_embeddings()
                # Override input channels for magnitude
                model.backbone = model.backbone.__class__(input_channels=3)
            else:
                print("   📝 V1 with magnitude (u,v,mag)")
                from mots_exp.models.mv_center import create_mv_center_v1
                model = create_mv_center_v1(
                    input_channels=3,
                    fpn_type=args.config if args.config in ['tiny', 'simple'] else 'tiny',
                    use_embeddings=use_embeddings
                )
        else:
            if use_embeddings:
                print("   📝 V1 motion-only with embeddings")
                model = create_mv_center_v1_with_embeddings()
            else:
                print("   📝 V1 motion-only baseline")
                model = create_mv_center_v1_motion_only()
    
    elif args.version == 'v2':
        # MV-Center V2: Motion + residuals (future implementation)
        raise NotImplementedError("MV-Center V2 not implemented yet")
    
    elif args.version == 'v3':
        # MV-Center V3: Motion + residuals + SSM (future implementation)
        raise NotImplementedError("MV-Center V3 not implemented yet")
    
    model = model.to(device)
    
    # Setup loss function
    model.set_loss_function(
        center_weight=args.center_weight,
        box_weight=args.box_weight,
        giou_weight=args.giou_weight,
        conf_weight=args.conf_weight,
        use_confidence=False  # Start without confidence loss
    )
    
    # Print model info (if available)
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"   📊 Model: {info['model_name']} {info['version']}")
        print(f"   📊 Parameters: {info['total_parameters']:,} ({info['total_parameters']/1000:.1f}K)")
    
    # Return model (and None for criterion for non-memory models)
    return model, None
    print(f"   📊 Est. FLOPs: {info['estimated_flops']:.2f}M ({info['estimated_flops']/1000:.3f} GFLOPs)")
    print(f"   📊 Input: {info['input_channels']} channels")
    
    # Check performance target
    gflops = info['estimated_flops'] / 1000.0
    if gflops < 0.3:
        print(f"   ✅ Performance target met: {gflops:.3f} < 0.3 GFLOPs")
    else:
        print(f"   ⚠️ Performance target exceeded: {gflops:.3f} > 0.3 GFLOPs")
    
    return model


def create_dataset(args):
    """Create training dataset - REAL DATA ONLY."""
    input_channels = 3 if args.use_magnitude else 2
    
    print(f"\n📊 Creating dataset...")
    print(f"   Input channels: {input_channels} ({'u,v,magnitude' if args.use_magnitude else 'u,v'})")
    
    if not HAS_REAL_DATASET or create_mots_dataset is None:
        raise RuntimeError("❌ Real MOTS dataset factory not available! Cannot train without real data.")
    
    print(f"   🔍 Loading real MOTS datasets (MOT15+MOT17+MOT20 at {args.resolution}x{args.resolution})...")
    print(f"   ⏳ This may take 60-90 seconds while scanning video sequences...")
    
    # Create combined training dataset with MOT15, MOT17, and MOT20
    # Determine DCT coefficient count and whether to load residuals
    dct_coeffs = getattr(args, 'dct_coeffs', 64)  # Default to 64 for backward compatibility
    load_mv = not getattr(args, 'no_mv', False)   # Load MV unless --no-mv is set
    load_residuals = args.use_dct and not args.use_mv_only and dct_coeffs > 0  # Only load if DCT enabled and coeffs > 0
    
    dataset = create_mots_dataset(
        dataset_type=['mot15', 'mot17', 'mot20'],  # Use all three datasets
        resolution=args.resolution,
        mode="train",
        load_iframe=False,
        load_pframe=False,
        load_motion_vectors=load_mv,
        load_residuals=load_residuals,
        dct_coeffs=dct_coeffs,  # ✨ NEW: Control number of DCT coefficients (for ablation study)
        load_annotations=True,
        sequence_length=48,
        data_format="separate",
        combine_datasets=True  # Combine all datasets into one
    )
    
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("❌ Real dataset is empty or failed to load! Cannot train without real data.")
    
    print(f"   ✅ Real MOTS datasets loaded: {len(dataset)} samples")
    print(f"   📝 Dataset type: {type(dataset).__name__}")
    
    return dataset


def generate_gaussian_heatmap(center, size, sigma=2):
    """Generate Gaussian heatmap for a single object."""
    h, w = size
    cx, cy = center
    
    # Create coordinate grids
    x = torch.arange(0, w, dtype=torch.float32)
    y = torch.arange(0, h, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate Gaussian
    gaussian = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    
    return gaussian


def generate_targets_from_boxes(boxes, labels, ids, motion_shape, input_size=640):
    """
    Generate P3/P4 targets from bounding boxes.
    
    Args:
        boxes: List of boxes in [x,y,w,h] format (normalized 0-1 or pixel coords)
        labels: List of class labels
        ids: List of object IDs
        motion_shape: Shape of motion vectors (C, H, W)
        input_size: Input image size
        
    Returns:
        targets: Dict with P3 and P4 levels
    """
    # Based on actual model output (for 40x40 input):
    # P3: 10x10 (stride ~4 in motion vector space)
    # P4: 5x5 (stride ~8 in motion vector space)
    
    targets = {
        'P3': {
            'center': torch.zeros(1, 10, 10),
            'box': torch.zeros(4, 10, 10),
            'mask': torch.zeros(1, 10, 10)
        },
        'P4': {
            'center': torch.zeros(1, 5, 5),
            'box': torch.zeros(4, 5, 5),
            'mask': torch.zeros(1, 5, 5)
        }
    }
    
    if len(boxes) == 0:
        return targets
    
    # Convert boxes to tensor if needed
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    
    # Process each box
    for i, box in enumerate(boxes):
        x, y, w, h = box
        
        # If boxes are normalized (0-1), convert to pixels
        if x <= 1.0 and y <= 1.0:
            x *= input_size
            y *= input_size
            w *= input_size
            h *= input_size
        
        # Calculate center
        cx = x + w / 2
        cy = y + h / 2
        
        # Assign to appropriate level based on box size
        box_area = w * h
        
        # Small/medium objects -> P3 (10x10), large objects -> P4 (5x5)
        if box_area < (input_size * 0.5) ** 2:  # < 50% of image
            # P3 level (10x10 grid)
            stride = input_size / 10
            cx_grid = cx / stride
            cy_grid = cy / stride
            
            if 0 <= cx_grid < 10 and 0 <= cy_grid < 10:
                # Add Gaussian heatmap
                gaussian = generate_gaussian_heatmap((cx_grid, cy_grid), (10, 10), sigma=2)
                targets['P3']['center'][0] = torch.maximum(targets['P3']['center'][0], gaussian)
                
                # Add box regression targets (offset + size)
                cx_int, cy_int = int(cx_grid), int(cy_grid)
                targets['P3']['box'][0, cy_int, cx_int] = (cx_grid - cx_int)  # x offset
                targets['P3']['box'][1, cy_int, cx_int] = (cy_grid - cy_int)  # y offset
                targets['P3']['box'][2, cy_int, cx_int] = torch.log(torch.tensor(w / stride + 1e-6))  # log width
                targets['P3']['box'][3, cy_int, cx_int] = torch.log(torch.tensor(h / stride + 1e-6))  # log height
                
                # Mark as valid
                targets['P3']['mask'][0, cy_int, cx_int] = 1.0
        else:
            # P4 level (5x5 grid)
            stride = input_size / 5
            cx_grid = cx / stride
            cy_grid = cy / stride
            
            if 0 <= cx_grid < 5 and 0 <= cy_grid < 5:
                # Add Gaussian heatmap
                gaussian = generate_gaussian_heatmap((cx_grid, cy_grid), (5, 5), sigma=1.5)
                targets['P4']['center'][0] = torch.maximum(targets['P4']['center'][0], gaussian)
                
                # Add box regression targets
                cx_int, cy_int = int(cx_grid), int(cy_grid)
                targets['P4']['box'][0, cy_int, cx_int] = (cx_grid - cx_int)
                targets['P4']['box'][1, cy_int, cx_int] = (cy_grid - cy_int)
                targets['P4']['box'][2, cy_int, cx_int] = torch.log(torch.tensor(w / stride + 1e-6))
                targets['P4']['box'][3, cy_int, cx_int] = torch.log(torch.tensor(h / stride + 1e-6))
                
                # Mark as valid
                targets['P4']['mask'][0, cy_int, cx_int] = 1.0
    
    return targets


def process_batch(batch, device, debug=False):
    """Process a batch of data for MV-Center training."""
    motion_vectors = []
    targets_list = []
    
    for sample in batch:
        if sample is None:
            continue
        
        # Extract motion vectors
        mv = sample['motion_vectors']
        if not isinstance(mv, torch.Tensor):
            mv = torch.tensor(mv, dtype=torch.float32)
        
        # Fix motion vector shape: [2, H, W, 2] -> [2, H, W] (u,v channels)
        if len(mv.shape) == 4:  # [2, H, W, 2]
            # Take the mean across the first dimension and extract u,v from last dimension
            mv_x = mv[:, :, :, 0].mean(dim=0)  # Average across 2 frames, get u
            mv_y = mv[:, :, :, 1].mean(dim=0)  # Average across 2 frames, get v
            mv = torch.stack([mv_x, mv_y], dim=0)  # Shape: [2, H, W]
        
        motion_vectors.append(mv)
        
        # Extract targets - generate from raw annotations
        boxes = sample.get('boxes', [])
        labels = sample.get('labels', [])
        ids = sample.get('ids', [])
        
        if debug and len(boxes) > 0:
            print(f"📦 Sample has {len(boxes)} boxes, generating targets...")
        
        # Generate targets from boxes
        targets = generate_targets_from_boxes(boxes, labels, ids, mv.shape)
        
        if debug and targets:
            num_p3 = targets['P3']['mask'].sum().item()
            num_p4 = targets['P4']['mask'].sum().item()
            print(f"   ✅ Generated targets: P3={int(num_p3)} objects, P4={int(num_p4)} objects")
        
        targets_list.append(targets)
    
    if not motion_vectors:
        if debug:
            print("⚠️ No motion vectors in batch")
        return None, None
    
    # Stack motion vectors
    motion_vectors = torch.stack(motion_vectors).to(device)
    
    # Collate targets by level
    batch_targets = {}
    if targets_list and targets_list[0]:
        if debug:
            print(f"📋 First target keys: {targets_list[0].keys()}")
        for level in ['P3', 'P4']:
            if level in targets_list[0]:
                batch_targets[level] = {}
                for key in ['center', 'box', 'mask']:
                    if key in targets_list[0][level]:
                        tensors = [t[level][key] for t in targets_list if level in t and key in t[level]]
                        if tensors:
                            batch_targets[level][key] = torch.stack(tensors).to(device)
                            if debug:
                                print(f"   ✅ {level}.{key}: shape {batch_targets[level][key].shape}")
    else:
        if debug:
            print("⚠️ No targets in targets_list")
    
    if debug:
        print(f"📊 Final batch_targets keys: {batch_targets.keys()}")
    
    return motion_vectors, batch_targets


def custom_collate_fn(batch):
    """Custom collate function to handle None samples."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch


def _train_on_gop_sequence(model, gop_frames, criterion, optimizer, scaler, args, device, progress_bar):
    """
    Train on a single GOP sequence.
    
    Args:
        model: Memory tracker model
        gop_frames: List of frame dicts with 'motion_vectors' and 'boxes'
        criterion: Loss function
        optimizer: Optimizer
        scaler: AMP scaler (or None)
        args: Training arguments
        device: Device
        progress_bar: Progress bar for updates
        
    Returns:
        loss_dict: Dictionary with loss components
    """
    if len(gop_frames) < 2:
        # Need at least 2 frames for velocity loss
        return None
    
    # Debug: Log GOP length occasionally
    if hasattr(_train_on_gop_sequence, 'gop_count'):
        _train_on_gop_sequence.gop_count += 1
    else:
        _train_on_gop_sequence.gop_count = 1
    
    if _train_on_gop_sequence.gop_count <= 3:
        print(f"\n🔍 DEBUG: GOP #{_train_on_gop_sequence.gop_count} has {len(gop_frames)} frames (expected ~48)")
    
    optimizer.zero_grad()
    
    # Check if this is a memory tracker or DCT-MV tracker
    is_memory_tracker = hasattr(model, 'reset') and hasattr(model, 'tracker')
    is_dct_mv_tracker = hasattr(model, 'dct_mv_encoder') or hasattr(model, 'encoder')
    
    # 🔄 RESET LSTM hidden state at start of GOP (memory tracker only)
    if is_memory_tracker:
        model.reset()
        
        # Frame 0: I-frame initialization
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        iframe_ids = gop_frames[0].get('ids', None)
        if iframe_ids is not None:
            iframe_ids = iframe_ids.to(device)
        model.tracker.init_from_iframe(iframe_boxes, iframe_ids)
    
    # For DCT-MV tracker, initialize hidden state
    hidden_state = None
    current_boxes = gop_frames[0]['boxes'].to(device) if is_dct_mv_tracker else None
    
    # Detect which loss function we're using
    from mots_exp.models.mv_center.components.detection_loss import DetectionLossWithNoObject
    use_detection_loss = isinstance(criterion, DetectionLossWithNoObject)
    
    # Collect predictions, targets, and optionally embeddings/logits across the GOP
    predictions = []
    targets = []
    logits_list = [] if use_detection_loss else []  # For new detection loss (raw logits)
    confidences_list = [] if not use_detection_loss else []  # For old loss (probabilities)
    embeddings_list = [] if hasattr(model, 'use_id_embedding') and model.use_id_embedding else None
    ids_list = [] if hasattr(model, 'use_id_embedding') and model.use_id_embedding else None
    motion_vectors_list = [] if hasattr(criterion, 'use_negative_sampling') and criterion.use_negative_sampling else None
    
    # Process P-frames sequentially (LSTM state persists across frames)
    for t in range(1, len(gop_frames)):
        # Get motion vectors for this P-frame
        # ✅ ABLATION FIX: motion_vectors can be None for DCT-only mode
        mv = gop_frames[t].get('motion_vectors', None)
        if mv is not None:
            mv = mv.to(device)
        target_boxes = gop_frames[t]['boxes'].to(device)
        
        # Get DCT residuals if available (for DCT-MV model)
        dct = gop_frames[t].get('residuals', None)
        if dct is not None:
            dct = dct.to(device)
            # Add batch dimension if needed: [H, W, C] → [1, H, W, C]
            if dct.ndim == 3:
                dct = dct.unsqueeze(0)
        
        # 🔍 DEBUG: Log motion vector properties for first few GOPs
        if _train_on_gop_sequence.gop_count <= 2 and t == 1:
            print(f"\n🔍 DEBUG Motion Vectors (GOP #{_train_on_gop_sequence.gop_count}, Frame {t}):")
            if mv is not None:
                print(f"   - Original mv.shape = {mv.shape}")
                print(f"   - mv dtype = {mv.dtype}")
                print(f"   - mv min/max = {mv.min():.4f} / {mv.max():.4f}")
                print(f"   - mv mean/std = {mv.mean():.4f} / {mv.std():.4f}")
                print(f"   - Unique values in mv = {torch.unique(mv).numel()}")
                print(f"   - Sample values (first 10 flattened): {mv.flatten()[:10].tolist()}")
            else:
                print(f"   - mv = None (DCT-only mode)")
            print(f"   - target_boxes.shape = {target_boxes.shape}")
            if dct is not None:
                print(f"   - DCT residuals.shape = {dct.shape}")
                print(f"   - DCT min/max = {dct.min():.4f} / {dct.max():.4f}")
        
        # Process motion vectors to correct shape
        # ✅ ABLATION FIX: Handle None motion vectors for DCT-only mode
        mv_processed = None
        if mv is not None:
            if mv.ndim == 4:
                # [T, H, W, 2] or [2, H, W, 2] → [2, H, W]
                mv_avg = mv.mean(dim=0)  # [H, W, 2]
                mv_processed = mv_avg.permute(2, 0, 1)  # [2, H, W]
            elif mv.ndim == 3:
                # [H, W, 2] → [2, H, W]
                mv_processed = mv.permute(2, 0, 1)
            else:
                # Already [2, H, W]
                mv_processed = mv
            
            # Add batch dimension for DCT-MV model and enhanced MV-only model: [2, H, W] → [1, 2, H, W]
            if is_dct_mv_tracker or args.use_mv_enhanced:
                mv_processed = mv_processed.unsqueeze(0)
        
        # 🔍 DEBUG: Log processed motion vectors
        if _train_on_gop_sequence.gop_count <= 2 and t == 1:
            if mv_processed is not None:
                print(f"   - Processed mv_processed.shape = {mv_processed.shape}")
                print(f"   - Processed mv_processed min/max = {mv_processed.min():.4f} / {mv_processed.max():.4f}")
                print(f"   - Processed unique values = {torch.unique(mv_processed).numel()}")
                print(f"   - Processed sample (channel 0, first 10): {mv_processed[0].flatten()[:10].tolist()}")
            else:
                print(f"   - mv_processed = None (DCT-only mode)")
        
        # Forward pass (LSTM state maintained from previous frame)
        # For DCT-MV model, pass both motion vectors and DCT residuals
        if args.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                if is_dct_mv_tracker:
                    # DCT-MV model: pass motion vectors, DCT (can be None for MV-only ablation), boxes, and hidden state
                    # Use logits for detection loss, probabilities for old loss
                    pred_boxes, conf, hidden_state = model.forward_single_frame(
                        mv_processed, dct, current_boxes, hidden_state,
                        return_logits=use_detection_loss  # ✅ Get logits for new loss
                    )
                    current_boxes = pred_boxes  # Update for next frame
                else:
                    # MV-only memory tracker
                    outputs = model.forward(mv_processed, mode='single_frame')
        else:
            if is_dct_mv_tracker:
                # DCT-MV model: pass motion vectors, DCT (can be None for MV-only ablation), boxes, and hidden state
                # Use logits for detection loss, probabilities for old loss
                pred_boxes, conf, hidden_state = model.forward_single_frame(
                    mv_processed, dct, current_boxes, hidden_state,
                    return_logits=use_detection_loss  # ✅ Get logits for new loss
                )
                current_boxes = pred_boxes  # Update for next frame
            else:
                # MV-only memory tracker
                outputs = model.forward(mv_processed, mode='single_frame')
        
        # Unpack outputs for memory tracker
        if not is_dct_mv_tracker:
            # Unpack outputs (may include embeddings for enhanced model)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                # Enhanced model: (boxes, confidences, embeddings)
                pred_boxes, conf, emb = outputs
                if embeddings_list is not None:
                    embeddings_list.append(emb)
            else:
                # Standard model: (boxes, confidences)
                pred_boxes, conf = outputs
        
        predictions.append(pred_boxes)
        targets.append(target_boxes)
        
        # Collect logits (for detection loss) or confidences (for old loss)
        if use_detection_loss:
            logits_list.append(conf)  # conf is raw logits when return_logits=True
        else:
            confidences_list.append(conf)  # conf is probabilities when return_logits=False
        
        # Collect IDs and motion vectors for enhanced loss
        if ids_list is not None:
            frame_ids = gop_frames[t].get('ids', None)
            if frame_ids is not None:
                ids_list.append(frame_ids.to(device))
        if motion_vectors_list is not None:
            motion_vectors_list.append(mv_processed)
    
    # ✅ CRITICAL: Ensure we have valid predictions before computing loss
    if len(predictions) == 0 or len(targets) == 0:
        gop_id = gop_frames[0].get('sequence_id', 'unknown') if len(gop_frames) > 0 else 'unknown'
        print(f"⚠️ INFO: GOP '{gop_id}' has empty annotations (predictions={len(predictions)}, targets={len(targets)}) - this is expected for some GOPs, skipping")
        return None
    
    # Ensure all predictions require gradients
    for i, pred in enumerate(predictions):
        if not pred.requires_grad:
            print(f"⚠️ WARNING: Prediction {i} does not require grad! This will cause backprop to fail.")
            # This should not happen - indicates a bug in forward pass
    
    # Compute loss over the entire sequence
    # Pass different arguments depending on which loss function we're using
    if use_detection_loss:
        # New detection loss: needs predictions, targets, and logits (raw outputs)
        loss_kwargs = {
            'predictions': predictions,
            'targets': targets,
            'pred_logits': logits_list  # ✅ Raw logits for FocalLoss/BCEWithLogitsLoss
        }
        loss, loss_dict = criterion(**loss_kwargs)  # ✅ Returns (loss, loss_dict) tuple
    else:
        # Old loss: needs predictions, targets, and confidences (probabilities)
        loss_kwargs = {
            'predictions': predictions,
            'targets': targets,
            'confidences': confidences_list
        }
        
        # Add enhanced loss arguments if using enhanced model
        if embeddings_list is not None and len(embeddings_list) > 0:
            loss_kwargs['embeddings'] = embeddings_list
        if ids_list is not None and len(ids_list) > 0:
            loss_kwargs['ids'] = ids_list
        if motion_vectors_list is not None and len(motion_vectors_list) > 0:
            loss_kwargs['motion_vectors'] = torch.stack(motion_vectors_list)  # [T, 2, H, W]
            loss_kwargs['model'] = model
            loss_kwargs['grid_size'] = 40
            loss_kwargs['image_size'] = args.resolution
        
        loss, loss_dict = criterion(**loss_kwargs)
    
    # ✅ CRITICAL: Check if loss requires gradients before backward
    if not loss.requires_grad:
        print(f"❌ ERROR: Loss does not require gradients!")
        print(f"   Loss value: {loss.item()}")
        print(f"   Number of predictions: {len(predictions)}")
        print(f"   Predictions require_grad: {[p.requires_grad for p in predictions[:3]]}")
        if use_detection_loss:
            print(f"   Number of logits: {len(logits_list)}")
            print(f"   Logits require_grad: {[l.requires_grad for l in logits_list[:3]]}")
        return None
    
    # Backward pass
    if args.amp and scaler is not None:
        scaler.scale(loss).backward()
        if args.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
    
    # Update progress bar
    postfix = {
        'loss': f'{loss.item():.6f}',
        'box': f'{loss_dict.get("box", 0.0) if isinstance(loss_dict.get("box"), (int, float)) else loss_dict["box"].item():.6f}',
        'vel': f'{loss_dict.get("velocity", 0.0) if isinstance(loss_dict.get("velocity"), (int, float)) else loss_dict.get("velocity", torch.tensor(0.0)).item():.6f}',
        'frames': len(gop_frames)
    }
    
    # Add ID and negative losses if present
    if 'id' in loss_dict:
        postfix['id'] = f'{loss_dict["id"] if isinstance(loss_dict["id"], (int, float)) else loss_dict["id"].item():.6f}'
    if 'negative' in loss_dict:
        postfix['neg'] = f'{loss_dict["negative"] if isinstance(loss_dict["negative"], (int, float)) else loss_dict["negative"].item():.6f}'
    
    progress_bar.set_postfix(postfix)
    
    result_dict = {
        'total': loss.item(),
        'box': loss_dict['box'].item() if hasattr(loss_dict['box'], 'item') else float(loss_dict['box']),
        'velocity': loss_dict.get('velocity', torch.tensor(0.0)).item() if hasattr(loss_dict.get('velocity', 0.0), 'item') else float(loss_dict.get('velocity', 0.0)),
        'confidence': loss_dict['confidence'] if isinstance(loss_dict['confidence'], (int, float)) else loss_dict['confidence'].item()
    }
    
    # Add detection loss components if present (GIoU, classification, no-object)
    if 'giou' in loss_dict:
        result_dict['giou'] = loss_dict['giou'].item() if hasattr(loss_dict['giou'], 'item') else float(loss_dict['giou'])
    if 'classification' in loss_dict:
        result_dict['classification'] = loss_dict['classification'].item() if hasattr(loss_dict['classification'], 'item') else float(loss_dict['classification'])
    if 'no_object' in loss_dict:
        result_dict['no_object'] = loss_dict['no_object'].item() if hasattr(loss_dict['no_object'], 'item') else float(loss_dict['no_object'])
    
    # Add ID and negative losses to result dict if present
    if 'id' in loss_dict:
        result_dict['id'] = loss_dict['id'] if isinstance(loss_dict['id'], (int, float)) else loss_dict['id'].item()
    if 'negative' in loss_dict:
        result_dict['negative'] = loss_dict['negative'] if isinstance(loss_dict['negative'], (int, float)) else loss_dict['negative'].item()
    
    # ⚠️ DISABLED: mAP calculation during training (too slow, now only in validation)
    # Set dummy values - real mAP will be calculated during validation
    result_dict['map'] = 0.0
    result_dict['static_baseline_map'] = 0.0
    
    # # OLD CODE: Calculate mAP@0.5 for this GOP (detach from computation graph for efficiency)
    # gop_map = 0.0
    # static_baseline_map = 0.0  # Using I-frame boxes for all frames
    # motion_baseline_map = 0.0  # Using I-frame boxes + mean motion translation
    # num_frames_with_boxes = 0
    # 
    # with torch.no_grad():
    #     # Get I-frame boxes (first frame predictions)
    #     iframe_boxes = predictions[0] if len(predictions) > 0 else None
    #     
    #     for frame_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
    #         if len(pred) > 0 and len(tgt) > 0:
    #             # Model mAP
    #             frame_map = simple_map(pred, tgt, thresh=0.5)
    #             gop_map += frame_map
    #             
    #             # Static baseline: use I-frame boxes for all frames
    #             if iframe_boxes is not None and len(iframe_boxes) > 0 and frame_idx > 0:
    #                 static_map = simple_map(iframe_boxes, tgt, thresh=0.5)
    #                 static_baseline_map += static_map
    #             
    #             # Motion baseline: I-frame boxes + average motion (simplified)
    #             # For training we skip this to save time (only do in validation)
    #             
    #             num_frames_with_boxes += 1
    # 
    # if num_frames_with_boxes > 0:
    #     result_dict['map'] = gop_map / num_frames_with_boxes
    #     # Compute static baseline mAP (excluding I-frame itself)
    #     if num_frames_with_boxes > 1:
    #         result_dict['static_baseline_map'] = static_baseline_map / (num_frames_with_boxes - 1)
    #     else:
    #         result_dict['static_baseline_map'] = 0.0
    # else:
    #     result_dict['map'] = 0.0
    #     result_dict['static_baseline_map'] = 0.0
    
    return result_dict


class GOPCache:
    """
    LRU cache for GOP sequences with memory-bounded storage.
    
    Loads GOPs on-demand and caches them for reuse across epochs.
    Automatically evicts least-recently-used GOPs when memory limit reached.
    """
    def __init__(self, max_gops=100):
        """
        Args:
            max_gops: Maximum number of GOPs to cache (default 100 for ~50-60% of dataset)
        """
        self.max_gops = max_gops
        self.cache = OrderedDict()  # Preserves insertion order for LRU
        self.hits = 0
        self.misses = 0
        
    def get(self, gop_id, load_fn):
        """
        Get GOP from cache or load it.
        
        Args:
            gop_id: Unique identifier for the GOP
            load_fn: Function to call if GOP not in cache (returns list of frames)
        
        Returns:
            List of frames for the GOP
        """
        if gop_id in self.cache:
            # Cache hit - move to end (most recently used)
            self.hits += 1
            self.cache.move_to_end(gop_id)
            return self.cache[gop_id]
        
        # Cache miss - load GOP
        self.misses += 1
        gop_frames = load_fn()
        
        # Add to cache
        self.cache[gop_id] = gop_frames
        
        # Evict oldest if over limit
        if len(self.cache) > self.max_gops:
            evicted_id = next(iter(self.cache))  # Get oldest key
            del self.cache[evicted_id]
        
        return gop_frames
    
    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cached_gops': len(self.cache),
            'max_gops': self.max_gops
        }
    
    def clear(self):
        """Clear cache and reset statistics"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


def load_gop_sequences_once(dataloader, gop_length=48, selected_indices=None):
    """
    Load GOP sequences from the dataloader ONCE at the start of training.
    
    Args:
        dataloader: DataLoader with the dataset
        gop_length: Length of each GOP in frames
        selected_indices: Optional list of GOP indices to load (for balanced sampling)
    
    Returns:
        Ordered list of (seq_id, frames) tuples that can be reused across epochs.
    """
    base_dataset = dataloader.dataset
    # Handle Subset wrapper
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    
    # Determine if DataLoader is using a Subset
    subset_indices = None
    try:
        from torch.utils.data import Subset
        if isinstance(dataloader.dataset, Subset):
            subset_indices = dataloader.dataset.indices
            base_dataset = dataloader.dataset.dataset
    except Exception:
        subset_indices = None

    # Build sequence index list to iterate in order
    if selected_indices is not None:
        # Use provided balanced selection
        seq_indices = selected_indices
        print(f"\n📦 Loading {len(seq_indices)} balanced GOP sequences once for all epochs...", flush=True)
    elif subset_indices is not None:
        seq_indices = sorted(set(idx // gop_length for idx in subset_indices))
        print(f"\n📦 Loading {len(seq_indices)} GOP sequences (from Subset) once for all epochs...", flush=True)
    else:
        seq_indices = list(range(len(base_dataset.sequences)))
        print(f"\n📦 Loading {len(seq_indices)} complete GOP sequences once for all epochs...", flush=True)

    # Load complete GOPs in parallel using threads (IO-bound)
    gop_sequences = {}
    seq_index_to_id = {}

    # Worker function to load one sequence (preserving frame order)
    def _load_sequence(seq_idx):
        sequence_info = base_dataset.sequences[seq_idx]
        sequence_id = f"{sequence_info['video_name']}_gop{sequence_info['gop_index']}"
        frames = []
        for frame_idx in range(gop_length):
            global_idx = seq_idx * gop_length + frame_idx
            if subset_indices is not None and global_idx not in subset_indices:
                continue
            try:
                sample = base_dataset[global_idx]
                mv = sample.get('motion_vectors')
                boxes = sample.get('boxes')
                residuals = sample.get('residuals')
                # ✅ ABLATION FIX: Allow frames with only DCT residuals (mv can be None)
                if boxes is None or (mv is None and residuals is None):
                    continue
                frame_data = {'motion_vectors': mv, 'boxes': boxes, 'frame_id': frame_idx}
                if residuals is not None:
                    frame_data['residuals'] = residuals
                frames.append(frame_data)
            except Exception:
                continue
        return seq_idx, sequence_id, frames

    max_workers = min(8, (os.cpu_count() or 4))
    futures = {}
    seq_bar = tqdm(total=len(seq_indices), desc="Loading GOPs", unit="GOP")
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for seq_idx in seq_indices:
            futures[exe.submit(_load_sequence, seq_idx)] = seq_idx

        for fut in as_completed(futures):
            seq_idx, sequence_id, frames = fut.result()
            seq_index_to_id[seq_idx] = sequence_id
            if len(frames) >= 2:
                gop_sequences[sequence_id] = frames
            seq_bar.update(1)
            seq_bar.set_postfix({'loaded': len(gop_sequences), 'frames': len(frames)})
    seq_bar.close()

    # Preserve original sequence order when building the training list
    ordered_gops = []
    for seq_idx in seq_indices:
        seq_id = seq_index_to_id.get(seq_idx)
        if seq_id is None:
            continue
        if seq_id in gop_sequences:
            ordered_gops.append((seq_id, gop_sequences[seq_id]))

    print(f"✅ Loaded {len(ordered_gops)} complete GOPs (will be reused for all epochs)", flush=True)
    
    # Print GOP statistics
    gop_lengths = [len(frames) for _, frames in ordered_gops]
    if gop_lengths:
        print(f"   📊 GOP lengths: min={min(gop_lengths)}, max={max(gop_lengths)}, avg={sum(gop_lengths)/len(gop_lengths):.1f}")
        print(f"   🎯 GOPs with >= 2 frames: {sum(1 for l in gop_lengths if l >= 2)}")
    
    return ordered_gops


def load_balanced_gops(dataloader, max_gops=70, gop_length=48):
    """
    Load a limited number of GOPs with balanced distribution across all videos.
    
    Args:
        dataloader: DataLoader with the full dataset
        max_gops: Maximum number of GOPs to load (default 70)
        gop_length: Length of each GOP in frames (default 48)
    
    Returns:
        List of (seq_id, frames) tuples with balanced sampling across videos
    """
    print(f"\n🎯 Loading balanced GOP subset: max_gops={max_gops}")
    
    base_dataset = dataloader.dataset.dataset if hasattr(dataloader.dataset, 'dataset') else dataloader.dataset
    
    # Get all available sequences grouped by video
    video_gops = {}  # {video_name: [gop_indices]}
    
    # Scan dataset to group GOPs by video
    if hasattr(base_dataset, 'sequences'):
        # Use sequences list to group by video name
        for seq_idx, seq_info in enumerate(base_dataset.sequences):
            video_name = seq_info['video_name']
            if video_name not in video_gops:
                video_gops[video_name] = []
            video_gops[video_name].append(seq_idx)
    elif hasattr(base_dataset, 'sequence_ids'):
        for seq_idx, seq_id in enumerate(base_dataset.sequence_ids):
            # Extract video name from sequence_id (format: "video_name_gop_X")
            video_name = '_'.join(seq_id.split('_')[:-2]) if '_gop_' in seq_id else seq_id
            if video_name not in video_gops:
                video_gops[video_name] = []
            video_gops[video_name].append(seq_idx)
    else:
        # Fallback: use all available GOPs without grouping
        print("   ⚠️ Warning: Could not detect video grouping, using all GOPs")
        num_seqs = len(base_dataset.sequences) if hasattr(base_dataset, 'sequences') else len(base_dataset)
        # Just return random selection of GOP indices
        import random
        selected_indices = random.sample(range(num_seqs), min(max_gops, num_seqs))
        print(f"   ✅ Randomly selected {len(selected_indices)} GOPs for training")
        return load_gop_sequences_once(dataloader, gop_length, selected_indices=selected_indices)
    
    print(f"   📹 Found {len(video_gops)} videos with total {sum(len(v) for v in video_gops.values())} GOPs")
    
    # Calculate balanced sampling: equal GOPs per video
    num_videos = len(video_gops)
    gops_per_video = max(1, max_gops // num_videos)
    
    print(f"   ⚖️  Sampling {gops_per_video} GOPs per video (balanced distribution)")
    
    # Sample GOPs evenly from each video
    selected_indices = []
    for video_name, gop_indices in sorted(video_gops.items()):
        # Sample evenly spaced GOPs from this video
        num_available = len(gop_indices)
        if num_available <= gops_per_video:
            # Take all GOPs if video has fewer than target
            selected_indices.extend(gop_indices)
        else:
            # Sample evenly spaced GOPs
            step = num_available / gops_per_video
            sampled = [gop_indices[int(i * step)] for i in range(gops_per_video)]
            selected_indices.extend(sampled)
    
    # Limit to max_gops in case of rounding
    selected_indices = selected_indices[:max_gops]
    
    print(f"   ✅ Selected {len(selected_indices)} GOPs for balanced training")
    print(f"   📊 Distribution: {len(selected_indices)}/{sum(len(v) for v in video_gops.values())} GOPs ({len(selected_indices)/sum(len(v) for v in video_gops.values())*100:.1f}%)")
    
    # Now load these selected GOPs using the existing loading logic
    return load_gop_sequences_once(dataloader, gop_length, selected_indices=selected_indices)


def train_epoch_memory(model, ordered_gops, criterion, optimizer, scaler, args, device, epoch, train_loader=None, gop_cache=None):
    """
    Train memory-based tracker for one epoch using pre-loaded GOP sequences.
    
    Args:
        model: The model to train
        ordered_gops: Pre-loaded list of (seq_id, frames) tuples (required for balanced loading)
        criterion: Loss function
        optimizer: Optimizer
        scaler: AMP scaler
        args: Training arguments
        device: Device
        epoch: Current epoch number
        train_loader: Not used (kept for compatibility)
        gop_cache: Not used (kept for compatibility)
    """
    print(f"\n🔍 DEBUG: Entered train_epoch_memory for epoch {epoch+1}", flush=True)
    
    # Use pre-loaded GOPs (balanced sampling)
    if ordered_gops is None:
        raise ValueError("ordered_gops must be provided for balanced GOP training")
    
    print(f"🔍 DEBUG: Using preloaded GOPs ({len(ordered_gops)} sequences)", flush=True)
    print(f"🔍 DEBUG: About to call model.train()", flush=True)
    model.train()
    print(f"🔍 DEBUG: model.train() completed", flush=True)
    
    total_loss = 0.0
    total_box_loss = 0.0
    total_conf_loss = 0.0
    total_giou_loss = 0.0
    total_classification_loss = 0.0
    total_no_object_loss = 0.0
    total_map_score = 0.0  # Track mAP during training
    total_static_baseline_map = 0.0  # Track static baseline mAP
    num_gops = 0
    
    # Detect if using detection loss (for tracking additional metrics)
    from mots_exp.models.mv_center.components.detection_loss import DetectionLossWithNoObject
    use_detection_loss = isinstance(criterion, DetectionLossWithNoObject)
    print(f"🔍 DEBUG: Detection loss check completed (use_detection_loss={use_detection_loss})", flush=True)
    
    # Train on each pre-loaded GOP
    progress_bar = tqdm(ordered_gops, 
                       desc=f"Epoch {epoch+1}/{args.epochs} [Memory GOP]", 
                       unit="GOP")
    
    for seq_id, gop_frames in progress_bar:
        if len(gop_frames) < 2:
            # Skip GOPs with less than 2 frames (can't compute velocity)
            continue
        
        # Train on this GOP sequence
        loss_dict = _train_on_gop_sequence(
            model, gop_frames, criterion, optimizer,
            scaler, args, device, progress_bar
        )
        
        if loss_dict is not None:
            total_loss += loss_dict['total']
            total_box_loss += loss_dict['box']
            # Handle both old loss (confidence) and new loss (classification)
            total_conf_loss += loss_dict.get('confidence', loss_dict.get('classification', 0.0))
            
            # Track additional metrics for detection loss
            if use_detection_loss:
                total_giou_loss += loss_dict.get('giou', 0.0)
                total_classification_loss += loss_dict.get('classification', 0.0)
                total_no_object_loss += loss_dict.get('no_object', 0.0)
            
            # ⚠️ DISABLED: mAP tracking during training (moved to validation only)
            # total_map_score += loss_dict.get('map', 0.0)
            # total_static_baseline_map += loss_dict.get('static_baseline_map', 0.0)
            
            num_gops += 1
            
            # Update progress bar with current stats (no mAP during training)
            avg_loss = total_loss / max(num_gops, 1)
            progress_bar.set_postfix({
                'loss': f'{loss_dict["total"]:.4f}',
                'avg': f'{avg_loss:.4f}',
                'frames': len(gop_frames)
            })
    
    # Calculate epoch averages
    avg_loss = total_loss / max(num_gops, 1)
    avg_box_loss = total_box_loss / max(num_gops, 1)
    avg_conf_loss = total_conf_loss / max(num_gops, 1)
    # ⚠️ DISABLED: Training mAP calculation (moved to validation)
    # avg_map = total_map_score / max(num_gops, 1)
    # avg_static_baseline = total_static_baseline_map / max(num_gops, 1)
    
    print(f"\n📊 Memory Tracker Epoch {epoch+1} Summary:")
    print(f"   Total Loss: {avg_loss:.6f}")
    print(f"   Box Loss: {avg_box_loss:.6f}")
    # ⚠️ Training mAP now only calculated during validation
    # print(f"   Training mAP@0.5: {avg_map:.4f}")
    # print(f"   Static Baseline mAP: {avg_static_baseline:.4f} (I-frame boxes reused)")
    # if avg_map > avg_static_baseline:
    #     improvement = ((avg_map - avg_static_baseline) / (avg_static_baseline + 1e-8)) * 100
    #     print(f"   📈 Improvement over static: +{improvement:.1f}%")
    # else:
    #     decline = ((avg_static_baseline - avg_map) / (avg_static_baseline + 1e-8)) * 100
    #     print(f"   ⚠️ Below static baseline: -{decline:.1f}%")
    # Velocity loss has been removed from training metrics
    
    # Print detection loss components if using new loss
    if use_detection_loss:
        avg_giou_loss = total_giou_loss / max(num_gops, 1)
        avg_classification_loss = total_classification_loss / max(num_gops, 1)
        avg_no_object_loss = total_no_object_loss / max(num_gops, 1)
        print(f"   GIoU Loss: {avg_giou_loss:.6f}")
        print(f"   Classification Loss: {avg_classification_loss:.6f}")
        print(f"   No-Object Loss: {avg_no_object_loss:.6f}")
    else:
        print(f"   Confidence Loss: {avg_conf_loss:.6f}")
    
    print(f"   GOPs Processed: {num_gops}")
    
    # Return metrics (include new ones for detection loss)
    metrics = {
        'total_loss': avg_loss,
        'box_loss': avg_box_loss,
        'confidence_loss': avg_conf_loss,
        # ⚠️ REMOVED: Training mAP (now only calculated during validation)
        # 'train_map': avg_map,
        # 'static_baseline_map': avg_static_baseline,
        'num_batches': num_gops
    }
    
    if use_detection_loss:
        metrics['giou_loss'] = avg_giou_loss
        metrics['classification_loss'] = avg_classification_loss
        metrics['no_object_loss'] = avg_no_object_loss
    
    return metrics


def train_epoch(model, dataloader, optimizer, scaler, args, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_center_loss = 0.0
    total_box_loss = 0.0
    total_giou_loss = 0.0
    num_batches = 0
    
    # Debug first batch
    debug_first = True
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
    
    for batch_idx, batch in enumerate(progress_bar):
        if batch is None:
            if debug_first:
                print("⚠️ Batch is None")
            continue
        
        # Process batch (debug first batch only)
        motion_vectors, targets = process_batch(batch, device, debug=debug_first)
        if motion_vectors is None or not targets:
            if debug_first:
                print(f"⚠️ Skipping batch: motion_vectors={motion_vectors is not None}, targets={bool(targets)}")
            debug_first = False
            continue
        
        debug_first = False  # Disable debug after first valid batch
        
        optimizer.zero_grad()
        
        # Forward pass with AMP if enabled
        if args.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict, predictions = model(motion_vectors, targets)
                loss = loss_dict['total_loss']
        else:
            loss_dict, predictions = model(motion_vectors, targets)
            loss = loss_dict['total_loss']
        
        # Backward pass
        if args.amp and scaler is not None:
            scaler.scale(loss).backward()
            if args.gradient_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            else:
                grad_norm = 0.0
            optimizer.step()
        
        # Accumulate losses (handle both tensors and floats)
        total_loss += loss.item() if hasattr(loss, 'item') else float(loss)
        total_center_loss += loss_dict['center_loss'].item() if hasattr(loss_dict['center_loss'], 'item') else float(loss_dict['center_loss'])
        total_box_loss += loss_dict['box_loss'].item() if hasattr(loss_dict['box_loss'], 'item') else float(loss_dict['box_loss'])
        total_giou_loss += loss_dict['giou_loss'].item() if hasattr(loss_dict['giou_loss'], 'item') else float(loss_dict['giou_loss'])
        num_batches += 1
        
        # Update progress bar
        if batch_idx % args.log_freq == 0:
            avg_loss = total_loss / num_batches
            postfix = {
                'loss': f'{loss.item() if hasattr(loss, "item") else float(loss):.6f}',
                'avg': f'{avg_loss:.6f}',
                'center': f'{loss_dict["center_loss"].item() if hasattr(loss_dict["center_loss"], "item") else float(loss_dict["center_loss"]):.6f}',
                'box': f'{loss_dict["box_loss"].item() if hasattr(loss_dict["box_loss"], "item") else float(loss_dict["box_loss"]):.6f}',
                'giou': f'{loss_dict["giou_loss"].item() if hasattr(loss_dict["giou_loss"], "item") else float(loss_dict["giou_loss"]):.6f}'
            }
            if args.gradient_clip > 0:
                postfix['grad'] = f'{grad_norm:.2f}'
            progress_bar.set_postfix(postfix)
    
    # Calculate epoch averages
    avg_loss = total_loss / max(num_batches, 1)
    avg_center_loss = total_center_loss / max(num_batches, 1)
    avg_box_loss = total_box_loss / max(num_batches, 1)
    avg_giou_loss = total_giou_loss / max(num_batches, 1)
    
    return {
        'total_loss': avg_loss,
        'center_loss': avg_center_loss,
        'box_loss': avg_box_loss,
        'giou_loss': avg_giou_loss,
        'num_batches': num_batches
    }


def save_checkpoint(model, optimizer, epoch, loss_history, args):
    """Save model checkpoint."""
    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_info': model_info,
        'loss_history': loss_history,
        'args': vars(args)
    }
    
    # Create filename with model configuration
    version = args.version
    config = args.config
    magnitude = "magnitude" if args.use_magnitude else "baseline"
    embeddings = "emb" if args.use_embeddings else ""
    
    filename = f'mv_center_{version}_{config}_{magnitude}'
    if embeddings:
        filename += f'_{embeddings}'
    filename += f'_epoch_{epoch+1}.pt'
    
    checkpoint_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def save_best_checkpoint(model, optimizer, epoch, loss_history, val_map, args):
    """Save best model checkpoint based on validation mAP."""
    model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
    
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_info': model_info,
        'loss_history': loss_history,
        'val_map': val_map,
        'args': vars(args)
    }
    
    # Save as best_model.pt
    best_path = os.path.join(args.output_dir, 'best_model.pt')
    torch.save(checkpoint, best_path)
    
    print(f"🏆 Best model saved: {best_path} (mAP@0.5: {val_map:.4f})")
    return best_path


def plot_training_curves(loss_history, output_dir):
    """Plot and save training curves."""
    epochs = range(1, len(loss_history['total_loss']) + 1)
    
    # Check what losses are available
    has_giou = 'giou_loss' in loss_history and len(loss_history['giou_loss']) > 0
    has_classification = 'classification_loss' in loss_history and len(loss_history['classification_loss']) > 0
    has_no_object = 'no_object_loss' in loss_history and len(loss_history['no_object_loss']) > 0
    has_val_map = 'val_map' in loss_history and len(loss_history['val_map']) > 0
    
    # Determine layout based on available metrics
    if has_giou or has_classification:
        # Detection loss: use 3x2 grid for 6 plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
        
        # Total loss
        ax1.plot(epochs, loss_history['total_loss'], 'b-', linewidth=2, marker='o')
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Box loss
        ax2.plot(epochs, loss_history['box_loss'], 'g-', linewidth=2, marker='^')
        ax2.set_title('Box Regression Loss (L1)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Box Loss')
        ax2.grid(True, alpha=0.3)
        
        # GIoU loss
        if has_giou:
            ax3.plot(epochs, loss_history['giou_loss'], 'm-', linewidth=2, marker='d')
            ax3.set_title('GIoU Loss', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('GIoU Loss')
            ax3.grid(True, alpha=0.3)
        
        # Classification loss
        if has_classification:
            ax4.plot(epochs, loss_history['classification_loss'], 'r-', linewidth=2, marker='s')
            ax4.set_title('Classification Loss (Focal)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Classification Loss')
            ax4.grid(True, alpha=0.3)
        
        # No-object loss
        if has_no_object:
            ax5.plot(epochs, loss_history['no_object_loss'], 'c-', linewidth=2, marker='x')
            ax5.set_title('No-Object Loss', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('No-Object Loss')
            ax5.grid(True, alpha=0.3)
        
        # Validation mAP only (training mAP removed for speed)
        if has_val_map:
            ax6.plot(epochs, loss_history['val_map'], 'orange', linewidth=2, marker='*', label='Val mAP')
            ax6.set_title('Validation mAP@0.5', fontsize=14, fontweight='bold')
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('mAP')
            ax6.legend(loc='best', fontsize=9)
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('DCT-MV Detection Training Progress', fontsize=16, fontweight='bold')
    else:
        # Legacy tracker: 2x2 grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        ax1.plot(epochs, loss_history['total_loss'], 'b-', linewidth=2, marker='o')
        ax1.set_title('Total Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Box loss
        ax2.plot(epochs, loss_history['box_loss'], 'g-', linewidth=2, marker='^')
        ax2.set_title('Box Regression Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Box Loss')
        ax2.grid(True, alpha=0.3)
        
        # Confidence loss (if available)
        if 'confidence_loss' in loss_history:
            ax3.plot(epochs, loss_history['confidence_loss'], 'm-', linewidth=2, marker='d')
            ax3.set_title('Confidence Loss', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Confidence Loss')
            ax3.grid(True, alpha=0.3)
        
        # Training & Validation mAP with Baselines (if available)
        if has_val_map:
            ax4.plot(epochs, loss_history['val_map'], 'orange', linewidth=2, marker='*', label='Val mAP')
            ax4.set_title('Validation mAP@0.5', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('mAP')
            ax4.legend(loc='best', fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Memory Tracker Training Progress', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training curves saved: {plot_path}")


def evaluate_map(model, dataset, device, max_eval_sequences=None, ordered_val_gops=None):
    """
    Proper validation: compute mAP@0.5 for specific frame indices.
    Evaluates at frames [1, 10, 20, 30, 48] to measure tracking drift over time.
    Also computes separate metrics for moving camera vs static camera videos.
    
    Args:
        model: Model to evaluate
        dataset: Full dataset (used if ordered_val_gops is None)
        device: Device to use
        max_eval_sequences: Max number of sequences to evaluate (legacy, ignored if ordered_val_gops provided)
        ordered_val_gops: Pre-loaded balanced GOP subset for validation (optional)
    """
    from mots_exp.metrics import simple_map
    model.eval()
    
    # Use pre-loaded GOPs if provided, otherwise use full dataset
    if ordered_val_gops is not None:
        print(f"   🎯 Validating on {len(ordered_val_gops)} pre-loaded balanced GOPs")
        seq_infos = []
        # Convert ordered_gops format to seq_infos format
        for seq_id, gop_frames in ordered_val_gops:
            # DEBUG: Print first few seq_ids to understand format
            if len(seq_infos) < 3:
                print(f"   🔍 DEBUG Val GOP format: seq_id='{seq_id}', num_frames={len(gop_frames)}")
            
            # Extract video name and gop index from seq_id
            # Format: 'VideoName_960x960_gop50_500frames_gop0'
            # We need to split from the RIGHT to get the last _gopX
            if '_gop' in seq_id:
                # Find the last occurrence of '_gop' followed by digits
                parts = seq_id.rsplit('_gop', 1)  # Split from right, max 1 split
                if len(parts) == 2:
                    video_name = parts[0]
                    try:
                        gop_index = int(parts[1])
                        seq_infos.append({
                            'sequence_id': video_name,
                            'gop_index': gop_index,
                            'frames': gop_frames  # Include frames for direct access
                        })
                    except ValueError:
                        print(f"   ⚠️ WARNING: Could not parse GOP index from '{parts[1]}' in seq_id '{seq_id}'")
                else:
                    print(f"   ⚠️ WARNING: Unexpected split result for '{seq_id}'")
            else:
                print(f"   ⚠️ WARNING: No '_gop' found in seq_id '{seq_id}'")
        
        print(f"   ✅ Converted {len(seq_infos)} validation GOPs to seq_infos format")
    else:
        seq_infos = dataset.get_sequence_info()
        if max_eval_sequences is not None:
            seq_infos = seq_infos[:max_eval_sequences]

    # Camera type classification (comprehensive list for all datasets)
    # Complete authoritative list covering all 37 validation sequences
    MOVING_CAMERA_SEQUENCES = [
        'ADL-Rundle-8', 'ADL-Rundle-1', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
        'KITTI-19', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
        'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP', 'MOT17-12-SDP', 'MOT17-14-SDP'
    ]
    
    STATIC_CAMERA_SEQUENCES = [
        'ADL-Rundle-6', 'ADL-Rundle-3', 'AVG-TownCentre', 'KITTI-16', 'PETS09-S2L2',
        'TUD-Crossing', 'Venice-1', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus',
        'TUD-Stadtmitte', 'Venice-2', 'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',
        'MOT17-01-SDP', 'MOT17-03-SDP', 'MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'
    ]
    
    # Frame indices to evaluate (measure tracking performance over time)
    # Evaluate all P-frames 1-12 for statistics, plus sparse frames for full GOP coverage
    eval_frame_indices = list(range(1, 13)) + [20, 30, 40, 47]  # P-frames 1-12, then 20, 30, 40, 47
    
    # Overall frame maps
    frame_maps = {idx: [] for idx in eval_frame_indices}
    
    # Per-camera-type frame maps
    moving_camera_maps = {idx: [] for idx in eval_frame_indices}
    static_camera_maps = {idx: [] for idx in eval_frame_indices}
    
    # Add progress bar for validation
    val_pbar = tqdm(seq_infos, desc="🔍 Validating", unit="seq")
    
    with torch.no_grad():
        for seq in val_pbar:
            seq_id = seq['sequence_id']
            gop_idx = seq['gop_index']
            
            # Check if we have pre-loaded frames
            use_preloaded = 'frames' in seq
            gop_frames = seq.get('frames', [])
            
            # Determine camera type for this sequence
            # Check if seq_id starts with any base name from our camera lists
            is_moving_camera = any(seq_id.startswith(cam_seq) for cam_seq in MOVING_CAMERA_SEQUENCES)
            is_static_camera = any(seq_id.startswith(cam_seq) for cam_seq in STATIC_CAMERA_SEQUENCES)
            
            # Reset LSTM state for new sequence
            if hasattr(model, 'reset'):
                model.reset()
            
            # Initialize with I-frame (frame 0)
            iframe_boxes_t = None
            if use_preloaded and len(gop_frames) > 0:
                # Use pre-loaded frame 0
                iframe_sample = gop_frames[0]
            else:
                # Load from dataset
                iframe_sample = dataset.data_loader.load_frame(seq, 0)
            
            if iframe_sample is not None:
                iframe_boxes = iframe_sample.get('boxes')
                if iframe_boxes is not None:
                    iframe_boxes_t = torch.tensor(iframe_boxes, dtype=torch.float32).to(device)
                    if hasattr(model, 'tracker'):
                        iframe_ids = iframe_sample.get('ids', None)
                        if iframe_ids is not None:
                            iframe_ids = torch.tensor(iframe_ids, dtype=torch.long).to(device)
                        model.tracker.init_from_iframe(iframe_boxes_t, iframe_ids)
            
            # Track through sequence and evaluate at specific frames
            hidden_state = None
            current_boxes = iframe_boxes_t
            
            # Determine sequence length
            seq_length = len(gop_frames) if use_preloaded else dataset.sequence_length
            
            for f in range(1, seq_length):  # Start from frame 1 (first P-frame)
                if use_preloaded:
                    # Use pre-loaded frame
                    if f >= len(gop_frames):
                        continue
                    sample = gop_frames[f]
                else:
                    # Load from dataset
                    sample = dataset.data_loader.load_frame(seq, f)
                
                if sample is None:
                    continue
                    
                mv = sample.get('motion_vectors')
                gt_boxes = sample.get('boxes')
                dct = sample.get('residuals', None)
                
                # ✅ ABLATION FIX: Allow validation with only DCT residuals
                if gt_boxes is None or (mv is None and dct is None):
                    continue
                
                # Process motion vectors (can be None for DCT-only mode)
                mv_t = None
                if mv is not None:
                    mv_t = torch.tensor(mv, dtype=torch.float32).to(device) if not isinstance(mv, torch.Tensor) else mv.to(device)
                    if mv_t.numel() == 0:
                        mv_t = None
                    elif mv_t.ndim == 4:
                        mv_avg = mv_t.mean(dim=0)
                        if mv_avg is None:
                            mv_t = None
                        else:
                            mv_t = mv_avg.permute(2,0,1).unsqueeze(0)
                    elif mv_t.ndim == 3:
                        mv_t = mv_t.permute(2,0,1).unsqueeze(0)
                    else:
                        if mv_t.ndim != 4:
                            mv_t = None
                
                # Process DCT residuals
                dct_t = None
                if dct is not None:
                    dct_t = torch.tensor(dct, dtype=torch.float32).to(device) if not isinstance(dct, torch.Tensor) else dct.to(device)
                    if dct_t.ndim == 3:
                        dct_t = dct_t.unsqueeze(0)
                
                # ✅ FIXED: Predict from previous frame boxes, NOT ground truth!
                if hasattr(model, 'forward_single_frame'):
                    # Use current_boxes (predictions from previous frame) as input
                    out = model.forward_single_frame(mv_t, dct_t, current_boxes, hidden_state)
                    if isinstance(out, tuple):
                        if len(out) == 3:
                            pred_boxes, _, hidden_state = out
                        else:
                            pred_boxes = out[0]
                        pred_boxes = pred_boxes.detach().cpu()
                        # Update current_boxes for next frame
                        current_boxes = pred_boxes.to(device) if len(pred_boxes) > 0 else current_boxes
                    else:
                        pred_boxes = out.detach().cpu()
                        current_boxes = pred_boxes.to(device) if len(pred_boxes) > 0 else current_boxes
                else:
                    pred_boxes = torch.empty((0,4))

                # Evaluate at specific frame indices
                if f in eval_frame_indices:
                    tgt_boxes = gt_boxes.cpu() if isinstance(gt_boxes, torch.Tensor) else torch.tensor(gt_boxes)
                    if len(pred_boxes) > 0 and len(tgt_boxes) > 0:
                        m = simple_map(pred_boxes, tgt_boxes, thresh=0.5)
                        frame_maps[f].append(m)
                        
                        # Also add to camera-specific maps
                        if is_moving_camera:
                            moving_camera_maps[f].append(m)
                        elif is_static_camera:
                            static_camera_maps[f].append(m)
            
            # Update progress bar with average mAP across all frames
            all_maps = [m for maps in frame_maps.values() for m in maps]
            if all_maps:
                val_pbar.set_postfix({'mAP': f'{sum(all_maps)/len(all_maps):.4f}'})
    
    val_pbar.close()
    model.train()
    
    # Compute mAP per frame index (overall)
    results = {}
    print(f"\n   📊 mAP@0.5 by frame index (Overall):")
    for frame_idx in eval_frame_indices:
        if frame_maps[frame_idx]:
            frame_map = sum(frame_maps[frame_idx]) / len(frame_maps[frame_idx])
            results[f'mAP50_frame_{frame_idx+1}'] = frame_map  # +1 for 1-indexed display
            print(f"      Frame {frame_idx+1:2d}: {frame_map:.4f}")
        else:
            results[f'mAP50_frame_{frame_idx+1}'] = 0.0
            print(f"      Frame {frame_idx+1:2d}: 0.0000")
    
    # Compute mAP per frame index (moving camera)
    moving_count = len(MOVING_CAMERA_SEQUENCES)
    print(f"\n   📹 mAP@0.5 by frame index (Moving Camera: {moving_count} sequences):")
    for frame_idx in eval_frame_indices:
        if moving_camera_maps[frame_idx]:
            frame_map = sum(moving_camera_maps[frame_idx]) / len(moving_camera_maps[frame_idx])
            results[f'mAP50_moving_frame_{frame_idx+1}'] = frame_map
            print(f"      Frame {frame_idx+1:2d}: {frame_map:.4f}")
        else:
            results[f'mAP50_moving_frame_{frame_idx+1}'] = 0.0
            print(f"      Frame {frame_idx+1:2d}: 0.0000")
    
    # Compute mAP per frame index (static camera)
    static_count = len(STATIC_CAMERA_SEQUENCES)
    print(f"\n   🎥 mAP@0.5 by frame index (Static Camera: {static_count} sequences):")
    for frame_idx in eval_frame_indices:
        if static_camera_maps[frame_idx]:
            frame_map = sum(static_camera_maps[frame_idx]) / len(static_camera_maps[frame_idx])
            results[f'mAP50_static_frame_{frame_idx+1}'] = frame_map
            print(f"      Frame {frame_idx+1:2d}: {frame_map:.4f}")
        else:
            results[f'mAP50_static_frame_{frame_idx+1}'] = 0.0
            print(f"      Frame {frame_idx+1:2d}: 0.0000")
    
    # Compute averages
    all_maps = [m for maps in frame_maps.values() for m in maps]
    moving_all_maps = [m for maps in moving_camera_maps.values() for m in maps]
    static_all_maps = [m for maps in static_camera_maps.values() for m in maps]
    
    if all_maps:
        avg_map = sum(all_maps) / len(all_maps)
        print(f"\n      📊 Overall Average:  {avg_map:.4f}")
        
        if moving_all_maps:
            moving_avg = sum(moving_all_maps) / len(moving_all_maps)
            results['mAP50_moving_avg'] = moving_avg
            print(f"      📹 Moving Camera Avg: {moving_avg:.4f}")
        
        if static_all_maps:
            static_avg = sum(static_all_maps) / len(static_all_maps)
            results['mAP50_static_avg'] = static_avg
            print(f"      🎥 Static Camera Avg: {static_avg:.4f}")
        
        return float(avg_map)
    return 0.0


def evaluate_per_sequence(model, dataset, device, ordered_val_gops=None, max_eval_sequences=None):
    """
    Comprehensive per-sequence evaluation with detailed mAP@0.5 metrics.
    Returns mAP50 per sequence broken down by camera type.
    
    Args:
        model: Trained model
        dataset: Dataset to evaluate
        device: Device (cuda/cpu)
        ordered_val_gops: Pre-loaded validation GOPs (optional)
        max_eval_sequences: Maximum sequences to evaluate (optional)
    
    Returns:
        dict: Per-sequence results including:
            - sequence_name -> mAP50 value
            - camera_type classifications (static/moving/mixed)
            - overall/static/moving averages
    """
    from mots_exp.metrics import simple_map
    from collections import defaultdict
    
    # Define camera sequence classifications (complete list)
    MOVING_CAMERA_SEQUENCES = [
        'ADL-Rundle-8', 'ADL-Rundle-1', 'ETH-Crossing', 'ETH-Jelmoli', 'ETH-Linthescher',
        'KITTI-19', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
        'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP', 'MOT17-12-SDP', 'MOT17-14-SDP'
    ]
    
    STATIC_CAMERA_SEQUENCES = [
        'ADL-Rundle-6', 'ADL-Rundle-3', 'AVG-TownCentre', 'KITTI-16', 'PETS09-S2L2',
        'TUD-Crossing', 'Venice-1', 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus',
        'TUD-Stadtmitte', 'Venice-2', 'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05',
        'MOT17-01-SDP', 'MOT17-03-SDP', 'MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08'
    ]
    
    model.eval()
    print(f"\n🔍 Running detailed per-sequence evaluation...")
    
    # Frame indices to evaluate (all P-frames 1-12 for statistics, plus sparse frames)
    eval_frame_indices = list(range(1, 13)) + [20, 30, 40, 47]  # P-frames 1-12, then 20, 30, 40, 47
    
    # Get validation sequences
    if ordered_val_gops is not None:
        # Extract unique sequence names from pre-loaded GOPs
        # ordered_val_gops is a list of tuples: (seq_id, frames)
        val_seq_names = sorted(list(set([gop[0].split('_gop_')[0] if '_gop_' in gop[0] else gop[0] for gop in ordered_val_gops])))
        if max_eval_sequences:
            val_seq_names = val_seq_names[:max_eval_sequences]
    else:
        # Use dataset sequences
        val_seq_names = dataset.sequences
        if max_eval_sequences:
            val_seq_names = val_seq_names[:max_eval_sequences]
    
    # Per-sequence mAP storage
    sequence_maps = defaultdict(list)  # sequence_name -> list of frame mAPs
    
    # Frame-by-frame mAP tracking (for mAP vs GOP frame index plot)
    frame_maps_by_camera = {
        'static': defaultdict(list),   # frame_idx -> list of mAPs
        'moving': defaultdict(list),
        'mixed': defaultdict(list)
    }
    
    # Track all P-frames (1-12) for statistical analysis
    pframe_maps_by_camera = {
        'static': [],  # All mAP values from P-frames 1-12
        'moving': [],
        'mixed': []
    }
    
    # Track first 6 P-frames (1-6) for early-frames analysis
    pframe_6_maps_by_camera = {
        'static': [],  # All mAP values from P-frames 1-6
        'moving': [],
        'mixed': []
    }
    
    # Classify sequences by camera type
    # Note: val_seq_names have full suffix (e.g., "ADL-Rundle-1_960x960_gop50_500frames_gop0")
    # but our lists have base names (e.g., "ADL-Rundle-1")
    # So we check if the base name is a prefix of the full sequence name
    static_sequences = []
    moving_sequences = []
    mixed_sequences = []
    
    for seq_name in val_seq_names:
        # Check if any static sequence base name is a prefix of this sequence
        is_static = any(seq_name.startswith(base_name) for base_name in STATIC_CAMERA_SEQUENCES)
        is_moving = any(seq_name.startswith(base_name) for base_name in MOVING_CAMERA_SEQUENCES)
        
        if is_static:
            static_sequences.append(seq_name)
        elif is_moving:
            moving_sequences.append(seq_name)
        else:
            mixed_sequences.append(seq_name)
    
    print(f"   📊 Evaluating {len(val_seq_names)} sequences:")
    print(f"      🎥 Static: {len(static_sequences)}")
    print(f"      📹 Moving: {len(moving_sequences)}")
    print(f"      🎬 Mixed: {len(mixed_sequences)}")
    
    # Progress bar
    seq_pbar = tqdm(val_seq_names, desc="   Per-Sequence Evaluation", leave=True)
    
    for seq in seq_pbar:
        seq_pbar.set_postfix({'seq': seq})
        
        # Get GOPs for this sequence
        if ordered_val_gops is not None:
            # ordered_val_gops is a list of tuples: (seq_id, frames)
            # Match sequences by checking if seq is in the seq_id
            seq_gops = [gop for gop in ordered_val_gops if seq in gop[0]]
        else:
            # Load GOPs from dataset
            continue  # Skip if no pre-loaded GOPs
        
        if not seq_gops:
            continue
        
        # Evaluate each GOP
        for gop_tuple in seq_gops:
            # Unpack tuple: (seq_id, frames)
            seq_id, gop_frames = gop_tuple
            
            # Initialize tracking state
            current_boxes = None
            hidden_state = None
            
            # Initialize with I-frame (frame 0) ground truth boxes
            if len(gop_frames) > 0:
                iframe_sample = gop_frames[0]
                if iframe_sample is not None:
                    iframe_boxes = iframe_sample.get('boxes')
                    if iframe_boxes is not None:
                        current_boxes = torch.tensor(iframe_boxes, dtype=torch.float32).to(device) if not isinstance(iframe_boxes, torch.Tensor) else iframe_boxes.to(device)
            
            # Process each frame in GOP
            for f in range(len(gop_frames)):
                sample = gop_frames[f]
                
                if sample is None:
                    continue
                
                mv = sample.get('motion_vectors')
                gt_boxes = sample.get('boxes')
                dct = sample.get('residuals', None)
                
                # Skip if no ground truth or no input data
                if gt_boxes is None or (mv is None and dct is None):
                    continue
                
                # Skip frame 0 (I-frame) - already used for initialization
                if f == 0:
                    continue
                
                # Process motion vectors
                mv_t = None
                if mv is not None:
                    mv_t = torch.tensor(mv, dtype=torch.float32).to(device) if not isinstance(mv, torch.Tensor) else mv.to(device)
                    if mv_t.numel() == 0:
                        mv_t = None
                    elif mv_t.ndim == 4:
                        mv_avg = mv_t.mean(dim=0)
                        if mv_avg is None:
                            mv_t = None
                        else:
                            mv_t = mv_avg.permute(2, 0, 1).unsqueeze(0)
                    elif mv_t.ndim == 3:
                        mv_t = mv_t.permute(2, 0, 1).unsqueeze(0)
                    else:
                        if mv_t.ndim != 4:
                            mv_t = None
                
                # Process DCT residuals
                dct_t = None
                if dct is not None:
                    dct_t = torch.tensor(dct, dtype=torch.float32).to(device) if not isinstance(dct, torch.Tensor) else dct.to(device)
                    if dct_t.ndim == 3:
                        dct_t = dct_t.unsqueeze(0)
                
                # Get predictions
                with torch.no_grad():
                    if hasattr(model, 'forward_single_frame'):
                        out = model.forward_single_frame(mv_t, dct_t, current_boxes, hidden_state)
                        if isinstance(out, tuple):
                            if len(out) == 3:
                                pred_boxes, _, hidden_state = out
                            else:
                                pred_boxes = out[0]
                            pred_boxes = pred_boxes.detach().cpu()
                            current_boxes = pred_boxes.to(device) if len(pred_boxes) > 0 else current_boxes
                        else:
                            pred_boxes = out.detach().cpu()
                            current_boxes = pred_boxes.to(device) if len(pred_boxes) > 0 else current_boxes
                    else:
                        pred_boxes = torch.empty((0, 4))
                
                # Evaluate at specific frame indices
                if f in eval_frame_indices:
                    tgt_boxes = gt_boxes.cpu() if isinstance(gt_boxes, torch.Tensor) else torch.tensor(gt_boxes)
                    if len(pred_boxes) > 0 and len(tgt_boxes) > 0:
                        m = simple_map(pred_boxes, tgt_boxes, thresh=0.5)
                        sequence_maps[seq].append(m)
                        
                        # Track by camera type for frame-by-frame analysis
                        if seq in static_sequences:
                            frame_maps_by_camera['static'][f].append(m)
                            # Also track for first 12 P-frames (1-12) for statistical analysis
                            if 1 <= f <= 12:
                                pframe_maps_by_camera['static'].append(m)
                            # Track for first 6 P-frames (1-6) for early-frames analysis
                            if 1 <= f <= 6:
                                pframe_6_maps_by_camera['static'].append(m)
                        elif seq in moving_sequences:
                            frame_maps_by_camera['moving'][f].append(m)
                            if 1 <= f <= 12:
                                pframe_maps_by_camera['moving'].append(m)
                            if 1 <= f <= 6:
                                pframe_6_maps_by_camera['moving'].append(m)
                        else:
                            frame_maps_by_camera['mixed'][f].append(m)
                            if 1 <= f <= 12:
                                pframe_maps_by_camera['mixed'].append(m)
                            if 1 <= f <= 6:
                                pframe_6_maps_by_camera['mixed'].append(m)
    
    seq_pbar.close()
    model.train()
    
    # Compute per-sequence averages
    results = {
        'per_sequence': {},
        'static_sequences': {},
        'moving_sequences': {},
        'mixed_sequences': {},
        'overall_avg': 0.0,
        'static_avg': 0.0,
        'moving_avg': 0.0,
        'mixed_avg': 0.0,
        'frame_evolution': {},  # frame_idx -> {static_avg, moving_avg, mixed_avg}
        'pframe_stats': {},  # Statistics over first 12 P-frames
        'pframe_6_stats': {}  # Statistics over first 6 P-frames
    }
    
    # Overall results
    print(f"\n   📊 Per-Sequence mAP@0.5 Results:")
    print(f"   {'='*60}")
    
    all_maps = []
    static_maps = []
    moving_maps = []
    mixed_maps = []
    
    for seq in sorted(sequence_maps.keys()):
        if sequence_maps[seq]:
            seq_map = sum(sequence_maps[seq]) / len(sequence_maps[seq])
            results['per_sequence'][seq] = seq_map
            all_maps.append(seq_map)
            
            # Classify by camera type
            if seq in static_sequences:
                results['static_sequences'][seq] = seq_map
                static_maps.append(seq_map)
                cam_type = "🎥 Static"
            elif seq in moving_sequences:
                results['moving_sequences'][seq] = seq_map
                moving_maps.append(seq_map)
                cam_type = "📹 Moving"
            else:
                results['mixed_sequences'][seq] = seq_map
                mixed_maps.append(seq_map)
                cam_type = "🎬 Mixed"
            
            print(f"      {cam_type:12s} {seq:20s}: {seq_map:.4f}")
        else:
            results['per_sequence'][seq] = 0.0
    
    print(f"   {'='*60}")
    
    # Compute category averages
    if all_maps:
        results['overall_avg'] = sum(all_maps) / len(all_maps)
        print(f"      📊 Overall Average:  {results['overall_avg']:.4f} ({len(all_maps)} sequences)")
    
    if static_maps:
        results['static_avg'] = sum(static_maps) / len(static_maps)
        print(f"      🎥 Static Average:   {results['static_avg']:.4f} ({len(static_maps)} sequences)")
    
    if moving_maps:
        results['moving_avg'] = sum(moving_maps) / len(moving_maps)
        print(f"      📹 Moving Average:   {results['moving_avg']:.4f} ({len(moving_maps)} sequences)")
    
    if mixed_maps:
        results['mixed_avg'] = sum(mixed_maps) / len(mixed_maps)
        print(f"      🎬 Mixed Average:    {results['mixed_avg']:.4f} ({len(mixed_maps)} sequences)")
    
    # Compute frame-by-frame evolution (mAP vs GOP frame index)
    for frame_idx in sorted(eval_frame_indices):
        results['frame_evolution'][frame_idx] = {}
        
        if frame_maps_by_camera['static'][frame_idx]:
            static_avg = sum(frame_maps_by_camera['static'][frame_idx]) / len(frame_maps_by_camera['static'][frame_idx])
            results['frame_evolution'][frame_idx]['static'] = static_avg
        else:
            results['frame_evolution'][frame_idx]['static'] = None
            
        if frame_maps_by_camera['moving'][frame_idx]:
            moving_avg = sum(frame_maps_by_camera['moving'][frame_idx]) / len(frame_maps_by_camera['moving'][frame_idx])
            results['frame_evolution'][frame_idx]['moving'] = moving_avg
        else:
            results['frame_evolution'][frame_idx]['moving'] = None
            
        if frame_maps_by_camera['mixed'][frame_idx]:
            mixed_avg = sum(frame_maps_by_camera['mixed'][frame_idx]) / len(frame_maps_by_camera['mixed'][frame_idx])
            results['frame_evolution'][frame_idx]['mixed'] = mixed_avg
        else:
            results['frame_evolution'][frame_idx]['mixed'] = None
    
    # Compute statistics over first 12 P-frames
    import numpy as np
    
    print(f"\n   📊 Statistics over first 12 P-frames:")
    print(f"   {'='*60}")
    
    for cam_type in ['static', 'moving', 'mixed']:
        if pframe_maps_by_camera[cam_type]:
            values = np.array(pframe_maps_by_camera[cam_type])
            mean = np.mean(values)
            std = np.std(values)
            stderr = std / np.sqrt(len(values))  # Standard error of the mean
            
            results['pframe_stats'][cam_type] = {
                'mean': float(mean),
                'std': float(std),
                'stderr': float(stderr),
                'n_samples': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
            # Display with uncertainty
            cam_icon = {'static': '🎥', 'moving': '📹', 'mixed': '🎬'}[cam_type]
            print(f"      {cam_icon} {cam_type.capitalize():8s}: {mean:.4f} ± {stderr:.4f} "
                  f"(std: {std:.4f}, n={len(values)})")
        else:
            results['pframe_stats'][cam_type] = None
    
    print(f"   {'='*60}")
    
    # Compute statistics over first 6 P-frames
    print(f"\n   📊 Statistics over first 6 P-frames:")
    print(f"   {'='*60}")
    
    for cam_type in ['static', 'moving', 'mixed']:
        if pframe_6_maps_by_camera[cam_type]:
            values = np.array(pframe_6_maps_by_camera[cam_type])
            mean = np.mean(values)
            std = np.std(values)
            stderr = std / np.sqrt(len(values))  # Standard error of the mean
            
            results['pframe_6_stats'][cam_type] = {
                'mean': float(mean),
                'std': float(std),
                'stderr': float(stderr),
                'n_samples': len(values),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
            # Display with uncertainty
            cam_icon = {'static': '🎥', 'moving': '📹', 'mixed': '🎬'}[cam_type]
            print(f"      {cam_icon} {cam_type.capitalize():8s}: {mean:.4f} ± {stderr:.4f} "
                  f"(std: {std:.4f}, n={len(values)})")
        else:
            results['pframe_6_stats'][cam_type] = None
    
    print(f"   {'='*60}")
    
    return results


def plot_per_sequence_results(results, output_dir, model_name="model"):
    """
    Generate comprehensive per-sequence mAP50 plots.
    Creates separate bar plots for overall, static, moving, and mixed camera sequences.
    
    Args:
        results: Dict from evaluate_per_sequence()
        output_dir: Directory to save plots
        model_name: Name of the model (for plot titles)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"\n📊 Generating per-sequence mAP50 plots...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'per_sequence_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Overall plot (all sequences)
    if results['per_sequence']:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sequences = sorted(results['per_sequence'].keys())
        maps = [results['per_sequence'][seq] for seq in sequences]
        
        # Color code by camera type
        colors = []
        for seq in sequences:
            if seq in results['static_sequences']:
                colors.append('#3498db')  # Blue for static
            elif seq in results['moving_sequences']:
                colors.append('#e74c3c')  # Red for moving
            else:
                colors.append('#95a5a6')  # Gray for mixed
        
        x = np.arange(len(sequences))
        bars = ax.bar(x, maps, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add average line
        avg_map = results['overall_avg']
        ax.axhline(y=avg_map, color='black', linestyle='--', linewidth=2, label=f'Average: {avg_map:.4f}')
        
        ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Per-Sequence mAP@0.5 (All Sequences)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label=f'Static ({len(results["static_sequences"])} seq)'),
            Patch(facecolor='#e74c3c', edgecolor='black', label=f'Moving ({len(results["moving_sequences"])} seq)'),
            Patch(facecolor='#95a5a6', edgecolor='black', label=f'Mixed ({len(results["mixed_sequences"])} seq)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        overall_path = os.path.join(plots_dir, 'per_sequence_overall.png')
        plt.savefig(overall_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Overall plot saved: {overall_path}")
    
    # 2. Static camera sequences
    if results['static_sequences']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sequences = sorted(results['static_sequences'].keys())
        maps = [results['static_sequences'][seq] for seq in sequences]
        
        x = np.arange(len(sequences))
        ax.bar(x, maps, color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add average line
        avg_map = results['static_avg']
        ax.axhline(y=avg_map, color='black', linestyle='--', linewidth=2, label=f'Average: {avg_map:.4f}')
        
        ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Static Camera Sequences mAP@0.5', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        static_path = os.path.join(plots_dir, 'per_sequence_static.png')
        plt.savefig(static_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Static plot saved: {static_path}")
    
    # 3. Moving camera sequences
    if results['moving_sequences']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sequences = sorted(results['moving_sequences'].keys())
        maps = [results['moving_sequences'][seq] for seq in sequences]
        
        x = np.arange(len(sequences))
        ax.bar(x, maps, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add average line
        avg_map = results['moving_avg']
        ax.axhline(y=avg_map, color='black', linestyle='--', linewidth=2, label=f'Average: {avg_map:.4f}')
        
        ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Moving Camera Sequences mAP@0.5', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        moving_path = os.path.join(plots_dir, 'per_sequence_moving.png')
        plt.savefig(moving_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Moving plot saved: {moving_path}")
    
    # 4. Mixed camera sequences (if any)
    if results['mixed_sequences']:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sequences = sorted(results['mixed_sequences'].keys())
        maps = [results['mixed_sequences'][seq] for seq in sequences]
        
        x = np.arange(len(sequences))
        ax.bar(x, maps, color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add average line
        avg_map = results['mixed_avg']
        ax.axhline(y=avg_map, color='black', linestyle='--', linewidth=2, label=f'Average: {avg_map:.4f}')
        
        ax.set_xlabel('Sequence', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Mixed Camera Sequences mAP@0.5', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sequences, rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        mixed_path = os.path.join(plots_dir, 'per_sequence_mixed.png')
        plt.savefig(mixed_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Mixed plot saved: {mixed_path}")
    
    # 5. Summary comparison plot (average per camera type)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = []
    avgs = []
    colors_summary = []
    
    if results['static_avg'] > 0:
        categories.append(f"Static\n({len(results['static_sequences'])} seq)")
        avgs.append(results['static_avg'])
        colors_summary.append('#3498db')
    
    if results['moving_avg'] > 0:
        categories.append(f"Moving\n({len(results['moving_sequences'])} seq)")
        avgs.append(results['moving_avg'])
        colors_summary.append('#e74c3c')
    
    if results['mixed_avg'] > 0:
        categories.append(f"Mixed\n({len(results['mixed_sequences'])} seq)")
        avgs.append(results['mixed_avg'])
        colors_summary.append('#95a5a6')
    
    categories.append(f"Overall\n({len(results['per_sequence'])} seq)")
    avgs.append(results['overall_avg'])
    colors_summary.append('#2c3e50')
    
    x = np.arange(len(categories))
    bars = ax.bar(x, avgs, color=colors_summary, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avgs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name} - mAP@0.5 by Camera Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(avgs) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    summary_path = os.path.join(plots_dir, 'per_sequence_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Summary plot saved: {summary_path}")
    
    # 6. Frame-by-frame evolution plot (mAP vs GOP frame index)
    if results.get('frame_evolution'):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        frame_indices = sorted(results['frame_evolution'].keys())
        
        # Extract data for each camera type
        static_vals = [results['frame_evolution'][f]['static'] for f in frame_indices]
        moving_vals = [results['frame_evolution'][f]['moving'] for f in frame_indices]
        mixed_vals = [results['frame_evolution'][f]['mixed'] for f in frame_indices]
        
        # Plot lines for each camera type
        legend_labels = []
        
        if any(v is not None for v in static_vals):
            static_plot = [v if v is not None else 0 for v in static_vals]
            line_static = ax.plot(frame_indices, static_plot, marker='o', linewidth=2.5, markersize=8,
                   color='#3498db', alpha=0.9, zorder=3)[0]
            
            # Add mean line and uncertainty band for first 12 P-frames
            if results.get('pframe_stats', {}).get('static'):
                stats = results['pframe_stats']['static']
                mean_val = stats['mean']
                stderr = stats['stderr']
                
                # Draw horizontal line for mean (frames 1-12)
                ax.axhline(y=mean_val, xmin=0, xmax=0.5, color='#3498db', 
                          linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
                # Draw uncertainty band (mean ± stderr)
                ax.fill_between([min(frame_indices), 12], mean_val - stderr, mean_val + stderr,
                               color='#3498db', alpha=0.15, zorder=0)
                
                legend_labels.append(f'Static (μ={mean_val:.3f}±{stderr:.3f}, n={stats["n_samples"]})')
            else:
                legend_labels.append(f'Static ({len(results["static_sequences"])} seq)')
        
        if any(v is not None for v in moving_vals):
            moving_plot = [v if v is not None else 0 for v in moving_vals]
            line_moving = ax.plot(frame_indices, moving_plot, marker='s', linewidth=2.5, markersize=8,
                   color='#e74c3c', alpha=0.9, zorder=3)[0]
            
            # Add mean line and uncertainty band for first 12 P-frames
            if results.get('pframe_stats', {}).get('moving'):
                stats = results['pframe_stats']['moving']
                mean_val = stats['mean']
                stderr = stats['stderr']
                
                ax.axhline(y=mean_val, xmin=0, xmax=0.5, color='#e74c3c',
                          linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
                ax.fill_between([min(frame_indices), 12], mean_val - stderr, mean_val + stderr,
                               color='#e74c3c', alpha=0.15, zorder=0)
                
                legend_labels.append(f'Moving (μ={mean_val:.3f}±{stderr:.3f}, n={stats["n_samples"]})')
            else:
                legend_labels.append(f'Moving ({len(results["moving_sequences"])} seq)')
        
        if any(v is not None for v in mixed_vals):
            mixed_plot = [v if v is not None else 0 for v in mixed_vals]
            line_mixed = ax.plot(frame_indices, mixed_plot, marker='^', linewidth=2.5, markersize=8,
                   color='#95a5a6', alpha=0.9, zorder=3)[0]
            
            # Add mean line and uncertainty band for first 12 P-frames
            if results.get('pframe_stats', {}).get('mixed'):
                stats = results['pframe_stats']['mixed']
                mean_val = stats['mean']
                stderr = stats['stderr']
                
                ax.axhline(y=mean_val, xmin=0, xmax=0.5, color='#95a5a6',
                          linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
                ax.fill_between([min(frame_indices), 12], mean_val - stderr, mean_val + stderr,
                               color='#95a5a6', alpha=0.15, zorder=0)
                
                legend_labels.append(f'Mixed (μ={mean_val:.3f}±{stderr:.3f}, n={stats["n_samples"]})')
            else:
                legend_labels.append(f'Mixed ({len(results["mixed_sequences"])} seq)')
        
        # Add vertical line at frame 12 to mark end of statistics window
        ax.axvline(x=12, color='gray', linestyle=':', linewidth=1, alpha=0.5, zorder=0)
        ax.text(12, ax.get_ylim()[1] * 0.98, 'P-frame 12', 
               ha='center', va='top', fontsize=9, color='gray', style='italic')
        
        ax.set_xlabel('GOP Frame Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - mAP@0.5 Evolution Through GOP\n(Shaded: mean±stderr over first 12 P-frames)', 
                    fontsize=13, fontweight='bold')
        ax.set_xticks(frame_indices)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(legend_labels, fontsize=9, loc='best', framealpha=0.9)
        
        plt.tight_layout()
        evolution_path = os.path.join(plots_dir, 'map_evolution_by_camera.png')
        plt.savefig(evolution_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Frame evolution plot saved: {evolution_path}")
    
    print(f"   📁 All plots saved to: {plots_dir}")


def generate_tracking_videos(model, val_loader, args, device, max_gops=2):
    """
    Generate tracking visualization videos directly using the trained model.
    
    Args:
        model: Trained DCT-MV model
        val_loader: Validation data loader (not used, we create a new one with full data)
        args: Training arguments
        device: Device (cuda/cpu)
        max_gops: Maximum number of GOPs to visualize
    """
    print(f"\n🎬 Generating tracking videos...")
    
    # Determine video output directory
    video_output_dir = args.video_output_dir if args.video_output_dir else os.path.join(args.output_dir, 'videos')
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f"   📁 Output: {video_output_dir}")
    print(f"   🎬 Max GOPs: {max_gops}")
    
    try:
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from collections import defaultdict
        
        # Create a NEW dataset with full data (including DCT residuals and RGB frames)
        print(f"   🔄 Creating video generation dataset with full data...")
        
        from dataset.factory.dataset_factory import create_mots_dataset
        
        # For video generation, use same ablation parameters as model
        dct_coeffs = getattr(args, 'dct_coeffs', 64)
        load_mv = not getattr(args, 'no_mv', False)
        load_residuals_viz = (args.use_dct and not args.use_mv_only and dct_coeffs > 0) or True  # Always load for viz
        
        video_dataset = create_mots_dataset(
            dataset_type="mot17",
            resolution=args.resolution,
            mode="train",
            load_iframe=True,  # Load I-frames for visualization
            load_pframe=True,  # Load P-frames for visualization
            load_motion_vectors=load_mv,  # Match model's MV usage
            load_residuals=load_residuals_viz,  # Load DCT residuals for video generation
            dct_coeffs=dct_coeffs,  # ✨ Match model's DCT coefficient count
            load_annotations=True,  # Need ground truth boxes
            sequence_length=48,
            data_format="separate"
        )
        
        # Get validation sequence indices (same as training)
        total_sequences = len(video_dataset.sequences)
        train_frac = args.train_split
        val_frac = args.val_split
        
        seq_indices = list(range(total_sequences))
        split_train = int(total_sequences * train_frac)
        split_val = split_train + int(total_sequences * val_frac)
        val_seq_idx = seq_indices[split_train:split_val]
        
        # Create validation subset with frame indices
        gop_len = 48
        val_frame_indices = []
        for s in val_seq_idx:
            start = s * gop_len
            val_frame_indices.extend(list(range(start, start + gop_len)))
        
        video_val_dataset = torch.utils.data.Subset(video_dataset, val_frame_indices)
        
        # Create data loader with batch_size=1 for easier processing
        video_val_loader = DataLoader(
            video_val_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=0,  # Use 0 workers for simpler debugging
            collate_fn=lambda x: x[0] if len(x) == 1 else x  # Return single sample
        )
        
        print(f"   ✅ Video dataset created: {len(video_val_dataset)} validation samples")
        
        model.eval()
        
        # Group validation samples by GOP
        gop_groups = defaultdict(list)
        
        print(f"   🔄 Collecting GOP sequences from validation set...")
        gop_names_collected = set()
        
        # Add progress bar for data collection
        from tqdm import tqdm
        pbar = tqdm(video_val_loader, desc="   Collecting frames", unit="frame")
        for sample_idx, sample in enumerate(pbar):
            if sample is None:
                continue
            
            seq_id = sample.get('sequence_id', f'seq_{sample_idx}')
            
            # Extract GOP information from sequence_id
            if '_gop' in seq_id:
                last_gop_pos = seq_id.rfind('_gop')
                if last_gop_pos != -1:
                    video_name = seq_id[:last_gop_pos]
                    gop_part = seq_id[last_gop_pos + 4:]
                    try:
                        gop_idx = int(gop_part)
                        gop_key = f"{video_name}_gop{gop_idx}"
                        
                        # Check if we already have enough complete GOPs
                        if gop_key not in gop_names_collected:
                            # If we already have max_gops, skip new GOPs
                            if len(gop_names_collected) >= max_gops:
                                continue
                            gop_names_collected.add(gop_key)
                        
                        # Store sample data
                        sample_data = {
                            'motion_vectors': sample.get('motion_vectors'),
                            'dct_residuals': sample.get('residuals'),  # Dataset uses 'residuals', not 'dct_residuals'
                            'boxes': sample.get('boxes'),
                            'frame_idx': sample.get('frame_index', 0),
                            'iframe': sample.get('iframe'),
                            'pframe': sample.get('pframe')
                        }
                        gop_groups[gop_key].append(sample_data)
                        
                        # Early exit once we have enough complete GOPs
                        # (each GOP has 48 frames, so check if we've collected enough frames)
                        if len(gop_names_collected) >= max_gops:
                            min_frames_per_gop = min(len(frames) for frames in gop_groups.values())
                            if min_frames_per_gop >= 45:  # Close to complete GOP (48 frames)
                                pbar.close()
                                break
                        
                    except:
                        continue
        
        print(f"   📊 Found {len(gop_groups)} GOPs to visualize")
        
        # Generate videos for each GOP
        videos_created = 0
        for gop_idx, (gop_name, frames) in enumerate(list(gop_groups.items())[:max_gops]):
            print(f"   🎬 Creating video {gop_idx + 1}/{min(max_gops, len(gop_groups))}: {gop_name} ({len(frames)} frames collected)")
            
            # Sort frames by frame index
            frames.sort(key=lambda x: x['frame_idx'])
            
            # Debug: Check first frame structure
            if len(frames) > 0:
                first_frame = frames[0]
                print(f"      🔍 Debug first frame: mv={type(first_frame.get('motion_vectors'))}, dct={type(first_frame.get('dct_residuals'))}, boxes={type(first_frame.get('boxes'))}")
            
            # Create video
            video_path = os.path.join(video_output_dir, f'{gop_name}_tracking.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_size = (1280, 720)
            out = cv2.VideoWriter(video_path, fourcc, 10.0, frame_size)
            
            frames_written = 0
            frames_skipped = 0
            
            # Initialize tracking state for the GOP (like in training)
            hidden_state = None
            current_boxes = frames[0]['boxes'].to(device) if len(frames) > 0 and frames[0].get('boxes') is not None else None
            
            # Add progress bar for frame processing
            with torch.no_grad():
                for frame_idx, frame_data in enumerate(tqdm(frames[:50], desc=f"   Processing {gop_name}", unit="frame")):  # Limit to 50 frames
                    try:
                        # Validate frame data
                        mv = frame_data.get('motion_vectors')
                        dct = frame_data.get('dct_residuals')
                        gt_boxes = frame_data.get('boxes')
                        iframe = frame_data.get('iframe')
                        pframe = frame_data.get('pframe')
                        
                        if mv is None:
                            frames_skipped += 1
                            continue
                        if dct is None:
                            frames_skipped += 1
                            continue
                        
                        # Get RGB frame for visualization (iframe for first frame, pframe for others)
                        rgb_frame = iframe if frame_idx == 0 and iframe is not None else pframe
                        if rgb_frame is None:
                            frames_skipped += 1
                            continue
                        
                        # Ensure tensors
                        if not isinstance(mv, torch.Tensor):
                            frames_skipped += 1
                            continue
                        if not isinstance(dct, torch.Tensor):
                            frames_skipped += 1
                            continue
                        
                        # Debug: Print original shapes for first frame only
                        if frame_idx == 0:
                            print(f"\n         🔍 First frame shapes: mv={mv.shape}, dct={dct.shape}")
                        
                        # Process tensors EXACTLY as in training pipeline (_train_on_gop_sequence)
                        # Motion vectors: [H, W, 2] -> [2, H, W] -> [1, 2, H, W]
                        if mv.dim() == 4 and mv.shape[0] > 1:
                            # Handle batch dimension artifact from data loader
                            mv = mv[0]  # Take first sample: [H, W, 2]
                        
                        if mv.dim() == 3:
                            if mv.shape[-1] == 2:  # [H, W, 2] - channels last
                                mv = mv.permute(2, 0, 1)  # [2, H, W]
                            # Now should be [2, H, W]
                            mv = mv.unsqueeze(0).to(device)  # [1, 2, H, W]
                        else:
                            frames_skipped += 1
                            continue
                        
                        # DCT residuals: [H, W, 64] -> [1, H, W, 64] (CHANNELS LAST!)
                        # ⚠️ CRITICAL: Training expects [B, H, W, C], NOT [B, C, H, W]
                        if dct.dim() == 3:
                            # Should be [H, W, 64] - channels last
                            dct = dct.unsqueeze(0).to(device)  # [1, H, W, 64] - keep channels last!
                        else:
                            frames_skipped += 1
                            continue
                        
                        # Debug: Print processed shapes for first frame only
                        if frame_idx == 0:
                            print(f"         🔍 Processed shapes: mv={mv.shape}, dct={dct.shape}")
                            if gt_boxes is not None and isinstance(gt_boxes, torch.Tensor):
                                print(f"         🔍 Ground truth boxes shape: {gt_boxes.shape}\n")
                        
                        if gt_boxes is not None and isinstance(gt_boxes, torch.Tensor):
                            gt_boxes = gt_boxes.to(device)
                        
                        # Get predictions
                        # In training, boxes are passed directly without batch dimension
                        # forward_single_frame expects boxes as [N, 4], not [1, N, 4]
                        # Also pass and update hidden_state like in training
                        if current_boxes is not None and len(current_boxes) > 0:
                            pred_boxes, pred_confs, hidden_state = model.forward_single_frame(mv, dct, current_boxes, hidden_state)
                            current_boxes = pred_boxes  # Update for next frame (like in training)
                        else:
                            # Create dummy initial boxes [N, 4] format
                            dummy_boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]], device=device)
                            pred_boxes, pred_confs, hidden_state = model.forward_single_frame(mv, dct, dummy_boxes, hidden_state)
                            current_boxes = pred_boxes  # Update for next frame
                        
                        # Create visualization with RGB frame as background
                        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                        
                        # Display RGB frame as background
                        if isinstance(rgb_frame, torch.Tensor):
                            rgb_np = rgb_frame.cpu().numpy()
                            
                            # Remove batch dimension if present: [1, H, W, C] -> [H, W, C]
                            if rgb_np.ndim == 4 and rgb_np.shape[0] == 1:
                                rgb_np = rgb_np[0]
                            
                            # Handle different RGB frame formats
                            if rgb_np.ndim == 3:
                                if rgb_np.shape[0] == 3:  # [C, H, W]
                                    rgb_np = rgb_np.transpose(1, 2, 0)  # [H, W, C]
                                # Now should be [H, W, C]
                            
                            # Normalize if needed
                            if rgb_np.max() > 1.0:
                                rgb_np = rgb_np / 255.0
                            
                            ax.imshow(rgb_np)
                        
                        # Visualize motion vectors as quiver plot overlay
                        mv_np = mv[0].cpu().numpy()
                        h, w = mv_np.shape[1:]
                        y, x = np.mgrid[0:h:4, 0:w:4]
                        u = mv_np[0, ::4, ::4]
                        v = mv_np[1, ::4, ::4]
                        ax.quiver(x, y, u, v, scale=50, color='cyan', alpha=0.6, width=0.003)
                        
                        # Draw predicted boxes (red)
                        # pred_boxes is [N, 4], not [1, N, 4]
                        if pred_boxes is not None and len(pred_boxes) > 0:
                            pred_boxes_np = pred_boxes.cpu().numpy()
                            for box in pred_boxes_np:
                                cx, cy, w_box, h_box = box
                                # Scale to image dimensions
                                img_h, img_w = rgb_np.shape[:2] if isinstance(rgb_frame, torch.Tensor) else (h, w)
                                x_min = (cx - w_box/2) * img_w
                                y_min = (cy - h_box/2) * img_h
                                width = w_box * img_w
                                height = h_box * img_h
                                rect = plt.Rectangle((x_min, y_min), width, height,
                                                   linewidth=3, edgecolor='red', facecolor='none', label='Prediction')
                                ax.add_patch(rect)
                        
                        # Draw ground truth boxes (green)
                        if gt_boxes is not None and isinstance(gt_boxes, torch.Tensor):
                            gt_boxes_np = gt_boxes.cpu().numpy()
                            for box in gt_boxes_np:
                                cx, cy, w_box, h_box = box
                                # Scale to image dimensions
                                img_h, img_w = rgb_np.shape[:2] if isinstance(rgb_frame, torch.Tensor) else (h, w)
                                x_min = (cx - w_box/2) * img_w
                                y_min = (cy - h_box/2) * img_h
                                width = w_box * img_w
                                height = h_box * img_h
                                rect = plt.Rectangle((x_min, y_min), width, height,
                                                   linewidth=3, edgecolor='green', facecolor='none', linestyle='--', label='Ground Truth')
                                ax.add_patch(rect)
                        
                        # Set axis limits based on RGB frame dimensions
                        if isinstance(rgb_frame, torch.Tensor):
                            img_h, img_w = rgb_np.shape[:2]
                            ax.set_xlim(0, img_w)
                            ax.set_ylim(img_h, 0)
                        else:
                            ax.set_xlim(0, w)
                            ax.set_ylim(h, 0)
                        ax.set_title(f'{gop_name} - Frame {frame_data["frame_idx"]} - RED=Pred | GREEN=GT | CYAN=MotionVectors')
                        ax.axis('off')  # Hide axis for cleaner video
                        
                        # Convert to video frame
                        plt.tight_layout()
                        fig.canvas.draw()
                        # Use buffer_rgba() instead of deprecated tostring_rgb()
                        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                        img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                        # Convert RGBA to RGB then to BGR
                        img_rgb = img[:, :, :3]  # Drop alpha channel
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        img_resized = cv2.resize(img_bgr, frame_size)
                        out.write(img_resized)
                        plt.close(fig)
                        
                        frames_written += 1
                    except Exception as e:
                        # Only print first error, then suppress to avoid spam
                        if frames_skipped == 0:
                            print(f"\n      ⚠️ Error processing frame {frame_idx}: {e}")
                            print(f"         (Further errors will be suppressed)")
                        frames_skipped += 1
                        continue
            
            out.release()
            
            if frames_written > 0:
                print(f"\n      ✅ Video created: {frames_written} frames written, {frames_skipped} skipped")
                videos_created += 1
            else:
                print(f"\n      ⚠️ No frames written ({frames_skipped} frames skipped)")
        
        print(f"   ✅ Generated {videos_created} videos!")
        print(f"   📁 Videos saved in: {video_output_dir}")
        return videos_created > 0
        
    except ImportError as e:
        print(f"   ⚠️ Missing required library: {e}")
        print(f"   Video generation skipped")
        return False
    except Exception as e:
        print(f"   ⚠️ Error generating videos: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training function."""
    args = parse_args()
    
    print(f"🚀 MV-Center Modular Training")
    print(f"   Version: {args.version.upper()}")
    print(f"   Config: {args.config}")
    print(f"   Motion: {'3-channel (u,v,mag)' if args.use_magnitude else '2-channel (u,v)'}")
    print(f"   Embeddings: {args.use_embeddings}")
    
    # Setup environment
    device = setup_environment(args)
    print(f"   Device: {device}")
    
    if args.dry_run:
        print("✅ Dry run completed - configuration looks good!")
        return 0
    
    try:
        # Create model (and criterion if memory tracker)
        model, criterion = create_model(args, device)
        
        # Determine if using memory tracker
        # ENHANCED MODEL: Always use memory tracker path (GOP-based training)
        use_memory = args.use_memory or args.version == 'memory' or args.use_dct or args.use_mv_enhanced
        
        # Create dataset and dataloaders (train/val split by sequences)
        # Load training dataset (MOT15+MOT17+MOT20 train splits)
        train_full_dataset = create_dataset(args)
        print(f"\n   📊 Preparing training and validation datasets...", flush=True)
        
        #  Create separate validation dataset from test splits
        print(f"   🔍 Loading validation dataset (MOT15+MOT17+MOT20 test splits)...")
        # Use same ablation parameters as training
        dct_coeffs = getattr(args, 'dct_coeffs', 64)
        load_mv = not getattr(args, 'no_mv', False)
        load_residuals = args.use_dct and not args.use_mv_only and dct_coeffs > 0
        
        val_full_dataset = create_mots_dataset(
            dataset_type=['mot15', 'mot17', 'mot20'],
            resolution=args.resolution,
            mode="test",  # Use test splits for validation
            load_iframe=False,
            load_pframe=False,
            load_motion_vectors=load_mv,
            load_residuals=load_residuals,
            dct_coeffs=dct_coeffs,  # ✨ NEW: Match training DCT coefficients
            load_annotations=True,
            sequence_length=48,
            data_format="separate",
            combine_datasets=True
        )

        total_train_sequences = len(train_full_dataset.sequences)
        total_val_sequences = len(val_full_dataset.sequences)
        
        # For training: use all training sequences
        train_seq_idx = list(range(total_train_sequences))
        # For validation: use all test sequences
        val_seq_idx = list(range(total_val_sequences))
        
        print(f"   Sequences: train={len(train_seq_idx)}, val={len(val_seq_idx)} (using separate test sets)", flush=True)

        # Create frame indices for training dataset
        gop_len = getattr(train_full_dataset, 'sequence_length', 48)
        train_frame_indices = []
        for s in train_seq_idx:
            start = s * gop_len
            train_frame_indices.extend(list(range(start, start + gop_len)))

        # Respect max_samples if provided (limit total frames used for training subset)
        if args.max_samples:
            train_frame_indices = train_frame_indices[:args.max_samples]

        train_dataset = torch.utils.data.Subset(train_full_dataset, train_frame_indices)
        
        # For validation: use all frames from validation dataset
        val_gop_len = getattr(val_full_dataset, 'sequence_length', 48)
        val_frame_indices = []
        for s in val_seq_idx:
            start = s * val_gop_len
            val_frame_indices.extend(list(range(start, start + val_gop_len)))
        
        val_dataset = val_full_dataset  # Use full validation dataset

        print(f"   📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}", flush=True)
        
        # Update full_dataset to be the training dataset for compatibility
        full_dataset = train_full_dataset
        
        # 📋 Track which GOPs are in training vs validation
        train_gops_info = []
        val_gops_info = []
        
        # Extract GOP information from dataset
        for gop_idx in train_seq_idx:
            if hasattr(full_dataset, 'sequences') and gop_idx < len(full_dataset.sequences):
                gop_info = full_dataset.sequences[gop_idx]
                train_gops_info.append({
                    'gop_index': gop_idx,
                    'sequence_id': gop_info.get('sequence_id', f'gop_{gop_idx}'),
                    'video_name': gop_info.get('video_name', 'unknown'),
                    'num_frames': gop_len
                })
        
        for gop_idx in val_seq_idx:
            if hasattr(full_dataset, 'sequences') and gop_idx < len(full_dataset.sequences):
                gop_info = full_dataset.sequences[gop_idx]
                val_gops_info.append({
                    'gop_index': gop_idx,
                    'sequence_id': gop_info.get('sequence_id', f'gop_{gop_idx}'),
                    'video_name': gop_info.get('video_name', 'unknown'),
                    'num_frames': gop_len
                })
        
        # Save GOP split information
        gop_split_info = {
            'train_gops': train_gops_info,
            'val_gops': val_gops_info,
            'train_count': len(train_gops_info),
            'val_count': len(val_gops_info),
            'gop_length': gop_len
        }
        
        gop_split_path = os.path.join(args.output_dir, 'gop_split_info.json')
        with open(gop_split_path, 'w') as f:
            json.dump(gop_split_info, f, indent=2)
        
        print(f"   📋 GOP split saved to: {gop_split_path}", flush=True)
        print(f"   📊 Training GOPs: {len(train_gops_info)}, Validation GOPs: {len(val_gops_info)}", flush=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=custom_collate_fn,
                                  pin_memory=device.type == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=max(0, args.num_workers//2), collate_fn=custom_collate_fn,
                                pin_memory=device.type == 'cuda')

        print(f"   ✅ DataLoaders ready: train_batches={len(train_loader)}, val_batches={len(val_loader)}", flush=True)
        
        # Setup optimizer and scheduler (MV-Center paper settings)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler (as per MV-Center paper)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
        
        # AMP scaler if enabled
        scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
        
        print(f"   🎯 Optimizer: AdamW (lr={args.learning_rate}, wd={args.weight_decay})", flush=True)
        print(f"   📅 Scheduler: CosineAnnealingLR", flush=True)
        print(f"   ⚡ AMP: {args.amp and device.type == 'cuda'}", flush=True)
        print(f"   ✂️  Gradient Clipping: {'ENABLED (max_norm=' + str(args.gradient_clip) + ')' if args.gradient_clip > 0 else 'DISABLED'}", flush=True)
        
        print(f"\n   📋 Initializing training loop...", flush=True)
        
        # Training loop
        if use_memory:
            # Memory tracker uses different loss structure
            loss_history = {
                'total_loss': [],
                'box_loss': [],
                'confidence_loss': [],
                'id_loss': [],
                'negative_loss': [],
                'giou_loss': [],
                'classification_loss': [],
                'no_object_loss': [],
                # ⚠️ REMOVED: Training mAP tracking (now only in validation)
                # 'train_map': [],
                # 'static_baseline_map': [],
                'val_map': []
            }
        else:
            # Regular tracker
            loss_history = {
                'total_loss': [],
                'center_loss': [],
                'box_loss': [],
                'giou_loss': []
            }
        
        print(f"   ✅ Loss history initialized", flush=True)
        print(f"\n🏃 Starting training for {args.epochs} epochs...", flush=True)
        
        # Track best model
        best_val_map = 0.0
        best_epoch = -1
        
        # Load balanced GOPs once for all epochs (memory-efficient)
        ordered_gops = None
        ordered_val_gops = None
        if use_memory:
            print(f"   🧠 Using Memory-Based LSTM Tracker (balanced GOP loading)", flush=True)
            print(f"   � Loading max_gops={args.max_gops} GOPs balanced across all videos", flush=True)
            
            # Load balanced subset of GOPs once
            gop_length = getattr(full_dataset, 'sequence_length', 48)
            ordered_gops = load_balanced_gops(train_loader, max_gops=args.max_gops, gop_length=gop_length)
            
            print(f"   ✅ Loaded {len(ordered_gops)} training GOPs (will be reused for all {args.epochs} epochs)", flush=True)
            
            # Load balanced subset of validation GOPs once
            print(f"   📊 Loading max_val_gops={args.max_val_gops} validation GOPs balanced across all videos", flush=True)
            ordered_val_gops = load_balanced_gops(val_loader, max_gops=args.max_val_gops, gop_length=gop_length)
            print(f"   ✅ Loaded {len(ordered_val_gops)} validation GOPs (will be reused for all validations)", flush=True)
        print("   ⏱️  Training begins now!\n", flush=True)
        
        for epoch in range(args.epochs):
            # Train one epoch (use appropriate training function)
            if use_memory:
                epoch_metrics = train_epoch_memory(model, ordered_gops, criterion, optimizer, scaler, args, device, epoch, train_loader=None, gop_cache=None)
            else:
                epoch_metrics = train_epoch(model, train_loader, optimizer, scaler, args, device, epoch)
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Store loss history
            if use_memory:
                loss_history['total_loss'].append(epoch_metrics['total_loss'])
                loss_history['box_loss'].append(epoch_metrics['box_loss'])
                # velocity_loss removed from metrics
                loss_history['confidence_loss'].append(epoch_metrics['confidence_loss'])
                # detection components
                loss_history['giou_loss'].append(epoch_metrics.get('giou_loss', 0.0))
                loss_history['classification_loss'].append(epoch_metrics.get('classification_loss', 0.0))
                loss_history['no_object_loss'].append(epoch_metrics.get('no_object_loss', 0.0))
                # id and negative losses
                loss_history['id_loss'].append(epoch_metrics.get('id_loss', 0.0))
                loss_history['negative_loss'].append(epoch_metrics.get('negative', 0.0))
                # ⚠️ REMOVED: Training mAP tracking (now only in validation)
                # loss_history['train_map'].append(epoch_metrics.get('train_map', 0.0))
                # loss_history['static_baseline_map'].append(epoch_metrics.get('static_baseline_map', 0.0))
                
                # ✅ Run validation every 5 epochs (or last epoch)
                if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                    print(f"\n🔍 Running validation at epoch {epoch+1}...", flush=True)
                    val_map = evaluate_map(model, full_dataset, device, max_eval_sequences=len(val_seq_idx), ordered_val_gops=ordered_val_gops)
                    loss_history['val_map'].append(val_map)
                    print(f"   📈 Validation mAP@0.5: {val_map:.4f}")
                    
                    # Track best model
                    if val_map > best_val_map:
                        best_val_map = val_map
                        best_epoch = epoch + 1
                        # Save best model checkpoint
                        save_best_checkpoint(model, optimizer, epoch, loss_history, val_map, args)
                        print(f"   🏆 New best model! mAP@0.5: {val_map:.4f} (epoch {epoch+1})")
                    
                else:
                    # Skip validation for this epoch (use last known value or 0)
                    last_val_map = loss_history['val_map'][-1] if loss_history['val_map'] else 0.0
                    loss_history['val_map'].append(last_val_map)
                
                # Print epoch summary
                print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
                print(f"   Total Loss: {epoch_metrics['total_loss']:.6f}")
                print(f"   Box Loss: {epoch_metrics['box_loss']:.6f}")
                # velocity_loss removed from prints
                print(f"   Confidence Loss: {epoch_metrics['confidence_loss']:.6f}")
                # ⚠️ Training mAP removed - only shown during validation
                print(f"   Batches: {epoch_metrics['num_batches']}")
                print(f"   Learning Rate: {current_lr:.2e}")
                
                # Print dynamic balancer stats if enabled
                if args.use_dynamic_balancing and hasattr(criterion, 'balancer') and criterion.balancer is not None:
                    criterion.balancer.print_stats()
            else:
                loss_history['total_loss'].append(epoch_metrics['total_loss'])
                loss_history['center_loss'].append(epoch_metrics['center_loss'])
                loss_history['box_loss'].append(epoch_metrics['box_loss'])
                loss_history['giou_loss'].append(epoch_metrics['giou_loss'])
                
                # Print epoch summary
                print(f"\n📈 Epoch {epoch+1}/{args.epochs} Summary:")
                print(f"   Total Loss: {epoch_metrics['total_loss']:.6f}")
                print(f"   Center Loss: {epoch_metrics['center_loss']:.6f}")
                print(f"   Box Loss: {epoch_metrics['box_loss']:.6f}")
                print(f"   GIoU Loss: {epoch_metrics['giou_loss']:.6f}")
                print(f"   Batches: {epoch_metrics['num_batches']}")
                print(f"   Learning Rate: {current_lr:.2e}")
            print("hello")
            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
                save_checkpoint(model, optimizer, epoch, loss_history, args)
            
            # Plot training curves
            if len(loss_history['total_loss']) > 1:
                plot_training_curves(loss_history, args.output_dir)
        
        # ✅ Per-sequence evaluation and plotting for best model
        print(f"\n{'='*80}")
        print(f"🏆 Best Model Evaluation and Per-Sequence Analysis")
        print(f"{'='*80}")
        
        if best_epoch > 0 and use_memory:
            print(f"\n📊 Best model achieved at epoch {best_epoch} with mAP@0.5: {best_val_map:.4f}")
            
            # Load best model for per-sequence evaluation
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                print(f"   📂 Loading best model from: {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Run comprehensive per-sequence evaluation
                print(f"\n🔍 Running detailed per-sequence evaluation on best model...")
                per_seq_results = evaluate_per_sequence(
                    model, 
                    full_dataset, 
                    device, 
                    ordered_val_gops=ordered_val_gops,
                    max_eval_sequences=len(val_seq_idx)
                )
                
                # Generate per-sequence plots
                model_name = f"{args.version}_{args.config}"
                if args.use_dct and not args.no_mv:
                    model_name += f"_MV+DCT-{args.dct_coeffs}"
                elif args.use_dct and args.no_mv:
                    model_name += f"_DCT-{args.dct_coeffs}"
                elif args.dct_coeffs == 0:
                    model_name += "_MV-only"
                
                plot_per_sequence_results(per_seq_results, args.output_dir, model_name=model_name)
                
                # Save detailed per-sequence results to JSON
                per_seq_path = os.path.join(args.output_dir, 'per_sequence_results.json')
                with open(per_seq_path, 'w') as f:
                    json.dump(per_seq_results, f, indent=2)
                print(f"\n   💾 Per-sequence results saved: {per_seq_path}")
                
                print(f"\n✅ Best model per-sequence analysis complete!")
                print(f"   📊 Overall mAP@0.5: {per_seq_results['overall_avg']:.4f}")
                print(f"   🎥 Static mAP@0.5:  {per_seq_results['static_avg']:.4f}")
                print(f"   📹 Moving mAP@0.5:  {per_seq_results['moving_avg']:.4f}")
                if per_seq_results['mixed_avg'] > 0:
                    print(f"   🎬 Mixed mAP@0.5:   {per_seq_results['mixed_avg']:.4f}")
            else:
                print(f"   ⚠️  Best model checkpoint not found: {best_model_path}")
        else:
            print(f"   ⚠️  No valid best model found (best_epoch={best_epoch})")
        
        print(f"\n{'='*80}\n")
        
        # Save final results
        results = {
            'model_version': args.version,
            'model_config': args.config,
            'use_magnitude': args.use_magnitude,
            'use_embeddings': args.use_embeddings,
            'final_loss': loss_history['total_loss'][-1],
            'best_val_map': best_val_map,
            'best_epoch': best_epoch,
            'loss_history': loss_history,
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
            'training_args': vars(args)
        }
        
        results_path = os.path.join(args.output_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Training completed successfully!")
        print(f"   Final loss: {loss_history['total_loss'][-1]:.6f}")
        print(f"   Results saved: {results_path}")
        
        # Generate tracking videos if requested
        if args.generate_videos:
            generate_tracking_videos(model, val_loader, args, device, max_gops=args.max_video_gops)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
