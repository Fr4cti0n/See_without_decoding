#!/usr/bin/env python3
"""
Model Speed Estimation Script for Ablation Study Variants

Measures inference speed and memory usage for all 9 ablation variants:
- MV-only (motion vectors only)
- DCT-8/16/32/64 (DCT residuals only)
- MV+DCT-8/16/32/64 (hybrid models)

Metrics:
- Inference speed: FPS, ms/frame, ms/GOP
- Memory usage: Model size (MB), RAM/VRAM usage (MB)
- Comparison across all variants (CPU and GPU)

Compatible with DCT-MV models and balanced validation GOP loading.
"""

import sys
import os
import torch
import time
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import psutil
import warnings
warnings.filterwarnings('ignore')

# Add project paths
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import dataset factory (same as training script)
try:
    dataset_dir = os.path.join(project_root, 'dataset')
    if dataset_dir not in sys.path:
        sys.path.insert(0, dataset_dir)
    
    from dataset.factory.dataset_factory import create_mots_dataset
    print("✅ Dataset factory imported successfully")
    HAS_REAL_DATASET = True
except ImportError as e:
    print(f"⚠️ Dataset factory import failed: {e}")
    HAS_REAL_DATASET = False
    create_mots_dataset = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ablation Variants Speed Benchmark")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to single model checkpoint (.pt file), or omit to benchmark all variants")
    parser.add_argument("--results-dir", type=str, default="experiments/ablation_fast",
                       help="Directory containing all ablation variant results (default: experiments/ablation_fast)")
    parser.add_argument("--benchmark-all", action="store_true",
                       help="Benchmark all 9 ablation variants automatically")
    parser.add_argument("--use-fast", action="store_true",
                       help="Use fast architecture (global pooling + 1-layer LSTM)")
    parser.add_argument("--use-ultra-fast", action="store_true",
                       help="Use ultra-fast architecture (global pooling + feedforward, no LSTM)")
    parser.add_argument("--resolution", type=int, default=960, choices=[640, 960], 
                       help="Video resolution (default: 960)")
    parser.add_argument("--num-gops", type=int, default=30, 
                       help="Number of validation GOPs to benchmark (default: 30)")
    parser.add_argument("--max-val-gops", type=int, default=30,
                       help="Maximum validation GOPs to load (default: 30)")
    parser.add_argument("--warmup", type=int, default=5, 
                       help="Number of warmup iterations (default: 5)")
    parser.add_argument("--device", type=str, default=None, 
                       help="Device (cuda/cpu), auto-detect if not specified")
    parser.add_argument("--output", type=str, default="speed_results.json", 
                       help="Output JSON file for results")
    parser.add_argument("--compare-baselines", action="store_true",
                       help="Compare with static baseline (I-frame copy)")
    parser.add_argument("--train-split", type=float, default=0.5,
                       help="Training split fraction (default: 0.5)")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split fraction (default: 0.2)")
    parser.add_argument("--gop-length", type=int, default=48,
                       help="GOP length in frames (default: 48)")
    parser.add_argument("--print-summary", action="store_true",
                       help="Print comparison table for all variants")
    parser.add_argument("--export-latex", action="store_true",
                       help="Export results to LaTeX table format")
    parser.add_argument("--rtdetr-model-size", type=float, default=63.0,
                       help="RT-DETR model size in MB (default: 63.0)")
    parser.add_argument("--rtdetr-peak-vram", type=float, default=1500.0,
                       help="RT-DETR peak VRAM in MB (default: 1500.0)")
    parser.add_argument("--rtdetr-fps", type=float, default=45.0,
                       help="RT-DETR FPS on same hardware (default: 45.0)")
    parser.add_argument("--rtdetr-map", type=float, default=0.851,
                       help="RT-DETR mAP@0.5 (default: 0.851)")
    return parser.parse_args()


def load_model(model_path, device, use_fast=False, use_ultra_fast=False):
    """Load trained model from checkpoint (supports all ablation variants + fast architectures)."""
    print(f"\n📦 Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    args_dict = checkpoint.get('args', {})
    
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Detect model type from args
    use_dct = args_dict.get('use_dct', False)
    dct_coeffs = args_dict.get('dct_coeffs', 64)
    no_mv = args_dict.get('no_mv', False)
    
    # Check if checkpoint was trained with fast architecture
    checkpoint_use_fast = args_dict.get('use_fast', False)
    checkpoint_use_ultra_fast = args_dict.get('use_ultra_fast', False)
    
    # Override with command-line flags if provided, otherwise use checkpoint flags
    if use_fast or use_ultra_fast:
        actual_use_fast = use_fast
        actual_use_ultra_fast = use_ultra_fast
    else:
        actual_use_fast = checkpoint_use_fast
        actual_use_ultra_fast = checkpoint_use_ultra_fast
    
    # Determine model configuration
    mv_channels = 0 if no_mv else 2
    dct_channels = dct_coeffs if use_dct else 0
    
    # Model is DCT-MV if use_dct is True (regardless of mv/dct channels)
    if use_dct:
        # Decide between fast and standard architecture
        if actual_use_ultra_fast:
            from mots_exp.models.dct_mv_center.fast_dct_mv_tracker import UltraFastDCTMVTracker
            
            # Extract model configuration from checkpoint
            feature_dim = args_dict.get('feature_dim', 64)
            hidden_dim = args_dict.get('hidden_dim', 128)  # Not used in ultra-fast but keep for consistency
            image_size = args_dict.get('resolution', 960)
            
            print(f"   Model type: Ultra-Fast DCT-MV Tracker (Global Pooling + Feedforward)")
            print(f"   Config: MV channels={mv_channels}, DCT channels={dct_channels}")
            print(f"   Feature dim={feature_dim}, image_size={image_size}")
            
            # CRITICAL: num_dct_coeffs must match dct_channels for the coefficient selector
            num_dct_coeffs = dct_channels if dct_channels > 0 else 16
            
            # Create ultra-fast model
            model = UltraFastDCTMVTracker(
                num_dct_coeffs=num_dct_coeffs,
                mv_channels=mv_channels,
                dct_channels=dct_channels,
                mv_feature_dim=32,
                dct_feature_dim=32,
                fused_feature_dim=feature_dim,
                feedforward_dim=hidden_dim,  # Used as feedforward hidden size
                dropout=0.1,
                image_size=image_size
            )
            
            model_type = "Ultra-Fast DCT-MV Tracker"
            
        elif actual_use_fast:
            from mots_exp.models.dct_mv_center.fast_dct_mv_tracker import FastDCTMVTracker
            
            # Extract model configuration from checkpoint
            feature_dim = args_dict.get('feature_dim', 64)
            hidden_dim = args_dict.get('hidden_dim', 128)
            image_size = args_dict.get('resolution', 960)
            
            print(f"   Model type: Fast DCT-MV Tracker (Global Pooling + 1-layer LSTM)")
            print(f"   Config: MV channels={mv_channels}, DCT channels={dct_channels}")
            print(f"   Feature dim={feature_dim}, hidden_dim={hidden_dim}, image_size={image_size}")
            
            # CRITICAL: num_dct_coeffs must match dct_channels for the coefficient selector
            num_dct_coeffs = dct_channels if dct_channels > 0 else 16
            
            # Create fast model
            model = FastDCTMVTracker(
                num_dct_coeffs=num_dct_coeffs,
                mv_channels=mv_channels,
                dct_channels=dct_channels,
                mv_feature_dim=32,
                dct_feature_dim=32,
                fused_feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                lstm_layers=1,  # Fast model uses 1 LSTM layer
                dropout=0.1,
                image_size=image_size
            )
            
            model_type = "Fast DCT-MV Tracker"
            
        else:
            # Standard architecture with ROI pooling and attention
            from mots_exp.models.dct_mv_center import DCTMVCenterTracker
            
            # Extract model configuration from checkpoint
            feature_dim = args_dict.get('feature_dim', 64)
            hidden_dim = args_dict.get('hidden_dim', 128)
            image_size = args_dict.get('resolution', 960)
            use_multiscale = args_dict.get('use_multiscale_roi', False)
            use_parallel = args_dict.get('use_parallel_heads', False)
            use_attention = args_dict.get('use_attention', False)
            
            print(f"   Model type: Standard DCT-MV Tracker")
            print(f"   Config: MV channels={mv_channels}, DCT channels={dct_channels}")
            print(f"   Feature dim={feature_dim}, hidden_dim={hidden_dim}, image_size={image_size}")
            print(f"   Architecture: multiscale={use_multiscale}, parallel={use_parallel}, attention={use_attention}")
            
            # CRITICAL: num_dct_coeffs must match dct_channels for the coefficient selector
            num_dct_coeffs = dct_channels if dct_channels > 0 else 16
            
            # Create model with same architecture as training
            model = DCTMVCenterTracker(
                num_dct_coeffs=num_dct_coeffs,  # MUST match what was used during training!
                mv_channels=mv_channels,
                dct_channels=dct_channels,
                mv_feature_dim=32,
                dct_feature_dim=32,
                fused_feature_dim=feature_dim,
                roi_feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                lstm_layers=2,
                dropout=0.1,
                image_size=image_size,
                use_multiscale_roi=use_multiscale,
                roi_sizes=args_dict.get('roi_sizes', [3, 7, 11]),
                use_parallel_heads=use_parallel,
                use_attention=use_attention,
                attention_heads=args_dict.get('attention_heads', 4)
            )
            
            model_type = "Standard DCT-MV Tracker"
    else:
        raise ValueError(f"Unknown model type in checkpoint: use_dct={use_dct}")
    
    # Load weights with error handling for architecture mismatches
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"   ⚠️ Warning: State dict mismatch, trying strict=False loading...")
        print(f"      Error: {str(e)[:200]}...")
        # Try loading with strict=False to handle minor mismatches
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"   ⚠️ Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys: {unexpected_keys[:5]}...")
        print(f"   ✅ Loaded weights with strict=False (some layers may not match)")
    
    model = model.to(device)
    model.eval()
    
    # Count parameters and calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1024 / 1024  # FP32 = 4 bytes per parameter
    
    print(f"✅ Model loaded successfully")
    print(f"   Total parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"   Model size: {model_size_mb:.2f} MB (FP32)")
    
    model_info = {
        'model_type': model_type,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': round(model_size_mb, 2),
        'mv_channels': mv_channels,
        'dct_channels': dct_channels,
        'feature_dim': feature_dim if use_dct else None,
        'hidden_dim': hidden_dim if use_dct else None,
        'image_size': image_size if use_dct else None,
        'use_fast': actual_use_fast,
        'use_ultra_fast': actual_use_ultra_fast,
        'use_multiscale_roi': use_multiscale if not (actual_use_fast or actual_use_ultra_fast) else False,
        'use_parallel_heads': use_parallel if not (actual_use_fast or actual_use_ultra_fast) else False,
        'use_attention': use_attention if not (actual_use_fast or actual_use_ultra_fast) else False,
    }
    
    return model, model_info


def get_memory_usage():
    """Get current CPU and GPU memory usage."""
    memory_info = {}
    
    # CPU memory
    process = psutil.Process()
    memory_info['cpu_mb'] = process.memory_info().rss / 1024 / 1024
    
    # GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    return memory_info


def load_validation_gops(args):
    """
    Load balanced validation GOP sequences (same as training script).
    
    Returns:
        List of (seq_id, gop_frames) tuples
    """
    print(f"\n📚 Loading validation dataset...")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   Max validation GOPs: {args.max_val_gops}")
    
    if not HAS_REAL_DATASET or create_mots_dataset is None:
        raise RuntimeError("❌ Real MOTS dataset factory not available!")
    
    # Determine what to load based on whether we need DCT residuals
    # For speed benchmarking all variants, we load both MV and DCT
    # Models will ignore what they don't need
    
    # Create combined validation dataset (same as training)
    dataset = create_mots_dataset(
        dataset_type=['mot15', 'mot17', 'mot20'],
        resolution=args.resolution,
        mode="train",
        load_iframe=False,
        load_pframe=False,
        load_motion_vectors=True,  # Always load MV
        load_residuals=True,  # Always load DCT (models will ignore if not needed)
        dct_coeffs=64,  # Load max coefficients (models will use subset)
        load_annotations=True,
        sequence_length=48,
        data_format="separate",
        combine_datasets=True
    )
    
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("❌ Dataset is empty or failed to load!")
    
    print(f"   ✅ Dataset loaded: {len(dataset)} total samples")
    
    # Create validation split (same logic as training script)
    from torch.utils.data import DataLoader, Subset
    import random
    
    total_sequences = len(dataset.sequences)
    seq_indices = list(range(total_sequences))
    random.seed(42)  # Same seed as training for consistency
    random.shuffle(seq_indices)
    
    split_train = int(total_sequences * args.train_split)
    split_val = split_train + int(total_sequences * args.val_split)
    val_seq_idx = seq_indices[split_train:split_val]
    
    print(f"   📊 Total sequences: {total_sequences}")
    print(f"   🎯 Validation sequences: {len(val_seq_idx)} ({len(val_seq_idx)/total_sequences*100:.1f}%)")
    
    # Create validation subset
    gop_len = args.gop_length
    val_frame_indices = []
    for s in val_seq_idx:
        start_idx = s * gop_len
        end_idx = start_idx + gop_len
        val_frame_indices.extend(range(start_idx, min(end_idx, len(dataset))))
    
    val_dataset = Subset(dataset, val_frame_indices)
    
    # Load balanced validation GOPs with parallel I/O and RAM caching
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"\n🎯 Loading validation GOP sequences (RAM caching with parallel I/O)...")
    
    # Get base dataset
    base_dataset = val_dataset.dataset
    
    # Worker function to load one sequence in parallel
    def _load_sequence(seq_idx):
        """Load a single GOP sequence with all frames"""
        sequence_info = base_dataset.sequences[seq_idx]
        sequence_id = f"{sequence_info['video_name']}_gop{sequence_info['gop_index']}"
        
        frames = []
        for frame_idx in range(gop_len):
            global_idx = seq_idx * gop_len + frame_idx
            if global_idx >= len(base_dataset):
                break
            
            try:
                sample = base_dataset[global_idx]
                mv = sample.get('motion_vectors')
                boxes = sample.get('boxes')
                residuals = sample.get('residuals')
                
                # Allow frames with MV or DCT (for ablation variants)
                if boxes is None or (mv is None and residuals is None):
                    continue
                
                frame_data = {
                    'motion_vectors': mv,
                    'boxes': boxes,
                    'frame_id': frame_idx
                }
                if residuals is not None:
                    frame_data['residuals'] = residuals
                
                frames.append(frame_data)
            except Exception as e:
                continue
        
        return seq_idx, sequence_id, frames
    
    # Load GOPs in parallel using ThreadPoolExecutor (IO-bound task)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    max_workers = min(8, os.cpu_count() or 4)
    gop_sequences = {}
    seq_index_to_id = {}
    
    selected_indices = [s for s in val_seq_idx[:args.max_val_gops] if s < total_sequences]
    
    print(f"   🔧 Using {max_workers} parallel workers for I/O")
    
    futures = {}
    seq_bar = tqdm(total=len(selected_indices), desc="Loading GOPs", unit="GOP")
    
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        # Submit all loading tasks
        for seq_idx in selected_indices:
            futures[exe.submit(_load_sequence, seq_idx)] = seq_idx
        
        # Collect results as they complete
        for fut in as_completed(futures):
            seq_idx, sequence_id, frames = fut.result()
            seq_index_to_id[seq_idx] = sequence_id
            
            # Only keep GOPs with at least 2 frames (I-frame + at least 1 P-frame)
            if len(frames) >= 2:
                gop_sequences[sequence_id] = frames
            
            seq_bar.update(1)
            seq_bar.set_postfix({'loaded': len(gop_sequences), 'frames': len(frames)})
    
    seq_bar.close()
    
    # Build ordered list preserving sequence order
    ordered_val_gops = []
    for seq_idx in selected_indices:
        seq_id = seq_index_to_id.get(seq_idx)
        if seq_id and seq_id in gop_sequences:
            ordered_val_gops.append((seq_id, gop_sequences[seq_id]))
    
    print(f"   ✅ Loaded {len(ordered_val_gops)} validation GOPs (cached in RAM)")
    
    # Print GOP statistics
    gop_lengths = [len(frames) for _, frames in ordered_val_gops]
    if gop_lengths:
        print(f"   📊 GOP lengths: min={min(gop_lengths)}, max={max(gop_lengths)}, avg={sum(gop_lengths)/len(gop_lengths):.1f}")

    
    return ordered_val_gops


def benchmark_model(model, model_info, val_gops, device, num_gops, warmup):
    """
    Benchmark model inference speed on validation GOPs (supports all ablation variants).
    
    Args:
        model: Trained model (DCT-MV with variable mv_channels/dct_channels)
        model_info: Model configuration dict
        val_gops: List of (seq_id, gop_frames) tuples
        device: Device (cuda/cpu)
        num_gops: Number of GOPs to benchmark
        warmup: Number of warmup iterations
    
    Returns:
        dict with timing and memory statistics
    """
    print(f"\n⏱️  Benchmarking model inference speed...")
    print(f"   Available GOPs: {len(val_gops)}")
    print(f"   Warmup iterations: {warmup}")
    print(f"   Benchmark iterations: {num_gops}")
    
    model.eval()
    
    # Get model configuration
    mv_channels = model_info.get('mv_channels', 2)
    dct_channels = model_info.get('dct_channels', 64)
    
    print(f"   Model type: MV={mv_channels} ch, DCT={dct_channels} ch")
    
    # Limit to available GOPs
    total_available = len(val_gops)
    num_gops = min(num_gops, total_available - warmup)
    
    if num_gops <= 0:
        raise ValueError(f"Not enough GOPs! Need at least {warmup + 1} GOPs, but only {total_available} available.")
    
    print(f"   ✅ Will benchmark {num_gops} GOPs after {warmup} warmup iterations")
    
    # Get initial memory baseline
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        baseline_memory_mb = 0.0
    
    # Warmup phase
    print(f"\n🔥 Warming up {device.type.upper()}...")
    for i in tqdm(range(min(warmup, total_available)), desc="Warmup"):
        seq_id, gop_frames = val_gops[i]
        
        if len(gop_frames) < 2:
            continue
        
        # Extract I-frame boxes
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        
        # Skip GOPs with no objects (can't track nothing)
        if iframe_boxes.shape[0] == 0:
            continue
        
        # Collect motion vectors and DCT residuals
        mvs = [] if mv_channels > 0 else None
        dcts = [] if dct_channels > 0 else None
        
        for t in range(1, len(gop_frames)):
            # Motion vectors (if model uses them)
            if mv_channels > 0:
                mv = gop_frames[t]['motion_vectors'].to(device)
                
                # Motion vectors are [2, 60, 60, 2] -> take first channel, get [2, 60, 60]
                if mv.ndim == 4 and mv.shape[-1] == 2:
                    mv = mv[..., 0]  # Take only u,v components: [2, 60, 60]
                elif mv.ndim == 4:
                    mv = mv.squeeze(0)  # Remove batch dim if present
                
                mvs.append(mv)
            
            # DCT residuals (if model uses them)
            if dct_channels > 0:
                dct = gop_frames[t].get('residuals', None)
                
                if dct is not None:
                    # DCT should be [H, W, C] = [120, 120, 64]
                    if dct.ndim == 4:
                        dct = dct.squeeze(0)
                    
                    # Slice to requested number of coefficients
                    dct = dct[..., :dct_channels]
                    dcts.append(dct.to(device))
        
        # Stack sequences and add batch dimension (BEFORE timing)
        if mvs:
            mv_seq = torch.stack(mvs)  # [T, 2, 60, 60]
            mv_seq = mv_seq.unsqueeze(1)  # [T, 1, 2, 60, 60]
        else:
            mv_seq = None
            
        if dcts:
            dct_seq = torch.stack(dcts)  # [T, 120, 120, C]
            dct_seq = dct_seq.unsqueeze(1)  # [T, 1, 120, 120, C]
        else:
            dct_seq = None
        
        with torch.no_grad():
            # Call model's forward_single_frame for each frame
            hidden_state = None
            for t in range(len(gop_frames) - 1):  # -1 because frame 0 is I-frame
                # Data already has batch dimension
                mv_frame = mv_seq[t] if mv_seq is not None else None  # [1, 2, 60, 60]
                dct_frame = dct_seq[t] if dct_seq is not None else None  # [1, 120, 120, C]
                boxes_frame = iframe_boxes  # [N, 4]
                
                # Forward single frame (returns 3 values: boxes, confs, hidden_state)
                _, _, hidden_state = model.forward_single_frame(
                    mv_frame, dct_frame, boxes_frame, hidden_state
                )
    
    # Synchronize before benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Get memory after warmup (model + activations loaded)
    if torch.cuda.is_available() and device.type == 'cuda':
        warmup_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        model_memory_overhead_mb = warmup_memory_mb - baseline_memory_mb
    else:
        warmup_memory_mb = 0.0
        model_memory_overhead_mb = 0.0
    
    # Benchmark phase
    print(f"\n⏱️  Running benchmark...")
    
    times = []
    frame_counts = []
    object_counts = []
    peak_memory_mb = []
    
    start_idx = warmup
    end_idx = warmup + num_gops
    
    for i in tqdm(range(start_idx, end_idx), desc="Benchmarking"):
        seq_id, gop_frames = val_gops[i]
        
        if len(gop_frames) < 2:
            continue
        
        # Extract I-frame boxes
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        N = iframe_boxes.shape[0]  # Number of objects
        T = len(gop_frames) - 1  # Number of P-frames
        
        # Skip GOPs with no objects (can't track nothing)
        if N == 0:
            continue
        
        # DEBUG: Check what data is available in first P-frame
        if i == start_idx:
            first_pframe = gop_frames[1]
            has_mv = 'motion_vectors' in first_pframe and first_pframe['motion_vectors'] is not None
            has_dct = 'residuals' in first_pframe and first_pframe['residuals'] is not None
            print(f"\n   🔍 DEBUG - First GOP data availability:")
            print(f"      Motion Vectors: {'✅ Available' if has_mv else '❌ Missing'}")
            print(f"      DCT Residuals: {'✅ Available' if has_dct else '❌ Missing'}")
            if has_mv:
                mv = first_pframe['motion_vectors']
                print(f"      MV shape: {mv.shape}")
                print(f"      MV range: [{mv.min():.2f}, {mv.max():.2f}], mean={mv.mean():.2f}, std={mv.std():.2f}")
            if has_dct:
                dct = first_pframe['residuals']
                print(f"      DCT shape: {dct.shape}")
                print(f"      DCT range: [{dct.min():.2f}, {dct.max():.2f}], mean={dct.mean():.2f}, std={dct.std():.2f}")
            print(f"      Number of objects: {N}")
            print(f"      Number of frames: {T}")
        
        # Collect motion vectors and DCT residuals
        mvs = [] if mv_channels > 0 else None
        dcts = [] if dct_channels > 0 else None
        
        for t in range(1, len(gop_frames)):
            # Motion vectors (if model uses them)
            if mv_channels > 0:
                mv = gop_frames[t]['motion_vectors'].to(device)
                
                # Motion vectors are [2, 60, 60, 2] -> take first channel, get [2, 60, 60]
                if mv.ndim == 4 and mv.shape[-1] == 2:
                    mv = mv[..., 0]  # Take only u,v components: [2, 60, 60]
                elif mv.ndim == 4:
                    mv = mv.squeeze(0)  # Remove batch dim if present
                
                mvs.append(mv)
            
            # DCT residuals (if model uses them)
            if dct_channels > 0:
                dct = gop_frames[t].get('residuals', None)
                
                if dct is not None:
                    # DCT should be [H, W, C] = [120, 120, 64]
                    if dct.ndim == 4:
                        dct = dct.squeeze(0)
                    
                    # Slice to requested number of coefficients
                    dct = dct[..., :dct_channels]
                    dcts.append(dct.to(device))
        
        # DEBUG: Log what data was actually collected
        if i == start_idx:
            print(f"   🔍 DEBUG - Data collected for this model:")
            print(f"      Model config: MV={mv_channels} ch, DCT={dct_channels} ch")
            if mvs is not None:
                print(f"      MV frames collected: {len(mvs)}")
                if len(mvs) > 0:
                    print(f"      MV tensor shape per frame: {mvs[0].shape}")
            if dcts is not None:
                print(f"      DCT frames collected: {len(dcts)}")
                if len(dcts) > 0:
                    print(f"      DCT tensor shape per frame: {dcts[0].shape}")
        
        # Stack sequences and add batch dimension
        if mvs:
            mv_seq = torch.stack(mvs)  # [T, 2, 60, 60]
            # Add batch dimension: [T, 2, 60, 60] -> [T, 1, 2, 60, 60]
            mv_seq = mv_seq.unsqueeze(1)
        else:
            mv_seq = None
            
        if dcts:
            dct_seq = torch.stack(dcts)  # [T, 120, 120, C]
            # Add batch dimension: [T, 120, 120, C] -> [T, 1, 120, 120, C]
            dct_seq = dct_seq.unsqueeze(1)
        else:
            dct_seq = None
        
        # Reset peak memory stats for this GOP
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        # ==================================================================
        # TIMING STARTS HERE - Only measure model inference, not data prep
        # ==================================================================
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Forward pass through complete GOP - use frame-by-frame for proper hidden state handling
        with torch.no_grad():
            hidden_state = None
            for t in range(T):
                # Data already has batch dimension from preparation above
                mv_frame = mv_seq[t] if mv_seq is not None else None  # [1, 2, 60, 60]
                dct_frame = dct_seq[t] if dct_seq is not None else None  # [1, 120, 120, C]
                boxes_frame = iframe_boxes
                
                # Forward single frame (returns 3 values: boxes, confs, hidden_state)
                _, _, hidden_state = model.forward_single_frame(
                    mv_frame, dct_frame, boxes_frame, hidden_state
                )
        
        # End timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        
        # End timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        # ==================================================================
        # TIMING ENDS HERE
        # ==================================================================
        
        # Record peak memory for this GOP
        if torch.cuda.is_available() and device.type == 'cuda':
            peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            peak_memory_mb.append(peak_mb)
        
        # Record statistics
        elapsed = end_time - start_time
        times.append(elapsed)
        frame_counts.append(T)
        object_counts.append(N)
    
    # Calculate statistics
    times = np.array(times)
    frame_counts = np.array(frame_counts)
    object_counts = np.array(object_counts)
    
    total_frames = frame_counts.sum()
    total_time = times.sum()
    
    # Get final memory usage
    if torch.cuda.is_available() and device.type == 'cuda':
        final_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        peak_memory_max_mb = max(peak_memory_mb) if peak_memory_mb else 0.0
    else:
        final_memory_mb = 0.0
        peak_memory_max_mb = 0.0
    
    stats = {
        'gop_time_mean_ms': float(times.mean() * 1000),
        'gop_time_std_ms': float(times.std() * 1000),
        'gop_time_min_ms': float(times.min() * 1000),
        'gop_time_max_ms': float(times.max() * 1000),
        'gop_time_median_ms': float(np.median(times) * 1000),
        'frame_time_mean_ms': float((total_time / total_frames) * 1000),
        'fps': float(total_frames / total_time),
        'total_gops': int(num_gops),
        'total_frames': int(total_frames),
        'avg_objects_per_gop': float(object_counts.mean()),
        'avg_frames_per_gop': float(frame_counts.mean()),
        'min_objects': int(object_counts.min()),
        'max_objects': int(object_counts.max()),
        # Memory statistics
        'model_size_mb': model_info.get('model_size_mb', 0.0),
        'baseline_memory_mb': round(baseline_memory_mb, 2),
        'warmup_memory_mb': round(warmup_memory_mb, 2),
        'model_overhead_mb': round(model_memory_overhead_mb, 2),
        'final_memory_mb': round(final_memory_mb, 2),
        'peak_memory_mb': round(peak_memory_max_mb, 2),
    }
    
    return stats


def benchmark_static_baseline(val_gops, device, num_gops, warmup):
    """
    Benchmark static baseline (copy I-frame boxes to all P-frames).
    
    This represents the simplest tracking approach with no computation.
    """
    print(f"\n⏱️  Benchmarking Static Baseline (I-frame copy)...")
    
    total_available = len(val_gops)
    num_gops = min(num_gops, total_available - warmup)
    
    # Warmup (minimal for baseline)
    for i in range(min(warmup, total_available)):
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) < 2:
            continue
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        _ = [iframe_boxes.clone() for _ in range(len(gop_frames) - 1)]
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    frame_counts = []
    object_counts = []
    
    start_idx = warmup
    end_idx = warmup + num_gops
    
    for i in tqdm(range(start_idx, end_idx), desc="Static Baseline"):
        seq_id, gop_frames = val_gops[i]
        
        if len(gop_frames) < 2:
            continue
        
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        N = iframe_boxes.shape[0]
        T = len(gop_frames) - 1
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        # Simply copy I-frame boxes to all frames
        predictions = [iframe_boxes.clone() for _ in range(T)]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        frame_counts.append(T)
        object_counts.append(N)
    
    # Calculate statistics
    times = np.array(times)
    frame_counts = np.array(frame_counts)
    
    total_frames = frame_counts.sum()
    total_time = times.sum()
    
    return {
        'gop_time_mean_ms': float(times.mean() * 1000),
        'gop_time_std_ms': float(times.std() * 1000),
        'gop_time_min_ms': float(times.min() * 1000),
        'gop_time_max_ms': float(times.max() * 1000),
        'frame_time_mean_ms': float((total_time / total_frames) * 1000),
        'fps': float(total_frames / total_time),
        'total_gops': int(num_gops),
        'total_frames': int(total_frames),
    }


def print_results(results):
    """Print benchmark results in a formatted table."""
    print("\n" + "="*90)
    print("MODEL SPEED BENCHMARK RESULTS")
    print("="*90)
    
    model_stats = results['model']
    model_info = results['model_info']
    
    print(f"\n📊 Model Information:")
    print(f"   Architecture:      {model_info['model_type']}")
    print(f"   Total Parameters:  {model_info['total_params']:,}")
    print(f"   Trainable Params:  {model_info['trainable_params']:,}")
    print(f"   Model Size:        {model_info['model_size_mb']:.2f} MB (FP32)")
    print(f"   MV Channels:       {model_info['mv_channels']}")
    print(f"   DCT Channels:      {model_info['dct_channels']}")
    if model_info.get('feature_dim'):
        print(f"   Feature Dim:       {model_info['feature_dim']}")
        print(f"   Hidden Dim:        {model_info['hidden_dim']}")
        print(f"   Image Size:        {model_info['image_size']}x{model_info['image_size']}")
        print(f"   Multi-scale ROI:   {model_info['use_multiscale_roi']}")
        print(f"   Parallel Heads:    {model_info['use_parallel_heads']}")
        print(f"   Attention:         {model_info['use_attention']}")
    
    print(f"\n⚡ Device Information:")
    print(f"   Device:            {results['device']}")
    print(f"   Resolution:        {results['resolution']}x{results['resolution']}")
    
    print(f"\n⏱️  Inference Speed:")
    print(f"   GOP Time (avg):    {model_stats['gop_time_mean_ms']:>8.2f} ± {model_stats['gop_time_std_ms']:>6.2f} ms")
    print(f"   GOP Time (min):    {model_stats['gop_time_min_ms']:>8.2f} ms")
    print(f"   GOP Time (max):    {model_stats['gop_time_max_ms']:>8.2f} ms")
    print(f"   GOP Time (median): {model_stats['gop_time_median_ms']:>8.2f} ms")
    print(f"   Frame Time (avg):  {model_stats['frame_time_mean_ms']:>8.2f} ms/frame")
    print(f"   FPS:               {model_stats['fps']:>8.1f} frames/sec")
    
    print(f"\n💾 Memory Usage:")
    print(f"   Model Size:        {model_stats['model_size_mb']:>8.2f} MB")
    if results['device'] == 'cuda' or results['device'].startswith('cuda'):
        print(f"   Baseline VRAM:     {model_stats['baseline_memory_mb']:>8.2f} MB")
        print(f"   After Warmup:      {model_stats['warmup_memory_mb']:>8.2f} MB")
        print(f"   Model Overhead:    {model_stats['model_overhead_mb']:>8.2f} MB")
        print(f"   Peak VRAM:         {model_stats['peak_memory_mb']:>8.2f} MB")
    else:
        print(f"   (CPU mode - VRAM metrics not available)")
    
    print(f"\n📈 Benchmark Statistics:")
    print(f"   Total GOPs:        {model_stats['total_gops']}")
    print(f"   Total Frames:      {model_stats['total_frames']}")
    print(f"   Avg Frames/GOP:    {model_stats['avg_frames_per_gop']:.1f}")
    print(f"   Avg Objects/GOP:   {model_stats['avg_objects_per_gop']:.1f}")
    print(f"   Min Objects:       {model_stats['min_objects']}")
    print(f"   Max Objects:       {model_stats['max_objects']}")
    
    # Compare with baseline if available
    if 'static_baseline' in results:
        baseline_stats = results['static_baseline']
        speedup = baseline_stats['fps'] / model_stats['fps']
        overhead = model_stats['frame_time_mean_ms'] - baseline_stats['frame_time_mean_ms']
        
        print(f"\n📊 Static Baseline Comparison (I-frame copy):")
        print(f"   Baseline GOP Time:  {baseline_stats['gop_time_mean_ms']:>8.2f} ± {baseline_stats['gop_time_std_ms']:>6.2f} ms")
        print(f"   Baseline Frame Time:{baseline_stats['frame_time_mean_ms']:>8.2f} ms/frame")
        print(f"   Baseline FPS:       {baseline_stats['fps']:>8.1f} frames/sec")
        print(f"   Model Slowdown:     {speedup:>8.2f}x slower than baseline")
        print(f"   Model Overhead:     {overhead:>8.2f} ms/frame")
    
    print("\n" + "="*90)


def get_all_variant_checkpoints(results_dir):
    """
    Discover all ablation variant checkpoints in results directory.
    
    Returns:
        List of (variant_name, checkpoint_path) tuples
    """
    results_dir = Path(results_dir)
    
    variants = [
        ('mv_only', 'MV-only', 2, 0),
        ('dct_8', 'DCT-8', 0, 8),
        ('dct_16', 'DCT-16', 0, 16),
        ('dct_32', 'DCT-32', 0, 32),
        ('dct_64', 'DCT-64', 0, 64),
        ('mv_dct_8', 'MV+DCT-8', 2, 8),
        ('mv_dct_16', 'MV+DCT-16', 2, 16),
        ('mv_dct_32', 'MV+DCT-32', 2, 32),
        ('mv_dct_64_baseline', 'MV+DCT-64', 2, 64),
    ]
    
    checkpoints = []
    for dir_name, display_name, mv_ch, dct_ch in variants:
        # Try best_model.pt first, then checkpoint_epoch_*.pt
        variant_dir = results_dir / dir_name
        best_model = variant_dir / 'best_model.pt'
        
        if best_model.exists():
            checkpoints.append((display_name, str(best_model), mv_ch, dct_ch))
        else:
            # Try to find latest checkpoint
            checkpoint_files = list(variant_dir.glob('checkpoint_epoch_*.pt'))
            if checkpoint_files:
                # Sort by epoch number
                latest = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))[-1]
                checkpoints.append((display_name, str(latest), mv_ch, dct_ch))
    
    return checkpoints


def benchmark_all_variants(args, device, val_gops):
    """
    Benchmark all 9 ablation variants and return results.
    
    Returns:
        List of (variant_name, results_dict) tuples
    """
    checkpoints = get_all_variant_checkpoints(args.results_dir)
    
    if not checkpoints:
        raise RuntimeError(f"❌ No model checkpoints found in {args.results_dir}!")
    
    print(f"\n🔍 Found {len(checkpoints)} variant checkpoints:")
    for name, path, mv, dct in checkpoints:
        print(f"   • {name:15s} (MV={mv}, DCT={dct:2d}): {Path(path).name}")
    
    all_results = []
    
    for i, (variant_name, checkpoint_path, mv_ch, dct_ch) in enumerate(checkpoints, 1):
        print(f"\n{'='*90}")
        print(f"BENCHMARKING VARIANT {i}/{len(checkpoints)}: {variant_name}")
        print(f"{'='*90}")
        
        try:
            # Load model with architecture flags from args
            model, model_info = load_model(
                checkpoint_path, device,
                use_fast=args.use_fast,
                use_ultra_fast=args.use_ultra_fast
            )
            
            # Clear GPU memory before benchmarking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Benchmark model
            model_stats = benchmark_model(
                model, model_info, val_gops, device,
                args.num_gops, args.warmup
            )
            
            # Store results
            results = {
                'variant_name': variant_name,
                'model': model_stats,
                'model_info': model_info,
                'device': str(device),
                'resolution': args.resolution,
                'checkpoint': checkpoint_path,
            }
            
            all_results.append(results)
            
            # Free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n✅ {variant_name} benchmark complete!")
            print(f"   FPS: {model_stats['fps']:.1f}, Frame time: {model_stats['frame_time_mean_ms']:.2f} ms")
            print(f"   Model size: {model_stats['model_size_mb']:.2f} MB")
            if 'peak_memory_mb' in model_stats:
                print(f"   Peak VRAM: {model_stats['peak_memory_mb']:.2f} MB")
        
        except Exception as e:
            print(f"\n❌ Failed to benchmark {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def export_latex_speed_table(all_results, output_file, device_name='GPU', rtdetr_baseline=None):
    """
    Export speed comparison table to LaTeX format with GPU capacity analysis.
    Creates two tables: standalone deployment and GOP-based deployment.
    
    Args:
        all_results: List of results dicts from benchmark_all_variants
        output_file: Path to output .tex file
        device_name: 'GPU' or 'CPU'
        rtdetr_baseline: Optional dict with RT-DETR baseline stats:
                        {'model_size_mb': float, 'peak_memory_mb': float, 'fps': float, 'map': float}
    """
    print(f"\n📄 Exporting LaTeX speed tables to: {output_file}")
    
    GPU_TOTAL_MEMORY_GB = 16.0  # 16GB GPU
    GPU_TOTAL_MEMORY_MB = GPU_TOTAL_MEMORY_GB * 1024  # 16384 MB
    GPU_RESERVED_MB = 1024  # Reserve 1GB for system/overhead
    GPU_AVAILABLE_MB = GPU_TOTAL_MEMORY_MB - GPU_RESERVED_MB  # 15360 MB available
    
    # Get RT-DETR VRAM for GOP calculation
    rtdetr_total_vram = 0.0
    rtdetr_fps = 0.0
    if rtdetr_baseline:
        rtdetr_model_size = rtdetr_baseline.get('model_size_mb', 0.0)
        rtdetr_peak_vram = rtdetr_baseline.get('peak_memory_mb', 0.0)
        rtdetr_total_vram = rtdetr_model_size + rtdetr_peak_vram
        rtdetr_fps = rtdetr_baseline.get('fps', 0.0)
    
    with open(output_file, 'w') as f:
        # ============================================
        # TABLE 1: Standalone Deployment (no GOP)
        # ============================================
        f.write("% Table 1: Standalone Deployment Analysis\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Standalone Deployment: Compressed Domain vs RGB Baseline (16GB GPU)}\n")
        f.write("\\label{tab:speed_standalone}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Variant & Total VRAM & Frame Time & \\multicolumn{3}{c}{16GB GPU Capacity} \\\\\n")
        f.write(" & (MB) & (ms) & \\# Instances & Total FPS & 30 FPS Cameras \\\\\n")
        f.write("\\midrule\n")
        
        # Add RT-DETR baseline
        if rtdetr_baseline:
            frame_time = 1000.0 / rtdetr_fps if rtdetr_fps > 0 else 0.0
            
            if rtdetr_total_vram > 0:
                num_instances = int(GPU_AVAILABLE_MB / rtdetr_total_vram)
                total_fps = num_instances * rtdetr_fps if num_instances > 0 else 0.0
            else:
                num_instances = 0
                total_fps = 0.0
            
            num_cameras = int(total_fps / 30.0) if total_fps > 0 else 0
            
            f.write(f"\\textbf{{RT-DETR (RGB)}} & {rtdetr_total_vram:.2f} & {frame_time:.2f} & {num_instances} & {total_fps:.1f} & {num_cameras} \\\\\n")
            f.write("\\midrule\n")
        
        # Compressed domain variants
        for result in all_results:
            variant_name = result['variant_name']
            model_stats = result['model']
            
            model_size_mb = model_stats['model_size_mb']
            peak_vram_mb = model_stats.get('peak_memory_mb', 0.0)
            total_vram_mb = model_size_mb + peak_vram_mb
            fps = model_stats['fps']
            frame_time_ms = model_stats['frame_time_mean_ms']
            
            if total_vram_mb > 0:
                num_instances = int(GPU_AVAILABLE_MB / total_vram_mb)
                total_fps = num_instances * fps if num_instances > 0 else 0.0
            else:
                num_instances = 0
                total_fps = 0.0
            
            num_cameras = int(total_fps / 30.0) if total_fps > 0 else 0
            variant_name_latex = variant_name.replace('+', '$+$')
            
            f.write(f"{variant_name_latex} & {total_vram_mb:.2f} & {frame_time_ms:.2f} & {num_instances} & {total_fps:.1f} & {num_cameras} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n\n")
        
        # ============================================
        # TABLE 2: GOP-Based Deployment
        # ============================================
        f.write("% Table 2: GOP-Based Deployment Analysis\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{GOP-Based Deployment: RT-DETR (I-frame) + Compressed Domain (P-frames) on 16GB GPU}\n")
        f.write("\\label{tab:speed_gop}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Variant & GOP-6 (1I+5P) & GOP-12 (1I+11P) & GOP-50 (1I+49P) \\\\\n")
        f.write(" & 30 FPS Cameras & 30 FPS Cameras & 30 FPS Cameras \\\\\n")
        f.write("\\midrule\n")
        
        # Add RT-DETR baseline (same for GOP deployment)
        if rtdetr_baseline:
            if rtdetr_total_vram > 0:
                num_instances = int(GPU_AVAILABLE_MB / rtdetr_total_vram)
                total_fps = num_instances * rtdetr_fps if num_instances > 0 else 0.0
                num_cameras = int(total_fps / 30.0) if total_fps > 0 else 0
            else:
                num_instances = 0
                num_cameras = 0
            
            f.write(f"\\textbf{{RT-DETR (RGB)}} & {num_cameras} & {num_cameras} & {num_cameras} \\\\\n")
            f.write("\\midrule\n")
        
        # Compressed domain variants with GOP-6, GOP-12, and GOP-50
        for result in all_results:
            variant_name = result['variant_name']
            model_stats = result['model']
            
            model_size_mb = model_stats['model_size_mb']
            peak_vram_mb = model_stats.get('peak_memory_mb', 0.0)
            total_vram_mb = model_size_mb + peak_vram_mb
            compressed_fps = model_stats['fps']
            
            # GOP Total VRAM = RT-DETR + Compressed
            gop_total_vram = rtdetr_total_vram + total_vram_mb
            
            # GOP-6: 1 I-frame + 5 P-frames
            if gop_total_vram > 0 and rtdetr_fps > 0 and compressed_fps > 0:
                gop6_instances = int(GPU_AVAILABLE_MB / gop_total_vram)
                gop6_avg_fps = 6.0 / (1.0/rtdetr_fps + 5.0/compressed_fps)
                gop6_total_fps = gop6_instances * gop6_avg_fps
                gop6_cameras = int(gop6_total_fps / 30.0)
            else:
                gop6_instances = 0
                gop6_cameras = 0
            
            # GOP-12: 1 I-frame + 11 P-frames
            if gop_total_vram > 0 and rtdetr_fps > 0 and compressed_fps > 0:
                gop12_instances = int(GPU_AVAILABLE_MB / gop_total_vram)
                gop12_avg_fps = 12.0 / (1.0/rtdetr_fps + 11.0/compressed_fps)
                gop12_total_fps = gop12_instances * gop12_avg_fps
                gop12_cameras = int(gop12_total_fps / 30.0)
            else:
                gop12_instances = 0
                gop12_cameras = 0
            
            # GOP-50: 1 I-frame + 49 P-frames
            if gop_total_vram > 0 and rtdetr_fps > 0 and compressed_fps > 0:
                gop50_instances = int(GPU_AVAILABLE_MB / gop_total_vram)
                gop50_avg_fps = 50.0 / (1.0/rtdetr_fps + 49.0/compressed_fps)
                gop50_total_fps = gop50_instances * gop50_avg_fps
                gop50_cameras = int(gop50_total_fps / 30.0)
            else:
                gop50_instances = 0
                gop50_cameras = 0
            
            variant_name_latex = variant_name.replace('+', '$+$')
            
            f.write(f"{variant_name_latex} & {gop6_cameras} & {gop12_cameras} & {gop50_cameras} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    
    print(f"   ✅ LaTeX tables exported successfully")
    print(f"   📊 Two tables created:")
    print(f"      - Table 1: Standalone deployment (no GOP)")
    print(f"      - Table 2: GOP-based deployment (GOP-6 and GOP-12)")
    print(f"   📊 GPU Capacity Analysis (16GB GPU):")
    print(f"      - Total memory: {GPU_TOTAL_MEMORY_GB:.0f} GB ({GPU_TOTAL_MEMORY_MB:.0f} MB)")
    print(f"      - Reserved for system: {GPU_RESERVED_MB:.0f} MB")
    print(f"      - Available for models: {GPU_AVAILABLE_MB:.0f} MB")
    
    if rtdetr_baseline:
        rtdetr_instances = int(GPU_AVAILABLE_MB / rtdetr_total_vram) if rtdetr_total_vram > 0 else 0
        print(f"      - RT-DETR instances: {rtdetr_instances}")
    
    if all_results:
        best_capacity = max(all_results, key=lambda x: int(GPU_AVAILABLE_MB / (x['model']['model_size_mb'] + x['model'].get('peak_memory_mb', 0.0))))
        best_vram = best_capacity['model']['model_size_mb'] + best_capacity['model'].get('peak_memory_mb', 0.0)
        best_instances = int(GPU_AVAILABLE_MB / best_vram)
        print(f"      - Best compressed (capacity): {best_capacity['variant_name']} ({best_instances} instances)")



def print_comparison_table(all_results, device_name):
    """
    Print comparison table for all ablation variants.
    
    Args:
        all_results: List of results dicts from benchmark_all_variants
        device_name: 'CPU' or 'GPU'
    """
    print(f"\n{'='*120}")
    print(f"ABLATION STUDY SPEED COMPARISON ({device_name})")
    print(f"{'='*120}")
    
    # Load accuracy results if available
    accuracy_map = {}
    for result in all_results:
        variant_name = result['variant_name']
        checkpoint_path = Path(result['checkpoint'])
        variant_dir = checkpoint_path.parent
        
        # Try to load validation mAP from training_results.json
        results_json = variant_dir / 'training_results.json'
        if results_json.exists():
            with open(results_json, 'r') as f:
                training_data = json.load(f)
                accuracy_map[variant_name] = training_data.get('best_val_map', None)
    
    # Print table header
    if device_name == 'GPU':
        print("┌───────────────┬──────┬──────┬──────────┬──────────┬───────────┬──────────┬────────────┬───────────────┐")
        print("│ Variant       │  MV  │ DCT  │  Model   │   Val    │   Frame   │   FPS    │ Peak VRAM  │  Model+Data   │")
        print("│               │  Ch  │  Ch  │ Size MB  │  mAP@50  │  Time ms  │          │     MB     │  VRAM (MB)    │")
        print("├───────────────┼──────┼──────┼──────────┼──────────┼───────────┼──────────┼────────────┼───────────────┤")
    else:
        print("┌───────────────┬──────┬──────┬──────────┬──────────┬───────────┬──────────┬──────────────┐")
        print("│ Variant       │  MV  │ DCT  │  Model   │   Val    │   Frame   │   FPS    │   Model      │")
        print("│               │  Ch  │  Ch  │ Size MB  │  mAP@50  │  Time ms  │          │   Size MB    │")
        print("├───────────────┼──────┼──────┼──────────┼──────────┼───────────┼──────────┼──────────────┤")
    
    # Print each variant
    for result in all_results:
        variant_name = result['variant_name']
        model_info = result['model_info']
        model_stats = result['model']
        
        mv_ch = model_info['mv_channels']
        dct_ch = model_info['dct_channels']
        model_size_mb = model_stats['model_size_mb']
        val_map = accuracy_map.get(variant_name, None)
        frame_time_ms = model_stats['frame_time_mean_ms']
        fps = model_stats['fps']
        
        val_map_str = f"{val_map:.4f}" if val_map is not None else "  N/A   "
        
        if device_name == 'GPU' and 'peak_memory_mb' in model_stats:
            peak_vram = model_stats['peak_memory_mb']
            total_vram = model_size_mb + peak_vram
            print(f"│ {variant_name:13s} │  {mv_ch:2d}  │  {dct_ch:2d}  │  {model_size_mb:6.2f}  │ {val_map_str} │  {frame_time_ms:7.2f}  │ {fps:7.1f}  │  {peak_vram:8.2f}  │   {total_vram:10.2f}  │")
        else:
            print(f"│ {variant_name:13s} │  {mv_ch:2d}  │  {dct_ch:2d}  │  {model_size_mb:6.2f}  │ {val_map_str} │  {frame_time_ms:7.2f}  │ {fps:7.1f}  │   {model_size_mb:9.2f}  │")
    
    if device_name == 'GPU':
        print("└───────────────┴──────┴──────┴──────────┴──────────┴───────────┴──────────┴────────────┴───────────────┘")
    else:
        print("└───────────────┴──────┴──────┴──────────┴──────────┴───────────┴──────────┴──────────────┘")
    
    print()
    
    # Find best variants
    if all_results:
        fastest = min(all_results, key=lambda x: x['model']['frame_time_mean_ms'])
        smallest = min(all_results, key=lambda x: x['model']['model_size_mb'])
        
        print("🏆 BEST VARIANTS:")
        print(f"   Fastest:         {fastest['variant_name']} ({fastest['model']['fps']:.1f} FPS, {fastest['model']['frame_time_mean_ms']:.2f} ms/frame)")
        print(f"   Smallest Model:  {smallest['variant_name']} ({smallest['model']['model_size_mb']:.2f} MB)")
        
        if accuracy_map:
            best_accuracy_name = max(accuracy_map.keys(), key=lambda k: accuracy_map[k] if accuracy_map[k] is not None else 0.0)
            best_accuracy_map = accuracy_map[best_accuracy_name]
            if best_accuracy_map is not None:
                print(f"   Best Accuracy:   {best_accuracy_name} (mAP@50 = {best_accuracy_map:.4f})")
        
        if device_name == 'GPU':
            min_vram = min(all_results, key=lambda x: x['model'].get('peak_memory_mb', float('inf')))
            print(f"   Lowest VRAM:     {min_vram['variant_name']} ({min_vram['model']['peak_memory_mb']:.2f} MB peak)")
    
    print()
    
    # Efficiency metrics
    print("📊 EFFICIENCY METRICS:")
    
    # Speed comparison
    if len(all_results) >= 2:
        mv_only = next((r for r in all_results if r['variant_name'] == 'MV-only'), None)
        dct_64 = next((r for r in all_results if r['variant_name'] == 'DCT-64'), None)
        mv_dct_64 = next((r for r in all_results if r['variant_name'] == 'MV+DCT-64'), None)
        
        if mv_only and dct_64:
            speedup = dct_64['model']['frame_time_mean_ms'] / mv_only['model']['frame_time_mean_ms']
            print(f"   • DCT-64 vs MV-only speed: {speedup:.2f}x slower")
        
        if mv_only and mv_dct_64:
            speedup = mv_dct_64['model']['frame_time_mean_ms'] / mv_only['model']['frame_time_mean_ms']
            print(f"   • MV+DCT-64 vs MV-only speed: {speedup:.2f}x slower")
        
        if dct_64 and mv_dct_64:
            speedup = mv_dct_64['model']['frame_time_mean_ms'] / dct_64['model']['frame_time_mean_ms']
            print(f"   • MV+DCT-64 vs DCT-64 speed: {speedup:.2f}x {'slower' if speedup > 1 else 'faster'}")
    
    # mAP per MB (efficiency metric)
    if accuracy_map:
        print()
        print("   mAP per MB (higher = better efficiency):")
        efficiency_list = []
        for result in all_results:
            variant_name = result['variant_name']
            if variant_name in accuracy_map and accuracy_map[variant_name] is not None:
                map_val = accuracy_map[variant_name]
                size_mb = result['model']['model_size_mb']
                efficiency = map_val / size_mb if size_mb > 0 else 0.0
                efficiency_list.append((variant_name, efficiency, map_val, size_mb))
        
        # Sort by efficiency
        efficiency_list.sort(key=lambda x: x[1], reverse=True)
        
        for variant_name, efficiency, map_val, size_mb in efficiency_list[:5]:
            print(f"     - {variant_name:15s}: {efficiency:.6f} (mAP={map_val:.4f}, size={size_mb:.2f}MB)")
    
    print()


def save_all_results(all_results, output_dir, device_name):
    """Save all benchmark results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save individual results
    for result in all_results:
        variant_name = result['variant_name']
        safe_name = variant_name.replace('+', '_').replace('-', '_').lower()
        output_file = output_dir / f"{safe_name}_{device_name.lower()}_speed.json"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"   💾 Saved {variant_name} results to {output_file}")
    
    # Save combined comparison
    combined_file = output_dir / f"all_variants_{device_name.lower()}_comparison.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"   💾 Saved combined results to {combined_file}")


def main():
    args = parse_arguments()
    
    print(f"🚀 Model Speed Benchmark for Ablation Study")
    print(f"="*90)
    
    # Device setup
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    device_name = 'GPU' if device.type == 'cuda' else 'CPU'
    print(f"🖥️  Device: {device} ({device_name})")
    
    # Load validation dataset (once for all variants)
    val_gops = load_validation_gops(args)
    
    # Benchmark all variants or single model
    if args.benchmark_all or args.model_path is None:
        print(f"\n{'='*90}")
        print(f"BENCHMARKING ALL ABLATION VARIANTS")
        print(f"{'='*90}")
        
        all_results = benchmark_all_variants(args, device, val_gops)
        
        if not all_results:
            print("\n❌ No variants were successfully benchmarked!")
            return 1
        
        # Print comparison table
        print_comparison_table(all_results, device_name)
        
        # Save all results
        output_dir = Path(args.results_dir) / 'speed_benchmarks'
        print(f"\n💾 Saving results to {output_dir}...")
        save_all_results(all_results, output_dir, device_name)
        
        # Export LaTeX table if requested
        if args.export_latex:
            latex_output = output_dir / 'speed_comparison_table.tex'
            
            # Prepare RT-DETR baseline if provided
            rtdetr_baseline = None
            if args.rtdetr_model_size > 0 and args.rtdetr_fps > 0:
                rtdetr_baseline = {
                    'model_size_mb': args.rtdetr_model_size,
                    'peak_memory_mb': args.rtdetr_peak_vram,
                    'fps': args.rtdetr_fps,
                    'map': args.rtdetr_map
                }
                print(f"\n📊 RT-DETR Baseline Configuration:")
                print(f"   Model Size: {args.rtdetr_model_size:.2f} MB")
                print(f"   Peak VRAM: {args.rtdetr_peak_vram:.2f} MB")
                print(f"   FPS: {args.rtdetr_fps:.1f}")
                print(f"   mAP@0.5: {args.rtdetr_map:.4f}")
            
            export_latex_speed_table(all_results, latex_output, device_name, rtdetr_baseline)
            print(f"   📄 LaTeX table saved to: {latex_output}")
        
        print(f"\n✅ All variants benchmarked successfully!")
        print(f"   Results saved to: {output_dir}")
    
    else:
        # Single model benchmark
        print(f"\n{'='*90}")
        print(f"BENCHMARKING SINGLE MODEL")
        print(f"{'='*90}")
        
        # Load model with architecture flags
        model, model_info = load_model(
            args.model_path, device,
            use_fast=args.use_fast,
            use_ultra_fast=args.use_ultra_fast
        )
        
        # Clear GPU memory before benchmarking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark model
        model_stats = benchmark_model(
            model, model_info, val_gops, device,
            args.num_gops, args.warmup
        )
        
        # Get memory usage after model benchmark
        memory_usage = get_memory_usage()
        
        # Prepare results
        results = {
            'model': model_stats,
            'model_info': model_info,
            'memory_usage': memory_usage,
            'device': str(device),
            'resolution': args.resolution,
            'checkpoint': args.model_path,
        }
        
        # Benchmark baseline if requested
        if args.compare_baselines:
            print(f"\n{'='*90}")
            print(f"BENCHMARKING BASELINE")
            print(f"{'='*90}")
            
            baseline_stats = benchmark_static_baseline(
                val_gops, device,
                args.num_gops, args.warmup
            )
            results['static_baseline'] = baseline_stats
        
        # Print formatted results
        print_results(results)
        
        # Save results to JSON
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        print("\n✅ Speed benchmark complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
