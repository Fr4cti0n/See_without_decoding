#!/usr/bin/env python3
"""
Fair Comparison: DCT-MV Model vs RT-DETR Baseline

Compares your compressed video tracking model (DCT-MV) with RT-DETR baseline
on the same validation GOPs using:
- DCT-MV: Uses motion vectors + DCT residuals (compressed domain)
- RT-DETR: Uses RGB I-frames and P-frames (pixel domain, Ultralytics)

Both models process the same sequences and are benchmarked for:
- Inference speed (FPS, frame time, GOP time)
- Memory usage (GPU/CPU)
- Model size (parameters, memory footprint)

Author: GitHub Copilot
Date: October 2024
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

# Import dataset factory
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
    parser = argparse.ArgumentParser(description="Fair Comparison: DCT-MV vs RT-DETR")
    
    # Model paths
    parser.add_argument("--dct-mv-model", type=str, required=True,
                       help="Path to trained DCT-MV model checkpoint (.pt file)")
    parser.add_argument("--rtdetr-model", type=str, default="rtdetr-l.pt",
                       help="RT-DETR model name or path (default: rtdetr-l.pt)")
    
    # Dataset parameters
    parser.add_argument("--resolution", type=int, default=960, choices=[640, 960],
                       help="Video resolution (default: 960)")
    parser.add_argument("--num-gops", type=int, default=10,
                       help="Number of validation GOPs to benchmark (default: 10)")
    parser.add_argument("--max-val-gops", type=int, default=20,
                       help="Maximum validation GOPs to load (default: 20)")
    parser.add_argument("--warmup", type=int, default=3,
                       help="Number of warmup iterations (default: 3)")
    
    # Training split parameters
    parser.add_argument("--train-split", type=float, default=0.5,
                       help="Training split fraction (default: 0.5)")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split fraction (default: 0.2)")
    parser.add_argument("--gop-length", type=int, default=48,
                       help="GOP length in frames (default: 48)")
    
    # Device and output
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu), auto-detect if not specified")
    parser.add_argument("--output", type=str, default="comparison_results.json",
                       help="Output JSON file for results")
    
    return parser.parse_args()


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


def load_dct_mv_model(model_path, device):
    """Load trained DCT-MV model from checkpoint."""
    print(f"\n📦 Loading DCT-MV model from {model_path}...")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    args_dict = checkpoint.get('args', {})
    
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Import DCT-MV model
    from mots_exp.models.dct_mv_center import DCTMVCenterTracker
    
    # Extract model configuration
    num_dct_coeffs = args_dict.get('num_dct_coeffs', 16)
    feature_dim = args_dict.get('feature_dim', 64)
    hidden_dim = args_dict.get('hidden_dim', 128)
    image_size = args_dict.get('resolution', 960)
    
    # Create model
    model = DCTMVCenterTracker(
        num_dct_coeffs=num_dct_coeffs,
        mv_feature_dim=32,
        dct_feature_dim=32,
        fused_feature_dim=feature_dim,
        roi_feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        lstm_layers=2,
        dropout=0.1,
        image_size=image_size,
        use_multiscale_roi=args_dict.get('use_multiscale_roi', False),
        roi_sizes=args_dict.get('roi_sizes', [3, 7, 11]),
        use_parallel_heads=args_dict.get('use_parallel_heads', False),
        use_attention=args_dict.get('use_attention', False),
        attention_heads=args_dict.get('attention_heads', 4)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✅ DCT-MV model loaded successfully")
    print(f"   Parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.1f} MB)")
    
    return model, total_params


def load_rtdetr_model(model_name, device):
    """Load RT-DETR model from Ultralytics."""
    print(f"\n📦 Loading RT-DETR model: {model_name}...")
    
    try:
        from ultralytics import RTDETR
        
        # Load model
        model = RTDETR(model_name)
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        
        print(f"✅ RT-DETR model loaded successfully")
        print(f"   Parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.1f} MB)")
        
        return model, total_params
        
    except Exception as e:
        print(f"❌ Failed to load RT-DETR model: {e}")
        return None, 0


def load_validation_gops_compressed(args):
    """
    Load validation GOPs with ONLY compressed data (MV + DCT) for DCT-MV benchmark.
    Does NOT load RGB frames to avoid wasting memory.
    
    Returns:
        List of (seq_id, gop_frames) tuples with compressed data in RAM
    """
    print(f"\n📚 Loading validation dataset with COMPRESSED DATA ONLY (no RGB)...")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   Max validation GOPs: {args.max_val_gops}")
    
    if not HAS_REAL_DATASET or create_mots_dataset is None:
        raise RuntimeError("❌ Real MOTS dataset factory not available!")
    
    # Create dataset with ONLY compressed data (no RGB!)
    dataset = create_mots_dataset(
        dataset_type=['mot15', 'mot17', 'mot20'],
        resolution=args.resolution,
        mode="train",
        load_iframe=False,  # ❌ NO RGB I-frames
        load_pframe=False,  # ❌ NO RGB P-frames
        load_motion_vectors=True,  # ✅ MV for DCT-MV
        load_residuals=True,  # ✅ DCT residuals for DCT-MV
        load_annotations=True,
        sequence_length=48,
        data_format="separate",
        combine_datasets=True
    )


def load_validation_gops_rgb(args):
    """
    Load validation GOPs with ONLY RGB frames for RT-DETR benchmark.
    Does NOT load compressed data to avoid wasting memory.
    
    Returns:
        List of (seq_id, gop_frames) tuples with RGB frames in RAM
    """
    print(f"\n📚 Loading validation dataset with RGB FRAMES ONLY (no compressed data)...")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   Max validation GOPs: {args.max_val_gops}")
    
    if not HAS_REAL_DATASET or create_mots_dataset is None:
        raise RuntimeError("❌ Real MOTS dataset factory not available!")
    
    # Create dataset with ONLY RGB frames (no compressed data!)
    dataset = create_mots_dataset(
        dataset_type=['mot15', 'mot17', 'mot20'],
        resolution=args.resolution,
        mode="train",
        load_iframe=True,  # ✅ RGB I-frames for RT-DETR
        load_pframe=True,  # ✅ RGB P-frames for RT-DETR
        load_motion_vectors=False,  # ❌ NO MV
        load_residuals=False,  # ❌ NO DCT residuals
        load_annotations=True,
        sequence_length=48,
        data_format="separate",
        combine_datasets=True
    )
    
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("❌ Dataset is empty or failed to load!")
    
    print(f"   ✅ Dataset loaded: {len(dataset)} total samples")
    
    # Create validation split
    import random
    
    total_sequences = len(dataset.sequences)
    seq_indices = list(range(total_sequences))
    random.seed(42)
    random.shuffle(seq_indices)
    
    split_train = int(total_sequences * args.train_split)
    split_val = split_train + int(total_sequences * args.val_split)
    val_seq_idx = seq_indices[split_train:split_val]
    
    print(f"   📊 Total sequences: {total_sequences}")
    print(f"   🎯 Validation sequences: {len(val_seq_idx)}")
    
    # Load validation GOPs directly from dataset INTO RAM
    print(f"   � Loading up to {args.max_val_gops} GOPs into RAM (this may take a moment)...")
    
    gop_len = args.gop_length
    ordered_val_gops = []
    
    for gop_count, seq_idx in enumerate(val_seq_idx[:args.max_val_gops]):
        sequence_info = dataset.sequences[seq_idx]
        sequence_id = f"{sequence_info['video_name']}_gop{sequence_info['gop_index']}"
        
        print(f"      Loading GOP {gop_count+1}/{min(args.max_val_gops, len(val_seq_idx))}: {sequence_id}...", end=' ', flush=True)
        
        frames = []
        for frame_idx in range(gop_len):
            global_idx = seq_idx * gop_len + frame_idx
            if global_idx >= len(dataset):
                break
            
            try:
                sample = dataset[global_idx]
                
                # Extract and CLONE all data into RAM (detach from disk-backed storage)
                frame_data = {
                    'boxes': sample.get('boxes').clone() if sample.get('boxes') is not None else None,
                    'motion_vectors': sample.get('motion_vectors').clone() if sample.get('motion_vectors') is not None else None,
                    'residuals': sample.get('residuals').clone() if sample.get('residuals') is not None else None,
                    'ids': sample.get('ids').clone() if sample.get('ids') is not None else None,
                    'sequence_id': sequence_id,
                    'frame_id': frame_idx,
                }
                
                # Add RGB frames if available - CLONE to RAM
                if 'iframe' in sample and sample['iframe'] is not None:
                    iframe = sample['iframe']
                    if isinstance(iframe, torch.Tensor):
                        frame_data['iframe_rgb'] = iframe.clone()
                    else:
                        frame_data['iframe_rgb'] = torch.from_numpy(iframe.copy()) if hasattr(iframe, 'copy') else torch.tensor(iframe)
                
                if 'pframe' in sample and sample['pframe'] is not None:
                    pframe = sample['pframe']
                    if isinstance(pframe, torch.Tensor):
                        frame_data['pframe_rgb'] = pframe.clone()
                    else:
                        frame_data['pframe_rgb'] = torch.from_numpy(pframe.copy()) if hasattr(pframe, 'copy') else torch.tensor(pframe)
                
                frames.append(frame_data)
                
            except Exception as e:
                print(f"\n      ⚠️ Failed to load frame {frame_idx} from GOP {seq_idx}: {e}")
                continue
        
        # Only keep GOPs with at least 2 frames
        if len(frames) >= 2:
            ordered_val_gops.append((sequence_id, frames))
            print(f"✓ ({len(frames)} frames)")
        else:
            print(f"✗ (insufficient frames)")
        
        if len(ordered_val_gops) >= args.max_val_gops:
            break
    
    print(f"   ✅ Loaded {len(ordered_val_gops)} balanced validation GOPs into RAM")
    
    # Print GOP statistics
    gop_lengths = [len(frames) for _, frames in ordered_val_gops]
    if gop_lengths:
        print(f"   📊 GOP lengths: min={min(gop_lengths)}, max={max(gop_lengths)}, avg={sum(gop_lengths)/len(gop_lengths):.1f}")
    
    # Estimate memory usage
    total_frames = sum(gop_lengths)
    est_memory_mb = total_frames * args.resolution * args.resolution * 3 * 2 / 1024 / 1024  # RGB (3 channels) + compressed data
    print(f"   💾 Estimated RAM usage: ~{est_memory_mb:.1f} MB for {total_frames} frames")
    
    return ordered_val_gops


def benchmark_dct_mv_model(model, val_gops, device, num_gops, warmup):
    """
    Benchmark DCT-MV model on validation GOPs.
    Uses motion vectors + DCT residuals (compressed domain).
    """
    print(f"\n⏱️  Benchmarking DCT-MV model (compressed domain)...")
    
    model.eval()
    total_available = len(val_gops)
    num_gops = min(num_gops, total_available - warmup)
    
    if num_gops <= 0:
        raise ValueError(f"Not enough GOPs! Need at least {warmup + 1}")
    
    # Warmup
    print(f"   🔥 Warming up ({warmup} iterations)...")
    for i in range(min(warmup, total_available)):
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) < 2:
            continue
        
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        
        mvs, dcts = [], []
        for t in range(1, len(gop_frames)):
            mv = gop_frames[t]['motion_vectors'].to(device)
            dct = gop_frames[t].get('residuals', None)
            
            if mv.ndim == 4 and mv.shape[-1] == 2:
                mv = mv[..., 0]
            mvs.append(mv)
            
            if dct is not None:
                if dct.ndim == 4:
                    dct = dct.squeeze(0)
                dcts.append(dct.to(device))
        
        mv_seq = torch.stack(mvs)
        dct_seq = torch.stack(dcts) if dcts else None
        
        with torch.no_grad():
            _ = model(mv_seq, dct_seq, iframe_boxes)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"   ⏱️  Running benchmark on {num_gops} GOPs...")
    times = []
    frame_counts = []
    object_counts = []
    
    for i in tqdm(range(warmup, warmup + num_gops), desc="DCT-MV"):
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) < 2:
            continue
        
        iframe_boxes = gop_frames[0]['boxes'].to(device)
        N = iframe_boxes.shape[0]
        T = len(gop_frames) - 1
        
        mvs, dcts = [], []
        for t in range(1, len(gop_frames)):
            mv = gop_frames[t]['motion_vectors'].to(device)
            dct = gop_frames[t].get('residuals', None)
            
            if mv.ndim == 4 and mv.shape[-1] == 2:
                mv = mv[..., 0]
            mvs.append(mv)
            
            if dct is not None:
                if dct.ndim == 4:
                    dct = dct.squeeze(0)
                dcts.append(dct.to(device))
        
        mv_seq = torch.stack(mvs)
        dct_seq = torch.stack(dcts) if dcts else None
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(mv_seq, dct_seq, iframe_boxes)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        frame_counts.append(T)
        object_counts.append(N)
    
    # Statistics
    times = np.array(times)
    frame_counts = np.array(frame_counts)
    total_frames = frame_counts.sum()
    total_time = times.sum()
    
    return {
        'gop_time_mean_ms': float(times.mean() * 1000),
        'gop_time_std_ms': float(times.std() * 1000),
        'frame_time_mean_ms': float((total_time / total_frames) * 1000),
        'fps': float(total_frames / total_time),
        'total_gops': int(num_gops),
        'total_frames': int(total_frames),
        'avg_objects_per_gop': float(np.array(object_counts).mean()),
    }


def benchmark_rtdetr_model(model, val_gops, device, num_gops, warmup, img_size=960):
    """
    Benchmark RT-DETR model on validation GOPs.
    Uses RGB I-frames and P-frames (pixel domain).
    Measures INFERENCE ONLY (model forward pass, no NMS/post-processing).
    """
    print(f"\n⏱️  Benchmarking RT-DETR model (RGB pixel domain - INFERENCE ONLY)...")
    
    model.eval()
    # Access the actual model for inference-only benchmarking
    inference_model = model.model
    
    total_available = len(val_gops)
    num_gops = min(num_gops, total_available - warmup)
    
    # Warmup
    print(f"   🔥 Warming up ({warmup} iterations)...")
    for i in range(min(warmup, total_available)):
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) < 2:
            continue
        
        # Process I-frame (frame 0 has iframe_rgb)
        iframe_rgb = gop_frames[0].get('iframe_rgb')
        if iframe_rgb is not None:
            iframe_rgb = iframe_rgb.to(device).float() / 255.0  # Normalize once
            # Convert from HWC to CHW if needed
            if iframe_rgb.ndim == 3 and iframe_rgb.shape[-1] == 3:
                iframe_rgb = iframe_rgb.permute(2, 0, 1)  # HWC -> CHW
            elif iframe_rgb.ndim == 4 and iframe_rgb.shape[-1] == 3:
                iframe_rgb = iframe_rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
            if iframe_rgb.ndim == 3:
                iframe_rgb = iframe_rgb.unsqueeze(0)  # CHW -> BCHW
            with torch.no_grad():
                _ = inference_model(iframe_rgb)  # Just forward pass, no predict()
        
        # Process first P-frame (frame 1 has pframe_rgb)
        pframe_rgb = gop_frames[1].get('pframe_rgb')
        if pframe_rgb is not None:
            pframe_rgb = pframe_rgb.to(device).float() / 255.0  # Normalize once
            # Convert from HWC to CHW if needed
            if pframe_rgb.ndim == 3 and pframe_rgb.shape[-1] == 3:
                pframe_rgb = pframe_rgb.permute(2, 0, 1)  # HWC -> CHW
            elif pframe_rgb.ndim == 4 and pframe_rgb.shape[-1] == 3:
                pframe_rgb = pframe_rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
            if pframe_rgb.ndim == 3:
                pframe_rgb = pframe_rgb.unsqueeze(0)  # CHW -> BCHW
            with torch.no_grad():
                _ = inference_model(pframe_rgb)  # Just forward pass, no predict()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"   ⏱️  Running benchmark on {num_gops} GOPs...")
    times = []
    frame_counts = []
    
    for i in tqdm(range(warmup, warmup + num_gops), desc="RT-DETR"):
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) < 2:
            continue
        
        # Count frames with RGB data
        frame_count = 0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            # Process I-frame (frame 0)
            iframe_rgb = gop_frames[0].get('iframe_rgb')
            if iframe_rgb is not None:
                iframe_rgb = iframe_rgb.to(device).float() / 255.0  # Normalize once
                # Convert from HWC to CHW if needed
                if iframe_rgb.ndim == 3 and iframe_rgb.shape[-1] == 3:
                    iframe_rgb = iframe_rgb.permute(2, 0, 1)  # HWC -> CHW
                elif iframe_rgb.ndim == 4 and iframe_rgb.shape[-1] == 3:
                    iframe_rgb = iframe_rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
                if iframe_rgb.ndim == 3:
                    iframe_rgb = iframe_rgb.unsqueeze(0)  # CHW -> BCHW
                _ = inference_model(iframe_rgb)  # Just forward pass
                frame_count += 1
            
            # Process all P-frames
            for t in range(1, len(gop_frames)):
                pframe_rgb = gop_frames[t].get('pframe_rgb')
                if pframe_rgb is not None:
                    pframe_rgb = pframe_rgb.to(device).float() / 255.0  # Normalize once
                    # Convert from HWC to CHW if needed
                    if pframe_rgb.ndim == 3 and pframe_rgb.shape[-1] == 3:
                        pframe_rgb = pframe_rgb.permute(2, 0, 1)  # HWC -> CHW
                    elif pframe_rgb.ndim == 4 and pframe_rgb.shape[-1] == 3:
                        pframe_rgb = pframe_rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
                    if pframe_rgb.ndim == 3:
                        pframe_rgb = pframe_rgb.unsqueeze(0)  # CHW -> BCHW
                    _ = inference_model(pframe_rgb)  # Just forward pass
                    frame_count += 1
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        times.append(end_time - start_time)
        frame_counts.append(frame_count)
    
    # Statistics
    times = np.array(times)
    frame_counts = np.array(frame_counts)
    total_frames = frame_counts.sum()
    total_time = times.sum()
    
    return {
        'gop_time_mean_ms': float(times.mean() * 1000),
        'gop_time_std_ms': float(times.std() * 1000),
        'frame_time_mean_ms': float((total_time / total_frames) * 1000),
        'fps': float(total_frames / total_time),
        'total_gops': int(num_gops),
        'total_frames': int(total_frames),
    }


def print_comparison_results(results):
    """Print comparison results in a formatted table."""
    print("\n" + "="*100)
    print("FAIR COMPARISON: DCT-MV vs RT-DETR BASELINE")
    print("="*100)
    
    dct_stats = results['dct_mv']
    rtdetr_stats = results['rtdetr']
    
    # Model information
    print(f"\n📊 Model Information:")
    print(f"   {'Model':<20} {'Parameters':<15} {'Model Size (MB)':<18} {'Input Type':<30}")
    print(f"   {'-'*20} {'-'*15} {'-'*18} {'-'*30}")
    
    dct_mv_size_mb = results['dct_mv_params'] * 4 / 1024 / 1024
    rtdetr_size_mb = results['rtdetr_params'] * 4 / 1024 / 1024
    
    print(f"   {'DCT-MV':<20} {results['dct_mv_params']:>14,} {dct_mv_size_mb:>17.1f} {'MV + DCT (compressed)':<30}")
    print(f"   {'RT-DETR':<20} {results['rtdetr_params']:>14,} {rtdetr_size_mb:>17.1f} {'RGB frames (pixel domain)':<30}")
    
    # Speed comparison
    print(f"\n⏱️  Inference Speed Comparison:")
    print(f"   {'Metric':<30} {'DCT-MV':<20} {'RT-DETR':<20} {'Speedup':<15}")
    print(f"   {'-'*30} {'-'*20} {'-'*20} {'-'*15}")
    
    speedup_fps = dct_stats['fps'] / rtdetr_stats['fps']
    speedup_frame = rtdetr_stats['frame_time_mean_ms'] / dct_stats['frame_time_mean_ms']
    speedup_gop = rtdetr_stats['gop_time_mean_ms'] / dct_stats['gop_time_mean_ms']
    
    print(f"   {'FPS (frames/sec)':<30} {dct_stats['fps']:>19.1f} {rtdetr_stats['fps']:>19.1f} {speedup_fps:>14.2f}x")
    print(f"   {'Frame Time (ms)':<30} {dct_stats['frame_time_mean_ms']:>19.2f} {rtdetr_stats['frame_time_mean_ms']:>19.2f} {speedup_frame:>14.2f}x")
    print(f"   {'GOP Time (ms)':<30} {dct_stats['gop_time_mean_ms']:>19.2f} {rtdetr_stats['gop_time_mean_ms']:>19.2f} {speedup_gop:>14.2f}x")
    
    # Memory comparison
    print(f"\n💾 GPU Memory Usage During Inference:")
    dct_mem = results['dct_mv_memory']
    rtdetr_mem = results['rtdetr_memory']
    
    if 'gpu_allocated_mb' in dct_mem and 'gpu_allocated_mb' in rtdetr_mem:
        print(f"   {'Metric':<30} {'DCT-MV':<20} {'RT-DETR':<20} {'Difference':<15}")
        print(f"   {'-'*30} {'-'*20} {'-'*20} {'-'*15}")
        print(f"   {'GPU Allocated (MB)':<30} {dct_mem['gpu_allocated_mb']:>19.1f} {rtdetr_mem['gpu_allocated_mb']:>19.1f} {rtdetr_mem['gpu_allocated_mb']-dct_mem['gpu_allocated_mb']:>14.1f}")
        print(f"   {'GPU Peak (MB)':<30} {dct_mem['gpu_max_allocated_mb']:>19.1f} {rtdetr_mem['gpu_max_allocated_mb']:>19.1f} {rtdetr_mem['gpu_max_allocated_mb']-dct_mem['gpu_max_allocated_mb']:>14.1f}")
        print(f"\n   ⚠️  Note: GPU memory includes model + activations + intermediate tensors")
    
    # Summary
    print(f"\n📈 Summary:")
    dct_mv_size_mb = results['dct_mv_params'] * 4 / 1024 / 1024
    rtdetr_size_mb = results['rtdetr_params'] * 4 / 1024 / 1024
    size_ratio = rtdetr_size_mb / dct_mv_size_mb
    
    print(f"   DCT-MV is {speedup_fps:.1f}x {'FASTER' if speedup_fps > 1 else 'SLOWER'} than RT-DETR")
    print(f"   DCT-MV is {size_ratio:.1f}x SMALLER than RT-DETR ({dct_mv_size_mb:.1f} MB vs {rtdetr_size_mb:.1f} MB)")
    print(f"   DCT-MV has {results['dct_mv_params']/results['rtdetr_params']*100:.2f}% of RT-DETR's parameters")
    print(f"   DCT-MV uses COMPRESSED domain data (MV + DCT)")
    print(f"   RT-DETR uses PIXEL domain data (RGB frames)")
    print(f"\n⚠️  Note: RT-DETR speed measured as INFERENCE ONLY (model.forward)")
    print(f"   - Excludes NMS and post-processing")
    print(f"   - Resolution: {results['resolution']}x{results['resolution']}")
    print(f"   - Ultralytics docs report ~110 FPS @ 640x640 on T4 GPU")
    
    print("\n" + "="*100)


def main():
    args = parse_arguments()
    
    print(f"🚀 Fair Comparison: DCT-MV vs RT-DETR Baseline")
    print(f"="*100)
    
    # Device setup
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Device: {device}")
    
    # Load DCT-MV model
    dct_mv_model, dct_mv_params = load_dct_mv_model(args.dct_mv_model, device)
    
    # Load RT-DETR model
    rtdetr_model, rtdetr_params = load_rtdetr_model(args.rtdetr_model, device)
    if rtdetr_model is None:
        print("❌ Cannot proceed without RT-DETR model")
        return 1
    
    # Load validation dataset with RGB frames
    val_gops = load_validation_gops_rgb(args)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark DCT-MV model
    print(f"\n{'='*100}")
    print("BENCHMARKING DCT-MV MODEL")
    print(f"{'='*100}")
    
    dct_mv_stats = benchmark_dct_mv_model(
        dct_mv_model, val_gops, device,
        args.num_gops, args.warmup
    )
    dct_mv_memory = get_memory_usage()
    
    # Clear GPU memory before RT-DETR benchmark
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark RT-DETR model
    print(f"\n{'='*100}")
    print("BENCHMARKING RT-DETR BASELINE")
    print(f"{'='*100}")
    
    rtdetr_stats = benchmark_rtdetr_model(
        rtdetr_model, val_gops, device,
        args.num_gops, args.warmup, img_size=args.resolution
    )
    rtdetr_memory = get_memory_usage()
    
    # Compile results
    results = {
        'dct_mv': dct_mv_stats,
        'rtdetr': rtdetr_stats,
        'dct_mv_params': dct_mv_params,
        'rtdetr_params': rtdetr_params,
        'dct_mv_memory': dct_mv_memory,
        'rtdetr_memory': rtdetr_memory,
        'device': str(device),
        'resolution': args.resolution,
        'dct_mv_checkpoint': args.dct_mv_model,
        'rtdetr_model': args.rtdetr_model,
    }
    
    # Print comparison
    print_comparison_results(results)
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("\n✅ Comparison complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
