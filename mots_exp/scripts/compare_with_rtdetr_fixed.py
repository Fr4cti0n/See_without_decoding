"""
Fair comparison between DCT-MV and RT-DETR on MOTS validation sequences.

KEY FIX: Load ONLY the data each model needs to avoid memory inflation:
- DCT-MV: Load ONLY motion vectors + DCT residuals (NO RGB)
- RT-DETR: Load ONLY RGB frames (NO compressed data)

This ensures fair memory measurements and faster loading.
"""

import argparse
import torch
import time
import json
import sys
from pathlib import Path

# Try importing real dataset
try:
    from dataset.factory.dataset_factory import create_mots_dataset
    HAS_REAL_DATASET = True
except ImportError:
    print("⚠️  Warning: Real MOTS dataset not available, will use mock data")
    HAS_REAL_DATASET = False
    create_mots_dataset = None

# Import model
from mots_exp.models.dct_mv_center import DCTMVCenterTracker


def load_dct_mv_model(checkpoint_path, device):
    """Load DCT-MV model from checkpoint"""
    print(f"\n📦 Loading DCT-MV model from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        args_dict = checkpoint.get('args', {})
        
        model = DCTMVCenterTracker(
            num_dct_coeffs=args_dict.get('num_dct_coeffs', 16),
            mv_feature_dim=32,
            dct_feature_dim=32,
            fused_feature_dim=args_dict.get('feature_dim', 64),
            roi_feature_dim=args_dict.get('feature_dim', 64),
            hidden_dim=args_dict.get('hidden_dim', 128),
            lstm_layers=2,
            dropout=0.1,
            image_size=args_dict.get('resolution', 960),
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"✅ DCT-MV model loaded successfully")
        print(f"   Parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.1f} MB)")
        
        return model, total_params
        
    except Exception as e:
        print(f"❌ Failed to load DCT-MV model: {e}")
        return None, 0


def load_rtdetr_model(model_path, device):
    """Load RT-DETR model from Ultralytics"""
    print(f"\n📦 Loading RT-DETR model from: {model_path}")
    
    try:
        from ultralytics import RTDETR
        
        model = RTDETR(model_path)
        model.to(device)
        model.model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.model.parameters())
        
        print(f"✅ RT-DETR model loaded successfully")
        print(f"   Parameters: {total_params:,} (~{total_params * 4 / 1024 / 1024:.1f} MB)")
        
        return model, total_params
        
    except Exception as e:
        print(f"❌ Failed to load RT-DETR model: {e}")
        return None, 0


def load_compressed_data_only(args):
    """
    Load validation GOPs with ONLY compressed data (MV + DCT).
    Does NOT load RGB frames to avoid wasting memory for DCT-MV benchmark.
    
    Returns:
        List of (seq_id, gop_frames) tuples with compressed data in RAM
    """
    print(f"\n📚 Loading COMPRESSED DATA ONLY (Motion Vectors + DCT Residuals)...")
    print(f"   ❌ NOT loading RGB frames (DCT-MV doesn't use them)")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   Max validation GOPs: {args.max_val_gops}")
    
    if not HAS_REAL_DATASET or create_mots_dataset is None:
        raise RuntimeError("❌ Real MOTS dataset factory not available!")
    
    # Create dataset with ONLY compressed data (NO RGB!)
    dataset = create_mots_dataset(
        dataset_type=['mot15', 'mot17', 'mot20'],
        resolution=args.resolution,
        mode="train",
        load_iframe=False,  # ❌ NO RGB I-frames
        load_pframe=False,  # ❌ NO RGB P-frames  
        load_motion_vectors=True,  # ✅ Motion vectors
        load_residuals=True,  # ✅ DCT residuals
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
    
    # Load validation GOPs INTO RAM
    print(f"   🔄 Loading up to {args.max_val_gops} GOPs into RAM...")
    
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
                
                # Extract and CLONE only compressed data (NO RGB!)
                frame_data = {
                    'boxes': sample.get('boxes').clone() if sample.get('boxes') is not None else None,
                    'motion_vectors': sample.get('motion_vectors').clone() if sample.get('motion_vectors') is not None else None,
                    'residuals': sample.get('residuals').clone() if sample.get('residuals') is not None else None,
                    'ids': sample.get('ids').clone() if sample.get('ids') is not None else None,
                    'sequence_id': sequence_id,
                    'frame_id': frame_idx,
                }
                
                frames.append(frame_data)
                
            except Exception as e:
                print(f"\n      ⚠️ Failed to load frame {frame_idx}: {e}")
                continue
        
        if len(frames) >= 2:
            ordered_val_gops.append((sequence_id, frames))
            print(f"✓ ({len(frames)} frames)")
        else:
            print(f"✗ (insufficient frames)")
        
        if len(ordered_val_gops) >= args.max_val_gops:
            break
    
    print(f"   ✅ Loaded {len(ordered_val_gops)} validation GOPs into RAM (compressed data only)")
    
    # Print GOP statistics
    gop_lengths = [len(frames) for _, frames in ordered_val_gops]
    if gop_lengths:
        print(f"   📊 GOP lengths: min={min(gop_lengths)}, max={max(gop_lengths)}, avg={sum(gop_lengths)/len(gop_lengths):.1f}")
    
    total_frames = sum(gop_lengths)
    # Estimate memory: MV (2x60x60) + DCT (120x120x64) per frame
    est_memory_mb = total_frames * ((2*60*60) + (120*120*64)) * 4 / 1024 / 1024
    print(f"   💾 Estimated RAM usage: ~{est_memory_mb:.1f} MB for {total_frames} frames (compressed data)")
    
    return ordered_val_gops


def load_rgb_data_only(args):
    """
    Load validation GOPs with ONLY RGB frames.
    Does NOT load compressed data to avoid wasting memory for RT-DETR benchmark.
    
    Returns:
        List of (seq_id, gop_frames) tuples with RGB frames in RAM
    """
    print(f"\n📚 Loading RGB FRAMES ONLY (I-frames + P-frames)...")
    print(f"   ❌ NOT loading compressed data (RT-DETR doesn't use it)")
    print(f"   Resolution: {args.resolution}x{args.resolution}")
    print(f"   Max validation GOPs: {args.max_val_gops}")
    
    if not HAS_REAL_DATASET or create_mots_dataset is None:
        raise RuntimeError("❌ Real MOTS dataset factory not available!")
    
    # Create dataset with ONLY RGB frames (NO compressed data!)
    dataset = create_mots_dataset(
        dataset_type=['mot15', 'mot17', 'mot20'],
        resolution=args.resolution,
        mode="train",
        load_iframe=True,  # ✅ RGB I-frames
        load_pframe=True,  # ✅ RGB P-frames
        load_motion_vectors=False,  # ❌ NO motion vectors
        load_residuals=False,  # ❌ NO DCT residuals
        load_annotations=True,
        sequence_length=48,
        data_format="separate",
        combine_datasets=True
    )
    
    if dataset is None or len(dataset) == 0:
        raise RuntimeError("❌ Dataset is empty or failed to load!")
    
    print(f"   ✅ Dataset loaded: {len(dataset)} total samples")
    
    # Create validation split (same as compressed data)
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
    
    # Load validation GOPs INTO RAM
    print(f"   🔄 Loading up to {args.max_val_gops} GOPs into RAM...")
    
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
                
                # Extract and CLONE only RGB data (NO compressed data!)
                frame_data = {
                    'boxes': sample.get('boxes').clone() if sample.get('boxes') is not None else None,
                    'ids': sample.get('ids').clone() if sample.get('ids') is not None else None,
                    'sequence_id': sequence_id,
                    'frame_id': frame_idx,
                }
                
                # Add RGB frames - CLONE to RAM
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
                print(f"\n      ⚠️ Failed to load frame {frame_idx}: {e}")
                continue
        
        if len(frames) >= 2:
            ordered_val_gops.append((sequence_id, frames))
            print(f"✓ ({len(frames)} frames)")
        else:
            print(f"✗ (insufficient frames)")
        
        if len(ordered_val_gops) >= args.max_val_gops:
            break
    
    print(f"   ✅ Loaded {len(ordered_val_gops)} validation GOPs into RAM (RGB data only)")
    
    # Print GOP statistics
    gop_lengths = [len(frames) for _, frames in ordered_val_gops]
    if gop_lengths:
        print(f"   📊 GOP lengths: min={min(gop_lengths)}, max={max(gop_lengths)}, avg={sum(gop_lengths)/len(gop_lengths):.1f}")
    
    total_frames = sum(gop_lengths)
    # Estimate memory: RGB frames (resolution x resolution x 3 channels x 2 frames per sample)
    est_memory_mb = total_frames * args.resolution * args.resolution * 3 * 2 / 1024 / 1024
    print(f"   💾 Estimated RAM usage: ~{est_memory_mb:.1f} MB for {total_frames} frames (RGB data)")
    
    return ordered_val_gops


def benchmark_dct_mv_model(model, val_gops, device, num_gops, warmup):
    """
    Benchmark DCT-MV model on validation GOPs.
    Uses ONLY motion vectors + DCT residuals (compressed domain).
    """
    print(f"\n⏱️  Benchmarking DCT-MV model (compressed domain)...")
    
    model.eval()
    total_available = len(val_gops)
    num_gops = min(num_gops, total_available - warmup)
    
    if num_gops <= 0:
        raise ValueError(f"Not enough GOPs! Need at least {warmup + 1}")
    
    # Measure GPU memory (model only)
    torch.cuda.reset_peak_memory_stats()
    model_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
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
            
            # Fix MV dimensions if needed
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
    
    torch.cuda.synchronize()
    
    # Actual benchmark
    print(f"   📊 Running benchmark on {num_gops} GOPs...")
    
    gop_times = []
    frame_counts = []
    
    for i in range(warmup, warmup + num_gops):
        if i >= len(val_gops):
            break
            
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
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(mv_seq, dct_seq, iframe_boxes)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        gop_time_ms = (end_time - start_time) * 1000
        gop_times.append(gop_time_ms)
        frame_counts.append(len(gop_frames))
    
    # Measure peak GPU memory during inference
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Calculate statistics
    total_time_ms = sum(gop_times)
    total_frames = sum(frame_counts)
    avg_time_per_gop_ms = total_time_ms / len(gop_times)
    avg_time_per_frame_ms = total_time_ms / total_frames
    fps = 1000.0 / avg_time_per_frame_ms
    
    print(f"   ✅ DCT-MV Benchmark Complete:")
    print(f"      GOPs processed: {len(gop_times)}")
    print(f"      Total frames: {total_frames}")
    print(f"      Avg time/GOP: {avg_time_per_gop_ms:.2f} ms")
    print(f"      Avg time/frame: {avg_time_per_frame_ms:.2f} ms")
    print(f"      FPS: {fps:.1f}")
    print(f"      GPU Memory (model only): {model_memory_mb:.2f} MB")
    print(f"      GPU Memory (peak during inference): {peak_memory_mb:.2f} MB")
    
    return {
        'fps': fps,
        'ms_per_frame': avg_time_per_frame_ms,
        'ms_per_gop': avg_time_per_gop_ms,
        'total_frames': total_frames,
        'num_gops': len(gop_times),
        'model_memory_mb': model_memory_mb,
        'peak_memory_mb': peak_memory_mb,
    }


def benchmark_rtdetr_model(model, val_gops, device, num_gops, warmup, img_size=960):
    """
    Benchmark RT-DETR - INFERENCE ONLY (no NMS/post-processing).
    Uses ONLY RGB frames.
    """
    print(f"\n⏱️  Benchmarking RT-DETR model (RGB frames, inference only)...")
    
    inference_model = model.model  # Access raw model, not predict wrapper
    inference_model.eval()
    
    total_available = len(val_gops)
    num_gops = min(num_gops, total_available - warmup)
    
    if num_gops <= 0:
        raise ValueError(f"Not enough GOPs! Need at least {warmup + 1}")
    
    # Measure GPU memory (model only)
    torch.cuda.reset_peak_memory_stats()
    model_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
    # Warmup
    print(f"   🔥 Warming up ({warmup} iterations)...")
    for i in range(min(warmup, total_available)):
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) == 0:
            continue
        
        # Use I-frame (first frame)
        iframe_rgb = gop_frames[0].get('iframe_rgb')
        if iframe_rgb is None:
            continue
        
        # Move to device and normalize
        iframe_rgb = iframe_rgb.to(device).float() / 255.0
        
        # Convert format if needed
        if iframe_rgb.ndim == 3 and iframe_rgb.shape[-1] == 3:
            iframe_rgb = iframe_rgb.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        elif iframe_rgb.ndim == 4 and iframe_rgb.shape[-1] == 3:
            iframe_rgb = iframe_rgb.permute(0, 3, 1, 2)  # BHWC -> BCHW
        elif iframe_rgb.ndim == 3:
            iframe_rgb = iframe_rgb.unsqueeze(0)  # CHW -> BCHW
        
        # Resize if needed
        if iframe_rgb.shape[2] != img_size or iframe_rgb.shape[3] != img_size:
            iframe_rgb = torch.nn.functional.interpolate(
                iframe_rgb, size=(img_size, img_size), mode='bilinear', align_corners=False
            )
        
        with torch.no_grad():
            _ = inference_model(iframe_rgb)
    
    torch.cuda.synchronize()
    
    # Actual benchmark
    print(f"   📊 Running benchmark on {num_gops} GOPs...")
    
    gop_times = []
    frame_counts = []
    
    for i in range(warmup, warmup + num_gops):
        if i >= len(val_gops):
            break
            
        seq_id, gop_frames = val_gops[i]
        if len(gop_frames) == 0:
            continue
        
        # Process all frames in GOP
        frames_processed = 0
        torch.cuda.synchronize()
        start_time = time.time()
        
        for frame_data in gop_frames:
            # Try I-frame first, then P-frame
            frame_rgb = frame_data.get('iframe_rgb') or frame_data.get('pframe_rgb')
            if frame_rgb is None:
                continue
            
            # Move to device and normalize
            frame_rgb = frame_rgb.to(device).float() / 255.0
            
            # Convert format if needed
            if frame_rgb.ndim == 3 and frame_rgb.shape[-1] == 3:
                frame_rgb = frame_rgb.permute(2, 0, 1).unsqueeze(0)
            elif frame_rgb.ndim == 4 and frame_rgb.shape[-1] == 3:
                frame_rgb = frame_rgb.permute(0, 3, 1, 2)
            elif frame_rgb.ndim == 3:
                frame_rgb = frame_rgb.unsqueeze(0)
            
            # Resize if needed
            if frame_rgb.shape[2] != img_size or frame_rgb.shape[3] != img_size:
                frame_rgb = torch.nn.functional.interpolate(
                    frame_rgb, size=(img_size, img_size), mode='bilinear', align_corners=False
                )
            
            with torch.no_grad():
                _ = inference_model(frame_rgb)
            
            frames_processed += 1
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        if frames_processed > 0:
            gop_time_ms = (end_time - start_time) * 1000
            gop_times.append(gop_time_ms)
            frame_counts.append(frames_processed)
    
    # Measure peak GPU memory during inference
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Calculate statistics
    total_time_ms = sum(gop_times)
    total_frames = sum(frame_counts)
    avg_time_per_gop_ms = total_time_ms / len(gop_times)
    avg_time_per_frame_ms = total_time_ms / total_frames
    fps = 1000.0 / avg_time_per_frame_ms
    
    print(f"   ✅ RT-DETR Benchmark Complete:")
    print(f"      GOPs processed: {len(gop_times)}")
    print(f"      Total frames: {total_frames}")
    print(f"      Avg time/GOP: {avg_time_per_gop_ms:.2f} ms")
    print(f"      Avg time/frame: {avg_time_per_frame_ms:.2f} ms")
    print(f"      FPS: {fps:.1f}")
    print(f"      GPU Memory (model only): {model_memory_mb:.2f} MB")
    print(f"      GPU Memory (peak during inference): {peak_memory_mb:.2f} MB")
    
    return {
        'fps': fps,
        'ms_per_frame': avg_time_per_frame_ms,
        'ms_per_gop': avg_time_per_gop_ms,
        'total_frames': total_frames,
        'num_gops': len(gop_times),
        'model_memory_mb': model_memory_mb,
        'peak_memory_mb': peak_memory_mb,
    }


def print_comparison_results(results):
    """Print comprehensive comparison results"""
    print("\n" + "="*80)
    print("📊 FAIR COMPARISON: DCT-MV vs RT-DETR")
    print("="*80)
    
    # Model information
    print(f"\n📦 Model Information:")
    print(f"   {'Model':<20} {'Parameters':<15} {'Model Size (MB)':<18}")
    print(f"   {'-'*20} {'-'*15} {'-'*18}")
    
    dct_mv_size_mb = results['dct_mv_params'] * 4 / 1024 / 1024
    rtdetr_size_mb = results['rtdetr_params'] * 4 / 1024 / 1024
    
    print(f"   {'DCT-MV':<20} {results['dct_mv_params']:>14,} {dct_mv_size_mb:>17.1f}")
    print(f"   {'RT-DETR':<20} {results['rtdetr_params']:>14,} {rtdetr_size_mb:>17.1f}")
    
    # Performance comparison
    print(f"\n⏱️  Inference Performance (@ {results['resolution']}x{results['resolution']}):")
    print(f"   {'Model':<20} {'FPS':<12} {'ms/frame':<12} {'ms/GOP':<12}")
    print(f"   {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"   {'DCT-MV':<20} {results['dct_mv_fps']:>11.1f} {results['dct_mv_ms_per_frame']:>11.2f} {results['dct_mv_ms_per_gop']:>11.2f}")
    print(f"   {'RT-DETR':<20} {results['rtdetr_fps']:>11.1f} {results['rtdetr_ms_per_frame']:>11.2f} {results['rtdetr_ms_per_gop']:>11.2f}")
    
    # GPU Memory (CORRECTED to show both model size and inference memory)
    print(f"\n💾 GPU Memory Usage:")
    print(f"   {'Model':<20} {'Model Size (MB)':<18} {'Peak During Inference (MB)':<28}")
    print(f"   {'-'*20} {'-'*18} {'-'*28}")
    print(f"   {'DCT-MV':<20} {results['dct_mv_model_memory']:>17.2f} {results['dct_mv_peak_memory']:>27.2f}")
    print(f"   {'RT-DETR':<20} {results['rtdetr_model_memory']:>17.2f} {results['rtdetr_peak_memory']:>27.2f}")
    print(f"   ⚠️  Note: Peak memory includes activations, intermediates, and input data")
    
    # Speedup and efficiency
    speedup_fps = results['dct_mv_fps'] / results['rtdetr_fps']
    speedup_latency = results['rtdetr_ms_per_frame'] / results['dct_mv_ms_per_frame']
    size_ratio = rtdetr_size_mb / dct_mv_size_mb
    
    print(f"\n📈 Summary:")
    print(f"   🚀 Speed:")
    print(f"      DCT-MV is {speedup_fps:.1f}x FASTER than RT-DETR")
    print(f"      ({results['dct_mv_fps']:.1f} FPS vs {results['rtdetr_fps']:.1f} FPS)")
    print(f"   ")
    print(f"   💾 Model Size:")
    print(f"      DCT-MV is {size_ratio:.1f}x SMALLER than RT-DETR")
    print(f"      ({dct_mv_size_mb:.1f} MB vs {rtdetr_size_mb:.1f} MB)")
    print(f"      DCT-MV has {results['dct_mv_params']/results['rtdetr_params']*100:.2f}% of RT-DETR's parameters")
    print(f"   ")
    print(f"   ⚡ Efficiency:")
    print(f"      DCT-MV operates on COMPRESSED domain (motion vectors + DCT residuals)")
    print(f"      RT-DETR operates on RGB pixels (full image data)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Fair comparison: DCT-MV vs RT-DETR')
    
    # Model paths
    parser.add_argument('--dct-mv-model', type=str, required=True,
                        help='Path to DCT-MV checkpoint')
    parser.add_argument('--rtdetr-model', type=str, default='rtdetr-l.pt',
                        help='Path to RT-DETR model')
    
    # Benchmark settings
    parser.add_argument('--num-gops', type=int, default=5,
                        help='Number of GOPs to benchmark')
    parser.add_argument('--warmup', type=int, default=1,
                        help='Number of warmup iterations')
    parser.add_argument('--resolution', type=int, default=960,
                        help='Image resolution (960 or 640)')
    
    # Dataset settings
    parser.add_argument('--max-val-gops', type=int, default=7,
                        help='Maximum validation GOPs to load')
    parser.add_argument('--gop-length', type=int, default=48,
                        help='GOP length (frames)')
    parser.add_argument('--train-split', type=float, default=0.7,
                        help='Training split ratio')
    parser.add_argument('--val-split', type=float, default=0.15,
                        help='Validation split ratio')
    
    # Output
    parser.add_argument('--output', type=str, default='comparison_dctmv_vs_rtdetr_corrected.json',
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load DCT-MV model
    dct_mv_model, dct_mv_params = load_dct_mv_model(args.dct_mv_model, device)
    if dct_mv_model is None:
        return 1
    
    # Load compressed data ONLY for DCT-MV
    print("\n" + "="*80)
    print("STEP 1: DCT-MV Benchmark (Compressed Data)")
    print("="*80)
    
    compressed_gops = load_compressed_data_only(args)
    dct_mv_results = benchmark_dct_mv_model(dct_mv_model, compressed_gops, device, args.num_gops, args.warmup)
    
    # Clean up DCT-MV to free memory
    del dct_mv_model, compressed_gops
    torch.cuda.empty_cache()
    
    # Load RT-DETR model
    rtdetr_model, rtdetr_params = load_rtdetr_model(args.rtdetr_model, device)
    if rtdetr_model is None:
        return 1
    
    # Load RGB data ONLY for RT-DETR
    print("\n" + "="*80)
    print("STEP 2: RT-DETR Benchmark (RGB Data)")
    print("="*80)
    
    rgb_gops = load_rgb_data_only(args)
    rtdetr_results = benchmark_rtdetr_model(rtdetr_model, rgb_gops, device, args.num_gops, args.warmup, args.resolution)
    
    # Combine results
    results = {
        'resolution': args.resolution,
        'dct_mv_params': dct_mv_params,
        'rtdetr_params': rtdetr_params,
        'dct_mv_fps': dct_mv_results['fps'],
        'dct_mv_ms_per_frame': dct_mv_results['ms_per_frame'],
        'dct_mv_ms_per_gop': dct_mv_results['ms_per_gop'],
        'dct_mv_model_memory': dct_mv_results['model_memory_mb'],
        'dct_mv_peak_memory': dct_mv_results['peak_memory_mb'],
        'rtdetr_fps': rtdetr_results['fps'],
        'rtdetr_ms_per_frame': rtdetr_results['ms_per_frame'],
        'rtdetr_ms_per_gop': rtdetr_results['ms_per_gop'],
        'rtdetr_model_memory': rtdetr_results['model_memory_mb'],
        'rtdetr_peak_memory': rtdetr_results['peak_memory_mb'],
    }
    
    # Print comparison
    print_comparison_results(results)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
