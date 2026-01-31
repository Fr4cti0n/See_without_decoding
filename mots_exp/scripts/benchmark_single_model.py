"""
Clean isolated speed benchmark for a single model
No interference from other models or complex data loading
"""

import torch
import time
import numpy as np
import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.dct_mv_tracker import DCTMVCenterTracker


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Detect variant from path
    ckpt_name = str(checkpoint_path).lower()
    if 'mv_dct' in ckpt_name:
        mv_channels = 2
        if 'dct_8' in ckpt_name or 'dct-8' in ckpt_name:
            dct_channels = 8
        elif 'dct_16' in ckpt_name or 'dct-16' in ckpt_name:
            dct_channels = 16
        elif 'dct_32' in ckpt_name or 'dct-32' in ckpt_name:
            dct_channels = 32
        elif 'dct_64' in ckpt_name or 'dct-64' in ckpt_name:
            dct_channels = 64
        else:
            dct_channels = 8
    elif 'dct' in ckpt_name:
        mv_channels = 0
        if 'dct_8' in ckpt_name or 'dct-8' in ckpt_name:
            dct_channels = 8
        elif 'dct_16' in ckpt_name or 'dct-16' in ckpt_name:
            dct_channels = 16
        elif 'dct_32' in ckpt_name or 'dct-32' in ckpt_name:
            dct_channels = 32
        elif 'dct_64' in ckpt_name or 'dct-64' in ckpt_name:
            dct_channels = 64
        else:
            dct_channels = 8
    elif 'mv' in ckpt_name:
        mv_channels = 2
        dct_channels = 0
    else:
        raise ValueError(f"Cannot determine variant from path: {checkpoint_path}")
    
    num_dct_coeffs = dct_channels if dct_channels > 0 else 16
    
    print(f"Model config: MV={mv_channels} ch, DCT={dct_channels} ch, num_dct_coeffs={num_dct_coeffs}")
    
    model = DCTMVCenterTracker(
        num_dct_coeffs=num_dct_coeffs,
        mv_channels=mv_channels,
        dct_channels=dct_channels,
        mv_feature_dim=32,
        dct_feature_dim=32,
        fused_feature_dim=64,
        roi_feature_dim=64,
        hidden_dim=128,
        lstm_layers=2,
        dropout=0.1,
        image_size=960,
        use_multiscale_roi=False,
        roi_sizes=[3, 7, 11],
        use_parallel_heads=True,
        use_attention=True,
        attention_heads=4
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model = model.to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # FP32
    
    return model, mv_channels, dct_channels, {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }


def create_realistic_data(mv_channels, dct_channels, device):
    """Create data with realistic ranges matching actual dataset"""
    mv_input = None
    dct_input = None
    
    if mv_channels > 0:
        # MV: mean=0.03, std=2.16, range [-15, 16]
        mv_input = torch.randn(1, mv_channels, 60, 60).to(device) * 2.16 + 0.03
        mv_input = torch.clamp(mv_input, -15, 16)
    
    if dct_channels > 0:
        # DCT: mean=1.40, std=40.84, range [-485, 1872]
        dct_input = torch.randn(1, 120, 120, dct_channels).to(device) * 40.84 + 1.40
        dct_input = torch.clamp(dct_input, -485, 1872)
    
    return mv_input, dct_input


def benchmark_single_frame(model, mv_input, dct_input, boxes, iterations=1000, warmup=100):
    """Benchmark single frame inference (no LSTM state carry-over)"""
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            model.forward_single_frame(mv_input, dct_input, boxes, None)
        
        # Benchmark
        torch.cuda.synchronize()
        times = []
        
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.time()
            model.forward_single_frame(mv_input, dct_input, boxes, None)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
    
    return np.array(times)


def benchmark_sequence(model, mv_input, dct_input, boxes, num_frames=47, warmup=5, repeats=10):
    """Benchmark full sequence with LSTM state carry-over"""
    times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            hidden_state = None
            for _ in range(num_frames):
                _, _, hidden_state = model.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        # Benchmark
        for _ in range(repeats):
            torch.cuda.synchronize()
            start = time.time()
            
            hidden_state = None
            for _ in range(num_frames):
                _, _, hidden_state = model.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
            
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    return np.array(times)


def main():
    parser = argparse.ArgumentParser(description='Clean isolated speed benchmark')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--num-objects', type=int, default=5, help='Number of objects to track')
    parser.add_argument('--iterations', type=int, default=1000, help='Iterations for single-frame benchmark')
    parser.add_argument('--warmup', type=int, default=100, help='Warmup iterations')
    parser.add_argument('--seq-frames', type=int, default=47, help='Frames per sequence')
    parser.add_argument('--seq-repeats', type=int, default=10, help='Sequence repetitions')
    parser.add_argument('--output', help='Output JSON file')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("="*80)
    print("ISOLATED SPEED BENCHMARK")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Model: {args.model_path}")
    print(f"Objects: {args.num_objects}")
    print(f"Single-frame iterations: {args.iterations} (warmup: {args.warmup})")
    print(f"Sequence: {args.seq_frames} frames × {args.seq_repeats} repeats")
    
    # Load model
    print("\n" + "="*80)
    print("Loading model...")
    print("="*80)
    model, mv_ch, dct_ch, model_info = load_model(args.model_path, device)
    
    print(f"\nModel info:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create data
    print("\n" + "="*80)
    print("Creating realistic test data...")
    print("="*80)
    mv_input, dct_input = create_realistic_data(mv_ch, dct_ch, device)
    
    # Create boxes
    boxes = torch.rand(args.num_objects, 4).to(device) * 960
    boxes[:, 2:] = boxes[:, :2] + 50  # Make boxes 50x50
    boxes = torch.clamp(boxes, 0, 960)
    
    if mv_input is not None:
        print(f"  MV shape: {mv_input.shape}, range: [{mv_input.min():.2f}, {mv_input.max():.2f}]")
    if dct_input is not None:
        print(f"  DCT shape: {dct_input.shape}, range: [{dct_input.min():.2f}, {dct_input.max():.2f}]")
    print(f"  Boxes shape: {boxes.shape}")
    
    # Measure memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)
    
    # Benchmark 1: Single frame (no state)
    print("\n" + "="*80)
    print("Benchmark 1: Single frame inference (no LSTM state carry-over)")
    print("="*80)
    times_single = benchmark_single_frame(model, mv_input, dct_input, boxes, args.iterations, args.warmup)
    
    print(f"\nResults:")
    print(f"  Mean:   {times_single.mean():.3f} ms/frame")
    print(f"  Median: {np.median(times_single):.3f} ms/frame")
    print(f"  Std:    {times_single.std():.3f} ms")
    print(f"  Min:    {times_single.min():.3f} ms")
    print(f"  Max:    {times_single.max():.3f} ms")
    print(f"  FPS:    {1000/times_single.mean():.1f}")
    
    # Benchmark 2: Sequence with LSTM state
    print("\n" + "="*80)
    print(f"Benchmark 2: Full sequence ({args.seq_frames} frames with LSTM state carry-over)")
    print("="*80)
    times_seq = benchmark_sequence(model, mv_input, dct_input, boxes, args.seq_frames, 5, args.seq_repeats)
    
    time_per_frame = (times_seq.mean() / args.seq_frames) * 1000
    
    print(f"\nResults:")
    print(f"  Sequence time (mean): {times_seq.mean()*1000:.2f} ms")
    print(f"  Time per frame:       {time_per_frame:.3f} ms/frame")
    print(f"  FPS:                  {1000/time_per_frame:.1f}")
    print(f"  Std:                  {times_seq.std()*1000:.2f} ms")
    
    # Memory usage
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\nMemory usage:")
        print(f"  Baseline: {baseline_memory:.2f} MB")
        print(f"  Peak:     {peak_memory:.2f} MB")
        print(f"  Used:     {peak_memory - baseline_memory:.2f} MB")
    
    # Save results
    results = {
        'model_path': str(args.model_path),
        'device': str(device),
        'mv_channels': mv_ch,
        'dct_channels': dct_ch,
        'num_objects': args.num_objects,
        'model_info': model_info,
        'single_frame': {
            'iterations': args.iterations,
            'warmup': args.warmup,
            'mean_ms': float(times_single.mean()),
            'median_ms': float(np.median(times_single)),
            'std_ms': float(times_single.std()),
            'min_ms': float(times_single.min()),
            'max_ms': float(times_single.max()),
            'fps': float(1000 / times_single.mean())
        },
        'sequence': {
            'num_frames': args.seq_frames,
            'repeats': args.seq_repeats,
            'mean_total_ms': float(times_seq.mean() * 1000),
            'mean_per_frame_ms': float(time_per_frame),
            'fps': float(1000 / time_per_frame),
            'std_ms': float(times_seq.std() * 1000)
        }
    }
    
    if device.type == 'cuda':
        results['memory'] = {
            'baseline_mb': float(baseline_memory),
            'peak_mb': float(peak_memory),
            'used_mb': float(peak_memory - baseline_memory)
        }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nModel: {'MV-only' if mv_ch > 0 and dct_ch == 0 else 'DCT-only' if mv_ch == 0 else f'MV+DCT-{dct_ch}'}")
    print(f"Single frame: {times_single.mean():.3f} ms/frame ({1000/times_single.mean():.1f} FPS)")
    print(f"Sequence:     {time_per_frame:.3f} ms/frame ({1000/time_per_frame:.1f} FPS)")
    print(f"Parameters:   {model_info['total_params']:,}")
    print(f"Model size:   {model_info['model_size_mb']:.2f} MB")
    if device.type == 'cuda':
        print(f"VRAM used:    {peak_memory - baseline_memory:.2f} MB")


if __name__ == "__main__":
    main()
