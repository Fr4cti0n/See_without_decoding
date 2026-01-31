"""
Profile COMPLETE forward pass for DCT-only vs MV+DCT
Step-by-step timing to find the bottleneck
"""

import torch
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.dct_mv_tracker import DCTMVCenterTracker


def benchmark_step(func, num_iters=100, warmup=20):
    """Benchmark a single step"""
    # Warmup
    for _ in range(warmup):
        func()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iters):
        func()
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / num_iters * 1000  # ms per iteration


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test parameters
    batch_size = 1
    num_objects = 5  # Based on DEBUG output
    num_iters = 100
    warmup = 20
    
    # Create dummy inputs
    mv_input = torch.randn(1, 2, 60, 60).to(device)
    dct_input = torch.randn(1, 120, 120, 8).to(device)
    boxes = torch.tensor([[100, 100, 200, 200],
                          [300, 300, 400, 400],
                          [500, 500, 600, 600],
                          [700, 700, 800, 800],
                          [150, 150, 250, 250]], dtype=torch.float32).to(device)
    
    print(f"Benchmark settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num objects: {num_objects}")
    print(f"  Iterations: {num_iters}")
    print(f"  Warmup: {warmup}")
    
    # ========================================================================
    # TEST 1: DCT-only model (DCT-8)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: DCT-only model (8 channels)")
    print("="*80)
    
    model_dct = DCTMVCenterTracker(
        num_dct_coeffs=8,
        mv_channels=0,
        dct_channels=8,
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
    ).to(device).eval()
    
    # Profile each component
    print("\n📊 Component-level profiling:")
    
    with torch.no_grad():
        # 1. Encoder
        def encoder_dct():
            return model_dct.encoder(None, dct_input)
        
        time_encoder = benchmark_step(encoder_dct, num_iters, warmup)
        print(f"  1. Encoder:        {time_encoder:.3f} ms")
        
        # Get encoder output for next steps
        fused_features, global_features = model_dct.encoder(None, dct_input)
        
        # 2. ROI Extractor
        def roi_extract_dct():
            return model_dct.roi_extractor(fused_features, boxes)
        
        time_roi = benchmark_step(roi_extract_dct, num_iters, warmup)
        print(f"  2. ROI Extractor:  {time_roi:.3f} ms")
        
        # Get ROI features for LSTM
        roi_features, _ = model_dct.roi_extractor(fused_features, boxes)
        
        # 3. LSTM Tracker
        def lstm_track_dct():
            return model_dct.lstm_tracker(roi_features, global_features[0], None, return_logits=False)
        
        time_lstm = benchmark_step(lstm_track_dct, num_iters, warmup)
        print(f"  3. LSTM Tracker:   {time_lstm:.3f} ms")
        
        # 4. Full forward pass
        def full_forward_dct():
            return model_dct.forward_single_frame(None, dct_input, boxes, None, return_logits=False)
        
        time_full = benchmark_step(full_forward_dct, num_iters, warmup)
        print(f"  4. Full forward:   {time_full:.3f} ms")
        
        print(f"\n  Sum of components:  {time_encoder + time_roi + time_lstm:.3f} ms")
        print(f"  Overhead:           {time_full - (time_encoder + time_roi + time_lstm):.3f} ms")
    
    # ========================================================================
    # TEST 2: MV+DCT model (MV + DCT-8)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: MV+DCT model (2 MV + 8 DCT channels)")
    print("="*80)
    
    model_mv_dct = DCTMVCenterTracker(
        num_dct_coeffs=8,
        mv_channels=2,
        dct_channels=8,
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
    ).to(device).eval()
    
    # Profile each component
    print("\n📊 Component-level profiling:")
    
    with torch.no_grad():
        # 1. Encoder
        def encoder_mv_dct():
            return model_mv_dct.encoder(mv_input, dct_input)
        
        time_encoder = benchmark_step(encoder_mv_dct, num_iters, warmup)
        print(f"  1. Encoder:        {time_encoder:.3f} ms")
        
        # Get encoder output for next steps
        fused_features, global_features = model_mv_dct.encoder(mv_input, dct_input)
        
        # 2. ROI Extractor
        def roi_extract_mv_dct():
            return model_mv_dct.roi_extractor(fused_features, boxes)
        
        time_roi = benchmark_step(roi_extract_mv_dct, num_iters, warmup)
        print(f"  2. ROI Extractor:  {time_roi:.3f} ms")
        
        # Get ROI features for LSTM
        roi_features, _ = model_mv_dct.roi_extractor(fused_features, boxes)
        
        # 3. LSTM Tracker
        def lstm_track_mv_dct():
            return model_mv_dct.lstm_tracker(roi_features, global_features[0], None, return_logits=False)
        
        time_lstm = benchmark_step(lstm_track_mv_dct, num_iters, warmup)
        print(f"  3. LSTM Tracker:   {time_lstm:.3f} ms")
        
        # 4. Full forward pass
        def full_forward_mv_dct():
            return model_mv_dct.forward_single_frame(mv_input, dct_input, boxes, None, return_logits=False)
        
        time_full = benchmark_step(full_forward_mv_dct, num_iters, warmup)
        print(f"  4. Full forward:   {time_full:.3f} ms")
        
        print(f"\n  Sum of components:  {time_encoder + time_roi + time_lstm:.3f} ms")
        print(f"  Overhead:           {time_full - (time_encoder + time_roi + time_lstm):.3f} ms")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*80)
    
    # Re-run with same iterations for fair comparison
    with torch.no_grad():
        # DCT-only
        def dct_full():
            return model_dct.forward_single_frame(None, dct_input, boxes, None, return_logits=False)
        
        time_dct_full = benchmark_step(dct_full, num_iters=200, warmup=50)
        
        # MV+DCT
        def mv_dct_full():
            return model_mv_dct.forward_single_frame(mv_input, dct_input, boxes, None, return_logits=False)
        
        time_mv_dct_full = benchmark_step(mv_dct_full, num_iters=200, warmup=50)
    
    print(f"\nFull forward pass timing:")
    print(f"  DCT-only:  {time_dct_full:.3f} ms/frame")
    print(f"  MV+DCT:    {time_mv_dct_full:.3f} ms/frame")
    print(f"  Speedup:   {time_dct_full / time_mv_dct_full:.2f}x")
    
    if time_dct_full > time_mv_dct_full:
        print(f"\n⚠️  PARADOX REPRODUCED: MV+DCT is {time_dct_full/time_mv_dct_full:.2f}x faster!")
    else:
        print(f"\n✅ Expected: DCT-only is {time_mv_dct_full/time_dct_full:.2f}x faster")
    
    print(f"\nFPS equivalent:")
    print(f"  DCT-only:  {1000/time_dct_full:.1f} FPS")
    print(f"  MV+DCT:    {1000/time_mv_dct_full:.1f} FPS")


if __name__ == "__main__":
    main()
