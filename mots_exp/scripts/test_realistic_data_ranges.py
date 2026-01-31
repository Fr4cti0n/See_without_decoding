"""
Test if data range affects performance
Compare torch.randn() vs realistic data ranges
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.dct_mv_tracker import DCTMVCenterTracker


def load_model(checkpoint_path, mv_ch, dct_ch, device):
    """Load model"""
    num_dct_coeffs = dct_ch if dct_ch > 0 else 16
    
    model = DCTMVCenterTracker(
        num_dct_coeffs=num_dct_coeffs,
        mv_channels=mv_ch,
        dct_channels=dct_ch,
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.to(device).eval()


def benchmark_with_data(model, mv_input, dct_input, boxes, num_frames=47, warmup=10):
    """Benchmark model"""
    with torch.no_grad():
        # Warmup
        hidden_state = None
        for _ in range(warmup):
            _, _, hidden_state = model.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        hidden_state = None
        for _ in range(num_frames):
            _, _, hidden_state = model.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
    
    return elapsed


def main():
    device = torch.device('cuda')
    
    # Settings
    num_objects = 5
    num_frames = 47
    
    boxes = torch.tensor([[100, 100, 200, 200],
                          [300, 300, 400, 400],
                          [500, 500, 600, 600],
                          [700, 700, 800, 800],
                          [150, 150, 250, 250]], dtype=torch.float32).to(device)
    
    print("="*80)
    print("TESTING: Does data range affect performance?")
    print("="*80)
    print(f"\nActual data ranges from dataset:")
    print(f"  MV:  [-15, 16], mean=0.03, std=2.16")
    print(f"  DCT: [-485, 1872], mean=1.40, std=40.84")
    
    # ========================================================================
    # Load models
    # ========================================================================
    print(f"\nLoading models...")
    model_dct = load_model('experiments/ablation_validation/dct_8/best_model.pt', 0, 8, device)
    model_mv_dct = load_model('experiments/ablation_validation/mv_dct_8/best_model.pt', 2, 8, device)
    
    # ========================================================================
    # TEST 1: torch.randn() (unrealistic, centered at 0, std=1)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: torch.randn() (unrealistic data)")
    print("="*80)
    
    mv_randn = torch.randn(1, 2, 60, 60).to(device)
    dct_randn = torch.randn(1, 120, 120, 8).to(device)
    
    print(f"\nGenerated data ranges:")
    print(f"  MV:  [{mv_randn.min():.2f}, {mv_randn.max():.2f}], std={mv_randn.std():.2f}")
    print(f"  DCT: [{dct_randn.min():.2f}, {dct_randn.max():.2f}], std={dct_randn.std():.2f}")
    
    time_dct_randn = benchmark_with_data(model_dct, None, dct_randn, boxes, num_frames)
    time_mv_dct_randn = benchmark_with_data(model_mv_dct, mv_randn, dct_randn, boxes, num_frames)
    
    print(f"\nResults with torch.randn():")
    print(f"  DCT-only:  {(time_dct_randn/num_frames)*1000:.3f} ms/frame ({num_frames/time_dct_randn:.1f} FPS)")
    print(f"  MV+DCT:    {(time_mv_dct_randn/num_frames)*1000:.3f} ms/frame ({num_frames/time_mv_dct_randn:.1f} FPS)")
    print(f"  Ratio:     {time_dct_randn/time_mv_dct_randn:.2f}x")
    
    # ========================================================================
    # TEST 2: Realistic ranges (matching actual data)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: Realistic data ranges (matching dataset)")
    print("="*80)
    
    # MV: mean=0.03, std=2.16, range [-15, 16]
    mv_realistic = torch.randn(1, 2, 60, 60).to(device) * 2.16 + 0.03
    mv_realistic = torch.clamp(mv_realistic, -15, 16)
    
    # DCT: mean=1.40, std=40.84, range [-485, 1872]
    dct_realistic = torch.randn(1, 120, 120, 8).to(device) * 40.84 + 1.40
    dct_realistic = torch.clamp(dct_realistic, -485, 1872)
    
    print(f"\nGenerated data ranges:")
    print(f"  MV:  [{mv_realistic.min():.2f}, {mv_realistic.max():.2f}], std={mv_realistic.std():.2f}")
    print(f"  DCT: [{dct_realistic.min():.2f}, {dct_realistic.max():.2f}], std={dct_realistic.std():.2f}")
    
    time_dct_realistic = benchmark_with_data(model_dct, None, dct_realistic, boxes, num_frames)
    time_mv_dct_realistic = benchmark_with_data(model_mv_dct, mv_realistic, dct_realistic, boxes, num_frames)
    
    print(f"\nResults with realistic ranges:")
    print(f"  DCT-only:  {(time_dct_realistic/num_frames)*1000:.3f} ms/frame ({num_frames/time_dct_realistic:.1f} FPS)")
    print(f"  MV+DCT:    {(time_mv_dct_realistic/num_frames)*1000:.3f} ms/frame ({num_frames/time_mv_dct_realistic:.1f} FPS)")
    print(f"  Ratio:     {time_dct_realistic/time_mv_dct_realistic:.2f}x")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON: Impact of data range")
    print("="*80)
    
    print(f"\nDCT-only model:")
    print(f"  With randn():    {(time_dct_randn/num_frames)*1000:.3f} ms/frame")
    print(f"  With realistic:  {(time_dct_realistic/num_frames)*1000:.3f} ms/frame")
    print(f"  Slowdown:        {time_dct_realistic/time_dct_randn:.2f}x")
    
    print(f"\nMV+DCT model:")
    print(f"  With randn():    {(time_mv_dct_randn/num_frames)*1000:.3f} ms/frame")
    print(f"  With realistic:  {(time_mv_dct_realistic/num_frames)*1000:.3f} ms/frame")
    print(f"  Slowdown:        {time_mv_dct_realistic/time_mv_dct_randn:.2f}x")
    
    print(f"\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    dct_slowdown = time_dct_realistic / time_dct_randn
    mv_dct_slowdown = time_mv_dct_realistic / time_mv_dct_randn
    
    if dct_slowdown > mv_dct_slowdown:
        print(f"\n🎯 KEY FINDING:")
        print(f"  DCT-only affected MORE by realistic data: {dct_slowdown:.2f}x vs {mv_dct_slowdown:.2f}x")
        print(f"  Difference: {dct_slowdown - mv_dct_slowdown:.2f}x")
        print(f"\n  This partially explains the paradox!")
        print(f"  Large DCT values (±1800) are harder to process than small MV values (±15)")
    else:
        print(f"\n  Both models affected similarly by data range")
    
    print(f"\nFull benchmark reported (for reference):")
    print(f"  DCT-only:  4.78 ms/frame (209 FPS)")
    print(f"  MV+DCT:    2.75 ms/frame (364 FPS)")
    print(f"  Ratio:     1.74x")


if __name__ == "__main__":
    main()
