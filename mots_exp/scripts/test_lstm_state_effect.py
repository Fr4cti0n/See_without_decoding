"""
Test if LSTM hidden state carry-over causes the speed difference
"""

import torch
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.dct_mv_tracker import DCTMVCenterTracker


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test parameters matching actual benchmark
    num_objects = 5
    num_frames = 47  # GOP length minus I-frame
    
    # Create dummy inputs
    mv_input = torch.randn(1, 2, 60, 60).to(device)
    dct_input = torch.randn(1, 120, 120, 8).to(device)
    boxes = torch.tensor([[100, 100, 200, 200],
                          [300, 300, 400, 400],
                          [500, 500, 600, 600],
                          [700, 700, 800, 800],
                          [150, 150, 250, 250]], dtype=torch.float32).to(device)
    
    print(f"Test settings (matching actual benchmark):")
    print(f"  Num objects: {num_objects}")
    print(f"  Num frames: {num_frames}")
    print(f"  GOP sequence processing with LSTM state carry-over\n")
    
    # ========================================================================
    # TEST 1: DCT-only model
    # ========================================================================
    print("="*80)
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
    
    with torch.no_grad():
        # Warmup
        hidden_state = None
        for t in range(10):
            _, _, hidden_state = model_dct.forward_single_frame(None, dct_input, boxes, hidden_state)
        
        # Actual test: Process full GOP sequence
        torch.cuda.synchronize()
        start_time = time.time()
        
        hidden_state = None
        for t in range(num_frames):
            _, _, hidden_state = model_dct.forward_single_frame(None, dct_input, boxes, hidden_state)
        
        torch.cuda.synchronize()
        elapsed_dct = time.time() - start_time
        
    print(f"\nResults:")
    print(f"  Total time: {elapsed_dct*1000:.2f} ms for {num_frames} frames")
    print(f"  Time per frame: {(elapsed_dct/num_frames)*1000:.3f} ms")
    print(f"  FPS: {num_frames/elapsed_dct:.1f}")
    
    # ========================================================================
    # TEST 2: MV+DCT model
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
    
    with torch.no_grad():
        # Warmup
        hidden_state = None
        for t in range(10):
            _, _, hidden_state = model_mv_dct.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        # Actual test: Process full GOP sequence
        torch.cuda.synchronize()
        start_time = time.time()
        
        hidden_state = None
        for t in range(num_frames):
            _, _, hidden_state = model_mv_dct.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        torch.cuda.synchronize()
        elapsed_mv_dct = time.time() - start_time
        
    print(f"\nResults:")
    print(f"  Total time: {elapsed_mv_dct*1000:.2f} ms for {num_frames} frames")
    print(f"  Time per frame: {(elapsed_mv_dct/num_frames)*1000:.3f} ms")
    print(f"  FPS: {num_frames/elapsed_mv_dct:.1f}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nPer-frame timing:")
    print(f"  DCT-only:  {(elapsed_dct/num_frames)*1000:.3f} ms/frame ({num_frames/elapsed_dct:.1f} FPS)")
    print(f"  MV+DCT:    {(elapsed_mv_dct/num_frames)*1000:.3f} ms/frame ({num_frames/elapsed_mv_dct:.1f} FPS)")
    print(f"  Ratio:     {elapsed_dct / elapsed_mv_dct:.2f}x")
    
    if elapsed_dct > elapsed_mv_dct:
        print(f"\n⚠️  PARADOX: MV+DCT is {elapsed_dct/elapsed_mv_dct:.2f}x faster!")
        print(f"  This matches the full benchmark results!")
    else:
        print(f"\n✅ Expected: DCT-only is {elapsed_mv_dct/elapsed_dct:.2f}x faster")
    
    # Now test WITHOUT hidden state carry-over (reset each frame)
    print("\n" + "="*80)
    print("CONTROL TEST: Without LSTM state carry-over")
    print("="*80)
    
    with torch.no_grad():
        # DCT-only (no state carry)
        torch.cuda.synchronize()
        start_time = time.time()
        
        for t in range(num_frames):
            _, _, _ = model_dct.forward_single_frame(None, dct_input, boxes, None)  # Reset state each time
        
        torch.cuda.synchronize()
        elapsed_dct_no_state = time.time() - start_time
        
        # MV+DCT (no state carry)
        torch.cuda.synchronize()
        start_time = time.time()
        
        for t in range(num_frames):
            _, _, _ = model_mv_dct.forward_single_frame(mv_input, dct_input, boxes, None)  # Reset state each time
        
        torch.cuda.synchronize()
        elapsed_mv_dct_no_state = time.time() - start_time
    
    print(f"\nWithout LSTM state carry-over:")
    print(f"  DCT-only:  {(elapsed_dct_no_state/num_frames)*1000:.3f} ms/frame ({num_frames/elapsed_dct_no_state:.1f} FPS)")
    print(f"  MV+DCT:    {(elapsed_mv_dct_no_state/num_frames)*1000:.3f} ms/frame ({num_frames/elapsed_mv_dct_no_state:.1f} FPS)")
    print(f"  Ratio:     {elapsed_dct_no_state / elapsed_mv_dct_no_state:.2f}x")
    
    print(f"\nEffect of LSTM state carry-over:")
    print(f"  DCT-only slowdown:  {(elapsed_dct/elapsed_dct_no_state):.2f}x")
    print(f"  MV+DCT slowdown:    {(elapsed_mv_dct/elapsed_mv_dct_no_state):.2f}x")
    
    if (elapsed_dct/elapsed_dct_no_state) > (elapsed_mv_dct/elapsed_mv_dct_no_state):
        diff = (elapsed_dct/elapsed_dct_no_state) - (elapsed_mv_dct/elapsed_mv_dct_no_state)
        print(f"\n🔍 KEY FINDING: DCT-only is affected {diff:.2f}x MORE by LSTM state carry-over!")
        print(f"   This could explain the paradox!")


if __name__ == "__main__":
    main()
