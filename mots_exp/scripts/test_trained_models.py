"""
Test ACTUAL trained models to see if there's a model-specific issue
"""

import torch
import time
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.dct_mv_tracker import DCTMVCenterTracker


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config from checkpoint
    config = checkpoint.get('model_config', {})
    
    # Check model variant from checkpoint path
    ckpt_name = str(checkpoint_path).lower()
    if 'mv_dct' in ckpt_name:
        # MV+DCT variant
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
            dct_channels = 8  # Default
    elif 'dct' in ckpt_name:
        # DCT-only variant
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
            dct_channels = 8  # Default
    else:
        # MV-only variant
        mv_channels = 2
        dct_channels = 0
    
    num_dct_coeffs = dct_channels if dct_channels > 0 else 16
    
    print(f"Loading model: MV={mv_channels} ch, DCT={dct_channels} ch, num_dct_coeffs={num_dct_coeffs}")
    
    # Create model
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
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    
    model = model.to(device).eval()
    
    return model, mv_channels, dct_channels


def benchmark_model(model, mv_input, dct_input, boxes, num_frames=47):
    """Benchmark a model on sequence with LSTM state carry-over"""
    with torch.no_grad():
        # Warmup
        hidden_state = None
        for t in range(10):
            _, _, hidden_state = model.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        # Actual test
        torch.cuda.synchronize()
        start_time = time.time()
        
        hidden_state = None
        for t in range(num_frames):
            _, _, hidden_state = model.forward_single_frame(mv_input, dct_input, boxes, hidden_state)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dct-model', required=True, help='Path to DCT-only model checkpoint')
    parser.add_argument('--mv-dct-model', required=True, help='Path to MV+DCT model checkpoint')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Test parameters
    num_objects = 5
    num_frames = 47
    
    # Create dummy inputs
    mv_input = torch.randn(1, 2, 60, 60).to(device)
    dct_input = torch.randn(1, 120, 120, 8).to(device)
    boxes = torch.tensor([[100, 100, 200, 200],
                          [300, 300, 400, 400],
                          [500, 500, 600, 600],
                          [700, 700, 800, 800],
                          [150, 150, 250, 250]], dtype=torch.float32).to(device)
    
    print(f"Test settings:")
    print(f"  Num objects: {num_objects}")
    print(f"  Num frames: {num_frames}\n")
    
    # ========================================================================
    # Load DCT-only model
    # ========================================================================
    print("="*80)
    print("Loading DCT-only model")
    print("="*80)
    
    model_dct, mv_ch_dct, dct_ch_dct = load_model(args.dct_model, device)
    
    print(f"\nBenchmarking DCT-only model...")
    elapsed_dct = benchmark_model(model_dct, None, dct_input, boxes, num_frames)
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed_dct*1000:.2f} ms for {num_frames} frames")
    print(f"  Time per frame: {(elapsed_dct/num_frames)*1000:.3f} ms")
    print(f"  FPS: {num_frames/elapsed_dct:.1f}")
    
    # ========================================================================
    # Load MV+DCT model
    # ========================================================================
    print("\n" + "="*80)
    print("Loading MV+DCT model")
    print("="*80)
    
    model_mv_dct, mv_ch_mvdct, dct_ch_mvdct = load_model(args.mv_dct_model, device)
    
    print(f"\nBenchmarking MV+DCT model...")
    elapsed_mv_dct = benchmark_model(model_mv_dct, mv_input, dct_input, boxes, num_frames)
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed_mv_dct*1000:.2f} ms for {num_frames} frames")
    print(f"  Time per frame: {(elapsed_mv_dct/num_frames)*1000:.3f} ms")
    print(f"  FPS: {num_frames/elapsed_mv_dct:.1f}")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    print(f"\nUsing ACTUAL trained models:")
    print(f"  DCT-only:  {(elapsed_dct/num_frames)*1000:.3f} ms/frame ({num_frames/elapsed_dct:.1f} FPS)")
    print(f"  MV+DCT:    {(elapsed_mv_dct/num_frames)*1000:.3f} ms/frame ({num_frames/elapsed_mv_dct:.1f} FPS)")
    print(f"  Ratio:     {elapsed_dct / elapsed_mv_dct:.2f}x")
    
    if elapsed_dct > elapsed_mv_dct:
        print(f"\n⚠️  PARADOX REPRODUCED: MV+DCT is {elapsed_dct/elapsed_mv_dct:.2f}x faster!")
    else:
        print(f"\n✅ Expected: DCT-only is {elapsed_mv_dct/elapsed_dct:.2f}x faster")


if __name__ == "__main__":
    main()
