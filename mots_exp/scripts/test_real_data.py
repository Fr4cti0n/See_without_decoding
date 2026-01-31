"""
Reproduce the exact benchmark conditions with real data loading
This should finally reproduce the paradox
"""

import torch
import time
import numpy as np
from pathlib import Path
import sys
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.dct_mv_tracker import DCTMVCenterTracker
from mots_exp.dataset.mots_exp_dataset import MOTSExperimentDataset
from torch.utils.data import DataLoader, Subset


def load_model(checkpoint_path, device):
    """Load model from checkpoint - EXACT copy from estimate_model_speed.py"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    ckpt_name = str(checkpoint_path).lower()
    if 'mv_dct' in ckpt_name:
        mv_channels = 2
        if 'dct_8' in ckpt_name:
            dct_channels = 8
        else:
            dct_channels = 8
    elif 'dct' in ckpt_name:
        mv_channels = 0
        if 'dct_8' in ckpt_name:
            dct_channels = 8
        else:
            dct_channels = 8
    else:
        mv_channels = 2
        dct_channels = 0
    
    num_dct_coeffs = dct_channels if dct_channels > 0 else 16
    
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
    
    return model, mv_channels, dct_channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dct-model', required=True)
    parser.add_argument('--mv-dct-model', required=True)
    parser.add_argument('--dataset-dir', default='/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/MOTS-experiments/data/MOTS_data')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    print(f"Using device: {device}\n")
    
    # Create dataset - EXACT same as benchmark
    print("Loading dataset...")
    dataset = MOTSExperimentDataset(
        data_dir=args.dataset_dir,
        split='val',
        mode='track',
        gop_length=48,
        image_size=960
    )
    
    # Select first GOP for testing
    first_gop_frames = []
    base_dataset = dataset.dataset
    seq_info = base_dataset.sequences[0]
    
    print(f"Loading GOP from: {seq_info['video_name']}, GOP {seq_info['gop_index']}")
    
    for frame_idx in range(48):
        global_idx = frame_idx
        sample = base_dataset[global_idx]
        mv = sample.get('motion_vectors')
        boxes = sample.get('boxes')
        residuals = sample.get('residuals')
        
        if boxes is None or (mv is None and residuals is None):
            continue
        
        frame_data = {
            'motion_vectors': mv,
            'boxes': boxes,
            'frame_id': frame_idx
        }
        if residuals is not None:
            frame_data['residuals'] = residuals
        
        first_gop_frames.append(frame_data)
    
    print(f"Loaded {len(first_gop_frames)} frames\n")
    
    # ========================================================================
    # TEST DCT-only - EXACT data prep from benchmark
    # ========================================================================
    print("="*80)
    print("TEST 1: DCT-only model with REAL data")
    print("="*80)
    
    model_dct, mv_ch, dct_ch = load_model(args.dct_model, device)
    
    # Data prep - EXACT copy from estimate_model_speed.py
    iframe_boxes = first_gop_frames[0]['boxes'].to(device)
    N = len(iframe_boxes)
    T = len(first_gop_frames) - 1
    
    # Collect DCT residuals
    dcts = []
    for t in range(1, len(first_gop_frames)):
        dct = first_gop_frames[t].get('residuals', None)
        if dct is not None:
            if dct.ndim == 4:
                dct = dct.squeeze(0)
            dct = dct[..., :dct_ch]
            dcts.append(dct.to(device))
    
    # Stack
    dct_seq = torch.stack(dcts).unsqueeze(1)  # [T, 1, 120, 120, 8]
    
    print(f"Data prepared:")
    print(f"  Objects: {N}")
    print(f"  Frames: {T}")
    print(f"  DCT shape: {dct_seq.shape}")
    
    # Benchmark - EXACT timing from estimate_model_speed.py
    with torch.no_grad():
        # Warmup
        hidden_state = None
        for t in range(5):
            dct_frame = dct_seq[t]
            boxes_frame = iframe_boxes
            _, _, hidden_state = model_dct.forward_single_frame(
                None, dct_frame, boxes_frame, hidden_state
            )
        
        # Actual benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        hidden_state = None
        for t in range(T):
            dct_frame = dct_seq[t]
            boxes_frame = iframe_boxes
            _, _, hidden_state = model_dct.forward_single_frame(
                None, dct_frame, boxes_frame, hidden_state
            )
        
        torch.cuda.synchronize()
        elapsed_dct = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed_dct*1000:.2f} ms")
    print(f"  Time per frame: {(elapsed_dct/T)*1000:.3f} ms")
    print(f"  FPS: {T/elapsed_dct:.1f}")
    
    # ========================================================================
    # TEST MV+DCT - EXACT data prep from benchmark
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: MV+DCT model with REAL data")
    print("="*80)
    
    model_mv_dct, mv_ch, dct_ch = load_model(args.mv_dct_model, device)
    
    # Data prep - EXACT copy from estimate_model_speed.py
    iframe_boxes = first_gop_frames[0]['boxes'].to(device)
    
    # Collect MVs and DCTs
    mvs = []
    dcts = []
    for t in range(1, len(first_gop_frames)):
        # MV
        mv = first_gop_frames[t]['motion_vectors'].to(device)
        if mv.ndim == 4 and mv.shape[-1] == 2:
            mv = mv[..., 0]
        mvs.append(mv)
        
        # DCT
        dct = first_gop_frames[t].get('residuals', None)
        if dct is not None:
            if dct.ndim == 4:
                dct = dct.squeeze(0)
            dct = dct[..., :dct_ch]
            dcts.append(dct.to(device))
    
    # Stack
    mv_seq = torch.stack(mvs).unsqueeze(1)  # [T, 1, 2, 60, 60]
    dct_seq = torch.stack(dcts).unsqueeze(1)  # [T, 1, 120, 120, 8]
    
    print(f"Data prepared:")
    print(f"  Objects: {N}")
    print(f"  Frames: {T}")
    print(f"  MV shape: {mv_seq.shape}")
    print(f"  DCT shape: {dct_seq.shape}")
    
    # Benchmark
    with torch.no_grad():
        # Warmup
        hidden_state = None
        for t in range(5):
            mv_frame = mv_seq[t]
            dct_frame = dct_seq[t]
            boxes_frame = iframe_boxes
            _, _, hidden_state = model_mv_dct.forward_single_frame(
                mv_frame, dct_frame, boxes_frame, hidden_state
            )
        
        # Actual benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        hidden_state = None
        for t in range(T):
            mv_frame = mv_seq[t]
            dct_frame = dct_seq[t]
            boxes_frame = iframe_boxes
            _, _, hidden_state = model_mv_dct.forward_single_frame(
                mv_frame, dct_frame, boxes_frame, hidden_state
            )
        
        torch.cuda.synchronize()
        elapsed_mv_dct = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed_mv_dct*1000:.2f} ms")
    print(f"  Time per frame: {(elapsed_mv_dct/T)*1000:.3f} ms")
    print(f"  FPS: {T/elapsed_mv_dct:.1f}")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON WITH REAL DATA")
    print("="*80)
    
    print(f"\nFull benchmark results (for comparison):")
    print(f"  DCT-only:  4.78 ms/frame (209 FPS)")
    print(f"  MV+DCT:    2.75 ms/frame (364 FPS)")
    print(f"  Ratio:     1.74x (MV+DCT faster)")
    
    print(f"\nThis test results:")
    print(f"  DCT-only:  {(elapsed_dct/T)*1000:.3f} ms/frame ({T/elapsed_dct:.1f} FPS)")
    print(f"  MV+DCT:    {(elapsed_mv_dct/T)*1000:.3f} ms/frame ({T/elapsed_mv_dct:.1f} FPS)")
    print(f"  Ratio:     {elapsed_dct/elapsed_mv_dct:.2f}x")
    
    if elapsed_dct > elapsed_mv_dct:
        print(f"\n⚠️  PARADOX REPRODUCED!")
    else:
        print(f"\n✅ Normal behavior (DCT-only faster)")


if __name__ == "__main__":
    main()
