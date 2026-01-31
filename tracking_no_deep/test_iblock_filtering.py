#!/usr/bin/env python3
"""
Test script to verify I-block filtering and motion vector processing.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add path exactly like residuals study
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

def analyze_motion_and_blocks():
    """Analyze motion vectors and block types to verify I-block filtering."""
    print("🔍 Analyzing Motion Vectors and Block Types")
    print("=" * 50)
    
    # Initialize data loader
    factory = MOTSDataLoaderFactory(verbose=True)
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    sequence_name = sequences[0] if sequences else 'MOT17-09-SDP_960x960_gop50_500frames'
    data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
    
    print(f"✅ Using sequence: {sequence_name}")
    
    # Load data for GOP 0
    gop_idx = 0
    motion_data = data_loader.load_motion_vectors(gop_idx)
    macroblock_data = data_loader.load_macroblocks(gop_idx)
    
    if motion_data is None:
        print("❌ No motion data available")
        return
        
    if macroblock_data is None:
        print("❌ No macroblock data available")
        return
    
    print(f"📊 Motion data shape: {motion_data.shape}")
    print(f"📊 Macroblock data shape: {macroblock_data.shape}")
    
    gop_frames = motion_data.shape[0]
    
    # Analyze first few frames
    for frame_idx in range(min(5, gop_frames)):
        print(f"\n📋 Frame {frame_idx}:")
        
        # Count block types
        i_blocks = 0
        p_blocks = 0
        total_motion_magnitude = 0
        non_zero_motions = 0
        
        for mb_row in range(5):  # Check first 5 rows
            for mb_col in range(5):  # Check first 5 columns
                # Check block type
                mb_type = macroblock_data[frame_idx, mb_row, mb_col, 0]
                
                if mb_type == 0:
                    i_blocks += 1
                else:
                    p_blocks += 1
                    
                # Get motion vectors
                mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
                mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
                mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
                mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
                
                # Calculate magnitudes
                mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
                mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
                max_magnitude = max(mag_type0, mag_type1)
                
                total_motion_magnitude += max_magnitude
                if max_magnitude > 0.1:
                    non_zero_motions += 1
                    
                # Print details for first few macroblocks
                if mb_row < 2 and mb_col < 2:
                    block_type_str = "I-block" if mb_type == 0 else "P-block"
                    print(f"  MB({mb_col},{mb_row}): {block_type_str}, "
                          f"type0_mag={mag_type0:.2f}, type1_mag={mag_type1:.2f}, max={max_magnitude:.2f}")
        
        avg_motion = total_motion_magnitude / 25  # 5x5 = 25 macroblocks
        print(f"  Block types: {i_blocks} I-blocks, {p_blocks} P-blocks")
        print(f"  Motion: avg={avg_motion:.3f}, non-zero={non_zero_motions}/25")
        
        if i_blocks == 25:
            print("  ⚠️  All blocks are I-blocks - this explains zero motion!")
        elif non_zero_motions == 0:
            print("  ⚠️  No motion detected - object might be stationary")
    
    # Test our I-block filtering logic
    print(f"\n🧪 Testing I-block filtering logic:")
    test_frame = 0
    test_mb_row, test_mb_col = 0, 0
    
    # Get motion vectors
    mv_x_type0 = float(motion_data[test_frame, 0, test_mb_row, test_mb_col, 0])
    mv_y_type0 = float(motion_data[test_frame, 0, test_mb_row, test_mb_col, 1])
    mv_x_type1 = float(motion_data[test_frame, 1, test_mb_row, test_mb_col, 0])
    mv_y_type1 = float(motion_data[test_frame, 1, test_mb_row, test_mb_col, 1])
    
    # Check block type
    mb_type = macroblock_data[test_frame, test_mb_row, test_mb_col, 0]
    is_i_block = (mb_type == 0)
    
    print(f"  Test macroblock ({test_mb_col},{test_mb_row}) in frame {test_frame}:")
    print(f"    Block type: {'I-block' if is_i_block else 'P-block'}")
    print(f"    Original motion vectors: type0=({mv_x_type0:.3f}, {mv_y_type0:.3f}), type1=({mv_x_type1:.3f}, {mv_y_type1:.3f})")
    
    # Apply our filtering logic
    if is_i_block:
        # Force zero motion for I-blocks
        filtered_mv_x, filtered_mv_y = 0.0, 0.0
        motion_type = -1
        print(f"    Filtered motion (I-block): ({filtered_mv_x:.3f}, {filtered_mv_y:.3f}) [FORCED TO ZERO]")
    else:
        # Choose motion vector with larger magnitude for P-blocks
        mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
        mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
        
        if mag_type1 > mag_type0:
            filtered_mv_x, filtered_mv_y = mv_x_type1, mv_y_type1
            motion_type = 1
        else:
            filtered_mv_x, filtered_mv_y = mv_x_type0, mv_y_type0
            motion_type = 0
        
        print(f"    Filtered motion (P-block): ({filtered_mv_x:.3f}, {filtered_mv_y:.3f}) [type={motion_type}]")
    
    print(f"\n✅ I-block filtering analysis completed!")
    print(f"📝 Summary:")
    print(f"   - I-blocks have motion vectors forced to zero")
    print(f"   - P-blocks use motion vector with highest magnitude")
    print(f"   - If all blocks are I-blocks, no motion will be detected")
    print(f"   - This is expected behavior for stationary objects or intra-coded frames")

if __name__ == "__main__":
    analyze_motion_and_blocks()
