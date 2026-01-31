#!/usr/bin/env python3
"""
Detailed motion analysis to understand why motion vectors are zero.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add path exactly like residuals study
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

def analyze_motion_in_detail():
    """Detailed analysis of motion vectors across the entire frame."""
    print("🔍 Detailed Motion Vector Analysis")
    print("=" * 50)
    
    # Initialize data loader
    factory = MOTSDataLoaderFactory(verbose=True)
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    sequence_name = sequences[0] if sequences else 'MOT17-09-SDP_960x960_gop50_500frames'
    data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
    
    print(f"✅ Using sequence: {sequence_name}")
    
    # Load data for multiple GOPs to find motion
    for gop_idx in range(min(3, 3)):  # Check first 3 GOPs
        print(f"\n🎯 GOP {gop_idx} Analysis:")
        
        motion_data = data_loader.load_motion_vectors(gop_idx)
        macroblock_data = data_loader.load_macroblocks(gop_idx)
        
        if motion_data is None:
            print(f"  ❌ No motion data for GOP {gop_idx}")
            continue
            
        if macroblock_data is None:
            print(f"  ❌ No macroblock data for GOP {gop_idx}")
            continue
        
        print(f"  📊 Motion data shape: {motion_data.shape}")
        print(f"  📊 Macroblock data shape: {macroblock_data.shape}")
        
        gop_frames = motion_data.shape[0]
        
        # Analyze entire frame for motion
        for frame_idx in range(min(10, gop_frames)):
            print(f"\n  📋 Frame {frame_idx}:")
            
            # Count block types across entire frame
            total_i_blocks = 0
            total_p_blocks = 0
            total_motion_magnitude = 0
            significant_motions = 0
            max_motion = 0
            max_motion_pos = None
            
            # Check entire 60x60 grid
            for mb_row in range(60):
                for mb_col in range(60):
                    # Check block type
                    mb_type = macroblock_data[frame_idx, mb_row, mb_col, 0]
                    
                    if mb_type == 0:
                        total_i_blocks += 1
                    else:
                        total_p_blocks += 1
                        
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
                    if max_magnitude > 0.5:
                        significant_motions += 1
                        
                    if max_magnitude > max_motion:
                        max_motion = max_magnitude
                        max_motion_pos = (mb_col, mb_row)
                        
            avg_motion = total_motion_magnitude / (60 * 60)
            i_block_percentage = (total_i_blocks / (60 * 60)) * 100
            
            print(f"    Block types: {total_i_blocks} I-blocks ({i_block_percentage:.1f}%), {total_p_blocks} P-blocks")
            print(f"    Motion: avg={avg_motion:.3f}, significant_motions={significant_motions}/3600")
            print(f"    Max motion: {max_motion:.3f} at position {max_motion_pos}")
            
            if significant_motions > 0:
                print(f"    🎉 Found {significant_motions} macroblocks with significant motion!")
                
                # Show details of top motion macroblocks
                motion_details = []
                for mb_row in range(60):
                    for mb_col in range(60):
                        mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
                        mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
                        mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
                        mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
                        
                        mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
                        mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
                        max_magnitude = max(mag_type0, mag_type1)
                        
                        if max_magnitude > 0.5:
                            mb_type = macroblock_data[frame_idx, mb_row, mb_col, 0]
                            block_type_str = "I-block" if mb_type == 0 else "P-block"
                            motion_details.append({
                                'pos': (mb_col, mb_row),
                                'magnitude': max_magnitude,
                                'type': block_type_str,
                                'mv_type0': (mv_x_type0, mv_y_type0),
                                'mv_type1': (mv_x_type1, mv_y_type1)
                            })
                
                # Sort by magnitude and show top 5
                motion_details.sort(key=lambda x: x['magnitude'], reverse=True)
                print(f"    Top motion macroblocks:")
                for i, detail in enumerate(motion_details[:5]):
                    pos = detail['pos']
                    mag = detail['magnitude']
                    mb_type = detail['type']
                    mv0 = detail['mv_type0']
                    mv1 = detail['mv_type1']
                    print(f"      {i+1}. MB({pos[0]},{pos[1]}): {mb_type}, mag={mag:.3f}, mv0={mv0}, mv1={mv1}")
                
                return True  # Found motion, can stop here
            
            elif frame_idx == 0 and total_i_blocks == 3600:
                print(f"    ⚠️  All blocks are I-blocks - typical for first frame of GOP")
            elif significant_motions == 0:
                print(f"    ⚠️  No significant motion detected")
        
        if significant_motions > 0:
            break  # Found motion in this GOP
    
    print(f"\n📝 Summary:")
    print(f"   - Searched through multiple GOPs and frames")
    print(f"   - Looking for motion vectors with magnitude > 0.5 pixels")
    print(f"   - This sequence appears to have very limited motion")
    print(f"   - Consider testing with a sequence that has more object movement")
    
    return False

def create_synthetic_motion_test():
    """Create a test with synthetic motion to verify our visualization works."""
    print(f"\n🧪 Creating Synthetic Motion Test")
    print("=" * 40)
    
    # This would create a test scenario with artificial motion
    # to verify our low macroblock count detection and new macroblock visualization
    
    print("To test the new features properly, we need:")
    print("1. A sequence with actual object motion")
    print("2. Or create synthetic motion data")
    print("3. Or manually trigger the low count scenario")
    
    return True

if __name__ == "__main__":
    found_motion = analyze_motion_in_detail()
    
    if not found_motion:
        print(f"\n💡 Recommendation:")
        print(f"   - The current sequence has minimal motion")
        print(f"   - Try a different sequence with more dynamic objects")
        print(f"   - Or we can create a synthetic test to verify the new features")
        
        create_synthetic_motion_test()
