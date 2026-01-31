#!/usr/bin/env python3
"""
Test script to generate only the cumulative temporal profile video
"""

import sys
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from pathlib import Path
from core.data_loader import MOTSDataLoaderFactory
from dataset.visualization_toolkit.residuals_study.generate_all_residual_video import ResidualVideoGenerator

def main():
    print("Testing Enhanced Cumulative Temporal Profile Video Generation")
    print("=" * 60)
    
    # Create factory and data loader
    factory = MOTSDataLoaderFactory(verbose=False)
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    sequence_name = sequences[0]
    data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
    
    print(f"Using sequence: {sequence_name}")
    
    # Create video generator
    output_dir = Path("outputs")
    video_generator = ResidualVideoGenerator(data_loader, output_dir)
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Frame rate: {video_generator.fps} fps")
    print()
    
    # Generate only cumulative temporal profile video
    print("Generating Enhanced Cumulative Temporal Profile Video...")
    if video_generator.generate_cumulative_temporal_video(max_frames=25):
        print("✓ Enhanced cumulative temporal profile video generated successfully!")
    else:
        print("✗ Failed to generate cumulative temporal profile video")
    
    print("\nTest complete!")
    print("Check: outputs/residual_cumulative_temporal_profile.mp4")

if __name__ == "__main__":
    main()
