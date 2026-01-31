#!/usr/bin/env python3
"""
Test script for technique comparison video with real temporal analysis
"""

import sys
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/MOTS-experiments')

import os
from pathlib import Path
from dataset.visualization_toolkit.residuals_study.generate_all_residual_video import ResidualVideoGenerator

def main():
    print("Testing Technique Comparison Video with Real Temporal Analysis")
    print("=" * 65)
    
    # Use the data path directly
    data_path = '/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/MOTS-experiments/dataset/kitti_mots/training/images'
    
    print(f"Data path: {data_path}")
    
    # Create video generator
    output_dir = Path("outputs")
    video_generator = ResidualVideoGenerator(data_path, output_dir, fps=3)
    
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Frame rate: {video_generator.fps} fps")
    print()
    
    # Generate technique comparison video
    print("Generating Technique Comparison Video...")
    success = video_generator.generate_technique_comparison_video()
    if success:
        print("✓ Technique comparison video generated successfully!")
        print("  File: outputs/residual_technique_comparison.mp4")
        print("  Features:")
        print("  - Instantaneous P-Block Residuals (cyan annotations)")
        print("  - Rolling Window Analysis (yellow annotations)")
        print("  - Cumulative Integration (magenta annotations)")
        print("  - RGB with Annotations (lime annotations)")
        print("  - Real temporal analysis (NO synthetic data)")
    else:
        print("✗ Failed to generate technique comparison video")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()
