#!/usr/bin/env python3
"""
Bounding Box Motion Vector Analyzer
Visualizes bounding boxes with their mean motion vectors to understand average motion within objects.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import argparse
from typing import List, Dict, Tuple, Optional

# Add dataset path for imports
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory


class BoundingBoxMotionAnalyzer:
    """Analyzes and visualizes mean motion vectors within bounding boxes."""
    
    def __init__(self, data_path: str = None):
        """Initialize the analyzer with data loader."""
        if data_path is None:
            data_path = os.path.join(os.path.dirname(__file__), 'dataset', 'data_prepared')
        
        self.data_path = data_path
        self.factory = MOTSDataLoaderFactory(verbose=True)
        self.data_loader = None
        
        # Colors for different objects
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
        ]
    
    def load_sequence(self, sequence_name: str = None) -> bool:
        """Load a specific sequence or the first available one."""
        try:
            # Get available sequences
            sequences = self.factory.list_sequences(['MOT17'], ['960x960'])
            if not sequences:
                print("❌ No sequences found!")
                return False
            
            # Use specified sequence or first available
            if sequence_name and sequence_name in sequences:
                target_sequence = sequence_name
            else:
                target_sequence = sequences[0]
                if sequence_name:
                    print(f"⚠️  Sequence '{sequence_name}' not found, using '{target_sequence}'")
            
            print(f"📂 Loading sequence: {target_sequence}")
            self.data_loader = self.factory.create_loader(target_sequence, ['MOT17'], ['960x960'])
            
            # Test data loading
            motion_data = self.data_loader.load_motion_vectors(0)  # Load GOP 0
            annotation_data = self.data_loader.load_corrected_annotations(0)  # Load GOP 0
            rgb_data = self.data_loader.load_rgb_frames(0, 'pframe')  # Load GOP 0
            
            print(f"✅ Sequence loaded successfully!")
            print(f"   📊 Motion vectors shape: {motion_data.shape}")
            if hasattr(annotation_data, 'files'):
                ann_shape = annotation_data['annotations'].shape
                print(f"   📋 Annotations shape: {ann_shape}")
            else:
                print(f"   📋 Annotations type: {type(annotation_data)}")
            print(f"   🖼️  RGB frames shape: {rgb_data.shape}")
            print(f"   🎯 Available GOPs: {len(self.data_loader.get_available_gops())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading sequence: {str(e)}")
            return False
    
    def calculate_bbox_mean_motion(self, frame_idx: int) -> List[Dict]:
        """Calculate mean motion vectors for each bounding box in a frame."""
        if not self.data_loader:
            return []
        
        try:
            # Convert frame index to GOP and frame within GOP
            gop_length = 49  # Standard GOP length
            gop_idx = frame_idx // gop_length
            frame_in_gop = frame_idx % gop_length
            
            # Load data for this GOP
            motion_data = self.data_loader.load_motion_vectors(gop_idx)
            annotation_data_file = self.data_loader.load_corrected_annotations(gop_idx)
            annotation_list = annotation_data_file['annotations'] if hasattr(annotation_data_file, 'files') else annotation_data_file
            
            if frame_in_gop >= len(annotation_list):
                print(f"⚠️  Frame {frame_idx} (GOP {gop_idx}, frame {frame_in_gop}) out of range for annotations")
                return []
            
            if frame_in_gop >= motion_data.shape[0]:
                print(f"⚠️  Frame {frame_idx} (GOP {gop_idx}, frame {frame_in_gop}) out of range for motion")
                return []
            
            frame_motion = motion_data[frame_in_gop]  # Shape: (2, 60, 60, 2)
            frame_annotations = annotation_list[frame_in_gop]  # Individual annotation matrix
            
            bbox_motions = []
            
            # Parse each annotation (format: [obj_id, obj_class, x_norm, y_norm, w_norm, h_norm])
            for ann in frame_annotations:
                if len(ann) >= 6:
                    obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                    
                    # Convert normalized coordinates to pixel coordinates
                    frame_width = frame_height = 960
                    x_center = x_norm * frame_width
                    y_center = y_norm * frame_height
                    width = w_norm * frame_width
                    height = h_norm * frame_height
                    
                    # Calculate bounding box corners
                    x_min = max(0, int(x_center - width/2))
                    y_min = max(0, int(y_center - height/2))
                    x_max = min(959, int(x_center + width/2))
                    y_max = min(959, int(y_center + height/2))
                    
                    # Convert to macroblock coordinates (motion vectors are at 60x60 resolution)
                    # Each macroblock represents 16x16 pixels, so scale factor is 60/960 = 1/16
                    mb_x_min = max(0, int(x_min / 16))
                    mb_y_min = max(0, int(y_min / 16))
                    mb_x_max = min(59, int(x_max / 16))
                    mb_y_max = min(59, int(y_max / 16))
                    
                    # Extract motion vectors within bounding box for both GOP layers
                    motion_in_bbox_layer0 = frame_motion[0, mb_y_min:mb_y_max+1, mb_x_min:mb_x_max+1, :]
                    motion_in_bbox_layer1 = frame_motion[1, mb_y_min:mb_y_max+1, mb_x_min:mb_x_max+1, :]
                    
                    # Calculate mean motion for each layer
                    mean_motion_layer0 = np.mean(motion_in_bbox_layer0.reshape(-1, 2), axis=0)
                    mean_motion_layer1 = np.mean(motion_in_bbox_layer1.reshape(-1, 2), axis=0)
                    
                    # Calculate overall mean motion
                    overall_mean_motion = (mean_motion_layer0 + mean_motion_layer1) / 2
                    
                    # Calculate motion magnitude
                    motion_magnitude = np.linalg.norm(overall_mean_motion)
                    
                    # Count non-zero motion vectors
                    non_zero_layer0 = np.sum(np.any(motion_in_bbox_layer0 != 0, axis=2))
                    non_zero_layer1 = np.sum(np.any(motion_in_bbox_layer1 != 0, axis=2))
                    total_pixels = motion_in_bbox_layer0.shape[0] * motion_in_bbox_layer0.shape[1]
                    
                    bbox_info = {
                        'object_id': int(obj_id),
                        'object_class': int(obj_class),
                        'bbox': (x_min, y_min, x_max, y_max),
                        'bbox_mb': (mb_x_min, mb_y_min, mb_x_max, mb_y_max),
                        'center': (x_center, y_center),
                        'size': (width, height),
                        'mean_motion': overall_mean_motion,
                        'mean_motion_layer0': mean_motion_layer0,
                        'mean_motion_layer1': mean_motion_layer1,
                        'motion_magnitude': motion_magnitude,
                        'non_zero_pixels_layer0': non_zero_layer0,
                        'non_zero_pixels_layer1': non_zero_layer1,
                        'total_pixels': total_pixels,
                        'motion_coverage_layer0': non_zero_layer0 / total_pixels if total_pixels > 0 else 0,
                        'motion_coverage_layer1': non_zero_layer1 / total_pixels if total_pixels > 0 else 0,
                        'gop_idx': gop_idx,
                        'frame_in_gop': frame_in_gop,
                    }
                    
                    bbox_motions.append(bbox_info)
            
            return bbox_motions
            
        except Exception as e:
            print(f"❌ Error calculating bbox motion for frame {frame_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def visualize_frame_with_motion(self, frame_idx: int, output_path: str = None, 
                                   motion_scale: float = 5.0, min_motion_threshold: float = 0.1) -> bool:
        """Visualize a frame with bounding boxes and their mean motion vectors."""
        if not self.data_loader:
            print("❌ No data loader available")
            return False
        
        try:
            # Convert frame index to GOP and frame within GOP
            gop_length = 49  # Standard GOP length
            gop_idx = frame_idx // gop_length
            frame_in_gop = frame_idx % gop_length
            
            # Get RGB frame
            rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
            if frame_in_gop >= rgb_data.shape[0]:
                print(f"⚠️  Frame {frame_idx} out of range")
                return False
            
            frame_rgb = rgb_data[frame_in_gop]
            
            # Calculate bounding box motions
            bbox_motions = self.calculate_bbox_mean_motion(frame_idx)
            
            if not bbox_motions:
                print(f"⚠️  No objects found in frame {frame_idx}")
                return False
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.imshow(frame_rgb)
            ax.set_title(f'Frame {frame_idx}: Bounding Boxes with Mean Motion Vectors\n'
                        f'Found {len(bbox_motions)} objects', fontsize=14, fontweight='bold')
            
            print(f"\n📊 Frame {frame_idx} Analysis:")
            print(f"   🎯 Objects detected: {len(bbox_motions)}")
            
            # Draw each bounding box and its motion
            for i, bbox_info in enumerate(bbox_motions):
                obj_id = bbox_info['object_id']
                x_min, y_min, x_max, y_max = bbox_info['bbox']
                center_x, center_y = bbox_info['center']
                mean_motion = bbox_info['mean_motion']
                motion_mag = bbox_info['motion_magnitude']
                
                # Choose color
                color = self.colors[i % len(self.colors)]
                color_norm = (color[0]/255, color[1]/255, color[2]/255)
                
                # Draw bounding box
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), bbox_width, bbox_height,
                                       linewidth=2, edgecolor=color_norm, facecolor='none')
                ax.add_patch(rect)
                
                # Add object ID label
                ax.text(x_min, y_min - 5, f'ID: {obj_id}', 
                       fontsize=10, fontweight='bold', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color_norm, alpha=0.8))
                
                # Draw motion vector if significant
                if motion_mag > min_motion_threshold:
                    # Scale motion vector for visibility
                    motion_x = mean_motion[0] * motion_scale
                    motion_y = mean_motion[1] * motion_scale
                    
                    # Draw arrow
                    arrow = FancyArrowPatch((center_x, center_y), 
                                          (center_x + motion_x, center_y + motion_y),
                                          arrowstyle='->', mutation_scale=20, 
                                          color=color_norm, linewidth=3, alpha=0.8)
                    ax.add_patch(arrow)
                    
                    # Add motion magnitude text
                    ax.text(center_x + motion_x/2, center_y + motion_y/2 - 10, 
                           f'{motion_mag:.2f}', 
                           fontsize=9, fontweight='bold', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Print detailed info
                print(f"\n   🔍 Object {obj_id}:")
                print(f"      📦 Bbox: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                print(f"      📍 Center: ({center_x:.1f}, {center_y:.1f})")
                print(f"      ➡️  Mean Motion: ({mean_motion[0]:.3f}, {mean_motion[1]:.3f})")
                print(f"      📏 Motion Magnitude: {motion_mag:.3f}")
                print(f"      📊 Layer 0 Motion Coverage: {bbox_info['motion_coverage_layer0']:.1%}")
                print(f"      📊 Layer 1 Motion Coverage: {bbox_info['motion_coverage_layer1']:.1%}")
            
            # Customize plot
            ax.set_xlim(0, frame_rgb.shape[1])
            ax.set_ylim(frame_rgb.shape[0], 0)  # Invert y-axis for image coordinates
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X coordinate', fontsize=12)
            ax.set_ylabel('Y coordinate', fontsize=12)
            
            # Add legend
            legend_text = f"Motion Scale: {motion_scale}x\nMin Threshold: {min_motion_threshold}\n"
            legend_text += f"Arrow length = motion magnitude × {motion_scale}"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
            
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"✅ Saved visualization to: {output_path}")
            else:
                plt.show()
            
            plt.close()
            return True
            
        except Exception as e:
            print(f"❌ Error visualizing frame {frame_idx}: {str(e)}")
            return False
    
    def analyze_multiple_frames(self, start_frame: int = 5, num_frames: int = 10, 
                               output_dir: str = "bbox_motion_analysis") -> bool:
        """Analyze multiple frames and create comprehensive visualization."""
        if not self.data_loader:
            print("❌ No data loader available")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🎬 Analyzing {num_frames} frames starting from frame {start_frame}")
        
        # Analyze each frame
        all_motions = []
        
        for i in range(num_frames):
            frame_idx = start_frame + i
            print(f"\n📍 Processing frame {frame_idx}...")
            
            # Create visualization for this frame
            output_path = os.path.join(output_dir, f"frame_{frame_idx:03d}_motion_analysis.png")
            success = self.visualize_frame_with_motion(frame_idx, output_path)
            
            if success:
                # Collect motion data
                bbox_motions = self.calculate_bbox_mean_motion(frame_idx)
                all_motions.extend(bbox_motions)
        
        # Create summary statistics
        if all_motions:
            self._create_motion_summary(all_motions, output_dir)
        
        print(f"\n✅ Analysis complete! Check results in: {output_dir}")
        return True
    
    def _create_motion_summary(self, all_motions: List[Dict], output_dir: str):
        """Create summary statistics and plots for all motion data."""
        try:
            # Extract motion magnitudes
            magnitudes = [m['motion_magnitude'] for m in all_motions]
            motion_x = [m['mean_motion'][0] for m in all_motions]
            motion_y = [m['mean_motion'][1] for m in all_motions]
            
            # Create summary plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Motion magnitude histogram
            ax1.hist(magnitudes, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_xlabel('Motion Magnitude')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Motion Magnitudes')
            ax1.grid(True, alpha=0.3)
            
            # Motion vector scatter plot
            ax2.scatter(motion_x, motion_y, alpha=0.6, s=50)
            ax2.set_xlabel('Motion X')
            ax2.set_ylabel('Motion Y')
            ax2.set_title('Motion Vector Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            
            # Motion magnitude over time (if we have frame info)
            object_ids = [m['object_id'] for m in all_motions]
            ax3.scatter(range(len(magnitudes)), magnitudes, c=object_ids, alpha=0.6, cmap='tab10')
            ax3.set_xlabel('Detection Index')
            ax3.set_ylabel('Motion Magnitude')
            ax3.set_title('Motion Magnitude Over Detections')
            ax3.grid(True, alpha=0.3)
            
            # Coverage statistics
            coverage_layer0 = [m['motion_coverage_layer0'] for m in all_motions]
            coverage_layer1 = [m['motion_coverage_layer1'] for m in all_motions]
            
            ax4.boxplot([coverage_layer0, coverage_layer1], labels=['Layer 0', 'Layer 1'])
            ax4.set_ylabel('Motion Coverage Ratio')
            ax4.set_title('Motion Coverage by GOP Layer')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save summary plot
            summary_path = os.path.join(output_dir, "motion_summary.png")
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create text summary
            summary_text = f"""
Motion Vector Analysis Summary
=============================

Total detections analyzed: {len(all_motions)}
Unique objects: {len(set(object_ids))}

Motion Statistics:
- Mean magnitude: {np.mean(magnitudes):.3f}
- Std magnitude: {np.std(magnitudes):.3f}
- Max magnitude: {np.max(magnitudes):.3f}
- Min magnitude: {np.min(magnitudes):.3f}

Motion Direction:
- Mean X motion: {np.mean(motion_x):.3f}
- Mean Y motion: {np.mean(motion_y):.3f}
- Std X motion: {np.std(motion_x):.3f}
- Std Y motion: {np.std(motion_y):.3f}

Coverage Statistics:
- Mean Layer 0 coverage: {np.mean(coverage_layer0):.1%}
- Mean Layer 1 coverage: {np.mean(coverage_layer1):.1%}

Objects with significant motion (>0.5): {len([m for m in magnitudes if m > 0.5])}
Objects with minimal motion (<0.1): {len([m for m in magnitudes if m < 0.1])}
"""
            
            summary_file = os.path.join(output_dir, "motion_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(summary_text)
            
            print(f"📊 Motion summary saved to: {summary_file}")
            print(f"📈 Summary plots saved to: {summary_path}")
            
        except Exception as e:
            print(f"❌ Error creating motion summary: {str(e)}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Analyze motion vectors within bounding boxes')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Sequence name to analyze (default: first available)')
    parser.add_argument('--start-frame', type=int, default=5,
                       help='Starting frame index (default: 5)')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames to analyze (default: 10)')
    parser.add_argument('--output-dir', type=str, default='bbox_motion_analysis',
                       help='Output directory (default: bbox_motion_analysis)')
    parser.add_argument('--single-frame', type=int, default=None,
                       help='Analyze only a single frame')
    parser.add_argument('--motion-scale', type=float, default=5.0,
                       help='Scale factor for motion vector arrows (default: 5.0)')
    parser.add_argument('--motion-threshold', type=float, default=0.1,
                       help='Minimum motion threshold to display arrows (default: 0.1)')
    
    args = parser.parse_args()
    
    print("🎯 Bounding Box Motion Vector Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = BoundingBoxMotionAnalyzer()
    
    # Load sequence
    if not analyzer.load_sequence(args.sequence):
        return 1
    
    # Analyze frames
    if args.single_frame is not None:
        # Single frame analysis
        output_path = os.path.join(args.output_dir, f"frame_{args.single_frame:03d}_motion_analysis.png")
        os.makedirs(args.output_dir, exist_ok=True)
        success = analyzer.visualize_frame_with_motion(
            args.single_frame, output_path, args.motion_scale, args.motion_threshold)
        if not success:
            return 1
    else:
        # Multiple frame analysis
        success = analyzer.analyze_multiple_frames(
            args.start_frame, args.num_frames, args.output_dir)
        if not success:
            return 1
    
    print("\n✅ Analysis completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
