#!/usr/bin/env python3
"""
Motion Vector Field Visualizer
Creates a video showing every motion vector as arrows across the entire frame for a GOP.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import argparse
from typing import List, Dict, Tuple, Optional
import time

# Add path for imports
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory


class MotionVectorFieldVisualizer:
    """Visualizes complete motion vector fields as arrow overlays on video frames."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.factory = MOTSDataLoaderFactory(verbose=True)
        self.data_loader = None
        
        # Visualization parameters
        self.arrow_scale = 3.0  # Scale factor for arrow lengths
        self.arrow_subsample = 2  # Show every Nth vector (1=all, 2=every other, etc.)
        self.min_magnitude_threshold = 0.1  # Minimum magnitude to display arrow
        self.arrow_alpha = 0.7  # Arrow transparency
        
        # Colors for different GOP layers
        self.layer_colors = {
            0: (1.0, 0.0, 0.0),  # Red for layer 0
            1: (0.0, 0.0, 1.0),  # Blue for layer 1
        }
    
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
            motion_data = self.data_loader.load_motion_vectors(0)
            rgb_data = self.data_loader.load_rgb_frames(0, 'pframe')
            
            print(f"✅ Sequence loaded successfully!")
            print(f"   📊 Motion vectors shape: {motion_data.shape}")
            print(f"   🖼️  RGB frames shape: {rgb_data.shape}")
            print(f"   🎯 Available GOPs: {len(self.data_loader.get_available_gops())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading sequence: {str(e)}")
            return False
    
    def create_motion_field_frame(self, rgb_frame: np.ndarray, motion_data: np.ndarray, 
                                 frame_idx: int, layer_idx: int = None) -> np.ndarray:
        """Create a frame with motion vectors overlaid as arrows."""
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(rgb_frame)
            
            # Get motion vector dimensions
            if layer_idx is not None:
                # Single layer
                motion_layer = motion_data[layer_idx]  # Shape: (60, 60, 2)
                layers_to_process = [(layer_idx, motion_layer)]
                title_suffix = f" - Layer {layer_idx}"
            else:
                # Both layers
                layers_to_process = [(0, motion_data[0]), (1, motion_data[1])]
                title_suffix = " - Both Layers"
            
            total_arrows = 0
            significant_arrows = 0
            
            for layer_id, motion_layer in layers_to_process:
                color = self.layer_colors[layer_id]
                
                # Get motion field dimensions
                h, w = motion_layer.shape[:2]
                
                # Calculate scaling from motion grid to image coordinates
                scale_y = rgb_frame.shape[0] / h  # 960 / 60 = 16
                scale_x = rgb_frame.shape[1] / w  # 960 / 60 = 16
                
                # Process motion vectors with subsampling
                for y in range(0, h, self.arrow_subsample):
                    for x in range(0, w, self.arrow_subsample):
                        # Get motion vector
                        mv_x = motion_layer[y, x, 0]
                        mv_y = motion_layer[y, x, 1]
                        
                        # Calculate magnitude
                        magnitude = np.sqrt(mv_x**2 + mv_y**2)
                        
                        total_arrows += 1
                        
                        # Skip if below threshold
                        if magnitude < self.min_magnitude_threshold:
                            continue
                        
                        significant_arrows += 1
                        
                        # Convert to image coordinates
                        start_x = x * scale_x + scale_x / 2  # Center of macroblock
                        start_y = y * scale_y + scale_y / 2
                        
                        # Scale motion vector for visibility
                        end_x = start_x + mv_x * self.arrow_scale
                        end_y = start_y + mv_y * self.arrow_scale
                        
                        # Create arrow
                        arrow = FancyArrowPatch(
                            (start_x, start_y), (end_x, end_y),
                            arrowstyle='->', 
                            mutation_scale=15,
                            color=color,
                            alpha=self.arrow_alpha,
                            linewidth=1.5
                        )
                        ax.add_patch(arrow)
                        
                        # Add magnitude text for very significant vectors
                        if magnitude > 2.0:
                            ax.text(end_x + 5, end_y - 5, f'{magnitude:.1f}', 
                                   fontsize=8, color=color, 
                                   bbox=dict(boxstyle="round,pad=0.2", 
                                           facecolor='white', alpha=0.6))
            
            # Set title and labels
            ax.set_title(f'Motion Vector Field - Frame {frame_idx}{title_suffix}\n'
                        f'Showing {significant_arrows}/{total_arrows} vectors (magnitude > {self.min_magnitude_threshold})',
                        fontsize=14, fontweight='bold')
            
            ax.set_xlim(0, rgb_frame.shape[1])
            ax.set_ylim(rgb_frame.shape[0], 0)  # Invert y-axis
            ax.axis('off')
            
            # Add legend
            if layer_idx is None:
                legend_elements = [
                    plt.Line2D([0], [0], color=self.layer_colors[0], lw=3, label='Layer 0'),
                    plt.Line2D([0], [0], color=self.layer_colors[1], lw=3, label='Layer 1')
                ]
                ax.legend(handles=legend_elements, loc='upper right')
            
            # Add parameter info
            info_text = f"Scale: {self.arrow_scale}x | Subsample: {self.arrow_subsample} | Threshold: {self.min_magnitude_threshold}"
            ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            # Convert to numpy array
            fig.canvas.draw()
            
            # Get the image buffer - use the correct method for newer matplotlib
            try:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            except AttributeError:
                # Newer matplotlib versions use buffer_rgba
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                buf = buf[:, :, :3]  # Remove alpha channel
            else:
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            
            return buf
            
        except Exception as e:
            print(f"❌ Error creating motion field frame {frame_idx}: {str(e)}")
            return rgb_frame
    
    def create_motion_field_video(self, gop_idx: int = 0, output_path: str = "motion_field_complete.mp4",
                                 show_both_layers: bool = True, max_frames: int = None) -> bool:
        """Create a video showing motion vector fields for an entire GOP."""
        if not self.data_loader:
            print("❌ No data loader available")
            return False
        
        try:
            print(f"\n🎬 Creating motion vector field video for GOP {gop_idx}")
            
            # Load GOP data
            motion_data = self.data_loader.load_motion_vectors(gop_idx)
            rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
            
            if motion_data is None or rgb_data is None:
                print(f"❌ No data available for GOP {gop_idx}")
                return False
            
            # Process entire GOP unless max_frames is specified
            total_frames = motion_data.shape[0]
            frames_to_process = min(max_frames, total_frames) if max_frames else total_frames
            
            print(f"   📊 Motion data shape: {motion_data.shape}")
            print(f"   🖼️  RGB data shape: {rgb_data.shape}")
            print(f"   🎯 Processing {frames_to_process} frames out of {total_frames} total frames")
            print(f"   📁 Output: {output_path}")
            
            # Setup video writer
            height, width = 960, 960
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 15.0, (width, height))
            
            # Process each frame
            for frame_idx in range(frames_to_process):
                print(f"   📍 Processing frame {frame_idx+1}/{frames_to_process} (GOP frame {frame_idx})")
                
                # Handle case where RGB frames might be fewer than motion frames
                rgb_frame_idx = min(frame_idx, rgb_data.shape[0] - 1)
                rgb_frame = rgb_data[rgb_frame_idx]
                
                # Get motion vectors for this frame
                motion_layer0 = motion_data[frame_idx, 0]  # Shape: (60, 60, 2)
                motion_layer1 = motion_data[frame_idx, 1]  # Shape: (60, 60, 2)
                
                if show_both_layers:
                    # Create combined visualization
                    frame_image = self.create_motion_field_frame(
                        rgb_frame, motion_layer0, motion_layer1, 
                        frame_idx=frame_idx, title_suffix=f" (GOP {gop_idx})"
                    )
                else:
                    # Show only layer 0
                    frame_image = self.create_motion_field_frame(
                        rgb_frame, motion_layer0, None,
                        frame_idx=frame_idx, title_suffix=f" (GOP {gop_idx}, Layer 0)"
                    )
                
                # Convert RGB to BGR for OpenCV
                if frame_image is not None:
                    frame_bgr = cv2.cvtColor(frame_image, cv2.COLOR_RGB2BGR)
                    # Resize to ensure correct dimensions
                    frame_bgr = cv2.resize(frame_bgr, (width, height))
                    out.write(frame_bgr)
                else:
                    print(f"⚠️  Skipped frame {frame_idx} due to processing error")
            
            out.release()
            
            def main():
    """Main function to create motion vector field video for complete GOP."""
    print("🎯 Motion Vector Field Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = MotionVectorFieldVisualizer()
    
    # Load sequence
    if not visualizer.load_sequence():
        return 1
    
    # Create motion field video for complete GOP 0
    print(f"🎬 Creating complete GOP motion field video...")
    success = visualizer.create_motion_field_video(
        gop_idx=0, 
        output_path="motion_field_complete.mp4",
        show_both_layers=True,
        max_frames=None  # Process all frames in GOP
    )
    
    if success:
        print(f"✅ Motion field video created successfully!")
        print(f"   📹 File: motion_field_complete.mp4")
        print(f"   🔍 Shows all motion vectors as arrows for entire GOP")
        print(f"   ⭐ Use: ffplay motion_field_complete.mp4 to view")
    else:
        print(f"❌ Failed to create motion field video")
        return 1
    
    return 0
            print(f"   📹 File: {output_path}")
            print(f"   🎬 Frames: {frames_to_process}")
            print(f"   ⏱️  Duration: ~{frames_to_process/15.0:.1f} seconds")
            print(f"   📏 Resolution: {width}x{height}")
            
            return True
            
            print(f"📊 GOP {gop_idx} data:")
            print(f"   Motion vectors: {motion_data.shape}")
            print(f"   RGB frames: {rgb_data.shape}")
            
            # Determine number of frames to process
            num_frames = min(motion_data.shape[0], rgb_data.shape[0])
            if max_frames:
                num_frames = min(num_frames, max_frames)
            
            print(f"🎥 Processing {num_frames} frames...")
            
            # Initialize video writer
            first_frame = rgb_data[0]
            if show_both_layers:
                # Create test frame to get dimensions
                test_frame = self.create_motion_field_frame(first_frame, motion_data[0], 0)
                frame_height, frame_width = test_frame.shape[:2]
            else:
                # Will create separate videos for each layer
                test_frame = self.create_motion_field_frame(first_frame, motion_data[0], 0, layer_idx=0)
                frame_height, frame_width = test_frame.shape[:2]
            
            # Video settings
            fps = 5  # Slower for better observation
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            if show_both_layers:
                # Single video with both layers
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                for frame_idx in range(num_frames):
                    print(f"🎬 Processing frame {frame_idx + 1}/{num_frames}...", end='\r')
                    
                    rgb_frame = rgb_data[frame_idx]
                    motion_frame = motion_data[frame_idx]
                    
                    # Create frame with both layers
                    output_frame = self.create_motion_field_frame(rgb_frame, motion_frame, frame_idx)
                    
                    # Convert RGB to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(output_frame_bgr)
                
                video_writer.release()
                print(f"\n✅ Video saved to: {output_path}")
                
            else:
                # Separate videos for each layer
                for layer_idx in [0, 1]:
                    layer_output_path = output_path.replace('.mp4', f'_layer{layer_idx}.mp4')
                    video_writer = cv2.VideoWriter(layer_output_path, fourcc, fps, (frame_width, frame_height))
                    
                    print(f"\n🎬 Creating video for layer {layer_idx}...")
                    
                    for frame_idx in range(num_frames):
                        print(f"   Processing frame {frame_idx + 1}/{num_frames}...", end='\r')
                        
                        rgb_frame = rgb_data[frame_idx]
                        motion_frame = motion_data[frame_idx]
                        
                        # Create frame with single layer
                        output_frame = self.create_motion_field_frame(rgb_frame, motion_frame, frame_idx, layer_idx)
                        
                        # Convert RGB to BGR for OpenCV
                        output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(output_frame_bgr)
                    
                    video_writer.release()
                    print(f"\n✅ Layer {layer_idx} video saved to: {layer_output_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating motion field video: {str(e)}")
            return False
    
    def analyze_motion_statistics(self, gop_idx: int = 0) -> Dict:
        """Analyze motion vector statistics for a GOP."""
        if not self.data_loader:
            return {}
        
        try:
            motion_data = self.data_loader.load_motion_vectors(gop_idx)
            if motion_data is None:
                return {}
            
            stats = {
                'gop_idx': gop_idx,
                'num_frames': motion_data.shape[0],
                'grid_size': (motion_data.shape[2], motion_data.shape[3]),
                'total_vectors_per_frame': motion_data.shape[2] * motion_data.shape[3] * 2,  # 2 layers
            }
            
            # Analyze each layer
            for layer_idx in [0, 1]:
                layer_data = motion_data[:, layer_idx, :, :, :]  # (frames, height, width, 2)
                
                # Calculate magnitudes
                magnitudes = np.sqrt(layer_data[:, :, :, 0]**2 + layer_data[:, :, :, 1]**2)
                
                # Statistics
                non_zero_mask = magnitudes > 0
                significant_mask = magnitudes > self.min_magnitude_threshold
                
                layer_stats = {
                    'mean_magnitude': np.mean(magnitudes),
                    'max_magnitude': np.max(magnitudes),
                    'std_magnitude': np.std(magnitudes),
                    'non_zero_count': np.sum(non_zero_mask),
                    'significant_count': np.sum(significant_mask),
                    'total_count': magnitudes.size,
                    'non_zero_percentage': np.sum(non_zero_mask) / magnitudes.size * 100,
                    'significant_percentage': np.sum(significant_mask) / magnitudes.size * 100,
                }
                
                stats[f'layer_{layer_idx}'] = layer_stats
            
            return stats
            
        except Exception as e:
            print(f"❌ Error analyzing motion statistics: {str(e)}")
            return {}
    
    def print_motion_statistics(self, stats: Dict):
        """Print motion vector statistics in a formatted way."""
        if not stats:
            print("❌ No statistics available")
            return
        
        print(f"\n📊 Motion Vector Statistics for GOP {stats['gop_idx']}")
        print("=" * 60)
        print(f"Frames: {stats['num_frames']}")
        print(f"Grid size: {stats['grid_size']}")
        print(f"Total vectors per frame: {stats['total_vectors_per_frame']}")
        
        for layer_idx in [0, 1]:
            if f'layer_{layer_idx}' not in stats:
                continue
                
            layer_stats = stats[f'layer_{layer_idx}']
            print(f"\n🔍 Layer {layer_idx}:")
            print(f"   Mean magnitude: {layer_stats['mean_magnitude']:.3f}")
            print(f"   Max magnitude: {layer_stats['max_magnitude']:.3f}")
            print(f"   Std magnitude: {layer_stats['std_magnitude']:.3f}")
            print(f"   Non-zero vectors: {layer_stats['non_zero_count']:,} ({layer_stats['non_zero_percentage']:.1f}%)")
            print(f"   Significant vectors: {layer_stats['significant_count']:,} ({layer_stats['significant_percentage']:.1f}%)")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Create motion vector field visualization videos')
    parser.add_argument('--sequence', type=str, default=None,
                       help='Sequence name to analyze (default: first available)')
    parser.add_argument('--gop', type=int, default=0,
                       help='GOP index to visualize (default: 0)')
    parser.add_argument('--output', type=str, default='motion_field_video.mp4',
                       help='Output video path (default: motion_field_video.mp4)')
    parser.add_argument('--max-frames', type=int, default=20,
                       help='Maximum number of frames to process (default: 20)')
    parser.add_argument('--arrow-scale', type=float, default=3.0,
                       help='Scale factor for arrow lengths (default: 3.0)')
    parser.add_argument('--subsample', type=int, default=2,
                       help='Show every Nth vector (default: 2)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Minimum magnitude threshold (default: 0.1)')
    parser.add_argument('--separate-layers', action='store_true',
                       help='Create separate videos for each GOP layer')
    parser.add_argument('--stats-only', action='store_true',
                       help='Only show statistics, don\'t create video')
    
    args = parser.parse_args()
    
    print("🎯 Motion Vector Field Visualizer")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = MotionVectorFieldVisualizer()
    visualizer.arrow_scale = args.arrow_scale
    visualizer.arrow_subsample = args.subsample
    visualizer.min_magnitude_threshold = args.threshold
    
    # Load sequence
    if not visualizer.load_sequence(args.sequence):
        return 1
    
    # Show statistics
    stats = visualizer.analyze_motion_statistics(args.gop)
    visualizer.print_motion_statistics(stats)
    
    if args.stats_only:
        return 0
    
    # Create video
    print(f"\n🎬 Creating motion vector field video...")
    print(f"   GOP: {args.gop}")
    print(f"   Max frames: {args.max_frames}")
    print(f"   Arrow scale: {args.arrow_scale}")
    print(f"   Subsample: {args.subsample}")
    print(f"   Threshold: {args.threshold}")
    print(f"   Separate layers: {args.separate_layers}")
    
    success = visualizer.create_motion_field_video(
        gop_idx=args.gop,
        output_path=args.output,
        show_both_layers=not args.separate_layers,
        max_frames=args.max_frames
    )
    
    if not success:
        return 1
    
    print("\n✅ Motion vector field visualization completed!")
    print(f"📹 Check the output video(s) in the current directory")
    
    return 0


if __name__ == "__main__":
    exit(main())
