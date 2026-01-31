#!/usr/bin/env python3
"""
Improved Motion-Guided Macroblock Tracker

This tracker uses the global motion vector field to guide macroblock tracking,
rather than relying solely on individual macroblock motion vectors.
It addresses the issue where tracking "stops" by using motion field interpolation
and predictive tracking based on surrounding motion patterns.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from pathlib import Path
import cv2
import colorsys
from scipy import ndimage
from scipy.interpolate import griddata

# Add path for data loader
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory


class MotionGuidedTracker:
    """Enhanced tracker that uses global motion field to guide macroblock tracking."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.factory = MOTSDataLoaderFactory(verbose=verbose)
        self.data_loader = None
        
        # Tracking parameters
        self.motion_smoothing_sigma = 1.0  # Gaussian smoothing for motion field
        self.prediction_weight = 0.7  # Weight for motion prediction vs observation
        self.neighborhood_radius = 2  # Radius for neighborhood motion analysis
        self.motion_threshold = 0.1  # Minimum motion to consider valid
        
    def load_sequence(self, sequence_name=None):
        """Load a MOTS sequence."""
        sequences = self.factory.list_sequences(['MOT17'], ['960x960'])
        if not sequences:
            print("❌ No sequences found!")
            return False
        
        target_sequence = sequence_name if sequence_name else sequences[0]
        print(f"📂 Loading sequence: {target_sequence}")
        
        self.data_loader = self.factory.create_loader(target_sequence, ['MOT17'], ['960x960'])
        return True
    
    def smooth_motion_field(self, motion_field):
        """Apply Gaussian smoothing to motion field for better interpolation."""
        # motion_field shape: (60, 60, 2)
        smoothed = np.zeros_like(motion_field)
        smoothed[:, :, 0] = ndimage.gaussian_filter(motion_field[:, :, 0], self.motion_smoothing_sigma)
        smoothed[:, :, 1] = ndimage.gaussian_filter(motion_field[:, :, 1], self.motion_smoothing_sigma)
        return smoothed
    
    def interpolate_motion_at_position(self, motion_field, x, y):
        """Interpolate motion vector at a specific position using bilinear interpolation."""
        # Convert pixel coordinates to motion field coordinates
        grid_x = x / 16.0  # 960/60 = 16 pixels per motion vector
        grid_y = y / 16.0
        
        # Clamp to valid range
        grid_x = np.clip(grid_x, 0, motion_field.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, motion_field.shape[0] - 1)
        
        # Get integer and fractional parts
        x0, x1 = int(grid_x), min(int(grid_x) + 1, motion_field.shape[1] - 1)
        y0, y1 = int(grid_y), min(int(grid_y) + 1, motion_field.shape[0] - 1)
        
        wx = grid_x - x0
        wy = grid_y - y0
        
        # Bilinear interpolation
        motion_x = (1-wx)*(1-wy)*motion_field[y0,x0,0] + wx*(1-wy)*motion_field[y0,x1,0] + \
                   (1-wx)*wy*motion_field[y1,x0,0] + wx*wy*motion_field[y1,x1,0]
        
        motion_y = (1-wx)*(1-wy)*motion_field[y0,x0,1] + wx*(1-wy)*motion_field[y0,x1,1] + \
                   (1-wx)*wy*motion_field[y1,x0,1] + wx*wy*motion_field[y1,x1,1]
        
        return np.array([motion_x, motion_y])
    
    def get_neighborhood_motion(self, motion_field, x, y, radius=2):
        """Get average motion in neighborhood around a position."""
        grid_x = int(x / 16.0)
        grid_y = int(y / 16.0)
        
        # Define neighborhood bounds
        x_min = max(0, grid_x - radius)
        x_max = min(motion_field.shape[1], grid_x + radius + 1)
        y_min = max(0, grid_y - radius)
        y_max = min(motion_field.shape[0], grid_y + radius + 1)
        
        # Extract neighborhood
        neighborhood = motion_field[y_min:y_max, x_min:x_max, :]
        
        # Calculate weighted average (center gets more weight)
        weights = np.zeros((y_max-y_min, x_max-x_min))
        center_y, center_x = weights.shape[0]//2, weights.shape[1]//2
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                dist = np.sqrt((i-center_y)**2 + (j-center_x)**2)
                weights[i, j] = np.exp(-dist**2 / (2 * radius**2))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted average motion
        avg_motion_x = np.sum(neighborhood[:, :, 0] * weights)
        avg_motion_y = np.sum(neighborhood[:, :, 1] * weights)
        
        return np.array([avg_motion_x, avg_motion_y])
    
    def predict_next_position(self, current_pos, motion_field_current, motion_field_prev=None):
        """Predict next position using multiple motion cues."""
        x, y = current_pos
        
        # Method 1: Direct interpolation at current position
        direct_motion = self.interpolate_motion_at_position(motion_field_current, x, y)
        
        # Method 2: Neighborhood average
        neighborhood_motion = self.get_neighborhood_motion(motion_field_current, x, y, self.neighborhood_radius)
        
        # Method 3: Temporal consistency (if previous frame available)
        if motion_field_prev is not None:
            prev_motion = self.interpolate_motion_at_position(motion_field_prev, x, y)
            # Blend current and previous motion
            temporal_motion = 0.6 * direct_motion + 0.4 * prev_motion
        else:
            temporal_motion = direct_motion
        
        # Combine different predictions with weights
        combined_motion = (0.4 * direct_motion + 
                          0.4 * neighborhood_motion + 
                          0.2 * temporal_motion)
        
        # Only use prediction if motion is significant
        motion_magnitude = np.linalg.norm(combined_motion)
        if motion_magnitude < self.motion_threshold:
            return current_pos, False  # No significant motion
        
        # Predict next position
        next_x = x + combined_motion[0]
        next_y = y + combined_motion[1]
        
        # Clamp to frame bounds
        next_x = np.clip(next_x, 0, 959)
        next_y = np.clip(next_y, 0, 959)
        
        return (next_x, next_y), True
    
    def track_object_with_motion_field(self, gop_idx=0, object_id=None):
        """Track an object using global motion field guidance."""
        # Load GOP data
        motion_data = self.data_loader.load_motion_vectors(gop_idx)
        annotation_data = self.data_loader.load_corrected_annotations(gop_idx)
        rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
        
        # Get first frame annotations to select object
        annotations = annotation_data['annotations']
        first_frame_anns = annotations[0]
        
        # Select object to track
        if object_id is None:
            # Select a good object for tracking (medium size, not at edges)
            best_object = None
            best_score = 0
            
            for ann in first_frame_anns:
                if len(ann) >= 6:
                    obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                    
                    # Convert to pixel coordinates
                    x_center = x_norm * 960
                    y_center = y_norm * 960
                    width = w_norm * 960
                    height = h_norm * 960
                    
                    # Score based on size and position
                    size_score = 1.0 if (50 < width < 200 and 50 < height < 200) else 0.5
                    pos_score = 1.0 if (100 < x_center < 860 and 100 < y_center < 860) else 0.3
                    
                    total_score = size_score * pos_score
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_object = {
                            'id': int(obj_id),
                            'center': [x_center, y_center],
                            'size': [width, height],
                            'bbox': [x_center - width/2, y_center - height/2, 
                                   x_center + width/2, y_center + height/2]
                        }
        
        if best_object is None:
            print("❌ No suitable object found for tracking")
            return None
        
        print(f"🎯 Tracking object {best_object['id']} with motion field guidance")
        print(f"   📍 Initial position: ({best_object['center'][0]:.1f}, {best_object['center'][1]:.1f})")
        print(f"   📦 Size: {best_object['size'][0]:.1f}x{best_object['size'][1]:.1f}")
        
        # Initialize tracking
        tracked_positions = []
        current_pos = tuple(best_object['center'])
        tracked_positions.append(current_pos)
        
        # Track through all frames
        for frame_idx in range(1, motion_data.shape[0]):
            # Get motion fields (combine both layers)
            motion_layer0 = motion_data[frame_idx, 0]  # Current frame
            motion_layer1 = motion_data[frame_idx, 1]
            
            # Smooth motion fields
            smooth_layer0 = self.smooth_motion_field(motion_layer0)
            smooth_layer1 = self.smooth_motion_field(motion_layer1)
            
            # Combine layers (weighted average)
            combined_motion = 0.6 * smooth_layer0 + 0.4 * smooth_layer1
            
            # Get previous motion field for temporal consistency
            prev_motion = None
            if frame_idx > 1:
                prev_layer0 = self.smooth_motion_field(motion_data[frame_idx-1, 0])
                prev_layer1 = self.smooth_motion_field(motion_data[frame_idx-1, 1])
                prev_motion = 0.6 * prev_layer0 + 0.4 * prev_layer1
            
            # Predict next position
            next_pos, has_motion = self.predict_next_position(
                current_pos, combined_motion, prev_motion
            )
            
            if has_motion:
                current_pos = next_pos
                tracked_positions.append(current_pos)
                
                if self.verbose and frame_idx % 10 == 0:
                    print(f"   📍 Frame {frame_idx}: ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
            else:
                # No significant motion, keep current position
                tracked_positions.append(current_pos)
        
        return {
            'object': best_object,
            'positions': tracked_positions,
            'frames': list(range(len(tracked_positions)))
        }
    
    def visualize_improved_tracking(self, tracking_result, gop_idx=0, output_path="improved_motion_tracking.mp4"):
        """Create visualization of improved motion-guided tracking."""
        if tracking_result is None:
            return False
        
        # Load data
        motion_data = self.data_loader.load_motion_vectors(gop_idx)
        rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
        
        print(f"🎬 Creating improved tracking visualization...")
        print(f"   📹 Output: {output_path}")
        print(f"   🎯 Tracking {len(tracking_result['positions'])} positions")
        
        # Video setup
        height, width = 960, 1920  # Side by side
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10.0, (width, height))
        
        positions = tracking_result['positions']
        object_info = tracking_result['object']
        
        for frame_idx in range(len(positions)):
            print(f"   📍 Creating frame {frame_idx+1}/{len(positions)}")
            
            # Get RGB frame
            rgb_frame_idx = min(frame_idx, rgb_data.shape[0] - 1)
            rgb_frame = rgb_data[rgb_frame_idx]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Left plot: RGB frame with tracking
            ax1.imshow(rgb_frame)
            
            # Draw tracking trail
            if len(positions) > 1:
                trail_x = [pos[0] for pos in positions[:frame_idx+1]]
                trail_y = [pos[1] for pos in positions[:frame_idx+1]]
                
                # Draw trail with fading
                for i in range(len(trail_x)-1):
                    alpha = 0.3 + 0.7 * (i / len(trail_x))
                    ax1.plot([trail_x[i], trail_x[i+1]], [trail_y[i], trail_y[i+1]], 
                            'r-', linewidth=3, alpha=alpha)
            
            # Draw current position
            current_pos = positions[frame_idx]
            ax1.plot(current_pos[0], current_pos[1], 'ro', markersize=12, markeredgecolor='white', markeredgewidth=2)
            
            # Draw bounding box at current position
            bbox_width, bbox_height = object_info['size']
            bbox_x = current_pos[0] - bbox_width/2
            bbox_y = current_pos[1] - bbox_height/2
            
            bbox_rect = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height,
                                        linewidth=3, edgecolor='red', facecolor='none', alpha=0.8)
            ax1.add_patch(bbox_rect)
            
            ax1.set_title(f'Improved Motion-Guided Tracking\nFrame {frame_idx} - Object {object_info["id"]}', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlim(0, 960)
            ax1.set_ylim(960, 0)
            ax1.axis('off')
            
            # Right plot: Motion field visualization
            if frame_idx < motion_data.shape[0]:
                motion_layer0 = motion_data[frame_idx, 0]
                motion_layer1 = motion_data[frame_idx, 1]
                
                # Show smoothed combined motion field
                smooth_layer0 = self.smooth_motion_field(motion_layer0)
                smooth_layer1 = self.smooth_motion_field(motion_layer1)
                combined_motion = 0.6 * smooth_layer0 + 0.4 * smooth_layer1
                
                ax2.imshow(rgb_frame, alpha=0.6)
                
                # Draw motion field (subsampled)
                subsample = 4
                for y in range(0, combined_motion.shape[0], subsample):
                    for x in range(0, combined_motion.shape[1], subsample):
                        motion_vec = combined_motion[y, x]
                        magnitude = np.linalg.norm(motion_vec)
                        
                        if magnitude > 0.3:
                            start_x = x * 16 + 8
                            start_y = y * 16 + 8
                            end_x = start_x + motion_vec[0] * 4
                            end_y = start_y + motion_vec[1] * 4
                            
                            alpha = min(1.0, magnitude / 2.0)
                            ax2.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                                       arrowprops=dict(arrowstyle='->', color='cyan', 
                                                     lw=1.5, alpha=alpha))
                
                # Highlight current tracking position
                ax2.plot(current_pos[0], current_pos[1], 'yo', markersize=10, markeredgecolor='red', markeredgewidth=2)
            
            ax2.set_title(f'Motion Field Guidance\nSmoothed Combined Motion Vectors', 
                         fontsize=14, fontweight='bold')
            ax2.set_xlim(0, 960)
            ax2.set_ylim(960, 0)
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Convert to video frame
            fig.canvas.draw()
            try:
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                buf = buf[:, :, :3]
            except AttributeError:
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.resize(frame_bgr, (width, height))  # Proper side by side
            out.write(frame_bgr)
            
            plt.close(fig)
        
        out.release()
        print(f"✅ Improved tracking video created: {output_path}")
        return True


def main():
    """Main function to demonstrate improved motion-guided tracking."""
    print("🚀 Improved Motion-Guided Macroblock Tracker")
    print("=" * 60)
    
    # Initialize tracker
    tracker = MotionGuidedTracker(verbose=True)
    
    # Load sequence
    if not tracker.load_sequence():
        return 1
    
    # Track object with motion field guidance
    print(f"\n🎯 Starting improved motion-guided tracking...")
    tracking_result = tracker.track_object_with_motion_field(gop_idx=0)
    
    if tracking_result:
        # Create visualization
        success = tracker.visualize_improved_tracking(
            tracking_result, 
            gop_idx=0, 
            output_path="improved_motion_tracking.mp4"
        )
        
        if success:
            print(f"\n✅ Improved motion-guided tracking completed!")
            print(f"   📹 Video: improved_motion_tracking.mp4")
            print(f"   🔍 Shows object tracking guided by global motion field")
            print(f"   🎯 Tracks object through entire GOP using motion interpolation")
            print(f"   ⭐ Use: ffplay improved_motion_tracking.mp4 to view")
            
            # Print tracking statistics
            positions = tracking_result['positions']
            total_distance = 0
            for i in range(1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                total_distance += dist
            
            print(f"\n📊 Tracking Statistics:")
            print(f"   🎯 Object ID: {tracking_result['object']['id']}")
            print(f"   📍 Frames tracked: {len(positions)}")
            print(f"   📏 Total distance: {total_distance:.1f} pixels")
            print(f"   📈 Average movement per frame: {total_distance/len(positions):.1f} pixels")
        else:
            print(f"\n❌ Failed to create tracking visualization")
            return 1
    else:
        print(f"\n❌ Tracking failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
