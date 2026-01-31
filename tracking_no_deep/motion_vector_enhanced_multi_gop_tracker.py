#!/usr/bin/env python3
"""
Motion Vector Enhanced Multi-GOP Tracker

This script combines the advanced multi-GOP macroblock tracking with
comprehensive motion vector field visualization overlaid on the same video.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import ndimage

class KalmanFilter2D:
    """
    2D Kalman Filter for object tracking (Option 2)
    
    State vector: [x, y, vx, vy] (position and velocity)
    Measurement: [x, y] (observed position)
    """
    
    def __init__(self, initial_pos, process_noise=1.0, measurement_noise=10.0):
        # State vector [x, y, vx, vy]
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])
        
        # State covariance matrix
        self.P = np.eye(4) * 100  # Initial uncertainty
        
        # Process noise (motion model uncertainty)
        self.Q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0], 
            [0, 0, process_noise, 0],
            [0, 0, 0, process_noise]
        ])
        
        # Measurement noise (observation uncertainty)
        self.R = np.eye(2) * measurement_noise
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Measurement matrix (observe position only)
        self.H = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ])
    
    def predict(self):
        """Predict the next state"""
        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement):
        """Update with measurement"""
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update step
        y = measurement - self.H @ self.state  # Innovation
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        
        return self.state[:2]  # Return updated position
    
    def get_velocity(self):
        """Get current velocity estimate"""
        return self.state[2:4]

# Add path for data loader
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory


class MotionVectorEnhancedTracker:
    """Enhanced tracker that combines macroblock tracking with motion vector field visualization."""
    
    def __init__(self, verbose=True, validation_mode=False):
        self.verbose = verbose
        self.validation_mode = validation_mode
        self.factory = MOTSDataLoaderFactory(verbose=verbose)
        self.data_loader = None
        
        # Motion field visualization parameters
        self.motion_smoothing_sigma = 1.0
        self.arrow_subsample = 3  # Show every 3rd motion vector
        self.arrow_scale = 6  # Scale factor for arrow size
        self.motion_threshold = 0.2  # Minimum motion to show arrow
        
        # Enhanced tracking parameters
        self.motion_history = {}  # Store motion history for temporal consistency
        self.history_length = 5  # Number of frames to keep in history
        
        # Kalman filter for state estimation (Option 2)
        self.kalman_filters = {}  # Store Kalman filter for each object
        self.use_kalman = True
        
        # Motion amplification parameters to fix slow tracking
        self.motion_scale_factor = 2.0  # Amplify motion vectors 
        self.fast_motion_threshold = 3.0  # Threshold for detecting fast motion
        self.adaptive_scaling = True  # Enable adaptive scaling based on motion magnitude
        
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
        """Apply Gaussian smoothing to motion field."""
        smoothed = np.zeros_like(motion_field)
        smoothed[:, :, 0] = ndimage.gaussian_filter(motion_field[:, :, 0], self.motion_smoothing_sigma)
        smoothed[:, :, 1] = ndimage.gaussian_filter(motion_field[:, :, 1], self.motion_smoothing_sigma)
        return smoothed
    
    def apply_motion_scaling(self, motion_vector, obj_id=None):
        """
        Apply adaptive motion scaling to fix slow tracking issue.
        
        The issue: Objects move faster than bounding box displacement
        Solution: Intelligently amplify motion vectors based on magnitude
        """
        motion_magnitude = np.linalg.norm(motion_vector)
        
        if motion_magnitude < 0.1:
            # Very small motion - don't amplify noise
            return motion_vector
        
        if self.adaptive_scaling:
            # Adaptive scaling based on motion magnitude
            if motion_magnitude < 1.0:
                # Small motion: amplify significantly
                scale_factor = self.motion_scale_factor * 2.0  # 4x amplification
            elif motion_magnitude < self.fast_motion_threshold:
                # Medium motion: standard amplification  
                scale_factor = self.motion_scale_factor  # 2x amplification
            else:
                # Already fast motion: minimal amplification
                scale_factor = max(1.2, self.motion_scale_factor * 0.6)  # 1.2x amplification
        else:
            # Fixed scaling
            scale_factor = self.motion_scale_factor
        
        scaled_motion = motion_vector * scale_factor
        
        if obj_id is not None and self.verbose and scale_factor > 1.5:
            print(f"   ⚡ Object {obj_id}: Motion scaling {motion_magnitude:.2f} → {np.linalg.norm(scaled_motion):.2f} (factor: {scale_factor:.1f}x)")
        
        return scaled_motion
    
    def select_tracking_objects(self, annotations):
        """Select objects for tracking from first frame annotations."""
        selected_objects = []
        
        for ann in annotations[:4]:  # Take first 4 objects
            if len(ann) >= 6:
                obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                
                # Convert to pixel coordinates
                x_center = x_norm * 960
                y_center = y_norm * 960
                width = w_norm * 960
                height = h_norm * 960
                
                # Select center macroblock
                mb_col = int(x_center // 16)
                mb_row = int(y_center // 16)
                
                selected_objects.append({
                    'id': int(obj_id),
                    'center': [x_center, y_center],
                    'size': [width, height],
                    'original_size': [width, height],  # Store original size for resets
                    'macroblock': [mb_col, mb_row],
                    'color': self.get_object_color(len(selected_objects))
                })
        
        return selected_objects
    
    def get_object_color(self, index):
        """Get distinct color for object."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
        return colors[index % len(colors)]
    
    def get_ground_truth_annotations(self, global_frame, all_gop_data):
        """Get ground truth annotations for a specific global frame."""
        gop_size = 49  # Frames per GOP
        gop_idx = global_frame // gop_size
        frame_idx = global_frame % gop_size
        
        if gop_idx >= len(all_gop_data):
            return None
        
        annotation_data = all_gop_data[gop_idx]['annotation_data']
        if not annotation_data or frame_idx >= len(annotation_data['annotations']):
            return None
        
        # Convert annotations to pixel coordinates
        annotations = annotation_data['annotations'][frame_idx]
        ground_truth_objects = {}
        
        for ann in annotations:
            if len(ann) >= 6:
                obj_id = int(ann[0])
                x_norm, y_norm, w_norm, h_norm = ann[2:6]
                
                # Convert to pixel coordinates
                x_center = x_norm * 960
                y_center = y_norm * 960
                width = w_norm * 960
                height = h_norm * 960
                
                ground_truth_objects[obj_id] = {
                    'center': [x_center, y_center],
                    'size': [width, height]
                }
        
        return ground_truth_objects
    
    def extract_enhanced_motion(self, combined_motion, current_pos, current_size, obj_id, global_frame):
        """
        Enhanced motion extraction using multiple approaches:
        - Option 1: Multi-macroblock sampling  
        - Option 2: Kalman filtering for state estimation
        - Option 3: Edge motion priority
        - Option 4: Temporal consistency
        """
        center_x, center_y = current_pos
        bbox_width, bbox_height = current_size
        
        # Convert to macroblock coordinates
        center_mb_x = int(center_x // 16)
        center_mb_y = int(center_y // 16)
        
        # Calculate bounding box in macroblock space
        bbox_mb_left = max(0, int((center_x - bbox_width/2) // 16))
        bbox_mb_right = min(combined_motion.shape[1] - 1, int((center_x + bbox_width/2) // 16))
        bbox_mb_top = max(0, int((center_y - bbox_height/2) // 16))
        bbox_mb_bottom = min(combined_motion.shape[0] - 1, int((center_y + bbox_height/2) // 16))
        
        # 1. Multi-Macroblock Sampling: Collect all motion vectors within bounding box
        internal_motions = []
        edge_motions = []
        
        for mb_y in range(bbox_mb_top, bbox_mb_bottom + 1):
            for mb_x in range(bbox_mb_left, bbox_mb_right + 1):
                motion_vec = combined_motion[mb_y, mb_x]
                motion_magnitude = np.linalg.norm(motion_vec)
                
                # 3. Edge Motion Priority: Prioritize motion vectors at object boundaries
                is_edge = (mb_x == bbox_mb_left or mb_x == bbox_mb_right or 
                          mb_y == bbox_mb_top or mb_y == bbox_mb_bottom)
                
                if motion_magnitude > 0.1:  # Only consider significant motion
                    if is_edge:
                        edge_motions.append(motion_vec)
                    else:
                        internal_motions.append(motion_vec)
        
        # Calculate weighted motion
        if edge_motions:
            # Prioritize edge motion (weight = 0.7)
            edge_motion = np.mean(edge_motions, axis=0)
            if internal_motions:
                internal_motion = np.mean(internal_motions, axis=0)
                combined_motion = 0.7 * edge_motion + 0.3 * internal_motion
            else:
                combined_motion = edge_motion
        elif internal_motions:
            # Fallback to internal motion
            combined_motion = np.mean(internal_motions, axis=0)
        else:
            # No significant motion found
            combined_motion = np.array([0.0, 0.0])
        
        # 4. Temporal Consistency: Use motion history for smoothing
        if obj_id not in self.motion_history:
            self.motion_history[obj_id] = []
        
        # Add current motion to history
        self.motion_history[obj_id].append(combined_motion.copy())
        
        # Keep only recent history
        if len(self.motion_history[obj_id]) > self.history_length:
            self.motion_history[obj_id].pop(0)
        
        # Detect if object is in fast motion phase
        fast_motion_detected = False
        if len(self.motion_history[obj_id]) >= 3:
            recent_magnitudes = [np.linalg.norm(m) for m in self.motion_history[obj_id][-3:]]
            avg_magnitude = np.mean(recent_magnitudes)
            if avg_magnitude > self.fast_motion_threshold:
                fast_motion_detected = True
                if self.verbose:
                    print(f"   🏃 Object {obj_id}: Fast motion detected (avg: {avg_magnitude:.2f})")
        
        # Apply temporal smoothing
        if len(self.motion_history[obj_id]) >= 2:
            # Weight recent motions more heavily
            weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5][-len(self.motion_history[obj_id]):])
            weights = weights / weights.sum()  # Normalize
            
            smoothed_motion = np.average(self.motion_history[obj_id], axis=0, weights=weights)
            
            # If current motion is very weak but history shows motion, use prediction
            current_magnitude = np.linalg.norm(combined_motion)
            history_magnitude = np.linalg.norm(smoothed_motion)
            
            if current_magnitude < 0.3 and history_magnitude > 0.5:
                # Use predicted motion based on history
                predicted_motion = smoothed_motion * 1.0  # Keep full motion (was 0.8)
                if self.verbose:
                    print(f"   🔮 Object {obj_id}: Using predicted motion {predicted_motion} (current was weak: {combined_motion})")
                final_motion = predicted_motion
            else:
                # Blend current with history - adjust based on motion speed
                if fast_motion_detected:
                    # For fast motion, trust current motion more heavily
                    final_motion = 0.9 * combined_motion + 0.1 * smoothed_motion
                else:
                    # Standard blending - favor current motion more
                    final_motion = 0.8 * combined_motion + 0.2 * smoothed_motion
        else:
            final_motion = combined_motion
        
        # 2. Kalman Filter State Estimation (Option 2)
        if self.use_kalman:
            # Initialize Kalman filter for this object if needed
            if obj_id not in self.kalman_filters:
                self.kalman_filters[obj_id] = KalmanFilter2D(
                    initial_pos=current_pos,
                    process_noise=5.0,    # Increased: Allow faster motion changes
                    measurement_noise=3.0  # Decreased: Trust observations more
                )
            
            kalman = self.kalman_filters[obj_id]
            
            # Predict next position using Kalman filter
            predicted_pos = kalman.predict()
            
            # Calculate motion-based position estimate
            motion_pos = [current_pos[0] + final_motion[0], current_pos[1] + final_motion[1]]
            
            # Update Kalman filter with motion-based observation
            kalman_pos = kalman.update(motion_pos)
            
            # Extract Kalman-smoothed motion
            kalman_motion = kalman_pos - np.array(current_pos)
            
            # Blend Kalman prediction with motion-based estimate
            alpha = 0.7  # Weight for Kalman filter
            enhanced_motion = alpha * kalman_motion + (1 - alpha) * final_motion
            
            if self.verbose and np.linalg.norm(enhanced_motion - final_motion) > 0.5:
                print(f"   🎯 Object {obj_id}: Kalman correction: {final_motion} → {enhanced_motion}")
            
            # Apply motion scaling to fix slow tracking
            scaled_motion = self.apply_motion_scaling(enhanced_motion, obj_id)
            return scaled_motion
        
        # Apply motion scaling to fix slow tracking  
        scaled_motion = self.apply_motion_scaling(final_motion, obj_id)
        return scaled_motion
    
    def reset_bounding_boxes(self, tracking_objects, current_annotations, frame_idx, global_frame):
        """Reset bounding boxes to original annotations every 20 frames."""
        if current_annotations is None:
            print(f"⚠️  No annotations available for reset at frame {global_frame} (local frame {frame_idx})")
            return tracking_objects
        
        print(f"🔄 Resetting bounding boxes at global frame {global_frame} (local frame {frame_idx})")
        print(f"   📋 Available annotations: {len(current_annotations)} objects")
        
        # Create a mapping of object IDs to annotations
        ann_dict = {}
        for i, ann in enumerate(current_annotations):
            if len(ann) >= 6:
                obj_id = int(ann[0])
                ann_dict[obj_id] = ann
                print(f"   📦 Annotation {i}: ID={obj_id}, pos=({ann[2]:.3f},{ann[3]:.3f}), size=({ann[4]:.3f}x{ann[5]:.3f})")
        
        # Reset each tracked object if annotation is available
        reset_count = 0
        for obj in tracking_objects:
            obj_id = obj['id']
            if obj_id in ann_dict:
                ann = ann_dict[obj_id]
                x_norm, y_norm, w_norm, h_norm = ann[2:6]
                
                # Convert to pixel coordinates
                x_center = x_norm * 960
                y_center = y_norm * 960
                width = w_norm * 960
                height = h_norm * 960
                
                # Reset to original annotation size
                obj['size'] = [width, height]
                obj['center'] = [x_center, y_center]
                obj['macroblock'] = [int(x_center // 16), int(y_center // 16)]
                
                print(f"   ✅ Reset Object {obj_id}: center=({x_center:.1f},{y_center:.1f}), size=({width:.1f}x{height:.1f})")
                reset_count += 1
            else:
                print(f"   ❌ No annotation found for Object {obj_id}")
        
        print(f"   📊 Reset {reset_count}/{len(tracking_objects)} objects")
        return tracking_objects
    
    def analyze_motion_divergence(self, motion_field, center_x, center_y, bbox_width, bbox_height):
        """Analyze motion vector divergence around bounding box to adapt size."""
        # Convert to motion field coordinates
        center_mb_x = int(center_x // 16)
        center_mb_y = int(center_y // 16)
        
        # Define bounding box region in motion field coordinates
        bbox_mb_width = max(1, int(bbox_width // 16))
        bbox_mb_height = max(1, int(bbox_height // 16))
        
        # Sample motion vectors around the bounding box
        sample_radius = max(2, max(bbox_mb_width, bbox_mb_height) // 2)
        
        # Get motion vectors in different regions around the object
        regions = {
            'center': [],
            'left': [],
            'right': [],
            'top': [],
            'bottom': [],
            'corners': []
        }
        
        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                mb_x = center_mb_x + dx
                mb_y = center_mb_y + dy
                
                # Check bounds
                if (0 <= mb_x < motion_field.shape[1] and 
                    0 <= mb_y < motion_field.shape[0]):
                    
                    motion_vec = motion_field[mb_y, mb_x]
                    
                    # Classify into regions
                    if abs(dx) <= 1 and abs(dy) <= 1:
                        regions['center'].append(motion_vec)
                    elif dx < -1:
                        regions['left'].append(motion_vec)
                    elif dx > 1:
                        regions['right'].append(motion_vec)
                    elif dy < -1:
                        regions['top'].append(motion_vec)
                    elif dy > 1:
                        regions['bottom'].append(motion_vec)
                    
                    # Corner regions
                    if abs(dx) > 1 and abs(dy) > 1:
                        regions['corners'].append(motion_vec)
        
        # Calculate motion statistics for each region
        region_stats = {}
        for region_name, vectors in regions.items():
            if vectors:
                vectors = np.array(vectors)
                avg_motion = np.mean(vectors, axis=0)
                motion_variance = np.var(vectors, axis=0)
                region_stats[region_name] = {
                    'avg_motion': avg_motion,
                    'variance': motion_variance,
                    'magnitude': np.linalg.norm(avg_motion)
                }
        
        return region_stats
    
    def calculate_adaptive_bbox_size(self, original_size, motion_stats, frame_idx):
        """Calculate adaptive bounding box size based on motion divergence."""
        original_width, original_height = original_size
        
        if not motion_stats or 'center' not in motion_stats:
            return original_size
        
        # Get motion statistics
        center_motion = motion_stats.get('center', {}).get('avg_motion', np.array([0, 0]))
        left_motion = motion_stats.get('left', {}).get('avg_motion', np.array([0, 0]))
        right_motion = motion_stats.get('right', {}).get('avg_motion', np.array([0, 0]))
        top_motion = motion_stats.get('top', {}).get('avg_motion', np.array([0, 0]))
        bottom_motion = motion_stats.get('bottom', {}).get('avg_motion', np.array([0, 0]))
        
        # Calculate motion divergence
        # Horizontal divergence: left motion vs right motion
        horizontal_divergence = np.linalg.norm(left_motion - right_motion)
        
        # Vertical divergence: top motion vs bottom motion  
        vertical_divergence = np.linalg.norm(top_motion - bottom_motion)
        
        # Calculate expansion/contraction factors
        # High divergence = expansion, low divergence = contraction
        max_divergence = 3.0  # Threshold for maximum divergence
        min_size_factor = 0.8  # Minimum size factor
        max_size_factor = 1.4  # Maximum size factor
        
        # Horizontal size adaptation
        h_factor = min_size_factor + (horizontal_divergence / max_divergence) * (max_size_factor - min_size_factor)
        h_factor = np.clip(h_factor, min_size_factor, max_size_factor)
        
        # Vertical size adaptation
        v_factor = min_size_factor + (vertical_divergence / max_divergence) * (max_size_factor - min_size_factor)
        v_factor = np.clip(v_factor, min_size_factor, max_size_factor)
        
        # Apply temporal smoothing to avoid rapid size changes
        smoothing_factor = 0.1  # Lower = more smoothing
        if hasattr(self, 'prev_size_factors'):
            h_factor = (1 - smoothing_factor) * self.prev_size_factors[0] + smoothing_factor * h_factor
            v_factor = (1 - smoothing_factor) * self.prev_size_factors[1] + smoothing_factor * v_factor
        
        self.prev_size_factors = [h_factor, v_factor]
        
        # Calculate new size
        new_width = original_width * h_factor
        new_height = original_height * v_factor
        
        return [new_width, new_height]
    
    def track_objects_with_motion_field(self, num_gops=3):
        """Track objects across multiple GOPs with adaptive bounding box sizing."""
        mode_text = "Validation Mode" if self.validation_mode else "Adaptive Bounding Boxes"
        print(f"🎯 Enhanced Multi-GOP Tracking with {mode_text}")
        
        # Load first GOP to select objects
        motion_data_gop0 = self.data_loader.load_motion_vectors(0)
        annotation_data_gop0 = self.data_loader.load_corrected_annotations(0)
        
        if motion_data_gop0 is None or annotation_data_gop0 is None:
            print("❌ Failed to load GOP 0 data")
            return None
        
        # Select tracking objects
        first_frame_anns = annotation_data_gop0['annotations'][0]
        tracking_objects = self.select_tracking_objects(first_frame_anns)
        
        if not tracking_objects:
            print("❌ No objects selected for tracking")
            return None
        
        print(f"🎯 Selected {len(tracking_objects)} objects for {'validation' if self.validation_mode else 'adaptive'} tracking")
        for obj in tracking_objects:
            print(f"   📦 Object {obj['id']}: center=({obj['center'][0]:.1f},{obj['center'][1]:.1f}), size=({obj['size'][0]:.1f}x{obj['size'][1]:.1f})")
        
        # Initialize size tracking
        self.prev_size_factors = [1.0, 1.0]  # Initialize size factors
        
        # Collect data from all GOPs
        all_gop_data = []
        
        for gop_idx in range(num_gops):
            print(f"📖 Loading GOP {gop_idx}...")
            
            motion_data = self.data_loader.load_motion_vectors(gop_idx)
            rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
            annotation_data = self.data_loader.load_corrected_annotations(gop_idx)
            
            if motion_data is None or rgb_data is None:
                print(f"⚠️  GOP {gop_idx} data incomplete")
                break
            
            all_gop_data.append({
                'gop_idx': gop_idx,
                'motion_data': motion_data,
                'rgb_data': rgb_data,
                'annotation_data': annotation_data
            })
        
        print(f"✅ Loaded {len(all_gop_data)} GOPs")
        
        # Track objects through all frames with adaptive sizing
        tracked_positions = {obj['id']: [] for obj in tracking_objects}
        
        for gop_data in all_gop_data:
            motion_data = gop_data['motion_data']
            annotation_data = gop_data['annotation_data']
            gop_idx = gop_data['gop_idx']
            
            for frame_idx in range(motion_data.shape[0]):
                # Calculate global frame number for reset logic
                global_frame = gop_idx * motion_data.shape[0] + frame_idx
                
                # Reset bounding boxes every 20 frames
                if global_frame > 0 and global_frame % 20 == 0:
                    current_annotations = None
                    # Calculate the correct annotation frame index within this GOP
                    annotation_frame_idx = frame_idx
                    print(f"🔍 Reset at global frame {global_frame}: GOP {gop_idx}, local frame {frame_idx}")
                    
                    if annotation_data and annotation_frame_idx < len(annotation_data['annotations']):
                        current_annotations = annotation_data['annotations'][annotation_frame_idx]
                        print(f"   📋 Using annotations from GOP {gop_idx}, frame {annotation_frame_idx}")
                    else:
                        print(f"   ⚠️  No annotations available for GOP {gop_idx}, frame {annotation_frame_idx}")
                    
                    tracking_objects = self.reset_bounding_boxes(tracking_objects, current_annotations, annotation_frame_idx, global_frame)
                
                # Get current motion field
                motion_layer0 = motion_data[frame_idx, 0]
                motion_layer1 = motion_data[frame_idx, 1]
                
                # Smooth and combine layers
                smooth_layer0 = self.smooth_motion_field(motion_layer0)
                smooth_layer1 = self.smooth_motion_field(motion_layer1)
                combined_motion = 0.6 * smooth_layer0 + 0.4 * smooth_layer1
                
                # Track each object with enhanced motion extraction
                for obj in tracking_objects:
                    obj_id = obj['id']
                    
                    # Update macroblock position based on motion
                    if global_frame == 0:
                        # Initialize position and size
                        current_pos = obj['center'].copy()
                        current_size = obj['size'].copy()
                        # Use simple center motion for first frame
                        mb_col, mb_row = obj['macroblock']
                        if (0 <= mb_row < combined_motion.shape[0] and 
                            0 <= mb_col < combined_motion.shape[1]):
                            motion_vec = combined_motion[mb_row, mb_col]
                        else:
                            motion_vec = np.array([0.0, 0.0])
                        
                        # Initialize motion_stats for first frame
                        motion_stats = {
                            'mean_magnitude': 0.0,
                            'std_magnitude': 0.0,
                            'direction_variance': 0.0,
                            'edge_coherence': 0.0
                        }
                    else:
                        # Get last position and size
                        if tracked_positions[obj_id]:
                            last_track = tracked_positions[obj_id][-1]
                            last_pos = last_track['position']
                            last_size = last_track['adaptive_size']
                            
                            # 🚀 ENHANCED MOTION EXTRACTION
                            motion_vec = self.extract_enhanced_motion(
                                combined_motion, last_pos, last_size, obj_id, global_frame
                            )
                            
                            current_pos = [
                                last_pos[0] + motion_vec[0],
                                last_pos[1] + motion_vec[1]
                            ]
                            # Clamp to frame bounds
                            current_pos[0] = np.clip(current_pos[0], 0, 959)
                            current_pos[1] = np.clip(current_pos[1], 0, 959)
                            
                            # Analyze motion divergence around current position
                            motion_stats = self.analyze_motion_divergence(
                                combined_motion, current_pos[0], current_pos[1], 
                                last_size[0], last_size[1]
                            )
                            
                            # Calculate adaptive bounding box size
                            current_size = self.calculate_adaptive_bbox_size(
                                obj['size'], motion_stats, global_frame
                            )
                        else:
                            current_pos = obj['center'].copy()
                            current_size = obj['size'].copy()
                            motion_vec = np.array([0.0, 0.0])
                            
                            # Initialize motion_stats for objects without tracking history
                            motion_stats = {
                                'mean_magnitude': 0.0,
                                'std_magnitude': 0.0,
                                'direction_variance': 0.0,
                                'edge_coherence': 0.0
                            }
                        
                        # Update macroblock coordinates
                        obj['macroblock'] = [int(current_pos[0] // 16), int(current_pos[1] // 16)]
                        
                        # Get ground truth for validation mode
                        ground_truth = None
                        if self.validation_mode:
                            gt_annotations = self.get_ground_truth_annotations(global_frame, all_gop_data)
                            if gt_annotations and obj['id'] in gt_annotations:
                                ground_truth = gt_annotations[obj['id']]
                        
                        # Store tracking data with adaptive sizing
                        track_data = {
                            'gop': gop_idx,
                            'frame': frame_idx,
                            'global_frame': global_frame,
                            'position': current_pos,
                            'original_size': obj['size'].copy(),
                            'adaptive_size': current_size,
                            'motion_vector': motion_vec,
                            'macroblock': obj['macroblock'].copy(),
                            'motion_stats': motion_stats if frame_idx > 0 or gop_idx > 0 else None
                        }
                        
                        # Add ground truth data in validation mode
                        if self.validation_mode and ground_truth:
                            track_data['ground_truth_position'] = ground_truth['center']
                            track_data['ground_truth_size'] = ground_truth['size']
                            # Calculate tracking error
                            error_x = current_pos[0] - ground_truth['center'][0]
                            error_y = current_pos[1] - ground_truth['center'][1]
                            track_data['tracking_error'] = np.sqrt(error_x**2 + error_y**2)
                        
                        tracked_positions[obj['id']].append(track_data)
        
        return {
            'tracking_objects': tracking_objects,
            'tracked_positions': tracked_positions,
            'gop_data': all_gop_data
        }
    
    def create_motion_vector_enhanced_video(self, tracking_result, output_path="motion_vector_enhanced_tracking.mp4"):
        """Create video with motion vector field overlay and object tracking."""
        if tracking_result is None:
            return False
        
        print(f"🎬 Creating motion vector enhanced video...")
        print(f"   📹 Output: {output_path}")
        
        tracking_objects = tracking_result['tracking_objects']
        tracked_positions = tracking_result['tracked_positions']
        all_gop_data = tracking_result['gop_data']
        
        # Video setup
        height, width = 960, 960
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))
        
        frame_count = 0
        
        # Process each GOP
        for gop_data in all_gop_data:
            motion_data = gop_data['motion_data']
            rgb_data = gop_data['rgb_data']
            gop_idx = gop_data['gop_idx']
            
            print(f"   📖 Processing GOP {gop_idx} ({motion_data.shape[0]} frames)...")
            
            for frame_idx in range(motion_data.shape[0]):
                print(f"   📍 Creating frame {frame_count+1}")
                
                # Get RGB frame
                rgb_frame_idx = min(frame_idx, rgb_data.shape[0] - 1)
                rgb_frame = rgb_data[rgb_frame_idx]
                
                # Get motion field
                motion_layer0 = motion_data[frame_idx, 0]
                motion_layer1 = motion_data[frame_idx, 1]
                
                # Smooth and combine motion fields
                smooth_layer0 = self.smooth_motion_field(motion_layer0)
                smooth_layer1 = self.smooth_motion_field(motion_layer1)
                combined_motion = 0.6 * smooth_layer0 + 0.4 * smooth_layer1
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 12))
                
                # Display RGB frame
                ax.imshow(rgb_frame, extent=[0, 960, 960, 0], alpha=0.7)
                
                # Draw motion vector field
                self.draw_motion_vector_field(ax, combined_motion)
                
                # Draw object tracking
                self.draw_object_tracking(ax, tracking_objects, tracked_positions, frame_count)
                
                # Add comprehensive info with size adaptation details
                info_text = f'Motion Vector Enhanced Tracking with Adaptive Sizing\\n'
                info_text += f'Frame {frame_count+1} - GOP {gop_idx}\\n'
                info_text += f'Objects: {len(tracking_objects)}\\n'
                info_text += f'Solid boxes: Adaptive size | Dashed: Original size'
                
                ax.text(20, 80, info_text, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9),
                       fontweight='bold', verticalalignment='top')
                
                ax.set_title(f'Adaptive Bounding Box Tracking + Motion Vector Field\\nGOP {gop_idx} - Frame {frame_idx}', 
                           fontsize=16, fontweight='bold')
                ax.set_xlim(0, 960)
                ax.set_ylim(960, 0)
                ax.axis('off')
                
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
                frame_bgr = cv2.resize(frame_bgr, (width, height))
                out.write(frame_bgr)
                
                plt.close(fig)
                frame_count += 1
        
        out.release()
        print(f"✅ Motion vector enhanced video with adaptive sizing created: {output_path}")
        print(f"   🎯 Total frames: {frame_count}")
        print(f"   📦 Features:")
        print(f"     • Adaptive bounding box sizing based on motion divergence")
        print(f"     • Solid boxes show adaptive size, dashed show original")
        print(f"     • Size change percentages displayed")
        print(f"     • Motion divergence visualization around objects")
        return True
    
    def draw_motion_vector_field(self, ax, motion_field):
        """Draw motion vector field as arrows."""
        for y in range(0, motion_field.shape[0], self.arrow_subsample):
            for x in range(0, motion_field.shape[1], self.arrow_subsample):
                motion_vec = motion_field[y, x]
                magnitude = np.linalg.norm(motion_vec)
                
                if magnitude > self.motion_threshold:
                    # Convert to pixel coordinates
                    start_x = x * 16 + 8
                    start_y = y * 16 + 8
                    end_x = start_x + motion_vec[0] * self.arrow_scale
                    end_y = start_y + motion_vec[1] * self.arrow_scale
                    
                    # Color based on magnitude
                    alpha = min(1.0, magnitude / 3.0)
                    if magnitude < 0.5:
                        color = 'cyan'
                    elif magnitude < 1.5:
                        color = 'yellow'
                    else:
                        color = 'red'
                    
                    ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                               arrowprops=dict(arrowstyle='->', color=color, 
                                             lw=1.5, alpha=alpha))
    
    def draw_object_tracking(self, ax, tracking_objects, tracked_positions, frame_count):
        """Draw object tracking with adaptive bounding boxes and trails."""
        for obj in tracking_objects:
            obj_id = obj['id']
            color = obj['color']
            
            if obj_id not in tracked_positions:
                continue
            
            positions = tracked_positions[obj_id]
            
            # Get positions up to current frame
            current_positions = [pos for pos in positions if pos['global_frame'] <= frame_count]
            
            if not current_positions:
                continue
            
            # Draw trail with size indication
            if len(current_positions) > 1:
                trail_x = [pos['position'][0] for pos in current_positions]
                trail_y = [pos['position'][1] for pos in current_positions]
                
                # Fade trail
                for i in range(len(trail_x)-1):
                    alpha = 0.3 + 0.7 * (i / len(trail_x))
                    ax.plot([trail_x[i], trail_x[i+1]], [trail_y[i], trail_y[i+1]], 
                           color=color, linewidth=3, alpha=alpha)
                
                # In validation mode, also draw ground truth trail
                if self.validation_mode:
                    gt_positions = [pos for pos in current_positions if 'ground_truth_position' in pos]
                    if len(gt_positions) > 1:
                        gt_trail_x = [pos['ground_truth_position'][0] for pos in gt_positions]
                        gt_trail_y = [pos['ground_truth_position'][1] for pos in gt_positions]
                        
                        # Ground truth trail in lighter color with dots
                        for i in range(len(gt_trail_x)-1):
                            alpha = 0.4 + 0.6 * (i / len(gt_trail_x))
                            ax.plot([gt_trail_x[i], gt_trail_x[i+1]], [gt_trail_y[i], gt_trail_y[i+1]], 
                                   color=color, linewidth=2, alpha=alpha, linestyle=':')
            
            # Current position and size
            current_track = current_positions[-1]
            current_pos = current_track['position']
            current_motion = current_track['motion_vector']
            adaptive_size = current_track['adaptive_size']
            original_size = current_track['original_size']
            
            # Draw object center
            ax.plot(current_pos[0], current_pos[1], 'o', color=color, 
                   markersize=12, markeredgecolor='white', markeredgewidth=2)
            
            # In validation mode, draw ground truth center
            if self.validation_mode and 'ground_truth_position' in current_track:
                gt_pos = current_track['ground_truth_position']
                ax.plot(gt_pos[0], gt_pos[1], 's', color=color, 
                       markersize=10, markeredgecolor='white', markeredgewidth=2, alpha=0.7)
                
                # Draw connection line between tracked and ground truth
                ax.plot([current_pos[0], gt_pos[0]], [current_pos[1], gt_pos[1]], 
                       color='red', linewidth=2, alpha=0.8, linestyle='--')
            
            # Draw object's motion vector
            if np.linalg.norm(current_motion) > 0.1:
                ax.arrow(current_pos[0], current_pos[1], 
                        current_motion[0] * 8, current_motion[1] * 8,
                        head_width=8, head_length=8, fc=color, ec='white', 
                        linewidth=2, alpha=0.9)
            
            # Draw adaptive bounding box
            bbox_width, bbox_height = adaptive_size
            bbox_x = current_pos[0] - bbox_width/2
            bbox_y = current_pos[1] - bbox_height/2
            
            # Adaptive bounding box (solid line)
            adaptive_bbox = patches.Rectangle((bbox_x, bbox_y), bbox_width, bbox_height,
                                            linewidth=3, edgecolor=color, facecolor='none', alpha=0.9)
            ax.add_patch(adaptive_bbox)
            
            # Original bounding box (dashed line for comparison)
            orig_width, orig_height = original_size
            orig_bbox_x = current_pos[0] - orig_width/2
            orig_bbox_y = current_pos[1] - orig_height/2
            
            original_bbox = patches.Rectangle((orig_bbox_x, orig_bbox_y), orig_width, orig_height,
                                            linewidth=2, edgecolor=color, facecolor='none', 
                                            alpha=0.5, linestyle='--')
            ax.add_patch(original_bbox)
            
            # Calculate size change percentage
            size_change_w = ((bbox_width - orig_width) / orig_width) * 100
            size_change_h = ((bbox_height - orig_height) / orig_height) * 100
            
            # Label with size information or validation info
            if self.validation_mode and 'tracking_error' in current_track:
                tracking_error = current_track['tracking_error']
                label_text = f'Obj {obj_id}\\nError: {tracking_error:.1f}px\\nW: {size_change_w:+.1f}%\\nH: {size_change_h:+.1f}%'
            else:
                label_text = f'Obj {obj_id}\\nW: {size_change_w:+.1f}%\\nH: {size_change_h:+.1f}%'
            
            ax.text(current_pos[0], current_pos[1] - bbox_height/2 - 15, label_text, 
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                   color='white')
            
            # In validation mode, draw ground truth bounding box
            if self.validation_mode and 'ground_truth_size' in current_track:
                gt_pos = current_track['ground_truth_position']
                gt_size = current_track['ground_truth_size']
                gt_bbox_x = gt_pos[0] - gt_size[0]/2
                gt_bbox_y = gt_pos[1] - gt_size[1]/2
                
                # Ground truth bounding box (dotted line)
                gt_bbox = patches.Rectangle((gt_bbox_x, gt_bbox_y), gt_size[0], gt_size[1],
                                          linewidth=2, edgecolor='white', facecolor='none', 
                                          alpha=0.8, linestyle=':')
                ax.add_patch(gt_bbox)
            
            # Draw motion divergence visualization if available
            if current_track.get('motion_stats'):
                self.draw_motion_divergence_visualization(ax, current_pos, adaptive_size, 
                                                        current_track['motion_stats'], color)
    
    def draw_motion_divergence_visualization(self, ax, center_pos, bbox_size, motion_stats, color):
        """Draw visualization of motion divergence analysis."""
        if not motion_stats:
            return
        
        center_x, center_y = center_pos
        bbox_width, bbox_height = bbox_size
        
        # Draw motion direction indicators around the bounding box
        regions = ['left', 'right', 'top', 'bottom']
        positions = {
            'left': [center_x - bbox_width/3, center_y],
            'right': [center_x + bbox_width/3, center_y],
            'top': [center_x, center_y - bbox_height/3],
            'bottom': [center_x, center_y + bbox_height/3]
        }
        
        for region in regions:
            if region in motion_stats:
                pos = positions[region]
                motion = motion_stats[region]['avg_motion']
                magnitude = motion_stats[region]['magnitude']
                
                if magnitude > 0.2:  # Only show significant motion
                    # Draw small arrow indicating regional motion
                    ax.arrow(pos[0], pos[1], 
                            motion[0] * 4, motion[1] * 4,
                            head_width=4, head_length=4, 
                            fc=color, ec='white', 
                            linewidth=1, alpha=0.6)
                    
                    # Small circle to mark the region
                    ax.plot(pos[0], pos[1], 'o', color='white', 
                           markersize=3, markeredgecolor=color, markeredgewidth=1)


def main():
    """Main function for motion vector enhanced tracking."""
    print("🚀 Motion Vector Enhanced Multi-GOP Tracker")
    print("=" * 80)
    
    # Initialize tracker in validation mode
    tracker = MotionVectorEnhancedTracker(verbose=True, validation_mode=True)
    
    # Load sequence
    if not tracker.load_sequence():
        return 1
    
    # Track objects with motion field
    print(f"\\n🎯 Starting motion vector enhanced tracking...")
    tracking_result = tracker.track_objects_with_motion_field(num_gops=3)
    
    if tracking_result:
        # Create enhanced video
        success = tracker.create_motion_vector_enhanced_video(
            tracking_result, 
            output_path="motion_vector_enhanced_tracking.mp4"
        )
        
        if success:
            print(f"\\n✅ Motion vector enhanced tracking completed!")
            print(f"   📹 Video: motion_vector_enhanced_tracking.mp4")
            print(f"   🎯 Features:")
            print(f"     • Complete motion vector field overlay")
            print(f"     • VALIDATION MODE: Ground truth vs tracked positions")
            print(f"     • Circles = tracked positions, Squares = ground truth")
            print(f"     • Solid trails = tracking, Dotted trails = ground truth")
            print(f"     • White dotted boxes = ground truth bounding boxes")
            print(f"     • Red dashed lines = tracking error visualization")
            print(f"     • Real-time tracking error displayed in pixels")
            print(f"     • Adaptive bounding box sizing based on motion divergence")
            print(f"     • Object tracking with trails and size adaptation")
            print(f"     • Motion vectors color-coded by magnitude")
            print(f"     • 3 consecutive GOPs with size evolution")
            print(f"     • Object motion vectors and regional motion analysis")
            print(f"   ⭐ Use: ffplay motion_vector_enhanced_tracking.mp4 to view")
            
            return 0
        else:
            print(f"\\n❌ Failed to create enhanced video")
            return 1
    else:
        print(f"\\n❌ Enhanced tracking failed")
        return 1


if __name__ == "__main__":
    exit(main())
