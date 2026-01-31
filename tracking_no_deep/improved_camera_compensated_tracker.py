#!/usr/bin/env python3
"""
Improved Camera Motion Compensated Tracker

This fixed version:
1. Tracks all 3 sequences with 3 objects each (9 total objects)
2. Properly loads and displays updated object annotations for each frame
3. Implements better camera motion detection using multiple methods
4. Shows comprehensive multi-panel video with proper object tracking
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
from scipy import stats
from sklearn.linear_model import RANSACRegressor, LinearRegression

# Add path exactly like residuals study
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

def estimate_global_camera_motion(motion_data, frame_idx, method='robust_median'):
    """
    Improved global camera motion estimation.
    
    Args:
        motion_data: Motion vector data [frames, types, rows, cols, 2]
        frame_idx: Current frame index
        method: 'robust_median', 'ransac_consensus', or 'mode_analysis'
    
    Returns:
        Estimated camera motion (dx, dy) and confidence score
    """
    if frame_idx >= motion_data.shape[0]:
        return (0.0, 0.0), 0.0
    
    # Extract motion vectors for current frame (use both types for better estimation)
    mv_x_type0 = motion_data[frame_idx, 0, :, :, 0]  # I-frame/baseline motion
    mv_y_type0 = motion_data[frame_idx, 0, :, :, 1]
    mv_x_type1 = motion_data[frame_idx, 1, :, :, 0]  # P-frame motion
    mv_y_type1 = motion_data[frame_idx, 1, :, :, 1]
    
    # Combine motion vectors from both types for more robust estimation
    all_mv_x = np.concatenate([mv_x_type0.flatten(), mv_x_type1.flatten()])
    all_mv_y = np.concatenate([mv_y_type0.flatten(), mv_y_type1.flatten()])
    
    # Filter valid motion vectors (non-zero with reasonable magnitude)
    magnitude_threshold = 0.5  # Minimum motion to consider
    max_magnitude = 50.0  # Maximum reasonable motion
    
    magnitudes = np.sqrt(all_mv_x**2 + all_mv_y**2)
    valid_mask = (magnitudes > magnitude_threshold) & (magnitudes < max_magnitude)
    
    mv_x_valid = all_mv_x[valid_mask]
    mv_y_valid = all_mv_y[valid_mask]
    
    if len(mv_x_valid) < 20:  # Need sufficient vectors for reliable estimation
        return (0.0, 0.0), 0.0
    
    if method == 'robust_median':
        # Robust median-based estimation with outlier rejection
        median_x = np.median(mv_x_valid)
        median_y = np.median(mv_y_valid)
        
        # Calculate deviations from median
        dev_x = np.abs(mv_x_valid - median_x)
        dev_y = np.abs(mv_y_valid - median_y)
        
        # Use MAD (Median Absolute Deviation) for robust thresholding
        mad_x = np.median(dev_x)
        mad_y = np.median(dev_y)
        
        # Select vectors within 2*MAD of median (roughly 95% for normal distribution)
        threshold_factor = 2.5
        consistent_mask = (dev_x < threshold_factor * mad_x) & (dev_y < threshold_factor * mad_y)
        
        if np.sum(consistent_mask) > len(mv_x_valid) * 0.3:  # At least 30% consensus
            camera_dx = np.mean(mv_x_valid[consistent_mask])
            camera_dy = np.mean(mv_y_valid[consistent_mask])
            confidence = np.sum(consistent_mask) / len(mv_x_valid)
        else:
            camera_dx, camera_dy = median_x, median_y
            confidence = 0.3
            
    elif method == 'ransac_consensus':
        # RANSAC-based consensus estimation
        try:
            # Prepare data for RANSAC
            X = np.ones((len(mv_x_valid), 1))  # Constant global motion model
            
            # RANSAC for X and Y components
            ransac_x = RANSACRegressor(
                LinearRegression(), 
                residual_threshold=3.0, 
                max_trials=100,
                random_state=42
            )
            ransac_y = RANSACRegressor(
                LinearRegression(), 
                residual_threshold=3.0, 
                max_trials=100,
                random_state=42
            )
            
            ransac_x.fit(X, mv_x_valid.reshape(-1, 1))
            ransac_y.fit(X, mv_y_valid.reshape(-1, 1))
            
            camera_dx = float(ransac_x.estimator_.intercept_[0])
            camera_dy = float(ransac_y.estimator_.intercept_[0])
            
            # Confidence based on inlier percentage
            inlier_ratio_x = np.sum(ransac_x.inlier_mask_) / len(mv_x_valid)
            inlier_ratio_y = np.sum(ransac_y.inlier_mask_) / len(mv_y_valid)
            confidence = (inlier_ratio_x + inlier_ratio_y) / 2
            
        except Exception as e:
            # Fallback to median
            camera_dx = np.median(mv_x_valid)
            camera_dy = np.median(mv_y_valid)
            confidence = 0.2
            
    elif method == 'mode_analysis':
        # Histogram mode analysis for dominant motion
        try:
            # Create 2D histogram of motion vectors
            bins = 25
            hist_range = [[-20, 20], [-20, 20]]  # Reasonable motion range
            
            hist, x_edges, y_edges = np.histogram2d(mv_x_valid, mv_y_valid, 
                                                   bins=bins, range=hist_range)
            
            # Find the bin with maximum count (mode)
            max_bin_idx = np.unravel_index(np.argmax(hist), hist.shape)
            
            # Camera motion is center of dominant bin
            camera_dx = (x_edges[max_bin_idx[0]] + x_edges[max_bin_idx[0] + 1]) / 2
            camera_dy = (y_edges[max_bin_idx[1]] + y_edges[max_bin_idx[1] + 1]) / 2
            
            # Confidence based on dominance of the mode
            total_vectors = len(mv_x_valid)
            mode_count = hist[max_bin_idx]
            confidence = mode_count / total_vectors
            
        except Exception as e:
            camera_dx = np.median(mv_x_valid)
            camera_dy = np.median(mv_y_valid)
            confidence = 0.1
    
    else:
        raise ValueError(f"Unknown camera motion estimation method: {method}")
    
    # Apply reasonableness checks
    if abs(camera_dx) > 30 or abs(camera_dy) > 30:  # Unreasonably large motion
        confidence *= 0.5
    
    return (float(camera_dx), float(camera_dy)), float(confidence)

def get_macroblocks_for_object(object_info):
    """Generate macroblock grid for object bounding box."""
    x1, y1, x2, y2 = object_info['bbox']
    mb_size = 16
    
    # Calculate macroblock grid boundaries
    mb_start_col = int(x1 // mb_size)
    mb_end_col = int(x2 // mb_size)
    mb_start_row = int(y1 // mb_size)
    mb_end_row = int(y2 // mb_size)
    
    macroblocks = []
    for mb_row in range(mb_start_row, mb_end_row + 1):
        for mb_col in range(mb_start_col, mb_end_col + 1):
            # Only include macroblocks that significantly overlap with bbox
            mb_x1 = mb_col * mb_size
            mb_y1 = mb_row * mb_size
            mb_x2 = mb_x1 + mb_size
            mb_y2 = mb_y1 + mb_size
            
            # Check overlap with object bbox
            overlap_x1 = max(mb_x1, x1)
            overlap_y1 = max(mb_y1, y1)
            overlap_x2 = min(mb_x2, x2)
            overlap_y2 = min(mb_y2, y2)
            
            if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                mb_area = mb_size * mb_size
                
                # Include if at least 25% overlap
                if overlap_area / mb_area >= 0.25:
                    mb_center_x = mb_x1 + mb_size // 2
                    mb_center_y = mb_y1 + mb_size // 2
                    
                    macroblocks.append({
                        'position': (mb_col, mb_row),
                        'pixel_center': (mb_center_x, mb_center_y),
                        'bounds': (mb_x1, mb_y1, mb_x2, mb_y2),
                        'overlap_ratio': overlap_area / mb_area
                    })
    
    return macroblocks

def load_frame_annotations(data_loader, gop_idx=0):
    """Load and parse frame annotations for object tracking."""
    try:
        annotation_data = data_loader.load_corrected_annotations(gop_idx)
        if annotation_data is None:
            annotation_data = data_loader.load_annotations(gop_idx)
        return annotation_data
    except Exception as e:
        print(f"   ⚠️  Could not load annotations: {e}")
        return None

def parse_frame_annotations(annotation_data, frame_idx):
    """Parse annotations for a specific frame."""
    frame_objects = []
    
    if annotation_data is None:
        return frame_objects
    
    try:
        if hasattr(annotation_data, 'files') and 'annotations' in annotation_data.files:
            annotations = annotation_data['annotations']
            if frame_idx < len(annotations):
                frame_annotations = annotations[frame_idx]
                
                for ann in frame_annotations:
                    if len(ann) >= 6:
                        obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                        
                        # Convert normalized coordinates to pixel coordinates
                        frame_width = frame_height = 960
                        x_center = x_norm * frame_width
                        y_center = y_norm * frame_height
                        width = w_norm * frame_width
                        height = h_norm * frame_height
                        
                        # Calculate bbox coordinates
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        x2 = x_center + width/2
                        y2 = y_center + height/2
                        
                        frame_objects.append({
                            'id': int(obj_id),
                            'class': int(obj_class),
                            'bbox': [x1, y1, x2, y2],
                            'center': [x_center, y_center],
                            'size': [width, height]
                        })
                        
    except Exception as e:
        print(f"   ⚠️  Frame {frame_idx}: Annotation parsing error: {e}")
    
    return frame_objects

def select_top_objects_from_sequences(factory, num_sequences=3, objects_per_sequence=3):
    """Select top objects from multiple sequences."""
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    
    if len(sequences) < num_sequences:
        num_sequences = len(sequences)
    
    selected_sequences = []
    
    for seq_idx in range(num_sequences):
        if seq_idx >= len(sequences):
            break
            
        sequence_name = sequences[seq_idx]
        print(f"📁 Sequence {seq_idx + 1}: {sequence_name}")
        
        data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
        
        # Load first frame annotations to find objects using same pattern as working tracker
        try:
            annotation_data = data_loader.load_corrected_annotations(0)
            if annotation_data is None:
                annotation_data = data_loader.load_annotations(0)
        except:
            annotation_data = None
        
        first_frame_objects = parse_frame_annotations(annotation_data, frame_idx=0)
        
        if first_frame_objects:
            print(f"   Found {len(first_frame_objects)} annotations in first frame")
            
            # Calculate areas and select top objects
            for obj in first_frame_objects:
                x1, y1, x2, y2 = obj['bbox']
                obj['area'] = (x2 - x1) * (y2 - y1)
            
            # Sort by area and select top objects
            first_frame_objects.sort(key=lambda x: x['area'], reverse=True)
            selected_objects = first_frame_objects[:objects_per_sequence]
            
            for i, obj in enumerate(selected_objects):
                x1, y1, x2, y2 = obj['bbox']
                width, height = x2 - x1, y2 - y1
                print(f"   🎯 Object {i+1}: ID={obj['id']}, Area={obj['area']:.0f}, Size={width:.0f}x{height:.0f}")
            
            selected_sequences.append({
                'sequence_index': seq_idx,
                'sequence_name': sequence_name,
                'data_loader': data_loader,
                'objects': selected_objects,
                'annotation_data': annotation_data
            })
        else:
            print(f"   ❌ No objects found in first frame")
    
    return selected_sequences

def track_object_with_camera_compensation(data_loader, object_info, object_idx, sequence_idx, gop_idx=0):
    """Track object with improved camera motion compensation and annotation updates."""
    print(f"\\n🎥 Camera-Compensated Tracking: Object {object_idx+1} in Sequence {sequence_idx+1}")
    
    # Get initial macroblocks for this object
    macroblocks = get_macroblocks_for_object(object_info)
    total_macroblocks = len(macroblocks)
    
    if total_macroblocks == 0:
        print(f"   ❌ No macroblocks found for object {object_info['id']}")
        return None
    
    print(f"   📦 Object {object_info['id']}: {total_macroblocks} initial macroblocks")
    
    # Load data
    motion_data = data_loader.load_motion_vectors(gop_idx)
    rgb_data = data_loader.load_rgb_frames(gop_idx, 'pframe')
    annotation_data = load_frame_annotations(data_loader, gop_idx)
    
    if motion_data is None:
        print(f"   ❌ Missing motion data for GOP {gop_idx}")
        return None
    
    gop_frames = min(motion_data.shape[0], 45)
    print(f"   📊 Data loaded: motion={motion_data.shape}, processing {gop_frames} frames")
    
    # Initialize tracking system
    tracking_system = {
        'object_info': object_info,
        'sequence_idx': sequence_idx,
        'object_idx': object_idx,
        'gop_frames': gop_frames,
        'frames_data': [],
        'annotation_data': annotation_data
    }
    
    # Process each frame with camera motion compensation
    for frame_idx in range(gop_frames):
        print(f"   🎬 Frame {frame_idx+1}/{gop_frames}: Estimating camera motion...", end='')
        
        # 1. Estimate global camera motion for this frame
        camera_motion, confidence = estimate_global_camera_motion(
            motion_data, frame_idx, method='robust_median'
        )
        
        # 2. Load frame annotations to get updated object position
        frame_objects = parse_frame_annotations(annotation_data, frame_idx)
        current_object = None
        
        # Find our tracked object in current frame
        for obj in frame_objects:
            if obj['id'] == object_info['id']:
                current_object = obj
                break
        
        # If object not found, use previous position or original
        if current_object is None:
            if frame_idx > 0 and tracking_system['frames_data']:
                # Use previous frame's object position
                prev_frame = tracking_system['frames_data'][-1]
                if prev_frame['current_object']:
                    current_object = prev_frame['current_object'].copy()
                else:
                    current_object = object_info.copy()
            else:
                current_object = object_info.copy()
        
        # 3. Update macroblocks based on current object position
        if current_object != object_info:
            # Recalculate macroblocks for updated position
            updated_macroblocks = get_macroblocks_for_object(current_object)
            if updated_macroblocks:  # Use updated if available
                macroblocks = updated_macroblocks
        
        print(f" Camera: ({camera_motion[0]:.1f}, {camera_motion[1]:.1f}) confidence: {confidence:.2f}")
        
        # 4. Extract motion vectors for all macroblocks
        raw_motion_vectors = []
        compensated_vectors = []
        relative_magnitudes = []
        
        for mb_info in macroblocks:
            mb_col, mb_row = mb_info['position']
            
            # Check bounds to prevent index errors
            if (mb_row >= motion_data.shape[2] or mb_col >= motion_data.shape[3] or 
                mb_row < 0 or mb_col < 0):
                # Skip out of bounds macroblocks
                continue
            
            # Get motion vectors (prefer type1, fallback to type0)
            mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
            mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
            
            if abs(mv_x_type1) < 0.1 and abs(mv_y_type1) < 0.1:
                mv_x = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
                mv_y = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
            else:
                mv_x, mv_y = mv_x_type1, mv_y_type1
            
            raw_motion_vectors.append((mv_x, mv_y))
            
            # 5. Compensate for camera motion
            comp_x = mv_x - camera_motion[0]
            comp_y = mv_y - camera_motion[1]
            compensated_vectors.append((comp_x, comp_y))
            
            # Calculate relative motion magnitude
            relative_mag = np.sqrt(comp_x**2 + comp_y**2)
            relative_magnitudes.append(relative_mag)
        
        # 6. Adaptive macroblock selection
        if confidence > 0.4:  # High confidence camera motion
            selection_strategy = 'motion_guided'
            threshold = np.percentile(relative_magnitudes, 60) if relative_magnitudes else 1.0
        else:  # Low confidence, be more conservative
            selection_strategy = 'dynamic'
            threshold = np.percentile(relative_magnitudes, 75) if relative_magnitudes else 1.0
        
        selected_indices = [i for i, mag in enumerate(relative_magnitudes) 
                          if mag > max(0.3, threshold * 0.4)]
        
        # Ensure minimum coverage
        if len(selected_indices) < max(5, len(macroblocks) * 0.2):
            sorted_indices = np.argsort(relative_magnitudes)[::-1]
            for i in sorted_indices[:max(8, len(macroblocks) // 3)]:
                if i not in selected_indices:
                    selected_indices.append(i)
        
        print(f"      📍 Selected {len(selected_indices)}/{total_macroblocks} macroblocks using {selection_strategy}")
        
        # 7. Store frame data
        frame_data = {
            'frame_idx': frame_idx,
            'camera_motion': camera_motion,
            'camera_confidence': confidence,
            'current_object': current_object,
            'frame_objects': frame_objects,
            'raw_motion_vectors': raw_motion_vectors,
            'compensated_vectors': compensated_vectors,
            'relative_magnitudes': relative_magnitudes,
            'selected_indices': selected_indices,
            'selection_strategy': selection_strategy,
            'macroblocks': macroblocks,
            'rgb_frame': rgb_data[frame_idx] if rgb_data is not None and frame_idx < rgb_data.shape[0] else None
        }
        
        tracking_system['frames_data'].append(frame_data)
    
    # Calculate summary statistics
    avg_camera_motion = np.mean([fd['camera_motion'] for fd in tracking_system['frames_data']], axis=0)
    avg_confidence = np.mean([fd['camera_confidence'] for fd in tracking_system['frames_data']])
    avg_selected = np.mean([len(fd['selected_indices']) for fd in tracking_system['frames_data']])
    
    # Analyze motion patterns
    camera_magnitudes = [np.sqrt(fd['camera_motion'][0]**2 + fd['camera_motion'][1]**2) 
                        for fd in tracking_system['frames_data']]
    avg_camera_magnitude = np.mean(camera_magnitudes)
    
    tracking_system['summary_stats'] = {
        'avg_camera_motion': avg_camera_motion,
        'avg_camera_confidence': avg_confidence,
        'avg_selected_macroblocks': avg_selected,
        'avg_camera_magnitude': avg_camera_magnitude,
        'total_frames': gop_frames
    }
    
    motion_type = ("🎥 Camera-dominant" if avg_camera_magnitude > 3.0 
                  else "🎯 Object-dominant" if avg_camera_magnitude > 0.8 
                  else "⚪ Static scene")
    
    print(f"   ✅ Camera compensation complete:")
    print(f"      📈 Avg camera motion: ({avg_camera_motion[0]:.1f}, {avg_camera_motion[1]:.1f}) | {motion_type}")
    print(f"      🎯 Avg confidence: {avg_confidence:.2f}")
    print(f"      📍 Avg selected macroblocks: {avg_selected:.1f}/{total_macroblocks}")
    
    return tracking_system

def create_improved_camera_compensated_video(all_tracking_results, output_filename="improved_camera_compensated_tracking.mp4"):
    """Create improved video showing camera motion compensation with all 3 sequences."""
    print(f"\\n🎬 Creating Improved Camera Motion Compensated Video")
    print("=" * 60)
    
    if not all_tracking_results:
        print("❌ No tracking results to visualize")
        return False
    
    # Create figure with 3x3 grid for 9 objects
    fig = plt.figure(figsize=(24, 18))
    
    # Determine max frames
    max_frames = max(
        obj_result['gop_frames'] for seq_results in all_tracking_results
        for obj_result in seq_results if obj_result
    )
    
    def animate(frame_num):
        fig.clear()
        
        subplot_idx = 1
        
        for seq_idx, seq_results in enumerate(all_tracking_results):
            for obj_idx, tracking_result in enumerate(seq_results):
                if tracking_result is None or frame_num >= len(tracking_result['frames_data']):
                    continue
                
                ax = fig.add_subplot(3, 3, subplot_idx)
                subplot_idx += 1
                
                frame_width = frame_height = 960
                ax.set_xlim(0, frame_width)
                ax.set_ylim(0, frame_height)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                
                # Get frame data
                frame_data = tracking_result['frames_data'][frame_num]
                original_object = tracking_result['object_info']
                current_object = frame_data['current_object']
                
                # Display RGB background
                if frame_data['rgb_frame'] is not None:
                    try:
                        ax.imshow(frame_data['rgb_frame'], extent=[0, frame_width, frame_height, 0], alpha=0.7)
                    except:
                        ax.set_facecolor('black')
                else:
                    ax.set_facecolor('black')
                
                # Draw current object bounding box (updated position)
                if current_object and 'bbox' in current_object:
                    x1, y1, x2, y2 = current_object['bbox']
                    colors = ['yellow', 'cyan', 'orange', 'lime', 'magenta', 'red', 'blue', 'pink', 'white']
                    color = colors[current_object['id'] % len(colors)]
                    
                    bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=3, edgecolor=color, facecolor='none',
                                                linestyle='-', alpha=0.9)
                    ax.add_patch(bbox_rect)
                    
                    # Object ID label
                    ax.text(x1 + 5, y1 + 20, f"ID:{current_object['id']}", 
                           fontsize=10, color=color, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
                
                # Draw selected macroblocks and their motion vectors
                selected_indices = frame_data['selected_indices']
                compensated_vectors = frame_data['compensated_vectors']
                relative_magnitudes = frame_data['relative_magnitudes']
                macroblocks = frame_data['macroblocks']
                
                for i in selected_indices:
                    if i >= len(macroblocks):
                        continue
                        
                    mb_info = macroblocks[i]
                    mb_x, mb_y = mb_info['pixel_center']
                    
                    # Color based on motion magnitude
                    if i < len(relative_magnitudes):
                        rel_mag = relative_magnitudes[i]
                        if rel_mag > 3.0:
                            mb_color, mb_size = 'red', 8
                        elif rel_mag > 1.5:
                            mb_color, mb_size = 'orange', 6
                        else:
                            mb_color, mb_size = 'yellow', 4
                    else:
                        mb_color, mb_size = 'white', 4
                    
                    # Draw macroblock position
                    ax.plot(mb_x, mb_y, 'o', color=mb_color, markersize=mb_size, 
                           markeredgecolor='white', markeredgewidth=1, alpha=0.8)
                    
                    # Draw compensated motion vector (red arrows)
                    if i < len(compensated_vectors) and i < len(relative_magnitudes):
                        comp_x, comp_y = compensated_vectors[i]
                        rel_mag = relative_magnitudes[i]
                        
                        if rel_mag > 0.2:  # Only draw significant motion
                            # Scale for visibility
                            scale = 4.0 if rel_mag < 1.0 else 3.0
                            arrow_dx = comp_x * scale
                            arrow_dy = comp_y * scale
                            
                            # Red arrow for compensated object motion
                            ax.arrow(mb_x, mb_y, arrow_dx, arrow_dy,
                                    head_width=6, head_length=6, 
                                    fc='red', ec='red', alpha=0.9, linewidth=2)
                
                # Draw global camera motion indicator
                camera_motion = frame_data['camera_motion']
                confidence = frame_data['camera_confidence']
                
                if confidence > 0.2:  # Show camera motion if detected
                    # Draw camera motion vector at top-right
                    cam_start_x, cam_start_y = frame_width - 120, 50
                    cam_scale = 15.0  # Larger scale for visibility
                    cam_dx = camera_motion[0] * cam_scale
                    cam_dy = camera_motion[1] * cam_scale
                    
                    # Camera motion in blue
                    ax.arrow(cam_start_x, cam_start_y, cam_dx, cam_dy,
                            head_width=10, head_length=10, 
                            fc='cyan', ec='blue', alpha=0.8, linewidth=3)
                    
                    ax.text(cam_start_x - 30, cam_start_y - 25, 
                           f'Cam: {confidence:.2f}', 
                           fontsize=8, color='cyan', fontweight='bold')
                
                # Motion analysis
                strategy = frame_data['selection_strategy']
                avg_rel_motion = np.mean(relative_magnitudes) if relative_magnitudes else 0
                camera_magnitude = np.sqrt(camera_motion[0]**2 + camera_motion[1]**2)
                
                motion_type = ("🎥 Camera" if camera_magnitude > 3.0 
                              else "🎯 Object" if avg_rel_motion > 2.0 else "⚪ Static")
                
                title = (f"Seq {seq_idx+1} Obj {obj_idx+1} (ID:{original_object['id']})\\n"
                        f"Frame {frame_num+1}/{tracking_result['gop_frames']} | {motion_type}\\n"
                        f"{strategy} | {len(selected_indices)}/{len(macroblocks)} MBs")
                
                ax.set_title(title, fontsize=9, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Main title
        fig.suptitle(f'Improved Camera Motion Compensated Tracking (3 Sequences × 3 Objects)\\n'
                    f'Frame {frame_num+1}/{max_frames} | Red=Object Motion | Blue=Camera Motion | Updated Object Positions', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
    
    # Create animation
    print(f"📹 Animating {max_frames} frames with improved camera motion compensation...")
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                  interval=1000//7, blit=False, repeat=True)
    
    # Save video
    try:
        writer = animation.FFMpegWriter(fps=7, metadata=dict(artist='Improved Camera Compensated Tracker'),
                                      bitrate=10000)
        anim.save(output_filename, writer=writer, dpi=100)
        print(f"✅ Improved camera compensated video saved: {output_filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False
    finally:
        plt.close(fig)

def main():
    """Main function for improved camera motion compensated tracking."""
    print("🎥 Improved Camera Motion Compensated Tracker")
    print("=" * 60)
    print("🔴 Red arrows = Object motion (camera compensated)")
    print("🔵 Blue arrows = Estimated camera motion")
    print("📍 Color-coded macroblocks = Motion magnitude")
    print("📦 Updated bounding boxes = Real object tracking")
    print("=" * 60)
    
    # Initialize data loader factory
    factory = MOTSDataLoaderFactory(verbose=False)
    
    # Select 3 sequences with 3 objects each (9 total objects)
    selected_sequences = select_top_objects_from_sequences(factory, num_sequences=3, objects_per_sequence=3)
    
    if not selected_sequences:
        print("❌ No sequences available")
        return False
    
    print(f"\\n🎯 Selected {len(selected_sequences)} sequences with 3 objects each for improved camera compensation")
    
    # Track with improved camera compensation
    all_tracking_results = []
    
    for seq_data in selected_sequences:
        seq_idx = seq_data['sequence_index']
        data_loader = seq_data['data_loader']
        objects = seq_data['objects']
        
        print(f"\\n📁 Processing Sequence {seq_idx + 1}: {seq_data['sequence_name']}")
        
        sequence_tracking_results = []
        
        for obj_idx, object_info in enumerate(objects):
            tracking_result = track_object_with_camera_compensation(
                data_loader, object_info, obj_idx, seq_idx, gop_idx=0
            )
            sequence_tracking_results.append(tracking_result)
        
        all_tracking_results.append(sequence_tracking_results)
    
    # Create improved video
    success = create_improved_camera_compensated_video(all_tracking_results)
    
    if success:
        print("\\n🎉 Improved Camera Motion Compensated Tracking completed!")
        print("\\n📊 Comprehensive Camera Motion Analysis:")
        
        total_objects = 0
        total_camera_motion = 0
        total_confidence = 0
        
        for seq_idx, seq_results in enumerate(all_tracking_results):
            print(f"\\n   📁 Sequence {seq_idx + 1}:")
            for obj_idx, result in enumerate(seq_results):
                if result:
                    stats = result['summary_stats']
                    total_objects += 1
                    total_camera_motion += stats['avg_camera_magnitude']
                    total_confidence += stats['avg_camera_confidence']
                    
                    motion_classification = ("🎥 Camera-dominant" if stats['avg_camera_magnitude'] > 3.0 
                                           else "🎯 Object-dominant" if stats['avg_camera_magnitude'] > 0.8 
                                           else "⚪ Static scene")
                    
                    print(f"      🎯 Object {obj_idx + 1}: {motion_classification}")
                    print(f"         Camera motion: ({stats['avg_camera_motion'][0]:.1f}, {stats['avg_camera_motion'][1]:.1f})px")
                    print(f"         Confidence: {stats['avg_camera_confidence']:.2f}")
                    print(f"         Adaptive selection: {stats['avg_selected_macroblocks']:.1f} macroblocks/frame")
        
        avg_camera_motion = total_camera_motion / total_objects if total_objects > 0 else 0
        avg_confidence = total_confidence / total_objects if total_objects > 0 else 0
        
        print(f"\\n📈 Overall Analysis:")
        print(f"   Total objects tracked: {total_objects}")
        print(f"   Average camera motion magnitude: {avg_camera_motion:.2f}px")
        print(f"   Average detection confidence: {avg_confidence:.2f}")
        
        print("\\n🔬 Enhanced Features:")
        print("   ✅ All 3 sequences with 3 objects each (9 total objects)")
        print("   ✅ Frame-by-frame object position updates from annotations")
        print("   ✅ Improved camera motion detection with robust algorithms")
        print("   ✅ Adaptive macroblock selection based on motion patterns")
        print("   ✅ Motion vector compensation for object-relative motion")
        print("   ✅ Confidence-based algorithm switching")
        print("   ✅ Comprehensive 3x3 video panel display")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create improved camera motion compensated tracking video")
