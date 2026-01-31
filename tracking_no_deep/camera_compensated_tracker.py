#!/usr/bin/env python3
"""
Camera Motion Compensated Tracker

This tracker implements camera motion compensation by:
1. Estimating global camera motion from the motion vector field
2. Subtracting camera motion from individual macroblock motion vectors
3. Adapting macroblock selection based on compensated motion
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

def estimate_camera_motion(motion_data, frame_idx, method='median_filter'):
    """
    Estimate global camera motion from the motion vector field.
    
    Args:
        motion_data: Motion vector data [frames, types, rows, cols, 2]
        frame_idx: Current frame index
        method: 'median_filter', 'ransac', or 'dominant_mode'
    
    Returns:
        Estimated camera motion (dx, dy) and confidence score
    """
    if frame_idx >= motion_data.shape[0]:
        return (0.0, 0.0), 0.0
    
    # Extract motion vectors for current frame
    # Use type 1 (P-frame motion vectors) as they're more reliable
    mv_x = motion_data[frame_idx, 1, :, :, 0]  # Shape: (rows, cols)
    mv_y = motion_data[frame_idx, 1, :, :, 1]
    
    # Flatten motion vectors and filter out zero/invalid vectors
    mv_x_flat = mv_x.flatten()
    mv_y_flat = mv_y.flatten()
    
    # Filter out zero motion vectors and outliers
    valid_mask = (np.abs(mv_x_flat) > 0.1) | (np.abs(mv_y_flat) > 0.1)
    mv_x_valid = mv_x_flat[valid_mask]
    mv_y_valid = mv_y_flat[valid_mask]
    
    if len(mv_x_valid) < 10:  # Not enough valid motion vectors
        return (0.0, 0.0), 0.0
    
    if method == 'median_filter':
        # Simple robust estimation using median
        camera_dx = np.median(mv_x_valid)
        camera_dy = np.median(mv_y_valid)
        
        # Confidence based on how many vectors are close to the median
        diff_x = np.abs(mv_x_valid - camera_dx)
        diff_y = np.abs(mv_y_valid - camera_dy)
        close_vectors = np.sum((diff_x < 2.0) & (diff_y < 2.0))
        confidence = close_vectors / len(mv_x_valid)
        
    elif method == 'ransac':
        # RANSAC-based robust estimation
        try:
            # Prepare data for RANSAC (we assume constant global motion)
            X = np.ones((len(mv_x_valid), 1))  # Constant model
            y_x = mv_x_valid.reshape(-1, 1)
            y_y = mv_y_valid.reshape(-1, 1)
            
            # RANSAC for X motion
            ransac_x = RANSACRegressor(LinearRegression(), residual_threshold=2.0, random_state=42)
            ransac_x.fit(X, y_x)
            camera_dx = ransac_x.estimator_.intercept_[0]
            
            # RANSAC for Y motion
            ransac_y = RANSACRegressor(LinearRegression(), residual_threshold=2.0, random_state=42)
            ransac_y.fit(X, y_y)
            camera_dy = ransac_y.estimator_.intercept_[0]
            
            # Confidence based on inlier ratio
            confidence = (np.sum(ransac_x.inlier_mask_) + np.sum(ransac_y.inlier_mask_)) / (2 * len(mv_x_valid))
            
        except:
            # Fallback to median if RANSAC fails
            camera_dx = np.median(mv_x_valid)
            camera_dy = np.median(mv_y_valid)
            confidence = 0.5
            
    elif method == 'dominant_mode':
        # Find dominant motion mode using histogram analysis
        # Bin the motion vectors
        bins = 20
        hist_x, edges_x = np.histogram(mv_x_valid, bins=bins)
        hist_y, edges_y = np.histogram(mv_y_valid, bins=bins)
        
        # Find dominant bins
        dominant_bin_x = np.argmax(hist_x)
        dominant_bin_y = np.argmax(hist_y)
        
        # Camera motion is the center of dominant bins
        camera_dx = (edges_x[dominant_bin_x] + edges_x[dominant_bin_x + 1]) / 2
        camera_dy = (edges_y[dominant_bin_y] + edges_y[dominant_bin_y + 1]) / 2
        
        # Confidence based on how dominant the mode is
        confidence = (hist_x[dominant_bin_x] + hist_y[dominant_bin_y]) / (2 * len(mv_x_valid))
    
    else:
        raise ValueError(f"Unknown camera motion estimation method: {method}")
    
    return (float(camera_dx), float(camera_dy)), float(confidence)

def compensate_motion_vectors(motion_vectors, camera_motion):
    """
    Compensate motion vectors by subtracting estimated camera motion.
    
    Args:
        motion_vectors: List of (mv_x, mv_y) tuples
        camera_motion: Tuple of (camera_dx, camera_dy)
    
    Returns:
        Compensated motion vectors and relative motion magnitudes
    """
    camera_dx, camera_dy = camera_motion
    compensated_vectors = []
    relative_magnitudes = []
    
    for mv_x, mv_y in motion_vectors:
        # Subtract camera motion to get object-relative motion
        comp_x = mv_x - camera_dx
        comp_y = mv_y - camera_dy
        
        # Calculate relative motion magnitude
        relative_mag = np.sqrt(comp_x**2 + comp_y**2)
        
        compensated_vectors.append((comp_x, comp_y))
        relative_magnitudes.append(relative_mag)
    
    return compensated_vectors, relative_magnitudes

def adaptive_macroblock_selection(object_info, compensated_vectors, relative_magnitudes, selection_strategy='dynamic'):
    """
    Adaptively select macroblocks based on compensated motion vectors.
    
    Args:
        object_info: Object bounding box information
        compensated_vectors: Motion vectors after camera compensation
        relative_magnitudes: Magnitudes of compensated motion vectors
        selection_strategy: 'static', 'dynamic', or 'motion_guided'
    
    Returns:
        Selected macroblock indices and their properties
    """
    if selection_strategy == 'static':
        # Original static selection - all macroblocks in bbox
        return list(range(len(compensated_vectors))), ['static'] * len(compensated_vectors)
    
    elif selection_strategy == 'dynamic':
        # Select macroblocks with significant object motion after compensation
        threshold = np.percentile(relative_magnitudes, 75) if relative_magnitudes else 1.0
        selected_indices = [i for i, mag in enumerate(relative_magnitudes) if mag > max(0.5, threshold * 0.3)]
        properties = ['dynamic_high_motion' if relative_magnitudes[i] > threshold else 'dynamic_low_motion' 
                     for i in selected_indices]
        
        # Ensure we have at least some macroblocks
        if len(selected_indices) < 5:
            # Add some of the highest motion macroblocks
            sorted_indices = np.argsort(relative_magnitudes)[::-1]
            for i in sorted_indices[:10]:
                if i not in selected_indices:
                    selected_indices.append(i)
                    properties.append('dynamic_fallback')
        
        return selected_indices, properties
    
    elif selection_strategy == 'motion_guided':
        # Sophisticated selection based on motion patterns
        if not relative_magnitudes:
            return list(range(len(compensated_vectors))), ['motion_guided'] * len(compensated_vectors)
        
        # Classify macroblocks by motion characteristics
        selected_indices = []
        properties = []
        
        mean_motion = np.mean(relative_magnitudes)
        std_motion = np.std(relative_magnitudes)
        
        for i, (mag, (comp_x, comp_y)) in enumerate(zip(relative_magnitudes, compensated_vectors)):
            # Select based on multiple criteria
            is_significant_motion = mag > mean_motion + 0.5 * std_motion
            is_consistent_direction = abs(comp_x) > 0.5 or abs(comp_y) > 0.5
            is_not_noise = mag > 0.3
            
            if is_significant_motion and is_consistent_direction and is_not_noise:
                selected_indices.append(i)
                if mag > mean_motion + std_motion:
                    properties.append('motion_guided_high')
                else:
                    properties.append('motion_guided_medium')
            elif is_not_noise:
                # Include some low-motion macroblocks for stability
                if len([p for p in properties if 'low' in p]) < len(compensated_vectors) * 0.3:
                    selected_indices.append(i)
                    properties.append('motion_guided_low')
        
        # Ensure minimum coverage
        if len(selected_indices) < 8:
            sorted_indices = np.argsort(relative_magnitudes)[::-1]
            for i in sorted_indices[:15]:
                if i not in selected_indices:
                    selected_indices.append(i)
                    properties.append('motion_guided_coverage')
        
        return selected_indices, properties
    
    else:
        raise ValueError(f"Unknown selection strategy: {selection_strategy}")

def track_with_camera_compensation(data_loader, object_info, object_idx, sequence_idx, gop_idx=0):
    """Track object with camera motion compensation."""
    print(f"\n🎥 Camera-Compensated Tracking: Object {object_idx+1} in Sequence {sequence_idx+1}")
    
    # Get initial macroblocks for this object
    from enhanced_motion_vector_tracker import get_macroblocks_for_object
    macroblocks = get_macroblocks_for_object(object_info)
    total_macroblocks = len(macroblocks)
    
    if total_macroblocks == 0:
        print(f"   ❌ No macroblocks found for object {object_info['id']}")
        return None
    
    print(f"   📦 Object {object_info['id']}: {total_macroblocks} initial macroblocks")
    
    # Load data
    motion_data = data_loader.load_motion_vectors(gop_idx)
    rgb_data = data_loader.load_rgb_frames(gop_idx, 'pframe')
    
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
        'frames_data': []
    }
    
    # Process each frame with camera motion compensation
    for frame_idx in range(gop_frames):
        print(f"   🎬 Frame {frame_idx+1}/{gop_frames}: Estimating camera motion...", end='')
        
        # 1. Estimate camera motion for this frame
        camera_motion, confidence = estimate_camera_motion(motion_data, frame_idx, method='median_filter')
        print(f" Camera: ({camera_motion[0]:.1f}, {camera_motion[1]:.1f}) confidence: {confidence:.2f}")
        
        # 2. Extract motion vectors for all macroblocks
        raw_motion_vectors = []
        for mb_info in macroblocks:
            mb_col, mb_row = mb_info['position']
            
            # Get motion vectors (prefer type1)
            mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
            mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
            
            # Fallback to type0 if type1 is zero
            if abs(mv_x_type1) < 0.1 and abs(mv_y_type1) < 0.1:
                mv_x = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
                mv_y = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
            else:
                mv_x, mv_y = mv_x_type1, mv_y_type1
            
            raw_motion_vectors.append((mv_x, mv_y))
        
        # 3. Compensate motion vectors for camera motion
        compensated_vectors, relative_magnitudes = compensate_motion_vectors(raw_motion_vectors, camera_motion)
        
        # 4. Adaptive macroblock selection based on compensated motion
        selection_strategy = 'motion_guided' if confidence > 0.3 else 'dynamic'
        selected_indices, properties = adaptive_macroblock_selection(
            object_info, compensated_vectors, relative_magnitudes, selection_strategy
        )
        
        print(f"      📍 Selected {len(selected_indices)}/{total_macroblocks} macroblocks using {selection_strategy} strategy")
        
        # 5. Store frame data
        frame_data = {
            'frame_idx': frame_idx,
            'camera_motion': camera_motion,
            'camera_confidence': confidence,
            'raw_motion_vectors': raw_motion_vectors,
            'compensated_vectors': compensated_vectors,
            'relative_magnitudes': relative_magnitudes,
            'selected_indices': selected_indices,
            'selection_properties': properties,
            'selection_strategy': selection_strategy,
            'macroblocks': macroblocks,
            'rgb_frame': rgb_data[frame_idx] if rgb_data is not None and frame_idx < rgb_data.shape[0] else None
        }
        
        tracking_system['frames_data'].append(frame_data)
    
    # Calculate summary statistics
    avg_camera_motion = np.mean([fd['camera_motion'] for fd in tracking_system['frames_data']], axis=0)
    avg_confidence = np.mean([fd['camera_confidence'] for fd in tracking_system['frames_data']])
    avg_selected = np.mean([len(fd['selected_indices']) for fd in tracking_system['frames_data']])
    
    tracking_system['summary_stats'] = {
        'avg_camera_motion': avg_camera_motion,
        'avg_camera_confidence': avg_confidence,
        'avg_selected_macroblocks': avg_selected,
        'total_frames': gop_frames
    }
    
    print(f"   ✅ Camera compensation complete:")
    print(f"      📈 Avg camera motion: ({avg_camera_motion[0]:.1f}, {avg_camera_motion[1]:.1f})")
    print(f"      🎯 Avg confidence: {avg_confidence:.2f}")
    print(f"      📍 Avg selected macroblocks: {avg_selected:.1f}/{total_macroblocks}")
    
    return tracking_system

def create_camera_compensated_video(all_tracking_results, output_filename="camera_compensated_tracking.mp4"):
    """Create video showing camera motion compensation."""
    print(f"\n🎬 Creating Camera Motion Compensated Video")
    print("=" * 60)
    
    if not all_tracking_results:
        print("❌ No tracking results to visualize")
        return False
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    
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
                object_info = tracking_result['object_info']
                
                # Display RGB background
                if frame_data['rgb_frame'] is not None:
                    try:
                        ax.imshow(frame_data['rgb_frame'], extent=[0, frame_width, frame_height, 0], alpha=0.7)
                    except:
                        ax.set_facecolor('black')
                else:
                    ax.set_facecolor('black')
                
                # Draw object bounding box
                x1, y1, x2, y2 = object_info['bbox']
                colors = ['yellow', 'cyan', 'orange', 'lime', 'magenta', 'red', 'blue']
                color = colors[object_info['id'] % len(colors)]
                
                bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                            linewidth=3, edgecolor=color, facecolor='none',
                                            linestyle='-', alpha=0.9)
                ax.add_patch(bbox_rect)
                
                # Draw selected macroblocks and their motion vectors
                selected_indices = frame_data['selected_indices']
                compensated_vectors = frame_data['compensated_vectors']
                relative_magnitudes = frame_data['relative_magnitudes']
                properties = frame_data['selection_properties']
                macroblocks = frame_data['macroblocks']
                
                for i, (idx, prop) in enumerate(zip(selected_indices, properties)):
                    if idx >= len(macroblocks):
                        continue
                        
                    mb_info = macroblocks[idx]
                    mb_x, mb_y = mb_info['pixel_center']
                    
                    # Color code by selection property
                    if 'high' in prop:
                        mb_color = 'red'
                        mb_size = 8
                    elif 'medium' in prop:
                        mb_color = 'orange'
                        mb_size = 6
                    else:
                        mb_color = 'yellow'
                        mb_size = 4
                    
                    # Draw macroblock position
                    ax.plot(mb_x, mb_y, 'o', color=mb_color, markersize=mb_size, 
                           markeredgecolor='white', markeredgewidth=1, alpha=0.8)
                    
                    # Draw compensated motion vector (clean red arrows)
                    if idx < len(compensated_vectors) and idx < len(relative_magnitudes):
                        comp_x, comp_y = compensated_vectors[idx]
                        rel_mag = relative_magnitudes[idx]
                        
                        if rel_mag > 0.1:  # Only draw significant motion
                            # Scale for visibility
                            scale = 3.0 if rel_mag < 1.0 else 2.0
                            arrow_dx = comp_x * scale
                            arrow_dy = comp_y * scale
                            
                            # Red arrow for compensated motion
                            ax.arrow(mb_x, mb_y, arrow_dx, arrow_dy,
                                    head_width=4, head_length=4, 
                                    fc='red', ec='red', alpha=0.9, linewidth=2)
                
                # Draw global camera motion indicator
                camera_motion = frame_data['camera_motion']
                confidence = frame_data['camera_confidence']
                
                if confidence > 0.2:  # Only show if confident about camera motion
                    # Draw camera motion vector at top-right
                    cam_start_x, cam_start_y = frame_width - 100, 50
                    cam_scale = 10.0  # Larger scale for visibility
                    cam_dx = camera_motion[0] * cam_scale
                    cam_dy = camera_motion[1] * cam_scale
                    
                    # Camera motion in blue
                    ax.arrow(cam_start_x, cam_start_y, cam_dx, cam_dy,
                            head_width=8, head_length=8, 
                            fc='blue', ec='blue', alpha=0.8, linewidth=3)
                    
                    ax.text(cam_start_x - 20, cam_start_y - 20, 'Camera', 
                           fontsize=8, color='blue', fontweight='bold')
                
                # Title with motion analysis
                strategy = frame_data['selection_strategy']
                avg_rel_motion = np.mean(relative_magnitudes) if relative_magnitudes else 0
                
                motion_type = ("🎥 Camera Motion" if np.sqrt(camera_motion[0]**2 + camera_motion[1]**2) > 5.0 
                              else "🎯 Object Motion" if avg_rel_motion > 2.0 else "⚪ Static")
                
                title = (f"Seq {seq_idx+1} Obj {obj_idx+1} (ID:{object_info['id']})\\n"
                        f"Frame {frame_num+1}/{tracking_result['gop_frames']} | {motion_type}\\n"
                        f"Strategy: {strategy} | Selected: {len(selected_indices)}/{len(macroblocks)}")
                
                ax.set_title(title, fontsize=8, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Main title
        fig.suptitle(f'Camera Motion Compensated Tracking\\n'
                    f'Frame {frame_num+1}/{max_frames} | Red=Object Motion | Blue=Camera Motion', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    # Create animation
    print(f"📹 Animating {max_frames} frames with camera motion compensation...")
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                  interval=1000//7, blit=False, repeat=True)
    
    # Save video
    try:
        writer = animation.FFMpegWriter(fps=7, metadata=dict(artist='Camera Compensated Tracker'),
                                      bitrate=8000)
        anim.save(output_filename, writer=writer, dpi=100)
        print(f"✅ Camera compensated video saved: {output_filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False
    finally:
        plt.close(fig)

def main():
    """Main function for camera motion compensated tracking."""
    print("🎥 Camera Motion Compensated Tracker")
    print("=" * 50)
    print("🔴 Red arrows = Object motion (camera compensated)")
    print("🔵 Blue arrows = Estimated camera motion")
    print("📍 Color-coded macroblocks = Selection strategy")
    print("=" * 50)
    
    # Initialize data loader factory
    factory = MOTSDataLoaderFactory(verbose=False)
    
    # Select sequences (use just 2 for demonstration)
    from enhanced_motion_vector_tracker import select_top_objects_from_sequences
    selected_sequences = select_top_objects_from_sequences(factory, num_sequences=2, objects_per_sequence=2)
    
    if not selected_sequences:
        print("❌ No sequences available")
        return False
    
    print(f"\\n🎯 Selected {len(selected_sequences)} sequences with 2 objects each for camera compensation demo")
    
    # Track with camera compensation
    all_tracking_results = []
    
    for seq_data in selected_sequences:
        seq_idx = seq_data['sequence_index']
        data_loader = seq_data['data_loader']
        objects = seq_data['objects'][:2]  # Only 2 objects per sequence
        
        print(f"\\n📁 Processing Sequence {seq_idx + 1}: {seq_data['sequence_name']}")
        
        sequence_tracking_results = []
        
        for obj_idx, object_info in enumerate(objects):
            tracking_result = track_with_camera_compensation(
                data_loader, object_info, obj_idx, seq_idx, gop_idx=0
            )
            sequence_tracking_results.append(tracking_result)
        
        all_tracking_results.append(sequence_tracking_results)
    
    # Create video
    success = create_camera_compensated_video(all_tracking_results)
    
    if success:
        print("\\n🎉 Camera Motion Compensated Tracking completed!")
        print("\\n📊 Camera Motion Analysis Summary:")
        
        for seq_idx, seq_results in enumerate(all_tracking_results):
            print(f"\\n   📁 Sequence {seq_idx + 1}:")
            for obj_idx, result in enumerate(seq_results):
                if result:
                    stats = result['summary_stats']
                    cam_motion_mag = np.sqrt(stats['avg_camera_motion'][0]**2 + stats['avg_camera_motion'][1]**2)
                    
                    motion_classification = ("🎥 Camera-dominant" if cam_motion_mag > 5.0 
                                           else "🎯 Object-dominant" if cam_motion_mag > 1.0 
                                           else "⚪ Static scene")
                    
                    print(f"      🎯 Object {obj_idx + 1}: {motion_classification}")
                    print(f"         Camera motion: ({stats['avg_camera_motion'][0]:.1f}, {stats['avg_camera_motion'][1]:.1f})px")
                    print(f"         Confidence: {stats['avg_camera_confidence']:.2f}")
                    print(f"         Adaptive selection: {stats['avg_selected_macroblocks']:.1f} macroblocks/frame")
        
        print("\\n🔬 Camera Motion Compensation Features:")
        print("   ✅ Global camera motion estimation using median filtering")
        print("   ✅ Motion vector compensation (subtract camera motion)")
        print("   ✅ Adaptive macroblock selection based on compensated motion")
        print("   ✅ RANSAC-based robust estimation (fallback)")
        print("   ✅ Motion-guided selection strategies")
        print("   ✅ Confidence-based algorithm switching")
        print("   ✅ Visual distinction between camera and object motion")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create camera motion compensated tracking video")
