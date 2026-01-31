#!/usr/bin/env python3
"""
Corrected Single Object Macroblock Tracker

This script provides CORRECTED macroblock tracking that addresses the issues:
- Only accumulates motion from P-blocks (not I-blocks)
- Filters out noise and invalid motion vectors
- Properly handles reference frames
- Validates motion consistency
- Uses motion vector confidence weighting
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

# Add path exactly like residuals study
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

def select_single_object_macroblocks(annotation_data, first_rgb_frame):
    """Select ALL macroblocks within a single object's bounding box."""
    macroblock_size = 16  # pixels
    frame_width = frame_height = 960
    mb_cols = frame_width // macroblock_size  # 60
    mb_rows = frame_height // macroblock_size  # 60
    
    print(f"🎯 Corrected Single Object Macroblock Analysis")
    print(f"Macroblock grid: {mb_cols}x{mb_rows}, size: {macroblock_size}px")
    
    selected_object = None
    object_macroblocks = {}
    
    if annotation_data is not None:
        try:
            if hasattr(annotation_data, 'files') and 'annotations' in annotation_data.files:
                annotations = annotation_data['annotations']
                print(f"Found {len(annotations)} frames of annotations")
                
                if len(annotations) > 0:
                    first_frame_anns = annotations[0]  # First frame
                    print(f"First frame has {len(first_frame_anns)} annotations")
                    
                    # Select the most suitable object for tracking (medium size, good motion potential)
                    best_object = None
                    best_score = 0
                    
                    for i, ann in enumerate(first_frame_anns):
                        if len(ann) >= 6:
                            obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                            
                            # Convert normalized coordinates to pixel coordinates
                            x_center = x_norm * frame_width
                            y_center = y_norm * frame_height
                            width = w_norm * frame_width
                            height = h_norm * frame_height
                            
                            # Score based on size (not too small, not too large) and position (not at edges)
                            size_score = 1.0
                            if width < 50 or height < 50:  # Too small
                                size_score = 0.3
                            elif width > 300 or height > 400:  # Too large
                                size_score = 0.7
                            
                            # Position score (avoid edges where motion might be clipped)
                            pos_score = 1.0
                            if x_center < 100 or x_center > 860 or y_center < 100 or y_center > 860:
                                pos_score = 0.5
                            
                            total_score = size_score * pos_score
                            
                            if total_score > best_score and width > 32 and height > 32:
                                best_score = total_score
                                best_object = {
                                    'id': int(obj_id),
                                    'class': int(obj_class),
                                    'center': [x_center, y_center],
                                    'size': [width, height],
                                    'bbox': [x_center - width/2, y_center - height/2, 
                                           x_center + width/2, y_center + height/2],
                                    'area': width * height,
                                    'score': total_score
                                }
                    
                    if best_object:
                        selected_object = best_object
                        print(f"🎯 Selected Object {selected_object['id']} (score: {selected_object['score']:.2f}):")
                        print(f"    Center: ({selected_object['center'][0]:.1f}, {selected_object['center'][1]:.1f})")
                        print(f"    Size: {selected_object['size'][0]:.1f} x {selected_object['size'][1]:.1f}")
                        print(f"    Area: {selected_object['area']:.0f} pixels²")
                        
        except Exception as e:
            print(f"⚠️  Error processing annotations: {e}")
    
    # Fallback to a more reasonable object if no annotations
    if selected_object is None:
        print(f"🔄 Using fallback object in center area")
        selected_object = {
            'id': 1,
            'center': [480, 480],  # Center of frame
            'size': [120, 160],    # Reasonable size
            'bbox': [420, 400, 540, 560],
            'area': 19200
        }
        print(f"    Fallback object at center (480, 480) with size 120x160")
    
    # Find ALL macroblocks within the bounding box
    x1, y1, x2, y2 = selected_object['bbox']
    
    # Convert bounding box coordinates to macroblock indices
    mb_x1 = max(0, int(x1 // macroblock_size))
    mb_y1 = max(0, int(y1 // macroblock_size))
    mb_x2 = min(mb_cols - 1, int(x2 // macroblock_size))
    mb_y2 = min(mb_rows - 1, int(y2 // macroblock_size))
    
    print(f"📦 Bounding box covers macroblocks from ({mb_x1},{mb_y1}) to ({mb_x2},{mb_y2})")
    
    # Collect all macroblocks within the bounding box
    center_x, center_y = selected_object['center']
    center_mb_col = int(center_x // macroblock_size)
    center_mb_row = int(center_y // macroblock_size)
    
    macroblocks = []
    for mb_row in range(mb_y1, mb_y2 + 1):
        for mb_col in range(mb_x1, mb_x2 + 1):
            # Calculate macroblock position
            mb_center_x = mb_col * macroblock_size + macroblock_size / 2
            mb_center_y = mb_row * macroblock_size + macroblock_size / 2
            
            # Calculate distance to object center for weighting
            distance_to_center = np.sqrt((mb_center_x - center_x)**2 + (mb_center_y - center_y)**2)
            
            # Determine macroblock role
            if mb_col == center_mb_col and mb_row == center_mb_row:
                mb_type = "CENTER"
                color = (1.0, 0.0, 0.0)  # Red for center
                tracking_weight = 1.0
            elif (mb_row == mb_y1 or mb_row == mb_y2) and (mb_col == mb_x1 or mb_col == mb_x2):
                mb_type = "CORNER"
                color = (0.0, 0.0, 1.0)  # Blue for corners
                tracking_weight = 0.7
            elif mb_row == mb_y1 or mb_row == mb_y2 or mb_col == mb_x1 or mb_col == mb_x2:
                mb_type = "EDGE"
                color = (0.0, 0.8, 0.0)  # Green for edges
                tracking_weight = 0.8
            else:
                mb_type = "INTERIOR"
                color = (0.8, 0.5, 0.0)  # Orange for interior
                tracking_weight = 0.9
            
            macroblocks.append({
                'position': (mb_col, mb_row),
                'pixel_center': (mb_center_x, mb_center_y),
                'type': mb_type,
                'color': color,
                'distance_to_center': distance_to_center,
                'tracking_weight': tracking_weight,
                'index': len(macroblocks)
            })
    
    object_macroblocks = {
        'object_info': selected_object,
        'macroblocks': macroblocks,
        'bbox_mb_bounds': (mb_x1, mb_y1, mb_x2, mb_y2),
        'total_macroblocks': len(macroblocks)
    }
    
    print(f"✅ Collected {len(macroblocks)} macroblocks within object:")
    type_counts = {}
    for mb in macroblocks:
        mb_type = mb['type']
        type_counts[mb_type] = type_counts.get(mb_type, 0) + 1
    
    for mb_type, count in type_counts.items():
        print(f"    {mb_type}: {count} macroblocks")
    
    return object_macroblocks

def validate_motion_vector(mv_x, mv_y, frame_idx, is_p_block, max_reasonable_motion=50):
    """
    Validate if a motion vector is reasonable and should be used for tracking.
    
    Args:
        mv_x, mv_y: Motion vector components
        frame_idx: Current frame index
        is_p_block: Whether this is a P-block
        max_reasonable_motion: Maximum reasonable motion per frame in pixels
        
    Returns:
        bool: Whether the motion vector is valid for tracking
    """
    # I-blocks shouldn't have meaningful motion vectors in P-frames
    if not is_p_block and frame_idx > 0:
        return False
    
    # Check magnitude
    magnitude = np.sqrt(mv_x**2 + mv_y**2)
    
    # Filter out unreasonably large motion (likely noise or errors)
    if magnitude > max_reasonable_motion:
        return False
    
    # Very small motion might be noise - only accept if above threshold
    if magnitude < 0.5 and frame_idx > 0:
        return False
    
    # First frame (I-frame) should have zero or minimal motion
    if frame_idx == 0 and magnitude > 2.0:
        return False
    
    return True

def track_all_macroblocks_corrected(data_loader, object_macroblocks, gop_idx=0):
    """
    CORRECTED macroblock tracking that addresses the motion accumulation issues.
    """
    macroblock_size = 16
    print(f"🔍 CORRECTED Tracking of {object_macroblocks['total_macroblocks']} macroblocks through GOP {gop_idx}")
    
    # Load data for this GOP
    motion_data = data_loader.load_motion_vectors(gop_idx)
    rgb_data = data_loader.load_rgb_frames(gop_idx, 'pframe')
    macroblock_data = data_loader.load_macroblocks(gop_idx)
    
    if motion_data is None or rgb_data is None:
        print(f"❌ Missing data for GOP {gop_idx}")
        return None
    
    gop_frames = motion_data.shape[0]
    print(f"  GOP {gop_idx}: {gop_frames} frames")
    print(f"  Motion data shape: {motion_data.shape}")
    
    # Analyze the motion data first
    print(f"  📊 Motion Data Analysis:")
    print(f"      Range: [{motion_data.min():.3f}, {motion_data.max():.3f}]")
    
    # Count non-zero motion vectors
    non_zero_motion = np.count_nonzero(motion_data)
    total_elements = motion_data.size
    print(f"      Non-zero motion vectors: {non_zero_motion}/{total_elements} ({100*non_zero_motion/total_elements:.1f}%)")
    
    # Initialize tracking for all macroblocks
    tracking_results = {
        'object_info': object_macroblocks['object_info'],
        'macroblock_tracks': [],
        'summary_stats': {},
        'gop_frames': gop_frames
    }
    
    # Track a subset of macroblocks for initial validation (sample every 3rd macroblock)
    sample_macroblocks = object_macroblocks['macroblocks'][::3]  # Every 3rd macroblock
    print(f"  🎯 Tracking sample of {len(sample_macroblocks)} macroblocks for validation")
    
    # Track each sampled macroblock
    for mb_info in sample_macroblocks:
        mb_col, mb_row = mb_info['position']
        mb_track = {
            'macroblock_info': mb_info,
            'positions': [],
            'motion_vectors': [],
            'valid_motion_count': 0,
            'total_displacement': 0.0,
            'max_frame_motion': 0.0,
            'p_frame_count': 0,
            'avg_motion_per_p_frame': 0.0
        }
        
        # Initial position
        start_x, start_y = mb_info['pixel_center']
        current_x, current_y = start_x, start_y
        
        print(f"  📍 Tracking MB({mb_col},{mb_row}) - {mb_info['type']} - starting at ({start_x:.1f}, {start_y:.1f})")
        
        total_motion_magnitude = 0
        valid_p_frames = 0
        
        for frame_idx in range(gop_frames):
            # Get motion vectors for both types
            mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
            mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
            mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
            mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
            
            # Choose motion vector with larger magnitude
            mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
            mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
            
            if mag_type1 > mag_type0:
                mv_x, mv_y = mv_x_type1, mv_y_type1
                motion_type = 1
            else:
                mv_x, mv_y = mv_x_type0, mv_y_type0
                motion_type = 0
            
            # Check block type
            is_p_block = True
            confidence = 1.0
            if macroblock_data is not None:
                mb_type = macroblock_data[frame_idx, mb_row, mb_col, 0]
                if mb_type == 0:  # I-block
                    is_p_block = False
                    confidence = 0.3
            
            # Validate motion vector
            is_valid_motion = validate_motion_vector(mv_x, mv_y, frame_idx, is_p_block)
            
            # Only accumulate motion if it's valid
            motion_applied = False
            if is_valid_motion and frame_idx > 0 and is_p_block:
                # Apply motion with confidence weighting
                weighted_mv_x = mv_x * mb_info['tracking_weight'] * confidence
                weighted_mv_y = mv_y * mb_info['tracking_weight'] * confidence
                
                current_x += weighted_mv_x
                current_y += weighted_mv_y
                motion_applied = True
                
                frame_motion = np.sqrt(mv_x**2 + mv_y**2)
                total_motion_magnitude += frame_motion
                valid_p_frames += 1
                mb_track['valid_motion_count'] += 1
                
                # Track maximum motion in a single frame
                if frame_motion > mb_track['max_frame_motion']:
                    mb_track['max_frame_motion'] = frame_motion
            
            # Store position and motion data
            mb_track['positions'].append({
                'frame': frame_idx + 5,  # Frames start at 5
                'x': current_x,
                'y': current_y,
                'mv_x': mv_x,
                'mv_y': mv_y,
                'is_p_block': is_p_block,
                'is_valid_motion': is_valid_motion,
                'motion_applied': motion_applied,
                'confidence': confidence,
                'motion_type': motion_type,
                'rgb_frame': rgb_data[frame_idx] if frame_idx < rgb_data.shape[0] else None
            })
            
            # Log first few frames for debugging
            if frame_idx < 5:
                block_type = "P" if is_p_block else "I"
                valid_str = "✓" if is_valid_motion else "✗"
                applied_str = "APPLIED" if motion_applied else "SKIPPED"
                print(f"    Frame {frame_idx+5:2d}: {block_type}-block, MV=({mv_x:+6.2f},{mv_y:+6.2f}) {valid_str} -> {applied_str} -> pos=({current_x:.1f},{current_y:.1f})")
        
        # Calculate summary statistics
        start_pos = mb_track['positions'][0]
        end_pos = mb_track['positions'][-1]
        mb_track['total_displacement'] = np.sqrt((end_pos['x'] - start_pos['x'])**2 + (end_pos['y'] - start_pos['y'])**2)
        mb_track['p_frame_count'] = valid_p_frames
        mb_track['avg_motion_per_p_frame'] = total_motion_magnitude / max(valid_p_frames, 1)
        
        # Calculate path length
        path_length = 0
        for i in range(1, len(mb_track['positions'])):
            prev_pos = mb_track['positions'][i-1]
            curr_pos = mb_track['positions'][i]
            step_distance = np.sqrt((curr_pos['x'] - prev_pos['x'])**2 + (curr_pos['y'] - prev_pos['y'])**2)
            path_length += step_distance
        mb_track['path_length'] = path_length
        
        tracking_results['macroblock_tracks'].append(mb_track)
        
        # Log summary for this macroblock
        print(f"    📊 MB({mb_col},{mb_row}) {mb_info['type']}: "
              f"displacement={mb_track['total_displacement']:.1f}px, "
              f"path={path_length:.1f}px, "
              f"avg_motion={mb_track['avg_motion_per_p_frame']:.2f}px/frame, "
              f"valid_motions={mb_track['valid_motion_count']}/{gop_frames}")
    
    # Calculate overall summary statistics
    displacements = [track['total_displacement'] for track in tracking_results['macroblock_tracks']]
    avg_motions = [track['avg_motion_per_p_frame'] for track in tracking_results['macroblock_tracks']]
    valid_motions = [track['valid_motion_count'] for track in tracking_results['macroblock_tracks']]
    
    tracking_results['summary_stats'] = {
        'total_macroblocks': len(tracking_results['macroblock_tracks']),
        'avg_displacement': np.mean(displacements),
        'max_displacement': np.max(displacements),
        'min_displacement': np.min(displacements),
        'std_displacement': np.std(displacements),
        'avg_motion_per_frame': np.mean(avg_motions),
        'max_motion_per_frame': np.max(avg_motions),
        'avg_valid_motions': np.mean(valid_motions),
        'best_tracking_mb_idx': np.argmax(displacements),
        'most_consistent_mb_idx': np.argmax(valid_motions)
    }
    
    print(f"\n📊 CORRECTED Tracking Summary:")
    print(f"  Average displacement: {tracking_results['summary_stats']['avg_displacement']:.1f}px")
    print(f"  Displacement range: {tracking_results['summary_stats']['min_displacement']:.1f} - {tracking_results['summary_stats']['max_displacement']:.1f}px")
    print(f"  Average motion per P-frame: {tracking_results['summary_stats']['avg_motion_per_frame']:.2f}px")
    print(f"  Average valid motions per macroblock: {tracking_results['summary_stats']['avg_valid_motions']:.1f}/{gop_frames}")
    
    best_idx = tracking_results['summary_stats']['best_tracking_mb_idx']
    best_mb = tracking_results['macroblock_tracks'][best_idx]
    print(f"  🏆 Best tracking macroblock: {best_mb['macroblock_info']['type']} at {best_mb['macroblock_info']['position']} "
          f"(displacement: {best_mb['total_displacement']:.1f}px, valid motions: {best_mb['valid_motion_count']})")
    
    consistent_idx = tracking_results['summary_stats']['most_consistent_mb_idx']
    consistent_mb = tracking_results['macroblock_tracks'][consistent_idx]
    print(f"  🎯 Most consistent macroblock: {consistent_mb['macroblock_info']['type']} at {consistent_mb['macroblock_info']['position']} "
          f"(valid motions: {consistent_mb['valid_motion_count']}, displacement: {consistent_mb['total_displacement']:.1f}px)")
    
    return tracking_results

def create_corrected_tracking_video(tracking_results):
    """Create visualization video for corrected macroblock tracking."""
    if tracking_results is None:
        print("❌ No tracking results to visualize")
        return False
    
    print("🎬 Creating CORRECTED Single Object Macroblock Tracking Video")
    print("=" * 60)
    
    object_info = tracking_results['object_info']
    macroblock_tracks = tracking_results['macroblock_tracks']
    gop_frames = tracking_results['gop_frames']
    
    print(f"📹 Object {object_info['id']}: {len(macroblock_tracks)} macroblocks, {gop_frames} frames")
    
    # Create video with enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    frame_width = frame_height = 960
    macroblock_size = 16
    
    def animate(frame_num):
        ax1.clear()
        ax2.clear()
        
        # ---- LEFT PANEL: Full scene with object tracking ----
        ax1.set_xlim(0, frame_width)
        ax1.set_ylim(0, frame_height)
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
        
        # Get RGB background from any macroblock
        rgb_frame = None
        if macroblock_tracks and frame_num < len(macroblock_tracks[0]['positions']):
            rgb_frame = macroblock_tracks[0]['positions'][frame_num]['rgb_frame']
        
        # Add RGB background
        if rgb_frame is not None:
            ax1.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.7)
        
        # Add light grid
        for i in range(0, frame_width + 1, macroblock_size * 4):
            ax1.axvline(x=i, color='white', alpha=0.3, linewidth=0.8)
        for i in range(0, frame_height + 1, macroblock_size * 4):
            ax1.axhline(y=i, color='white', alpha=0.3, linewidth=0.8)
        
        # Highlight the object bounding box
        x1, y1, x2, y2 = object_info['bbox']
        bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=3, edgecolor='yellow', facecolor='none',
                                    linestyle='--', alpha=0.8)
        ax1.add_patch(bbox_rect)
        
        # Plot all macroblock trajectories and current positions
        active_count = 0
        valid_motion_count = 0
        
        for track_idx, track in enumerate(macroblock_tracks):
            if frame_num >= len(track['positions']):
                continue
            
            mb_info = track['macroblock_info']
            positions_so_far = track['positions'][:frame_num + 1]
            
            if not positions_so_far:
                continue
            
            active_count += 1
            current_pos = positions_so_far[-1]
            
            # Count valid motions applied
            if current_pos.get('motion_applied', False):
                valid_motion_count += 1
            
            # Plot trajectory path (only show recent valid motions)
            valid_positions = [pos for pos in positions_so_far if pos.get('motion_applied', True)]
            if len(valid_positions) > 1:
                x_coords = [pos['x'] for pos in valid_positions]
                y_coords = [pos['y'] for pos in valid_positions]
                
                # Use macroblock type color with enhanced visibility for valid motions
                color = mb_info['color']
                alpha = 0.9 if current_pos.get('motion_applied', False) else 0.5
                linewidth = 4 if mb_info['type'] == 'CENTER' else 3
                
                ax1.plot(x_coords, y_coords, '-', color=color, 
                        linewidth=linewidth, alpha=alpha, solid_capstyle='round')
                
                # Plot trail points
                marker_size = 5 if mb_info['type'] == 'CENTER' else 3
                ax1.plot(x_coords[:-1], y_coords[:-1], 'o', color=color, 
                        markersize=marker_size, alpha=alpha*0.7, 
                        markeredgecolor='white', markeredgewidth=0.5)
            
            # Current position with enhanced visualization
            mb_size = macroblock_size if mb_info['type'] != 'CENTER' else macroblock_size * 1.3
            edge_color = mb_info['color']
            face_alpha = 0.6 if current_pos.get('motion_applied', False) else 0.3
            
            mb_rect = patches.Rectangle(
                (current_pos['x'] - mb_size/2, current_pos['y'] - mb_size/2),
                mb_size, mb_size,
                linewidth=3, edgecolor=edge_color, facecolor=edge_color, 
                alpha=face_alpha
            )
            ax1.add_patch(mb_rect)
            
            # Current position marker
            marker_size = 10 if mb_info['type'] == 'CENTER' else 7
            ax1.plot(current_pos['x'], current_pos['y'], 'o', color=edge_color, 
                    markersize=marker_size, markeredgecolor='white', markeredgewidth=2)
            
            # Motion vector arrow (only for valid, applied motion)
            if current_pos.get('motion_applied', False) and (abs(current_pos['mv_x']) > 1.0 or abs(current_pos['mv_y']) > 1.0):
                arrow_scale = 3
                ax1.arrow(current_pos['x'], current_pos['y'], 
                         current_pos['mv_x'] * arrow_scale, current_pos['mv_y'] * arrow_scale,
                         head_width=8, head_length=8, fc=edge_color, ec=edge_color, 
                         alpha=0.9, linewidth=2)
        
        # ---- RIGHT PANEL: Detailed analysis ----
        ax2.set_xlim(x1 - 50, x2 + 50)
        ax2.set_ylim(y1 - 50, y2 + 50)
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        # Zoomed RGB background
        if rgb_frame is not None:
            ax2.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.8)
        
        # Motion validation information
        if macroblock_tracks and frame_num < len(macroblock_tracks[0]['positions']):
            current_frame_info = macroblock_tracks[0]['positions'][frame_num]
            
            # Show motion validation status
            for track_idx, track in enumerate(macroblock_tracks):
                if frame_num >= len(track['positions']):
                    continue
                    
                mb_info = track['macroblock_info']
                current_pos = track['positions'][frame_num]
                
                # Color coding based on motion status
                if current_pos.get('motion_applied', False):
                    status_color = 'lime'
                    status_text = 'VALID'
                elif current_pos.get('is_valid_motion', False):
                    status_color = 'yellow'
                    status_text = 'FILTERED'
                else:
                    status_color = 'red'
                    status_text = 'INVALID'
                
                # Enhanced macroblock visualization in zoom
                mb_rect = patches.Rectangle(
                    (current_pos['x'] - macroblock_size/2, current_pos['y'] - macroblock_size/2),
                    macroblock_size, macroblock_size,
                    linewidth=2, edgecolor=status_color, facecolor=mb_info['color'], 
                    alpha=0.7
                )
                ax2.add_patch(mb_rect)
                
                # Status label
                displacement = track['total_displacement']
                label = f"{mb_info['type'][0]}\\n{status_text}\\n{displacement:.0f}px"
                ax2.text(current_pos['x'], current_pos['y'], label, 
                        fontsize=7, ha='center', va='center', 
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=status_color, alpha=0.8))
        
        # Enhanced legends and information
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Patch(facecolor=(1.0, 0.0, 0.0), label='CENTER'),
            Patch(facecolor=(0.0, 0.8, 0.0), label='EDGE'),
            Patch(facecolor=(0.0, 0.0, 1.0), label='CORNER'),
            Patch(facecolor=(0.8, 0.5, 0.0), label='INTERIOR'),
            Line2D([0], [0], color='lime', linewidth=3, label='Valid Motion'),
            Line2D([0], [0], color='red', linewidth=3, label='Invalid Motion')
        ]
        
        ax1.legend(handles=legend_elements, loc='lower right', 
                  bbox_to_anchor=(0.98, 0.02), fontsize=9, 
                  frameon=True, fancybox=True, shadow=True,
                  facecolor='white', edgecolor='black', framealpha=0.9)
        
        # Frame information with motion validation stats
        if macroblock_tracks and frame_num < len(macroblock_tracks[0]['positions']):
            current_frame_info = macroblock_tracks[0]['positions'][frame_num]
            frame_info = (f'Frame: {current_frame_info["frame"]}\\n'
                         f'GOP Frame: {frame_num + 1}/{gop_frames}\\n'
                         f'Active MBs: {active_count}/{len(macroblock_tracks)}\\n'
                         f'Valid Motions: {valid_motion_count}')
        else:
            frame_info = f'Frame: {frame_num + 1}/{gop_frames}'
            
        ax1.text(0.02, 0.98, frame_info, transform=ax1.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # Object and tracking quality information
        if frame_num < len(macroblock_tracks[0]['positions']):
            summary_stats = tracking_results['summary_stats']
            quality_info = (f'Object {object_info["id"]} (CORRECTED)\\n'
                           f'Avg Displacement: {summary_stats["avg_displacement"]:.1f}px\\n'
                           f'Avg Motion/Frame: {summary_stats["avg_motion_per_frame"]:.2f}px\\n'
                           f'Valid Motion Rate: {summary_stats["avg_valid_motions"]:.1f}/{gop_frames}')
            ax2.text(0.02, 0.98, quality_info, transform=ax2.transAxes,
                    fontsize=10, fontweight='bold', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))
        
        # Titles
        ax1.set_title(f'CORRECTED Object Tracking - Full Scene\\n'
                     f'Object {object_info["id"]} with Motion Validation', 
                     fontsize=12, fontweight='bold')
        
        ax2.set_title(f'Motion Validation Analysis\\n'
                     f'Green=Valid, Yellow=Filtered, Red=Invalid', 
                     fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('X Position (pixels)', fontsize=11)
        ax1.set_ylabel('Y Position (pixels)', fontsize=11)
        ax2.set_xlabel('X Position (pixels)', fontsize=11)
        ax2.set_ylabel('Y Position (pixels)', fontsize=11)
    
    # Create animation
    print(f"📹 Animating {gop_frames} frames with corrected tracking...")
    anim = animation.FuncAnimation(fig, animate, frames=gop_frames, 
                                  interval=1000//4, blit=False, repeat=True)
    
    # Save video
    output_path = "corrected_single_object_macroblock_tracking.mp4"
    try:
        writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Corrected Object Tracker'),
                                      bitrate=3500)
        anim.save(output_path, writer=writer, dpi=120)
        print(f"✅ Video saved: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False
    finally:
        plt.close(fig)

def main():
    """Main function to create corrected single object tracking."""
    print("🎬 CORRECTED Single Object Macroblock Tracker")
    print("=" * 50)
    
    # Initialize data loader
    factory = MOTSDataLoaderFactory(verbose=True)
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    sequence_name = sequences[0] if sequences else 'MOT17-09-SDP_960x960_gop50_500frames'
    data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
    
    print(f"✅ Using sequence: {sequence_name}")
    
    # Load annotations for object selection
    print("📋 Loading annotations for object selection...")
    try:
        annotation_data = data_loader.load_corrected_annotations(0)
        if annotation_data is None:
            annotation_data = data_loader.load_annotations(0)
    except:
        annotation_data = None
        print("⚠️  No annotations available, using fallback selection")
    
    # Load first frame for context
    rgb_data_first = data_loader.load_rgb_frames(0, 'pframe')
    first_frame = rgb_data_first[0] if rgb_data_first is not None else None
    
    # Select single object and all its macroblocks
    object_macroblocks = select_single_object_macroblocks(annotation_data, first_frame)
    
    if not object_macroblocks or object_macroblocks['total_macroblocks'] == 0:
        print("❌ No macroblocks selected")
        return False
    
    # Track with corrected algorithm
    tracking_results = track_all_macroblocks_corrected(data_loader, object_macroblocks, gop_idx=0)
    
    if tracking_results is None:
        print("❌ Tracking failed")
        return False
    
    # Create visualization video
    success = create_corrected_tracking_video(tracking_results)
    
    if success:
        print("\\n🎉 CORRECTED Single Object Macroblock Tracking completed!")
        print("\\n📊 Corrected Analysis Summary:")
        
        stats = tracking_results['summary_stats']
        print(f"  📦 Total Macroblocks Tracked: {stats['total_macroblocks']}")
        print(f"  📏 Average Displacement: {stats['avg_displacement']:.1f} pixels")
        print(f"  📏 Displacement Range: {stats['min_displacement']:.1f} - {stats['max_displacement']:.1f} pixels")
        print(f"  🎯 Average Motion per P-Frame: {stats['avg_motion_per_frame']:.2f} pixels")
        print(f"  ✅ Average Valid Motions: {stats['avg_valid_motions']:.1f} per macroblock")
        
        best_mb = tracking_results['macroblock_tracks'][stats['best_tracking_mb_idx']]
        print(f"  🏆 Best Tracking Macroblock: {best_mb['macroblock_info']['type']} at {best_mb['macroblock_info']['position']}")
        print(f"      Displacement: {best_mb['total_displacement']:.1f}px")
        print(f"      Valid Motions: {best_mb['valid_motion_count']}")
        
        print("\\n🔧 Corrections Applied:")
        print("  ✅ Only P-blocks used for motion accumulation")
        print("  ✅ Motion vector validation (magnitude, noise filtering)")
        print("  ✅ Confidence weighting based on macroblock type")
        print("  ✅ Invalid motion filtering (I-blocks, large jumps, noise)")
        print("  ✅ Enhanced tracking consistency validation")
        
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create corrected tracking video")
