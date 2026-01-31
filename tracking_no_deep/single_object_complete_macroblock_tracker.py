#!/usr/bin/env python3
"""
Single Object Complete Macroblock Tracker

This script focuses on tracking ALL macroblocks within a single object's bounding box
to analyze trajectory patterns and motion vector behavior across the entire object area.

Features:
- Selects ONE object from annotations
- Tracks ALL macroblocks within the object's bounding box
- Color-codes macroblocks based on their position (center, edge, corner)
- Analyzes motion patterns across the entire object
- Shows individual trajectories for each macroblock
- Identifies best-performing macroblocks for tracking
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
    """
    Select ALL macroblocks within a single object's bounding box.
    
    Args:
        annotation_data: Annotation data from data loader  
        first_rgb_frame: First RGB frame (960x960x3)
    
    Returns:
        Dictionary with macroblock info: positions, types, and color coding
    """
    macroblock_size = 16  # pixels
    frame_width = frame_height = 960
    mb_cols = frame_width // macroblock_size  # 60
    mb_rows = frame_height // macroblock_size  # 60
    
    print(f"🎯 Single Object Complete Macroblock Analysis")
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
                    
                    # Select the largest object (usually most reliable for tracking)
                    largest_object = None
                    largest_area = 0
                    
                    for i, ann in enumerate(first_frame_anns):
                        if len(ann) >= 6:
                            obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                            
                            # Convert normalized coordinates to pixel coordinates
                            x_center = x_norm * frame_width
                            y_center = y_norm * frame_height
                            width = w_norm * frame_width
                            height = h_norm * frame_height
                            
                            area = width * height
                            
                            # Check if this is the largest valid object
                            if area > largest_area and width > 32 and height > 32:  # Minimum size requirement
                                largest_area = area
                                largest_object = {
                                    'id': int(obj_id),
                                    'class': int(obj_class),
                                    'center': [x_center, y_center],
                                    'size': [width, height],
                                    'bbox': [x_center - width/2, y_center - height/2, 
                                           x_center + width/2, y_center + height/2],
                                    'area': area
                                }
                    
                    if largest_object:
                        selected_object = largest_object
                        print(f"🎯 Selected Object {selected_object['id']}:")
                        print(f"    Center: ({selected_object['center'][0]:.1f}, {selected_object['center'][1]:.1f})")
                        print(f"    Size: {selected_object['size'][0]:.1f} x {selected_object['size'][1]:.1f}")
                        print(f"    Area: {selected_object['area']:.0f} pixels²")
                        
        except Exception as e:
            print(f"⚠️  Error processing annotations: {e}")
    
    # Fallback to a default object if no annotations
    if selected_object is None:
        print(f"🔄 Using fallback object")
        selected_object = {
            'id': 1,
            'center': [500, 400],
            'size': [100, 150],
            'bbox': [450, 325, 550, 475],
            'area': 15000
        }
        print(f"    Fallback object at center (500, 400) with size 100x150")
    
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
            
            # Classify macroblock type based on position relative to object center
            distance_to_center = np.sqrt((mb_center_x - center_x)**2 + (mb_center_y - center_y)**2)
            
            # Determine macroblock role
            if mb_col == center_mb_col and mb_row == center_mb_row:
                mb_type = "CENTER"
                color = (1.0, 0.0, 0.0)  # Red for center
            elif (mb_row == mb_y1 or mb_row == mb_y2) and (mb_col == mb_x1 or mb_col == mb_x2):
                mb_type = "CORNER"
                color = (0.0, 0.0, 1.0)  # Blue for corners
            elif mb_row == mb_y1 or mb_row == mb_y2 or mb_col == mb_x1 or mb_col == mb_x2:
                mb_type = "EDGE"
                color = (0.0, 0.8, 0.0)  # Green for edges
            else:
                mb_type = "INTERIOR"
                color = (0.8, 0.5, 0.0)  # Orange for interior
            
            macroblocks.append({
                'position': (mb_col, mb_row),
                'pixel_center': (mb_center_x, mb_center_y),
                'type': mb_type,
                'color': color,
                'distance_to_center': distance_to_center,
                'index': len(macroblocks)  # For tracking purposes
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

def track_all_macroblocks_in_object(data_loader, object_macroblocks, gop_idx=0):
    """
    Track all macroblocks within the selected object through a GOP.
    
    Args:
        data_loader: MOTS data loader
        object_macroblocks: Dictionary with object and macroblock info
        gop_idx: GOP index to track
        
    Returns:
        Dictionary with tracking data for all macroblocks
    """
    macroblock_size = 16
    print(f"🔍 Tracking {object_macroblocks['total_macroblocks']} macroblocks through GOP {gop_idx}")
    
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
    
    # Initialize tracking for all macroblocks
    tracking_results = {
        'object_info': object_macroblocks['object_info'],
        'macroblock_tracks': [],
        'summary_stats': {},
        'gop_frames': gop_frames,
        'frame_macroblock_counts': [],  # Track how many original MBs remain in bbox per frame
        'low_count_frames': [],  # Frames where original MB count < 20%
        'new_macroblocks_per_frame': []  # New MBs found in bbox when count is low
    }
    
    original_bbox = object_macroblocks['object_info']['bbox']
    original_mb_count = object_macroblocks['total_macroblocks']
    threshold_count = int(original_mb_count * 0.2)  # 20% threshold
    
    print(f"📊 Original macroblock count: {original_mb_count}, 20% threshold: {threshold_count}")
    
    # Track each macroblock
    for mb_info in object_macroblocks['macroblocks']:
        mb_col, mb_row = mb_info['position']
        mb_track = {
            'macroblock_info': mb_info,
            'positions': [],
            'motion_vectors': [],
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
        p_frames = 0
        
        for frame_idx in range(gop_frames):
            # Get motion vectors for both types (include all blocks for now)
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
            
            # Check block type for information but don't filter
            is_p_block = True
            confidence = 1.0
            if macroblock_data is not None:
                mb_type = macroblock_data[frame_idx, mb_row, mb_col, 0]
                if mb_type == 0:  # I-block
                    is_p_block = False
                    confidence = 0.7  # Lower confidence for I-blocks but still include them
            
            # Update position (accumulate motion from frame 1 onwards)
            if frame_idx > 0:
                current_x += mv_x
                current_y += mv_y
                
                # Count motion for all blocks now
                frame_motion = np.sqrt(mv_x**2 + mv_y**2)
                total_motion_magnitude += frame_motion
                if is_p_block:
                    p_frames += 1
                
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
                'confidence': confidence,
                'motion_type': motion_type,
                'rgb_frame': rgb_data[frame_idx] if frame_idx < rgb_data.shape[0] else None
            })
            
            mb_track['motion_vectors'].append({
                'frame': frame_idx + 5,
                'mv_x': mv_x,
                'mv_y': mv_y,
                'magnitude': np.sqrt(mv_x**2 + mv_y**2),
                'is_p_block': is_p_block,
                'motion_type': motion_type
            })
        
        # Calculate summary statistics
        start_pos = mb_track['positions'][0]
        end_pos = mb_track['positions'][-1]
        mb_track['total_displacement'] = np.sqrt((end_pos['x'] - start_pos['x'])**2 + (end_pos['y'] - start_pos['y'])**2)
        mb_track['p_frame_count'] = p_frames
        mb_track['avg_motion_per_p_frame'] = total_motion_magnitude / max(p_frames, 1)
        
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
              f"avg_motion={mb_track['avg_motion_per_p_frame']:.2f}px/frame")
    
    # NOW analyze macroblock counts per frame after tracking is complete
    for frame_idx in range(gop_frames):
        macroblocks_in_bbox = 0
        new_macroblocks = []
        
        # Count how many original macroblocks are still within the bbox
        for track_idx, track in enumerate(tracking_results['macroblock_tracks']):
            if frame_idx < len(track['positions']):
                # Get the actual tracked position
                tracked_pos = track['positions'][frame_idx]
                tracked_x = tracked_pos['x']
                tracked_y = tracked_pos['y']
                
                # Check if tracked position is still within original bounding box
                x1, y1, x2, y2 = original_bbox
                if x1 <= tracked_x <= x2 and y1 <= tracked_y <= y2:
                    macroblocks_in_bbox += 1
        
        tracking_results['frame_macroblock_counts'].append(macroblocks_in_bbox)
        
        # If below threshold, find new macroblocks in the current bounding box area
        if macroblocks_in_bbox < threshold_count:
            tracking_results['low_count_frames'].append(frame_idx)
            
            # Find new macroblocks that could be added to maintain tracking
            x1, y1, x2, y2 = original_bbox
            mb_x1 = max(0, int(x1 // 16))
            mb_y1 = max(0, int(y1 // 16))
            mb_x2 = min(59, int(x2 // 16))  # 60x60 grid
            mb_y2 = min(59, int(y2 // 16))
            
            for mb_row in range(mb_y1, mb_y2 + 1):
                for mb_col in range(mb_x1, mb_x2 + 1):
                    # Check if this is not an original macroblock
                    is_original = any(mb['position'] == (mb_col, mb_row) 
                                    for mb in object_macroblocks['macroblocks'])
                    
                    if not is_original:
                        mb_center_x = mb_col * 16 + 8
                        mb_center_y = mb_row * 16 + 8
                        
                        # Get motion vector for this new macroblock
                        if frame_idx < motion_data.shape[0]:
                            mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
                            mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
                            mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
                            mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
                            
                            # Choose motion vector with larger magnitude
                            mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
                            mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
                            
                            if mag_type1 > mag_type0:
                                mv_x, mv_y = mv_x_type1, mv_y_type1
                            else:
                                mv_x, mv_y = mv_x_type0, mv_y_type0
                            
                            # Check if it's a P-block or I-block
                            is_p_block = True
                            block_type_info = "P-block"
                            if macroblock_data is not None:
                                mb_type = macroblock_data[frame_idx, mb_row, mb_col, 0]
                                if mb_type == 0:  # I-block
                                    is_p_block = False
                                    block_type_info = "I-block"
                            
                            magnitude = np.sqrt(mv_x**2 + mv_y**2)
                            
                            # Only add if it has some motion or is a good candidate
                            if magnitude > 0.1:  # Small threshold to avoid completely static blocks
                                new_macroblocks.append({
                                    'position': (mb_col, mb_row),
                                    'pixel_center': (mb_center_x, mb_center_y),
                                    'motion_vector': (mv_x, mv_y),
                                    'magnitude': magnitude,
                                    'is_p_block': is_p_block,
                                    'block_type_info': block_type_info,
                                    'type': 'NEW_CANDIDATE'
                                })
            
            print(f"⚠️  Frame {frame_idx}: Only {macroblocks_in_bbox}/{original_mb_count} original MBs in bbox "
                  f"({macroblocks_in_bbox/original_mb_count*100:.1f}%), found {len(new_macroblocks)} new candidates")
        
        tracking_results['new_macroblocks_per_frame'].append(new_macroblocks)    # Calculate overall summary statistics
    displacements = [track['total_displacement'] for track in tracking_results['macroblock_tracks']]
    avg_motions = [track['avg_motion_per_p_frame'] for track in tracking_results['macroblock_tracks']]
    
    tracking_results['summary_stats'] = {
        'total_macroblocks': len(tracking_results['macroblock_tracks']),
        'avg_displacement': np.mean(displacements),
        'max_displacement': np.max(displacements),
        'min_displacement': np.min(displacements),
        'std_displacement': np.std(displacements),
        'avg_motion_per_frame': np.mean(avg_motions),
        'max_motion_per_frame': np.max(avg_motions),
        'best_tracking_mb_idx': np.argmax(displacements),  # Macroblock with highest displacement
        'most_stable_mb_idx': np.argmin([track['max_frame_motion'] for track in tracking_results['macroblock_tracks']])
    }
    
    print(f"\n📊 Overall Summary:")
    print(f"  Average displacement: {tracking_results['summary_stats']['avg_displacement']:.1f}px")
    print(f"  Displacement range: {tracking_results['summary_stats']['min_displacement']:.1f} - {tracking_results['summary_stats']['max_displacement']:.1f}px")
    print(f"  Average motion per frame: {tracking_results['summary_stats']['avg_motion_per_frame']:.2f}px")
    
    best_idx = tracking_results['summary_stats']['best_tracking_mb_idx']
    best_mb = tracking_results['macroblock_tracks'][best_idx]
    print(f"  🏆 Best tracking macroblock: {best_mb['macroblock_info']['type']} at {best_mb['macroblock_info']['position']} "
          f"(displacement: {best_mb['total_displacement']:.1f}px)")
    
    return tracking_results

def create_single_object_tracking_video(tracking_results):
    """Create visualization video for single object complete macroblock tracking."""
    if tracking_results is None:
        print("❌ No tracking results to visualize")
        return False
    
    print("🎬 Creating Single Object Complete Macroblock Tracking Video")
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
        original_mb_count = tracking_results['frame_macroblock_counts'][frame_num] if frame_num < len(tracking_results['frame_macroblock_counts']) else len(macroblock_tracks)
        is_low_count_frame = frame_num in tracking_results['low_count_frames']
        new_macroblocks = tracking_results['new_macroblocks_per_frame'][frame_num] if frame_num < len(tracking_results['new_macroblocks_per_frame']) else []
        
        for track_idx, track in enumerate(macroblock_tracks):
            if frame_num >= len(track['positions']):
                continue
            
            mb_info = track['macroblock_info']
            positions_so_far = track['positions'][:frame_num + 1]
            
            if not positions_so_far:
                continue
            
            active_count += 1
            
            # Plot trajectory path
            if len(positions_so_far) > 1:
                x_coords = [pos['x'] for pos in positions_so_far]
                y_coords = [pos['y'] for pos in positions_so_far]
                
                # Use macroblock type color with some transparency
                color = mb_info['color']
                alpha = 0.8 if mb_info['type'] == 'CENTER' else 0.6
                linewidth = 3 if mb_info['type'] == 'CENTER' else 2
                
                ax1.plot(x_coords, y_coords, '-', color=color, 
                        linewidth=linewidth, alpha=alpha, solid_capstyle='round')
                
                # Plot trail points (smaller for non-center macroblocks)
                marker_size = 4 if mb_info['type'] == 'CENTER' else 2
                ax1.plot(x_coords[:-1], y_coords[:-1], 'o', color=color, 
                        markersize=marker_size, alpha=alpha*0.7, 
                        markeredgecolor='white', markeredgewidth=0.5)
            
            # Current position
            current_pos = positions_so_far[-1]
            
            # Draw macroblock square (more prominent for center)
            mb_size = macroblock_size if mb_info['type'] != 'CENTER' else macroblock_size * 1.2
            mb_rect = patches.Rectangle(
                (current_pos['x'] - mb_size/2, current_pos['y'] - mb_size/2),
                mb_size, mb_size,
                linewidth=2, edgecolor=mb_info['color'], facecolor=mb_info['color'], 
                alpha=0.4 if mb_info['type'] != 'CENTER' else 0.6
            )
            ax1.add_patch(mb_rect)
            
            # Current position marker
            marker_size = 8 if mb_info['type'] == 'CENTER' else 5
            ax1.plot(current_pos['x'], current_pos['y'], 'o', color=mb_info['color'], 
                    markersize=marker_size, markeredgecolor='white', markeredgewidth=2)
            
            # Motion vector arrow (show for all blocks with significant motion)
            if abs(current_pos['mv_x']) > 0.5 or abs(current_pos['mv_y']) > 0.5:
                arrow_scale = 2
                # Color arrow based on block type
                arrow_color = mb_info['color'] if current_pos['is_p_block'] else 'red'
                arrow_alpha = 0.8 if current_pos['is_p_block'] else 0.6
                ax1.arrow(current_pos['x'], current_pos['y'], 
                         current_pos['mv_x'] * arrow_scale, current_pos['mv_y'] * arrow_scale,
                         head_width=5, head_length=5, fc=arrow_color, ec=arrow_color, 
                         alpha=arrow_alpha)
        
        # Add new macroblocks visualization when count is low
        if is_low_count_frame and new_macroblocks:
            for new_mb in new_macroblocks:
                mb_center_x, mb_center_y = new_mb['pixel_center']
                mv_x, mv_y = new_mb['motion_vector']
                magnitude = new_mb['magnitude']
                block_type = new_mb['block_type_info']
                
                # Draw new macroblock with distinct visualization
                new_mb_rect = patches.Rectangle(
                    (mb_center_x - macroblock_size/2, mb_center_y - macroblock_size/2),
                    macroblock_size, macroblock_size,
                    linewidth=3, edgecolor='magenta', facecolor='magenta', 
                    alpha=0.3, linestyle='--'
                )
                ax1.add_patch(new_mb_rect)
                
                # New macroblock marker
                ax1.plot(mb_center_x, mb_center_y, 's', color='magenta', 
                        markersize=6, markeredgecolor='white', markeredgewidth=2)
                
                # Motion vector arrow for new macroblock
                if magnitude > 0.5:
                    arrow_scale = 2
                    ax1.arrow(mb_center_x, mb_center_y, 
                             mv_x * arrow_scale, mv_y * arrow_scale,
                             head_width=4, head_length=4, fc='magenta', ec='magenta', 
                             alpha=0.9, linestyle='--')
                
                # Label with displacement magnitude and block type
                ax1.text(mb_center_x + 8, mb_center_y - 8, f'{block_type}\n{magnitude:.1f}px', 
                        fontsize=7, color='magenta', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # ---- RIGHT PANEL: Zoomed view of object area ----
        ax2.set_xlim(x1 - 50, x2 + 50)
        ax2.set_ylim(y1 - 50, y2 + 50)
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        # Zoomed RGB background
        if rgb_frame is not None:
            ax2.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.8)
        
        # Dense macroblock grid for zoomed view
        zoom_x1, zoom_x2 = ax2.get_xlim()
        zoom_y1, zoom_y2 = ax2.get_ylim()
        
        for i in range(int(zoom_x1//macroblock_size)*macroblock_size, 
                      int(zoom_x2//macroblock_size)*macroblock_size + macroblock_size, 
                      macroblock_size):
            ax2.axvline(x=i, color='lightgray', alpha=0.5, linewidth=1)
        for i in range(int(zoom_y1//macroblock_size)*macroblock_size, 
                      int(zoom_y2//macroblock_size)*macroblock_size + macroblock_size, 
                      macroblock_size):
            ax2.axhline(y=i, color='lightgray', alpha=0.5, linewidth=1)
        
        # Plot macroblocks in zoomed view with labels
        for track_idx, track in enumerate(macroblock_tracks):
            if frame_num >= len(track['positions']):
                continue
                
            mb_info = track['macroblock_info']
            positions_so_far = track['positions'][:frame_num + 1]
            
            if not positions_so_far:
                continue
            
            current_pos = positions_so_far[-1]
            
            # Enhanced macroblock visualization in zoom
            mb_rect = patches.Rectangle(
                (current_pos['x'] - macroblock_size/2, current_pos['y'] - macroblock_size/2),
                macroblock_size, macroblock_size,
                linewidth=2, edgecolor=mb_info['color'], facecolor=mb_info['color'], 
                alpha=0.5
            )
            ax2.add_patch(mb_rect)
            
            # Label with macroblock type and displacement
            displacement = track['total_displacement']
            label = f"{mb_info['type'][0]}\n{displacement:.0f}px"
            ax2.text(current_pos['x'], current_pos['y'], label, 
                    fontsize=8, ha='center', va='center', 
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=mb_info['color'], alpha=0.8))
            
            # Trajectory in zoom (only recent path to avoid clutter)
            if len(positions_so_far) > 1:
                recent_positions = positions_so_far[-min(10, len(positions_so_far)):]  # Last 10 positions
                x_coords = [pos['x'] for pos in recent_positions]
                y_coords = [pos['y'] for pos in recent_positions]
                ax2.plot(x_coords, y_coords, '-', color=mb_info['color'], 
                        linewidth=3, alpha=0.8)
        
        # Add new macroblocks in zoomed view when count is low
        if is_low_count_frame and new_macroblocks:
            # Show only the most promising new macroblocks (highest motion magnitude)
            sorted_new_mbs = sorted(new_macroblocks, key=lambda x: x['magnitude'], reverse=True)[:10]  # Top 10
            
            for new_mb in sorted_new_mbs:
                mb_center_x, mb_center_y = new_mb['pixel_center']
                mv_x, mv_y = new_mb['motion_vector']
                magnitude = new_mb['magnitude']
                block_type = new_mb['block_type_info']
                
                # Check if in zoom area
                zoom_x1, zoom_x2 = ax2.get_xlim()
                zoom_y1, zoom_y2 = ax2.get_ylim()
                if not (zoom_x1 <= mb_center_x <= zoom_x2 and zoom_y1 <= mb_center_y <= zoom_y2):
                    continue
                
                # Draw new macroblock with distinct visualization
                new_mb_rect = patches.Rectangle(
                    (mb_center_x - macroblock_size/2, mb_center_y - macroblock_size/2),
                    macroblock_size, macroblock_size,
                    linewidth=3, edgecolor='magenta', facecolor='magenta', 
                    alpha=0.4, linestyle='--'
                )
                ax2.add_patch(new_mb_rect)
                
                # Label for new macroblock
                label = f"NEW\n{block_type}\n{magnitude:.1f}px"
                ax2.text(mb_center_x, mb_center_y, label, 
                        fontsize=6, ha='center', va='center', 
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='magenta', alpha=0.8))
                
                # Motion vector arrow
                if magnitude > 0.5:
                    arrow_scale = 1.5
                    ax2.arrow(mb_center_x, mb_center_y, 
                             mv_x * arrow_scale, mv_y * arrow_scale,
                             head_width=3, head_length=3, fc='magenta', ec='magenta', 
                             alpha=0.9, linestyle='--')
        
        # Add legends and information
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(1.0, 0.0, 0.0), label='CENTER'),
            Patch(facecolor=(0.0, 0.8, 0.0), label='EDGE'),
            Patch(facecolor=(0.0, 0.0, 1.0), label='CORNER'),
            Patch(facecolor=(0.8, 0.5, 0.0), label='INTERIOR')
        ]
        
        # Add new macroblock legend if low count frame
        if is_low_count_frame and new_macroblocks:
            legend_elements.append(Patch(facecolor='magenta', alpha=0.6, label='NEW_CANDIDATES'))
        
        ax1.legend(handles=legend_elements, loc='lower right', 
                  bbox_to_anchor=(0.98, 0.02), fontsize=10, 
                  frameon=True, fancybox=True, shadow=True,
                  facecolor='white', edgecolor='black', framealpha=0.9)
        
        # Frame information with macroblock count status
        if macroblock_tracks and frame_num < len(macroblock_tracks[0]['positions']):
            current_frame_info = macroblock_tracks[0]['positions'][frame_num]
            
            # Add macroblock count information
            mb_count_info = ""
            if frame_num < len(tracking_results['frame_macroblock_counts']):
                current_mb_count = tracking_results['frame_macroblock_counts'][frame_num]
                original_total = len(macroblock_tracks)
                percentage = (current_mb_count / original_total) * 100
                mb_count_info = f'MBs in BBox: {current_mb_count}/{original_total} ({percentage:.1f}%)\n'
                
                if is_low_count_frame:
                    new_mb_count = len([mb for mb in new_macroblocks if mb['is_p_block']])
                    mb_count_info += f'⚠️ LOW COUNT! New candidates: {new_mb_count}\n'
            
            frame_info = (f'Frame: {current_frame_info["frame"]}\n'
                         f'GOP Frame: {frame_num + 1}/{gop_frames}\n'
                         f'{mb_count_info}'
                         f'Active MBs: {active_count}/{len(macroblock_tracks)}')
        else:
            frame_info = f'Frame: {frame_num + 1}/{gop_frames}'
            
        ax1.text(0.02, 0.98, frame_info, transform=ax1.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # Object information
        obj_info = (f'Object {object_info["id"]}\\n'
                   f'Size: {object_info["size"][0]:.0f}x{object_info["size"][1]:.0f}\\n'
                   f'Macroblocks: {len(macroblock_tracks)}')
        ax2.text(0.02, 0.98, obj_info, transform=ax2.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        # Summary statistics
        if frame_num < len(macroblock_tracks[0]['positions']):
            summary_stats = tracking_results['summary_stats']
            stats_info = (f'Avg Displacement: {summary_stats["avg_displacement"]:.1f}px\\n'
                         f'Max Displacement: {summary_stats["max_displacement"]:.1f}px\\n'
                         f'Avg Motion/Frame: {summary_stats["avg_motion_per_frame"]:.2f}px')
            ax1.text(0.02, 0.02, stats_info, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))
        
        # Titles
        title_suffix = " (All blocks included - I&P)"
        if is_low_count_frame:
            title_suffix += " ⚠️ LOW MB COUNT!"
            
        ax1.set_title(f'Complete Object Tracking - Full Scene{title_suffix}\\n'
                     f'Object {object_info["id"]} with {len(macroblock_tracks)} Macroblocks', 
                     fontsize=12, fontweight='bold')
        
        zoom_title = f'Zoomed Object View - Detailed Analysis{title_suffix}\\n'
        if is_low_count_frame and new_macroblocks:
            total_candidates = len(new_macroblocks)
            zoom_title += f'Original MBs + {total_candidates} New Candidates (I&P blocks)'
        else:
            zoom_title += f'Macroblock Types and Individual Trajectories'
            
        ax2.set_title(zoom_title, fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('X Position (pixels)', fontsize=11)
        ax1.set_ylabel('Y Position (pixels)', fontsize=11)
        ax2.set_xlabel('X Position (pixels)', fontsize=11)
        ax2.set_ylabel('Y Position (pixels)', fontsize=11)
    
    # Create animation
    print(f"📹 Animating {gop_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=gop_frames, 
                                  interval=1000//4, blit=False, repeat=True)
    
    # Save video
    output_path = "single_object_complete_macroblock_tracking.mp4"
    try:
        writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Complete Object Tracker'),
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
    """Main function to create complete single object tracking."""
    print("🎬 Single Object Complete Macroblock Tracker")
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
    
    # Track all macroblocks through GOP 0
    tracking_results = track_all_macroblocks_in_object(data_loader, object_macroblocks, gop_idx=0)
    
    if tracking_results is None:
        print("❌ Tracking failed")
        return False
    
    # Create visualization video
    success = create_single_object_tracking_video(tracking_results)
    
    if success:
        print("\\n🎉 Single Object Complete Macroblock Tracking completed!")
        print("\\n📊 Analysis Summary:")
        
        stats = tracking_results['summary_stats']
        print(f"  📦 Total Macroblocks Tracked: {stats['total_macroblocks']}")
        print(f"  📏 Average Displacement: {stats['avg_displacement']:.1f} pixels")
        print(f"  📏 Displacement Range: {stats['min_displacement']:.1f} - {stats['max_displacement']:.1f} pixels")
        print(f"  🎯 Average Motion per Frame: {stats['avg_motion_per_frame']:.2f} pixels")
        
        # Block type information
        print(f"\\n� Block Type Processing:")
        print(f"  - Both I-blocks and P-blocks are included in tracking")
        print(f"  - I-block motion vectors are used (with lower confidence)")
        print(f"  - P-block motion vectors have higher confidence")
        print(f"  - Motion arrows: P-blocks (original colors), I-blocks (red)")
        
        # Low count detection information
        low_count_frames = tracking_results['low_count_frames']
        if low_count_frames:
            print(f"\\n⚠️  Low Macroblock Count Detection:")
            print(f"  - {len(low_count_frames)} frames had <20% original macroblocks in bounding box")
            print(f"  - Low count frames: {low_count_frames}")
            print(f"  - New candidates (both I&P blocks) are highlighted in magenta")
            print(f"  - This helps identify when object tracking may need additional macroblocks")
        else:
            print(f"\\n✅ Macroblock Count Status:")
            print(f"  - All frames maintained >20% of original macroblocks")
            print(f"  - No new candidate macroblocks needed")
        
        best_mb = tracking_results['macroblock_tracks'][stats['best_tracking_mb_idx']]
        print(f"\\n🏆 Best Tracking Macroblock: {best_mb['macroblock_info']['type']} at {best_mb['macroblock_info']['position']}")
        print(f"      Displacement: {best_mb['total_displacement']:.1f}px")
        print(f"      Path Length: {best_mb['path_length']:.1f}px")
        
        print("\\n🎯 Enhanced Trajectory Analysis:")
        print("  - CENTER macroblocks typically provide most stable tracking")
        print("  - EDGE macroblocks can show boundary motion effects")
        print("  - CORNER macroblocks may be affected by object deformation")
        print("  - NEW_CANDIDATES (magenta) show potential replacement macroblocks")
        print("  - Both I-blocks and P-blocks contribute to motion analysis")
        print("  - Red motion arrows indicate I-block motion (lower confidence)")
        print("  - Compare individual trajectories to identify best tracking strategy")
        
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create single object tracking video")
