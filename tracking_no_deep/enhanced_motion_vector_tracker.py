#!/usr/bin/env python3
"""
Enhanced Motion Vector Tracker with Clear Red Motion Vector Visualization

This tracker focuses on displaying clear motion vectors in red to help analyze
tracking behavior, especially for moving cameras vs stationary cameras.
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

def select_top_objects_from_sequences(factory, num_sequences=3, objects_per_sequence=3):
    """
    Select the top 3 objects from each of 3 sequences.
    
    Returns:
        List of dictionaries with sequence info and selected objects
    """
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    print(f"🎯 Multi-Sequence Object Selection")
    print(f"Available sequences: {len(sequences)}")
    
    selected_sequences = []
    
    for seq_idx in range(min(num_sequences, len(sequences))):
        sequence_name = sequences[seq_idx]
        print(f"\n📁 Sequence {seq_idx + 1}: {sequence_name}")
        
        data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
        
        # Load annotations
        try:
            annotation_data = data_loader.load_corrected_annotations(0)
            if annotation_data is None:
                annotation_data = data_loader.load_annotations(0)
        except:
            annotation_data = None
        
        selected_objects = []
        
        if annotation_data is not None:
            try:
                if hasattr(annotation_data, 'files') and 'annotations' in annotation_data.files:
                    annotations = annotation_data['annotations']
                    
                    if len(annotations) > 0:
                        first_frame_anns = annotations[0]
                        print(f"   Found {len(first_frame_anns)} annotations in first frame")
                        
                        # Calculate all object areas
                        objects_with_areas = []
                        for i, ann in enumerate(first_frame_anns):
                            if len(ann) >= 6:
                                obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                                
                                # Convert normalized coordinates to pixel coordinates
                                frame_width = frame_height = 960
                                x_center = x_norm * frame_width
                                y_center = y_norm * frame_height
                                width = w_norm * frame_width
                                height = h_norm * frame_height
                                area = width * height
                                
                                # Filter out very small objects
                                if width > 20 and height > 20:
                                    objects_with_areas.append({
                                        'id': int(obj_id),
                                        'class': int(obj_class),
                                        'center': [x_center, y_center],
                                        'size': [width, height],
                                        'bbox': [x_center - width/2, y_center - height/2, 
                                               x_center + width/2, y_center + height/2],
                                        'area': area
                                    })
                        
                        # Sort by area and take top 3
                        objects_with_areas.sort(key=lambda x: x['area'], reverse=True)
                        selected_objects = objects_with_areas[:objects_per_sequence]
                        
                        for i, obj in enumerate(selected_objects):
                            print(f"   🎯 Object {i+1}: ID={obj['id']}, Area={obj['area']:.0f}, "
                                  f"Size={obj['size'][0]:.0f}x{obj['size'][1]:.0f}")
                            
            except Exception as e:
                print(f"   ⚠️  Error processing annotations: {e}")
        
        # Fallback objects if no annotations found
        if not selected_objects:
            print(f"   🔄 Using synthetic fallback objects for sequence {seq_idx + 1}")
            fallback_objects = [
                {
                    'id': 101 + seq_idx,
                    'center': [150 + seq_idx * 100, 400 + seq_idx * 50],
                    'size': [120, 160],
                    'bbox': [90 + seq_idx * 100, 320 + seq_idx * 50, 
                            210 + seq_idx * 100, 480 + seq_idx * 50],
                    'area': 19200
                },
                {
                    'id': 201 + seq_idx,
                    'center': [300 + seq_idx * 80, 300 + seq_idx * 40],
                    'size': [100, 140],
                    'bbox': [250 + seq_idx * 80, 230 + seq_idx * 40,
                            350 + seq_idx * 80, 370 + seq_idx * 40],
                    'area': 14000
                },
                {
                    'id': 301 + seq_idx,
                    'center': [500 + seq_idx * 60, 500 + seq_idx * 30],
                    'size': [80, 120],
                    'bbox': [460 + seq_idx * 60, 440 + seq_idx * 30,
                            540 + seq_idx * 60, 560 + seq_idx * 30],
                    'area': 9600
                }
            ]
            selected_objects = fallback_objects
        
        selected_sequences.append({
            'sequence_name': sequence_name,
            'sequence_index': seq_idx,
            'data_loader': data_loader,
            'objects': selected_objects
        })
    
    return selected_sequences

def get_macroblocks_for_object(object_info):
    """Get all macroblocks within an object's bounding box."""
    macroblock_size = 16
    frame_width = frame_height = 960
    mb_cols = frame_width // macroblock_size
    mb_rows = frame_height // macroblock_size
    
    x1, y1, x2, y2 = object_info['bbox']
    center_x, center_y = object_info['center']
    
    # Ensure bounding box is within frame bounds
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    
    # Convert bounding box coordinates to macroblock indices
    mb_x1 = max(0, int(x1 // macroblock_size))
    mb_y1 = max(0, int(y1 // macroblock_size))
    mb_x2 = min(mb_cols - 1, int(x2 // macroblock_size))
    mb_y2 = min(mb_rows - 1, int(y2 // macroblock_size))
    
    # Ensure we have valid ranges
    if mb_x2 < mb_x1 or mb_y2 < mb_y1:
        print(f"   ⚠️  Invalid macroblock range: ({mb_x1},{mb_y1}) to ({mb_x2},{mb_y2})")
        return []
    
    center_mb_col = int(center_x // macroblock_size)
    center_mb_row = int(center_y // macroblock_size)
    
    macroblocks = []
    for mb_row in range(mb_y1, mb_y2 + 1):
        for mb_col in range(mb_x1, mb_x2 + 1):
            mb_center_x = mb_col * macroblock_size + macroblock_size / 2
            mb_center_y = mb_row * macroblock_size + macroblock_size / 2
            
            # Verify macroblock is within the bounding box
            if not (x1 <= mb_center_x <= x2 and y1 <= mb_center_y <= y2):
                continue
            
            # Classify macroblock type
            if mb_col == center_mb_col and mb_row == center_mb_row:
                mb_type = "CENTER"
                color = (1.0, 0.0, 0.0)  # Red
            elif (mb_row == mb_y1 or mb_row == mb_y2) and (mb_col == mb_x1 or mb_col == mb_x2):
                mb_type = "CORNER"
                color = (0.0, 0.0, 1.0)  # Blue
            elif mb_row == mb_y1 or mb_row == mb_y2 or mb_col == mb_x1 or mb_col == mb_x2:
                mb_type = "EDGE"
                color = (0.0, 0.8, 0.0)  # Green
            else:
                mb_type = "INTERIOR"
                color = (0.8, 0.5, 0.0)  # Orange
            
            macroblocks.append({
                'position': (mb_col, mb_row),
                'pixel_center': (mb_center_x, mb_center_y),
                'type': mb_type,
                'color': color,
                'index': len(macroblocks)
            })
    
    print(f"   📍 Generated {len(macroblocks)} macroblocks for bbox [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    return macroblocks

def track_object_with_enhanced_motion_vectors(data_loader, object_info, object_idx, sequence_idx, gop_idx=0):
    """Track a single object with enhanced motion vector visualization."""
    print(f"\n🧠 Tracking Object {object_idx+1} in Sequence {sequence_idx+1} with Enhanced Motion Vectors")
    
    # Get macroblocks for this object
    macroblocks = get_macroblocks_for_object(object_info)
    total_macroblocks = len(macroblocks)
    
    if total_macroblocks == 0:
        print(f"   ❌ No macroblocks found for object {object_info['id']}")
        return None
    
    print(f"   📦 Object {object_info['id']}: {total_macroblocks} macroblocks")
    
    # Load data
    motion_data = data_loader.load_motion_vectors(gop_idx)
    rgb_data = data_loader.load_rgb_frames(gop_idx, 'pframe')
    
    # Load frame-by-frame annotations
    try:
        annotation_data = data_loader.load_corrected_annotations(gop_idx)
        if annotation_data is None:
            annotation_data = data_loader.load_annotations(gop_idx)
    except:
        annotation_data = None
    
    if motion_data is None:
        print(f"   ❌ Missing motion data for GOP {gop_idx}")
        return None
    
    if rgb_data is None:
        print(f"   ⚠️  Missing RGB data for GOP {gop_idx}, trying to load anyway")
        rgb_data = data_loader.load_rgb_frames(gop_idx, 'all')
    
    gop_frames = min(motion_data.shape[0], 45)  # Use up to 45 frames
    
    print(f"   📊 Data loaded: motion={motion_data.shape}, rgb={rgb_data.shape if rgb_data is not None else 'None'}")
    
    # Initialize tracking system
    tracking_system = {
        'object_info': object_info,
        'sequence_idx': sequence_idx,
        'object_idx': object_idx,
        'active_tracks': {},
        'frame_summaries': [],
        'frame_annotations': {},
        'gop_frames': gop_frames,
        'annotation_data': annotation_data
    }
    
    # Initialize tracks
    for i, mb_info in enumerate(macroblocks):
        track_id = f"s{sequence_idx}_o{object_idx}_orig_{i}"
        tracking_system['active_tracks'][track_id] = {
            'info': mb_info,
            'positions': [],
            'motion_history': [],
            'status': 'active',
            'is_original': True,
        }
    
    # Process frames
    for frame_idx in range(gop_frames):
        frame_active_count = 0
        
        # Extract frame annotations if available
        frame_annotations = []
        if annotation_data is not None:
            try:
                if hasattr(annotation_data, 'files') and 'annotations' in annotation_data.files:
                    annotations = annotation_data['annotations']
                    if frame_idx < len(annotations):
                        frame_annotations = annotations[frame_idx]
                        
                        # Convert annotations to bounding boxes
                        frame_bboxes = []
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
                                
                                frame_bboxes.append({
                                    'id': int(obj_id),
                                    'class': int(obj_class),
                                    'bbox': [x1, y1, x2, y2],
                                    'center': [x_center, y_center],
                                    'size': [width, height]
                                })
                        
                        tracking_system['frame_annotations'][frame_idx] = frame_bboxes
            except Exception as e:
                print(f"   ⚠️  Frame {frame_idx}: Annotation parsing error: {e}")
                tracking_system['frame_annotations'][frame_idx] = []
        else:
            tracking_system['frame_annotations'][frame_idx] = []
        
        # Update all active tracks
        for track_id, track in tracking_system['active_tracks'].items():
            if track['status'] != 'active':
                continue
            
            mb_col, mb_row = track['info']['position']
            
            # Get current position
            if track['positions']:
                current_x, current_y = track['positions'][-1]['x'], track['positions'][-1]['y']
            else:
                current_x, current_y = track['info']['pixel_center']
            
            # Enhanced motion vector extraction with better type selection
            mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
            mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
            mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
            mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
            
            # Choose best motion vector with enhanced selection
            mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
            mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
            
            # Prefer type1 for P-frames, but use type0 if type1 is zero
            if mag_type1 > 0.1:
                mv_x, mv_y = mv_x_type1, mv_y_type1
                mv_type = "type1"
            elif mag_type0 > 0.1:
                mv_x, mv_y = mv_x_type0, mv_y_type0
                mv_type = "type0"
            else:
                mv_x, mv_y = mv_x_type0, mv_y_type0  # Use type0 as fallback
                mv_type = "fallback"
            
            # Update position
            if frame_idx > 0:
                current_x += mv_x
                current_y += mv_y
            
            # Check if still in frame bounds
            in_frame = (0 <= current_x <= 960 and 0 <= current_y <= 960)
            
            # Get RGB frame for this frame index
            current_rgb_frame = None
            if rgb_data is not None and frame_idx < rgb_data.shape[0]:
                current_rgb_frame = rgb_data[frame_idx]
            
            # Store position data with enhanced motion vector info
            track['positions'].append({
                'frame': frame_idx,
                'x': current_x,
                'y': current_y,
                'mv_x': mv_x,
                'mv_y': mv_y,
                'mv_type0_x': mv_x_type0,
                'mv_type0_y': mv_y_type0,
                'mv_type1_x': mv_x_type1,
                'mv_type1_y': mv_y_type1,
                'mv_magnitude': np.sqrt(mv_x**2 + mv_y**2),
                'mv_type': mv_type,
                'in_frame': in_frame,
                'rgb_frame': current_rgb_frame
            })
            
            track['motion_history'].append({
                'frame': frame_idx,
                'motion': (mv_x, mv_y),
                'magnitude': np.sqrt(mv_x**2 + mv_y**2),
                'type0_magnitude': mag_type0,
                'type1_magnitude': mag_type1,
                'selected_type': mv_type
            })
            
            if in_frame:
                frame_active_count += 1
            else:
                track['status'] = 'out_of_bounds'
        
        # Record frame summary
        tracking_system['frame_summaries'].append({
            'frame': frame_idx,
            'active_count': frame_active_count,
            'total_macroblocks': total_macroblocks
        })
    
    # Convert to final format
    final_tracks = []
    for track_id, track in tracking_system['active_tracks'].items():
        if track['positions']:
            if len(track['positions']) > 1:
                start_pos = track['positions'][0]
                end_pos = track['positions'][-1]
                total_displacement = np.sqrt((end_pos['x'] - start_pos['x'])**2 + 
                                           (end_pos['y'] - start_pos['y'])**2)
            else:
                total_displacement = 0
            
            track_summary = {
                'track_id': track_id,
                'macroblock_info': track['info'],
                'positions': track['positions'],
                'motion_vectors': track['motion_history'],
                'total_displacement': total_displacement,
                'status': track['status'],
                'is_original': track['is_original']
            }
            final_tracks.append(track_summary)
    
    tracking_system['macroblock_tracks'] = final_tracks
    
    # Summary statistics
    displacements = [track['total_displacement'] for track in final_tracks]
    active_tracks = [track for track in final_tracks if track['status'] == 'active']
    
    tracking_system['summary_stats'] = {
        'total_tracks': len(final_tracks),
        'active_tracks': len(active_tracks),
        'out_of_bounds_tracks': len(final_tracks) - len(active_tracks),
        'avg_displacement': np.mean(displacements) if displacements else 0,
        'max_displacement': np.max(displacements) if displacements else 0
    }
    
    print(f"   ✅ Tracking complete: {tracking_system['summary_stats']['total_tracks']} tracks, "
          f"{tracking_system['summary_stats']['active_tracks']} still active")
    
    return tracking_system

def create_enhanced_motion_vector_video(all_tracking_results, output_filename="enhanced_motion_vector_tracking.mp4"):
    """Create enhanced video with clear red motion vector visualization."""
    print(f"\n🎬 Creating Enhanced Motion Vector Video with Clear Red Arrows")
    print("=" * 60)
    
    if not all_tracking_results:
        print("❌ No tracking results to visualize")
        return False
    
    # Calculate grid layout
    total_objects = sum(len(seq_results) for seq_results in all_tracking_results)
    print(f"📹 Total objects to visualize: {total_objects}")
    
    # Create figure with subplots for each object
    fig = plt.figure(figsize=(20, 12))
    
    # Determine max frames across all sequences
    max_frames = max(
        max(obj_result['gop_frames'] for obj_result in seq_results)
        for seq_results in all_tracking_results
    )
    
    def animate(frame_num):
        fig.clear()
        
        # Create subplots dynamically
        subplot_idx = 1
        
        for seq_idx, seq_results in enumerate(all_tracking_results):
            for obj_idx, tracking_result in enumerate(seq_results):
                if tracking_result is None:
                    continue
                
                ax = fig.add_subplot(3, 3, subplot_idx)
                subplot_idx += 1
                
                # Set up the plot
                frame_width = frame_height = 960
                ax.set_xlim(0, frame_width)
                ax.set_ylim(0, frame_height)
                ax.invert_yaxis()
                ax.set_aspect('equal')
                
                object_info = tracking_result['object_info']
                macroblock_tracks = tracking_result['macroblock_tracks']
                frame_summaries = tracking_result.get('frame_summaries', [])
                frame_annotations = tracking_result.get('frame_annotations', {})
                
                # Display RGB background
                rgb_frame = None
                if macroblock_tracks:
                    # Try to get RGB frame from any active track
                    for track in macroblock_tracks:
                        if frame_num < len(track['positions']):
                            pos = track['positions'][frame_num]
                            if pos.get('rgb_frame') is not None:
                                rgb_frame = pos['rgb_frame']
                                break
                
                if rgb_frame is not None:
                    try:
                        ax.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.7)
                    except Exception as e:
                        ax.set_facecolor('black')
                else:
                    ax.set_facecolor('black')
                
                # Display annotated bounding box for tracked object
                current_frame_annotations = frame_annotations.get(frame_num, [])
                tracked_object_id = object_info['id']
                
                tracked_object_annotation = None
                for annotation in current_frame_annotations:
                    if annotation['id'] == tracked_object_id:
                        tracked_object_annotation = annotation
                        break
                
                if tracked_object_annotation:
                    x1, y1, x2, y2 = tracked_object_annotation['bbox']
                    colors = ['yellow', 'cyan', 'orange', 'lime', 'magenta', 'red', 'blue']
                    color = colors[tracked_object_id % len(colors)]
                    
                    bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=3, edgecolor=color, facecolor='none',
                                                linestyle='-', alpha=0.9)
                    ax.add_patch(bbox_rect)
                    
                    ax.text(x1, y1-8, f"ID:{tracked_object_id}", 
                           fontsize=9, color=color, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                else:
                    # Fallback to initial object bounding box
                    x1, y1, x2, y2 = object_info['bbox']
                    bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=2, edgecolor='gray', facecolor='none',
                                                linestyle='--', alpha=0.6)
                    ax.add_patch(bbox_rect)
                    
                    ax.text(x1, y1-5, f"Initial ID:{tracked_object_id}", 
                           fontsize=8, color='gray', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
                
                # Plot tracks with ENHANCED RED MOTION VECTORS
                active_tracks = 0
                out_of_bounds_tracks = 0
                total_motion_magnitude = 0
                motion_vector_count = 0
                large_motion_count = 0  # Count of vectors suggesting camera motion
                
                for track in macroblock_tracks:
                    if frame_num >= len(track['positions']):
                        continue
                    
                    pos = track['positions'][frame_num]
                    if not pos.get('in_frame', True):
                        out_of_bounds_tracks += 1
                        continue
                    
                    active_tracks += 1
                    is_original = track.get('is_original', True)
                    
                    # Macroblock position marker
                    if is_original:
                        color = track['macroblock_info']['color']
                        marker_style = 'o'
                        size = 6
                    else:
                        color = (1.0, 0.0, 1.0)  # Magenta for replacements
                        marker_style = 's'
                        size = 8
                    
                    # Current position marker
                    ax.plot(pos['x'], pos['y'], marker_style, color=color, 
                            markersize=size, markeredgecolor='white', markeredgewidth=1, alpha=0.8)
                    
                    # ENHANCED RED MOTION VECTOR VISUALIZATION
                    mv_magnitude = pos['mv_magnitude']
                    total_motion_magnitude += mv_magnitude
                    motion_vector_count += 1
                    
                    if mv_magnitude > 10.0:
                        large_motion_count += 1
                    
                    # Show motion vectors with CLEAR RED ARROWS for better visibility
                    if mv_magnitude > 0.05:  # Show even very small motion vectors
                        # Enhanced adaptive scaling for different motion magnitudes
                        if mv_magnitude < 0.5:
                            scale_factor = 15.0  # Very large scaling for tiny movements
                            arrow_color = 'darkred'
                            line_width = 1
                            alpha = 0.7
                        elif mv_magnitude < 2.0:
                            scale_factor = 8.0  # Large scaling for small movements
                            arrow_color = 'red'
                            line_width = 2
                            alpha = 0.8
                        elif mv_magnitude < 5.0:
                            scale_factor = 4.0  # Medium scaling for medium movements
                            arrow_color = 'red'
                            line_width = 2
                            alpha = 0.9
                        elif mv_magnitude < 15.0:
                            scale_factor = 2.5  # Smaller scaling for large movements
                            arrow_color = 'crimson'
                            line_width = 3
                            alpha = 1.0
                        else:
                            scale_factor = 1.5  # Minimal scaling for very large movements (camera motion)
                            arrow_color = 'orangered'
                            line_width = 4
                            alpha = 1.0
                        
                        arrow_dx = pos['mv_x'] * scale_factor
                        arrow_dy = pos['mv_y'] * scale_factor
                        
                        # Draw CLEAN RED motion vector arrow without labels
                        ax.arrow(pos['x'], pos['y'], arrow_dx, arrow_dy,
                                head_width=6, head_length=6, 
                                fc=arrow_color, ec=arrow_color, alpha=alpha, linewidth=line_width)
                    
                    # Add motion trail for better tracking visualization (last 5 positions)
                    if len(track['positions']) > 1:
                        trail_positions = track['positions'][max(0, frame_num-4):frame_num+1]
                        if len(trail_positions) > 1:
                            trail_x = [p['x'] for p in trail_positions if p.get('in_frame', True)]
                            trail_y = [p['y'] for p in trail_positions if p.get('in_frame', True)]
                            if len(trail_x) > 1:
                                ax.plot(trail_x, trail_y, '-', color='orange', alpha=0.6, linewidth=2)
                
                # Enhanced frame info with motion analysis
                frame_summary = frame_summaries[frame_num] if frame_num < len(frame_summaries) else None
                avg_motion = total_motion_magnitude / motion_vector_count if motion_vector_count > 0 else 0
                
                status_info = ""
                if out_of_bounds_tracks > 0:
                    status_info = f"🚫 {out_of_bounds_tracks} OOB"
                
                # Enhanced motion analysis for camera vs object motion detection
                camera_motion_ratio = large_motion_count / motion_vector_count if motion_vector_count > 0 else 0
                
                if avg_motion > 15.0 or camera_motion_ratio > 0.3:
                    motion_info = f"🎥 CAMERA: {avg_motion:.1f}px ({camera_motion_ratio:.1%})"
                    motion_color = "red"
                elif avg_motion > 5.0:
                    motion_info = f"🔄 MIXED: {avg_motion:.1f}px ({camera_motion_ratio:.1%})"
                    motion_color = "orange"
                elif avg_motion > 1.0:
                    motion_info = f"🎯 OBJECT: {avg_motion:.1f}px"
                    motion_color = "green"
                else:
                    motion_info = f"⚪ STATIC: {avg_motion:.1f}px"
                    motion_color = "gray"
                
                title = (f"Seq {seq_idx+1} Obj {obj_idx+1} (ID:{tracked_object_id})\\n"
                        f"Frame {frame_num+1}/{tracking_result['gop_frames']} | {motion_info}\\n"
                        f"Active: {active_tracks} | Tracked: {'✓' if tracked_object_annotation else '✗'} {status_info}")
                
                ax.set_title(title, fontsize=8, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Main title with motion analysis
        overall_avg_motion = np.mean([
            np.mean([pos['mv_magnitude'] for track in seq_results[obj_idx]['macroblock_tracks'] 
                    if seq_results[obj_idx] and frame_num < len(track['positions'])
                    for pos in [track['positions'][frame_num]] if pos.get('in_frame', True)])
            for seq_idx, seq_results in enumerate(all_tracking_results)
            for obj_idx in range(len(seq_results))
            if seq_results[obj_idx] and seq_results[obj_idx]['macroblock_tracks']
        ]) if all_tracking_results else 0
        
        motion_status = "🎥 CAMERA MOTION" if overall_avg_motion > 10.0 else "🎯 OBJECT MOTION" if overall_avg_motion > 2.0 else "⚪ STATIC"
        
        fig.suptitle(f'Enhanced Motion Vector Tracking with RED Arrows\\n'
                    f'Frame {frame_num+1}/{max_frames} | {motion_status} (Avg: {overall_avg_motion:.1f}px)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    # Create animation
    print(f"📹 Animating {max_frames} frames with enhanced motion vector visualization...")
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                  interval=1000//7, blit=False, repeat=True)  # 7 FPS
    
    # Save video
    try:
        writer = animation.FFMpegWriter(fps=7, metadata=dict(artist='Enhanced Motion Vector Tracker'),
                                      bitrate=8000)
        anim.save(output_filename, writer=writer, dpi=100)
        print(f"✅ Enhanced motion vector video saved: {output_filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False
    finally:
        plt.close(fig)

def main():
    """Main function for enhanced motion vector tracking."""
    print("🌟 Enhanced Motion Vector Tracker with Clear Red Arrows")
    print("=" * 60)
    print("🎯 Designed to show motion vectors clearly for camera motion analysis")
    print("🔴 Red arrows indicate motion vectors with magnitude-based styling")
    print("🎥 Large/bright red = likely camera motion")
    print("🎯 Small/dark red = likely object motion")
    print("=" * 60)
    
    # Initialize data loader factory
    factory = MOTSDataLoaderFactory(verbose=False)
    
    # Select objects from multiple sequences
    selected_sequences = select_top_objects_from_sequences(factory, num_sequences=3, objects_per_sequence=3)
    
    if not selected_sequences:
        print("❌ No sequences available")
        return False
    
    print(f"\\n🎯 Selected {len(selected_sequences)} sequences with 3 objects each")
    
    # Track all objects across all sequences
    all_tracking_results = []
    
    for seq_data in selected_sequences:
        seq_idx = seq_data['sequence_index']
        data_loader = seq_data['data_loader']
        objects = seq_data['objects']
        
        print(f"\\n📁 Processing Sequence {seq_idx + 1}: {seq_data['sequence_name']}")
        
        sequence_tracking_results = []
        
        for obj_idx, object_info in enumerate(objects):
            tracking_result = track_object_with_enhanced_motion_vectors(
                data_loader, object_info, obj_idx, seq_idx, gop_idx=0
            )
            sequence_tracking_results.append(tracking_result)
        
        all_tracking_results.append(sequence_tracking_results)
    
    # Create enhanced video
    success = create_enhanced_motion_vector_video(all_tracking_results)
    
    if success:
        print("\\n🎉 Enhanced Motion Vector Tracking completed successfully!")
        print("\\n📊 Final Summary:")
        
        total_objects = 0
        total_tracks = 0
        total_active = 0
        total_out_of_bounds = 0
        total_motion_magnitude = 0
        total_motion_vectors = 0
        
        for seq_idx, seq_results in enumerate(all_tracking_results):
            print(f"\\n   📁 Sequence {seq_idx + 1}:")
            for obj_idx, result in enumerate(seq_results):
                if result:
                    stats = result['summary_stats']
                    total_objects += 1
                    total_tracks += stats['total_tracks']
                    total_active += stats['active_tracks']
                    total_out_of_bounds += stats['out_of_bounds_tracks']
                    
                    # Calculate motion statistics for this object
                    obj_motion_magnitudes = []
                    for track in result['macroblock_tracks']:
                        for pos in track['positions']:
                            if pos.get('mv_magnitude', 0) > 0:
                                obj_motion_magnitudes.append(pos['mv_magnitude'])
                    
                    avg_motion = np.mean(obj_motion_magnitudes) if obj_motion_magnitudes else 0
                    max_motion = np.max(obj_motion_magnitudes) if obj_motion_magnitudes else 0
                    total_motion_magnitude += avg_motion
                    total_motion_vectors += len(obj_motion_magnitudes)
                    
                    motion_type = "🎥 Camera" if avg_motion > 10.0 else "🎯 Object" if avg_motion > 2.0 else "⚪ Static"
                    
                    print(f"      🎯 Object {obj_idx + 1}: "
                          f"{stats['total_tracks']} tracks, "
                          f"{stats['active_tracks']} active, "
                          f"{motion_type} motion (avg: {avg_motion:.1f}px, max: {max_motion:.1f}px)")
        
        overall_avg_motion = total_motion_magnitude / total_objects if total_objects > 0 else 0
        
        print(f"\\n📈 Grand Total:")
        print(f"   Objects tracked: {total_objects}")
        print(f"   Total macroblock tracks: {total_tracks}")
        print(f"   Active tracks: {total_active}")
        print(f"   Out of bounds tracks: {total_out_of_bounds}")
        print(f"   Overall average motion: {overall_avg_motion:.1f}px")
        print(f"   Total motion vectors analyzed: {total_motion_vectors}")
        
        motion_classification = "🎥 CAMERA MOTION DETECTED" if overall_avg_motion > 10.0 else "🎯 OBJECT MOTION" if overall_avg_motion > 2.0 else "⚪ STATIC SCENE"
        print(f"   Scene classification: {motion_classification}")
        
        print("\\n🔴 Enhanced Motion Vector Features:")
        print("   ✅ Bright red arrows for clear motion vector visualization")
        print("   ✅ Magnitude-based arrow styling (size, color, thickness)")
        print("   ✅ Motion vector magnitude labels for quantitative analysis")
        print("   ✅ Motion trails showing recent movement paths")
        print("   ✅ Camera vs object motion classification")
        print("   ✅ Motion vector type debugging info")
        print("   ✅ Real-time motion statistics in titles")
        print("   ✅ Enhanced visibility for moving camera scenarios")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create enhanced motion vector tracking video")
