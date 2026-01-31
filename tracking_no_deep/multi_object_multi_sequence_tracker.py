#!/usr/bin/env python3
"""
Multi-Object Multi-Sequence Tracker with Intelligent Linking

This tracks the 3 biggest objects across 3 different video sequences,
demonstrating intelligent macroblock replacement and linking for each object.
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

from core.data_loader import MOTSDataLoaderFac                # Enhanced frame info with motion analysis
                frame_summary = frame_summaries[frame_num] if frame_num < len(frame_summaries) else None
                avg_motion = total_motion_magnitude / motion_vector_count if motion_vector_count > 0 else 0
                
                status_info = ""
                if out_of_bounds_tracks > 0:
                    status_info = f"🚫 {out_of_bounds_tracks} out of bounds"
                
                # Enhanced motion analysis for camera motion detection
                if avg_motion > 10.0:
                    motion_info = f"🎥 Camera Motion: {avg_motion:.1f}px"
                    motion_color = "red"
                elif avg_motion > 3.0:
                    motion_info = f"🔄 Mixed Motion: {avg_motion:.1f}px"
                    motion_color = "orange"
                elif avg_motion > 0.5:
                    motion_info = f"🎯 Object Motion: {avg_motion:.1f}px"
                    motion_color = "green"
                else:
                    motion_info = f"⚪ Static: {avg_motion:.1f}px"
                    motion_color = "gray"
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
                                if width > 50 and height > 50:
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
            print(f"   ⚠️  No valid annotations found for sequence {seq_idx + 1}")
            print(f"   � Trying to extract from annotations anyway...")
            
            # Try to get any objects from first frame even if small
            if annotation_data is not None:
                try:
                    if hasattr(annotation_data, 'files') and 'annotations' in annotation_data.files:
                        annotations = annotation_data['annotations']
                        if len(annotations) > 0:
                            first_frame_anns = annotations[0]
                            print(f"   📋 Raw annotations in first frame: {len(first_frame_anns)}")
                            
                            # Get any objects, even small ones
                            all_objects = []
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
                                    
                                    # Accept objects with reasonable size (lowered threshold)
                                    if width > 20 and height > 20:
                                        all_objects.append({
                                            'id': int(obj_id),
                                            'class': int(obj_class),
                                            'center': [x_center, y_center],
                                            'size': [width, height],
                                            'bbox': [x_center - width/2, y_center - height/2, 
                                                   x_center + width/2, y_center + height/2],
                                            'area': area
                                        })
                            
                            if all_objects:
                                # Sort by area and take top 3
                                all_objects.sort(key=lambda x: x['area'], reverse=True)
                                selected_objects = all_objects[:objects_per_sequence]
                                
                                print(f"   ✅ Found {len(selected_objects)} objects with lower threshold:")
                                for i, obj in enumerate(selected_objects):
                                    print(f"      🎯 Object {i+1}: ID={obj['id']}, Area={obj['area']:.0f}, "
                                          f"Size={obj['size'][0]:.0f}x{obj['size'][1]:.0f}")
                except Exception as e:
                    print(f"   ❌ Error extracting objects: {e}")
        
        # Final fallback if still no objects
        if not selected_objects:
            print(f"   �🔄 Using synthetic fallback objects for sequence {seq_idx + 1}")
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

def track_object_with_linking(data_loader, object_info, object_idx, sequence_idx, gop_idx=0):
    """Track a single object with intelligent linking."""
    print(f"\n🧠 Tracking Object {object_idx+1} in Sequence {sequence_idx+1}")
    
    # Get macroblocks for this object
    macroblocks = get_macroblocks_for_object(object_info)
    total_macroblocks = len(macroblocks)
    
    if total_macroblocks == 0:
        print(f"   ❌ No macroblocks found for object {object_info['id']}")
        return None
    
    print(f"   📦 Object {object_info['id']}: {total_macroblocks} macroblocks")
    print(f"   � Intelligent linking disabled - tracking original macroblocks only")
    
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
    
    gop_frames = min(motion_data.shape[0], 45)  # Use up to 45 frames for comprehensive tracking
    
    print(f"   📊 Data loaded: motion={motion_data.shape}, rgb={rgb_data.shape if rgb_data is not None else 'None'}")
    
    # Initialize tracking system
    tracking_system = {
        'object_info': object_info,
        'sequence_idx': sequence_idx,
        'object_idx': object_idx,
        'active_tracks': {},
        'replacement_links': {},
        'frame_summaries': [],
        'frame_annotations': {},  # Store annotations for each frame
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
            'linked_to': None
        }
    
    # Process frames
    for frame_idx in range(gop_frames):
        frame_active_count = 0
        frame_replacements = []
        
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
        
        # Update all active tracks (original macroblocks only)
        for track_id, track in tracking_system['active_tracks'].items():
            if track['status'] != 'active':
                continue
            
            mb_col, mb_row = track['info']['position']
            
            # Get current position
            if track['positions']:
                current_x, current_y = track['positions'][-1]['x'], track['positions'][-1]['y']
            else:
                current_x, current_y = track['info']['pixel_center']
            
            # Get motion vectors with enhanced analysis
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
            
            # Store position data
            track['positions'].append({
                'frame': frame_idx,
                'x': current_x,
                'y': current_y,
                'mv_x': mv_x,
                'mv_y': mv_y,
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

def create_multi_object_video(all_tracking_results, output_filename="multi_object_multi_sequence_tracking.mp4"):
    """Create comprehensive video showing all objects across all sequences."""
    print(f"\n🎬 Creating Multi-Object Multi-Sequence Video")
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
                
                # Display annotated bounding box only for the tracked object
                current_frame_annotations = frame_annotations.get(frame_num, [])
                tracked_object_id = object_info['id']
                
                # Filter annotations to show only the currently tracked object
                tracked_object_annotation = None
                for annotation in current_frame_annotations:
                    if annotation['id'] == tracked_object_id:
                        tracked_object_annotation = annotation
                        break
                
                if tracked_object_annotation:
                    # Draw only the tracked object's bounding box
                    x1, y1, x2, y2 = tracked_object_annotation['bbox']
                    
                    # Use object-specific color
                    colors = ['yellow', 'cyan', 'orange', 'lime', 'magenta', 'red', 'blue']
                    color = colors[tracked_object_id % len(colors)]
                    
                    bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=3, edgecolor=color, facecolor='none',
                                                linestyle='-', alpha=0.9)
                    ax.add_patch(bbox_rect)
                    
                    # Add object ID label
                    ax.text(x1, y1-8, f"Tracked ID:{tracked_object_id}", 
                           fontsize=9, color=color, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
                else:
                    # Fallback to initial object bounding box if no current annotation
                    x1, y1, x2, y2 = object_info['bbox']
                    bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                linewidth=2, edgecolor='gray', facecolor='none',
                                                linestyle='--', alpha=0.6)
                    ax.add_patch(bbox_rect)
                    
                    # Add label for initial bbox
                    ax.text(x1, y1-5, f"Initial ID:{tracked_object_id}", 
                           fontsize=8, color='gray', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
                
                # RGB background
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
                        ax.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.6)
                    except Exception as e:
                        print(f"Warning: Could not display RGB frame for seq {seq_idx+1} obj {obj_idx+1} frame {frame_num}: {e}")
                else:
                    # Set a dark background if no RGB frame is available
                    ax.set_facecolor('black')
                
                # Plot tracks
                active_tracks = 0
                out_of_bounds_tracks = 0
                total_motion_magnitude = 0
                motion_vector_count = 0
                
                for track in macroblock_tracks:
                    if frame_num >= len(track['positions']):
                        continue
                    
                    pos = track['positions'][frame_num]
                    if not pos.get('in_frame', True):
                        out_of_bounds_tracks += 1
                        continue
                    
                    active_tracks += 1
                    is_original = track.get('is_original', True)
                    
                    if is_original:
                        color = track['macroblock_info']['color']
                        marker_style = 'o'
                        size = 6
                    else:
                        color = (1.0, 0.0, 1.0)  # Magenta for replacements
                        marker_style = 's'
                        size = 8
                    
                    # Current position
                    ax.plot(pos['x'], pos['y'], marker_style, color=color, 
                            markersize=size, markeredgecolor='white', markeredgewidth=1, alpha=0.8)
                    
                    # Enhanced motion vector arrow in RED for better visibility - especially for moving camera scenarios
                    mv_magnitude = np.sqrt(pos['mv_x']**2 + pos['mv_y']**2)
                    total_motion_magnitude += mv_magnitude
                    motion_vector_count += 1
                    
                    # Show motion vectors with different styling based on magnitude (useful for camera motion analysis)
                    if mv_magnitude > 0.05:  # Show very small motion vectors too
                        # Enhanced adaptive scaling for better visibility
                        if mv_magnitude < 1.0:
                            scale_factor = 8.0  # Large scaling for tiny movements
                        elif mv_magnitude < 5.0:
                            scale_factor = 4.0  # Medium scaling for small movements
                        else:
                            scale_factor = max(2.0, min(3.0, 30.0 / mv_magnitude))  # Adaptive scaling for large movements
                        
                        arrow_dx = pos['mv_x'] * scale_factor
                        arrow_dy = pos['mv_y'] * scale_factor
                        
                        # Use different red shades based on motion magnitude to distinguish camera vs object motion
                        if mv_magnitude > 10.0:
                            # Very large motion - likely camera movement - bright red
                            arrow_color = 'red'
                            arrow_alpha = 1.0
                            line_width = 3
                        elif mv_magnitude > 3.0:
                            # Medium motion - mixed camera/object motion - medium red
                            arrow_color = 'crimson'
                            arrow_alpha = 0.9
                            line_width = 2
                        else:
                            # Small motion - likely object motion - darker red
                            arrow_color = 'darkred'
                            arrow_alpha = 0.8
                            line_width = 2
                        
                        # Draw enhanced motion vector arrow with better visibility
                        ax.arrow(pos['x'], pos['y'], arrow_dx, arrow_dy,
                                head_width=5, head_length=5, 
                                fc=arrow_color, ec=arrow_color, alpha=arrow_alpha, linewidth=line_width)
                        
                        # Add motion vector magnitude text with color coding
                        if mv_magnitude > 0.5:  # Show text for noticeable motion
                            text_color = 'white' if mv_magnitude > 5.0 else 'yellow'
                            text_bg_color = 'red' if mv_magnitude > 5.0 else 'darkred'
                            
                            ax.text(pos['x'] + arrow_dx * 0.7, pos['y'] + arrow_dy * 0.7, 
                                   f'{mv_magnitude:.1f}',
                                   fontsize=7, color=text_color, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=text_bg_color, alpha=0.8))
                    
                    # Add motion trail for better tracking visualization (last 3 positions)
                    if len(track['positions']) > 1:
                        trail_positions = track['positions'][max(0, frame_num-2):frame_num+1]
                        if len(trail_positions) > 1:
                            trail_x = [p['x'] for p in trail_positions if p.get('in_frame', True)]
                            trail_y = [p['y'] for p in trail_positions if p.get('in_frame', True)]
                            if len(trail_x) > 1:
                                ax.plot(trail_x, trail_y, '-', color='orange', alpha=0.5, linewidth=1)
                
                # Frame info
                frame_summary = frame_summaries[frame_num] if frame_num < len(frame_summaries) else None
                status_info = ""
                if out_of_bounds_tracks > 0:
                    status_info = f"� {out_of_bounds_tracks} out of bounds"
                
                title = (f"Seq {seq_idx+1} Obj {obj_idx+1} (ID:{tracked_object_id})\\n"
                        f"Frame {frame_num+1}/{tracking_result['gop_frames']}\\n"
                        f"Active: {active_tracks} | Tracked: {'✓' if tracked_object_annotation else '✗'} {status_info}")
                
                ax.set_title(title, fontsize=8, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Main title
        fig.suptitle(f'Multi-Object Multi-Sequence Tracking (No Linking)\\n'
                    f'Frame {frame_num+1}/{max_frames} - 3 Sequences × 3 Objects Each', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
    
    # Create animation
    print(f"📹 Animating {max_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=max_frames, 
                                  interval=1000//7, blit=False, repeat=True)  # 7 FPS for 45 frames
    
    # Save video
    try:
        writer = animation.FFMpegWriter(fps=7, metadata=dict(artist='Multi-Object Tracker'),
                                      bitrate=6000)  # Adjusted bitrate for 45 frames
        anim.save(output_filename, writer=writer, dpi=100)
        print(f"✅ Multi-object video saved: {output_filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False
    finally:
        plt.close(fig)

def main():
    """Main function for multi-object multi-sequence tracking."""
    print("🌟 Multi-Object Multi-Sequence Tracker with Intelligent Linking")
    print("=" * 70)
    
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
            tracking_result = track_object_with_linking(
                data_loader, object_info, obj_idx, seq_idx, gop_idx=0
            )
            sequence_tracking_results.append(tracking_result)
        
        all_tracking_results.append(sequence_tracking_results)
    
    # Create comprehensive video
    success = create_multi_object_video(all_tracking_results)
    
    if success:
        print("\\n🎉 Multi-Object Multi-Sequence Tracking completed successfully!")
        print("\\n📊 Final Summary:")
        
        total_objects = 0
        total_tracks = 0
        total_active = 0
        total_out_of_bounds = 0
        
        for seq_idx, seq_results in enumerate(all_tracking_results):
            print(f"\\n   📁 Sequence {seq_idx + 1}:")
            for obj_idx, result in enumerate(seq_results):
                if result:
                    stats = result['summary_stats']
                    total_objects += 1
                    total_tracks += stats['total_tracks']
                    total_active += stats['active_tracks']
                    total_out_of_bounds += stats['out_of_bounds_tracks']
                    
                    print(f"      🎯 Object {obj_idx + 1}: "
                          f"{stats['total_tracks']} tracks, "
                          f"{stats['active_tracks']} active, "
                          f"avg displacement: {stats['avg_displacement']:.1f}px")
        
        print(f"\\n📈 Grand Total:")
        print(f"   Objects tracked: {total_objects}")
        print(f"   Total macroblock tracks: {total_tracks}")
        print(f"   Active tracks: {total_active}")
        print(f"   Out of bounds tracks: {total_out_of_bounds}")
        
        print("\\n🧠 System Features Demonstrated:")
        print("   ✅ Multi-sequence object selection (3 biggest objects per sequence)")
        print("   ✅ Accurate macroblock positioning within bounding boxes")
        print("   ✅ Real-time trajectory tracking with motion vectors")
        print("   ✅ Frame-by-frame object-specific annotation display")
        print("   ✅ Comprehensive multi-panel video visualization (45 frames)")
        print("   ✅ Simplified tracking without unreliable linking system")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create multi-object multi-sequence tracking video")
