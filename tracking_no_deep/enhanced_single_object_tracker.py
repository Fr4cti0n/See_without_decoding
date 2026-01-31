#!/usr/bin/env python3
"""
Enhanced Single Object Complete Macroblock Tracker with Intelligent Linking

This version implements advanced macroblock replacement and linking when original
macroblocks leave the object boundaries, creating continuity in tracking.
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
    
    print(f"🎯 Enhanced Single Object Macroblock Analysis with Intelligent Linking")
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
                        
        except Exception as e:
            print(f"⚠️  Error processing annotations: {e}")
    
    # Force using a smaller object for demonstration (comment out for normal operation)
    selected_object = {
        'id': 999,
        'center': [120, 480],  
        'size': [80, 120],     # Small size to demonstrate replacement
        'bbox': [80, 420, 160, 540],  
        'area': 9600
    }
    print(f"🎯 Demo Object {selected_object['id']} (small size for replacement demo):")
    print(f"    Center: ({selected_object['center'][0]:.1f}, {selected_object['center'][1]:.1f})")
    print(f"    Size: {selected_object['size'][0]:.1f} x {selected_object['size'][1]:.1f}")
    print(f"    Area: {selected_object['area']:.0f} pixels²")
    
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

def track_with_intelligent_linking(data_loader, object_macroblocks, gop_idx=0):
    """
    Enhanced tracking with intelligent macroblock replacement and linking.
    """
    macroblock_size = 16
    print(f"🧠 Enhanced Tracking with Intelligent Linking: {object_macroblocks['total_macroblocks']} macroblocks through GOP {gop_idx}")
    
    # Load data for this GOP
    motion_data = data_loader.load_motion_vectors(gop_idx)
    rgb_data = data_loader.load_rgb_frames(gop_idx, 'pframe')
    macroblock_data = data_loader.load_macroblocks(gop_idx)
    
    if motion_data is None or rgb_data is None:
        print(f"❌ Missing data for GOP {gop_idx}")
        return None
    
    gop_frames = motion_data.shape[0]
    print(f"  GOP {gop_idx}: {gop_frames} frames")
    
    # Enhanced tracking system
    tracking_system = {
        'object_info': object_macroblocks['object_info'],
        'active_tracks': {},  # Currently active macroblock tracks
        'replacement_links': {},  # Links between original and replacement macroblocks
        'replacement_events': [],  # History of all replacement events
        'frame_summaries': [],  # Summary of each frame's tracking status
        'gop_frames': gop_frames
    }
    
    original_bbox = object_macroblocks['object_info']['bbox']
    threshold_count = int(object_macroblocks['total_macroblocks'] * 0.5)  # Use 50% threshold to trigger replacement
    
    print(f"📊 Threshold for replacement: {threshold_count} macroblocks (50% of original for demo)")
    
    # Initialize tracking with original macroblocks
    for i, mb_info in enumerate(object_macroblocks['macroblocks']):
        track_id = f"orig_{i}"
        tracking_system['active_tracks'][track_id] = {
            'info': mb_info,
            'positions': [],
            'motion_history': [],
            'status': 'active',  # 'active', 'lost', 'replaced'
            'is_original': True,
            'linked_to': None,  # ID of replacement macroblock
            'link_strength': None
        }
    
    # Process each frame
    for frame_idx in range(gop_frames):
        frame_active_count = 0
        frame_replacements = []
        
        # Update all active tracks (both original and replacement)
        for track_id, track in tracking_system['active_tracks'].items():
            if track['status'] != 'active':
                continue
                
            # For replacement tracks, use their actual position, not the original info
            if track['is_original']:
                mb_info = track['info']
                mb_col, mb_row = mb_info['position']
            else:
                # For replacement tracks, use the macroblock position from their info
                mb_col, mb_row = track['info']['position']
            
            # Get current position from tracking history
            if track['positions']:
                current_x, current_y = track['positions'][-1]['x'], track['positions'][-1]['y']
            else:
                # First time tracking this macroblock
                current_x, current_y = track['info']['pixel_center']
            
            # Get motion vectors at the macroblock position
            mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
            mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
            mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
            mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
            
            # Choose best motion vector
            mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
            mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
            
            if mag_type1 > mag_type0:
                mv_x, mv_y = mv_x_type1, mv_y_type1
            else:
                mv_x, mv_y = mv_x_type0, mv_y_type0
            
            # Update position with motion vector
            if frame_idx > 0 or not track['is_original']:  # Always apply motion for replacement tracks
                current_x += mv_x
                current_y += mv_y
            
            # Check if still in tracking area
            x1, y1, x2, y2 = original_bbox
            margin = 50  # Allow some margin
            in_tracking_area = (x1 - margin <= current_x <= x2 + margin and 
                              y1 - margin <= current_y <= y2 + margin)
            
            # Store position data
            track['positions'].append({
                'frame': frame_idx,
                'x': current_x,
                'y': current_y,
                'mv_x': mv_x,
                'mv_y': mv_y,
                'in_area': in_tracking_area,
                'rgb_frame': rgb_data[frame_idx] if frame_idx < rgb_data.shape[0] else None
            })
            
            track['motion_history'].append({
                'frame': frame_idx,
                'motion': (mv_x, mv_y),
                'magnitude': np.sqrt(mv_x**2 + mv_y**2)
            })
            
            if in_tracking_area:
                frame_active_count += 1
            else:
                print(f"    📤 Track {track_id} left tracking area at frame {frame_idx}")
                track['status'] = 'lost'
        
        # Check if we need replacements
        if frame_active_count < threshold_count:
            print(f"    ⚠️  Only {frame_active_count} active tracks (threshold: {threshold_count}) - finding replacements...")
            
            # Find replacement candidates
            candidates = find_replacement_candidates(
                motion_data, macroblock_data, frame_idx, original_bbox, tracking_system['active_tracks']
            )
            
            # Link lost tracks to best candidates
            new_links = create_intelligent_links(
                tracking_system['active_tracks'], candidates, frame_idx
            )
            
            # Add new replacement tracks with proper trajectory tracking
            for lost_track_id, replacement_info in new_links.items():
                new_track_id = f"repl_{frame_idx}_{len(tracking_system['active_tracks'])}"
                
                # Start tracking from current position and continue forward
                initial_x, initial_y = replacement_info['pixel_center']
                
                # Create replacement track with initial position
                tracking_system['active_tracks'][new_track_id] = {
                    'info': replacement_info,
                    'positions': [{
                        'frame': frame_idx,
                        'x': initial_x,
                        'y': initial_y,
                        'mv_x': replacement_info['motion_vector'][0],
                        'mv_y': replacement_info['motion_vector'][1],
                        'in_area': True,
                        'rgb_frame': rgb_data[frame_idx] if frame_idx < rgb_data.shape[0] else None
                    }],
                    'motion_history': [{
                        'frame': frame_idx,
                        'motion': replacement_info['motion_vector'],
                        'magnitude': replacement_info['magnitude']
                    }],
                    'status': 'active',
                    'is_original': False,
                    'linked_to': lost_track_id,
                    'link_strength': replacement_info.get('link_strength', 0.5),
                    'created_at_frame': frame_idx
                }
                
                # Update the lost track
                tracking_system['active_tracks'][lost_track_id]['status'] = 'replaced'
                tracking_system['active_tracks'][lost_track_id]['linked_to'] = new_track_id
                
                # Record the link
                tracking_system['replacement_links'][lost_track_id] = new_track_id
                
                frame_replacements.append({
                    'lost_track': lost_track_id,
                    'replacement_track': new_track_id,
                    'link_strength': replacement_info.get('link_strength', 0.5),
                    'frame': frame_idx
                })
                
                print(f"    🔗 Created link: {lost_track_id} → {new_track_id} "
                      f"(strength: {replacement_info.get('link_strength', 0.5):.2f})")
        
        # Record frame summary
        tracking_system['frame_summaries'].append({
            'frame': frame_idx,
            'active_count': frame_active_count,
            'replacements': frame_replacements,
            'needs_replacement': frame_active_count < threshold_count
        })
        
        print(f"    📊 Frame {frame_idx}: {frame_active_count} active tracks (threshold: {threshold_count})")
        if frame_active_count < threshold_count:
            print(f"    ⚡ THRESHOLD REACHED - Would create replacements!")
    
    # Convert to final format
    final_tracks = []
    for track_id, track in tracking_system['active_tracks'].items():
        if track['positions']:  # Only include tracks with position data
            # Calculate displacement
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
                'is_original': track['is_original'],
                'linked_to': track['linked_to'],
                'link_strength': track.get('link_strength', 1.0 if track['is_original'] else 0.5)
            }
            final_tracks.append(track_summary)
    
    tracking_system['macroblock_tracks'] = final_tracks
    
    # Calculate summary statistics
    displacements = [track['total_displacement'] for track in final_tracks]
    tracking_system['summary_stats'] = {
        'total_tracks': len(final_tracks),
        'original_tracks': sum(1 for track in final_tracks if track['is_original']),
        'replacement_tracks': sum(1 for track in final_tracks if not track['is_original']),
        'total_links': len(tracking_system['replacement_links']),
        'avg_displacement': np.mean(displacements) if displacements else 0,
        'max_displacement': np.max(displacements) if displacements else 0,
        'frames_with_replacements': len([f for f in tracking_system['frame_summaries'] if f['replacements']])
    }
    
    print(f"\n🧠 Enhanced Tracking Summary:")
    print(f"  Total tracks: {tracking_system['summary_stats']['total_tracks']}")
    print(f"  Original tracks: {tracking_system['summary_stats']['original_tracks']}")
    print(f"  Replacement tracks: {tracking_system['summary_stats']['replacement_tracks']}")
    print(f"  Total links created: {tracking_system['summary_stats']['total_links']}")
    
    return tracking_system

def find_replacement_candidates(motion_data, macroblock_data, frame_idx, original_bbox, active_tracks):
    """Find potential replacement macroblocks with current frame bounding box update."""
    
    # Calculate updated bounding box based on active track positions at current frame
    active_positions = []
    for track_id, track in active_tracks.items():
        if track['status'] == 'active' and track['positions']:
            last_pos = track['positions'][-1]
            active_positions.append((last_pos['x'], last_pos['y']))
    
    if active_positions:
        # Update bounding box based on current active macroblock positions
        xs, ys = zip(*active_positions)
        current_x1 = min(xs) - 32  # Add margin
        current_y1 = min(ys) - 32
        current_x2 = max(xs) + 32
        current_y2 = max(ys) + 32
        
        # Ensure it doesn't shrink too much from original
        x1, y1, x2, y2 = original_bbox
        current_x1 = min(current_x1, x1)
        current_y1 = min(current_y1, y1)
        current_x2 = max(current_x2, x2)
        current_y2 = max(current_y2, y2)
    else:
        # Fall back to original bounding box
        current_x1, current_y1, current_x2, current_y2 = original_bbox
    
    mb_x1 = max(0, int(current_x1 // 16))
    mb_y1 = max(0, int(current_y1 // 16))
    mb_x2 = min(59, int(current_x2 // 16))
    mb_y2 = min(59, int(current_y2 // 16))
    
    print(f"    📦 Updated search area: ({current_x1:.0f}, {current_y1:.0f}) to ({current_x2:.0f}, {current_y2:.0f})")
    print(f"    📊 Macroblock search: ({mb_x1}, {mb_y1}) to ({mb_x2}, {mb_y2})")
    
    candidates = []
    
    for mb_row in range(mb_y1, mb_y2 + 1):
        for mb_col in range(mb_x1, mb_x2 + 1):
            # Check if position is already tracked
            is_tracked = any(
                track['info']['position'] == (mb_col, mb_row) and track['status'] == 'active'
                for track in active_tracks.values()
            )
            
            if not is_tracked:
                mb_center_x = mb_col * 16 + 8
                mb_center_y = mb_row * 16 + 8
                
                # Get motion data for current frame
                mv_x_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 0])
                mv_y_type0 = float(motion_data[frame_idx, 0, mb_row, mb_col, 1])
                mv_x_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
                mv_y_type1 = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
                
                mag_type0 = np.sqrt(mv_x_type0**2 + mv_y_type0**2)
                mag_type1 = np.sqrt(mv_x_type1**2 + mv_y_type1**2)
                
                # Choose the motion vector with higher magnitude
                if mag_type1 > mag_type0:
                    mv_x, mv_y = mv_x_type1, mv_y_type1
                    motion_magnitude = mag_type1
                else:
                    mv_x, mv_y = mv_x_type0, mv_y_type0
                    motion_magnitude = mag_type0
                
                # Only consider macroblocks with significant motion (above threshold)
                if motion_magnitude > 0.5:  # Minimum motion threshold
                    candidates.append({
                        'position': (mb_col, mb_row),
                        'pixel_center': (mb_center_x, mb_center_y),
                        'motion_vector': (mv_x, mv_y),
                        'magnitude': motion_magnitude,
                        'type': 'REPLACEMENT',
                        'color': (1.0, 0.0, 1.0),  # Magenta for replacements
                        'quality_score': motion_magnitude
                    })
    
    print(f"    🔍 Found {len(candidates)} moving replacement candidates")
    return sorted(candidates, key=lambda x: x['quality_score'], reverse=True)[:10]  # Top 10 candidates

def create_intelligent_links(active_tracks, candidates, frame_idx):
    """Create intelligent links between lost tracks and replacement candidates."""
    links = {}
    
    # Find recently lost tracks
    lost_tracks = [(track_id, track) for track_id, track in active_tracks.items() 
                   if track['status'] == 'lost' and track.get('linked_to') is None]
    
    if not lost_tracks or not candidates:
        return links
    
    used_candidates = set()
    
    for track_id, track in lost_tracks:
        if not track['positions']:
            continue
        
        last_pos = track['positions'][-1]
        last_x, last_y = last_pos['x'], last_pos['y']
        
        # Calculate recent motion pattern
        recent_motions = track['motion_history'][-3:] if len(track['motion_history']) >= 3 else track['motion_history']
        if recent_motions:
            avg_mv_x = sum(m['motion'][0] for m in recent_motions) / len(recent_motions)
            avg_mv_y = sum(m['motion'][1] for m in recent_motions) / len(recent_motions)
            avg_magnitude = sum(m['magnitude'] for m in recent_motions) / len(recent_motions)
        else:
            avg_mv_x, avg_mv_y, avg_magnitude = 0, 0, 0
        
        best_candidate = None
        best_link_strength = 0
        
        for candidate in candidates:
            if candidate['position'] in used_candidates:
                continue
            
            cand_x, cand_y = candidate['pixel_center']
            cand_mv_x, cand_mv_y = candidate['motion_vector']
            cand_magnitude = candidate['magnitude']
            
            # Spatial proximity score
            spatial_distance = np.sqrt((cand_x - last_x)**2 + (cand_y - last_y)**2)
            spatial_score = max(0, 1.0 - spatial_distance / 80.0)
            
            # Motion pattern similarity
            motion_diff_x = abs(cand_mv_x - avg_mv_x)
            motion_diff_y = abs(cand_mv_y - avg_mv_y)
            motion_similarity = max(0, 1.0 - np.sqrt(motion_diff_x**2 + motion_diff_y**2) / 8.0)
            
            # Motion magnitude similarity
            magnitude_diff = abs(cand_magnitude - avg_magnitude)
            magnitude_similarity = max(0, 1.0 - magnitude_diff / 5.0)
            
            # Combined link strength
            link_strength = (0.4 * spatial_score + 0.3 * motion_similarity + 0.3 * magnitude_similarity)
            
            if link_strength > best_link_strength and link_strength > 0.4:  # Minimum threshold
                best_link_strength = link_strength
                best_candidate = candidate
        
        if best_candidate:
            best_candidate['link_strength'] = best_link_strength
            links[track_id] = best_candidate
            used_candidates.add(best_candidate['position'])
    
    return links

def create_enhanced_tracking_video(tracking_results):
    """Create visualization for enhanced tracking with intelligent linking."""
    if tracking_results is None:
        print("❌ No tracking results to visualize")
        return False

    print("🎬 Creating Enhanced Single Object Tracking Video with Intelligent Linking")
    print("=" * 70)
    
    object_info = tracking_results['object_info']
    macroblock_tracks = tracking_results['macroblock_tracks']
    gop_frames = tracking_results['gop_frames']
    replacement_links = tracking_results.get('replacement_links', {})
    frame_summaries = tracking_results.get('frame_summaries', [])
    
    print(f"📹 Object {object_info['id']}: {len(macroblock_tracks)} total tracks, {gop_frames} frames")
    print(f"🔗 Total replacement links: {len(replacement_links)}")
    
    # Create video with enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    frame_width = frame_height = 960
    macroblock_size = 16
    
    def animate(frame_num):
        ax1.clear()
        ax2.clear()
        
        # Get frame summary
        frame_summary = frame_summaries[frame_num] if frame_num < len(frame_summaries) else None
        
        # ---- LEFT PANEL: Full scene ----
        ax1.set_xlim(0, frame_width)
        ax1.set_ylim(0, frame_height)
        ax1.invert_yaxis()
        ax1.set_aspect('equal')
        
        # RGB background
        rgb_frame = None
        for track in macroblock_tracks:
            if frame_num < len(track['positions']) and track['positions'][frame_num]['rgb_frame'] is not None:
                rgb_frame = track['positions'][frame_num]['rgb_frame']
                break
        
        if rgb_frame is not None:
            ax1.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.7)
        
        # Grid
        for i in range(0, frame_width + 1, macroblock_size * 4):
            ax1.axvline(x=i, color='white', alpha=0.3, linewidth=0.8)
        for i in range(0, frame_height + 1, macroblock_size * 4):
            ax1.axhline(y=i, color='white', alpha=0.3, linewidth=0.8)
        
        # Object bounding box
        x1, y1, x2, y2 = object_info['bbox']
        bbox_rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                    linewidth=3, edgecolor='yellow', facecolor='none',
                                    linestyle='--', alpha=0.8)
        ax1.add_patch(bbox_rect)
        
        # Plot all active tracks
        active_tracks = 0
        original_tracks = 0
        replacement_tracks = 0
        
        for track in macroblock_tracks:
            if frame_num >= len(track['positions']):
                continue
            
            pos = track['positions'][frame_num]
            if not pos.get('in_area', True):
                continue  # Skip tracks that left the area
                
            active_tracks += 1
            
            mb_info = track['macroblock_info']
            is_original = track.get('is_original', True)
            
            if is_original:
                original_tracks += 1
                color = mb_info['color']
                alpha = 0.8
                marker_style = 'o'
                linewidth = 2
            else:
                replacement_tracks += 1
                color = (1.0, 0.0, 1.0)  # Magenta for replacements
                alpha = 0.9
                marker_style = 's'  # Square for replacements
                linewidth = 3
            
            # Draw trajectory
            if len(track['positions']) > 1:
                positions_so_far = track['positions'][:frame_num + 1]
                x_coords = [p['x'] for p in positions_so_far if p.get('in_area', True)]
                y_coords = [p['y'] for p in positions_so_far if p.get('in_area', True)]
                
                if len(x_coords) > 1:
                    ax1.plot(x_coords, y_coords, '-', color=color, 
                            linewidth=linewidth, alpha=alpha)
            
            # Current position
            ax1.plot(pos['x'], pos['y'], marker_style, color=color, 
                    markersize=8 if is_original else 10, 
                    markeredgecolor='white', markeredgewidth=2, alpha=alpha)
            
            # Motion vector arrow
            if abs(pos['mv_x']) > 0.5 or abs(pos['mv_y']) > 0.5:
                arrow_scale = 2
                ax1.arrow(pos['x'], pos['y'], 
                         pos['mv_x'] * arrow_scale, pos['mv_y'] * arrow_scale,
                         head_width=5, head_length=5, fc=color, ec=color, 
                         alpha=alpha)
        
        # ---- RIGHT PANEL: Zoomed view ----
        ax2.set_xlim(x1 - 50, x2 + 50)
        ax2.set_ylim(y1 - 50, y2 + 50)
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        
        # Zoomed background
        if rgb_frame is not None:
            ax2.imshow(rgb_frame, extent=[0, frame_width, frame_height, 0], alpha=0.8)
        
        # Detailed track visualization in zoom
        for track in macroblock_tracks:
            if frame_num >= len(track['positions']):
                continue
                
            pos = track['positions'][frame_num]
            if not pos.get('in_area', True):
                continue
                
            is_original = track.get('is_original', True)
            
            if is_original:
                color = track['macroblock_info']['color']
                label = track['macroblock_info']['type'][0]
            else:
                color = (1.0, 0.0, 1.0)
                label = "R"  # Replacement
                link_strength = track.get('link_strength', 0.5)
                
                # Show link strength
                ax2.text(pos['x'] + 8, pos['y'] - 8, f"{link_strength:.2f}", 
                        fontsize=7, color='magenta', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
            
            # Macroblock rectangle
            mb_rect = patches.Rectangle(
                (pos['x'] - macroblock_size/2, pos['y'] - macroblock_size/2),
                macroblock_size, macroblock_size,
                linewidth=2, edgecolor=color, facecolor=color, alpha=0.4
            )
            ax2.add_patch(mb_rect)
            
            # Label
            ax2.text(pos['x'], pos['y'], label, 
                    fontsize=8, ha='center', va='center', 
                    color='white', fontweight='bold')
        
        # Add legends
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(1.0, 0.0, 0.0), label='CENTER'),
            Patch(facecolor=(0.0, 0.8, 0.0), label='EDGE'),
            Patch(facecolor=(0.0, 0.0, 1.0), label='CORNER'),
            Patch(facecolor=(0.8, 0.5, 0.0), label='INTERIOR'),
            Patch(facecolor=(1.0, 0.0, 1.0), label='REPLACEMENT')
        ]
        
        ax1.legend(handles=legend_elements, loc='lower right', 
                  bbox_to_anchor=(0.98, 0.02), fontsize=10)
        
        # Frame information
        frame_info = f'Frame: {frame_num + 5}\\nGOP Frame: {frame_num + 1}/{gop_frames}\\n'
        frame_info += f'Active: {active_tracks} ({original_tracks} orig, {replacement_tracks} repl)'
        
        if frame_summary and frame_summary.get('needs_replacement', False):
            frame_info += '\\n⚠️ REPLACEMENT NEEDED'
            
        if frame_summary and frame_summary.get('replacements'):
            frame_info += f'\\n🔗 {len(frame_summary["replacements"])} new links'
        
        ax1.text(0.02, 0.98, frame_info, transform=ax1.transAxes,
                fontsize=11, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # Object info
        obj_info = f'Enhanced Object Tracking\\nIntelligent Linking System\\nTotal Links: {len(replacement_links)}'
        ax2.text(0.02, 0.98, obj_info, transform=ax2.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        # Titles
        title_suffix = " - WITH INTELLIGENT LINKING"
        if frame_summary and frame_summary.get('needs_replacement', False):
            title_suffix += " ⚠️ REPLACEMENT ACTIVE!"
            
        ax1.set_title(f'Enhanced Object Tracking{title_suffix}\\n'
                     f'Original + Replacement Macroblocks', 
                     fontsize=12, fontweight='bold')
        
        ax2.set_title(f'Intelligent Linking System\\n'
                     f'Link Qualities and Replacement Status', 
                     fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('X Position (pixels)', fontsize=11)
        ax1.set_ylabel('Y Position (pixels)', fontsize=11)
        ax2.set_xlabel('X Position (pixels)', fontsize=11)
        ax2.set_ylabel('Y Position (pixels)', fontsize=11)
    
    # Create animation
    print(f"📹 Animating {gop_frames} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=gop_frames, 
                                  interval=1000//4, blit=False, repeat=True)
    
    # Save video
    output_path = "enhanced_single_object_tracking_with_linking.mp4"
    try:
        writer = animation.FFMpegWriter(fps=4, metadata=dict(artist='Enhanced Object Tracker'),
                                      bitrate=4000)
        anim.save(output_path, writer=writer, dpi=120)
        print(f"✅ Enhanced video saved: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        return False
    finally:
        plt.close(fig)

def main():
    """Main function for enhanced tracking with intelligent linking."""
    print("🧠 Enhanced Single Object Tracker with Intelligent Linking")
    print("=" * 60)
    
    # Initialize data loader
    factory = MOTSDataLoaderFactory(verbose=True)
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    sequence_name = sequences[0] if sequences else 'MOT17-09-SDP_960x960_gop50_500frames'
    data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
    
    print(f"✅ Using sequence: {sequence_name}")
    
    # Load annotations
    try:
        annotation_data = data_loader.load_corrected_annotations(0)
        if annotation_data is None:
            annotation_data = data_loader.load_annotations(0)
    except:
        annotation_data = None
    
    # Load first frame
    rgb_data_first = data_loader.load_rgb_frames(0, 'pframe')
    first_frame = rgb_data_first[0] if rgb_data_first is not None else None
    
    # Select object and macroblocks
    object_macroblocks = select_single_object_macroblocks(annotation_data, first_frame)
    
    if not object_macroblocks or object_macroblocks['total_macroblocks'] == 0:
        print("❌ No macroblocks selected")
        return False
    
    # Enhanced tracking with intelligent linking
    tracking_results = track_with_intelligent_linking(data_loader, object_macroblocks, gop_idx=0)
    
    if tracking_results is None:
        print("❌ Enhanced tracking failed")
        return False
    
    # Create enhanced visualization
    success = create_enhanced_tracking_video(tracking_results)
    
    if success:
        print("\\n🎉 Enhanced Single Object Tracking with Intelligent Linking completed!")
        print("\\n📊 Enhanced Analysis Summary:")
        
        stats = tracking_results['summary_stats']
        print(f"  📦 Total Tracks: {stats['total_tracks']}")
        print(f"  📊 Original Tracks: {stats['original_tracks']}")
        print(f"  🔄 Replacement Tracks: {stats['replacement_tracks']}")
        print(f"  🔗 Total Links Created: {stats['total_links']}")
        print(f"  📈 Average Displacement: {stats['avg_displacement']:.1f} pixels")
        print(f"  🎯 Frames with Replacements: {stats['frames_with_replacements']}")
        
        print("\\n🧠 Intelligent Linking Features:")
        print("  - Tracks macroblock trajectories and detects when they leave object area")
        print("  - Finds replacement macroblocks within the object boundaries")
        print("  - Creates intelligent links based on spatial proximity and motion similarity")
        print("  - Maintains tracking continuity even when original macroblocks are lost")
        print("  - Visualizes original (colored) and replacement (magenta) macroblocks")
        print("  - Shows link quality scores for replacement macroblocks")
        print("  - Provides frame-by-frame replacement status and warnings")
    
    return success

if __name__ == "__main__":
    success = main()
    if not success:
        print("❌ Failed to create enhanced tracking video")
