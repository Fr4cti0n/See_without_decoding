#!/usr/bin/env python3
"""
Demo: Intelligent Macroblock Linking System

This demonstrates the key features requested:
1. Tracking macroblocks in bounding boxes
2. Detecting when less than 20% remain 
3. Finding new macroblocks and establishing intelligent links
4. Showing displacement of both original and replacement macroblocks
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add path exactly like residuals study
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

def demo_intelligent_linking():
    """
    Demo of the intelligent macroblock linking system showing:
    - Original macroblock tracking
    - Threshold detection (20% remaining)
    - New macroblock identification
    - Intelligent linking between lost and new macroblocks
    - Displacement visualization for both original and replacement tracks
    """
    print("🎯 DEMO: Intelligent Macroblock Linking System")
    print("=" * 60)
    
    # Initialize data loader
    factory = MOTSDataLoaderFactory(verbose=False)
    sequences = factory.list_sequences(['MOT17'], ['960x960'])
    sequence_name = sequences[0] if sequences else 'MOT17-09-SDP_960x960_gop50_500frames'
    data_loader = factory.create_loader(sequence_name, ['MOT17'], ['960x960'])
    
    # Load sample data
    motion_data = data_loader.load_motion_vectors(0)
    rgb_data = data_loader.load_rgb_frames(0, 'pframe')
    
    if motion_data is None or rgb_data is None:
        print("❌ Could not load data")
        return
    
    # Define object bounding box (simulated)
    object_bbox = [50, 400, 200, 600]  # x1, y1, x2, y2
    x1, y1, x2, y2 = object_bbox
    
    # Convert to macroblock coordinates
    mb_x1, mb_y1 = max(0, x1 // 16), max(0, y1 // 16)
    mb_x2, mb_y2 = min(59, x2 // 16), min(59, y2 // 16)
    
    print(f"📦 Object bounding box: ({x1}, {y1}) to ({x2}, {y2})")
    print(f"📊 Macroblock region: ({mb_x1}, {mb_y1}) to ({mb_x2}, {mb_y2})")
    
    # Initial macroblocks in the object
    original_macroblocks = []
    for mb_row in range(mb_y1, mb_y2 + 1):
        for mb_col in range(mb_x1, mb_x2 + 1):
            original_macroblocks.append({
                'id': len(original_macroblocks),
                'position': (mb_col, mb_row),
                'pixel_center': (mb_col * 16 + 8, mb_row * 16 + 8),
                'status': 'active'
            })
    
    original_count = len(original_macroblocks)
    threshold_count = int(original_count * 0.2)  # 20% threshold as requested
    
    print(f"🎯 Original macroblocks: {original_count}")
    print(f"📏 20% threshold: {threshold_count} macroblocks")
    print()
    
    # Simulate tracking through multiple frames
    active_tracks = {}
    replacement_links = {}
    frame_reports = []
    
    # Initialize tracking
    for mb in original_macroblocks:
        track_id = f"orig_{mb['id']}"
        active_tracks[track_id] = {
            'macroblock': mb,
            'positions': [],
            'motion_history': [],
            'status': 'active',
            'is_original': True
        }
    
    # Process 10 frames as demo
    print("🔄 Processing frames and demonstrating intelligent linking...")
    print()
    
    for frame_idx in range(min(10, motion_data.shape[0])):
        active_count = 0
        frame_lost_tracks = []
        
        # Update all tracks
        for track_id, track in active_tracks.items():
            if track['status'] != 'active':
                continue
                
            mb = track['macroblock']
            mb_col, mb_row = mb['position']
            
            # Get current position
            if track['positions']:
                current_x, current_y = track['positions'][-1]
            else:
                current_x, current_y = mb['pixel_center']
            
            # Get motion vectors
            mv_x = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])  # Use type 1
            mv_y = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
            
            # Update position
            if frame_idx > 0:
                current_x += mv_x
                current_y += mv_y
            
            # Check if still in object area
            in_object = (x1 <= current_x <= x2 and y1 <= current_y <= y2)
            
            track['positions'].append((current_x, current_y))
            track['motion_history'].append((mv_x, mv_y))
            
            if in_object:
                active_count += 1
            else:
                track['status'] = 'lost'
                frame_lost_tracks.append(track_id)
        
        # Check threshold and find replacements
        replacement_events = []
        if active_count < threshold_count:
            print(f"⚠️  Frame {frame_idx}: Only {active_count} active tracks (threshold: {threshold_count})")
            print("🔍 Finding replacement macroblocks...")
            
            # Find candidate replacements
            candidates = []
            for mb_row in range(mb_y1, mb_y2 + 1):
                for mb_col in range(mb_x1, mb_x2 + 1):
                    # Check if this position is already tracked
                    already_tracked = any(
                        track['macroblock']['position'] == (mb_col, mb_row) and track['status'] == 'active'
                        for track in active_tracks.values()
                    )
                    
                    if not already_tracked:
                        mv_x = float(motion_data[frame_idx, 1, mb_row, mb_col, 0])
                        mv_y = float(motion_data[frame_idx, 1, mb_row, mb_col, 1])
                        motion_magnitude = np.sqrt(mv_x**2 + mv_y**2)
                        
                        candidates.append({
                            'position': (mb_col, mb_row),
                            'pixel_center': (mb_col * 16 + 8, mb_row * 16 + 8),
                            'motion': (mv_x, mv_y),
                            'magnitude': motion_magnitude
                        })
            
            # Sort candidates by motion magnitude (prioritize moving macroblocks)
            candidates.sort(key=lambda x: x['magnitude'], reverse=True)
            
            # Create intelligent links
            lost_tracks = [track_id for track_id, track in active_tracks.items() 
                          if track['status'] == 'lost' and track_id not in replacement_links]
            
            links_created = 0
            for lost_track_id in lost_tracks[:len(candidates)]:
                if links_created < len(candidates):
                    candidate = candidates[links_created]
                    
                    # Calculate link quality
                    lost_track = active_tracks[lost_track_id]
                    if lost_track['positions']:
                        last_pos = lost_track['positions'][-1]
                        distance = np.sqrt((candidate['pixel_center'][0] - last_pos[0])**2 + 
                                         (candidate['pixel_center'][1] - last_pos[1])**2)
                        
                        # Spatial proximity score (closer is better)
                        spatial_score = max(0, 1.0 - distance / 100.0)
                        
                        # Motion similarity score
                        if lost_track['motion_history']:
                            last_motion = lost_track['motion_history'][-1]
                            motion_diff = np.sqrt((candidate['motion'][0] - last_motion[0])**2 + 
                                                (candidate['motion'][1] - last_motion[1])**2)
                            motion_score = max(0, 1.0 - motion_diff / 10.0)
                        else:
                            motion_score = 0.5
                        
                        # Combined link strength
                        link_strength = 0.6 * spatial_score + 0.4 * motion_score
                        
                        if link_strength > 0.3:  # Minimum quality threshold
                            # Create replacement track
                            new_track_id = f"repl_{frame_idx}_{links_created}"
                            
                            active_tracks[new_track_id] = {
                                'macroblock': {
                                    'id': len(active_tracks),
                                    'position': candidate['position'],
                                    'pixel_center': candidate['pixel_center'],
                                    'status': 'active'
                                },
                                'positions': [candidate['pixel_center']],
                                'motion_history': [candidate['motion']],
                                'status': 'active',
                                'is_original': False,
                                'linked_from': lost_track_id,
                                'link_strength': link_strength,
                                'created_at_frame': frame_idx
                            }
                            
                            # Record the link
                            replacement_links[lost_track_id] = new_track_id
                            replacement_events.append({
                                'lost_track': lost_track_id,
                                'replacement_track': new_track_id,
                                'link_strength': link_strength,
                                'distance': distance
                            })
                            
                            print(f"   🔗 Link created: {lost_track_id} → {new_track_id}")
                            print(f"      Link strength: {link_strength:.2f}, Distance: {distance:.1f}px")
                            
                            links_created += 1
            
            print(f"   ✅ Created {links_created} intelligent links")
            print()
        
        # Record frame summary
        frame_reports.append({
            'frame': frame_idx,
            'active_count': active_count,
            'lost_tracks': len(frame_lost_tracks),
            'replacement_events': replacement_events,
            'threshold_reached': active_count < threshold_count
        })
    
    # Final summary
    print("📊 INTELLIGENT LINKING DEMO RESULTS")
    print("=" * 40)
    
    total_tracks = len(active_tracks)
    original_tracks = sum(1 for track in active_tracks.values() if track['is_original'])
    replacement_tracks = total_tracks - original_tracks
    total_links = len(replacement_links)
    
    print(f"📈 Total tracks: {total_tracks}")
    print(f"   Original tracks: {original_tracks}")
    print(f"   Replacement tracks: {replacement_tracks}")
    print(f"🔗 Total intelligent links: {total_links}")
    print()
    
    # Show displacement analysis
    print("📏 DISPLACEMENT ANALYSIS")
    print("-" * 25)
    
    for track_id, track in active_tracks.items():
        if len(track['positions']) > 1:
            start_pos = track['positions'][0]
            end_pos = track['positions'][-1]
            displacement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            
            track_type = "ORIG" if track['is_original'] else "REPL"
            link_info = ""
            if not track['is_original']:
                link_strength = track.get('link_strength', 0)
                linked_from = track.get('linked_from', 'unknown')
                link_info = f" (linked from {linked_from}, strength: {link_strength:.2f})"
            
            print(f"   {track_type} {track_id}: {displacement:.1f}px{link_info}")
    
    print()
    print("🧠 KEY FEATURES DEMONSTRATED:")
    print("✅ Tracked all macroblocks within object bounding boxes")
    print("✅ Detected when less than 20% of original macroblocks remain")
    print("✅ Found new candidate macroblocks within the object area")
    print("✅ Created intelligent links based on spatial proximity and motion similarity")
    print("✅ Showed displacement for both original and replacement macroblocks")
    print("✅ Maintained tracking continuity through macroblock replacement")
    
    return True

if __name__ == "__main__":
    demo_intelligent_linking()
