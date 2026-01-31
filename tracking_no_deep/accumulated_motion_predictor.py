#!/usr/bin/env python3
"""
Accumulated Motion Vector Predictor for All Fully Visible Objects

This script accumulates motion vectors from the beginning of each GOP until the last P-frame
and estimates the new bounding box positions for all objects that are fully visible throughout
each GOP (present in all 49 frames).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import ndimage

# Add path for data loader
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

class AccumulatedMotionPredictor:
    """Accumulates motion vectors and predicts bounding box positions for all fully visible objects."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.factory = MOTSDataLoaderFactory(verbose=verbose)
        self.data_loader = None
        
        # Target objects - will be determined dynamically per GOP
        self.target_objects = None  # Dynamic per GOP
        
        # Motion accumulation parameters
        self.motion_smoothing_sigma = 0.5  # Gaussian smoothing for motion field
        
        # Evaluation parameters
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # For mAP calculation
        
        # Visualization parameters
        self.base_colors = [
            (255, 100, 100),  # Red
            (100, 100, 255),  # Blue  
            (100, 255, 100),  # Green
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
            (100, 255, 255),  # Cyan
            (255, 150, 50),   # Orange
            (150, 50, 255),   # Purple
            (50, 255, 150),   # Light Green
            (255, 50, 150),   # Pink
        ]
        
    def get_colors_for_objects(self, object_ids):
        """Generate color mapping for a set of object IDs."""
        colors = {}
        for i, obj_id in enumerate(sorted(object_ids)):
            colors[obj_id] = self.base_colors[i % len(self.base_colors)]
        return colors
        
    def load_sequence(self, sequence_name=None):
        """Load a MOTS sequence."""
        sequences = self.factory.list_sequences(['MOT17'], ['960x960'])
        if not sequences:
            print("❌ No sequences found!")
            return False, []
        
        if sequence_name:
            if sequence_name not in sequences:
                print(f"❌ Sequence {sequence_name} not found!")
                print(f"Available sequences: {sequences}")
                return False, []
            target_sequence = sequence_name
        else:
            target_sequence = sequences[0]
            
        print(f"📂 Loading sequence: {target_sequence}")
        self.data_loader = self.factory.create_loader(target_sequence, ['MOT17'], ['960x960'])
        return True, sequences
    
    def load_gop_data(self, gop_idx):
        """Load motion vectors, RGB frames, and annotations for a GOP."""
        try:
            # Load motion vectors
            motion_data = self.data_loader.load_motion_vectors(gop_idx)
            if motion_data is None:
                return None
                
            # Load RGB frames  
            rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
            if rgb_data is None:
                return None
                
            # Load annotations
            annotation_data = self.data_loader.load_corrected_annotations(gop_idx)
            if annotation_data is None:
                return None
            
            # Combine into single data structure
            gop_data = {
                'motion_vectors': motion_data,
                'rgb_frames': rgb_data, 
                'annotations': annotation_data
            }
            
            return gop_data
            
        except Exception as e:
            print(f"❌ Error loading GOP {gop_idx}: {e}")
            return None
    
    def smooth_motion_field(self, motion_field):
        """Apply Gaussian smoothing to motion field."""
        smoothed = np.zeros_like(motion_field)
        smoothed[:, :, 0] = ndimage.gaussian_filter(motion_field[:, :, 0], self.motion_smoothing_sigma)
        smoothed[:, :, 1] = ndimage.gaussian_filter(motion_field[:, :, 1], self.motion_smoothing_sigma)
        return smoothed

    def find_fully_visible_objects(self, gop_data):
        """Find all objects that are fully visible throughout the entire GOP."""
        if not gop_data or 'annotations' not in gop_data:
            return []
        
        # Track object presence across all frames
        object_frame_count = {}
        annotations_data = gop_data['annotations']['annotations']
        total_frames = len(annotations_data)
        
        for frame_idx in range(total_frames):
            frame_annotations = annotations_data[frame_idx]
            for ann in frame_annotations:
                if len(ann) >= 6:
                    obj_id = int(ann[0])
                    if obj_id not in object_frame_count:
                        object_frame_count[obj_id] = 0
                    object_frame_count[obj_id] += 1
        
        # Find objects present in all frames (fully visible)
        fully_visible_objects = []
        for obj_id, frame_count in object_frame_count.items():
            if frame_count == total_frames:
                fully_visible_objects.append(obj_id)
        
        # Sort for consistent ordering
        fully_visible_objects.sort()
        
        if self.verbose:
            print(f"   🎯 Found {len(fully_visible_objects)} fully visible objects: {fully_visible_objects}")
            print(f"   📊 Total objects seen: {len(object_frame_count)}")
            for obj_id, count in sorted(object_frame_count.items()):
                visibility = (count / total_frames) * 100
                status = "✓ Fully visible" if count == total_frames else f"⚠ Partial ({visibility:.1f}%)"
                print(f"      Object {obj_id}: {count}/{total_frames} frames - {status}")
        
        return fully_visible_objects
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1, box2: [x, y, width, height] format
        """
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
        x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_map_per_object(self, predicted_boxes, ground_truth_boxes, object_ids):
        """Calculate mAP for each object across all frames.
        
        Args:
            predicted_boxes: dict {obj_id: [list of predicted boxes]}
            ground_truth_boxes: dict {obj_id: [list of ground truth boxes]}
            object_ids: list of object IDs to evaluate
        
        Returns:
            dict: {obj_id: {'mAP': value, 'AP_per_threshold': {threshold: AP}}}
        """
        results = {}
        
        for obj_id in object_ids:
            if obj_id not in predicted_boxes or obj_id not in ground_truth_boxes:
                results[obj_id] = {'mAP': 0.0, 'AP_per_threshold': {}}
                continue
                
            pred_boxes = predicted_boxes[obj_id]
            gt_boxes = ground_truth_boxes[obj_id]
            
            # Ensure same number of predictions and ground truths
            min_len = min(len(pred_boxes), len(gt_boxes))
            pred_boxes = pred_boxes[:min_len]
            gt_boxes = gt_boxes[:min_len]
            
            ap_per_threshold = {}
            
            for threshold in self.iou_thresholds:
                correct_predictions = 0
                
                for pred_box, gt_box in zip(pred_boxes, gt_boxes):
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou >= threshold:
                        correct_predictions += 1
                
                # Calculate Average Precision for this threshold
                ap = correct_predictions / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
                ap_per_threshold[threshold] = ap
            
            # Calculate mAP (mean over all thresholds)
            mAP = np.mean(list(ap_per_threshold.values()))
            
            results[obj_id] = {
                'mAP': mAP,
                'AP_per_threshold': ap_per_threshold,
                'total_predictions': len(pred_boxes),
                'total_ground_truths': len(gt_boxes)
            }
        
        return results
        
    def process_multiple_sequences(self, sequences_to_test=None, num_gops=3):
        """Process multiple sequences for comparison across different video conditions."""
        
        # Get available sequences
        success, all_sequences = self.load_sequence()
        if not success:
            return {}
        
        # Select sequences to test
        if sequences_to_test is None:
            # Take first 3 available sequences or all if less than 3
            sequences_to_test = all_sequences[:min(3, len(all_sequences))]
        else:
            # Validate provided sequences
            sequences_to_test = [seq for seq in sequences_to_test if seq in all_sequences]
            if not sequences_to_test:
                print("❌ None of the provided sequences are available!")
                return {}
        
        print(f"🎯 Processing {len(sequences_to_test)} sequences for comparison:")
        for i, seq in enumerate(sequences_to_test, 1):
            print(f"   {i}. {seq}")
        
        all_results = {}
        
        for seq_idx, sequence_name in enumerate(sequences_to_test):
            print(f"\n{'='*80}")
            print(f"🎬 SEQUENCE {seq_idx + 1}/{len(sequences_to_test)}: {sequence_name}")
            print(f"{'='*80}")
            
            # Load this sequence
            success, _ = self.load_sequence(sequence_name)
            if not success:
                print(f"❌ Failed to load sequence {sequence_name}")
                continue
            
            # Process GOPs for this sequence
            sequence_results = self.process_gops(num_gops, sequence_prefix=f"seq{seq_idx + 1}_")
            all_results[sequence_name] = sequence_results
        
        # Generate comparison summary
        self.generate_sequence_comparison_summary(all_results, sequences_to_test)
        
        return all_results
    
    def generate_sequence_comparison_summary(self, all_results, sequences):
        """Generate a comprehensive comparison summary across all sequences."""
        
        print(f"\n{'='*80}")
        print(f"📊 MULTI-SEQUENCE COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Collect overall statistics
        overall_stats = {}
        
        for seq_name, seq_results in all_results.items():
            if not seq_results:
                continue
                
            seq_stats = {
                'total_objects': 0,
                'total_gops': len(seq_results),
                'mean_mAP': [],
                'mean_AP50': [],
                'mean_AP75': [],
                'best_performing_objects': [],
                'worst_performing_objects': []
            }
            
            for gop_idx, gop_data in seq_results.items():
                if 'mean_metrics' not in gop_data:
                    continue
                    
                metrics = gop_data['mean_metrics']
                seq_stats['mean_mAP'].append(metrics['mean_mAP'])
                seq_stats['mean_AP50'].append(metrics['mean_AP50'])
                seq_stats['mean_AP75'].append(metrics['mean_AP75'])
                seq_stats['total_objects'] += len(gop_data.get('individual_results', {}))
                
                # Find best and worst performing objects
                for obj_id, obj_result in gop_data.get('individual_results', {}).items():
                    obj_mAP = obj_result['mAP']
                    if obj_mAP > 0.7:  # Good performance threshold
                        seq_stats['best_performing_objects'].append((obj_id, obj_mAP, gop_idx))
                    elif obj_mAP < 0.3:  # Poor performance threshold
                        seq_stats['worst_performing_objects'].append((obj_id, obj_mAP, gop_idx))
            
            # Calculate sequence averages
            if seq_stats['mean_mAP']:
                seq_stats['avg_mAP'] = np.mean(seq_stats['mean_mAP'])
                seq_stats['avg_AP50'] = np.mean(seq_stats['mean_AP50'])
                seq_stats['avg_AP75'] = np.mean(seq_stats['mean_AP75'])
            else:
                seq_stats['avg_mAP'] = 0.0
                seq_stats['avg_AP50'] = 0.0
                seq_stats['avg_AP75'] = 0.0
            
            overall_stats[seq_name] = seq_stats
        
        # Print sequence comparison
        print(f"\n🎯 SEQUENCE PERFORMANCE COMPARISON:")
        print(f"{'Sequence':<30} {'Avg mAP':<10} {'Avg AP@0.5':<12} {'Avg AP@0.75':<12} {'Objects':<8} {'GOPs':<5}")
        print("-" * 80)
        
        for seq_name, stats in overall_stats.items():
            short_name = seq_name.split('_')[0] if '_' in seq_name else seq_name[:25]
            print(f"{short_name:<30} {stats['avg_mAP']:<10.3f} {stats['avg_AP50']:<12.3f} "
                  f"{stats['avg_AP75']:<12.3f} {stats['total_objects']:<8} {stats['total_gops']:<5}")
        
        # Find best and worst sequences
        if overall_stats:
            best_seq = max(overall_stats.items(), key=lambda x: x[1]['avg_mAP'])
            worst_seq = min(overall_stats.items(), key=lambda x: x[1]['avg_mAP'])
            
            print(f"\n🏆 BEST PERFORMING SEQUENCE: {best_seq[0]}")
            print(f"   Average mAP: {best_seq[1]['avg_mAP']:.3f}")
            print(f"   Average AP@0.5: {best_seq[1]['avg_AP50']:.3f}")
            print(f"   Average AP@0.75: {best_seq[1]['avg_AP75']:.3f}")
            
            print(f"\n⚠️  WORST PERFORMING SEQUENCE: {worst_seq[0]}")
            print(f"   Average mAP: {worst_seq[1]['avg_mAP']:.3f}")
            print(f"   Average AP@0.5: {worst_seq[1]['avg_AP50']:.3f}")
            print(f"   Average AP@0.75: {worst_seq[1]['avg_AP75']:.3f}")
        
        # Overall insights
        print(f"\n💡 INSIGHTS:")
        total_objects = sum(stats['total_objects'] for stats in overall_stats.values())
        avg_overall_mAP = np.mean([stats['avg_mAP'] for stats in overall_stats.values()])
        
        print(f"   📊 Total objects tracked: {total_objects}")
        print(f"   📊 Overall average mAP: {avg_overall_mAP:.3f}")
        
        if avg_overall_mAP > 0.6:
            print(f"   ✅ Good overall performance - motion accumulation works well!")
        elif avg_overall_mAP > 0.4:
            print(f"   ⚠️  Moderate performance - some challenging conditions detected")
        else:
            print(f"   ❌ Poor overall performance - significant challenges with motion accumulation")
        
        print(f"\n📹 Generated videos: *seq*_accumulated_motion_prediction_gop*.mp4")
        print(f"🎯 Compare different sequences to understand performance variations")
        
        return overall_stats
    
    def extract_target_objects(self, annotations, target_object_ids=None):
        """Extract target objects from annotations."""
        target_objects = {}
        
        # Use provided IDs or default to objects 19 and 20
        if target_object_ids is None:
            target_object_ids = [19, 20]
        
        for ann in annotations:
            if len(ann) >= 6:
                obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                
                if int(obj_id) in target_object_ids:
                    # Convert to pixel coordinates
                    x_center = x_norm * 960
                    y_center = y_norm * 960
                    width = w_norm * 960
                    height = h_norm * 960
                    
                    # Calculate macroblock position (16x16 macroblocks)
                    mb_col = int(x_center // 16)
                    mb_row = int(y_center // 16)
                    
                    target_objects[int(obj_id)] = {
                        'id': int(obj_id),
                        'center': [x_center, y_center],
                        'size': [width, height],
                        'macroblock': [mb_col, mb_row],
                        'bbox': [x_center - width/2, y_center - height/2, width, height]
                    }
        
        return target_objects
    
    def accumulate_motion_vectors(self, gop_data, target_objects):
        """
        Accumulate motion vectors from I-frame to last P-frame.
        Also tracks ground truth bounding boxes for evaluation.
        
        Returns accumulated motion for each object and frame-by-frame motion.
        """
        if not gop_data or 'motion_vectors' not in gop_data:
            return {}, {}, {}
        
        motion_vectors = gop_data['motion_vectors']
        annotations_data = gop_data['annotations']['annotations']
        num_frames = motion_vectors.shape[0]
        
        # Initialize accumulation tracking
        accumulated_motion = {}
        frame_motions = {}
        ground_truth_boxes = {}
        
        for obj_id in target_objects.keys():
                
            accumulated_motion[obj_id] = {
                'total_displacement': np.array([0.0, 0.0]),
                'frame_displacements': [],
                'positions': [target_objects[obj_id]['center'].copy()],
                'bounding_boxes': [target_objects[obj_id]['bbox'].copy()]
            }
            frame_motions[obj_id] = []
            ground_truth_boxes[obj_id] = []
        
        print(f"🎯 Accumulating motion vectors across {num_frames} frames...")
        
        # Process each frame (0 = I-frame, 1-48 = P-frames)
        for frame_idx in range(num_frames):
            # Get motion field for this frame (use layer 0 for P-frames)
            motion_field = motion_vectors[frame_idx, 0]  # Shape: (60, 60, 2)
            
            # Apply smoothing
            smoothed_motion = self.smooth_motion_field(motion_field)
            
            print(f"   📍 Frame {frame_idx}: Motion field shape {motion_field.shape}")
            
            for obj_id in target_objects.keys():
                    
                # Get current object position
                current_pos = accumulated_motion[obj_id]['positions'][-1]
                current_bbox = accumulated_motion[obj_id]['bounding_boxes'][-1]
                
                # Calculate macroblock position
                mb_col = int(current_pos[0] // 16)
                mb_row = int(current_pos[1] // 16)
                
                # Clamp to valid range
                mb_col = np.clip(mb_col, 0, smoothed_motion.shape[1] - 1)
                mb_row = np.clip(mb_row, 0, smoothed_motion.shape[0] - 1)
                
                # Extract motion vector at object position
                motion_vec = smoothed_motion[mb_row, mb_col]
                
                # Accumulate motion
                accumulated_motion[obj_id]['total_displacement'] += motion_vec
                accumulated_motion[obj_id]['frame_displacements'].append(motion_vec.copy())
                frame_motions[obj_id].append(motion_vec.copy())
                
                # Calculate new position
                new_pos = [
                    current_pos[0] + motion_vec[0],
                    current_pos[1] + motion_vec[1]
                ]
                
                # Clamp to frame bounds
                new_pos[0] = np.clip(new_pos[0], 0, 959)
                new_pos[1] = np.clip(new_pos[1], 0, 959)
                
                # Calculate new bounding box
                obj_size = target_objects[obj_id]['size']
                new_bbox = [
                    new_pos[0] - obj_size[0]/2,
                    new_pos[1] - obj_size[1]/2,
                    obj_size[0],
                    obj_size[1]
                ]
                
                # Store new position and bbox
                accumulated_motion[obj_id]['positions'].append(new_pos)
                accumulated_motion[obj_id]['bounding_boxes'].append(new_bbox)
                
                if self.verbose and frame_idx % 10 == 0:
                    displacement = accumulated_motion[obj_id]['total_displacement']
                    print(f"      🔵 Object {obj_id}: Frame {frame_idx}, Motion: {motion_vec}, "
                          f"Total displacement: {displacement}, Position: {new_pos}")
            
            # Extract ground truth bounding boxes for this frame
            if frame_idx < len(annotations_data):
                frame_annotations = annotations_data[frame_idx]
                for ann in frame_annotations:
                    if len(ann) >= 6:
                        ann_obj_id, _, x_norm, y_norm, w_norm, h_norm = ann[:6]
                        ann_obj_id = int(ann_obj_id)
                        
                        if ann_obj_id in target_objects:
                            # Convert normalized coordinates to pixel coordinates
                            x_center = x_norm * 960
                            y_center = y_norm * 960
                            width = w_norm * 960
                            height = h_norm * 960
                            
                            # Convert to [x, y, width, height] format
                            gt_bbox = [
                                x_center - width/2,
                                y_center - height/2,
                                width,
                                height
                            ]
                            
                            ground_truth_boxes[ann_obj_id].append(gt_bbox)
        
        return accumulated_motion, frame_motions, ground_truth_boxes
    
    def visualize_accumulated_motion(self, gop_data, accumulated_motion, frame_motions, ground_truth_boxes, gop_idx=0, sequence_prefix=""):
        """Create visualization showing accumulated motion, predicted positions, and ground truth bounding boxes."""
        
        if not gop_data or 'rgb_frames' not in gop_data:
            print("❌ No RGB frames available for visualization")
            return
            
        rgb_frames = gop_data['rgb_frames']
        num_frames = len(rgb_frames)
        
        # Create video writer with sequence prefix
        if sequence_prefix:
            output_path = f'{sequence_prefix}_accumulated_motion_prediction_gop{gop_idx}.mp4'
        else:
            output_path = f'accumulated_motion_prediction_gop{gop_idx}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8
        frame_size = (960, 960)
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        print(f"🎬 Creating accumulated motion visualization: {output_path}")
        
        for frame_idx in range(num_frames):
            # Get RGB frame
            rgb_frame = rgb_frames[frame_idx].copy()
            
            # Convert to BGR for OpenCV
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw accumulated motion paths and predictions
            for obj_id in accumulated_motion.keys():
                    
                color = self.colors[obj_id]
                positions = accumulated_motion[obj_id]['positions']
                bboxes = accumulated_motion[obj_id]['bounding_boxes']
                
                # Draw motion trail
                for i in range(1, min(frame_idx + 2, len(positions))):
                    prev_pos = positions[i-1]
                    curr_pos = positions[i]
                    
                    # Draw motion vector arrow
                    pt1 = (int(prev_pos[0]), int(prev_pos[1]))
                    pt2 = (int(curr_pos[0]), int(curr_pos[1]))
                    
                    # Draw trail line with fading effect
                    alpha = 0.3 + 0.7 * (i / len(positions))
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.arrowedLine(bgr_frame, pt1, pt2, trail_color, 2, tipLength=0.3)
                
                # Draw current predicted bounding box
                if frame_idx < len(bboxes):
                    bbox = bboxes[frame_idx]
                    x, y, w, h = bbox
                    
                    # Draw predicted bounding box (solid line)
                    cv2.rectangle(bgr_frame, 
                                (int(x), int(y)), 
                                (int(x + w), int(y + h)), 
                                color, 3)
                    
                    # Draw object ID with "PRED" label
                    cv2.putText(bgr_frame, f'PRED Obj {obj_id}', 
                              (int(x), int(y) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw accumulated displacement info
                    total_disp = accumulated_motion[obj_id]['total_displacement']
                    disp_text = f'Disp: ({total_disp[0]:.1f}, {total_disp[1]:.1f})'
                    cv2.putText(bgr_frame, disp_text,
                              (int(x), int(y + h + 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw ground truth bounding box if available
                if obj_id in ground_truth_boxes and frame_idx < len(ground_truth_boxes[obj_id]):
                    gt_bbox = ground_truth_boxes[obj_id][frame_idx]
                    gt_x, gt_y, gt_w, gt_h = gt_bbox
                    
                    # Draw ground truth bounding box (dashed line style using smaller rectangles)
                    dash_length = 5
                    for i in range(0, int(gt_w), dash_length * 2):
                        start_x = int(gt_x + i)
                        end_x = min(int(gt_x + i + dash_length), int(gt_x + gt_w))
                        cv2.line(bgr_frame, (start_x, int(gt_y)), (end_x, int(gt_y)), color, 2)
                        cv2.line(bgr_frame, (start_x, int(gt_y + gt_h)), (end_x, int(gt_y + gt_h)), color, 2)
                    
                    for i in range(0, int(gt_h), dash_length * 2):
                        start_y = int(gt_y + i)
                        end_y = min(int(gt_y + i + dash_length), int(gt_y + gt_h))
                        cv2.line(bgr_frame, (int(gt_x), start_y), (int(gt_x), end_y), color, 2)
                        cv2.line(bgr_frame, (int(gt_x + gt_w), start_y), (int(gt_x + gt_w), end_y), color, 2)
                    
                    # Draw "GT" label
                    cv2.putText(bgr_frame, f'GT Obj {obj_id}', 
                              (int(gt_x), int(gt_y) - 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw frame info
            cv2.putText(bgr_frame, f'Frame {frame_idx} - GOP {gop_idx}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(bgr_frame, f'Predicted (solid) vs Ground Truth (dashed)', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, f'Accumulated Motion Prediction with mAP Evaluation', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(bgr_frame)
            
            if frame_idx % 10 == 0:
                print(f"   📍 Processing frame {frame_idx}/{num_frames}")
        
        out.release()
        print(f"✅ Accumulated motion visualization created: {output_path}")
        
        # Print final accumulated motion summary
        print(f"\n📊 Final Accumulated Motion Summary (GOP {gop_idx}):")
        for obj_id in accumulated_motion.keys():
            total_disp = accumulated_motion[obj_id]['total_displacement']
            initial_pos = accumulated_motion[obj_id]['positions'][0]
            final_pos = accumulated_motion[obj_id]['positions'][-1]
            
            print(f"   🔵 Object {obj_id}:")
            print(f"      Initial position: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f})")
            print(f"      Final position: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
            print(f"      Total displacement: ({total_disp[0]:.1f}, {total_disp[1]:.1f})")
            print(f"      Distance moved: {np.linalg.norm(total_disp):.1f} pixels")
    
    def process_gops(self, num_gops=3, sequence_prefix=""):
        """Process multiple GOPs and show accumulated motion predictions for all fully visible objects."""
        
        print(f"🎯 Processing {num_gops} GOPs for accumulated motion prediction...")
        
        # Track overall statistics
        all_gop_maps = []
        all_gop_ap50s = []
        all_gop_ap75s = []
        total_objects_processed = 0
        gop_results = {}
        
        for gop_idx in range(num_gops):
            print(f"\n📖 Loading GOP {gop_idx}...")
            
            # Load GOP data
            gop_data = self.load_gop_data(gop_idx)
            if not gop_data:
                print(f"❌ Failed to load GOP {gop_idx}")
                continue
            
            # Get annotations from first frame (I-frame)
            if 'annotations' not in gop_data or not gop_data['annotations']:
                print(f"❌ No annotations found for GOP {gop_idx}")
                continue
            
            # Find all fully visible objects in this GOP
            fully_visible_objects = self.find_fully_visible_objects(gop_data)
            
            if not fully_visible_objects:
                print(f"⚠️  No fully visible objects found in GOP {gop_idx}")
                continue
            
            # Extract target objects data
            first_frame_annotations = gop_data['annotations']['annotations'][0]
            target_objects = self.extract_target_objects(first_frame_annotations, fully_visible_objects)
            
            if not target_objects:
                print(f"⚠️  No target objects data found in GOP {gop_idx}")
                continue
                
            print(f"   🎯 Processing {len(target_objects)} fully visible objects: {list(target_objects.keys())}")
            
            # Set up colors for this GOP's objects
            self.colors = self.get_colors_for_objects(list(target_objects.keys()))
            
            # Accumulate motion vectors and get ground truth
            accumulated_motion, frame_motions, ground_truth_boxes = self.accumulate_motion_vectors(gop_data, target_objects)
            
            # Calculate mAP for each object
            predicted_boxes = {obj_id: accumulated_motion[obj_id]['bounding_boxes'] for obj_id in accumulated_motion}
            map_results = self.calculate_map_per_object(predicted_boxes, ground_truth_boxes, list(target_objects.keys()))
            
            # Create visualization
            self.visualize_accumulated_motion(gop_data, accumulated_motion, frame_motions, ground_truth_boxes, gop_idx, sequence_prefix)
            
            # Print mAP results
            print(f"\n📊 mAP Evaluation Results (GOP {gop_idx}):")
            
            # Collect metrics for mean calculation
            all_maps = []
            all_ap50s = []
            all_ap75s = []
            
            for obj_id in sorted(map_results.keys()):
                result = map_results[obj_id]
                print(f"   🔵 Object {obj_id}:")
                print(f"      mAP@[0.5:0.95]: {result['mAP']:.3f}")
                print(f"      AP@0.5: {result['AP_per_threshold'][0.5]:.3f}")
                print(f"      AP@0.75: {result['AP_per_threshold'][0.75]:.3f}")
                print(f"      Predictions/GT: {result['total_predictions']}/{result['total_ground_truths']}")
                
                # Collect for mean calculation
                all_maps.append(result['mAP'])
                all_ap50s.append(result['AP_per_threshold'][0.5])
                all_ap75s.append(result['AP_per_threshold'][0.75])
                
                # Show some individual IoU thresholds
                high_precision_thresholds = [0.8, 0.85, 0.9, 0.95]
                high_aps = [result['AP_per_threshold'][th] for th in high_precision_thresholds]
                avg_high_ap = np.mean(high_aps)
                print(f"      High-precision mAP@[0.8:0.95]: {avg_high_ap:.3f}")
            
            # Calculate and display mean metrics
            if all_maps:
                mean_map = np.mean(all_maps)
                mean_ap50 = np.mean(all_ap50s)
                mean_ap75 = np.mean(all_ap75s)
                
                # Store for overall summary
                all_gop_maps.append(mean_map)
                all_gop_ap50s.append(mean_ap50)
                all_gop_ap75s.append(mean_ap75)
                total_objects_processed += len(all_maps)
                
                # Store GOP results for return
                gop_results[gop_idx] = {
                    'mean_metrics': {
                        'mean_mAP': mean_map,
                        'mean_AP50': mean_ap50,
                        'mean_AP75': mean_ap75,
                        'num_objects': len(all_maps)
                    },
                    'individual_results': map_results
                }
                
                print(f"\n📈 Overall Performance (GOP {gop_idx}):")
                print(f"   🎯 Mean mAP@[0.5:0.95]: {mean_map:.3f}")
                print(f"   🎯 Mean AP@0.5: {mean_ap50:.3f}")
                print(f"   🎯 Mean AP@0.75: {mean_ap75:.3f}")
                print(f"   📊 Number of objects: {len(all_maps)}")
                
                # Performance assessment
                if mean_map >= 0.7:
                    performance_level = "🟢 Excellent"
                elif mean_map >= 0.5:
                    performance_level = "🟡 Good"
                elif mean_map >= 0.3:
                    performance_level = "🟠 Fair"
                else:
                    performance_level = "🔴 Poor"
                    
                print(f"   🏆 Performance Level: {performance_level}")
        
        print(f"\n✅ Accumulated motion prediction completed!")
        print(f"   📹 Generated videos: accumulated_motion_prediction_gop*.mp4")
        print(f"   🎯 Focus: All fully visible objects motion accumulation and position prediction")
        print(f"   ⭐ Use: ffplay accumulated_motion_prediction_gop0.mp4 to view")
        
        # Display overall summary across all GOPs
        if all_gop_maps:
            overall_mean_map = np.mean(all_gop_maps)
            overall_mean_ap50 = np.mean(all_gop_ap50s)
            overall_mean_ap75 = np.mean(all_gop_ap75s)
            
            print(f"\n🏆 FINAL SUMMARY - Overall Performance Across All GOPs:")
            print(f"   📊 Total GOPs processed: {len(all_gop_maps)}")
            print(f"   📊 Total objects tracked: {total_objects_processed}")
            print(f"   🎯 Overall Mean mAP@[0.5:0.95]: {overall_mean_map:.3f}")
            print(f"   🎯 Overall Mean AP@0.5: {overall_mean_ap50:.3f}")
            print(f"   🎯 Overall Mean AP@0.75: {overall_mean_ap75:.3f}")
            
            # Overall performance assessment
            if overall_mean_map >= 0.7:
                overall_performance = "🟢 Excellent"
            elif overall_mean_map >= 0.5:
                overall_performance = "🟡 Good"
            elif overall_mean_map >= 0.3:
                overall_performance = "🟠 Fair"
            else:
                overall_performance = "🔴 Poor"
                
            print(f"   🏆 Overall Performance Level: {overall_performance}")
            
            # Performance breakdown by GOP
            print(f"\n📈 GOP-by-GOP Performance Breakdown:")
            for i, (map_val, ap50_val, ap75_val) in enumerate(zip(all_gop_maps, all_gop_ap50s, all_gop_ap75s)):
                print(f"   GOP {i}: mAP={map_val:.3f}, AP@0.5={ap50_val:.3f}, AP@0.75={ap75_val:.3f}")
            
            # Standard deviation to show consistency
            map_std = np.std(all_gop_maps)
            print(f"   📊 mAP Standard Deviation: {map_std:.3f} {'(Consistent)' if map_std < 0.1 else '(Variable)'}")
        else:
            print(f"\n⚠️  No valid mAP results obtained across GOPs")
        
        return gop_results

def main():
    """Main function."""
    print("🚀 Accumulated Motion Vector Predictor for All Fully Visible Objects")
    print("=" * 80)
    
    # Create predictor
    predictor = AccumulatedMotionPredictor(verbose=True)
    
    # Process multiple sequences for validation
    predictor.process_multiple_sequences(num_gops=3)
    
    return 0

if __name__ == "__main__":
    exit(main())
