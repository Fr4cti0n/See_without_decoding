#!/usr/bin/env python3
"""
Motion Vector mAP Evaluator

This script evaluates motion vector tracking performance by calculating mAP and AP metrics
comparing tracked bounding boxes with initial/ground truth bounding boxes.
Integrates with existing motion vector tracking scripts.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
from scipy import ndimage
from collections import defaultdict

# Add path for data loader
sys.path.append('/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/utils/mots_dataset/visualization_toolkit')

from core.data_loader import MOTSDataLoaderFactory

class MotionVectorMAPEvaluator:
    """Evaluates motion vector tracking performance using mAP and AP metrics."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.factory = MOTSDataLoaderFactory(verbose=verbose)
        self.data_loader = None
        
        # Motion processing parameters
        self.motion_smoothing_sigma = 1.0
        self.motion_scale_factor = 2.0  # Motion amplification
        self.motion_threshold = 0.1  # Minimum motion to consider
        
        # Evaluation parameters
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.confidence_threshold = 0.5  # For detection filtering
        
        # Tracking state
        self.motion_history = {}  # Motion history for consistency
        self.history_length = 5
        
        # Color scheme for visualization
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
    
    def smooth_motion_field(self, motion_field):
        """Apply Gaussian smoothing to motion field."""
        smoothed = np.zeros_like(motion_field)
        smoothed[:, :, 0] = ndimage.gaussian_filter(motion_field[:, :, 0], self.motion_smoothing_sigma)
        smoothed[:, :, 1] = ndimage.gaussian_filter(motion_field[:, :, 1], self.motion_smoothing_sigma)
        return smoothed
    
    def extract_initial_objects(self, annotations):
        """Extract initial objects from first frame annotations."""
        initial_objects = {}
        
        for i, ann in enumerate(annotations):
            if len(ann) >= 6:
                obj_id, obj_class, x_norm, y_norm, w_norm, h_norm = ann[:6]
                obj_id = int(obj_id)
                
                # Convert normalized coordinates to pixel coordinates
                x_center = x_norm * 960
                y_center = y_norm * 960
                width = w_norm * 960
                height = h_norm * 960
                
                # Store initial bounding box
                initial_objects[obj_id] = {
                    'id': obj_id,
                    'class': obj_class,
                    'initial_center': [x_center, y_center],
                    'initial_size': [width, height],
                    'initial_bbox': [x_center - width/2, y_center - height/2, width, height],
                    'color': self.base_colors[i % len(self.base_colors)]
                }
        
        return initial_objects
    
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
    
    def extract_enhanced_motion(self, motion_field, current_pos, current_size, obj_id):
        """Extract enhanced motion using multi-macroblock sampling and temporal consistency."""
        center_x, center_y = current_pos
        bbox_width, bbox_height = current_size
        
        # Convert to macroblock coordinates
        center_mb_x = int(center_x // 16)
        center_mb_y = int(center_y // 16)
        
        # Calculate bounding box in macroblock space
        bbox_mb_left = max(0, int((center_x - bbox_width/2) // 16))
        bbox_mb_right = min(motion_field.shape[1] - 1, int((center_x + bbox_width/2) // 16))
        bbox_mb_top = max(0, int((center_y - bbox_height/2) // 16))
        bbox_mb_bottom = min(motion_field.shape[0] - 1, int((center_y + bbox_height/2) // 16))
        
        # Multi-macroblock sampling
        internal_motions = []
        edge_motions = []
        
        for mb_y in range(bbox_mb_top, bbox_mb_bottom + 1):
            for mb_x in range(bbox_mb_left, bbox_mb_right + 1):
                motion_vec = motion_field[mb_y, mb_x]
                motion_magnitude = np.linalg.norm(motion_vec)
                
                # Prioritize edge motion
                is_edge = (mb_x == bbox_mb_left or mb_x == bbox_mb_right or 
                          mb_y == bbox_mb_top or mb_y == bbox_mb_bottom)
                
                if motion_magnitude > self.motion_threshold:
                    if is_edge:
                        edge_motions.append(motion_vec)
                    else:
                        internal_motions.append(motion_vec)
        
        # Calculate weighted motion
        if edge_motions:
            edge_motion = np.mean(edge_motions, axis=0)
            if internal_motions:
                internal_motion = np.mean(internal_motions, axis=0)
                combined_motion = 0.7 * edge_motion + 0.3 * internal_motion
            else:
                combined_motion = edge_motion
        elif internal_motions:
            combined_motion = np.mean(internal_motions, axis=0)
        else:
            combined_motion = np.array([0.0, 0.0])
        
        # Temporal consistency using motion history
        if obj_id not in self.motion_history:
            self.motion_history[obj_id] = []
        
        self.motion_history[obj_id].append(combined_motion.copy())
        
        # Keep only recent history
        if len(self.motion_history[obj_id]) > self.history_length:
            self.motion_history[obj_id].pop(0)
        
        # Apply temporal smoothing
        if len(self.motion_history[obj_id]) >= 2:
            weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5][-len(self.motion_history[obj_id]):])
            weights = weights / weights.sum()
            smoothed_motion = np.average(self.motion_history[obj_id], axis=0, weights=weights)
            
            # Blend current with history
            final_motion = 0.8 * combined_motion + 0.2 * smoothed_motion
        else:
            final_motion = combined_motion
        
        # Apply motion scaling
        motion_magnitude = np.linalg.norm(final_motion)
        if motion_magnitude > self.motion_threshold:
            if motion_magnitude < 1.0:
                scale_factor = self.motion_scale_factor * 2.0  # 4x for small motion
            elif motion_magnitude < 3.0:
                scale_factor = self.motion_scale_factor  # 2x for medium motion
            else:
                scale_factor = max(1.2, self.motion_scale_factor * 0.6)  # 1.2x for large motion
            
            final_motion = final_motion * scale_factor
        
        return final_motion
    
    def track_objects_through_gop(self, gop_idx, initial_objects):
        """Track objects through a single GOP using motion vectors."""
        # Load GOP data
        motion_data = self.data_loader.load_motion_vectors(gop_idx)
        rgb_data = self.data_loader.load_rgb_frames(gop_idx, 'pframe')
        annotation_data = self.data_loader.load_corrected_annotations(gop_idx)
        
        if motion_data is None or rgb_data is None:
            print(f"❌ Failed to load GOP {gop_idx} data")
            return None
        
        num_frames = motion_data.shape[0]
        
        # Initialize tracking state
        tracked_objects = {}
        for obj_id, obj_data in initial_objects.items():
            tracked_objects[obj_id] = {
                'positions': [obj_data['initial_center'].copy()],
                'sizes': [obj_data['initial_size'].copy()],
                'bounding_boxes': [obj_data['initial_bbox'].copy()],
                'motion_vectors': [],
                'confidence_scores': [1.0]  # Start with full confidence
            }
        
        # Track through each frame
        for frame_idx in range(num_frames):
            # Get motion field
            motion_layer0 = motion_data[frame_idx, 0]
            motion_layer1 = motion_data[frame_idx, 1]
            
            # Smooth and combine layers
            smooth_layer0 = self.smooth_motion_field(motion_layer0)
            smooth_layer1 = self.smooth_motion_field(motion_layer1)
            combined_motion = 0.6 * smooth_layer0 + 0.4 * smooth_layer1
            
            # Track each object
            for obj_id in tracked_objects.keys():
                # Get current position and size
                current_pos = tracked_objects[obj_id]['positions'][-1]
                current_size = tracked_objects[obj_id]['sizes'][-1]
                
                # Extract enhanced motion
                motion_vec = self.extract_enhanced_motion(
                    combined_motion, current_pos, current_size, obj_id
                )
                
                # Calculate new position
                new_pos = [
                    current_pos[0] + motion_vec[0],
                    current_pos[1] + motion_vec[1]
                ]
                
                # Clamp to frame bounds
                new_pos[0] = np.clip(new_pos[0], 0, 959)
                new_pos[1] = np.clip(new_pos[1], 0, 959)
                
                # Keep same size for now (could add size adaptation)
                new_size = current_size.copy()
                
                # Calculate new bounding box
                new_bbox = [
                    new_pos[0] - new_size[0]/2,
                    new_pos[1] - new_size[1]/2,
                    new_size[0],
                    new_size[1]
                ]
                
                # Calculate confidence based on motion magnitude and consistency
                motion_magnitude = np.linalg.norm(motion_vec)
                confidence = min(1.0, max(0.1, 1.0 - motion_magnitude / 10.0))
                
                # Store tracking results
                tracked_objects[obj_id]['positions'].append(new_pos)
                tracked_objects[obj_id]['sizes'].append(new_size)
                tracked_objects[obj_id]['bounding_boxes'].append(new_bbox)
                tracked_objects[obj_id]['motion_vectors'].append(motion_vec.copy())
                tracked_objects[obj_id]['confidence_scores'].append(confidence)
        
        # Load ground truth annotations if available
        ground_truth_boxes = {}
        if annotation_data and 'annotations' in annotation_data:
            for frame_idx in range(min(num_frames, len(annotation_data['annotations']))):
                frame_annotations = annotation_data['annotations'][frame_idx]
                for ann in frame_annotations:
                    if len(ann) >= 6:
                        ann_obj_id = int(ann[0])
                        if ann_obj_id in initial_objects:
                            if ann_obj_id not in ground_truth_boxes:
                                ground_truth_boxes[ann_obj_id] = []
                            
                            # Convert to pixel coordinates
                            x_norm, y_norm, w_norm, h_norm = ann[2:6]
                            x_center = x_norm * 960
                            y_center = y_norm * 960
                            width = w_norm * 960
                            height = h_norm * 960
                            
                            gt_bbox = [
                                x_center - width/2,
                                y_center - height/2,
                                width,
                                height
                            ]
                            ground_truth_boxes[ann_obj_id].append(gt_bbox)
        
        return {
            'gop_idx': gop_idx,
            'tracked_objects': tracked_objects,
            'ground_truth_boxes': ground_truth_boxes,
            'rgb_data': rgb_data,
            'num_frames': num_frames
        }
    
    def calculate_map_metrics(self, predicted_boxes, ground_truth_boxes, confidence_scores=None):
        """Calculate mAP and AP metrics for tracking evaluation."""
        if not predicted_boxes or not ground_truth_boxes:
            return {'mAP': 0.0, 'AP_per_threshold': {}}
        
        # Ensure same number of predictions and ground truths
        min_len = min(len(predicted_boxes), len(ground_truth_boxes))
        predicted_boxes = predicted_boxes[:min_len]
        ground_truth_boxes = ground_truth_boxes[:min_len]
        
        if confidence_scores:
            confidence_scores = confidence_scores[:min_len]
        else:
            confidence_scores = [1.0] * len(predicted_boxes)
        
        # Calculate AP for each IoU threshold
        ap_per_threshold = {}
        
        for threshold in self.iou_thresholds:
            # Count true positives at this threshold
            true_positives = 0
            
            for pred_box, gt_box, conf in zip(predicted_boxes, ground_truth_boxes, confidence_scores):
                if conf >= self.confidence_threshold:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou >= threshold:
                        true_positives += 1
            
            # Calculate Average Precision
            ap = true_positives / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0.0
            ap_per_threshold[threshold] = ap
        
        # Calculate mAP (mean over all thresholds)
        mAP = np.mean(list(ap_per_threshold.values()))
        
        return {
            'mAP': mAP,
            'AP_per_threshold': ap_per_threshold,
            'total_predictions': len(predicted_boxes),
            'total_ground_truths': len(ground_truth_boxes),
            'mean_confidence': np.mean(confidence_scores)
        }
    
    def evaluate_gop_tracking(self, gop_result):
        """Evaluate tracking performance for a single GOP."""
        tracked_objects = gop_result['tracked_objects']
        ground_truth_boxes = gop_result['ground_truth_boxes']
        gop_idx = gop_result['gop_idx']
        
        evaluation_results = {}
        
        for obj_id in tracked_objects.keys():
            tracked_boxes = tracked_objects[obj_id]['bounding_boxes']
            confidence_scores = tracked_objects[obj_id]['confidence_scores']
            
            if obj_id in ground_truth_boxes:
                gt_boxes = ground_truth_boxes[obj_id]
            else:
                # If no ground truth, use initial box as reference
                gt_boxes = [tracked_boxes[0]] * len(tracked_boxes)
            
            # Calculate mAP metrics
            metrics = self.calculate_map_metrics(tracked_boxes, gt_boxes, confidence_scores)
            
            evaluation_results[obj_id] = metrics
            
            if self.verbose:
                print(f"   🎯 Object {obj_id} (GOP {gop_idx}):")
                print(f"      mAP@[0.5:0.95]: {metrics['mAP']:.3f}")
                print(f"      AP@0.5: {metrics['AP_per_threshold'][0.5]:.3f}")
                print(f"      AP@0.75: {metrics['AP_per_threshold'][0.75]:.3f}")
                print(f"      Mean confidence: {metrics['mean_confidence']:.3f}")
                print(f"      Predictions/GT: {metrics['total_predictions']}/{metrics['total_ground_truths']}")
        
        return evaluation_results
    
    def visualize_tracking_with_map(self, gop_result, evaluation_results, sequence_prefix="", save_frames=True):
        """Create visualization showing tracking results with mAP metrics."""
        tracked_objects = gop_result['tracked_objects']
        ground_truth_boxes = gop_result['ground_truth_boxes']
        rgb_data = gop_result['rgb_data']
        gop_idx = gop_result['gop_idx']
        num_frames = gop_result['num_frames']
        
        # Create video writer
        if sequence_prefix:
            output_path = f'{sequence_prefix}_motion_tracking_map_gop{gop_idx}.mp4'
        else:
            output_path = f'motion_tracking_map_gop{gop_idx}.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 8
        frame_size = (960, 960)
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        print(f"🎬 Creating tracking visualization with mAP: {output_path}")
        
        for frame_idx in range(num_frames):
            # Get RGB frame
            rgb_frame_idx = min(frame_idx, rgb_data.shape[0] - 1)
            rgb_frame = rgb_data[rgb_frame_idx].copy()
            
            # Convert to BGR for OpenCV
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw tracking results for each object
            for obj_id, obj_data in tracked_objects.items():
                if frame_idx < len(obj_data['bounding_boxes']):
                    # Get tracked bounding box
                    bbox = obj_data['bounding_boxes'][frame_idx]
                    confidence = obj_data['confidence_scores'][frame_idx]
                    x, y, w, h = bbox
                    
                    # Get object color
                    color = (0, 255, 0)  # Default green
                    if hasattr(self, 'initial_objects') and obj_id in self.initial_objects:
                        color = self.initial_objects[obj_id]['color']
                    
                    # Draw tracked bounding box (solid line)
                    cv2.rectangle(bgr_frame, 
                                (int(x), int(y)), 
                                (int(x + w), int(y + h)), 
                                color, 3)
                    
                    # Draw ground truth bounding box if available (dashed style)
                    if obj_id in ground_truth_boxes and frame_idx < len(ground_truth_boxes[obj_id]):
                        gt_bbox = ground_truth_boxes[obj_id][frame_idx]
                        gt_x, gt_y, gt_w, gt_h = gt_bbox
                        
                        # Draw dashed rectangle for ground truth
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
                    
                    # Get mAP metrics for this object
                    obj_metrics = evaluation_results.get(obj_id, {})
                    map_score = obj_metrics.get('mAP', 0.0)
                    ap50_score = obj_metrics.get('AP_per_threshold', {}).get(0.5, 0.0)
                    
                    # Draw object label with mAP info
                    label_text = f'Obj {obj_id} | mAP: {map_score:.3f} | AP@0.5: {ap50_score:.3f}'
                    cv2.putText(bgr_frame, label_text, 
                              (int(x), int(y) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Draw confidence score
                    conf_text = f'Conf: {confidence:.3f}'
                    cv2.putText(bgr_frame, conf_text,
                              (int(x), int(y + h + 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw frame info
            cv2.putText(bgr_frame, f'Frame {frame_idx} - GOP {gop_idx}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(bgr_frame, f'Motion Vector Tracking with mAP Evaluation', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, f'Solid: Tracked | Dashed: Ground Truth', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame
            out.write(bgr_frame)
            
            if frame_idx % 10 == 0:
                print(f"   📍 Processing frame {frame_idx}/{num_frames}")
        
        out.release()
        print(f"✅ Tracking visualization with mAP created: {output_path}")
        
        return output_path
    
    def process_sequence_evaluation(self, sequence_name=None, num_gops=3):
        """Process a sequence and evaluate tracking performance across multiple GOPs."""
        # Load sequence
        success, _ = self.load_sequence(sequence_name)
        if not success:
            return None
        
        print(f"🎯 Evaluating motion vector tracking performance...")
        
        # Get initial objects from first GOP's first frame
        annotation_data_gop0 = self.data_loader.load_corrected_annotations(0)
        if not annotation_data_gop0 or not annotation_data_gop0['annotations']:
            print("❌ No annotations found in first GOP")
            return None
        
        first_frame_annotations = annotation_data_gop0['annotations'][0]
        initial_objects = self.extract_initial_objects(first_frame_annotations)
        self.initial_objects = initial_objects  # Store for color access
        
        if not initial_objects:
            print("❌ No initial objects found")
            return None
        
        print(f"📦 Found {len(initial_objects)} initial objects:")
        for obj_id, obj_data in initial_objects.items():
            center = obj_data['initial_center']
            size = obj_data['initial_size']
            print(f"   🎯 Object {obj_id}: center=({center[0]:.1f},{center[1]:.1f}), size=({size[0]:.1f}x{size[1]:.1f})")
        
        # Track objects through multiple GOPs
        all_gop_results = []
        overall_metrics = defaultdict(list)
        
        for gop_idx in range(num_gops):
            print(f"\n📖 Processing GOP {gop_idx}...")
            
            # Track objects through this GOP
            gop_result = self.track_objects_through_gop(gop_idx, initial_objects)
            if not gop_result:
                print(f"⚠️  Failed to track GOP {gop_idx}")
                continue
            
            # Evaluate tracking performance
            evaluation_results = self.evaluate_gop_tracking(gop_result)
            
            # Store results
            gop_result['evaluation_results'] = evaluation_results
            all_gop_results.append(gop_result)
            
            # Accumulate metrics for overall statistics
            for obj_id, metrics in evaluation_results.items():
                overall_metrics[obj_id].append(metrics)
            
            # Create visualization
            video_path = self.visualize_tracking_with_map(
                gop_result, evaluation_results, 
                sequence_prefix=f"seq{gop_idx+1}"
            )
            
            # Print GOP summary
            self.print_gop_summary(gop_idx, evaluation_results)
        
        # Print overall summary
        self.print_overall_summary(overall_metrics, initial_objects)
        
        return {
            'initial_objects': initial_objects,
            'gop_results': all_gop_results,
            'overall_metrics': overall_metrics
        }
    
    def print_gop_summary(self, gop_idx, evaluation_results):
        """Print summary for a single GOP."""
        if not evaluation_results:
            return
        
        print(f"\n📊 GOP {gop_idx} Performance Summary:")
        
        all_maps = []
        all_ap50s = []
        all_ap75s = []
        
        for obj_id, metrics in evaluation_results.items():
            map_score = metrics['mAP']
            ap50_score = metrics['AP_per_threshold'][0.5]
            ap75_score = metrics['AP_per_threshold'][0.75]
            
            all_maps.append(map_score)
            all_ap50s.append(ap50_score)
            all_ap75s.append(ap75_score)
            
            print(f"   🎯 Object {obj_id}: mAP={map_score:.3f}, AP@0.5={ap50_score:.3f}, AP@0.75={ap75_score:.3f}")
        
        # GOP averages
        if all_maps:
            avg_map = np.mean(all_maps)
            avg_ap50 = np.mean(all_ap50s)
            avg_ap75 = np.mean(all_ap75s)
            
            print(f"   📈 GOP {gop_idx} Averages:")
            print(f"      Mean mAP: {avg_map:.3f}")
            print(f"      Mean AP@0.5: {avg_ap50:.3f}")
            print(f"      Mean AP@0.75: {avg_ap75:.3f}")
            
            # Performance assessment
            if avg_map >= 0.7:
                performance = "🟢 Excellent"
            elif avg_map >= 0.5:
                performance = "🟡 Good"
            elif avg_map >= 0.3:
                performance = "🟠 Fair"
            else:
                performance = "🔴 Poor"
            
            print(f"      Performance: {performance}")
    
    def print_overall_summary(self, overall_metrics, initial_objects):
        """Print overall performance summary across all GOPs."""
        print(f"\n{'='*80}")
        print(f"🏆 OVERALL MOTION VECTOR TRACKING PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        if not overall_metrics:
            print("❌ No metrics available for summary")
            return
        
        # Calculate per-object statistics
        print(f"\n📊 Per-Object Performance Across All GOPs:")
        print(f"{'Object':<8} {'Avg mAP':<10} {'Avg AP@0.5':<12} {'Avg AP@0.75':<12} {'GOPs':<5}")
        print("-" * 50)
        
        all_avg_maps = []
        all_avg_ap50s = []
        all_avg_ap75s = []
        
        for obj_id in sorted(overall_metrics.keys()):
            obj_metrics = overall_metrics[obj_id]
            
            maps = [m['mAP'] for m in obj_metrics]
            ap50s = [m['AP_per_threshold'][0.5] for m in obj_metrics]
            ap75s = [m['AP_per_threshold'][0.75] for m in obj_metrics]
            
            avg_map = np.mean(maps)
            avg_ap50 = np.mean(ap50s)
            avg_ap75 = np.mean(ap75s)
            num_gops = len(obj_metrics)
            
            all_avg_maps.append(avg_map)
            all_avg_ap50s.append(avg_ap50)
            all_avg_ap75s.append(avg_ap75)
            
            print(f"{obj_id:<8} {avg_map:<10.3f} {avg_ap50:<12.3f} {avg_ap75:<12.3f} {num_gops:<5}")
        
        # Overall statistics
        if all_avg_maps:
            overall_map = np.mean(all_avg_maps)
            overall_ap50 = np.mean(all_avg_ap50s)
            overall_ap75 = np.mean(all_avg_ap75s)
            map_std = np.std(all_avg_maps)
            
            print(f"\n🎯 FINAL METRICS:")
            print(f"   Overall mAP@[0.5:0.95]: {overall_map:.3f} ± {map_std:.3f}")
            print(f"   Overall AP@0.5: {overall_ap50:.3f}")
            print(f"   Overall AP@0.75: {overall_ap75:.3f}")
            print(f"   Total objects tracked: {len(overall_metrics)}")
            
            # Final performance assessment
            if overall_map >= 0.7:
                final_performance = "🟢 Excellent - Motion vector tracking works very well!"
            elif overall_map >= 0.5:
                final_performance = "🟡 Good - Motion vector tracking performs well with some room for improvement"
            elif overall_map >= 0.3:
                final_performance = "🟠 Fair - Motion vector tracking shows moderate performance"
            else:
                final_performance = "🔴 Poor - Motion vector tracking needs significant improvement"
            
            print(f"\n🏆 FINAL ASSESSMENT: {final_performance}")
            
            # Insights
            print(f"\n💡 INSIGHTS:")
            print(f"   📊 Tracking consistency: {'High' if map_std < 0.1 else 'Variable'} (std: {map_std:.3f})")
            
            best_obj = max(overall_metrics.keys(), key=lambda x: np.mean([m['mAP'] for m in overall_metrics[x]]))
            worst_obj = min(overall_metrics.keys(), key=lambda x: np.mean([m['mAP'] for m in overall_metrics[x]]))
            
            print(f"   🏆 Best performing object: {best_obj} (mAP: {np.mean([m['mAP'] for m in overall_metrics[best_obj]]):.3f})")
            print(f"   ⚠️  Most challenging object: {worst_obj} (mAP: {np.mean([m['mAP'] for m in overall_metrics[worst_obj]]):.3f})")
            
            print(f"\n📹 Generated videos: seq*_motion_tracking_map_gop*.mp4")
            print(f"🎯 Each video shows tracked vs ground truth boxes with real-time mAP scores")
            print(f"⭐ Use: ffplay seq1_motion_tracking_map_gop0.mp4 to view")


def main():
    """Main function for motion vector mAP evaluation."""
    print("🚀 Motion Vector mAP Evaluator")
    print("=" * 80)
    
    # Create evaluator
    evaluator = MotionVectorMAPEvaluator(verbose=True)
    
    # Process sequence evaluation
    results = evaluator.process_sequence_evaluation(num_gops=3)
    
    if results:
        print(f"\n✅ Motion vector tracking evaluation completed!")
        print(f"   📊 Evaluated {len(results['initial_objects'])} objects across {len(results['gop_results'])} GOPs")
        print(f"   📹 Generated tracking videos with mAP metrics")
        print(f"   🎯 Solid boxes = tracked, dashed = ground truth")
        print(f"   📈 Real-time mAP and AP scores displayed")
    else:
        print(f"\n❌ Motion vector tracking evaluation failed")
    
    return 0

if __name__ == "__main__":
    exit(main())
