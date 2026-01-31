#!/usr/bin/env python3
"""
Motion Vector Tracker with mAP Integration

This script modifies the existing motion vector enhanced tracker to include 
mAP and AP metrics evaluation comparing tracking results with ground truth.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import ndimage
from collections import defaultdict

# Try to import the existing tracker
try:
    from motion_vector_enhanced_multi_gop_tracker import MotionVectorEnhancedTracker
except ImportError:
    print("❌ Could not import MotionVectorEnhancedTracker. Make sure the file exists.")
    sys.exit(1)

class MotionVectorMAPTracker(MotionVectorEnhancedTracker):
    """Enhanced motion vector tracker with mAP evaluation capabilities."""
    
    def __init__(self, verbose=True, validation_mode=True):
        super().__init__(verbose=verbose, validation_mode=validation_mode)
        
        # mAP evaluation parameters
        self.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.confidence_threshold = 0.5
        
        # Tracking performance storage
        self.tracking_metrics = {}
        self.per_frame_metrics = {}
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes."""
        # Convert [center_x, center_y, width, height] to [x1, y1, x2, y2] if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume format is [x, y, width, height]
            x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
            x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        else:
            return 0.0
        
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
    
    def calculate_object_map(self, tracked_positions, ground_truth_positions):
        """Calculate mAP for a single object across all frames."""
        if not tracked_positions or not ground_truth_positions:
            return {'mAP': 0.0, 'AP_per_threshold': {}}
        
        # Extract bounding boxes
        predicted_boxes = []
        gt_boxes = []
        confidence_scores = []
        
        min_len = min(len(tracked_positions), len(ground_truth_positions))
        
        for i in range(min_len):
            # Get tracked bounding box
            tracked_pos = tracked_positions[i]['position']
            tracked_size = tracked_positions[i].get('adaptive_size', tracked_positions[i].get('original_size', [50, 50]))
            
            # Convert to [x, y, width, height]
            pred_box = [
                tracked_pos[0] - tracked_size[0]/2,
                tracked_pos[1] - tracked_size[1]/2,
                tracked_size[0],
                tracked_size[1]
            ]
            predicted_boxes.append(pred_box)
            
            # Get ground truth bounding box
            gt_pos = ground_truth_positions[i]['center']
            gt_size = ground_truth_positions[i]['size']
            
            gt_box = [
                gt_pos[0] - gt_size[0]/2,
                gt_pos[1] - gt_size[1]/2,
                gt_size[0],
                gt_size[1]
            ]
            gt_boxes.append(gt_box)
            
            # Generate confidence score based on motion consistency
            motion_vec = tracked_positions[i].get('motion_vector', np.array([0, 0]))
            motion_magnitude = np.linalg.norm(motion_vec)
            
            # Higher confidence for moderate motion
            if motion_magnitude < 0.5:
                confidence = 0.7 + 0.3 * (motion_magnitude / 0.5)
            elif motion_magnitude < 3.0:
                confidence = 1.0 - 0.2 * ((motion_magnitude - 0.5) / 2.5)
            else:
                confidence = max(0.4, 0.8 - 0.4 * ((motion_magnitude - 3.0) / 5.0))
            
            confidence_scores.append(confidence)
        
        # Calculate AP for each IoU threshold
        ap_per_threshold = {}
        
        for threshold in self.iou_thresholds:
            true_positives = 0
            
            for pred_box, gt_box, conf in zip(predicted_boxes, gt_boxes, confidence_scores):
                if conf >= self.confidence_threshold:
                    iou = self.calculate_iou(pred_box, gt_box)
                    if iou >= threshold:
                        true_positives += 1
            
            ap = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
            ap_per_threshold[threshold] = ap
        
        # Calculate mAP
        mAP = np.mean(list(ap_per_threshold.values()))
        
        return {
            'mAP': mAP,
            'AP_per_threshold': ap_per_threshold,
            'total_predictions': len(predicted_boxes),
            'total_ground_truths': len(gt_boxes),
            'mean_confidence': np.mean(confidence_scores)
        }
    
    def evaluate_tracking_performance(self, tracking_result):
        """Evaluate overall tracking performance using mAP metrics."""
        if not tracking_result:
            return {}
        
        tracked_positions = tracking_result['tracked_positions']
        all_gop_data = tracking_result['gop_data']
        
        # Collect ground truth data for all frames
        all_ground_truth = defaultdict(list)
        
        for gop_data in all_gop_data:
            gop_idx = gop_data['gop_idx']
            annotation_data = gop_data['annotation_data']
            
            if annotation_data and 'annotations' in annotation_data:
                for frame_idx, frame_annotations in enumerate(annotation_data['annotations']):
                    global_frame = gop_idx * 49 + frame_idx  # Assuming 49 frames per GOP
                    
                    for ann in frame_annotations:
                        if len(ann) >= 6:
                            obj_id = int(ann[0])
                            x_norm, y_norm, w_norm, h_norm = ann[2:6]
                            
                            # Convert to pixel coordinates
                            x_center = x_norm * 960
                            y_center = y_norm * 960
                            width = w_norm * 960
                            height = h_norm * 960
                            
                            all_ground_truth[obj_id].append({
                                'center': [x_center, y_center],
                                'size': [width, height],
                                'global_frame': global_frame
                            })
        
        # Calculate mAP for each tracked object
        object_metrics = {}
        
        for obj_id, positions in tracked_positions.items():
            if obj_id in all_ground_truth:
                # Align tracking results with ground truth by global frame
                aligned_gt = []
                aligned_tracking = []
                
                for track_data in positions:
                    global_frame = track_data['global_frame']
                    
                    # Find corresponding ground truth
                    gt_for_frame = None
                    for gt_data in all_ground_truth[obj_id]:
                        if gt_data['global_frame'] == global_frame:
                            gt_for_frame = gt_data
                            break
                    
                    if gt_for_frame:
                        aligned_tracking.append(track_data)
                        aligned_gt.append(gt_for_frame)
                
                # Calculate mAP for this object
                if aligned_tracking and aligned_gt:
                    metrics = self.calculate_object_map(aligned_tracking, aligned_gt)
                    object_metrics[obj_id] = metrics
                    
                    if self.verbose:
                        print(f"   🎯 Object {obj_id} mAP Analysis:")
                        print(f"      mAP@[0.5:0.95]: {metrics['mAP']:.3f}")
                        print(f"      AP@0.5: {metrics['AP_per_threshold'][0.5]:.3f}")
                        print(f"      AP@0.75: {metrics['AP_per_threshold'][0.75]:.3f}")
                        print(f"      Frames evaluated: {metrics['total_predictions']}")
        
        return object_metrics
    
    def create_map_evaluation_video(self, tracking_result, object_metrics, output_path="motion_vector_map_evaluation.mp4"):
        """Create video with mAP evaluation overlay."""
        if not tracking_result or not object_metrics:
            return False
        
        print(f"🎬 Creating mAP evaluation video: {output_path}")
        
        tracking_objects = tracking_result['tracking_objects']
        tracked_positions = tracking_result['tracked_positions']
        all_gop_data = tracking_result['gop_data']
        
        # Video setup
        height, width = 960, 1280  # Wider for metrics panel
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 8.0, (width, height))
        
        frame_count = 0
        
        # Process each GOP
        for gop_data in all_gop_data:
            motion_data = gop_data['motion_data']
            rgb_data = gop_data['rgb_data']
            gop_idx = gop_data['gop_idx']
            
            print(f"   📖 Processing GOP {gop_idx}...")
            
            for frame_idx in range(motion_data.shape[0]):
                # Get RGB frame
                rgb_frame_idx = min(frame_idx, rgb_data.shape[0] - 1)
                rgb_frame = rgb_data[rgb_frame_idx]
                
                # Create extended frame
                extended_frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Convert and place original frame
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                extended_frame[:, :960] = bgr_frame
                
                # Dark panel for metrics
                extended_frame[:, 960:] = (40, 40, 40)
                
                # Draw tracking results
                self.draw_tracking_with_map_overlay(extended_frame, tracking_objects, 
                                                  tracked_positions, frame_count, object_metrics)
                
                # Draw metrics panel
                self.draw_map_metrics_panel(extended_frame, object_metrics, frame_count, gop_idx, frame_idx)
                
                # Write frame
                out.write(extended_frame)
                frame_count += 1
                
                if frame_idx % 10 == 0:
                    print(f"   📍 Frame {frame_count}")
        
        out.release()
        print(f"✅ mAP evaluation video created: {output_path}")
        return True
    
    def draw_tracking_with_map_overlay(self, frame, tracking_objects, tracked_positions, frame_count, object_metrics):
        """Draw tracking results with mAP information overlay."""
        for obj in tracking_objects:
            obj_id = obj['id']
            color = self.get_bgr_color(obj['color'])
            
            if obj_id not in tracked_positions:
                continue
            
            # Get current tracking data
            current_positions = [pos for pos in tracked_positions[obj_id] if pos['global_frame'] <= frame_count]
            
            if not current_positions:
                continue
            
            current_track = current_positions[-1]
            current_pos = current_track['position']
            current_size = current_track.get('adaptive_size', current_track.get('original_size', [50, 50]))
            
            # Draw bounding box
            bbox_x = current_pos[0] - current_size[0]/2
            bbox_y = current_pos[1] - current_size[1]/2
            
            cv2.rectangle(frame, 
                         (int(bbox_x), int(bbox_y)), 
                         (int(bbox_x + current_size[0]), int(bbox_y + current_size[1])), 
                         color, 3)
            
            # Draw ground truth if available (validation mode)
            if 'ground_truth_position' in current_track:
                gt_pos = current_track['ground_truth_position']
                gt_size = current_track['ground_truth_size']
                gt_bbox_x = gt_pos[0] - gt_size[0]/2
                gt_bbox_y = gt_pos[1] - gt_size[1]/2
                
                # Draw dashed ground truth box
                self.draw_dashed_rectangle(frame, 
                                         (int(gt_bbox_x), int(gt_bbox_y)), 
                                         (int(gt_bbox_x + gt_size[0]), int(gt_bbox_y + gt_size[1])), 
                                         color, 2)
            
            # Get mAP metrics for this object
            if obj_id in object_metrics:
                metrics = object_metrics[obj_id]
                map_score = metrics['mAP']
                ap50_score = metrics['AP_per_threshold'][0.5]
                
                # Draw mAP info
                info_text = f'Obj {obj_id} | mAP: {map_score:.3f}'
                cv2.putText(frame, info_text, 
                          (int(bbox_x), int(bbox_y) - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                ap50_text = f'AP@0.5: {ap50_score:.3f}'
                cv2.putText(frame, ap50_text, 
                          (int(bbox_x), int(bbox_y + current_size[1] + 20)), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_map_metrics_panel(self, frame, object_metrics, frame_count, gop_idx, frame_idx):
        """Draw comprehensive mAP metrics panel."""
        panel_x = 970
        panel_y = 30
        line_height = 25
        
        # Panel title
        cv2.putText(frame, 'mAP Evaluation Metrics', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panel_y += line_height * 1.5
        
        # Frame info
        cv2.putText(frame, f'Frame: {frame_count}', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panel_y += line_height * 0.8
        
        cv2.putText(frame, f'GOP: {gop_idx} | Local: {frame_idx}', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panel_y += line_height * 1.5
        
        # Overall metrics
        if object_metrics:
            all_maps = [m['mAP'] for m in object_metrics.values()]
            all_ap50s = [m['AP_per_threshold'][0.5] for m in object_metrics.values()]
            all_ap75s = [m['AP_per_threshold'][0.75] for m in object_metrics.values()]
            
            avg_map = np.mean(all_maps)
            avg_ap50 = np.mean(all_ap50s)
            avg_ap75 = np.mean(all_ap75s)
            
            cv2.putText(frame, 'Overall Performance:', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            panel_y += line_height
            
            cv2.putText(frame, f'Mean mAP: {avg_map:.3f}', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panel_y += line_height * 0.8
            
            cv2.putText(frame, f'Mean AP@0.5: {avg_ap50:.3f}', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panel_y += line_height * 0.8
            
            cv2.putText(frame, f'Mean AP@0.75: {avg_ap75:.3f}', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            panel_y += line_height * 1.5
            
            # Per-object metrics
            for obj_id, metrics in object_metrics.items():
                cv2.putText(frame, f'Object {obj_id}:', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                panel_y += line_height * 0.9
                
                cv2.putText(frame, f'  mAP: {metrics["mAP"]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 0.7
                
                cv2.putText(frame, f'  AP@0.5: {metrics["AP_per_threshold"][0.5]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 0.7
                
                cv2.putText(frame, f'  Confidence: {metrics["mean_confidence"]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 1.2
        
        # Legend
        cv2.putText(frame, 'Legend:', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panel_y += line_height
        
        cv2.putText(frame, 'Solid: Tracked', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        panel_y += line_height * 0.8
        
        cv2.putText(frame, 'Dashed: Ground Truth', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def draw_dashed_rectangle(self, frame, pt1, pt2, color, thickness):
        """Draw a dashed rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        dash_length = 5
        
        # Top and bottom edges
        for x in range(x1, x2, dash_length * 2):
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left and right edges
        for y in range(y1, y2, dash_length * 2):
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def get_bgr_color(self, color_name):
        """Convert color name to BGR tuple."""
        color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'yellow': (0, 255, 255)
        }
        return color_map.get(color_name, (255, 255, 255))
    
    def run_map_evaluation(self, num_gops=3):
        """Run complete motion vector tracking with mAP evaluation."""
        print(f"🚀 Motion Vector Tracking with mAP Evaluation")
        print(f"=" * 60)
        
        # Load sequence
        if not self.load_sequence():
            return 1
        
        # Track objects
        print(f"\\n🎯 Starting motion vector tracking with mAP evaluation...")
        tracking_result = self.track_objects_with_motion_field(num_gops=num_gops)
        
        if not tracking_result:
            print(f"❌ Tracking failed")
            return 1
        
        # Evaluate performance
        print(f"\\n📊 Evaluating tracking performance with mAP metrics...")
        object_metrics = self.evaluate_tracking_performance(tracking_result)
        
        if not object_metrics:
            print(f"❌ mAP evaluation failed")
            return 1
        
        # Print summary
        self.print_map_summary(object_metrics)
        
        # Create evaluation video
        success = self.create_map_evaluation_video(
            tracking_result, object_metrics, 
            output_path="motion_vector_map_evaluation.mp4"
        )
        
        if success:
            print(f"\\n✅ Motion vector mAP evaluation completed!")
            print(f"   📹 Video: motion_vector_map_evaluation.mp4")
            print(f"   🎯 Features:")
            print(f"     • Motion vector field overlay")
            print(f"     • Real-time mAP and AP@0.5 scores")
            print(f"     • Tracked vs ground truth comparison")
            print(f"     • Comprehensive metrics panel")
            print(f"     • Frame-by-frame performance analysis")
            print(f"   ⭐ Use: ffplay motion_vector_map_evaluation.mp4")
            return 0
        else:
            print(f"❌ Failed to create evaluation video")
            return 1
    
    def print_map_summary(self, object_metrics):
        """Print comprehensive mAP evaluation summary."""
        print(f"\\n📊 mAP EVALUATION SUMMARY")
        print(f"=" * 50)
        
        if not object_metrics:
            print("❌ No metrics available")
            return
        
        # Per-object results
        print(f"\\n🎯 Per-Object Performance:")
        print(f"{'Object':<8} {'mAP':<8} {'AP@0.5':<8} {'AP@0.75':<8} {'Frames':<8}")
        print("-" * 45)
        
        all_maps = []
        all_ap50s = []
        all_ap75s = []
        
        for obj_id, metrics in sorted(object_metrics.items()):
            map_score = metrics['mAP']
            ap50_score = metrics['AP_per_threshold'][0.5]
            ap75_score = metrics['AP_per_threshold'][0.75]
            num_frames = metrics['total_predictions']
            
            all_maps.append(map_score)
            all_ap50s.append(ap50_score)
            all_ap75s.append(ap75_score)
            
            print(f"{obj_id:<8} {map_score:<8.3f} {ap50_score:<8.3f} {ap75_score:<8.3f} {num_frames:<8}")
        
        # Overall statistics
        print(f"\\n🏆 Overall Performance:")
        overall_map = np.mean(all_maps)
        overall_ap50 = np.mean(all_ap50s)
        overall_ap75 = np.mean(all_ap75s)
        map_std = np.std(all_maps)
        
        print(f"   Mean mAP@[0.5:0.95]: {overall_map:.3f} ± {map_std:.3f}")
        print(f"   Mean AP@0.5: {overall_ap50:.3f}")
        print(f"   Mean AP@0.75: {overall_ap75:.3f}")
        print(f"   Objects evaluated: {len(object_metrics)}")
        
        # Performance assessment
        if overall_map >= 0.7:
            performance = "🟢 Excellent"
        elif overall_map >= 0.5:
            performance = "🟡 Good"
        elif overall_map >= 0.3:
            performance = "🟠 Fair"
        else:
            performance = "🔴 Poor"
        
        print(f"   Performance level: {performance}")
        
        # Best and worst objects
        best_obj = max(object_metrics.keys(), key=lambda x: object_metrics[x]['mAP'])
        worst_obj = min(object_metrics.keys(), key=lambda x: object_metrics[x]['mAP'])
        
        print(f"\\n💡 Insights:")
        print(f"   🏆 Best performer: Object {best_obj} (mAP: {object_metrics[best_obj]['mAP']:.3f})")
        print(f"   ⚠️  Most challenging: Object {worst_obj} (mAP: {object_metrics[worst_obj]['mAP']:.3f})")
        print(f"   📊 Consistency: {'High' if map_std < 0.1 else 'Variable'} (std: {map_std:.3f})")


def main():
    """Main function for motion vector mAP evaluation."""
    print("🚀 Motion Vector Tracker with mAP Integration")
    print("=" * 80)
    
    # Create mAP-enabled tracker
    tracker = MotionVectorMAPTracker(verbose=True, validation_mode=True)
    
    # Run evaluation
    result = tracker.run_map_evaluation(num_gops=3)
    
    return result

if __name__ == "__main__":
    exit(main())
