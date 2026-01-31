#!/usr/bin/env python3
"""
Enhanced Motion Vector Tracker with mAP Evaluation

This script enhances the existing motion vector tracking by adding comprehensive
mAP and AP metrics evaluation, comparing tracking results with initial/ground truth bounding boxes.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
from collections import defaultdict

# Import existing motion vector tracker
try:
    from accumulated_motion_predictor import AccumulatedMotionPredictor
except ImportError:
    print("❌ Could not import AccumulatedMotionPredictor. Make sure the file exists.")
    sys.exit(1)

class EnhancedMotionVectorEvaluator(AccumulatedMotionPredictor):
    """Enhanced motion vector tracker with comprehensive mAP evaluation."""
    
    def __init__(self, verbose=True):
        super().__init__(verbose=verbose)
        
        # Additional evaluation parameters
        self.confidence_threshold = 0.5
        self.tracking_modes = ['standard', 'enhanced', 'predictive']
        
        # Performance tracking
        self.frame_by_frame_metrics = {}
        self.object_consistency_scores = {}
        
    def calculate_precision_recall_curve(self, predicted_boxes, ground_truth_boxes, confidence_scores, iou_threshold=0.5):
        """Calculate precision-recall curve for a specific IoU threshold."""
        if not predicted_boxes or not ground_truth_boxes:
            return [], [], 0.0
        
        # Create detection list with confidence scores
        detections = []
        for i, (pred_box, conf) in enumerate(zip(predicted_boxes, confidence_scores)):
            detections.append({
                'box': pred_box,
                'confidence': conf,
                'frame_idx': i
            })
        
        # Sort by confidence (descending)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision and recall at different confidence thresholds
        precisions = []
        recalls = []
        
        for conf_thresh in np.arange(0.1, 1.0, 0.05):
            true_positives = 0
            false_positives = 0
            
            for det in detections:
                if det['confidence'] >= conf_thresh:
                    frame_idx = det['frame_idx']
                    if frame_idx < len(ground_truth_boxes):
                        iou = self.calculate_iou(det['box'], ground_truth_boxes[frame_idx])
                        if iou >= iou_threshold:
                            true_positives += 1
                        else:
                            false_positives += 1
            
            # Calculate precision and recall
            total_predictions = true_positives + false_positives
            precision = true_positives / total_predictions if total_predictions > 0 else 0.0
            recall = true_positives / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate Average Precision (area under PR curve)
        if len(recalls) > 1:
            # Use trapezoidal integration
            ap = np.trapz(precisions, recalls)
        else:
            ap = 0.0
        
        return precisions, recalls, ap
    
    def calculate_frame_by_frame_metrics(self, predicted_boxes, ground_truth_boxes, object_id):
        """Calculate detailed frame-by-frame metrics."""
        frame_metrics = []
        
        min_len = min(len(predicted_boxes), len(ground_truth_boxes))
        
        for frame_idx in range(min_len):
            pred_box = predicted_boxes[frame_idx]
            gt_box = ground_truth_boxes[frame_idx]
            
            # Calculate IoU
            iou = self.calculate_iou(pred_box, gt_box)
            
            # Calculate center distance error
            pred_center = [pred_box[0] + pred_box[2]/2, pred_box[1] + pred_box[3]/2]
            gt_center = [gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2]
            center_error = np.sqrt((pred_center[0] - gt_center[0])**2 + (pred_center[1] - gt_center[1])**2)
            
            # Calculate size error
            pred_area = pred_box[2] * pred_box[3]
            gt_area = gt_box[2] * gt_box[3]
            size_error = abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0.0
            
            # Calculate aspect ratio error
            pred_aspect = pred_box[2] / pred_box[3] if pred_box[3] > 0 else 1.0
            gt_aspect = gt_box[2] / gt_box[3] if gt_box[3] > 0 else 1.0
            aspect_error = abs(pred_aspect - gt_aspect) / gt_aspect if gt_aspect > 0 else 0.0
            
            frame_metrics.append({
                'frame_idx': frame_idx,
                'iou': iou,
                'center_error': center_error,
                'size_error': size_error,
                'aspect_error': aspect_error,
                'pred_box': pred_box,
                'gt_box': gt_box
            })
        
        return frame_metrics
    
    def calculate_enhanced_map_metrics(self, predicted_boxes, ground_truth_boxes, confidence_scores=None, object_id=None):
        """Calculate enhanced mAP metrics with additional analysis."""
        if not predicted_boxes or not ground_truth_boxes:
            return {
                'mAP': 0.0, 'AP_per_threshold': {}, 'precision_recall_curves': {},
                'frame_metrics': [], 'consistency_score': 0.0
            }
        
        # Standard mAP calculation
        standard_metrics = self.calculate_map_per_object(
            {object_id: predicted_boxes}, 
            {object_id: ground_truth_boxes}, 
            [object_id]
        )[object_id]
        
        # Enhanced analysis
        if confidence_scores is None:
            confidence_scores = [1.0] * len(predicted_boxes)
        
        # Calculate precision-recall curves for key IoU thresholds
        pr_curves = {}
        key_thresholds = [0.5, 0.75, 0.9]
        
        for threshold in key_thresholds:
            precisions, recalls, ap = self.calculate_precision_recall_curve(
                predicted_boxes, ground_truth_boxes, confidence_scores, threshold
            )
            pr_curves[threshold] = {
                'precisions': precisions,
                'recalls': recalls,
                'ap': ap
            }
        
        # Frame-by-frame analysis
        frame_metrics = self.calculate_frame_by_frame_metrics(
            predicted_boxes, ground_truth_boxes, object_id
        )
        
        # Calculate consistency score (how stable is tracking over time)
        ious = [fm['iou'] for fm in frame_metrics]
        center_errors = [fm['center_error'] for fm in frame_metrics]
        
        if len(ious) > 1:
            iou_std = np.std(ious)
            center_error_std = np.std(center_errors)
            # Consistency score: lower std deviation = higher consistency
            consistency_score = max(0.0, 1.0 - (iou_std + center_error_std / 100.0))
        else:
            consistency_score = 1.0
        
        # Combine all metrics
        enhanced_metrics = {
            'mAP': standard_metrics['mAP'],
            'AP_per_threshold': standard_metrics['AP_per_threshold'],
            'precision_recall_curves': pr_curves,
            'frame_metrics': frame_metrics,
            'consistency_score': consistency_score,
            'mean_iou': np.mean(ious) if ious else 0.0,
            'mean_center_error': np.mean(center_errors) if center_errors else 0.0,
            'mean_confidence': np.mean(confidence_scores),
            'total_predictions': len(predicted_boxes),
            'total_ground_truths': len(ground_truth_boxes)
        }
        
        return enhanced_metrics
    
    def enhanced_process_gops(self, num_gops=3, sequence_prefix=""):
        """Enhanced GOP processing with detailed mAP evaluation."""
        
        print(f"🚀 Enhanced Motion Vector Tracking with Comprehensive mAP Evaluation")
        print(f"   📊 Processing {num_gops} GOPs with detailed metrics...")
        
        # Track overall statistics
        all_gop_metrics = []
        detailed_results = {}
        
        for gop_idx in range(num_gops):
            print(f"\n{'='*60}")
            print(f"📖 GOP {gop_idx} - Enhanced Evaluation")
            print(f"{'='*60}")
            
            # Load GOP data
            gop_data = self.load_gop_data(gop_idx)
            if not gop_data:
                print(f"❌ Failed to load GOP {gop_idx}")
                continue
            
            # Find fully visible objects
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
            
            print(f"   🎯 Processing {len(target_objects)} objects: {list(target_objects.keys())}")
            
            # Set up colors
            self.colors = self.get_colors_for_objects(list(target_objects.keys()))
            
            # Accumulate motion vectors and get ground truth
            accumulated_motion, frame_motions, ground_truth_boxes = self.accumulate_motion_vectors(gop_data, target_objects)
            
            # Enhanced evaluation for each object
            gop_object_metrics = {}
            
            for obj_id in target_objects.keys():
                if obj_id in accumulated_motion and obj_id in ground_truth_boxes:
                    predicted_boxes = accumulated_motion[obj_id]['bounding_boxes']
                    gt_boxes = ground_truth_boxes[obj_id]
                    
                    # Generate confidence scores based on motion consistency
                    confidence_scores = []
                    motion_vectors = accumulated_motion[obj_id]['frame_displacements']
                    
                    for i, motion_vec in enumerate(motion_vectors):
                        motion_magnitude = np.linalg.norm(motion_vec)
                        # Higher confidence for moderate motion, lower for very high or very low
                        if motion_magnitude < 0.5:
                            conf = 0.6 + 0.4 * (motion_magnitude / 0.5)  # 0.6 to 1.0
                        elif motion_magnitude < 3.0:
                            conf = 1.0 - 0.3 * ((motion_magnitude - 0.5) / 2.5)  # 1.0 to 0.7
                        else:
                            conf = max(0.3, 0.7 - 0.4 * ((motion_magnitude - 3.0) / 5.0))  # 0.7 to 0.3
                        
                        confidence_scores.append(conf)
                    
                    # Add initial confidence
                    confidence_scores = [1.0] + confidence_scores
                    
                    # Calculate enhanced metrics
                    enhanced_metrics = self.calculate_enhanced_map_metrics(
                        predicted_boxes, gt_boxes, confidence_scores, obj_id
                    )
                    
                    gop_object_metrics[obj_id] = enhanced_metrics
                    
                    # Print detailed results
                    print(f"\n   📊 Object {obj_id} Detailed Analysis:")
                    print(f"      🎯 mAP@[0.5:0.95]: {enhanced_metrics['mAP']:.3f}")
                    print(f"      🎯 AP@0.5: {enhanced_metrics['AP_per_threshold'][0.5]:.3f}")
                    print(f"      🎯 AP@0.75: {enhanced_metrics['AP_per_threshold'][0.75]:.3f}")
                    print(f"      📈 Mean IoU: {enhanced_metrics['mean_iou']:.3f}")
                    print(f"      📍 Mean Center Error: {enhanced_metrics['mean_center_error']:.1f}px")
                    print(f"      🔄 Consistency Score: {enhanced_metrics['consistency_score']:.3f}")
                    print(f"      📊 Mean Confidence: {enhanced_metrics['mean_confidence']:.3f}")
                    
                    # Performance assessment per object
                    if enhanced_metrics['mAP'] >= 0.8:
                        performance = "🟢 Excellent"
                    elif enhanced_metrics['mAP'] >= 0.6:
                        performance = "🟡 Good"
                    elif enhanced_metrics['mAP'] >= 0.4:
                        performance = "🟠 Fair"
                    else:
                        performance = "🔴 Poor"
                    
                    print(f"      🏆 Performance: {performance}")
                    
                    # Store frame-by-frame metrics
                    self.frame_by_frame_metrics[f"gop{gop_idx}_obj{obj_id}"] = enhanced_metrics['frame_metrics']
                    self.object_consistency_scores[f"gop{gop_idx}_obj{obj_id}"] = enhanced_metrics['consistency_score']
            
            # Create enhanced visualization
            self.create_enhanced_visualization(
                gop_data, accumulated_motion, frame_motions, ground_truth_boxes, 
                gop_object_metrics, gop_idx, sequence_prefix
            )
            
            # Store GOP results
            detailed_results[gop_idx] = {
                'object_metrics': gop_object_metrics,
                'target_objects': target_objects,
                'num_objects': len(target_objects)
            }
            
            # GOP summary
            if gop_object_metrics:
                gop_maps = [m['mAP'] for m in gop_object_metrics.values()]
                gop_ious = [m['mean_iou'] for m in gop_object_metrics.values()]
                gop_consistency = [m['consistency_score'] for m in gop_object_metrics.values()]
                
                gop_summary = {
                    'gop_idx': gop_idx,
                    'mean_mAP': np.mean(gop_maps),
                    'mean_IoU': np.mean(gop_ious),
                    'mean_consistency': np.mean(gop_consistency),
                    'num_objects': len(gop_object_metrics)
                }
                
                all_gop_metrics.append(gop_summary)
                
                print(f"\n   📈 GOP {gop_idx} Summary:")
                print(f"      Mean mAP: {gop_summary['mean_mAP']:.3f}")
                print(f"      Mean IoU: {gop_summary['mean_IoU']:.3f}")
                print(f"      Mean Consistency: {gop_summary['mean_consistency']:.3f}")
                print(f"      Objects processed: {gop_summary['num_objects']}")
        
        # Generate comprehensive summary
        self.generate_comprehensive_summary(all_gop_metrics, detailed_results)
        
        return {
            'gop_metrics': all_gop_metrics,
            'detailed_results': detailed_results,
            'frame_metrics': self.frame_by_frame_metrics,
            'consistency_scores': self.object_consistency_scores
        }
    
    def create_enhanced_visualization(self, gop_data, accumulated_motion, frame_motions, 
                                    ground_truth_boxes, object_metrics, gop_idx, sequence_prefix):
        """Create enhanced visualization with detailed mAP information."""
        
        if not gop_data or 'rgb_frames' not in gop_data:
            print("❌ No RGB frames available for enhanced visualization")
            return
        
        rgb_frames = gop_data['rgb_frames']
        num_frames = len(rgb_frames)
        
        # Create video writer
        if sequence_prefix:
            output_path = f'{sequence_prefix}_enhanced_motion_map_gop{gop_idx}.mp4'
        else:
            output_path = f'enhanced_motion_map_gop{gop_idx}.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 6  # Slower for detailed viewing
        frame_size = (1200, 960)  # Wider frame for metrics panel
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        print(f"🎬 Creating enhanced visualization: {output_path}")
        
        for frame_idx in range(num_frames):
            # Get RGB frame
            rgb_frame = rgb_frames[frame_idx].copy()
            
            # Convert to BGR for OpenCV
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Create extended frame with metrics panel
            extended_frame = np.zeros((960, 1200, 3), dtype=np.uint8)
            extended_frame[:, :960] = bgr_frame  # Original frame on left
            extended_frame[:, 960:] = (50, 50, 50)  # Dark panel on right
            
            # Draw tracking results
            for obj_id in accumulated_motion.keys():
                color = self.colors[obj_id]
                
                # Current frame bounding boxes
                if frame_idx < len(accumulated_motion[obj_id]['bounding_boxes']):
                    # Predicted bounding box
                    pred_bbox = accumulated_motion[obj_id]['bounding_boxes'][frame_idx]
                    x, y, w, h = pred_bbox
                    cv2.rectangle(extended_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 3)
                    
                    # Ground truth bounding box (if available)
                    if obj_id in ground_truth_boxes and frame_idx < len(ground_truth_boxes[obj_id]):
                        gt_bbox = ground_truth_boxes[obj_id][frame_idx]
                        gt_x, gt_y, gt_w, gt_h = gt_bbox
                        
                        # Draw dashed GT box
                        dash_length = 5
                        for i in range(0, int(gt_w), dash_length * 2):
                            start_x = int(gt_x + i)
                            end_x = min(int(gt_x + i + dash_length), int(gt_x + gt_w))
                            cv2.line(extended_frame, (start_x, int(gt_y)), (end_x, int(gt_y)), color, 2)
                            cv2.line(extended_frame, (start_x, int(gt_y + gt_h)), (end_x, int(gt_y + gt_h)), color, 2)
                        
                        for i in range(0, int(gt_h), dash_length * 2):
                            start_y = int(gt_y + i)
                            end_y = min(int(gt_y + i + dash_length), int(gt_y + gt_h))
                            cv2.line(extended_frame, (int(gt_x), start_y), (int(gt_x), end_y), color, 2)
                            cv2.line(extended_frame, (int(gt_x + gt_w), start_y), (int(gt_x + gt_w), end_y), color, 2)
                        
                        # Calculate and display current IoU
                        current_iou = self.calculate_iou(pred_bbox, gt_bbox)
                        cv2.putText(extended_frame, f'IoU: {current_iou:.3f}', 
                                  (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Object label
                    cv2.putText(extended_frame, f'Obj {obj_id}', 
                              (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw metrics panel
            panel_x = 970
            panel_y = 30
            line_height = 25
            
            # Panel title
            cv2.putText(extended_frame, 'Enhanced mAP Metrics', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            panel_y += line_height * 1.5
            
            # Frame info
            cv2.putText(extended_frame, f'Frame: {frame_idx}/{num_frames-1}', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            panel_y += line_height
            
            cv2.putText(extended_frame, f'GOP: {gop_idx}', 
                       (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            panel_y += line_height * 1.5
            
            # Object metrics
            for obj_id, metrics in object_metrics.items():
                color = self.colors[obj_id]
                
                cv2.putText(extended_frame, f'Object {obj_id}:', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                panel_y += line_height
                
                cv2.putText(extended_frame, f'  mAP: {metrics["mAP"]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 0.8
                
                cv2.putText(extended_frame, f'  AP@0.5: {metrics["AP_per_threshold"][0.5]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 0.8
                
                cv2.putText(extended_frame, f'  Mean IoU: {metrics["mean_iou"]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 0.8
                
                cv2.putText(extended_frame, f'  Consistency: {metrics["consistency_score"]:.3f}', 
                           (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                panel_y += line_height * 1.2
            
            # Overall frame info
            cv2.putText(extended_frame, 'Enhanced Motion Vector Tracking', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(extended_frame, 'Solid: Predicted | Dashed: Ground Truth', 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Write frame
            out.write(extended_frame)
            
            if frame_idx % 10 == 0:
                print(f"   📍 Processing frame {frame_idx}/{num_frames}")
        
        out.release()
        print(f"✅ Enhanced visualization created: {output_path}")
    
    def generate_comprehensive_summary(self, all_gop_metrics, detailed_results):
        """Generate comprehensive evaluation summary."""
        
        print(f"\n{'='*80}")
        print(f"🏆 COMPREHENSIVE MOTION VECTOR TRACKING EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        if not all_gop_metrics:
            print("❌ No metrics available for comprehensive summary")
            return
        
        # Overall statistics
        overall_maps = [gop['mean_mAP'] for gop in all_gop_metrics]
        overall_ious = [gop['mean_IoU'] for gop in all_gop_metrics]
        overall_consistency = [gop['mean_consistency'] for gop in all_gop_metrics]
        total_objects = sum(gop['num_objects'] for gop in all_gop_metrics)
        
        print(f"\n📊 OVERALL PERFORMANCE METRICS:")
        print(f"   🎯 Overall Mean mAP: {np.mean(overall_maps):.3f} ± {np.std(overall_maps):.3f}")
        print(f"   🎯 Overall Mean IoU: {np.mean(overall_ious):.3f} ± {np.std(overall_ious):.3f}")
        print(f"   🎯 Overall Consistency: {np.mean(overall_consistency):.3f} ± {np.std(overall_consistency):.3f}")
        print(f"   📊 Total Objects Tracked: {total_objects}")
        print(f"   📊 Total GOPs Processed: {len(all_gop_metrics)}")
        
        # GOP-by-GOP breakdown
        print(f"\n📈 GOP-BY-GOP PERFORMANCE:")
        print(f"{'GOP':<5} {'mAP':<8} {'IoU':<8} {'Consistency':<12} {'Objects':<8}")
        print("-" * 45)
        
        for gop_metric in all_gop_metrics:
            gop_idx = gop_metric['gop_idx']
            mean_map = gop_metric['mean_mAP']
            mean_iou = gop_metric['mean_IoU']
            mean_cons = gop_metric['mean_consistency']
            num_objs = gop_metric['num_objects']
            
            print(f"{gop_idx:<5} {mean_map:<8.3f} {mean_iou:<8.3f} {mean_cons:<12.3f} {num_objs:<8}")
        
        # Performance analysis
        excellent_gops = sum(1 for gop in all_gop_metrics if gop['mean_mAP'] >= 0.7)
        good_gops = sum(1 for gop in all_gop_metrics if 0.5 <= gop['mean_mAP'] < 0.7)
        fair_gops = sum(1 for gop in all_gop_metrics if 0.3 <= gop['mean_mAP'] < 0.5)
        poor_gops = sum(1 for gop in all_gop_metrics if gop['mean_mAP'] < 0.3)
        
        print(f"\n🎯 PERFORMANCE DISTRIBUTION:")
        print(f"   🟢 Excellent (mAP ≥ 0.7): {excellent_gops} GOPs")
        print(f"   🟡 Good (0.5 ≤ mAP < 0.7): {good_gops} GOPs")
        print(f"   🟠 Fair (0.3 ≤ mAP < 0.5): {fair_gops} GOPs")
        print(f"   🔴 Poor (mAP < 0.3): {poor_gops} GOPs")
        
        # Insights and recommendations
        print(f"\n💡 INSIGHTS & RECOMMENDATIONS:")
        
        if np.mean(overall_maps) >= 0.7:
            print(f"   ✅ Excellent overall performance! Motion vector tracking is highly effective.")
        elif np.mean(overall_maps) >= 0.5:
            print(f"   👍 Good performance with room for improvement in motion prediction accuracy.")
        else:
            print(f"   ⚠️  Performance needs improvement. Consider adjusting motion scaling or smoothing parameters.")
        
        if np.std(overall_maps) < 0.1:
            print(f"   🔄 Highly consistent performance across GOPs.")
        else:
            print(f"   ⚡ Variable performance - some GOPs more challenging than others.")
        
        if np.mean(overall_consistency) >= 0.8:
            print(f"   📈 Excellent tracking consistency - objects tracked smoothly over time.")
        else:
            print(f"   📊 Tracking consistency could be improved - consider temporal smoothing.")
        
        # Best and worst performing analysis
        best_gop = max(all_gop_metrics, key=lambda x: x['mean_mAP'])
        worst_gop = min(all_gop_metrics, key=lambda x: x['mean_mAP'])
        
        print(f"\n🏆 BEST PERFORMING GOP: {best_gop['gop_idx']}")
        print(f"   mAP: {best_gop['mean_mAP']:.3f}, IoU: {best_gop['mean_IoU']:.3f}, Consistency: {best_gop['mean_consistency']:.3f}")
        
        print(f"\n⚠️  MOST CHALLENGING GOP: {worst_gop['gop_idx']}")
        print(f"   mAP: {worst_gop['mean_mAP']:.3f}, IoU: {worst_gop['mean_IoU']:.3f}, Consistency: {worst_gop['mean_consistency']:.3f}")
        
        print(f"\n📹 GENERATED VIDEOS:")
        print(f"   • Enhanced visualizations: *enhanced_motion_map_gop*.mp4")
        print(f"   • Features: Real-time mAP metrics, IoU scores, consistency analysis")
        print(f"   • Panel shows: mAP, AP@0.5, Mean IoU, Consistency scores")
        print(f"   • Use: ffplay seq1_enhanced_motion_map_gop0.mp4")


def main():
    """Main function for enhanced motion vector evaluation."""
    print("🚀 Enhanced Motion Vector Tracker with Comprehensive mAP Evaluation")
    print("=" * 80)
    
    # Create enhanced evaluator
    evaluator = EnhancedMotionVectorEvaluator(verbose=True)
    
    # Process sequences with enhanced evaluation
    results = evaluator.enhanced_process_gops(num_gops=3, sequence_prefix="enhanced")
    
    if results:
        print(f"\n✅ Enhanced motion vector evaluation completed!")
        print(f"   📊 Processed {len(results['gop_metrics'])} GOPs with comprehensive metrics")
        print(f"   📈 Generated frame-by-frame analysis")
        print(f"   🎯 Created enhanced visualizations with real-time metrics")
        print(f"   📹 Videos show tracking accuracy, mAP scores, and consistency analysis")
    else:
        print(f"\n❌ Enhanced evaluation failed")
    
    return 0

if __name__ == "__main__":
    exit(main())
