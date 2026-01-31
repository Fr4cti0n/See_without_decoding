#!/usr/bin/env python3
"""
Simple mAP Enhancement for Motion Vector Tracking

This script adds mAP and AP evaluation to your existing motion vector tracking
by extending the accumulated_motion_predictor.py functionality.
"""

import sys
import os
import numpy as np
import cv2
import json
from collections import defaultdict

def calculate_iou(box1, box2):
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

def calculate_map_for_object(predicted_boxes, ground_truth_boxes, iou_thresholds=None):
    """Calculate mAP and AP metrics for a single object."""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    if not predicted_boxes or not ground_truth_boxes:
        return {'mAP': 0.0, 'AP_per_threshold': {th: 0.0 for th in iou_thresholds}}
    
    # Ensure same number of predictions and ground truths
    min_len = min(len(predicted_boxes), len(ground_truth_boxes))
    predicted_boxes = predicted_boxes[:min_len]
    ground_truth_boxes = ground_truth_boxes[:min_len]
    
    ap_per_threshold = {}
    
    for threshold in iou_thresholds:
        correct_predictions = 0
        
        for pred_box, gt_box in zip(predicted_boxes, ground_truth_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if iou >= threshold:
                correct_predictions += 1
        
        # Calculate Average Precision for this threshold
        ap = correct_predictions / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0.0
        ap_per_threshold[threshold] = ap
    
    # Calculate mAP (mean over all thresholds)
    mAP = np.mean(list(ap_per_threshold.values()))
    
    return {
        'mAP': mAP,
        'AP_per_threshold': ap_per_threshold,
        'total_predictions': len(predicted_boxes),
        'total_ground_truths': len(ground_truth_boxes)
    }

def calculate_additional_metrics(predicted_boxes, ground_truth_boxes):
    """Calculate additional tracking metrics beyond mAP."""
    if not predicted_boxes or not ground_truth_boxes:
        return {}
    
    min_len = min(len(predicted_boxes), len(ground_truth_boxes))
    
    center_errors = []
    size_errors = []
    aspect_errors = []
    ious = []
    
    for i in range(min_len):
        pred_box = predicted_boxes[i]
        gt_box = ground_truth_boxes[i]
        
        # Calculate IoU
        iou = calculate_iou(pred_box, gt_box)
        ious.append(iou)
        
        # Calculate center distance error
        pred_center = [pred_box[0] + pred_box[2]/2, pred_box[1] + pred_box[3]/2]
        gt_center = [gt_box[0] + gt_box[2]/2, gt_box[1] + gt_box[3]/2]
        center_error = np.sqrt((pred_center[0] - gt_center[0])**2 + (pred_center[1] - gt_center[1])**2)
        center_errors.append(center_error)
        
        # Calculate size error
        pred_area = pred_box[2] * pred_box[3]
        gt_area = gt_box[2] * gt_box[3]
        size_error = abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0.0
        size_errors.append(size_error)
        
        # Calculate aspect ratio error
        pred_aspect = pred_box[2] / pred_box[3] if pred_box[3] > 0 else 1.0
        gt_aspect = gt_box[2] / gt_box[3] if gt_box[3] > 0 else 1.0
        aspect_error = abs(pred_aspect - gt_aspect) / gt_aspect if gt_aspect > 0 else 0.0
        aspect_errors.append(aspect_error)
    
    return {
        'mean_iou': np.mean(ious),
        'std_iou': np.std(ious),
        'mean_center_error': np.mean(center_errors),
        'std_center_error': np.std(center_errors),
        'mean_size_error': np.mean(size_errors),
        'mean_aspect_error': np.mean(aspect_errors),
        'consistency_score': max(0.0, 1.0 - np.std(ious))  # Higher IoU consistency = higher score
    }

def analyze_tracking_results():
    """Analyze existing motion vector tracking results and calculate mAP metrics."""
    
    print("🚀 Motion Vector Tracking mAP Analysis")
    print("=" * 60)
    
    # Check if we have existing videos from motion vector tracking
    video_files = []
    for pattern in ['*accumulated_motion_prediction_gop*.mp4', '*motion_vector_enhanced_tracking*.mp4', 'seq*_accumulated_motion_prediction_gop*.mp4']:
        import glob
        video_files.extend(glob.glob(pattern))
    
    if video_files:
        print(f"📹 Found {len(video_files)} existing motion vector tracking videos:")
        for video in video_files:
            print(f"   • {video}")
    else:
        print("⚠️  No existing motion vector tracking videos found.")
        print("   Please run one of the motion vector tracking scripts first:")
        print("   • python accumulated_motion_predictor.py")
        print("   • python motion_vector_enhanced_multi_gop_tracker.py")
    
    # Try to run the accumulated motion predictor to get fresh results
    try:
        print(f"\n🎯 Running accumulated motion predictor to generate fresh tracking data...")
        
        # Import and run the existing predictor
        from accumulated_motion_predictor import AccumulatedMotionPredictor
        
        predictor = AccumulatedMotionPredictor(verbose=True)
        
        # Load sequence
        success, sequences = predictor.load_sequence()
        if not success:
            print("❌ Failed to load sequence")
            return
        
        print(f"✅ Loaded sequence successfully")
        
        # Process with enhanced evaluation
        print(f"\n📊 Processing GOPs with enhanced mAP evaluation...")
        enhanced_results = process_gops_with_map_evaluation(predictor)
        
        if enhanced_results:
            print(f"\n✅ Enhanced mAP evaluation completed!")
            print_comprehensive_map_summary(enhanced_results)
        else:
            print(f"\n❌ Enhanced evaluation failed")
            
    except ImportError as e:
        print(f"❌ Could not import AccumulatedMotionPredictor: {e}")
        print("   Make sure accumulated_motion_predictor.py is in the current directory")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def process_gops_with_map_evaluation(predictor, num_gops=3):
    """Process GOPs with enhanced mAP evaluation."""
    
    all_results = {}
    overall_metrics = []
    
    for gop_idx in range(num_gops):
        print(f"\n{'='*50}")
        print(f"📖 GOP {gop_idx} - Enhanced mAP Analysis")
        print(f"{'='*50}")
        
        # Load GOP data
        gop_data = predictor.load_gop_data(gop_idx)
        if not gop_data:
            print(f"❌ Failed to load GOP {gop_idx}")
            continue
        
        # Find fully visible objects
        fully_visible_objects = predictor.find_fully_visible_objects(gop_data)
        if not fully_visible_objects:
            print(f"⚠️  No fully visible objects found in GOP {gop_idx}")
            continue
        
        # Extract target objects data
        first_frame_annotations = gop_data['annotations']['annotations'][0]
        target_objects = predictor.extract_target_objects(first_frame_annotations, fully_visible_objects)
        
        if not target_objects:
            print(f"⚠️  No target objects data found in GOP {gop_idx}")
            continue
        
        print(f"   🎯 Processing {len(target_objects)} objects: {list(target_objects.keys())}")
        
        # Set up colors
        predictor.colors = predictor.get_colors_for_objects(list(target_objects.keys()))
        
        # Accumulate motion vectors and get ground truth
        accumulated_motion, frame_motions, ground_truth_boxes = predictor.accumulate_motion_vectors(gop_data, target_objects)
        
        # Enhanced evaluation for each object
        gop_results = {}
        gop_maps = []
        gop_ap50s = []
        gop_ap75s = []
        
        for obj_id in target_objects.keys():
            if obj_id in accumulated_motion and obj_id in ground_truth_boxes:
                predicted_boxes = accumulated_motion[obj_id]['bounding_boxes']
                gt_boxes = ground_truth_boxes[obj_id]
                
                # Calculate mAP metrics
                map_metrics = calculate_map_for_object(predicted_boxes, gt_boxes)
                
                # Calculate additional metrics
                additional_metrics = calculate_additional_metrics(predicted_boxes, gt_boxes)
                
                # Combine metrics
                combined_metrics = {**map_metrics, **additional_metrics}
                gop_results[obj_id] = combined_metrics
                
                # Collect for GOP summary
                gop_maps.append(map_metrics['mAP'])
                gop_ap50s.append(map_metrics['AP_per_threshold'][0.5])
                gop_ap75s.append(map_metrics['AP_per_threshold'][0.75])
                
                # Print detailed results
                print(f"\n   📊 Object {obj_id} Enhanced Analysis:")
                print(f"      🎯 mAP@[0.5:0.95]: {map_metrics['mAP']:.3f}")
                print(f"      🎯 AP@0.5: {map_metrics['AP_per_threshold'][0.5]:.3f}")
                print(f"      🎯 AP@0.75: {map_metrics['AP_per_threshold'][0.75]:.3f}")
                print(f"      📈 Mean IoU: {additional_metrics['mean_iou']:.3f}")
                print(f"      📍 Mean Center Error: {additional_metrics['mean_center_error']:.1f}px")
                print(f"      🔄 Consistency Score: {additional_metrics['consistency_score']:.3f}")
                print(f"      📦 Size Error: {additional_metrics['mean_size_error']:.3f}")
                print(f"      📐 Aspect Error: {additional_metrics['mean_aspect_error']:.3f}")
                
                # Performance classification
                if map_metrics['mAP'] >= 0.8:
                    performance = "🟢 Excellent"
                elif map_metrics['mAP'] >= 0.6:
                    performance = "🟡 Good"
                elif map_metrics['mAP'] >= 0.4:
                    performance = "🟠 Fair"
                else:
                    performance = "🔴 Poor"
                
                print(f"      🏆 Performance: {performance}")
        
        # Create enhanced visualization
        create_enhanced_map_visualization(gop_data, accumulated_motion, ground_truth_boxes, gop_results, gop_idx, predictor)
        
        # Store GOP results
        all_results[gop_idx] = {
            'object_results': gop_results,
            'gop_summary': {
                'mean_mAP': np.mean(gop_maps) if gop_maps else 0.0,
                'mean_AP50': np.mean(gop_ap50s) if gop_ap50s else 0.0,
                'mean_AP75': np.mean(gop_ap75s) if gop_ap75s else 0.0,
                'num_objects': len(gop_results)
            }
        }
        
        # Print GOP summary
        if gop_maps:
            print(f"\n   📈 GOP {gop_idx} Summary:")
            print(f"      Mean mAP: {np.mean(gop_maps):.3f}")
            print(f"      Mean AP@0.5: {np.mean(gop_ap50s):.3f}")
            print(f"      Mean AP@0.75: {np.mean(gop_ap75s):.3f}")
            print(f"      Objects: {len(gop_results)}")
            
            overall_metrics.append({
                'gop_idx': gop_idx,
                'mean_mAP': np.mean(gop_maps),
                'mean_AP50': np.mean(gop_ap50s),
                'mean_AP75': np.mean(gop_ap75s),
                'num_objects': len(gop_results)
            })
    
    return {
        'gop_results': all_results,
        'overall_metrics': overall_metrics
    }

def create_enhanced_map_visualization(gop_data, accumulated_motion, ground_truth_boxes, object_results, gop_idx, predictor):
    """Create enhanced visualization with mAP metrics overlay."""
    
    if not gop_data or 'rgb_frames' not in gop_data:
        print("❌ No RGB frames available for visualization")
        return
    
    rgb_frames = gop_data['rgb_frames']
    num_frames = len(rgb_frames)
    
    # Create video writer
    output_path = f'enhanced_map_evaluation_gop{gop_idx}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 6  # Slower for detailed viewing
    frame_size = (1280, 960)  # Wider for metrics panel
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print(f"🎬 Creating enhanced mAP visualization: {output_path}")
    
    for frame_idx in range(num_frames):
        # Get RGB frame
        rgb_frame = rgb_frames[frame_idx].copy()
        
        # Convert to BGR and create extended frame
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        extended_frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        extended_frame[:, :960] = bgr_frame
        extended_frame[:, 960:] = (40, 40, 40)  # Dark panel
        
        # Draw tracking results
        for obj_id in accumulated_motion.keys():
            color = predictor.colors[obj_id]
            
            if frame_idx < len(accumulated_motion[obj_id]['bounding_boxes']):
                # Predicted bounding box
                pred_bbox = accumulated_motion[obj_id]['bounding_boxes'][frame_idx]
                x, y, w, h = pred_bbox
                cv2.rectangle(extended_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 3)
                
                # Ground truth bounding box (dashed)
                if obj_id in ground_truth_boxes and frame_idx < len(ground_truth_boxes[obj_id]):
                    gt_bbox = ground_truth_boxes[obj_id][frame_idx]
                    gt_x, gt_y, gt_w, gt_h = gt_bbox
                    
                    # Draw dashed rectangle for ground truth
                    draw_dashed_rectangle(extended_frame, 
                                        (int(gt_x), int(gt_y)), 
                                        (int(gt_x + gt_w), int(gt_y + gt_h)), 
                                        color, 2)
                    
                    # Calculate current IoU
                    current_iou = calculate_iou(pred_bbox, gt_bbox)
                    cv2.putText(extended_frame, f'IoU: {current_iou:.3f}', 
                              (int(x), max(int(y) - 30, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(map(int, color)), 2)
                
                # Object label
                cv2.putText(extended_frame, f'Obj {obj_id}', 
                          (int(x), max(int(y) - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(map(int, color)), 2)
        
        # Draw metrics panel
        draw_metrics_panel(extended_frame, object_results, frame_idx, gop_idx)
        
        # Frame info
        cv2.putText(extended_frame, f'Enhanced mAP Evaluation - Frame {frame_idx}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(extended_frame, f'Solid: Predicted | Dashed: Ground Truth', 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Write frame
        out.write(extended_frame)
        
        if frame_idx % 10 == 0:
            print(f"   📍 Frame {frame_idx}/{num_frames}")
    
    out.release()
    print(f"✅ Enhanced mAP visualization created: {output_path}")

def draw_dashed_rectangle(frame, pt1, pt2, color, thickness):
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

def draw_metrics_panel(frame, object_results, frame_idx, gop_idx):
    """Draw comprehensive metrics panel."""
    panel_x = 970
    panel_y = 30
    line_height = 25
    
    # Panel title
    cv2.putText(frame, 'Enhanced mAP Metrics', 
               (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    panel_y += int(line_height * 1.5)
    
    # Frame info
    cv2.putText(frame, f'Frame: {frame_idx} | GOP: {gop_idx}', 
               (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    panel_y += int(line_height * 1.5)
    
    # Object metrics
    for obj_id, metrics in object_results.items():
        cv2.putText(frame, f'Object {obj_id}:', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        panel_y += line_height
        
        cv2.putText(frame, f'  mAP: {metrics["mAP"]:.3f}', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel_y += int(line_height * 0.8)
        
        cv2.putText(frame, f'  AP@0.5: {metrics["AP_per_threshold"][0.5]:.3f}', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel_y += int(line_height * 0.8)
        
        cv2.putText(frame, f'  Mean IoU: {metrics["mean_iou"]:.3f}', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel_y += int(line_height * 0.8)
        
        cv2.putText(frame, f'  Consistency: {metrics["consistency_score"]:.3f}', 
                   (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panel_y += int(line_height * 1.2)

def print_comprehensive_map_summary(results):
    """Print comprehensive mAP evaluation summary."""
    
    print(f"\n{'='*80}")
    print(f"🏆 COMPREHENSIVE MOTION VECTOR mAP EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    overall_metrics = results['overall_metrics']
    gop_results = results['gop_results']
    
    if not overall_metrics:
        print("❌ No metrics available for summary")
        return
    
    # Overall statistics
    all_maps = [m['mean_mAP'] for m in overall_metrics]
    all_ap50s = [m['mean_AP50'] for m in overall_metrics]
    all_ap75s = [m['mean_AP75'] for m in overall_metrics]
    total_objects = sum(m['num_objects'] for m in overall_metrics)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   🎯 Mean mAP@[0.5:0.95]: {np.mean(all_maps):.3f} ± {np.std(all_maps):.3f}")
    print(f"   🎯 Mean AP@0.5: {np.mean(all_ap50s):.3f}")
    print(f"   🎯 Mean AP@0.75: {np.mean(all_ap75s):.3f}")
    print(f"   📊 Total Objects: {total_objects}")
    print(f"   📊 GOPs Processed: {len(overall_metrics)}")
    
    # GOP breakdown
    print(f"\n📈 GOP-BY-GOP PERFORMANCE:")
    print(f"{'GOP':<5} {'mAP':<8} {'AP@0.5':<8} {'AP@0.75':<8} {'Objects':<8}")
    print("-" * 40)
    
    for metric in overall_metrics:
        gop_idx = metric['gop_idx']
        mean_map = metric['mean_mAP']
        mean_ap50 = metric['mean_AP50']
        mean_ap75 = metric['mean_AP75']
        num_objs = metric['num_objects']
        
        print(f"{gop_idx:<5} {mean_map:<8.3f} {mean_ap50:<8.3f} {mean_ap75:<8.3f} {num_objs:<8}")
    
    # Performance analysis
    excellent = sum(1 for m in all_maps if m >= 0.7)
    good = sum(1 for m in all_maps if 0.5 <= m < 0.7)
    fair = sum(1 for m in all_maps if 0.3 <= m < 0.5)
    poor = sum(1 for m in all_maps if m < 0.3)
    
    print(f"\n🎯 PERFORMANCE DISTRIBUTION:")
    print(f"   🟢 Excellent (≥0.7): {excellent} GOPs")
    print(f"   🟡 Good (0.5-0.7): {good} GOPs")
    print(f"   🟠 Fair (0.3-0.5): {fair} GOPs")
    print(f"   🔴 Poor (<0.3): {poor} GOPs")
    
    # Final assessment
    overall_performance = np.mean(all_maps)
    if overall_performance >= 0.7:
        assessment = "🟢 Excellent - Motion vector tracking performs very well!"
    elif overall_performance >= 0.5:
        assessment = "🟡 Good - Motion vector tracking shows solid performance"
    elif overall_performance >= 0.3:
        assessment = "🟠 Fair - Motion vector tracking needs some improvement"
    else:
        assessment = "🔴 Poor - Motion vector tracking requires significant enhancement"
    
    print(f"\n🏆 FINAL ASSESSMENT: {assessment}")
    print(f"   Overall Performance Score: {overall_performance:.3f}")
    
    # Best and worst GOP
    best_gop = max(overall_metrics, key=lambda x: x['mean_mAP'])
    worst_gop = min(overall_metrics, key=lambda x: x['mean_mAP'])
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   🏆 Best GOP: {best_gop['gop_idx']} (mAP: {best_gop['mean_mAP']:.3f})")
    print(f"   ⚠️  Most challenging GOP: {worst_gop['gop_idx']} (mAP: {worst_gop['mean_mAP']:.3f})")
    print(f"   📊 Consistency: {'High' if np.std(all_maps) < 0.1 else 'Variable'} (std: {np.std(all_maps):.3f})")
    
    print(f"\n📹 GENERATED CONTENT:")
    print(f"   • Enhanced videos: enhanced_map_evaluation_gop*.mp4")
    print(f"   • Real-time mAP metrics display")
    print(f"   • IoU calculations per frame")
    print(f"   • Tracking consistency analysis")
    print(f"   ⭐ Use: ffplay enhanced_map_evaluation_gop0.mp4")

if __name__ == "__main__":
    analyze_tracking_results()
