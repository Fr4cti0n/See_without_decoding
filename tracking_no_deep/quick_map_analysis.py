#!/usr/bin/env python3
"""
Quick mAP Metrics Calculator for Motion Vector Tracking

This script quickly calculates mAP metrics for your motion vector tracking results
without generating videos to avoid OpenCV issues.
"""

import sys
import os
import numpy as np
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x1_2, y1_2, x2_2, y2_2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def calculate_map_for_object(predicted_boxes, ground_truth_boxes):
    """Calculate mAP and AP metrics for a single object."""
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    if not predicted_boxes or not ground_truth_boxes:
        return {'mAP': 0.0, 'AP_per_threshold': {th: 0.0 for th in iou_thresholds}}
    
    min_len = min(len(predicted_boxes), len(ground_truth_boxes))
    predicted_boxes = predicted_boxes[:min_len]
    ground_truth_boxes = ground_truth_boxes[:min_len]
    
    ap_per_threshold = {}
    ious = []
    
    for threshold in iou_thresholds:
        correct_predictions = 0
        
        for pred_box, gt_box in zip(predicted_boxes, ground_truth_boxes):
            iou = calculate_iou(pred_box, gt_box)
            if threshold == 0.5:  # Store IoUs for first threshold
                ious.append(iou)
            if iou >= threshold:
                correct_predictions += 1
        
        ap = correct_predictions / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0.0
        ap_per_threshold[threshold] = ap
    
    mAP = np.mean(list(ap_per_threshold.values()))
    
    return {
        'mAP': mAP,
        'AP_per_threshold': ap_per_threshold,
        'mean_iou': np.mean(ious) if ious else 0.0,
        'total_predictions': len(predicted_boxes),
        'total_ground_truths': len(ground_truth_boxes)
    }

def quick_map_analysis():
    """Quick mAP analysis of motion vector tracking."""
    
    print("🚀 Quick mAP Analysis for Motion Vector Tracking")
    print("=" * 60)
    
    try:
        from accumulated_motion_predictor import AccumulatedMotionPredictor
        
        predictor = AccumulatedMotionPredictor(verbose=True)
        
        # Load sequence
        success, sequences = predictor.load_sequence()
        if not success:
            print("❌ Failed to load sequence")
            return
        
        print(f"✅ Loaded sequence successfully")
        
        # Process GOPs
        all_results = {}
        overall_metrics = []
        
        for gop_idx in range(3):  # Process 3 GOPs
            print(f"\n{'='*40}")
            print(f"📖 GOP {gop_idx} Analysis")
            print(f"{'='*40}")
            
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
            
            # Extract target objects
            first_frame_annotations = gop_data['annotations']['annotations'][0]
            target_objects = predictor.extract_target_objects(first_frame_annotations, fully_visible_objects)
            
            if not target_objects:
                print(f"⚠️  No target objects data found in GOP {gop_idx}")
                continue
            
            print(f"   🎯 Processing {len(target_objects)} objects: {list(target_objects.keys())}")
            
            # Set up colors (needed for the predictor)
            predictor.colors = predictor.get_colors_for_objects(list(target_objects.keys()))
            
            # Get tracking results
            accumulated_motion, frame_motions, ground_truth_boxes = predictor.accumulate_motion_vectors(gop_data, target_objects)
            
            # Calculate mAP for each object
            gop_results = {}
            gop_maps = []
            gop_ap50s = []
            gop_ap75s = []
            
            for obj_id in target_objects.keys():
                if obj_id in accumulated_motion and obj_id in ground_truth_boxes:
                    predicted_boxes = accumulated_motion[obj_id]['bounding_boxes']
                    gt_boxes = ground_truth_boxes[obj_id]
                    
                    # Calculate metrics
                    metrics = calculate_map_for_object(predicted_boxes, gt_boxes)
                    gop_results[obj_id] = metrics
                    
                    gop_maps.append(metrics['mAP'])
                    gop_ap50s.append(metrics['AP_per_threshold'][0.5])
                    gop_ap75s.append(metrics['AP_per_threshold'][0.75])
                    
                    # Print results
                    print(f"\n   📊 Object {obj_id}:")
                    print(f"      🎯 mAP@[0.5:0.95]: {metrics['mAP']:.3f}")
                    print(f"      🎯 AP@0.5: {metrics['AP_per_threshold'][0.5]:.3f}")
                    print(f"      🎯 AP@0.75: {metrics['AP_per_threshold'][0.75]:.3f}")
                    print(f"      📈 Mean IoU: {metrics['mean_iou']:.3f}")
                    print(f"      📊 Frames: {metrics['total_predictions']}")
                    
                    # Performance assessment
                    if metrics['mAP'] >= 0.7:
                        performance = "🟢 Excellent"
                    elif metrics['mAP'] >= 0.5:
                        performance = "🟡 Good"
                    elif metrics['mAP'] >= 0.3:
                        performance = "🟠 Fair"
                    else:
                        performance = "🔴 Poor"
                    
                    print(f"      🏆 Performance: {performance}")
            
            # GOP summary
            if gop_maps:
                gop_mean_map = np.mean(gop_maps)
                gop_mean_ap50 = np.mean(gop_ap50s)
                gop_mean_ap75 = np.mean(gop_ap75s)
                
                print(f"\n   📈 GOP {gop_idx} Summary:")
                print(f"      Mean mAP: {gop_mean_map:.3f}")
                print(f"      Mean AP@0.5: {gop_mean_ap50:.3f}")
                print(f"      Mean AP@0.75: {gop_mean_ap75:.3f}")
                
                overall_metrics.append({
                    'gop_idx': gop_idx,
                    'mean_mAP': gop_mean_map,
                    'mean_AP50': gop_mean_ap50,
                    'mean_AP75': gop_mean_ap75,
                    'num_objects': len(gop_results)
                })
            
            all_results[gop_idx] = gop_results
        
        # Overall summary
        print_overall_summary(overall_metrics)
        
    except ImportError as e:
        print(f"❌ Could not import AccumulatedMotionPredictor: {e}")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def print_overall_summary(overall_metrics):
    """Print comprehensive overall summary."""
    
    print(f"\n{'='*70}")
    print(f"🏆 OVERALL MOTION VECTOR TRACKING mAP SUMMARY")
    print(f"{'='*70}")
    
    if not overall_metrics:
        print("❌ No metrics available")
        return
    
    # Calculate overall statistics
    all_maps = [m['mean_mAP'] for m in overall_metrics]
    all_ap50s = [m['mean_AP50'] for m in overall_metrics]
    all_ap75s = [m['mean_AP75'] for m in overall_metrics]
    total_objects = sum(m['num_objects'] for m in overall_metrics)
    
    overall_map = np.mean(all_maps)
    overall_ap50 = np.mean(all_ap50s)
    overall_ap75 = np.mean(all_ap75s)
    map_std = np.std(all_maps)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   🎯 Overall mAP@[0.5:0.95]: {overall_map:.3f} ± {map_std:.3f}")
    print(f"   🎯 Overall AP@0.5: {overall_ap50:.3f}")
    print(f"   🎯 Overall AP@0.75: {overall_ap75:.3f}")
    print(f"   📊 Total Objects Tracked: {total_objects}")
    print(f"   📊 GOPs Processed: {len(overall_metrics)}")
    
    # GOP breakdown
    print(f"\n📈 GOP-BY-GOP BREAKDOWN:")
    print(f"{'GOP':<5} {'mAP':<8} {'AP@0.5':<8} {'AP@0.75':<8} {'Objects':<8}")
    print("-" * 40)
    
    for metric in overall_metrics:
        gop_idx = metric['gop_idx']
        mean_map = metric['mean_mAP']
        mean_ap50 = metric['mean_AP50']
        mean_ap75 = metric['mean_AP75']
        num_objs = metric['num_objects']
        
        print(f"{gop_idx:<5} {mean_map:<8.3f} {mean_ap50:<8.3f} {mean_ap75:<8.3f} {num_objs:<8}")
    
    # Performance classification
    if overall_map >= 0.7:
        assessment = "🟢 EXCELLENT - Motion vector tracking performs very well!"
        recommendation = "✅ Current approach is highly effective. Consider fine-tuning for edge cases."
    elif overall_map >= 0.5:
        assessment = "🟡 GOOD - Motion vector tracking shows solid performance"
        recommendation = "👍 Good results with room for improvement. Consider refining motion scaling or temporal consistency."
    elif overall_map >= 0.3:
        assessment = "🟠 FAIR - Motion vector tracking needs improvement"
        recommendation = "⚠️ Moderate performance. Consider adjusting motion amplification and smoothing parameters."
    else:
        assessment = "🔴 POOR - Motion vector tracking requires significant enhancement"
        recommendation = "❌ Low performance. Consider fundamental changes to motion extraction or prediction methods."
    
    print(f"\n🏆 FINAL ASSESSMENT: {assessment}")
    print(f"   Overall Performance Score: {overall_map:.3f}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   {recommendation}")
    
    # Consistency analysis
    if map_std < 0.1:
        consistency = "🔄 Highly consistent performance across GOPs"
    elif map_std < 0.2:
        consistency = "📊 Moderately consistent performance"
    else:
        consistency = "⚡ Variable performance - some GOPs more challenging"
    
    print(f"   {consistency} (std: {map_std:.3f})")
    
    # Best and worst GOP
    best_gop = max(overall_metrics, key=lambda x: x['mean_mAP'])
    worst_gop = min(overall_metrics, key=lambda x: x['mean_mAP'])
    
    print(f"\n📊 PERFORMANCE INSIGHTS:")
    print(f"   🏆 Best GOP: {best_gop['gop_idx']} (mAP: {best_gop['mean_mAP']:.3f})")
    print(f"   ⚠️ Most challenging GOP: {worst_gop['gop_idx']} (mAP: {worst_gop['mean_mAP']:.3f})")
    
    performance_gap = best_gop['mean_mAP'] - worst_gop['mean_mAP']
    if performance_gap > 0.3:
        print(f"   ⚡ Large performance gap ({performance_gap:.3f}) indicates scene-dependent challenges")
    else:
        print(f"   ✅ Consistent performance across different scene conditions")
    
    print(f"\n🎯 MOTION VECTOR TRACKING STRENGTHS:")
    if overall_ap50 >= 0.8:
        print(f"   ✅ Excellent object localization (AP@0.5: {overall_ap50:.3f})")
    elif overall_ap50 >= 0.6:
        print(f"   👍 Good object localization with minor drift")
    else:
        print(f"   ⚠️ Object localization needs improvement")
    
    if overall_ap75 >= 0.6:
        print(f"   ✅ High precision tracking (AP@0.75: {overall_ap75:.3f})")
    elif overall_ap75 >= 0.4:
        print(f"   👍 Moderate precision with some bounding box size issues")
    else:
        print(f"   ⚠️ Low precision - bounding box accuracy needs improvement")
    
    print(f"\n📹 EXISTING VIDEOS:")
    video_files = []
    for pattern in ['*accumulated_motion_prediction_gop*.mp4', '*motion_vector_enhanced_tracking*.mp4']:
        import glob
        video_files.extend(glob.glob(pattern))
    
    if video_files:
        print(f"   Found {len(video_files)} motion vector tracking videos:")
        for video in video_files[:3]:  # Show first 3
            print(f"   • {video}")
        if len(video_files) > 3:
            print(f"   • ... and {len(video_files) - 3} more")
        print(f"   ⭐ Use: ffplay {video_files[0]} to view tracking results")
    else:
        print(f"   ⚠️ No existing videos found. Run tracking scripts to generate visualizations.")

if __name__ == "__main__":
    quick_map_analysis()
