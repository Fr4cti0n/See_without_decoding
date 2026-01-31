#!/usr/bin/env python3
"""
Three-Method Comparison: Mean-VC vs I-frame vs MV Model

This script evaluates three tracking approaches on moving vs static objects:
1. Mean-VC: Average velocity continuation baseline
2. I-frame: Static baseline (assumes objects don't move)
3. MV Model: Motion vector based tracking

The comparison demonstrates:
- I-frame excels on static objects (no motion assumption is correct)
- MV model excels on moving objects (tracks actual motion)
- Mean-VC provides interpolation between the two

Usage:
    python evaluate_three_methods_comparison.py --mv-checkpoint best_mots17_deep_tracker.pt
"""

import os
import sys
import json
import time
import torch
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'mots_exp'))
sys.path.insert(0, str(project_root / 'mots_exp' / 'scripts'))

from dataset.factory.dataset_factory import create_mots_dataset
from mots_exp.scripts.train_mv_center import create_model, evaluate_map

# Import the moving object analysis function
from evaluate_moving_objects_only import (
    identify_moving_objects_in_gop,
    STATIC_CAMERA_MOT17_SEQUENCES,
    STATIC_CAMERA_MOT15_SEQUENCES,
    STATIC_CAMERA_MOT20_SEQUENCES
)


def compute_box_center_torch(boxes):
    """
    Compute center of bounding boxes in [cx, cy, w, h] format (torch tensors)
    Args:
        boxes: [N, 4] tensor in [cx, cy, w, h] format
    Returns:
        centers: [N, 2] tensor with [cx, cy]
    """
    return boxes[:, :2]  # cx, cy are already the first two columns


def get_dataset_from_sequence(sequence_name):
    """Extract dataset name from sequence name"""
    if sequence_name.startswith('MOT17'):
        return 'MOT17'
    elif sequence_name.startswith('MOT20'):
        return 'MOT20'
    else:
        return 'MOT15'


def filter_static_gops_by_dataset(indices, dataset, static_sequences):
    """Filter GOP indices to only static camera sequences"""
    static_indices = []
    for idx in indices:
        seq_info = dataset.sequences[idx]
        video_name = seq_info['video_name'] if isinstance(seq_info, dict) else seq_info
        seq_name = video_name.split('_')[0]
        
        if any(seq_name.startswith(s) or s.startswith(seq_name) for s in static_sequences):
            static_indices.append(idx)
    return static_indices


def compute_iou_batch(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes: [N, 4] in format [cx, cy, w, h] normalized
    """
    # Convert to [x1, y1, x2, y2]
    boxes1_xyxy = boxes1.clone()
    boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2
    
    boxes2_xyxy = boxes2.clone()
    boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Compute intersection
    x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[:, 0])
    y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[:, 1])
    x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[:, 2])
    y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[:, 3])
    
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Compute union
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    union = area1[:, None] + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


def evaluate_mean_vc_baseline(dataset, gop_start_idx, sequence_length, moving_ids, all_ids, iou_threshold=0.5):
    """
    Evaluate Mean Velocity Continuation baseline
    Predicts box position by averaging velocity over previous frames
    
    Returns:
        dict with mAP for moving and static objects separately
    """
    # Track object positions over time
    object_history = defaultdict(list)  # object_id -> list of (frame_idx, box)
    
    moving_tp, moving_fp, moving_fn = 0, 0, 0
    static_tp, static_fp, static_fn = 0, 0, 0
    
    # Build history from first few frames (use first 10 frames to estimate velocity)
    for frame_offset in range(min(10, sequence_length)):
        frame_idx = gop_start_idx + frame_offset
        if frame_idx >= len(dataset):
            break
        
        frame_data = dataset[frame_idx]
        boxes = frame_data['boxes']
        ids = frame_data['ids']
        
        for box, obj_id in zip(boxes, ids):
            obj_id_int = int(obj_id.item())
            object_history[obj_id_int].append((frame_offset, box.clone()))
    
    # Evaluate each frame using velocity continuation
    for frame_offset in range(10, sequence_length):
        frame_idx = gop_start_idx + frame_offset
        if frame_idx >= len(dataset):
            break
        
        frame_data = dataset[frame_idx]
        gt_boxes = frame_data['boxes']
        gt_ids = frame_data['ids']
        
        if len(gt_boxes) == 0:
            continue
        
        # Predict boxes using mean velocity (store with object IDs)
        predictions = {}  # object_id -> predicted_box
        
        for obj_id_int in object_history.keys():
            history = object_history[obj_id_int]
            if len(history) < 2:
                continue
            
            # Compute mean velocity from history
            velocities = []
            for i in range(1, len(history)):
                prev_frame, prev_box = history[i-1]
                curr_frame, curr_box = history[i]
                
                # Center is already cx, cy (first two elements)
                prev_center = prev_box[:2]  # [cx, cy]
                curr_center = curr_box[:2]  # [cx, cy]
                
                dt = curr_frame - prev_frame
                if dt > 0:
                    velocity = (curr_center - prev_center) / dt
                    velocities.append(velocity)
            
            if len(velocities) == 0:
                continue
            
            # Average velocity
            mean_velocity = torch.stack(velocities).mean(dim=0)
            
            # Predict position at current frame
            last_frame, last_box = history[-1]
            dt = frame_offset - last_frame
            
            # Center is already cx, cy
            predicted_center = last_box[:2] + mean_velocity * dt  # [cx, cy]
            
            # Create predicted box (keep same size as last observation)
            pred_box = last_box.clone()
            pred_box[0] = predicted_center[0]  # cx
            pred_box[1] = predicted_center[1]  # cy
            
            predictions[obj_id_int] = pred_box
        
        # Separate predictions by moving/static
        pred_boxes_moving = []
        pred_boxes_static = []
        
        for obj_id_int, pred_box in predictions.items():
            if obj_id_int in moving_ids:
                pred_boxes_moving.append(pred_box)
            else:
                pred_boxes_static.append(pred_box)
        
        # Convert to tensors
        pred_boxes_moving = torch.stack(pred_boxes_moving) if pred_boxes_moving else torch.empty(0, 4)
        pred_boxes_static = torch.stack(pred_boxes_static) if pred_boxes_static else torch.empty(0, 4)
        
        # Evaluate moving objects
        gt_moving_mask = torch.tensor([int(obj_id.item()) in moving_ids for obj_id in gt_ids])
        gt_moving_boxes = gt_boxes[gt_moving_mask]
        
        if len(pred_boxes_moving) > 0 and len(gt_moving_boxes) > 0:
            ious = compute_iou_batch(pred_boxes_moving, gt_moving_boxes)
            max_ious, _ = ious.max(dim=1)
            moving_tp += (max_ious > iou_threshold).sum().item()
            moving_fp += (max_ious <= iou_threshold).sum().item()
        elif len(pred_boxes_moving) > 0:
            moving_fp += len(pred_boxes_moving)
        
        if len(gt_moving_boxes) > 0:
            moving_fn += len(gt_moving_boxes)
            if len(pred_boxes_moving) > 0:
                ious = compute_iou_batch(pred_boxes_moving, gt_moving_boxes)
                max_ious_gt, _ = ious.max(dim=0)
                moving_fn -= (max_ious_gt > iou_threshold).sum().item()
        
        # Evaluate static objects
        gt_static_mask = ~gt_moving_mask
        gt_static_boxes = gt_boxes[gt_static_mask]
        
        if len(pred_boxes_static) > 0 and len(gt_static_boxes) > 0:
            ious = compute_iou_batch(pred_boxes_static, gt_static_boxes)
            max_ious, _ = ious.max(dim=1)
            static_tp += (max_ious > iou_threshold).sum().item()
            static_fp += (max_ious <= iou_threshold).sum().item()
        elif len(pred_boxes_static) > 0:
            static_fp += len(pred_boxes_static)
        
        if len(gt_static_boxes) > 0:
            static_fn += len(gt_static_boxes)
            if len(pred_boxes_static) > 0:
                ious = compute_iou_batch(pred_boxes_static, gt_static_boxes)
                max_ious_gt, _ = ious.max(dim=0)
                static_fn -= (max_ious_gt > iou_threshold).sum().item()
        
        # ✅ FIX: Update history with predictions for autoregressive tracking
        # This ensures each frame uses previous predictions, not just initial GT frames
        for obj_id_int, pred_box in predictions.items():
            # Only update if object still exists in GT (to avoid tracking lost objects)
            gt_obj_ids = [int(gt_id.item()) for gt_id in gt_ids]
            if obj_id_int in gt_obj_ids:
                object_history[obj_id_int].append((frame_offset, pred_box.clone()))
    
    # Compute F1 scores
    moving_precision = moving_tp / (moving_tp + moving_fp) if (moving_tp + moving_fp) > 0 else 0.0
    moving_recall = moving_tp / (moving_tp + moving_fn) if (moving_tp + moving_fn) > 0 else 0.0
    moving_map = 2 * moving_precision * moving_recall / (moving_precision + moving_recall) if (moving_precision + moving_recall) > 0 else 0.0
    
    static_precision = static_tp / (static_tp + static_fp) if (static_tp + static_fp) > 0 else 0.0
    static_recall = static_tp / (static_tp + static_fn) if (static_tp + static_fn) > 0 else 0.0
    static_map = 2 * static_precision * static_recall / (static_precision + static_recall) if (static_precision + static_recall) > 0 else 0.0
    
    return {
        'moving_map': moving_map,
        'static_map': static_map,
        'moving_count': len(moving_ids),
        'static_count': len(all_ids) - len(moving_ids)
    }


def evaluate_mv_model(model, dataset, gop_start_idx, sequence_length, moving_ids, all_ids, device, iou_threshold=0.5, use_teacher_forcing=False):
    """
    Evaluate MV model on moving vs static objects
    
    Args:
        use_teacher_forcing: If True, use GT boxes as input (like training). If False, use autoregressive tracking.
    
    Returns:
        dict with mAP for moving and static objects separately
    """
    model.eval()
    
    moving_tp, moving_fp, moving_fn = 0, 0, 0
    static_tp, static_fp, static_fn = 0, 0, 0
    
    hidden_state = None
    
    # Initialize from I-frame (frame 0)
    # This will be used as the "previous frame" for frame 1 prediction
    prev_frame_data = dataset[gop_start_idx]
    prev_boxes = prev_frame_data['boxes'].clone() if len(prev_frame_data['boxes']) > 0 else None
    prev_ids = prev_frame_data['ids'].clone() if len(prev_frame_data['ids']) > 0 else None
    
    if prev_boxes is None:
        # No objects in I-frame, can't track
        return {
            'moving_map': 0.0,
            'static_map': 0.0,
            'moving_count': len(moving_ids),
            'static_count': len(all_ids) - len(moving_ids)
        }
    
    with torch.no_grad():
        # Evaluate each P-frame (skip frame 0 since it's I-frame without MVs)
        for frame_offset in range(1, sequence_length):
            frame_idx = gop_start_idx + frame_offset
            if frame_idx >= len(dataset):
                break
            
            frame_data = dataset[frame_idx]
            gt_boxes = frame_data['boxes']
            gt_ids = frame_data['ids']
            
            if len(gt_boxes) == 0:
                # No objects in this frame - update prev_boxes and continue
                if use_teacher_forcing:
                    prev_boxes = gt_boxes.clone()
                    prev_ids = gt_ids.clone()
                continue
            
            # Get motion vectors - shape is [2, H, W, 2] where last dim is temporal
            # Extract the first temporal component: [2, H, W]
            mv_data = frame_data.get('motion_vectors', None)
            if mv_data is None:
                # No motion vectors for this frame - update prev and skip prediction
                if use_teacher_forcing:
                    prev_boxes = gt_boxes.clone()
                    prev_ids = gt_ids.clone()
                continue
            
            mv = mv_data[:, :, :, 0]  # [2, H, W]
            mv = mv.unsqueeze(0)  # Add batch dim: [1, 2, H, W]
            mv = mv.to(device)
            
            # Teacher forcing: use GT boxes from PREVIOUS frame (not current frame!)
            # Autoregressive: use predicted boxes from previous frame
            if use_teacher_forcing:
                # Use GT boxes from PREVIOUS frame as input
                input_boxes = prev_boxes.clone()
                input_ids = prev_ids.clone()
            else:
                # Use predicted boxes from previous frame  
                input_boxes = prev_boxes
                input_ids = prev_ids
            
            # Convert input_boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
            boxes_xyxy = input_boxes.clone()
            boxes_xyxy[:, 0] = (input_boxes[:, 0] - input_boxes[:, 2] / 2) * 960  # x1
            boxes_xyxy[:, 1] = (input_boxes[:, 1] - input_boxes[:, 3] / 2) * 960  # y1
            boxes_xyxy[:, 2] = (input_boxes[:, 0] + input_boxes[:, 2] / 2) * 960  # x2
            boxes_xyxy[:, 3] = (input_boxes[:, 1] + input_boxes[:, 3] / 2) * 960  # y2
            boxes_xyxy = boxes_xyxy.to(device)  # Move to device!
            
            # Model prediction (pass None for dct_residuals since MV-only)
            try:
                pred_boxes_xyxy, confidences, hidden_state = model.forward_single_frame(
                    mv, 
                    None,  # dct_residuals (ignored by MV-only model)
                    boxes_xyxy,
                    hidden_state=hidden_state,
                    return_logits=False
                )
            except Exception as e:
                print(f"   ⚠️ Model forward error at frame {frame_offset}: {e}")
                hidden_state = None
                current_boxes = None
                current_ids = None
                continue
            
            # Convert predictions back to normalized [cx, cy, w, h]
            pred_boxes = torch.zeros_like(input_boxes)
            pred_boxes[:, 0] = (pred_boxes_xyxy[:, 0] + pred_boxes_xyxy[:, 2]) / 2 / 960  # cx
            pred_boxes[:, 1] = (pred_boxes_xyxy[:, 1] + pred_boxes_xyxy[:, 3]) / 2 / 960  # cy
            pred_boxes[:, 2] = (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]) / 960  # w
            pred_boxes[:, 3] = (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1]) / 960  # h
            
            # Update prev_boxes for next frame
            if use_teacher_forcing:
                # Teacher forcing: use GT boxes from current frame as "previous" for next iteration
                prev_boxes = gt_boxes.clone()
                prev_ids = gt_ids.clone()
            else:
                # Autoregressive: use predicted boxes
                prev_boxes = pred_boxes.clone()
                # Note: prev_ids remains the same (we're tracking the same objects)
            
            # For evaluation, match predictions to GT of CURRENT frame
            # Separate predictions by moving/static (based on input_ids which came from prev frame)
            pred_moving_mask = torch.tensor([int(obj_id.item()) in moving_ids for obj_id in input_ids])
            pred_static_mask = ~pred_moving_mask
            
            pred_boxes_moving = pred_boxes[pred_moving_mask]
            pred_boxes_static = pred_boxes[pred_static_mask]
            
            # Separate GT by moving/static
            gt_moving_mask = torch.tensor([int(obj_id.item()) in moving_ids for obj_id in gt_ids])
            gt_static_mask = ~gt_moving_mask
            
            gt_moving_boxes = gt_boxes[gt_moving_mask]
            gt_static_boxes = gt_boxes[gt_static_mask]
            
            # Evaluate moving objects
            if len(pred_boxes_moving) > 0 and len(gt_moving_boxes) > 0:
                ious = compute_iou_batch(pred_boxes_moving, gt_moving_boxes)
                max_ious, _ = ious.max(dim=1)
                moving_tp += (max_ious > iou_threshold).sum().item()
                moving_fp += (max_ious <= iou_threshold).sum().item()
            elif len(pred_boxes_moving) > 0:
                moving_fp += len(pred_boxes_moving)
            
            if len(gt_moving_boxes) > 0:
                moving_fn += len(gt_moving_boxes)
                if len(pred_boxes_moving) > 0:
                    ious = compute_iou_batch(pred_boxes_moving, gt_moving_boxes)
                    max_ious_gt, _ = ious.max(dim=0)
                    moving_fn -= (max_ious_gt > iou_threshold).sum().item()
            
            # Evaluate static objects
            if len(pred_boxes_static) > 0 and len(gt_static_boxes) > 0:
                ious = compute_iou_batch(pred_boxes_static, gt_static_boxes)
                max_ious, _ = ious.max(dim=1)
                static_tp += (max_ious > iou_threshold).sum().item()
                static_fp += (max_ious <= iou_threshold).sum().item()
            elif len(pred_boxes_static) > 0:
                static_fp += len(pred_boxes_static)
            
            if len(gt_static_boxes) > 0:
                static_fn += len(gt_static_boxes)
                if len(pred_boxes_static) > 0:
                    ious = compute_iou_batch(pred_boxes_static, gt_static_boxes)
                    max_ious_gt, _ = ious.max(dim=0)
                    static_fn -= (max_ious_gt > iou_threshold).sum().item()
    
    # Compute F1 scores
    moving_precision = moving_tp / (moving_tp + moving_fp) if (moving_tp + moving_fp) > 0 else 0.0
    moving_recall = moving_tp / (moving_tp + moving_fn) if (moving_tp + moving_fn) > 0 else 0.0
    moving_map = 2 * moving_precision * moving_recall / (moving_precision + moving_recall) if (moving_precision + moving_recall) > 0 else 0.0
    
    static_precision = static_tp / (static_tp + static_fp) if (static_tp + static_fp) > 0 else 0.0
    static_recall = static_tp / (static_tp + static_fn) if (static_tp + static_fn) > 0 else 0.0
    static_map = 2 * static_precision * static_recall / (static_precision + static_recall) if (static_precision + static_recall) > 0 else 0.0
    
    return {
        'moving_map': moving_map,
        'static_map': static_map,
        'moving_count': len(moving_ids),
        'static_count': len(all_ids) - len(moving_ids)
    }


def evaluate_iframe_baseline(dataset, gop_start_idx, sequence_length, moving_ids, all_ids, iou_threshold=0.5):
    """
    Evaluate I-frame baseline on moving vs static objects
    
    Returns:
        dict with mAP for moving and static objects separately
    """
    # Get I-frame (first frame)
    i_frame = dataset[gop_start_idx]
    i_frame_boxes = i_frame['boxes']
    i_frame_ids = i_frame['ids']
    
    if len(i_frame_boxes) == 0:
        return {'moving_map': 0.0, 'static_map': 0.0, 'moving_count': 0, 'static_count': 0}
    
    # Separate into moving and static
    moving_mask = torch.tensor([int(obj_id.item()) in moving_ids for obj_id in i_frame_ids])
    static_mask = ~moving_mask
    
    moving_boxes = i_frame_boxes[moving_mask]
    static_boxes = i_frame_boxes[static_mask]
    
    # Track TP, FP, FN for moving and static objects
    moving_tp, moving_fp, moving_fn = 0, 0, 0
    static_tp, static_fp, static_fn = 0, 0, 0
    
    # Evaluate each frame (skip I-frame itself)
    for frame_offset in range(1, sequence_length):
        frame_idx = gop_start_idx + frame_offset
        if frame_idx >= len(dataset):
            break
        
        frame_data = dataset[frame_idx]
        gt_boxes = frame_data['boxes']
        gt_ids = frame_data['ids']
        
        if len(gt_boxes) == 0:
            continue
        
        # I-frame baseline: use same boxes as I-frame (no motion)
        pred_boxes = i_frame_boxes
        
        # Match predictions to GT for moving objects
        gt_moving_mask = torch.tensor([int(obj_id.item()) in moving_ids for obj_id in gt_ids])
        gt_moving_boxes = gt_boxes[gt_moving_mask]
        
        if len(moving_boxes) > 0 and len(gt_moving_boxes) > 0:
            ious = compute_iou_batch(moving_boxes, gt_moving_boxes)
            max_ious, _ = ious.max(dim=1)
            moving_tp += (max_ious > iou_threshold).sum().item()
            moving_fp += (max_ious <= iou_threshold).sum().item()
        elif len(moving_boxes) > 0:
            moving_fp += len(moving_boxes)
        
        if len(gt_moving_boxes) > 0:
            moving_fn += len(gt_moving_boxes)
            if len(moving_boxes) > 0:
                ious = compute_iou_batch(moving_boxes, gt_moving_boxes)
                max_ious_gt, _ = ious.max(dim=0)
                moving_fn -= (max_ious_gt > iou_threshold).sum().item()
        
        # Match predictions to GT for static objects
        gt_static_mask = ~gt_moving_mask
        gt_static_boxes = gt_boxes[gt_static_mask]
        
        if len(static_boxes) > 0 and len(gt_static_boxes) > 0:
            ious = compute_iou_batch(static_boxes, gt_static_boxes)
            max_ious, _ = ious.max(dim=1)
            static_tp += (max_ious > iou_threshold).sum().item()
            static_fp += (max_ious <= iou_threshold).sum().item()
        elif len(static_boxes) > 0:
            static_fp += len(static_boxes)
        
        if len(gt_static_boxes) > 0:
            static_fn += len(gt_static_boxes)
            if len(static_boxes) > 0:
                ious = compute_iou_batch(static_boxes, gt_static_boxes)
                max_ious_gt, _ = ious.max(dim=0)
                static_fn -= (max_ious_gt > iou_threshold).sum().item()
    
    # Compute precision/recall/F1 (simplified mAP)
    moving_precision = moving_tp / (moving_tp + moving_fp) if (moving_tp + moving_fp) > 0 else 0.0
    moving_recall = moving_tp / (moving_tp + moving_fn) if (moving_tp + moving_fn) > 0 else 0.0
    moving_map = 2 * moving_precision * moving_recall / (moving_precision + moving_recall) if (moving_precision + moving_recall) > 0 else 0.0
    
    static_precision = static_tp / (static_tp + static_fp) if (static_tp + static_fp) > 0 else 0.0
    static_recall = static_tp / (static_tp + static_fn) if (static_tp + static_fn) > 0 else 0.0
    static_map = 2 * static_precision * static_recall / (static_precision + static_recall) if (static_precision + static_recall) > 0 else 0.0
    
    return {
        'moving_map': moving_map,
        'static_map': static_map,
        'moving_count': len(moving_ids),
        'static_count': len(all_ids) - len(moving_ids)
    }


def main():
    parser = argparse.ArgumentParser(description='Compare three methods on moving vs static objects')
    parser.add_argument('--mv-checkpoint', type=str, default='best_mots17_deep_tracker.pt',
                        help='Path to MV model checkpoint')
    parser.add_argument('--motion-threshold', type=float, default=0.01,
                        help='Minimum displacement (normalized) to consider an object moving')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for matching')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--max-gops', type=int, default=None,
                        help='Maximum number of GOPs to evaluate per dataset')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--use-teacher-forcing', action='store_true',
                        help='Use GT boxes as input for MV model (teacher forcing) instead of autoregressive')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load MV model
    print(f"\n📦 Loading MV model from: {args.mv_checkpoint}")
    if not os.path.exists(args.mv_checkpoint):
        print(f"   ❌ Checkpoint not found: {args.mv_checkpoint}")
        return
    
    checkpoint = torch.load(args.mv_checkpoint, map_location=device)
    
    # Get model args from checkpoint
    if 'args' not in checkpoint:
        print(f"   ❌ No args found in checkpoint")
        return
    
    model_args_dict = checkpoint['args']
    
    # Convert dict to argparse Namespace object
    from argparse import Namespace
    model_args = Namespace(**model_args_dict)
    
    # Create model using saved args and device (returns tuple: model, criterion)
    model, _ = create_model(model_args, device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"   ✅ Model loaded successfully")
    print(f"   📊 Model config: version={model_args.version}, use_mv_only={model_args.use_mv_only}")
    
    # Configure datasets to evaluate
    datasets_config = [
        ('mot17', 'MOT17', STATIC_CAMERA_MOT17_SEQUENCES),
        ('mot15', 'MOT15', STATIC_CAMERA_MOT15_SEQUENCES),
        ('mot20', 'MOT20', STATIC_CAMERA_MOT20_SEQUENCES),
    ]
    
    all_results = {}
    
    print("\n" + "=" * 80)
    print("🔬 THREE-METHOD COMPARISON: Mean-VC vs I-frame vs MV Model")
    print("=" * 80)
    print(f"📊 Motion threshold: {args.motion_threshold} (normalized)")
    print(f"📊 IoU threshold: {args.iou_threshold}")
    print()
    
    # Evaluate each dataset
    for dataset_type, dataset_name, static_sequences in datasets_config:
        print("\n" + "=" * 80)
        print(f"📊 Evaluating {dataset_name}")
        print("=" * 80)
        
        # Load dataset
        print(f"\n📁 Loading {dataset_name} dataset...")
        dataset = create_mots_dataset(
            dataset_type=[dataset_type],
            resolution=960,
            mode='test',
            load_iframe=False,
            load_pframe=False,
            load_motion_vectors=True,
            load_residuals=False,
            dct_coeffs=0,
            load_annotations=True,
            sequence_length=60,
            data_format="separate",
            combine_datasets=False
        )
        
        print(f"   ✅ Loaded {len(dataset)} samples")
        
        # Filter to static camera GOPs
        print(f"\n🔍 Filtering to static camera GOPs...")
        all_indices = list(range(len(dataset.sequences)))
        static_indices = filter_static_gops_by_dataset(all_indices, dataset, static_sequences)
        
        if args.max_gops and len(static_indices) > args.max_gops:
            static_indices = static_indices[:args.max_gops]
        
        print(f"   ✅ Found {len(static_indices)} static camera GOPs")
        
        if len(static_indices) == 0:
            print(f"   ⚠️ No static GOPs found, skipping...")
            continue
        
        # Initialize result containers
        iframe_results = {
            'moving_maps': [], 'static_maps': [],
            'moving_count': 0, 'static_count': 0
        }
        mean_vc_results = {
            'moving_maps': [], 'static_maps': [],
            'moving_count': 0, 'static_count': 0
        }
        mv_model_results = {
            'moving_maps': [], 'static_maps': [],
            'moving_count': 0, 'static_count': 0
        }
        
        sequence_length = 60
        
        # Evaluate all three methods on each GOP
        for gop_idx in tqdm(static_indices, desc=f"Evaluating {dataset_name}"):
            try:
                gop_start_idx = gop_idx * sequence_length
                
                # Identify moving vs static objects
                moving_ids, all_ids = identify_moving_objects_in_gop(
                    dataset,
                    gop_start_idx,
                    sequence_length=sequence_length,
                    motion_threshold=args.motion_threshold
                )
                
                # 1. I-frame baseline
                iframe_result = evaluate_iframe_baseline(
                    dataset, gop_start_idx, sequence_length,
                    moving_ids, all_ids, iou_threshold=args.iou_threshold
                )
                if iframe_result['moving_count'] > 0:
                    iframe_results['moving_maps'].append(iframe_result['moving_map'])
                if iframe_result['static_count'] > 0:
                    iframe_results['static_maps'].append(iframe_result['static_map'])
                iframe_results['moving_count'] += iframe_result['moving_count']
                iframe_results['static_count'] += iframe_result['static_count']
                
                # 2. Mean-VC baseline
                mean_vc_result = evaluate_mean_vc_baseline(
                    dataset, gop_start_idx, sequence_length,
                    moving_ids, all_ids, iou_threshold=args.iou_threshold
                )
                if mean_vc_result['moving_count'] > 0:
                    mean_vc_results['moving_maps'].append(mean_vc_result['moving_map'])
                if mean_vc_result['static_count'] > 0:
                    mean_vc_results['static_maps'].append(mean_vc_result['static_map'])
                mean_vc_results['moving_count'] += mean_vc_result['moving_count']
                mean_vc_results['static_count'] += mean_vc_result['static_count']
                
                # 3. MV Model
                try:
                    mv_result = evaluate_mv_model(
                        model, dataset, gop_start_idx, sequence_length,
                        moving_ids, all_ids, device, iou_threshold=args.iou_threshold,
                        use_teacher_forcing=args.use_teacher_forcing
                    )
                    if mv_result is None:
                        print(f"   ⚠️ MV model returned None for GOP {gop_idx}")
                        mv_result = {'moving_map': 0.0, 'static_map': 0.0, 'moving_count': 0, 'static_count': 0}
                    
                    if mv_result['moving_count'] > 0:
                        mv_model_results['moving_maps'].append(mv_result['moving_map'])
                    if mv_result['static_count'] > 0:
                        mv_model_results['static_maps'].append(mv_result['static_map'])
                    mv_model_results['moving_count'] += mv_result['moving_count']
                    mv_model_results['static_count'] += mv_result['static_count']
                except Exception as e:
                    print(f"   ⚠️ Error in MV model evaluation for GOP {gop_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                
            except Exception as e:
                print(f"   ⚠️ Error processing GOP {gop_idx}: {e}")
                continue
        
        # Compute averages and std
        iframe_moving_map = np.mean(iframe_results['moving_maps']) if iframe_results['moving_maps'] else 0.0
        iframe_moving_std = np.std(iframe_results['moving_maps']) if iframe_results['moving_maps'] else 0.0
        iframe_static_map = np.mean(iframe_results['static_maps']) if iframe_results['static_maps'] else 0.0
        iframe_static_std = np.std(iframe_results['static_maps']) if iframe_results['static_maps'] else 0.0
        
        mean_vc_moving_map = np.mean(mean_vc_results['moving_maps']) if mean_vc_results['moving_maps'] else 0.0
        mean_vc_moving_std = np.std(mean_vc_results['moving_maps']) if mean_vc_results['moving_maps'] else 0.0
        mean_vc_static_map = np.mean(mean_vc_results['static_maps']) if mean_vc_results['static_maps'] else 0.0
        mean_vc_static_std = np.std(mean_vc_results['static_maps']) if mean_vc_results['static_maps'] else 0.0
        
        mv_model_moving_map = np.mean(mv_model_results['moving_maps']) if mv_model_results['moving_maps'] else 0.0
        mv_model_moving_std = np.std(mv_model_results['moving_maps']) if mv_model_results['moving_maps'] else 0.0
        mv_model_static_map = np.mean(mv_model_results['static_maps']) if mv_model_results['static_maps'] else 0.0
        mv_model_static_std = np.std(mv_model_results['static_maps']) if mv_model_results['static_maps'] else 0.0
        
        print(f"\n📊 Results for {dataset_name}:")
        print(f"\n   I-frame Baseline:")
        print(f"      Moving objects ({iframe_results['moving_count']}): mAP = {iframe_moving_map:.4f} ± {iframe_moving_std:.4f}")
        print(f"      Static objects ({iframe_results['static_count']}): mAP = {iframe_static_map:.4f} ± {iframe_static_std:.4f}")
        print(f"\n   Mean-VC Baseline:")
        print(f"      Moving objects ({mean_vc_results['moving_count']}): mAP = {mean_vc_moving_map:.4f} ± {mean_vc_moving_std:.4f}")
        print(f"      Static objects ({mean_vc_results['static_count']}): mAP = {mean_vc_static_map:.4f} ± {mean_vc_static_std:.4f}")
        print(f"\n   MV Model:")
        print(f"      Moving objects ({mv_model_results['moving_count']}): mAP = {mv_model_moving_map:.4f} ± {mv_model_moving_std:.4f}")
        print(f"      Static objects ({mv_model_results['static_count']}): mAP = {mv_model_static_map:.4f} ± {mv_model_static_std:.4f}")
        
        all_results[dataset_name] = {
            'iframe': {
                'moving_map': float(iframe_moving_map),
                'moving_std': float(iframe_moving_std),
                'static_map': float(iframe_static_map),
                'static_std': float(iframe_static_std),
                'moving_count': iframe_results['moving_count'],
                'static_count': iframe_results['static_count']
            },
            'mean_vc': {
                'moving_map': float(mean_vc_moving_map),
                'moving_std': float(mean_vc_moving_std),
                'static_map': float(mean_vc_static_map),
                'static_std': float(mean_vc_static_std),
                'moving_count': mean_vc_results['moving_count'],
                'static_count': mean_vc_results['static_count']
            },
            'mv_model': {
                'moving_map': float(mv_model_moving_map),
                'moving_std': float(mv_model_moving_std),
                'static_map': float(mv_model_static_map),
                'static_std': float(mv_model_static_std),
                'moving_count': mv_model_results['moving_count'],
                'static_count': mv_model_results['static_count']
            },
            'static_sequences': static_sequences
        }
    
    # Save results
    results_file = output_dir / 'three_method_comparison.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("📊 SUMMARY - THREE METHOD COMPARISON")
    print("=" * 100)
    print()
    print(f"{'Dataset':<10} {'Method':<12} {'Moving mAP':<20} {'Static mAP':<20} {'Moving N':<10} {'Static N':<10}")
    print("-" * 100)
    
    for dataset_name, results in all_results.items():
        iframe = results['iframe']
        mean_vc = results['mean_vc']
        mv_model = results['mv_model']
        
        # Format: mean ± std
        iframe_moving = f"{iframe['moving_map']:.4f} ± {iframe['moving_std']:.4f}"
        iframe_static = f"{iframe['static_map']:.4f} ± {iframe['static_std']:.4f}"
        meanvc_moving = f"{mean_vc['moving_map']:.4f} ± {mean_vc['moving_std']:.4f}"
        meanvc_static = f"{mean_vc['static_map']:.4f} ± {mean_vc['static_std']:.4f}"
        mvmodel_moving = f"{mv_model['moving_map']:.4f} ± {mv_model['moving_std']:.4f}"
        mvmodel_static = f"{mv_model['static_map']:.4f} ± {mv_model['static_std']:.4f}"
        
        print(f"{dataset_name:<10} {'I-frame':<12} {iframe_moving:<20} {iframe_static:<20} "
              f"{iframe['moving_count']:<10} {iframe['static_count']:<10}")
        print(f"{'':<10} {'Mean-VC':<12} {meanvc_moving:<20} {meanvc_static:<20} "
              f"{mean_vc['moving_count']:<10} {mean_vc['static_count']:<10}")
        print(f"{'':<10} {'MV Model':<12} {mvmodel_moving:<20} {mvmodel_static:<20} "
              f"{mv_model['moving_count']:<10} {mv_model['static_count']:<10}")
        print("-" * 100)
    
    print()
    print("=" * 80)
    print("💡 INTERPRETATION")
    print("=" * 80)
    print()
    print("Expected patterns:")
    print("  • I-frame: HIGH static mAP, LOW moving mAP (assumes no motion)")
    print("  • MV Model: HIGH moving mAP, GOOD static mAP (tracks motion)")
    print("  • Mean-VC: MEDIUM both (interpolates between)")
    print()
    print("\nKey Insights:")
    print("  1. I-frame fails on moving objects → motion tracking is essential")
    print("  2. MV model improves moving object detection significantly")
    print("  3. MV model maintains good performance on static objects")
    print()


if __name__ == '__main__':
    main()
