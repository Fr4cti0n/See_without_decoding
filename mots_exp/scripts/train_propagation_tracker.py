"""
Training script for Propagation-Focused Tracker with Fixed-Size Output

Key Features:
1. Fixed 300 box output slots (solves LSTM hidden state problem)
2. Track ID preservation with tracking loss
3. Hungarian matching for GT-prediction association by track ID
4. Propagation focus: predict frame N boxes given frame N-1 boxes + frame N motion vectors
5. Handles track birth/death gracefully

Author: Motion Tracking Team
Date: Oct 30, 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from mots_exp.models.tracking_propagator import TrackingPropagator
from mots_exp.models.rtdetr_loss import RTDETRTrackingLoss  # RT-DETR loss
from dataset.factory.dataset_factory import create_mots_dataset
from scipy.optimize import linear_sum_assignment
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F


def load_gop_sequences_once(dataset, gop_length=50, max_sequences=None):
    """
    Load GOP sequences from the dataset ONCE at the start of training.
    Groups individual frames into complete GOP sequences.
    
    Args:
        dataset: MOTSSequenceDataset or ConcatDataset of MOTSSequenceDataset
        gop_length: Length of each GOP in frames (default 50)
        max_sequences: Optional maximum TOTAL number of GOPs to load across all datasets
    
    Returns:
        Ordered list of (seq_id, frames) tuples that can be reused across epochs.
    """
    print(f"\n📦 Loading GOP sequences once for all epochs...", flush=True)
    if max_sequences is not None:
        print(f"   🎯 Limiting to {max_sequences} GOPs total (for RAM efficiency)")
    
    # Handle ConcatDataset (multiple datasets combined)
    if hasattr(dataset, 'datasets'):
        # ConcatDataset: iterate through each sub-dataset
        print(f"   Detected ConcatDataset with {len(dataset.datasets)} sub-datasets")
        all_gops = []
        remaining_quota = max_sequences if max_sequences is not None else float('inf')
        
        for ds_idx, sub_dataset in enumerate(dataset.datasets):
            if remaining_quota <= 0:
                print(f"   Reached GOP limit, skipping remaining datasets")
                break
            
            # Calculate how many GOPs to load from this dataset
            per_dataset_limit = min(remaining_quota, len(sub_dataset.sequences)) if max_sequences is not None else None
            
            print(f"   Loading GOPs from dataset {ds_idx + 1}/{len(dataset.datasets)} (max: {per_dataset_limit if per_dataset_limit else 'all'})...")
            sub_gops = _load_gops_from_single_dataset(
                sub_dataset, gop_length, per_dataset_limit
            )
            all_gops.extend(sub_gops)
            
            if max_sequences is not None:
                remaining_quota -= len(sub_gops)
        
        print(f"\n✅ Total GOPs loaded: {len(all_gops)}")
        return all_gops
    else:
        # Single dataset
        return _load_gops_from_single_dataset(dataset, gop_length, max_sequences)


def _load_gops_from_single_dataset(dataset, gop_length=50, max_sequences=None):
    """Helper function to load GOPs from a single dataset."""
    # Determine number of complete GOPs in dataset
    num_sequences = len(dataset.sequences)
    if max_sequences is not None:
        num_sequences = min(num_sequences, max_sequences)
    
    print(f"   Found {num_sequences} GOP sequences to load")
    
    # Load complete GOPs in parallel using threads (IO-bound)
    gop_sequences = {}
    seq_index_to_id = {}
    
    def _load_sequence(seq_idx):
        """Worker function to load one complete GOP sequence (preserving frame order)"""
        sequence_info = dataset.sequences[seq_idx]
        sequence_id = f"{sequence_info['video_name']}_gop{sequence_info['gop_index']}"
        frames = []
        
        for frame_idx in range(gop_length):
            global_idx = seq_idx * gop_length + frame_idx
            
            # Check if index is within dataset bounds
            if global_idx >= len(dataset):
                break
                
            try:
                sample = dataset[global_idx]
                mv = sample.get('motion_vectors')
                boxes = sample.get('boxes')
                ids = sample.get('ids')
                
                # Skip frames without motion vectors or boxes
                if mv is None or boxes is None or len(boxes) == 0:
                    continue
                
                frame_data = {
                    'motion_vectors': mv,
                    'boxes': boxes,
                    'ids': ids if ids is not None else torch.zeros(len(boxes), dtype=torch.long),
                    'frame_id': frame_idx
                }
                frames.append(frame_data)
            except Exception as e:
                print(f"Warning: Failed to load frame {frame_idx} from GOP {seq_idx}: {e}")
                continue
        
        return seq_idx, sequence_id, frames
    
    # Parallel loading with thread pool
    max_workers = min(8, (os.cpu_count() or 4))
    futures = {}
    seq_bar = tqdm(total=num_sequences, desc="Loading GOPs", unit="GOP")
    
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for seq_idx in range(num_sequences):
            futures[exe.submit(_load_sequence, seq_idx)] = seq_idx
        
        for fut in as_completed(futures):
            seq_idx, sequence_id, frames = fut.result()
            seq_index_to_id[seq_idx] = sequence_id
            
            # Only keep GOPs with at least 2 frames (need for sequential training)
            if len(frames) >= 2:
                gop_sequences[sequence_id] = frames
            
            seq_bar.update(1)
            seq_bar.set_postfix({'loaded': len(gop_sequences), 'frames': len(frames)})
    
    seq_bar.close()
    
    # Preserve original sequence order
    ordered_gops = []
    for seq_idx in range(num_sequences):
        seq_id = seq_index_to_id.get(seq_idx)
        if seq_id is not None and seq_id in gop_sequences:
            ordered_gops.append((seq_id, gop_sequences[seq_id]))
    
    print(f"✅ Loaded {len(ordered_gops)} complete GOPs (will be reused for all epochs)", flush=True)
    
    # Print GOP statistics
    gop_lengths = [len(frames) for _, frames in ordered_gops]
    if gop_lengths:
        print(f"   📊 GOP lengths: min={min(gop_lengths)}, max={max(gop_lengths)}, avg={sum(gop_lengths)/len(gop_lengths):.1f}")
        print(f"   🎯 GOPs with >= 2 frames: {sum(1 for l in gop_lengths if l >= 2)}")
    
    return ordered_gops


def select_confident_predictions(pred_boxes, pred_class_logits, pred_track_id_logits, 
                                 confidence_threshold=0.3, max_boxes=100):
    """
    Select confident predictions for autoregressive propagation.
    This prevents feeding garbage/drifted boxes back into the LSTM.
    
    Args:
        pred_boxes: [300, 4] - predicted boxes
        pred_class_logits: [300, 2] - class logits [no_object, pedestrian]
        pred_track_id_logits: [300, num_track_ids] - track ID logits
        confidence_threshold: Minimum p(pedestrian) to keep (default: 0.3)
        max_boxes: Maximum number of boxes to keep (default: 100)
    
    Returns:
        filtered_boxes: [K, 4] where K <= max_boxes
        filtered_track_ids: [K] - predicted track IDs
    """
    # Get pedestrian class probabilities
    class_probs = F.softmax(pred_class_logits, dim=-1)  # [300, 2]
    pedestrian_probs = class_probs[:, 1]  # [300]
    
    # Filter by confidence threshold
    confident_mask = pedestrian_probs > confidence_threshold
    confident_indices = confident_mask.nonzero(as_tuple=True)[0]
    
    if len(confident_indices) == 0:
        # No confident predictions - return empty
        return torch.zeros(0, 4, device=pred_boxes.device), torch.zeros(0, dtype=torch.long, device=pred_boxes.device)
    
    # Select confident predictions
    confident_boxes = pred_boxes[confident_indices]
    confident_probs = pedestrian_probs[confident_indices]
    confident_track_logits = pred_track_id_logits[confident_indices]
    
    # If too many, keep top-K by confidence
    if len(confident_indices) > max_boxes:
        topk_probs, topk_indices = torch.topk(confident_probs, k=max_boxes)
        confident_boxes = confident_boxes[topk_indices]
        confident_track_logits = confident_track_logits[topk_indices]
    
    # Get predicted track IDs
    predicted_track_ids = confident_track_logits.argmax(dim=-1)
    
    return confident_boxes, predicted_track_ids


def compute_box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes: [N, 4] in format [cx, cy, w, h] (normalized 0-1)
    Returns: [N, M] IoU matrix
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2))
    
    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    boxes1_xyxy = torch.zeros_like(boxes1)
    boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2  # x1
    boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2  # y1
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2  # x2
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2  # y2
    
    boxes2_xyxy = torch.zeros_like(boxes2)
    boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Compute intersection
    x1 = torch.max(boxes1_xyxy[:, None, 0], boxes2_xyxy[None, :, 0])
    y1 = torch.max(boxes1_xyxy[:, None, 1], boxes2_xyxy[None, :, 1])
    x2 = torch.min(boxes1_xyxy[:, None, 2], boxes2_xyxy[None, :, 2])
    y2 = torch.min(boxes1_xyxy[:, None, 3], boxes2_xyxy[None, :, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # Compute areas
    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1])
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1])
    
    # Compute IoU
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / (union + 1e-6)
    
    return iou


def compute_map50(pred_boxes, pred_scores, target_boxes, iou_threshold=0.5):
    """
    Compute mAP@0.5 for a single frame
    
    Args:
        pred_boxes: [N, 4] predicted boxes
        pred_scores: [N] confidence scores
        target_boxes: [M, 4] ground truth boxes
        iou_threshold: IoU threshold for matching (default 0.5)
    
    Returns:
        ap50: Average Precision at IoU=0.5
    """
    if len(target_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by confidence (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Compute IoU matrix
    iou_matrix = compute_box_iou(pred_boxes, target_boxes)
    
    # Track which GT boxes have been matched
    gt_matched = torch.zeros(len(target_boxes), dtype=torch.bool)
    
    # Track true positives and false positives
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    
    for i in range(len(pred_boxes)):
        # Find best matching GT box
        ious = iou_matrix[i]
        max_iou, max_idx = torch.max(ious, dim=0)
        
        if max_iou >= iou_threshold and not gt_matched[max_idx]:
            tp[i] = 1
            gt_matched[max_idx] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall at each threshold
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recall = tp_cumsum / len(target_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        if torch.any(recall >= t):
            ap += torch.max(precision[recall >= t])
    ap /= 11.0
    
    return ap.item()


def compute_map_over_sequence(model, ordered_gops, device, max_frames_list=[6, 12, 50]):
    """
    Compute mAP@0.5 over sequences of different lengths
    
    Args:
        model: TrackingPropagator model
        ordered_gops: List of (seq_id, frames) tuples
        device: Device
        max_frames_list: List of maximum frame counts to evaluate
    
    Returns:
        dict: mAP scores for each sequence length
    """
    model.eval()
    
    results = {f'map50_{n}frames': [] for n in max_frames_list}
    
    with torch.no_grad():
        for seq_id, gop_frames in tqdm(ordered_gops, desc="Computing mAP"):
            if len(gop_frames) < 2:
                continue
            
            # Reset LSTM for new sequence
            hidden_state = None
            
            # Initialize with I-frame - Keep normalized [0,1] coordinates
            current_boxes = gop_frames[0]['boxes'].to(device)
            current_ids = gop_frames[0]['ids'].to(device)
            
            if len(current_boxes) == 0:
                continue
            
            # Process frames and compute mAP at different checkpoints
            for max_frames in max_frames_list:
                num_frames = min(len(gop_frames), max_frames)
                frame_aps = []
                temp_hidden = hidden_state
                temp_boxes = current_boxes.clone()
                temp_ids = current_ids.clone()
                
                for t in range(1, num_frames):
                    # Get motion vectors
                    mv = gop_frames[t]['motion_vectors'].to(device)
                    target_boxes = gop_frames[t]['boxes'].to(device)
                    
                    if len(target_boxes) == 0:
                        continue
                    
                    # Process motion vectors
                    while mv.ndim > 4:
                        mv = mv.squeeze(0)
                    if mv.ndim == 4:
                        if mv.shape[0] == 2 and mv.shape[-1] != 2:
                            pass
                        else:
                            mv = mv.mean(dim=0)
                    if mv.ndim == 3:
                        if mv.shape[-1] == 2:
                            mv = mv.permute(2, 0, 1)
                    if mv.ndim == 3:
                        mv = mv.unsqueeze(0)
                    
                    # Forward pass
                    pred_boxes, pred_class_logits, pred_ids_logits, temp_hidden = model(
                        mv, None, temp_boxes, temp_ids, temp_hidden
                    )
                    
                    # Get confidence scores (probability of pedestrian class)
                    pred_scores = torch.softmax(pred_class_logits, dim=-1)[:, 1]  # P(pedestrian)
                    
                    # Compute mAP for this frame
                    ap = compute_map50(pred_boxes, pred_scores, target_boxes)
                    frame_aps.append(ap)
                    
                    # Update for next iteration
                    temp_boxes, temp_ids = select_confident_predictions(
                        pred_boxes, pred_class_logits, pred_ids_logits,
                        confidence_threshold=0.3, max_boxes=100
                    )
                
                # Average mAP over frames in this sequence
                if len(frame_aps) > 0:
                    results[f'map50_{max_frames}frames'].append(np.mean(frame_aps))
    
    # Average over all sequences
    final_results = {}
    for key, values in results.items():
        if len(values) > 0:
            final_results[key] = np.mean(values)
        else:
            final_results[key] = 0.0
    
    return final_results


def visualize_predictions_grid(model, datasets_config, device, output_dir, epoch, num_frames=6):
    """
    Create visualization grid showing GT vs Predictions for sample GOPs
    Loads images from each dataset to visualize tracking quality
    
    Args:
        model: TrackingPropagator model
        datasets_config: List of dataset names to sample from
        device: Device
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        num_frames: Number of frames to visualize per GOP (default 6)
    """
    import cv2
    from matplotlib.patches import Rectangle
    
    print(f"\n🎨 Creating prediction visualizations (epoch {epoch})...")
    
    model.eval()
    output_dir = Path(output_dir)
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Create JSON output directory for detailed predictions
    json_dir = output_dir / 'prediction_logs'
    json_dir.mkdir(exist_ok=True)
    
    # Dictionary to store all predictions for JSON export
    prediction_data = {
        'epoch': epoch,
        'datasets': {}
    }
    
    # Load datasets WITH images (iframe=True, pframe=True)
    sample_gops = []
    
    for dataset_name in datasets_config:
        print(f"  Loading sample GOP from {dataset_name}...")
        
        # Create dataset with images enabled (use test split like validation)
        dataset = create_mots_dataset(
            dataset_type=dataset_name,
            resolution=960,
            mode='test',  # Use test split (same as validation)
            load_iframe=True,  # Load I-frames
            load_pframe=True,  # Load P-frames
            load_motion_vectors=True,  # Still need motion vectors
            load_residuals=False,
            dct_coeffs=0,
            load_annotations=True,
            sequence_length=50,
            data_format='separate',
            combine_datasets=False
        )
        
        # Load just 1 GOP from this dataset WITH images
        # The dataset returns individual frames, so collect 50 consecutive frames for 1 GOP
        if len(dataset) >= 50:
            gop_frames = []
            
            # Collect frames for first GOP (frames 0-49)
            for frame_idx in range(min(50, len(dataset))):
                sample = dataset[frame_idx]
                
                # Extract frame data
                frame_data = {
                    'iframe': sample.get('iframe') if frame_idx == 0 else None,
                    'pframe': sample.get('pframe') if frame_idx > 0 else None,
                    'motion_vectors': sample.get('motion_vectors'),
                    'boxes': sample.get('boxes', torch.tensor([])),
                    'ids': sample.get('ids', torch.tensor([])),
                    'frame_id': frame_idx
                }
                gop_frames.append(frame_data)
            
            # Only add if we got frames with images
            if len(gop_frames) > 0 and gop_frames[0]['iframe'] is not None:
                sample_gops.append((dataset_name, ('seq_0', gop_frames)))
                print(f"    ✅ Loaded 1 GOP with {len(gop_frames)} frames")
            else:
                print(f"    ⚠️ No valid frames with images for {dataset_name}")
        else:
            print(f"    ⚠️ Not enough frames in dataset for {dataset_name}")
    
    if len(sample_gops) == 0:
        print("  ⚠️ No GOPs loaded for visualization")
        return
    
    # Create visualization grid
    num_gops = len(sample_gops)
    fig, axes = plt.subplots(num_gops, num_frames, figsize=(num_frames * 4, num_gops * 4))
    
    if num_gops == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for gop_idx, (dataset_name, (seq_id, gop_frames)) in enumerate(sample_gops):
            print(f"  Processing {dataset_name} GOP...")
            
            # Skip if no frames
            if len(gop_frames) == 0:
                print(f"    ⚠️ Skipping {dataset_name} - no frames")
                continue
            
            # Initialize prediction log for this dataset
            dataset_predictions = {
                'dataset_name': dataset_name,
                'num_gt_boxes_frame0': len(gop_frames[0]['boxes']),
                'frames': []
            }
            
            # Reset LSTM for new sequence
            hidden_state = None
            
            # Initialize with I-frame - Keep normalized [0,1] coordinates
            current_boxes = gop_frames[0]['boxes'].to(device)
            current_ids = gop_frames[0]['ids'].to(device)
            
            # Visualize first frame (I-frame with GT only)
            frame_idx = 0
            ax = axes[gop_idx, frame_idx]
            
            # Get I-frame image
            iframe_img = gop_frames[0]['iframe'].cpu().numpy()  # [1, H, W, C] or [H, W, C]
            # Remove batch dimension if present
            while len(iframe_img.shape) > 3:
                iframe_img = iframe_img.squeeze(0)
            # Now shape is [H, W, C]
            
            # Convert from [0, 255] to [0, 1] (images are NOT pre-normalized)
            iframe_img = iframe_img / 255.0
            iframe_img = np.clip(iframe_img, 0, 1)
            
            ax.imshow(iframe_img)
            
            # Draw GT boxes (green) - draw from ORIGINAL normalized boxes, not current_boxes
            h, w = iframe_img.shape[:2]
            gt_boxes_norm = gop_frames[0]['boxes'].cpu().numpy()  # Get original normalized boxes
            for box in gt_boxes_norm:
                cx, cy, bw, bh = box  # Normalized [0, 1]
                # Convert to pixel coordinates for drawing
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                box_w = bw * w
                box_h = bh * h
                rect = Rectangle((x1, y1), box_w, box_h, 
                                linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            
            ax.set_title(f'{dataset_name}\nFrame 0 (I-frame, GT)', fontsize=10)
            ax.axis('off')
            
            # Process P-frames
            for frame_idx in range(1, min(num_frames, len(gop_frames))):
                ax = axes[gop_idx, frame_idx]
                
                # Get P-frame data
                mv = gop_frames[frame_idx]['motion_vectors'].to(device)
                target_boxes = gop_frames[frame_idx]['boxes'].to(device)
                target_ids = gop_frames[frame_idx]['ids'].to(device)
                pframe_img = gop_frames[frame_idx]['pframe'].cpu().numpy()  # [1, H, W, C] or [H, W, C]
                
                # Remove batch dimension if present
                while len(pframe_img.shape) > 3:
                    pframe_img = pframe_img.squeeze(0)
                # Now shape is [H, W, C]
                
                # Convert from [0, 255] to [0, 1]
                pframe_img = pframe_img / 255.0
                pframe_img = np.clip(pframe_img, 0, 1)
                
                # Process motion vectors
                while mv.ndim > 4:
                    mv = mv.squeeze(0)
                if mv.ndim == 4:
                    if mv.shape[0] == 2 and mv.shape[-1] != 2:
                        pass
                    else:
                        mv = mv.mean(dim=0)
                if mv.ndim == 3:
                    if mv.shape[-1] == 2:
                        mv = mv.permute(2, 0, 1)
                if mv.ndim == 3:
                    mv = mv.unsqueeze(0)
                
                # Forward pass
                pred_boxes, pred_class_logits, pred_ids_logits, hidden_state = model(
                    mv, None, current_boxes, current_ids, hidden_state
                )
                
                # Debug: Print prediction info for first frame
                if frame_idx == 1:
                    print(f"    Frame {frame_idx}: pred_boxes shape={pred_boxes.shape}, "
                          f"pred_class_logits shape={pred_class_logits.shape}")
                    if len(pred_boxes) > 0:
                        print(f"    pred_boxes range: [{pred_boxes.min().item():.3f}, {pred_boxes.max().item():.3f}]")
                        pred_scores_debug = torch.softmax(pred_class_logits, dim=-1)
                        print(f"    pred_scores shape={pred_scores_debug.shape}, max={pred_scores_debug.max().item():.3f}")
                        print(f"    pred_scores[:, 1] (pedestrian class) max={pred_scores_debug[:, 1].max().item():.3f}")
                
                # Model outputs boxes in NORMALIZED [0,1] coordinates
                # Get confident predictions (very low threshold for propagation-based model)
                pred_scores = torch.softmax(pred_class_logits, dim=-1)[:, 1]  # P(pedestrian)
                confident_mask = pred_scores > 0.01  # Very low threshold - model propagates boxes, doesn't detect
                confident_boxes_norm = pred_boxes[confident_mask]  # Normalized [0,1]
                
                # Scale to pixels for JSON logging and visualization
                h_img, w_img = pframe_img.shape[:2]
                pred_boxes_pixel = pred_boxes.clone()
                pred_boxes_pixel[:, 0] = pred_boxes[:, 0] * w_img  # cx
                pred_boxes_pixel[:, 1] = pred_boxes[:, 1] * h_img  # cy
                pred_boxes_pixel[:, 2] = pred_boxes[:, 2] * w_img  # w
                pred_boxes_pixel[:, 3] = pred_boxes[:, 3] * h_img  # h
                confident_boxes_pixel = pred_boxes_pixel[confident_mask]
                
                # Log predictions to JSON
                frame_data = {
                    'frame_idx': frame_idx,
                    'num_predictions': len(pred_boxes),
                    'num_gt_boxes': len(target_boxes),
                    'predictions': {
                        'boxes_pixel': pred_boxes_pixel.cpu().tolist(),
                        'boxes_normalized': pred_boxes.cpu().tolist(),
                        'scores_pedestrian': pred_scores.cpu().tolist(),
                        'class_logits': pred_class_logits.cpu().tolist(),
                        'num_confident': confident_mask.sum().item(),
                        'confident_boxes_pixel': confident_boxes_pixel.cpu().tolist(),
                    },
                    'statistics': {
                        'score_max': pred_scores.max().item(),
                        'score_min': pred_scores.min().item(),
                        'score_mean': pred_scores.mean().item(),
                        'num_input_boxes': len(current_boxes),
                    }
                }
                dataset_predictions['frames'].append(frame_data)
                
                if frame_idx == 1:
                    print(f"    Pedestrian scores max={pred_scores.max().item():.3f}, "
                          f"confident predictions (>0.01): {confident_mask.sum().item()} / {len(pred_scores)}")
                
                # Display P-frame
                ax.imshow(pframe_img)
                
                # Draw GT boxes (green) - boxes are in normalized [0,1] format
                h, w = pframe_img.shape[:2]
                for box in target_boxes.cpu().numpy():
                    cx, cy, bw, bh = box  # Already normalized [0, 1]
                    # Convert to pixel coordinates
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    box_w = bw * w
                    box_h = bh * h
                    rect = Rectangle((x1, y1), box_w, box_h,
                                    linewidth=2, edgecolor='green', facecolor='none',
                                    label='GT' if box is target_boxes[0] else '')
                    ax.add_patch(rect)
                
                # Draw predicted boxes (red) - predictions are in NORMALIZED [0,1] coordinates
                for box in confident_boxes_norm.cpu().numpy():
                    cx, cy, bw, bh = box  # Normalized [0,1]
                    # Convert to pixel coordinates for drawing
                    x1 = (cx - bw/2) * w
                    y1 = (cy - bh/2) * h
                    box_w = bw * w
                    box_h = bh * h
                    rect = Rectangle((x1, y1), box_w, box_h,
                                    linewidth=2, edgecolor='red', facecolor='none',
                                    linestyle='--',
                                    label='Pred' if len(confident_boxes_norm) > 0 and (box == confident_boxes_norm[0].cpu().numpy()).all() else '')
                    ax.add_patch(rect)
                
                ax.set_title(f'Frame {frame_idx}\nGT:{len(target_boxes)} Pred:{len(confident_boxes_norm)}',
                           fontsize=10)
                ax.axis('off')
                
                # Update for next iteration (use very low threshold for propagation)
                current_boxes, current_ids = select_confident_predictions(
                    pred_boxes, pred_class_logits, pred_ids_logits,
                    confidence_threshold=0.01,  # Very low threshold - model propagates, doesn't detect
                    max_boxes=100
                )
            
            # Store dataset predictions
            prediction_data['datasets'][dataset_name] = dataset_predictions
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12,
              bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save visualization
    vis_path = vis_dir / f'predictions_epoch_{epoch:03d}.png'
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Visualization saved to: {vis_path}")
    
    # Save JSON predictions
    import json
    json_path = json_dir / f'epoch_{epoch:03d}.json'
    with open(json_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    print(f"  ✅ Predictions JSON saved to: {json_path}")
    print(f"  📊 Total datasets logged: {len(prediction_data['datasets'])}")
    for dataset_name, data in prediction_data['datasets'].items():
        print(f"    - {dataset_name}: {len(data['frames'])} frames")



def train_one_epoch(model, ordered_gops, criterion, optimizer, device, epoch, num_frames_per_gop=10):
    """
    Train for one epoch using pre-loaded GOP sequences with AUTOREGRESSIVE sequential propagation.
    
    Key behavior (matching train_mv_center.py):
    1. For EACH GOP:
       a. RESET LSTM hidden state (start fresh)
       b. Initialize with I-frame (frame 0) GT boxes
       c. Iterate through P-frames sequentially
       d. Use predictions from frame N-1 to predict frame N
       e. Maintain LSTM state WITHIN this GOP
    2. After GOP completes, reset for next GOP
    
    Args:
        model: TrackingPropagator model
        ordered_gops: List of (seq_id, frames) tuples from load_gop_sequences_once()
        criterion: RTDETRTrackingLoss
        optimizer: Optimizer
        device: Device
        epoch: Current epoch number
        num_frames_per_gop: Number of frames to process per GOP (default 10 for efficiency)
    """
    model.train()
    
    # RT-DETR loss components
    total_losses = {
        'total': 0.0,
        'cls': 0.0,
        'box_l1': 0.0,
        'box_giou': 0.0,
        'track_id': 0.0,
        'num_matched': 0,
        'num_gt': 0
    }
    # Mean-MV baseline accumulators (for comparison)
    baseline_map_sum = 0.0
    baseline_frames = 0
    
    num_gops = 0
    
    pbar = tqdm(ordered_gops, desc=f"Epoch {epoch}")
    for seq_id, gop_frames in pbar:
        # Skip GOPs with less than 2 frames
        if len(gop_frames) < 2:
            continue
        
        optimizer.zero_grad()
        
        # Limit frames processed for efficiency
        num_frames = min(len(gop_frames), num_frames_per_gop)
        
        # 🔄 RESET LSTM hidden state at start of GOP (key behavior!)
        hidden_state = None
        
        # Frame 0: I-frame initialization with GT boxes (normalized [0,1])
        current_boxes = gop_frames[0]['boxes'].to(device)
        current_ids = gop_frames[0]['ids'].to(device)
        
        if len(current_boxes) == 0:
            continue  # Skip GOPs with no objects in frame 0
        
        # Process P-frames sequentially (LSTM state persists across frames WITHIN this GOP)
        gop_loss = 0.0
        gop_loss_dict = {k: 0.0 for k in total_losses.keys()}
        frames_processed = 0
        
        for t in range(1, num_frames):
            # Get motion vectors for this P-frame
            mv = gop_frames[t]['motion_vectors'].to(device)
            target_boxes = gop_frames[t]['boxes'].to(device)
            target_ids = gop_frames[t]['ids'].to(device)
            
            if len(target_boxes) == 0:
                continue  # Skip frames with no GT
            
            # Process motion vectors: ensure [1, 2, H, W] shape
            # Handle various input shapes from dataset
            while mv.ndim > 4:
                # Remove extra dimensions: [1, 2, H, W, 2] → [2, H, W, 2]
                mv = mv.squeeze(0)
            
            if mv.ndim == 4:
                # [T, H, W, 2] → [H, W, 2] (take mean over temporal dim if exists)
                if mv.shape[0] == 2 and mv.shape[-1] != 2:
                    # Already [2, H, W, ?] format, keep as is
                    pass
                else:
                    mv = mv.mean(dim=0)  # [H, W, 2]
            
            if mv.ndim == 3:
                if mv.shape[-1] == 2:
                    # [H, W, 2] → [2, H, W]
                    mv = mv.permute(2, 0, 1)
                # else already [2, H, W]
            
            # Add batch dimension: [2, H, W] → [1, 2, H, W]
            if mv.ndim == 3:
                mv = mv.unsqueeze(0)
            
            # Forward pass: predict using PREVIOUS frame predictions + current MVs
            # RT-DETR returns: pred_boxes, pred_class_logits [no_object, pedestrian], pred_track_ids
            pred_boxes, pred_class_logits, pred_ids_logits, hidden_state = model(
                mv,
                None,  # No DCT
                current_boxes,  # Use predictions from previous frame!
                current_ids,
                hidden_state  # Maintain LSTM state within GOP
            )
            
            # Compute RT-DETR loss against current frame GT
            loss, loss_dict, num_matched = criterion(
                pred_boxes,
                pred_class_logits,  # Class logits [no_object, pedestrian]
                pred_ids_logits,
                target_boxes,
                target_ids
            )
            
            gop_loss += loss
            for k, v in loss_dict.items():
                if k != 'num_matched' and k != 'num_gt':  # Don't accumulate counts
                    gop_loss_dict[k] += v
            gop_loss_dict['num_matched'] += num_matched
            gop_loss_dict['num_gt'] += len(target_boxes)
            frames_processed += 1
            
            # CRITICAL: Select confident predictions for next frame (prevents garbage propagation!)
            current_boxes, current_ids = select_confident_predictions(
                pred_boxes,
                pred_class_logits,
                pred_ids_logits,
                confidence_threshold=0.3,  # Only keep pedestrian prob > 0.3
                max_boxes=100  # Limit to top 100 for efficiency
            )
        
        # Average loss over frames in this GOP
        if frames_processed > 0:
            gop_loss = gop_loss / frames_processed
            for k in gop_loss_dict.keys():
                gop_loss_dict[k] /= frames_processed
            
            # Backward pass
            gop_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            # Accumulate losses
            for k in total_losses.keys():
                if k in gop_loss_dict:  # Only accumulate matching keys
                    total_losses[k] += gop_loss_dict[k]
            
            num_gops += 1
            
            # Update progress bar with RT-DETR loss components
            pbar.set_postfix({
                'loss': f"{gop_loss_dict['total']:.4f}",
                'box': f"{gop_loss_dict['box_l1']:.4f}",
                'giou': f"{gop_loss_dict['box_giou']:.4f}",
                'cls': f"{gop_loss_dict['cls']:.4f}",
                'id': f"{gop_loss_dict['track_id']:.4f}",
                'match': f"{int(gop_loss_dict['num_matched'])}/{int(gop_loss_dict['num_gt'])}"
            })
    
    # Average losses
    if num_gops > 0:
        for k in total_losses.keys():
            total_losses[k] /= num_gops

    # Finalize baseline metric
    if baseline_frames > 0:
        baseline_map_avg = baseline_map_sum / baseline_frames
    else:
        baseline_map_avg = 0.0

    total_losses['baseline_map'] = baseline_map_avg

    return total_losses


def mean_mv_baseline_propagation(boxes, motion_vectors, image_size=960):
    """
    Simple mean-MV baseline: propagate boxes using mean motion vector within each box.
    
    This is the SIMPLEST motion-based tracking approach:
    - For each box, compute mean (dx, dy) from overlapping motion blocks
    - Translate box by (dx, dy)
    
    Args:
        boxes: [N, 4] in normalized [0, 1] coordinates (x1, y1, x2, y2)
        motion_vectors: [1, 2, 60, 60] or [2, 60, 60] - raw motion vectors
        image_size: Image size in pixels (default 960)
    
    Returns:
        propagated_boxes: [N, 4] in normalized [0, 1] coordinates
    """
    if len(boxes) == 0:
        return boxes
    
    # Remove batch dimension if present
    if motion_vectors.ndim == 4:
        mv = motion_vectors.squeeze(0)  # [2, 60, 60]
    else:
        mv = motion_vectors
    
    grid_size = 60
    block_size = image_size // grid_size  # 16 pixels per block
    
    # Convert boxes from normalized [0,1] to pixel coordinates
    boxes_pixel = boxes * image_size  # [N, 4]
    
    # Convert to grid coordinates
    boxes_grid = boxes_pixel / block_size  # [N, 4]
    boxes_grid = torch.clamp(boxes_grid, 0, grid_size - 1e-6)
    
    propagated_boxes = []
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes_grid[i]
        
        # Get integer bounds for block indices
        ix1 = int(torch.floor(x1).item())
        iy1 = int(torch.floor(y1).item())
        ix2 = int(torch.ceil(x2).item())
        iy2 = int(torch.ceil(y2).item())
        
        # Ensure at least 1x1 region
        ix2 = max(ix1 + 1, ix2)
        iy2 = max(iy1 + 1, iy2)
        
        # Extract motion vectors in this region
        mv_region = mv[:, iy1:iy2, ix1:ix2]  # [2, H', W']
        
        # Compute MEAN motion (this is the baseline!)
        mean_dx = mv_region[0].mean()
        mean_dy = mv_region[1].mean()
        
        # Apply motion to box (in pixel space)
        box_pixel = boxes_pixel[i]
        new_box_pixel = box_pixel.clone()
        new_box_pixel[0] += mean_dx  # x1
        new_box_pixel[1] += mean_dy  # y1
        new_box_pixel[2] += mean_dx  # x2
        new_box_pixel[3] += mean_dy  # y2
        
        # Clamp to image bounds
        new_box_pixel = torch.clamp(new_box_pixel, 0, image_size)
        
        propagated_boxes.append(new_box_pixel)
    
    propagated_boxes = torch.stack(propagated_boxes)  # [N, 4]
    
    # Convert back to normalized [0, 1]
    propagated_boxes = propagated_boxes / image_size
    
    return propagated_boxes


def validate(model, ordered_gops, criterion, device, num_frames_per_gop=10):
    """
    Validate the model with AUTOREGRESSIVE sequential propagation
    Same as training but without gradient updates
    
    Args:
        model: TrackingPropagator model
        ordered_gops: List of (seq_id, frames) tuples from load_gop_sequences_once()
        criterion: RT-DETR TrackingLoss
        device: Device
        num_frames_per_gop: Number of frames to process per GOP
    """
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'cls': 0.0,
        'box_l1': 0.0,
        'box_giou': 0.0,
        'track_id': 0.0,
        'num_matched': 0,
        'num_gt': 0
    }
    
    num_gops = 0
    
    with torch.no_grad():
        pbar = tqdm(ordered_gops, desc="Validation")
        for seq_id, gop_frames in pbar:
            # Skip GOPs with less than 2 frames
            if len(gop_frames) < 2:
                continue
            
            # Limit frames processed
            num_frames = min(len(gop_frames), num_frames_per_gop)
            
            # 🔄 RESET LSTM hidden state at start of GOP
            hidden_state = None
            
            # Frame 0: I-frame initialization
            current_boxes = gop_frames[0]['boxes'].to(device)
            current_ids = gop_frames[0]['ids'].to(device)
            
            if len(current_boxes) == 0:
                continue
            
            # Process P-frames sequentially
            gop_loss_dict = {k: 0.0 for k in total_losses.keys()}
            frames_processed = 0
            
            for t in range(1, num_frames):
                # Get motion vectors for this P-frame
                mv = gop_frames[t]['motion_vectors'].to(device)
                target_boxes = gop_frames[t]['boxes'].to(device)
                target_ids = gop_frames[t]['ids'].to(device)
                
                if len(target_boxes) == 0:
                    continue
                
                # Process motion vectors: ensure [1, 2, H, W] shape
                while mv.ndim > 4:
                    mv = mv.squeeze(0)
                
                if mv.ndim == 4:
                    if mv.shape[0] == 2 and mv.shape[-1] != 2:
                        pass
                    else:
                        mv = mv.mean(dim=0)
                
                if mv.ndim == 3:
                    if mv.shape[-1] == 2:
                        mv = mv.permute(2, 0, 1)
                
                if mv.ndim == 3:
                    mv = mv.unsqueeze(0)

                # --- Mean-MV baseline prediction (for comparison) ---
                try:
                    baseline_boxes = mean_mv_baseline_propagation(current_boxes, mv, image_size=960)
                    if len(baseline_boxes) > 0 and len(target_boxes) > 0:
                        baseline_scores = torch.ones(len(baseline_boxes), device=device)
                        baseline_map = compute_map50(baseline_boxes.to(device), baseline_scores, target_boxes)
                        baseline_map_sum += baseline_map
                        baseline_frames += 1
                except Exception:
                    # Keep validation robust even if baseline computation fails for edge cases
                    pass
                
                # Forward pass - RT-DETR returns class_logits instead of objectness
                pred_boxes, pred_class_logits, pred_ids_logits, hidden_state = model(
                    mv, None, current_boxes, current_ids, hidden_state
                )
                
                # Compute RT-DETR loss
                loss, loss_dict, num_matched = criterion(
                    pred_boxes,
                    pred_class_logits,  # Class logits [no_object, pedestrian]
                    pred_ids_logits,
                    target_boxes,
                    target_ids
                )
                
                for k, v in loss_dict.items():
                    if k != 'num_matched' and k != 'num_gt':
                        gop_loss_dict[k] += v
                gop_loss_dict['num_matched'] += num_matched
                gop_loss_dict['num_gt'] += len(target_boxes)
                frames_processed += 1
                
                # CRITICAL: Select confident predictions for next frame
                current_boxes, current_ids = select_confident_predictions(
                    pred_boxes,
                    pred_class_logits,
                    pred_ids_logits,
                    confidence_threshold=0.3,
                    max_boxes=100
                )
            
            # Average over frames in this GOP
            if frames_processed > 0:
                for k in gop_loss_dict.keys():
                    gop_loss_dict[k] /= frames_processed
                
                # Accumulate
                for k in total_losses.keys():
                    total_losses[k] += gop_loss_dict[k]
                
                num_gops += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{gop_loss_dict['total']:.4f}",
                    'match': f"{int(gop_loss_dict['num_matched'])}/{int(gop_loss_dict['num_gt'])}"
                })
    
    # Average losses
    if num_gops > 0:
        for k in total_losses.keys():
            total_losses[k] /= num_gops
    
    return total_losses


def plot_training_curves(history, output_dir):
    """Plot training curves for loss evolution"""
    output_dir = Path(output_dir)
    
    epochs = list(range(1, len(history['train_losses']) + 1))
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress - Propagation Tracker', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    ax = axes[0, 0]
    train_total = [x['total'] for x in history['train_losses']]
    val_total = [x['total'] for x in history['val_losses']]
    ax.plot(epochs, train_total, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_total, 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('Total Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Box Regression Loss (L1 + GIoU)
    ax = axes[0, 1]
    train_box = [x['box_l1'] for x in history['train_losses']]
    val_box = [x['box_l1'] for x in history['val_losses']]
    train_giou = [x['box_giou'] for x in history['train_losses']]
    val_giou = [x['box_giou'] for x in history['val_losses']]
    ax.plot(epochs, train_box, 'b-', label='Train L1', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_box, 'r-', label='Val L1', linewidth=2, marker='s', markersize=4)
    ax.plot(epochs, train_giou, 'b--', label='Train GIoU', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax.plot(epochs, val_giou, 'r--', label='Val GIoU', linewidth=2, marker='s', markersize=4, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Box Loss', fontsize=12)
    ax.set_title('Box Regression Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Classification Loss (Focal)
    ax = axes[1, 0]
    train_cls = [x['cls'] for x in history['train_losses']]
    val_cls = [x['cls'] for x in history['val_losses']]
    ax.plot(epochs, train_cls, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_cls, 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Classification Loss (Focal)', fontsize=12)
    ax.set_title('Classification Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Track ID Loss
    ax = axes[1, 1]
    train_id = [x['track_id'] for x in history['train_losses']]
    val_id = [x['track_id'] for x in history['val_losses']]
    ax.plot(epochs, train_id, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs, val_id, 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Track ID Loss (CE)', fontsize=12)
    ax.set_title('Track ID Preservation Loss Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plot_path = output_dir / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Training curves saved to: {plot_path}")
    
    # Also create a separate plot for matching statistics
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    train_matched = [x['num_matched'] for x in history['train_losses']]
    train_gts = [x['num_gt'] for x in history['train_losses']]
    val_matched = [x['num_matched'] for x in history['val_losses']]
    val_gts = [x['num_gt'] for x in history['val_losses']]
    
    train_match_rate = [m / (g + 1e-6) * 100 for m, g in zip(train_matched, train_gts)]
    val_match_rate = [m / (g + 1e-6) * 100 for m, g in zip(val_matched, val_gts)]
    
    ax2.plot(epochs, train_match_rate, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_match_rate, 'r-', label='Val', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Match Rate (%)', fontsize=12)
    ax2.set_title('Hungarian Matching Success Rate', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    match_plot_path = output_dir / 'matching_rate.png'
    plt.savefig(match_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Matching rate plot saved to: {match_plot_path}")
    
    # Create mAP@0.5 plots for different sequence lengths
    if len(history['map_scores']) > 0:
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
        fig3.suptitle('mAP@0.5 Evolution - Different Sequence Lengths', fontsize=16, fontweight='bold')
        
        map_6frames = [x['map50_6frames'] for x in history['map_scores']]
        map_12frames = [x['map50_12frames'] for x in history['map_scores']]
        map_50frames = [x['map50_50frames'] for x in history['map_scores']]
        
        # Plot 1: mAP@0.5 for 6 frames
        ax = axes3[0]
        ax.plot(epochs, map_6frames, 'g-', linewidth=2, marker='o', markersize=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP@0.5', fontsize=12)
        ax.set_title('First 6 Frames', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 baseline')
        ax.legend(fontsize=10)
        
        # Plot 2: mAP@0.5 for 12 frames
        ax = axes3[1]
        ax.plot(epochs, map_12frames, 'orange', linewidth=2, marker='s', markersize=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP@0.5', fontsize=12)
        ax.set_title('First 12 Frames', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 baseline')
        ax.legend(fontsize=10)
        
        # Plot 3: mAP@0.5 for full sequence (50 frames)
        ax = axes3[2]
        ax.plot(epochs, map_50frames, 'purple', linewidth=2, marker='^', markersize=5)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP@0.5', fontsize=12)
        ax.set_title('Full Sequence (50 frames)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 baseline')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        map_plot_path = output_dir / 'map50_evolution.png'
        plt.savefig(map_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 mAP@0.5 evolution plot saved to: {map_plot_path}")
    
    plt.close('all')


def main(args):
    """Main training loop"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets (support multiple datasets)
    print("\n=== Creating Datasets ===")
    
    # Handle dataset type
    if args.dataset_type == 'all':
        dataset_types = ['mot17', 'mot15', 'mot20']
        print(f"Training on ALL datasets: {dataset_types}")
    elif ',' in args.dataset_type:
        dataset_types = [d.strip() for d in args.dataset_type.split(',')]
        print(f"Training on multiple datasets: {dataset_types}")
    else:
        dataset_types = [args.dataset_type]
        print(f"Training on single dataset: {dataset_types}")
    
    # Store for later use in visualization
    datasets_for_viz = dataset_types
    
    # Create training datasets
    train_datasets = []
    for dt in dataset_types:
        ds = create_mots_dataset(
            dataset_type=dt,
            resolution=960,
            mode='train',
            load_iframe=False,  # No I-frames for propagation-only model
            load_pframe=False,  # No P-frames
            load_motion_vectors=True,  # Only MV
            load_residuals=False,  # No residuals
            dct_coeffs=0,  # No DCT coefficients
            load_annotations=True,
            sequence_length=args.sequence_length,
            data_format='separate',
            combine_datasets=False
        )
        train_datasets.append(ds)
        print(f"  {dt.upper()} train: {len(ds)} sequences")
    
    # Combine datasets
    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        print(f"  Combined train dataset: {len(train_dataset)} total sequences")
    else:
        train_dataset = train_datasets[0]
    
    # Create validation datasets (use test split as validation)
    val_datasets = []
    for dt in dataset_types:
        ds = create_mots_dataset(
            dataset_type=dt,
            resolution=960,
            mode='test',  # Use test split for validation
            load_iframe=False,
            load_pframe=False,
            load_motion_vectors=True,
            load_residuals=False,
            dct_coeffs=0,
            load_annotations=True,
            sequence_length=args.sequence_length,
            data_format='separate',
            combine_datasets=False
        )
        val_datasets.append(ds)
        print(f"  {dt.upper()} val (test split): {len(ds)} sequences")
    
    if len(val_datasets) > 1:
        val_dataset = ConcatDataset(val_datasets)
        print(f"  Combined val dataset: {len(val_dataset)} total sequences")
    else:
        val_dataset = val_datasets[0]
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    
    # 📦 Pre-load all GOP sequences ONCE (reused across epochs)
    print("\n=== Pre-loading GOP Sequences ===")
    print("Loading training GOPs...")
    train_gops = load_gop_sequences_once(
        train_dataset,
        gop_length=args.sequence_length,
        max_sequences=args.max_gops if hasattr(args, 'max_gops') else None
    )
    
    print("\nLoading validation GOPs...")
    val_gops = load_gop_sequences_once(
        val_dataset,
        gop_length=args.sequence_length,
        max_sequences=args.max_val_gops if hasattr(args, 'max_val_gops') else None
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = TrackingPropagator(
        max_slots=args.num_slots,
        mv_feature_dim=64,
        slot_dim=args.hidden_size,
        hidden_dim=args.hidden_size,
        num_lstm_layers=2,
        max_track_id=args.max_track_id,
        image_size=960
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create RT-DETR loss function
    criterion = RTDETRTrackingLoss(
        lambda_cls=args.obj_weight,  # Classification loss weight (was objectness)
        lambda_box=args.box_weight,   # L1 box loss weight
        lambda_giou=2.0,               # GIoU loss weight
        lambda_id=args.id_weight,      # Track ID loss weight
        focal_alpha=0.25,              # Focal loss alpha
        focal_gamma=2.0                # Focal loss gamma
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'map_scores': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\n=== Starting Training ===")
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_losses = train_one_epoch(
            model, train_gops, criterion, optimizer, device, epoch,
            num_frames_per_gop=args.num_frames
        )
        
        # Validate
        val_losses = validate(model, val_gops, criterion, device,
                             num_frames_per_gop=args.num_frames)
        
        # Compute mAP@0.5 and visualizations every 5 epochs (and always on epoch 1)
        if epoch % 5 == 0 or epoch == 1:
            print("\n📊 Computing mAP@0.5...")
            map_results = compute_map_over_sequence(model, val_gops, device, max_frames_list=[6, 12, 50])
            
            # Create prediction visualizations
            visualize_predictions_grid(
                model=model,
                datasets_config=datasets_for_viz,  # Use datasets from training
                device=device,
                output_dir=output_dir,
                epoch=epoch,
                num_frames=6  # Visualize first 6 frames
            )
        else:
            # Skip mAP computation for non-evaluation epochs
            map_results = {
                'map50_6frames': 0.0,
                'map50_12frames': 0.0,
                'map50_50frames': 0.0
            }
        
        # Update scheduler
        scheduler.step(val_losses['total'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f} "
              f"(box={train_losses['box_l1']:.4f}, "
              f"cls={train_losses['cls']:.4f}, "
              f"giou={train_losses['box_giou']:.4f}, "
              f"id={train_losses['track_id']:.4f})")
        print(f"  Val Loss: {val_losses['total']:.4f} "
              f"(box={val_losses['box_l1']:.4f}, "
              f"cls={val_losses['cls']:.4f}, "
              f"giou={val_losses['box_giou']:.4f}, "
              f"id={val_losses['track_id']:.4f})")
        print(f"  Train Matched: {train_losses['num_matched']:.1f}/{train_losses['num_gt']:.1f}")
        print(f"  Val Matched: {val_losses['num_matched']:.1f}/{val_losses['num_gt']:.1f}")
        
        # Only print mAP when it was actually computed
        if epoch % 5 == 0 or epoch == 1:
            print(f"  mAP@0.5 (6 frames): {map_results['map50_6frames']:.4f}")
            print(f"  mAP@0.5 (12 frames): {map_results['map50_12frames']:.4f}")
            print(f"  mAP@0.5 (50 frames): {map_results['map50_50frames']:.4f}")
        
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save history
        history['train_losses'].append(train_losses)
        history['val_losses'].append(val_losses)
        history['map_scores'].append(map_results)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'history': history
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'history': history
            }, best_model_path)
            print(f"  ✅ New best model! Val loss: {best_val_loss:.4f}")
        
        # Plot training curves after each epoch (overwrites previous)
        if epoch >= 2:  # Wait for at least 2 epochs to have meaningful curves
            plot_training_curves(history, output_dir)
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'history': history
    }, final_model_path)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert to JSON-serializable format
        history_json = {
            'train_losses': history['train_losses'],
            'val_losses': history['val_losses'],
            'learning_rates': history['learning_rates']
        }
        json.dump(history_json, f, indent=2)
    
    # Plot final training curves
    print("\n📊 Generating final training plots...")
    plot_training_curves(history, output_dir)
    
    print(f"\n{'='*50}")
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Propagation Tracker")
    
    # Dataset
    parser.add_argument('--dataset-type', type=str, default='mot17',
                        help='Dataset to use: mot17, mot15, mot20, all, or comma-separated (e.g., "mot17,mot15")')
    parser.add_argument('--data-dir', type=str, 
                        default='/home/aduche/Bureau/datasets/MOTS/videos/test',
                        help='Data directory')
    parser.add_argument('--sequence-length', type=int, default=60,
                        help='Sequence length for training')
    parser.add_argument('--num-frames', type=int, default=10,
                        help='Number of frames to process per GOP (for efficiency)')
    parser.add_argument('--max-gops', type=int, default=None,
                        help='Maximum number of GOPs to load for training (None=load all)')
    parser.add_argument('--max-val-gops', type=int, default=None,
                        help='Maximum number of GOPs to load for validation (None=load all)')
    
    # Model
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--num-slots', type=int, default=300,
                        help='Number of output slots (fixed)')
    parser.add_argument('--max-track-id', type=int, default=1000,
                        help='Maximum track ID for classification')
    
    # Loss weights
    parser.add_argument('--box-weight', type=float, default=5.0,
                        help='Box regression loss weight')
    parser.add_argument('--obj-weight', type=float, default=1.0,
                        help='Objectness loss weight')
    parser.add_argument('--id-weight', type=float, default=2.0,
                        help='Track ID loss weight')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    # Checkpointing
    parser.add_argument('--output-dir', type=str, 
                        default='experiments/propagation_tracker',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
