#!/usr/bin/env python3
"""
Motion Map Validation Script

This script validates the motion vector prediction model on MOTS dataset sequences.
It evaluates tracking performance using mAP metrics.
Compares deep learning models (baseline & magnitude) vs motion vector propagation.
"""

import sys
import os
import torch
import numpy as np
import cv2
import json
import argparse
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'dataset'))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Motion Vector mAP Validation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-path", type=str, default="/home/aduche/Bureau/datasets/MOTS/videos/", help="MOTS dataset path")
    parser.add_argument("--resolution", type=int, default=640, choices=[640, 960], help="Video resolution (must match model training resolution)")
    parser.add_argument("--max-videos", type=int, default=5, help="Maximum videos to process")
    parser.add_argument("--output-dir", type=str, default="outputs/motion_validation", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--create-plots", action="store_true", help="Create mAP analysis plots")
    parser.add_argument("--compare-baseline", action="store_true", help="Also run motion vector propagation baseline")
    parser.add_argument("--baseline-only", action="store_true", help="Run only motion vector propagation baseline (no deep learning)")
    return parser.parse_args()

class MotionValidationMetrics:
    """Compute motion vector validation metrics including mAP calculation."""
    
    def __init__(self, iou_thresholds=None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75, 0.9]
        self.reset()
    
    def reset(self):
        self.video_results = {}
        self.gop_results = {}
        self.frame_results = []
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes (center format)."""
        # Convert center format to corner format
        x1_min = box1[0] - box1[2] / 2
        y1_min = box1[1] - box1[3] / 2
        x1_max = box1[0] + box1[2] / 2
        y1_max = box1[1] + box1[3] / 2
        
        x2_min = box2[0] - box2[2] / 2
        y2_min = box2[1] - box2[3] / 2
        x2_max = box2[0] + box2[2] / 2
        y2_max = box2[1] + box2[3] / 2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_map_for_frame(self, pred_boxes, gt_boxes, valid_mask):
        """Calculate mAP for a single frame at all IoU thresholds."""
        frame_maps = {}
        
        if not valid_mask.any() or len(gt_boxes) == 0:
            for thresh in self.iou_thresholds:
                frame_maps[f'mAP@{thresh}'] = 0.0
            return frame_maps
        
        # Get valid predictions and ground truth
        pred_valid = pred_boxes[valid_mask]
        gt_filtered = gt_boxes[gt_boxes.sum(dim=1) != 0]  # Remove zero boxes
        
        if len(pred_valid) == 0 or len(gt_filtered) == 0:
            for thresh in self.iou_thresholds:
                frame_maps[f'mAP@{thresh}'] = 0.0
            return frame_maps
        
        # Calculate IoU matrix between all predictions and ground truth
        ious = torch.zeros(len(pred_valid), len(gt_filtered))
        for i, pred_box in enumerate(pred_valid):
            for j, gt_box in enumerate(gt_filtered):
                ious[i, j] = self.calculate_iou(pred_box, gt_box)
        
        # Calculate mAP for each threshold
        for thresh in self.iou_thresholds:
            # For each ground truth, find best matching prediction
            matches = []
            used_preds = set()
            
            for j in range(len(gt_filtered)):
                best_iou = 0.0
                best_pred = -1
                
                for i in range(len(pred_valid)):
                    if i not in used_preds and ious[i, j] > best_iou:
                        best_iou = ious[i, j]
                        best_pred = i
                
                if best_iou >= thresh:
                    matches.append(best_iou)
                    used_preds.add(best_pred)
            
            # Calculate precision (TP / (TP + FP))
            tp = len(matches)
            fp = len(pred_valid) - tp
            precision = tp / len(pred_valid) if len(pred_valid) > 0 else 0.0
            
            # Calculate recall (TP / (TP + FN))
            fn = len(gt_filtered) - tp
            recall = tp / len(gt_filtered) if len(gt_filtered) > 0 else 0.0
            
            # For simplicity, use precision as mAP approximation
            frame_maps[f'mAP@{thresh}'] = precision
        
        return frame_maps

def load_model(model_path, device):
    """Load the trained model (supports both IDMultiObjectTracker and MV-Center models)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect model type from checkpoint
    model_type = None
    if 'model_info' in checkpoint:
        model_info = checkpoint['model_info']
        # Check if it's a memory tracker (has LSTM-specific config)
        if 'feature_dim' in model_info and 'hidden_dim' in model_info and 'max_objects' in model_info:
            model_type = 'memory_tracker'
            print(f"📊 Detected Memory-Based LSTM Tracker:")
            print(f"   Feature dim: {model_info.get('feature_dim')}")
            print(f"   Hidden dim: {model_info.get('hidden_dim')}")
            print(f"   Max objects: {model_info.get('max_objects')}")
            print(f"   MV resolution: {model_info.get('mv_height')}x{model_info.get('mv_width')}")
        else:
            # MV-Center model
            model_type = 'mv_center'
            print(f"📊 Detected MV-Center model:")
            print(f"   Model: {model_info.get('model_name', 'Unknown')}")
            print(f"   Version: {model_info.get('version', 'Unknown')}")
            print(f"   Input channels: {model_info.get('input_channels', 2)}")
            print(f"   Parameters: {model_info.get('total_parameters', 0):,}")
    elif 'config' in checkpoint:
        # ID-aware tracker
        model_type = 'id_tracker'
        print(f"📊 Detected ID-aware tracker model")
    else:
        # Try to infer from state dict
        model_state = checkpoint.get('model_state_dict', checkpoint)
        if 'backbone.features.0.weight' in model_state:
            model_type = 'mv_center'
            print(f"📊 Detected MV-Center model (from state dict)")
        else:
            model_type = 'id_tracker'
            print(f"📊 Detected ID-aware tracker model (from state dict)")
    
    # Load appropriate model
    if model_type == 'memory_tracker':
        # Load Memory-Based LSTM Tracker (standard, enhanced, or fullbbox)
        model_info = checkpoint['model_info']
        
        # Check for full bbox model (has use_backbone flag)
        use_backbone = model_info.get('use_backbone', False)
        use_id_embedding = model_info.get('use_id_embedding', False)
        embedding_dim = model_info.get('embedding_dim', 128)
        
        if use_backbone:
            # Load full bbox model with CNN backbone
            from mots_exp.models.mv_center.mv_center_memory_fullbox import MVCenterMemoryTrackerFullBox
            print("   ✨ Loading FULL BBOX model with CNN backbone")
            ModelClass = MVCenterMemoryTrackerFullBox
        elif use_id_embedding:
            # Load enhanced model with ID embeddings
            from mots_exp.models.mv_center.mv_center_memory_enhanced import MVCenterMemoryTrackerEnhanced
            print("   ✨ Loading ENHANCED model with ID embeddings")
            ModelClass = MVCenterMemoryTrackerEnhanced
        else:
            # Load standard model
            from mots_exp.models.mv_center.mv_center_memory import MVCenterMemoryTracker
            ModelClass = MVCenterMemoryTracker
        
        # Check for ROI Align parameters (added in enhancement)
        use_roi_align = model_info.get('use_roi_align', False)
        roi_size = model_info.get('roi_size', (7, 7))
        
        # Create model with appropriate parameters
        model_kwargs = {
            'feature_dim': model_info['feature_dim'],
            'hidden_dim': model_info['hidden_dim'],
            'max_objects': model_info['max_objects'],
            'grid_size': 40,  # Motion vector grid is 40x40
            'image_size': 640,  # Image resolution
        }
        
        # Add full bbox model parameters
        if use_backbone:
            model_kwargs['use_backbone'] = use_backbone
        else:
            # Standard/enhanced models use ROI Align
            model_kwargs['use_roi_align'] = use_roi_align
            model_kwargs['roi_size'] = roi_size
        
        # Add enhanced model parameters if using enhanced model
        if use_id_embedding:
            model_kwargs['use_id_embedding'] = use_id_embedding
            model_kwargs['embedding_dim'] = embedding_dim
        
        model = ModelClass(**model_kwargs)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        use_magnitude = False  # Memory tracker uses X, Y only (2 channels)
        max_objects = model_info['max_objects']
        
        print(f"📊 Memory-Based LSTM Tracker loaded:")
        if use_roi_align:
            print(f"   ✨ Using ROI Align with {roi_size[0]}x{roi_size[1]} grid")
        print(f"   Input channels: 2 (u,v)")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, max_objects, use_magnitude, 'memory_tracker'
    
    elif model_type == 'mv_center':
        # Load MV-Center model
        from mots_exp.models.mv_center import create_mv_center_v1
        
        if 'model_info' in checkpoint:
            input_channels = checkpoint['model_info'].get('input_channels', 2)
            use_embeddings = checkpoint['model_info'].get('use_embeddings', False)
        else:
            # Infer from state dict
            backbone_weight = checkpoint.get('model_state_dict', checkpoint)['backbone.features.0.weight']
            input_channels = backbone_weight.shape[1]
            use_embeddings = False
        
        model = create_mv_center_v1(
            input_channels=input_channels,
            use_embeddings=use_embeddings
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        use_magnitude = (input_channels == 3)
        max_objects = 100  # MV-Center doesn't have explicit max_objects
        
        print(f"📊 MV-Center model loaded:")
        print(f"   Input channels: {input_channels} ({'u,v,mag' if use_magnitude else 'u,v'})")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, max_objects, use_magnitude, 'mv_center'
    
    else:
        # Load ID-aware tracker (original)
        from models.id_aware_tracker import IDMultiObjectTracker
        
        # Try to get model parameters from checkpoint config first, then infer from state
        if 'config' in checkpoint and 'motion_shape' in checkpoint['config']:
            # Use saved config (preferred method)
            motion_shape = checkpoint['config']['motion_shape']
            max_objects = checkpoint['config'].get('max_objects', 100)
            max_id = checkpoint['config'].get('max_id', 200)
            use_magnitude = checkpoint['config'].get('use_magnitude', False)
            motion_channels = checkpoint['config'].get('motion_channels', motion_shape[0])
            
            print(f"📊 Loaded from checkpoint config:")
            print(f"   Motion shape: {motion_shape}")
            print(f"   Use magnitude: {use_magnitude} ({motion_channels}-channel)")
            print(f"   Max objects: {max_objects}, Max ID: {max_id}")
        else:
            # Fallback: infer from model state (old checkpoints)
            model_state = checkpoint['model_state_dict']
            id_embedding_shape = model_state['id_encoder.embedding.weight'].shape[0]
            max_id = id_embedding_shape
            max_objects = max_id // 10
            motion_shape = (3, 40, 40)  # Assume 3 channels for old checkpoints
            use_magnitude = True
            motion_channels = 3
            
            print(f"📊 Inferred from model state (old checkpoint):")
            print(f"   Motion shape: {motion_shape}")
            print(f"   Max objects: {max_objects}, Max ID: {max_id}")
        
        model = IDMultiObjectTracker(
            motion_shape=motion_shape,
            hidden_dim=128,
            max_objects=max_objects,
            max_id=max_id
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        print(f"📊 ID-aware tracker loaded: {sum(p.numel() for p in model.parameters())} parameters")
        return model, max_objects, use_magnitude, 'id_tracker'


def convert_mv_center_predictions_to_boxes(predictions, max_objects=100, score_threshold=0.01):
    """
    Convert MV-Center raw predictions to bounding box tensor format.
    
    MV-Center raw output: dict with P3/P4 levels
    Each level: {'center': [B, 1, H, W], 'box': [B, 4, H, W]}
    
    Target format: torch.Tensor [1, max_objects, 4] in center format [cx, cy, w, h]
    
    Note: Model now outputs logits. Sigmoid is applied before thresholding.
          score_threshold of 0.01-0.5 should work properly now.
    """
    device = list(predictions.values())[0]['center'].device
    
    # Initialize output tensor
    boxes_tensor = torch.zeros(1, max_objects, 4, device=device)
    valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
    
    # Collect all detections from all levels
    all_detections = []
    
    for level_name, level_preds in predictions.items():
        if level_name not in ['P3', 'P4']:
            continue
        
        center_logits = level_preds['center'][0, 0]  # [H, W] - logits
        center_map = torch.sigmoid(center_logits)  # Apply sigmoid to get probabilities
        box_map = level_preds['box'][0]  # [4, H, W]
        
        # Find peaks in center heatmap
        h, w = center_map.shape
        
        # Apply threshold
        peak_mask = center_map > score_threshold
        
        if not peak_mask.any():
            continue
        
        # Get peak locations
        peak_coords = torch.nonzero(peak_mask, as_tuple=False)  # [N, 2] (y, x)
        peak_scores = center_map[peak_mask]  # [N]
        
        # Extract boxes at peak locations
        for i in range(len(peak_coords)):
            y, x = peak_coords[i]
            score = peak_scores[i].item()
            
            # Get box parameters at this location
            dx, dy, log_w, log_h = box_map[:, y, x]
            
            # Convert to actual box coordinates
            # The box regression predicts offset + log size
            stride = 640 / h  # P3: 640/10=64, P4: 640/5=128
            
            cx = (x + dx) * stride
            cy = (y + dy) * stride
            box_w = torch.exp(log_w) * stride
            box_h = torch.exp(log_h) * stride
            
            # Normalize to [0, 1]
            cx = cx / 640
            cy = cy / 640
            box_w = box_w / 640
            box_h = box_h / 640
            
            all_detections.append({
                'box': [cx.item(), cy.item(), box_w.item(), box_h.item()],
                'score': score,
                'level': level_name
            })
    
    # Sort by score and keep top max_objects
    all_detections.sort(key=lambda x: x['score'], reverse=True)
    all_detections = all_detections[:max_objects]
    
    # Fill tensor
    for i, det in enumerate(all_detections):
        box = det['box']
        boxes_tensor[0, i] = torch.tensor(box, dtype=torch.float32, device=device)
        valid_mask[0, i] = True
    
    return boxes_tensor, valid_mask


def run_model_inference(model, model_type, motion_tensor, prev_boxes, object_ids, valid_mask, max_objects):
    """
    Run model inference with appropriate interface for each model type.
    
    Args:
        model: The model instance
        model_type: 'mv_center' or 'id_tracker'
        motion_tensor: Motion vectors [B, C, H, W]
        prev_boxes: Previous boxes [B, max_objects, 4] (used by ID tracker)
        object_ids: Object IDs [B, max_objects] (used by ID tracker)
        valid_mask: Valid object mask [B, max_objects] (used by ID tracker)
        max_objects: Maximum number of objects
        
    Returns:
        predicted_boxes: [B, max_objects, 4]
        predicted_confidences: [B, max_objects] (if available)
        predicted_ids: [B, max_objects] (if available)
    """
    if model_type == 'mv_center':
        # MV-Center: Detection-based, no tracking state
        with torch.no_grad():
            predictions = model(motion_tensor)  # Returns raw predictions dict
        
        # Convert predictions to box tensor format
        predicted_boxes, new_valid_mask = convert_mv_center_predictions_to_boxes(
            predictions, max_objects=max_objects, score_threshold=0.01
        )
        
        # MV-Center doesn't track IDs, assign sequential IDs
        predicted_ids = torch.arange(1, max_objects + 1, dtype=torch.long, device=predicted_boxes.device).unsqueeze(0)
        predicted_confidences = torch.zeros(1, max_objects, device=predicted_boxes.device)
        
        return predicted_boxes, predicted_confidences, predicted_ids
        
    else:  # 'id_tracker'
        # ID-aware tracker: Sequential prediction with tracking state
        with torch.no_grad():
            predicted_boxes, predicted_confidences, predicted_ids = model(
                motion_tensor, prev_boxes, object_ids, valid_mask
            )
        
        return predicted_boxes, predicted_confidences, predicted_ids


def run_memory_tracker_gop_inference(model, gop_frames, device, max_objects):
    """
    Run inference for Memory-Based LSTM Tracker on a complete GOP sequence.
    
    Args:
        model: MVCenterMemoryTracker instance
        gop_frames: List of GOP frames (sorted by frame_id)
        device: Device to run on
        max_objects: Maximum number of objects
        
    Returns:
        List of predictions for each frame, each containing:
            - boxes: [N, 4] actual predicted boxes
            - confidences: [N] confidence scores
            - ids: [N] sequential object IDs
    """
    model.reset()  # Reset LSTM state for this GOP
    
    # Prepare GOP sequence tensors
    motion_sequence = []
    target_boxes_sequence = []
    
    for sample in gop_frames:
        # Extract motion vectors
        motion_vectors = sample.get('motion_vectors')
        if motion_vectors is None:
            continue
        
        # Ensure motion vectors are [C, H, W]
        if isinstance(motion_vectors, torch.Tensor):
            if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                # [1, H, W, 2] -> [2, H, W]
                motion_vectors = motion_vectors[0].permute(2, 0, 1)
            elif len(motion_vectors.shape) == 3 and motion_vectors.shape[2] == 2:
                # [H, W, 2] -> [2, H, W]
                motion_vectors = motion_vectors.permute(2, 0, 1)
        
        # Get target boxes for validation
        boxes = sample.get('boxes', torch.zeros(0, 4))
        
        motion_sequence.append(motion_vectors.to(device))
        target_boxes_sequence.append(boxes)
    
    if len(motion_sequence) == 0:
        return []
    
    # Stack into batch: [T, 2, H, W]
    motion_batch = torch.stack(motion_sequence, dim=0)
    
    # Get I-frame boxes (first frame's ground truth for initialization)
    first_frame_boxes = target_boxes_sequence[0]
    if isinstance(first_frame_boxes, torch.Tensor) and first_frame_boxes.numel() > 0:
        iframe_boxes = first_frame_boxes.view(-1, 4)
        iframe_boxes = iframe_boxes[iframe_boxes.sum(dim=1) != 0]  # Remove zero boxes
    else:
        iframe_boxes = torch.zeros(0, 4)
    
    iframe_boxes = iframe_boxes.to(device)
    
    # 🔍 DEBUG: Log motion vector properties
    print(f"\n🔍 DEBUG Motion Vectors Input:")
    print(f"   - motion_batch.shape = {motion_batch.shape}")
    print(f"   - motion_batch dtype = {motion_batch.dtype}")
    print(f"   - motion_batch min/max = {motion_batch.min():.4f} / {motion_batch.max():.4f}")
    print(f"   - motion_batch mean/std = {motion_batch.mean():.4f} / {motion_batch.std():.4f}")
    print(f"   - Number of unique values (first frame): {torch.unique(motion_batch[0]).numel()}")
    print(f"   - Sample values (first frame, first 10): {motion_batch[0].flatten()[:10].tolist()}")
    print(f"   - iframe_boxes.shape = {iframe_boxes.shape}")
    print(f"   - iframe_boxes = {iframe_boxes[:3] if len(iframe_boxes) > 0 else 'empty'}")
    
    # Run GOP inference
    with torch.no_grad():
        # Returns: (predictions, confidences) for standard model
        #       or (predictions, confidences, embeddings) for enhanced model
        outputs = model.forward_gop(motion_batch, iframe_boxes)
        
        # Handle both standard and enhanced models
        if isinstance(outputs, tuple) and len(outputs) == 3:
            # Enhanced model with ID embeddings
            predictions_list, confidences_list, embeddings_list = outputs
        else:
            # Standard model
            predictions_list, confidences_list = outputs
    
    # Convert to frame predictions
    frame_predictions = []
    for t in range(len(motion_sequence)):
        boxes = predictions_list[t]  # [N, 4]
        confidences = confidences_list[t]  # [N]
        
        # Create sequential IDs
        num_pred = len(boxes)
        ids = torch.arange(1, num_pred + 1, device=device)
        
        frame_predictions.append({
            'boxes': boxes,
            'confidences': confidences,
            'ids': ids
        })
    
    return frame_predictions


def scan_for_video_sequences(data_path, max_videos, resolution=640):
    """Scan for video sequences using dataset factory (same approach as validate.py)."""
    print(f"🔍 Using dataset factory to load motion vector data...")
    
    try:
        # Add dataset path for imports (same as validate.py)
        dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
        dataset_path = os.path.abspath(dataset_path)
        print(f"📁 Dataset path: {dataset_path}")
        
        if dataset_path not in sys.path:
            sys.path.append(dataset_path)
            
        # Also add the parent directory to ensure dataset module is found
        parent_path = os.path.join(os.path.dirname(__file__), '..', '..')
        parent_path = os.path.abspath(parent_path)
        if parent_path not in sys.path:
            sys.path.append(parent_path)
        
        # Import dataset factory (same as validate.py)
        from dataset.factory.dataset_factory import create_mots_dataset
        
        # Create dataset with motion vectors only (no RGB frames)
        dataset = create_mots_dataset(
            dataset_type="mot17",
            resolution=resolution,
            mode="train",  # Use train data for validation
            load_iframe=False,       # No RGB frames needed
            load_pframe=False,       # No RGB frames needed
            load_motion_vectors=True,  # Only motion vectors
            load_residuals=False,
            load_annotations=True,   # Need annotations for mAP calculation
            sequence_length=48  # GOP length
        )
        
        if dataset is None or len(dataset) == 0:
            print(f"❌ Dataset creation failed or empty")
            return []
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        return []

def motion_vector_propagation_baseline(dataset, metrics, max_objects, max_videos):
    """
    Baseline method: Track objects using only motion vector propagation (no deep learning).
    Uses motion vectors inside bounding boxes from step t-1 to predict position at step t.
    """
    print(f"\n🎯 Running Motion Vector Propagation Baseline (No Deep Learning)")
    print(f"=" * 70)
    
    from collections import defaultdict
    from scipy import ndimage
    
    # Group samples by video and GOP
    video_gop_groups = defaultdict(lambda: defaultdict(list))
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            if sample is not None and 'sequence_id' in sample:
                seq_id = sample['sequence_id']
                if '_gop' in seq_id:
                    last_gop_pos = seq_id.rfind('_gop')
                    if last_gop_pos != -1:
                        video_name = seq_id[:last_gop_pos]
                        gop_part = seq_id[last_gop_pos + 4:]
                        try:
                            gop_idx = int(gop_part)
                            video_gop_groups[video_name][gop_idx].append((idx, sample))
                        except ValueError:
                            continue
        except Exception as e:
            continue
    
    if max_videos > 0:
        video_names = list(video_gop_groups.keys())[:max_videos]
        video_gop_groups = {name: video_gop_groups[name] for name in video_names}
    
    validation_results = {}
    
    for video_name, gop_dict in video_gop_groups.items():
        print(f"\n🎬 Baseline tracking: {video_name}")
        
        video_maps = []
        video_gop_maps = {}
        video_frame_index_maps = defaultdict(list)
        
        for gop_idx in sorted(gop_dict.keys()):
            frames_list = gop_dict[gop_idx]
            frames_list.sort(key=lambda x: x[0])
            
            print(f"   GOP {gop_idx}: Propagating {len(frames_list)} frames...")
            
            # Track bounding boxes using motion vectors
            current_boxes = None
            current_ids = None
            
            gop_maps = []
            
            for frame_idx, (_, sample) in enumerate(frames_list):
                try:
                    motion_vectors = sample.get('motion_vectors')
                    boxes_data = sample.get('boxes', torch.zeros(1, 4))
                    
                    if motion_vectors is None:
                        continue
                    
                    # Process motion vectors
                    if isinstance(motion_vectors, torch.Tensor):
                        if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                            motion_vectors = motion_vectors[0].permute(2, 0, 1)
                        elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                            motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                        
                        if motion_vectors.shape[0] != 2:
                            motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else torch.zeros(2, 40, 40)
                    
                    # Get ground truth
                    if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                        gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                        gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
                    else:
                        gt_boxes = torch.zeros(0, 4)
                    
                    # Initialize with GT on first frame
                    if frame_idx == 0 or current_boxes is None:
                        if len(gt_boxes) > 0:
                            current_boxes = gt_boxes.clone()
                            current_ids = torch.arange(1, len(gt_boxes) + 1)
                        else:
                            continue
                    else:
                        # MOTION VECTOR PROPAGATION: Predict next position
                        propagated_boxes = []
                        
                        for box in current_boxes:
                            # Convert normalized box to pixel coordinates (assuming 960x960)
                            cx_px, cy_px = box[0].item() * 960, box[1].item() * 960
                            w_px, h_px = box[2].item() * 960, box[3].item() * 960
                            
                            # Get bounding box in motion vector grid (40x40)
                            cx_grid = int((cx_px / 960) * 40)
                            cy_grid = int((cy_px / 960) * 40)
                            w_grid = max(1, int((w_px / 960) * 40))
                            h_grid = max(1, int((h_px / 960) * 40))
                            
                            # Extract motion vectors inside bounding box
                            x_min = max(0, cx_grid - w_grid // 2)
                            x_max = min(40, cx_grid + w_grid // 2)
                            y_min = max(0, cy_grid - h_grid // 2)
                            y_max = min(40, cy_grid + h_grid // 2)
                            
                            if x_max > x_min and y_max > y_min:
                                mv_x_region = motion_vectors[0, y_min:y_max, x_min:x_max]
                                mv_y_region = motion_vectors[1, y_min:y_max, x_min:x_max]
                                
                                # Average motion in region
                                avg_mv_x = mv_x_region.mean().item()
                                avg_mv_y = mv_y_region.mean().item()
                            else:
                                avg_mv_x, avg_mv_y = 0.0, 0.0
                            
                            # Propagate box position (motion vectors are in pixel units)
                            new_cx_px = cx_px + avg_mv_x
                            new_cy_px = cy_px + avg_mv_y
                            
                            # Clamp to image bounds
                            new_cx_px = np.clip(new_cx_px, w_px/2, 960 - w_px/2)
                            new_cy_px = np.clip(new_cy_px, h_px/2, 960 - h_px/2)
                            
                            # Convert back to normalized coordinates
                            new_box = torch.tensor([
                                new_cx_px / 960,
                                new_cy_px / 960,
                                w_px / 960,
                                h_px / 960
                            ])
                            propagated_boxes.append(new_box)
                        
                        current_boxes = torch.stack(propagated_boxes) if propagated_boxes else current_boxes
                    
                    # Calculate mAP for this frame
                    if len(current_boxes) > 0 and len(gt_boxes) > 0:
                        frame_map = metrics.calculate_map_for_frame(
                            current_boxes, 
                            gt_boxes,
                            torch.ones(len(current_boxes), dtype=torch.bool)
                        )
                        
                        map_50 = frame_map.get('mAP@0.5', 0.0)
                        gop_maps.append(map_50)
                        video_maps.append(map_50)
                        video_frame_index_maps[frame_idx].append(map_50)
                
                except Exception as e:
                    print(f"      ⚠️ Frame {frame_idx} error: {e}")
                    continue
            
            # Store GOP results
            if gop_maps:
                gop_avg = np.mean(gop_maps)
                video_gop_maps[gop_idx] = {'avg_map': gop_avg, 'frame_maps': gop_maps}
                print(f"      ✓ GOP {gop_idx} mAP@0.5: {gop_avg:.4f}")
        
        # Store video results
        if video_maps:
            validation_results[video_name] = {
                'avg_map': {'mAP@0.5': np.mean(video_maps)},
                'gop_stats': video_gop_maps,
                'frame_index_maps': {k: np.mean(v) for k, v in video_frame_index_maps.items()},
                'total_frames': len(video_maps)
            }
            print(f"   ✅ Video mAP@0.5: {np.mean(video_maps):.4f} ({len(video_maps)} frames)")
    
    return validation_results

def validate_video_sequence_from_dataset(model, dataset, metrics, max_objects, device, max_videos, use_magnitude=True, model_type='id_tracker'):
    """Validate using dataset samples - process ALL GOPs for each video.
    
    Args:
        use_magnitude: If True, use 3-channel motion vectors (X, Y, Magnitude)
                      If False, use 2-channel motion vectors (X, Y only)
        model_type: Type of model ('mv_center' or 'id_tracker')
    """
    
    print(f"📊 Processing dataset samples for comprehensive mAP validation...")
    
    # Group samples by video name and GOP (more comprehensive approach)
    from collections import defaultdict
    video_gop_groups = defaultdict(lambda: defaultdict(list))
    
    print(f"🔄 Grouping {len(dataset)} samples by video and GOP...")
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            if sample is not None and 'sequence_id' in sample:
                seq_id = sample['sequence_id']
                
                # Extract video name and GOP index from sequence_id
                # Format: "MOT17-05-SDP_640x640_gop50_500frames_gop0"
                # We need to split from the right to get the actual GOP index
                if '_gop' in seq_id:
                    # Find the last occurrence of '_gop' which is the actual GOP index
                    last_gop_pos = seq_id.rfind('_gop')
                    if last_gop_pos != -1:
                        video_name = seq_id[:last_gop_pos]  # Everything before the last "_gop"
                        gop_part = seq_id[last_gop_pos + 4:]  # Everything after "_gop" 
                        try:
                            gop_idx = int(gop_part)  # Should be 0, 1, 2, etc.
                            video_gop_groups[video_name][gop_idx].append((idx, sample))
                        except ValueError:
                            # Skip if GOP index is not a valid integer
                            continue
                
        except Exception as e:
            print(f"⚠️ Error loading sample {idx}: {e}")
            continue
    
    print(f"📊 Found {len(video_gop_groups)} videos with GOP sequences:")
    for video_name, gops in video_gop_groups.items():
        print(f"   {video_name}: {len(gops)} GOPs (indices: {sorted(gops.keys())})")
    
    # Limit to max_videos if specified
    if max_videos > 0:
        video_names = list(video_gop_groups.keys())[:max_videos]
        video_gop_groups = {name: video_gop_groups[name] for name in video_names}
        print(f"📊 Limited to {len(video_gop_groups)} videos for processing")
    
    # Process each video with all its GOPs
    validation_results = {}
    
    for video_name, gop_dict in video_gop_groups.items():
        print(f"\n🎬 Validating video: {video_name}")
        print(f"   Processing {len(gop_dict)} GOPs: {sorted(gop_dict.keys())}")
        
        # Initialize video-level tracking
        video_maps = []
        video_gop_maps = {}
        video_frame_index_maps = defaultdict(list)  # Track mAP per frame index within GOP
        video_static_baseline_maps = defaultdict(list)  # Track static baseline mAP (I-frame boxes for all frames)
        video_motion_baseline_maps = defaultdict(list)  # Track motion baseline mAP (I-frame boxes + mean MV translation)
        total_frames_processed = 0
        video_motion_coverage = {'x_nonzero': 0, 'y_nonzero': 0, 'total_frames': 0}
        
        # Process each GOP in order
        for gop_idx in sorted(gop_dict.keys()):
            frames_list = gop_dict[gop_idx]
            print(f"   🔄 Processing GOP {gop_idx}: {len(frames_list)} frames")
            
            # Sort frames by index within GOP
            frames_list.sort(key=lambda x: x[0])
            
            gop_maps = []
            gop_motion_coverage = {'x_nonzero': 0, 'y_nonzero': 0, 'total_frames': 0}
            
            # MEMORY TRACKER: Process entire GOP at once
            if model_type == 'memory_tracker':
                # Extract frame samples
                gop_frames = [sample for _, sample in frames_list]
                
                # Run GOP inference
                try:
                    gop_predictions = run_memory_tracker_gop_inference(model, gop_frames, device, max_objects)
                    
                    # Get I-frame ground truth boxes for static baseline comparison
                    iframe_gt_boxes = None
                    perbox_motion_boxes = None  # Track accumulated per-box motion baseline
                    if len(gop_frames) > 0:
                        boxes_data = gop_frames[0].get('boxes', torch.zeros(1, 4))
                        if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                            iframe_gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                            iframe_gt_boxes = iframe_gt_boxes[iframe_gt_boxes.sum(dim=1) != 0]
                        else:
                            iframe_gt_boxes = torch.zeros(0, 4)
                        
                        # Initialize per-box motion tracking with I-frame boxes
                        if iframe_gt_boxes is not None and len(iframe_gt_boxes) > 0:
                            perbox_motion_boxes = iframe_gt_boxes.clone()
                    
                    # Process each frame's predictions
                    for frame_idx, ((_, sample), pred) in enumerate(zip(frames_list, gop_predictions)):
                        # Get ground truth
                        boxes_data = sample.get('boxes', torch.zeros(1, 4))
                        if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                            gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                            gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
                        else:
                            gt_boxes = torch.zeros(0, 4)
                        
                        # Get predictions
                        pred_boxes = pred['boxes']  # [N, 4]
                        pred_confidences = pred['confidences']  # [N]
                        
                        # Create valid mask for predictions
                        num_pred = len(pred_boxes)
                        valid_mask_frame = torch.zeros(max_objects, dtype=torch.bool, device=device)
                        if num_pred > 0:
                            valid_mask_frame[:num_pred] = True
                        
                        # Pad predictions to max_objects
                        padded_boxes = torch.zeros(max_objects, 4, device=device)
                        if num_pred > 0:
                            padded_boxes[:num_pred] = pred_boxes
                        
                        # Calculate mAP for model predictions
                        frame_maps = metrics.calculate_map_for_frame(
                            padded_boxes.cpu(), gt_boxes, valid_mask_frame.cpu()
                        )
                        
                        # Calculate static baseline mAP (using I-frame GT boxes as predictions)
                        # This shows what mAP would be if we didn't track at all (just kept I-frame boxes)
                        static_baseline_map = 0.0
                        motion_baseline_map = 0.0
                        
                        if iframe_gt_boxes is not None and len(iframe_gt_boxes) > 0:
                            # Pad I-frame boxes to max_objects
                            num_iframe_boxes = len(iframe_gt_boxes)
                            static_padded_boxes = torch.zeros(max_objects, 4)
                            static_valid_mask = torch.zeros(max_objects, dtype=torch.bool)
                            if num_iframe_boxes > 0:
                                static_padded_boxes[:num_iframe_boxes] = iframe_gt_boxes
                                static_valid_mask[:num_iframe_boxes] = True
                            
                            # Calculate mAP using I-frame boxes as "predictions"
                            static_frame_maps = metrics.calculate_map_for_frame(
                                static_padded_boxes, gt_boxes, static_valid_mask
                            )
                            static_baseline_map = static_frame_maps.get('mAP@0.5', 0)
                            
                            # Calculate motion baseline mAP (I-frame boxes + mean motion vector translation)
                            # This shows what mAP would be with simple motion-based prediction (no learning)
                            if frame_idx > 0:  # Only for P-frames (not I-frame)
                                # Get motion vectors for this frame
                                motion_data = sample.get('motion_vectors', None)
                                if motion_data is not None and isinstance(motion_data, torch.Tensor) and motion_data.numel() > 0:
                                    # Motion vectors shape: [2, H, W, 2]
                                    # First dimension (2): displacement types (forward/backward reference)
                                    # [H, W]: spatial dimensions
                                    # Last dimension (2): X and Y coordinates
                                    
                                    if len(motion_data.shape) == 4 and motion_data.shape[0] == 2:
                                        # Use first displacement type (index 0)
                                        # Shape becomes [H, W, 2]
                                        motion_frame_data = motion_data[0]
                                    elif len(motion_data.shape) == 3:
                                        # Already in [H, W, 2] format
                                        motion_frame_data = motion_data
                                    else:
                                        # Unexpected shape, skip
                                        motion_frame_data = None
                                    
                                    if motion_frame_data is not None:
                                        # Extract X and Y components
                                        mv_x = motion_frame_data[:, :, 0]  # [H, W] - X component
                                        mv_y = motion_frame_data[:, :, 1]  # [H, W] - Y component
                                    else:
                                        continue
                                    
                                    # 🔍 DEBUG: Inspect motion vector data (only for first video, first few frames)
                                    if gop_idx == 0 and frame_idx <= 5 and list(video_gop_groups.keys()).index(video_name) == 0:
                                        print(f"\n{'='*70}")
                                        print(f"🔍 MOTION VECTOR INSPECTION - Video {video_name}, GOP {gop_idx}, Frame {frame_idx}")
                                        print(f"{'='*70}")
                                        print(f"Raw motion data shape: {motion_data.shape}")
                                        print(f"Motion data dtype: {motion_data.dtype}")
                                        print(f"Motion data device: {motion_data.device}")
                                        if motion_frame_data is not None:
                                            print(f"Frame-specific motion data shape: {motion_frame_data.shape}")
                                        if len(motion_data.shape) == 4:
                                            print(f"Using first displacement type (index 0) from shape {motion_data.shape}")
                                        print(f"\nX component (mv_x):")
                                        print(f"  Shape: {mv_x.shape}")
                                        print(f"  Min: {mv_x.min().item():.4f}, Max: {mv_x.max().item():.4f}")
                                        print(f"  Mean (all): {mv_x.mean().item():.4f}, Std: {mv_x.std().item():.4f}")
                                        nonzero_x_debug = mv_x[mv_x != 0]
                                        print(f"  Non-zero count: {len(nonzero_x_debug)} / {mv_x.numel()} ({100*len(nonzero_x_debug)/mv_x.numel():.2f}%)")
                                        if len(nonzero_x_debug) > 0:
                                            print(f"  Non-zero mean: {nonzero_x_debug.mean().item():.4f}, std: {nonzero_x_debug.std().item():.4f}")
                                        
                                        print(f"\nY component (mv_y):")
                                        print(f"  Shape: {mv_y.shape}")
                                        print(f"  Min: {mv_y.min().item():.4f}, Max: {mv_y.max().item():.4f}")
                                        print(f"  Mean (all): {mv_y.mean().item():.4f}, Std: {mv_y.std().item():.4f}")
                                        nonzero_y_debug = mv_y[mv_y != 0]
                                        print(f"  Non-zero count: {len(nonzero_y_debug)} / {mv_y.numel()} ({100*len(nonzero_y_debug)/mv_y.numel():.2f}%)")
                                        if len(nonzero_y_debug) > 0:
                                            print(f"  Non-zero mean: {nonzero_y_debug.mean().item():.4f}, std: {nonzero_y_debug.std().item():.4f}")
                                        else:
                                            print(f"  ⚠️  ALL Y VALUES ARE ZERO!")
                                        
                                        print(f"\nSample values (top-left 5x5):")
                                        print(f"  X: {mv_x[:5, :5]}")
                                        print(f"  Y: {mv_y[:5, :5]}")
                                        print(f"{'='*70}\n")
                                    
                                    # Per-box motion baseline: extract motion vectors within each box region
                                    # Use previously updated boxes (accumulated motion through frames)
                                    motion_translated_boxes = perbox_motion_boxes.clone()
                                    
                                    # Get motion grid dimensions
                                    mv_height, mv_width = mv_x.shape
                                    
                                    # Debug: Print per-box motion for first video, first few frames
                                    if gop_idx == 0 and frame_idx <= 3 and list(video_gop_groups.keys()).index(video_name) == 0:
                                        print(f"\n🔍 PER-BOX MOTION BASELINE - Frame {frame_idx}")
                                        print(f"Number of boxes: {len(motion_translated_boxes)}")
                                        print(f"Motion grid size: {mv_height}x{mv_width}")
                                    
                                    # For each box, extract motion vectors in its region and translate
                                    for box_idx in range(len(motion_translated_boxes)):
                                        box = motion_translated_boxes[box_idx]  # [cx, cy, w, h] normalized
                                        
                                        # Convert normalized coords to motion vector grid coords
                                        cx, cy, w, h = box
                                        x1 = int(max(0, (cx - w/2) * mv_width))
                                        y1 = int(max(0, (cy - h/2) * mv_height))
                                        x2 = int(min(mv_width, (cx + w/2) * mv_width))
                                        y2 = int(min(mv_height, (cy + h/2) * mv_height))
                                        
                                        # Extract motion vectors within this box region
                                        if x2 > x1 and y2 > y1:
                                            box_mv_x = mv_x[y1:y2, x1:x2]
                                            box_mv_y = mv_y[y1:y2, x1:x2]
                                            
                                            # Calculate mean motion for this box (ignoring zeros)
                                            nonzero_mask_x = box_mv_x != 0
                                            nonzero_mask_y = box_mv_y != 0
                                            
                                            box_mean_mv_x = box_mv_x[nonzero_mask_x].mean() if nonzero_mask_x.any() else 0.0
                                            box_mean_mv_y = box_mv_y[nonzero_mask_y].mean() if nonzero_mask_y.any() else 0.0
                                            
                                            # Debug: Print per-box motion details
                                            if gop_idx == 0 and frame_idx <= 3 and list(video_gop_groups.keys()).index(video_name) == 0 and box_idx < 3:
                                                print(f"  Box {box_idx}: region=[{y1}:{y2}, {x1}:{x2}], "
                                                      f"mvs={nonzero_mask_x.sum().item()}/{box_mv_x.numel()}, "
                                                      f"motion=({box_mean_mv_x:.2f}, {box_mean_mv_y:.2f})")
                                            
                                            # Translate this box by its local motion (in pixels, normalized to image size)
                                            motion_translated_boxes[box_idx, 0] += box_mean_mv_x / 640.0  # cx += mv_x
                                            motion_translated_boxes[box_idx, 1] += box_mean_mv_y / 640.0  # cy += mv_y
                                            
                                            # Clamp to valid range [0, 1]
                                            motion_translated_boxes[box_idx, 0] = torch.clamp(motion_translated_boxes[box_idx, 0], 0, 1)
                                            motion_translated_boxes[box_idx, 1] = torch.clamp(motion_translated_boxes[box_idx, 1], 0, 1)
                                    
                                    # Update accumulated per-box motion for next frame
                                    perbox_motion_boxes = motion_translated_boxes.clone()
                                    
                                    # Pad motion-translated boxes
                                    motion_padded_boxes = torch.zeros(max_objects, 4)
                                    motion_valid_mask = torch.zeros(max_objects, dtype=torch.bool)
                                    if num_iframe_boxes > 0:
                                        motion_padded_boxes[:num_iframe_boxes] = motion_translated_boxes
                                        motion_valid_mask[:num_iframe_boxes] = True
                                    
                                    # Calculate mAP using motion-translated boxes
                                    motion_frame_maps = metrics.calculate_map_for_frame(
                                        motion_padded_boxes, gt_boxes, motion_valid_mask
                                    )
                                    motion_baseline_map = motion_frame_maps.get('mAP@0.5', 0)
                                else:
                                    # No motion vectors, use static baseline
                                    motion_baseline_map = static_baseline_map
                            else:
                                # For I-frame, motion baseline = static baseline (no motion to apply)
                                motion_baseline_map = static_baseline_map
                        
                        # Store frame result
                        frame_result = {
                            'video_name': video_name,
                            'gop_idx': gop_idx,
                            'frame_idx': frame_idx,
                            'global_frame_idx': total_frames_processed,
                            **frame_maps,
                            'num_predictions': num_pred,
                            'num_gt': len(gt_boxes),
                        }
                        
                        gop_maps.append(frame_result)
                        video_maps.append(frame_result)
                        video_frame_index_maps[frame_idx].append(frame_maps.get('mAP@0.5', 0))
                        video_static_baseline_maps[frame_idx].append(static_baseline_map)
                        video_motion_baseline_maps[frame_idx].append(motion_baseline_map)
                        total_frames_processed += 1
                        
                        print(f"     Frame {frame_idx}: Model={frame_maps.get('mAP@0.5', 0):.3f}, Static={static_baseline_map:.3f}, Motion={motion_baseline_map:.3f}, Pred={num_pred}, GT={len(gt_boxes)}")
                    
                except Exception as e:
                    print(f"     ❌ Error processing GOP {gop_idx} with memory tracker: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # FRAME-BY-FRAME MODELS (mv_center, id_tracker)
            else:
                # Initialize tracking state for this GOP
                prev_boxes = None
                object_ids = None
                valid_mask = None
            
                with torch.no_grad():
                    for frame_idx, (_, sample) in enumerate(frames_list):
                        try:
                            # Extract motion vectors from sample
                            motion_vectors = sample.get('motion_vectors')
                            boxes_data = sample.get('boxes', torch.zeros(1, 4))
                            
                            if motion_vectors is None:
                                continue
                            
                            # Handle motion vector format (same as before)
                            if isinstance(motion_vectors, torch.Tensor):
                                if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                                    motion_vectors = motion_vectors[0].permute(2, 0, 1)
                                elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                                    motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                                
                                if motion_vectors.shape[0] != 2:
                                    motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else torch.zeros(2, 40, 40)
                            
                            # Motion vector processing: optionally add magnitude
                            mv_x, mv_y = motion_vectors[0], motion_vectors[1]  # Extract X and Y components
                            
                            if use_magnitude:
                                # ENHANCEMENT: Add magnitude as third channel (X, Y, Magnitude)
                                mv_magnitude = torch.sqrt(mv_x**2 + mv_y**2)  # Calculate magnitude
                                motion_vectors_final = torch.stack([mv_x, mv_y, mv_magnitude], dim=0)  # (3, 40, 40)
                            else:
                                # BASELINE: Use only X and Y components (2 channels)
                                motion_vectors_final = torch.stack([mv_x, mv_y], dim=0)  # (2, 40, 40)
                            
                            motion_tensor = motion_vectors_final.unsqueeze(0).to(device)  # (1, C, 40, 40)
                            
                            # Track motion vector coverage
                            gop_motion_coverage['total_frames'] += 1
                            video_motion_coverage['total_frames'] += 1
                            
                            if torch.any(motion_vectors[0] != 0):  # X component
                                gop_motion_coverage['x_nonzero'] += 1
                                video_motion_coverage['x_nonzero'] += 1
                            if torch.any(motion_vectors[1] != 0):  # Y component
                                gop_motion_coverage['y_nonzero'] += 1
                                video_motion_coverage['y_nonzero'] += 1
                            
                            # Prepare ground truth boxes
                            if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                                gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                                gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
                            else:
                                gt_boxes = torch.zeros(0, 4)
                            
                            # Initialize tracking state for first frame of GOP
                            if frame_idx == 0 or prev_boxes is None:
                                prev_boxes = torch.zeros(1, max_objects, 4, device=device)
                                object_ids = torch.zeros(1, max_objects, dtype=torch.long, device=device)
                                valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
                                
                                # Initialize with ground truth if available
                                if len(gt_boxes) > 0:
                                    num_objects = min(len(gt_boxes), max_objects)
                                    prev_boxes[0, :num_objects] = gt_boxes[:num_objects].to(device)
                                    object_ids[0, :num_objects] = torch.arange(1, num_objects + 1, device=device)
                                    valid_mask[0, :num_objects] = True
                                    print(f"     🎯 I-FRAME {frame_idx}: Initialized with {num_objects} GT objects")
                            
                            # Handle I-frame vs P-frame prediction
                            if frame_idx == 0:
                                # I-FRAME: Use GT directly (perfect tracking baseline)
                                predicted_boxes = prev_boxes.clone()  # Use GT as "prediction"
                                print(f"     📊 I-FRAME {frame_idx}: Using GT as prediction (expected mAP = 1.0)")
                                print(f"         GT boxes: {len(gt_boxes)}")
                                print(f"         Predicted boxes (GT): {predicted_boxes[0][:len(gt_boxes) if len(gt_boxes) > 0 else 1]}")
                            else:
                                # P-FRAME: Use model prediction with motion vectors
                                predicted_boxes, _, _ = run_model_inference(
                                    model, model_type, motion_tensor, prev_boxes, object_ids, valid_mask, max_objects
                                )
                                print(f"     🔮 P-FRAME {frame_idx}: Model prediction ({model_type})")
                            
                            # Calculate mAP for this frame
                            frame_maps = metrics.calculate_map_for_frame(
                                predicted_boxes[0].cpu(), gt_boxes, valid_mask[0].cpu()
                            )
                            
                            # Store frame result
                            frame_result = {
                                'video_name': video_name,
                                'gop_idx': gop_idx,
                                'frame_idx': frame_idx,
                                'global_frame_idx': total_frames_processed,
                                **frame_maps,
                                'num_predictions': valid_mask[0].sum().item(),
                                'num_gt': len(gt_boxes),
                            }
                            
                            gop_maps.append(frame_result)
                            video_maps.append(frame_result)
                            
                            # Track mAP per frame index within GOP for cross-GOP analysis
                            video_frame_index_maps[frame_idx].append(frame_maps.get('mAP@0.5', 0))
                            
                            # Update tracking state
                            prev_boxes = predicted_boxes.detach().clone()
                            total_frames_processed += 1
                            
                        except Exception as e:
                            print(f"     ❌ Error processing GOP {gop_idx} frame {frame_idx}: {e}")
                            continue
            
            # Calculate GOP-level averages
            if gop_maps:
                gop_avg_maps = {}
                for thresh in metrics.iou_thresholds:
                    map_key = f'mAP@{thresh}'
                    gop_avg_maps[map_key] = np.mean([f[map_key] for f in gop_maps])
                
                # Calculate GOP motion coverage
                if gop_motion_coverage['total_frames'] > 0:
                    gop_x_coverage = (gop_motion_coverage['x_nonzero'] / gop_motion_coverage['total_frames']) * 100
                    gop_y_coverage = (gop_motion_coverage['y_nonzero'] / gop_motion_coverage['total_frames']) * 100
                else:
                    gop_x_coverage = gop_y_coverage = 0.0
                
                video_gop_maps[gop_idx] = {
                    'avg_map': gop_avg_maps,
                    'frames_processed': len(gop_maps),
                    'motion_coverage': {
                        'x_component': gop_x_coverage,
                        'y_component': gop_y_coverage,
                        'y_working': gop_y_coverage > 0
                    },
                    'frame_maps': gop_maps
                }
                
                print(f"     ✅ GOP {gop_idx}: {len(gop_maps)} frames, mAP@0.5: {gop_avg_maps.get('mAP@0.5', 0):.3f}")
            else:
                print(f"     ⚠️ GOP {gop_idx}: No frames processed")
        
        # Calculate video-level averages
        if video_maps:
            video_avg_maps = {}
            for thresh in metrics.iou_thresholds:
                map_key = f'mAP@{thresh}'
                video_avg_maps[map_key] = np.mean([f[map_key] for f in video_maps])
            
            # Calculate static baseline average mAP
            static_baseline_avg = {}
            if video_static_baseline_maps:
                all_static_maps = []
                for frame_maps in video_static_baseline_maps.values():
                    all_static_maps.extend(frame_maps)
                static_baseline_avg['mAP@0.5'] = np.mean(all_static_maps) if all_static_maps else 0.0
            
            # Calculate per-box motion baseline average mAP
            motion_baseline_avg = {}
            if video_motion_baseline_maps:
                all_motion_maps = []
                for frame_maps in video_motion_baseline_maps.values():
                    all_motion_maps.extend(frame_maps)
                motion_baseline_avg['mAP@0.5'] = np.mean(all_motion_maps) if all_motion_maps else 0.0
            
            # Calculate video motion coverage
            if video_motion_coverage['total_frames'] > 0:
                video_x_coverage = (video_motion_coverage['x_nonzero'] / video_motion_coverage['total_frames']) * 100
                video_y_coverage = (video_motion_coverage['y_nonzero'] / video_motion_coverage['total_frames']) * 100
            else:
                video_x_coverage = video_y_coverage = 0.0
            
            validation_results[video_name] = {
                'frames_processed': total_frames_processed,
                'gops_processed': len(video_gop_maps),
                'avg_map': video_avg_maps,
                'static_baseline_avg': static_baseline_avg,
                'motion_baseline_avg': motion_baseline_avg,
                'gop_maps': video_gop_maps,
                'frame_maps': video_maps,
                'frame_index_maps': dict(video_frame_index_maps),  # mAP per frame index within GOP
                'static_baseline_maps': dict(video_static_baseline_maps),  # Static baseline mAP (I-frame boxes only)
                'motion_baseline_maps': dict(video_motion_baseline_maps),  # Motion baseline mAP (I-frame + per-box MV)
                'motion_coverage': {
                    'x_component': video_x_coverage,
                    'y_component': video_y_coverage,
                    'y_working': video_y_coverage > 0
                }
            }
            
            print(f"   ✅ Video {video_name}: {total_frames_processed} frames, {len(video_gop_maps)} GOPs")
            print(f"      Model mAP@0.5: {video_avg_maps.get('mAP@0.5', 0):.3f}")
            print(f"      Static Baseline mAP@0.5: {static_baseline_avg.get('mAP@0.5', 0):.3f}")
            print(f"      Per-Box Motion Baseline mAP@0.5: {motion_baseline_avg.get('mAP@0.5', 0):.3f}")
        else:
            print(f"   ⚠️ Video {video_name}: No frames processed")
    
    return validation_results

def create_map_analysis_plots(validation_results, output_dir):
    """Create comprehensive mAP analysis plots for all videos and their GOPs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data to plot
    video_names = list(validation_results.keys())
    if not video_names:
        print("⚠️ No validation results to plot")
        return
    
    print(f"📊 Creating comprehensive mAP plots for {len(video_names)} videos...")
    
    # Create main analysis plot
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Comprehensive Motion Vector mAP Analysis', fontsize=18, fontweight='bold')
    
    # Plot 1: Video-level mAP comparison
    ax1 = fig.add_subplot(gs[0, 0])
    video_map_05 = [results['avg_map'].get('mAP@0.5', 0) for results in validation_results.values()]
    video_map_075 = [results['avg_map'].get('mAP@0.75', 0) for results in validation_results.values()]
    video_map_09 = [results['avg_map'].get('mAP@0.9', 0) for results in validation_results.values()]
    
    x = np.arange(len(video_names))
    width = 0.25
    
    ax1.bar(x - width, video_map_05, width, label='mAP@0.5', alpha=0.8, color='blue')
    ax1.bar(x, video_map_075, width, label='mAP@0.75', alpha=0.8, color='orange')
    ax1.bar(x + width, video_map_09, width, label='mAP@0.9', alpha=0.8, color='green')
    
    ax1.set_title('Video-level mAP Performance', fontweight='bold')
    ax1.set_xlabel('Videos')
    ax1.set_ylabel('mAP')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in video_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Motion coverage analysis
    ax2 = fig.add_subplot(gs[0, 1])
    x_coverage = [results['motion_coverage']['x_component'] for results in validation_results.values()]
    y_coverage = [results['motion_coverage']['y_component'] for results in validation_results.values()]
    
    ax2.bar(x - width/2, x_coverage, width, label='X-component', alpha=0.8, color='red')
    ax2.bar(x + width/2, y_coverage, width, label='Y-component', alpha=0.8, color='purple')
    
    ax2.set_title('Motion Vector Coverage', fontweight='bold')
    ax2.set_xlabel('Videos')
    ax2.set_ylabel('Coverage (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in video_names], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: GOP count per video
    ax3 = fig.add_subplot(gs[0, 2])
    gop_counts = [results['gops_processed'] for results in validation_results.values()]
    frame_counts = [results['frames_processed'] for results in validation_results.values()]
    
    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(x - width/2, gop_counts, width, label='GOPs', alpha=0.8, color='cyan')
    bars2 = ax3_twin.bar(x + width/2, frame_counts, width, label='Frames', alpha=0.8, color='yellow')
    
    ax3.set_title('Processing Statistics', fontweight='bold')
    ax3.set_xlabel('Videos')
    ax3.set_ylabel('GOP Count', color='cyan')
    ax3_twin.set_ylabel('Frame Count', color='orange')
    ax3.set_xticks(x)
    ax3.set_xticklabels([name[:20] + '...' if len(name) > 20 else name for name in video_names], rotation=45)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4-6: GOP-level mAP trends for each video (one subplot per video)
    video_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (video_name, results) in enumerate(validation_results.items()):
        if i >= 3:  # Limit to 3 videos for main plot
            break
            
        ax = fig.add_subplot(gs[1, i])
        
        if results['gop_maps']:
            gop_indices = sorted(results['gop_maps'].keys())
            gop_map_05 = [results['gop_maps'][gop]['avg_map']['mAP@0.5'] for gop in gop_indices]
            gop_map_075 = [results['gop_maps'][gop]['avg_map']['mAP@0.75'] for gop in gop_indices]
            
            color = video_colors[i % len(video_colors)]
            ax.plot(gop_indices, gop_map_05, 'o-', label='mAP@0.5', linewidth=2, color=color, alpha=0.8)
            ax.plot(gop_indices, gop_map_075, 's--', label='mAP@0.75', linewidth=2, color=color, alpha=0.6)
            
            ax.set_title(f'{video_name[:25]}...\nGOP-level mAP Trends', fontweight='bold', fontsize=10)
            ax.set_xlabel('GOP Index')
            ax.set_ylabel('mAP')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            avg_map_05 = np.mean(gop_map_05)
            std_map_05 = np.std(gop_map_05)
            ax.text(0.02, 0.98, f'μ={avg_map_05:.3f}\nσ={std_map_05:.3f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No GOP data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{video_name[:25]}...\nNo GOP Data', fontweight='bold', fontsize=10)
    
    # Plot 7: All videos Frame Index trends (mAP evolution within GOP structure)
    ax7 = fig.add_subplot(gs[2, :2])
    
    for i, (video_name, results) in enumerate(validation_results.items()):
        if results.get('frame_index_maps'):
            frame_indices = sorted(results['frame_index_maps'].keys())
            # Calculate average mAP for each frame index across all GOPs
            frame_avg_map = [np.mean(results['frame_index_maps'][idx]) for idx in frame_indices]
            
            color = video_colors[i % len(video_colors)]
            ax7.plot(frame_indices, frame_avg_map, 'o-', 
                    label=f'{video_name[:20]}...' if len(video_name) > 20 else video_name, 
                    linewidth=2, color=color, alpha=0.7)
    
    ax7.set_title('All Videos: mAP@0.5 per Frame Index within GOP', fontweight='bold')
    ax7.set_xlabel('Frame Index within GOP (0-47)')
    ax7.set_ylabel('Average mAP@0.5')
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # Add GOP structure annotation
    ax7.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax7.text(1, ax7.get_ylim()[1]*0.9, 'I-frame\n(GOP start)', fontsize=8, alpha=0.7)
    
    # Plot 8: Overall statistics
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Calculate overall statistics
    total_videos = len(validation_results)
    total_gops = sum(results['gops_processed'] for results in validation_results.values())
    total_frames = sum(results['frames_processed'] for results in validation_results.values())
    avg_map_05 = np.mean([results['avg_map'].get('mAP@0.5', 0) for results in validation_results.values()])
    avg_map_075 = np.mean([results['avg_map'].get('mAP@0.75', 0) for results in validation_results.values()])
    avg_x_coverage = np.mean([results['motion_coverage']['x_component'] for results in validation_results.values()])
    avg_y_coverage = np.mean([results['motion_coverage']['y_component'] for results in validation_results.values()])
    y_working_count = sum(1 for results in validation_results.values() if results['motion_coverage']['y_working'])
    
    stats_text = f"""Overall Statistics:

Videos processed: {total_videos}
GOPs processed: {total_gops}
Total frames: {total_frames}

Average mAP Performance:
  mAP@0.5:  {avg_map_05:.3f}
  mAP@0.75: {avg_map_075:.3f}

Motion Vector Coverage:
  X-component: {avg_x_coverage:.1f}%
  Y-component: {avg_y_coverage:.1f}%
  
Y-component working: {y_working_count}/{total_videos} videos

GOP Statistics:
  Avg GOPs per video: {total_gops/total_videos:.1f}
  Avg frames per GOP: {total_frames/total_gops:.1f}
"""
    
    ax8.text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top', 
             fontfamily='monospace', transform=ax8.transAxes)
    
    # Plot 9: mAP distribution histogram
    ax9 = fig.add_subplot(gs[3, 0])
    all_gop_map_05 = []
    for results in validation_results.values():
        if results['gop_maps']:
            for gop_data in results['gop_maps'].values():
                all_gop_map_05.append(gop_data['avg_map']['mAP@0.5'])
    
    if all_gop_map_05:
        ax9.hist(all_gop_map_05, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax9.set_title('GOP-level mAP@0.5 Distribution', fontweight='bold')
        ax9.set_xlabel('mAP@0.5')
        ax9.set_ylabel('Frequency')
        ax9.grid(True, alpha=0.3)
        
        # Add statistics
        mean_map = np.mean(all_gop_map_05)
        std_map = np.std(all_gop_map_05)
        ax9.axvline(mean_map, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_map:.3f}')
        ax9.axvline(mean_map + std_map, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_map + std_map:.3f}')
        ax9.axvline(mean_map - std_map, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_map - std_map:.3f}')
        ax9.legend()
    else:
        ax9.text(0.5, 0.5, 'No mAP data available', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('mAP Distribution')
    
    # Plot 10: Frame-by-frame evolution for first video
    ax10 = fig.add_subplot(gs[3, 1:])
    
    first_video_name = list(validation_results.keys())[0] if validation_results else None
    if first_video_name and validation_results[first_video_name]['frame_maps']:
        frame_maps = validation_results[first_video_name]['frame_maps']
        frame_indices = [f['global_frame_idx'] for f in frame_maps]
        gop_indices = [f['gop_idx'] for f in frame_maps]
        frame_map_05 = [f['mAP@0.5'] for f in frame_maps]
        
        # Color by GOP
        scatter = ax10.scatter(frame_indices, frame_map_05, c=gop_indices, cmap='tab10', alpha=0.7, s=30)
        ax10.plot(frame_indices, frame_map_05, 'b-', alpha=0.3, linewidth=1)
        
        ax10.set_title(f'Frame-by-Frame mAP@0.5 Evolution\n({first_video_name[:30]}...)', fontweight='bold')
        ax10.set_xlabel('Global Frame Index')
        ax10.set_ylabel('mAP@0.5')
        ax10.grid(True, alpha=0.3)
        
        # Add colorbar for GOP indices
        cbar = plt.colorbar(scatter, ax=ax10)
        cbar.set_label('GOP Index')
        
        # Add GOP boundaries
        gop_boundaries = []
        current_gop = -1
        for i, gop_idx in enumerate(gop_indices):
            if gop_idx != current_gop:
                if i > 0:
                    gop_boundaries.append(frame_indices[i])
                current_gop = gop_idx
        
        for boundary in gop_boundaries:
            ax10.axvline(boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax10.text(0.5, 0.5, 'No frame data available', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Frame-by-Frame Evolution')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comprehensive_motion_map_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comprehensive mAP analysis plot saved: {plot_path}")
    
    # Create individual plots for each video
    for video_name, results in validation_results.items():
        if results['gop_maps']:
            create_individual_video_plot(video_name, results, output_dir)

def create_individual_video_plot(video_name, results, output_dir):
    """Create detailed plot for individual video."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Detailed Analysis: {video_name}', fontsize=16, fontweight='bold')
    
    # Extract GOP data for other plots
    gop_indices = sorted(results['gop_maps'].keys())
    gop_map_05 = [results['gop_maps'][gop]['avg_map']['mAP@0.5'] for gop in gop_indices]
    gop_map_075 = [results['gop_maps'][gop]['avg_map']['mAP@0.75'] for gop in gop_indices]
    gop_map_09 = [results['gop_maps'][gop]['avg_map']['mAP@0.9'] for gop in gop_indices]
    
    # Plot 1: mAP trends across Frame Indices within GOP
    if results.get('frame_index_maps'):
        frame_indices = sorted(results['frame_index_maps'].keys())
        frame_avg_map = [np.mean(results['frame_index_maps'][idx]) for idx in frame_indices]
        frame_std_map = [np.std(results['frame_index_maps'][idx]) if len(results['frame_index_maps'][idx]) > 1 else 0 
                        for idx in frame_indices]
        
        axes[0, 0].plot(frame_indices, frame_avg_map, 'o-', label='Average mAP@0.5', linewidth=2, markersize=6, color='blue')
        axes[0, 0].fill_between(frame_indices, 
                               [m - s for m, s in zip(frame_avg_map, frame_std_map)],
                               [m + s for m, s in zip(frame_avg_map, frame_std_map)],
                               alpha=0.3, color='blue', label='±1 Std Dev')
        
        axes[0, 0].set_title('mAP@0.5 Evolution per Frame Index within GOP')
        axes[0, 0].set_xlabel('Frame Index within GOP (0-47)')
        axes[0, 0].set_ylabel('mAP@0.5')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add I-frame indicator
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        axes[0, 0].text(1, max(frame_avg_map)*0.9, 'I-frame\n(GOP start)', fontsize=9, alpha=0.8)
    else:
        axes[0, 0].text(0.5, 0.5, 'No frame index data available', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Frame Index Analysis - No Data')
    
    # Plot 2: Frames processed per GOP
    gop_frame_counts = [results['gop_maps'][gop]['frames_processed'] for gop in gop_indices]
    axes[0, 1].bar(gop_indices, gop_frame_counts, alpha=0.7, color='orange')
    axes[0, 1].set_title('Frames Processed per GOP')
    axes[0, 1].set_xlabel('GOP Index')
    axes[0, 1].set_ylabel('Frame Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Motion coverage per GOP
    gop_x_coverage = [results['gop_maps'][gop]['motion_coverage']['x_component'] for gop in gop_indices]
    gop_y_coverage = [results['gop_maps'][gop]['motion_coverage']['y_component'] for gop in gop_indices]
    
    width = 0.35
    x_pos = np.array(gop_indices)
    axes[1, 0].bar(x_pos - width/2, gop_x_coverage, width, label='X-component', alpha=0.7)
    axes[1, 0].bar(x_pos + width/2, gop_y_coverage, width, label='Y-component', alpha=0.7)
    
    axes[1, 0].set_title('Motion Vector Coverage per GOP')
    axes[1, 0].set_xlabel('GOP Index')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics
    axes[1, 1].axis('off')
    
    stats_text = f"""Video Statistics:

Total GOPs: {len(gop_indices)}
Total Frames: {results['frames_processed']}
Avg Frames per GOP: {results['frames_processed']/len(gop_indices):.1f}

Overall Performance:
  mAP@0.5:  {results['avg_map']['mAP@0.5']:.3f}
  mAP@0.75: {results['avg_map']['mAP@0.75']:.3f}
  mAP@0.9:  {results['avg_map']['mAP@0.9']:.3f}

GOP-level Statistics:
  Best mAP@0.5: {max(gop_map_05):.3f} (GOP {gop_indices[np.argmax(gop_map_05)]})
  Worst mAP@0.5: {min(gop_map_05):.3f} (GOP {gop_indices[np.argmin(gop_map_05)]})
  mAP@0.5 Std: {np.std(gop_map_05):.3f}

Motion Coverage:
  X-component: {results['motion_coverage']['x_component']:.1f}%
  Y-component: {results['motion_coverage']['y_component']:.1f}%
  Y-component working: {results['motion_coverage']['y_working']}
"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, fontsize=11, verticalalignment='top', 
                    fontfamily='monospace', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    safe_video_name = video_name.replace('/', '_').replace('\\', '_')
    plot_path = os.path.join(output_dir, f'video_analysis_{safe_video_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Individual video plot saved: {plot_path}")

def create_comparison_plots(dl_results, baseline_results, output_dir, model_name):
    """Create comparison plots between deep learning and motion vector baseline."""
    print(f"\n📊 Creating comparison plots: {model_name} vs Motion Vector Propagation")
    
    # Prepare data for comparison
    video_names = []
    dl_maps = []
    baseline_maps = []
    
    for video_name in dl_results.keys():
        if video_name in baseline_results:
            video_names.append(video_name.split('_gop')[0])  # Clean video name
            dl_maps.append(dl_results[video_name]['avg_map']['mAP@0.5'])
            baseline_maps.append(baseline_results[video_name]['avg_map']['mAP@0.5'])
    
    if not video_names:
        print("⚠️ No overlapping videos for comparison")
        return
    
    # Create comprehensive comparison plot
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Bar chart comparing mAP scores
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(video_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_maps, width, label='Motion Vector Propagation', 
                    color='#FF6B6B', alpha=0.8)
    bars2 = ax1.bar(x + width/2, dl_maps, width, label=f'{model_name}',
                    color='#4ECDC4', alpha=0.8)
    
    ax1.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
    ax1.set_title('Method Comparison: mAP@0.5 per Video', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(video_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Improvement percentage
    ax2 = plt.subplot(2, 3, 2)
    improvements = [(dl - bl) / bl * 100 if bl > 0 else 0 
                   for dl, bl in zip(dl_maps, baseline_maps)]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(x, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name} Improvement over Baseline', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(video_names, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    
    # 3. Scatter plot: baseline vs deep learning
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(baseline_maps, dl_maps, s=100, alpha=0.6, c='blue', edgecolors='black')
    
    # Add diagonal line (y=x) for reference
    min_val = min(min(baseline_maps), min(dl_maps))
    max_val = max(max(baseline_maps), max(dl_maps))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance')
    
    ax3.set_xlabel('Motion Vector Propagation mAP@0.5', fontsize=12, fontweight='bold')
    ax3.set_ylabel(f'{model_name} mAP@0.5', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Correlation', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Add video labels to points
    for i, name in enumerate(video_names):
        ax3.annotate(name, (baseline_maps[i], dl_maps[i]), 
                    fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    # 4. Overall statistics
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    baseline_mean = np.mean(baseline_maps)
    dl_mean = np.mean(dl_maps)
    overall_improvement = (dl_mean - baseline_mean) / baseline_mean * 100 if baseline_mean > 0 else 0
    
    stats_text = f"""
📊 OVERALL STATISTICS

Motion Vector Propagation:
  Mean mAP@0.5:  {baseline_mean:.4f}
  Std Dev:       {np.std(baseline_maps):.4f}
  Min:           {min(baseline_maps):.4f}
  Max:           {max(baseline_maps):.4f}

{model_name}:
  Mean mAP@0.5:  {dl_mean:.4f}
  Std Dev:       {np.std(dl_maps):.4f}
  Min:           {min(dl_maps):.4f}
  Max:           {max(dl_maps):.4f}

COMPARISON:
  Overall Improvement: {overall_improvement:+.2f}%
  Videos Analyzed:     {len(video_names)}
  Winner:              {"Deep Learning" if overall_improvement > 0 else "Baseline" if overall_improvement < 0 else "Tie"}
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Box plot comparison
    ax5 = plt.subplot(2, 3, 5)
    box_data = [baseline_maps, dl_maps]
    bp = ax5.boxplot(box_data, labels=['Baseline', model_name.replace('_', ' ')],
                     patch_artist=True, showmeans=True)
    
    colors_box = ['#FF6B6B', '#4ECDC4']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
    ax5.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Frame-by-frame comparison (if available)
    ax6 = plt.subplot(2, 3, 6)
    
    # Aggregate frame index mAP for both methods
    baseline_frame_maps = defaultdict(list)
    dl_frame_maps = defaultdict(list)
    
    for video_name in dl_results.keys():
        if video_name in baseline_results:
            # Baseline frame maps
            if 'frame_index_maps' in baseline_results[video_name]:
                for frame_idx, map_val in baseline_results[video_name]['frame_index_maps'].items():
                    baseline_frame_maps[frame_idx].append(map_val)
            
            # DL frame maps
            if 'frame_index_maps' in dl_results[video_name]:
                for frame_idx, map_val in dl_results[video_name]['frame_index_maps'].items():
                    dl_frame_maps[frame_idx].append(map_val)
    
    if baseline_frame_maps and dl_frame_maps:
        frame_indices = sorted(set(baseline_frame_maps.keys()) & set(dl_frame_maps.keys()))
        baseline_avg = [np.mean(baseline_frame_maps[i]) for i in frame_indices]
        dl_avg = [np.mean(dl_frame_maps[i]) for i in frame_indices]
        
        ax6.plot(frame_indices, baseline_avg, 'o-', label='Baseline', color='#FF6B6B', linewidth=2, markersize=4)
        ax6.plot(frame_indices, dl_avg, 's-', label=model_name, color='#4ECDC4', linewidth=2, markersize=4)
        ax6.set_xlabel('Frame Index in GOP', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Average mAP@0.5', fontsize=12, fontweight='bold')
        ax6.set_title('Temporal Performance Degradation', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'Frame-by-frame data\nnot available', 
                ha='center', va='center', fontsize=12)
        ax6.axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'comparison_{model_name}_vs_baseline.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved: {plot_path}")
    
    # Also create individual comparison plots for each video
    for video_name in dl_results.keys():
        if video_name in baseline_results:
            create_video_comparison_plot(
                dl_results[video_name], 
                baseline_results[video_name],
                video_name,
                model_name,
                output_dir
            )

def create_temporal_dependency_plot(validation_results, output_dir, model_name="Model", static_baseline_maps=None, motion_baseline_maps=None):
    """
    Create a dedicated plot showing mAP vs GOP frame index (0-47).
    This visualizes if the model handles temporal dependencies correctly.
    A well-trained temporal model should maintain stable mAP across all frame indices.
    
    Args:
        validation_results: Dict of video results with frame_index_maps
        output_dir: Where to save the plot
        model_name: Name to display in title
        static_baseline_maps: Optional dict mapping frame_idx -> list of mAP values
                              for static baseline (I-frame boxes repeated)
        motion_baseline_maps: Optional dict mapping frame_idx -> list of mAP values
                              for motion baseline (I-frame boxes + mean MV translation)
    """
    print(f"\n📈 Creating temporal dependency analysis plot...")
    
    # Aggregate frame_index_maps from all videos
    all_frame_indices = set()
    frame_index_to_maps = defaultdict(list)
    
    for video_name, results in validation_results.items():
        if 'frame_index_maps' in results and results['frame_index_maps']:
            for frame_idx, map_values in results['frame_index_maps'].items():
                all_frame_indices.add(frame_idx)
                if isinstance(map_values, list):
                    frame_index_to_maps[frame_idx].extend(map_values)
                else:
                    frame_index_to_maps[frame_idx].append(map_values)
    
    if not frame_index_to_maps:
        print("⚠️  No frame index data available for temporal dependency plot")
        return
    
    # Calculate statistics for each frame index
    frame_indices = sorted(frame_index_to_maps.keys())
    mean_maps = [np.mean(frame_index_to_maps[idx]) for idx in frame_indices]
    std_maps = [np.std(frame_index_to_maps[idx]) for idx in frame_indices]
    min_maps = [np.min(frame_index_to_maps[idx]) for idx in frame_indices]
    max_maps = [np.max(frame_index_to_maps[idx]) for idx in frame_indices]
    
    # Calculate static baseline if provided
    static_mean_maps = None
    if static_baseline_maps:
        static_mean_maps = [np.mean(static_baseline_maps.get(idx, [0])) for idx in frame_indices]
    
    # Calculate motion baseline if provided
    motion_mean_maps = None
    if motion_baseline_maps:
        motion_mean_maps = [np.mean(motion_baseline_maps.get(idx, [0])) for idx in frame_indices]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(f'Temporal Dependency Analysis: {model_name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean mAP with standard deviation + Static Baseline
    ax1.plot(frame_indices, mean_maps, 'o-', linewidth=2.5, markersize=8, 
             color='#2E86C1', label='Model mAP@0.5 (Temporal Tracking)', zorder=3)
    ax1.fill_between(frame_indices,
                     [m - s for m, s in zip(mean_maps, std_maps)],
                     [m + s for m, s in zip(mean_maps, std_maps)],
                     alpha=0.3, color='#2E86C1', label='±1 Std Dev', zorder=2)
    
    # Add static baseline (I-frame boxes repeated for all frames)
    if static_mean_maps:
        ax1.plot(frame_indices, static_mean_maps, 'x--', linewidth=2, markersize=7,
                 color='#E67E22', label='Static Baseline (No Tracking)', alpha=0.8, zorder=2)
    
    # Add motion baseline (I-frame boxes + mean motion vector translation)
    if motion_mean_maps:
        ax1.plot(frame_indices, motion_mean_maps, 'd--', linewidth=2, markersize=6,
                 color='#9B59B6', label='Motion Baseline (Per-Box MV)', alpha=0.8, zorder=2)
        
        # Calculate improvement over baselines
        improvement_static = np.mean(mean_maps) - np.mean(static_mean_maps) if static_mean_maps else 0
        improvement_motion = np.mean(mean_maps) - np.mean(motion_mean_maps)
        improvement_pct_static = (improvement_static / np.mean(static_mean_maps) * 100) if static_mean_maps and np.mean(static_mean_maps) > 0 else 0
        improvement_pct_motion = (improvement_motion / np.mean(motion_mean_maps) * 100) if np.mean(motion_mean_maps) > 0 else 0
        
        info_text = f'Model Mean: {np.mean(mean_maps):.4f}\n'
        if static_mean_maps:
            info_text += f'Static Mean: {np.mean(static_mean_maps):.4f} (Δ {improvement_static:+.4f}, {improvement_pct_static:+.1f}%)\n'
        info_text += f'Motion Mean: {np.mean(motion_mean_maps):.4f} (Δ {improvement_motion:+.4f}, {improvement_pct_motion:+.1f}%)'
        
        ax1.text(0.02, 0.98, info_text,
                 transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    elif static_mean_maps:
        # Only static baseline available
        improvement = np.mean(mean_maps) - np.mean(static_mean_maps)
        improvement_pct = (improvement / np.mean(static_mean_maps) * 100) if np.mean(static_mean_maps) > 0 else 0
        
        ax1.text(0.02, 0.98, 
                 f'Improvement over Static Baseline:\n'
                 f'  Δ mAP = {improvement:+.4f} ({improvement_pct:+.1f}%)',
                 transform=ax1.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Add I-frame marker
    ax1.axvline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, 
                label='I-frame (GOP start)', zorder=1)
    ax1.axhline(np.mean(mean_maps), color='green', linestyle=':', linewidth=2, 
                alpha=0.6, label=f'Model Mean: {np.mean(mean_maps):.3f}', zorder=1)
    
    ax1.set_xlabel('Frame Index within GOP (0 = I-frame, 1-47 = P-frames)', fontsize=13)
    ax1.set_ylabel('mAP@0.5', fontsize=13)
    ax1.set_title('Mean mAP@0.5 Evolution Across GOP Frames', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-1, max(frame_indices) + 1)
    
    # Add annotation for temporal stability
    mAP_range = max(mean_maps) - min(mean_maps)
    stability_text = f"mAP Range: {mAP_range:.4f}\n"
    if mAP_range < 0.05:
        stability_text += "✅ Excellent temporal stability"
        color = 'green'
    elif mAP_range < 0.10:
        stability_text += "⚠️  Moderate temporal stability"
        color = 'orange'
    else:
        stability_text += "❌ Poor temporal stability"
        color = 'red'
    
    ax1.text(0.98, 0.02, stability_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    # Plot 2: Min/Max range (shows variance) + Static Baseline
    ax2.fill_between(frame_indices, min_maps, max_maps, 
                     alpha=0.4, color='#E74C3C', label='Min-Max Range', zorder=2)
    ax2.plot(frame_indices, mean_maps, 'o-', linewidth=2.5, markersize=7,
             color='#2E86C1', label='Model Mean mAP@0.5', zorder=3)
    ax2.plot(frame_indices, min_maps, 's--', linewidth=1.5, markersize=5,
             color='#C0392B', label='Model Min mAP@0.5', alpha=0.7, zorder=3)
    ax2.plot(frame_indices, max_maps, '^--', linewidth=1.5, markersize=5,
             color='#27AE60', label='Model Max mAP@0.5', alpha=0.7, zorder=3)
    
    # Add static baseline (I-frame boxes repeated for all frames)
    if static_mean_maps:
        ax2.plot(frame_indices, static_mean_maps, 'x--', linewidth=2, markersize=7,
                 color='#E67E22', label='Static Baseline (No Tracking)', alpha=0.8, zorder=2)
    
    # Add motion baseline (I-frame boxes + mean motion vector translation)
    if motion_mean_maps:
        ax2.plot(frame_indices, motion_mean_maps, 'd--', linewidth=2, markersize=6,
                 color='#9B59B6', label='Motion Baseline (Per-Box MV)', alpha=0.8, zorder=2)
    
    # Add I-frame marker
    ax2.axvline(0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, 
                label='I-frame (GOP start)', zorder=1)
    
    ax2.set_xlabel('Frame Index within GOP (0 = I-frame, 1-47 = P-frames)', fontsize=13)
    ax2.set_ylabel('mAP@0.5', fontsize=13)
    ax2.set_title('mAP@0.5 Variability Across GOP Frames (Min/Mean/Max)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_xlim(-1, max(frame_indices) + 1)
    
    # Add interpretation guide
    interpretation = """
    Interpretation Guide:
    • Flat line → Good temporal handling (consistent performance across frames)
    • Declining trend → Model struggles with long-term dependencies
    • Increasing trend → Model improves over time (accumulates context)
    • High variance → Unstable predictions across different GOPs
    """
    ax2.text(0.02, 0.02, interpretation, transform=ax2.transAxes,
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'temporal_dependency_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Temporal dependency plot saved: {plot_path}")
    print(f"   Frame indices analyzed: {len(frame_indices)}")
    print(f"   mAP range: {min(mean_maps):.4f} - {max(mean_maps):.4f}")
    print(f"   Overall mean mAP: {np.mean(mean_maps):.4f}")
    
    # Print frame-by-frame statistics
    print(f"\n📊 Frame-by-frame mAP@0.5 statistics:")
    print(f"   Frame 0 (I-frame):  {mean_maps[0]:.4f} ± {std_maps[0]:.4f}")
    if len(mean_maps) >= 24:
        print(f"   Frame 24 (mid-GOP): {mean_maps[23]:.4f} ± {std_maps[23]:.4f}")
    if len(mean_maps) >= 48:
        print(f"   Frame 47 (end-GOP): {mean_maps[47]:.4f} ± {std_maps[47]:.4f}")

def create_video_comparison_plot(dl_video_result, baseline_video_result, video_name, model_name, output_dir):
    """Create detailed comparison plot for a single video."""
    fig = plt.figure(figsize=(16, 10))
    
    # Clean video name
    clean_name = video_name.split('_gop')[0]
    fig.suptitle(f'Video Comparison: {clean_name}', fontsize=16, fontweight='bold')
    
    # 1. GOP-level mAP comparison
    ax1 = plt.subplot(2, 2, 1)
    
    baseline_gop_maps = []
    dl_gop_maps = []
    gop_indices = []
    
    if 'gop_stats' in baseline_video_result and 'gop_stats' in dl_video_result:
        for gop_idx in sorted(set(baseline_video_result['gop_stats'].keys()) & 
                             set(dl_video_result['gop_stats'].keys())):
            gop_indices.append(gop_idx)
            baseline_gop_maps.append(baseline_video_result['gop_stats'][gop_idx]['avg_map'])
            dl_gop_maps.append(dl_video_result['gop_stats'][gop_idx]['avg_map']['mAP@0.5'])
        
        x = np.arange(len(gop_indices))
        width = 0.35
        
        ax1.bar(x - width/2, baseline_gop_maps, width, label='Baseline', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, dl_gop_maps, width, label=model_name, color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('GOP Index', fontweight='bold')
        ax1.set_ylabel('mAP@0.5', fontweight='bold')
        ax1.set_title('GOP-level Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(gop_indices)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
    
    # 2. Frame-level degradation
    ax2 = plt.subplot(2, 2, 2)
    
    if 'frame_index_maps' in baseline_video_result and 'frame_index_maps' in dl_video_result:
        baseline_frames = baseline_video_result['frame_index_maps']
        dl_frames = dl_video_result['frame_index_maps']
        
        frame_indices = sorted(set(baseline_frames.keys()) & set(dl_frames.keys()))
        baseline_vals = [baseline_frames[i] for i in frame_indices]
        dl_vals = [dl_frames[i] for i in frame_indices]
        
        ax2.plot(frame_indices, baseline_vals, 'o-', label='Baseline', color='#FF6B6B', linewidth=2)
        ax2.plot(frame_indices, dl_vals, 's-', label=model_name, color='#4ECDC4', linewidth=2)
        
        ax2.set_xlabel('Frame Index in GOP', fontweight='bold')
        ax2.set_ylabel('mAP@0.5', fontweight='bold')
        ax2.set_title('Frame-by-Frame Performance')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3. Statistics table
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    baseline_map = baseline_video_result['avg_map']['mAP@0.5']
    dl_map = dl_video_result['avg_map']['mAP@0.5']
    improvement = (dl_map - baseline_map) / baseline_map * 100 if baseline_map > 0 else 0
    
    stats_text = f"""
VIDEO: {clean_name}

Baseline (Motion Vector Propagation):
  mAP@0.5:    {baseline_map:.4f}
  Frames:     {baseline_video_result.get('total_frames', 'N/A')}

Deep Learning ({model_name}):
  mAP@0.5:    {dl_map:.4f}
  Frames:     {dl_video_result.get('frames_processed', 'N/A')}

Improvement:  {improvement:+.2f}%
Winner:       {"Deep Learning" if improvement > 0 else "Baseline" if improvement < 0 else "Tie"}
    """
    
    ax3.text(0.1, 0.5, stats_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    safe_video_name = video_name.replace('/', '_').replace('\\', '_')
    plot_path = os.path.join(output_dir, f'video_comparison_{safe_video_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    args = parse_arguments()
    
    print(f"🚀 Starting Motion Vector mAP Validation")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Data: {args.data_path}")
    print(f"📁 Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"🔍 Loading dataset with motion vectors...")
    dataset = scan_for_video_sequences(args.data_path, args.max_videos, args.resolution)
    
    if not dataset:
        print(f"❌ No dataset created")
        return
    
    # Initialize metrics
    metrics = MotionValidationMetrics()
    
    # Storage for all results
    all_results = {}
    
    # 1. Run motion vector propagation baseline (if requested)
    baseline_results = None
    if args.baseline_only or args.compare_baseline:
        print(f"\n{'='*70}")
        print(f"� METHOD 1: Motion Vector Propagation (No Deep Learning)")
        print(f"{'='*70}")
        baseline_results = motion_vector_propagation_baseline(dataset, metrics, 100, args.max_videos)
        all_results['Motion_Vector_Propagation'] = baseline_results
        
        if args.baseline_only:
            # Save baseline-only results
            results_path = os.path.join(args.output_dir, 'baseline_only_results.json')
            with open(results_path, 'w') as f:
                json.dump({'baseline': baseline_results}, f, indent=2)
            print(f"\n✅ Baseline-only validation completed! Results saved to: {results_path}")
            return
    
    # 2. Run deep learning model validation
    device = torch.device(args.device)
    try:
        model, max_objects, use_magnitude, model_type = load_model(args.model_path, device)
        
        # Determine model name based on type
        if model_type == 'mv_center':
            model_name = 'MV_Center_Magnitude' if use_magnitude else 'MV_Center_Baseline'
            model_desc = f"MV-Center {'(u,v,mag)' if use_magnitude else '(u,v)'}"
        else:
            model_name = 'Deep_Learning_Magnitude' if use_magnitude else 'Deep_Learning_Baseline'
            model_desc = f"ID-Tracker {'Magnitude-enhanced (3-channel)' if use_magnitude else 'Baseline (2-channel)'}"
        
        print(f"\n{'='*70}")
        print(f"🎯 METHOD 2: {model_name}")
        print(f"🎨 Model type: {model_desc}")
        print(f"{'='*70}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process sequences using dataset
    print(f"\n🎬 Validating motion vectors for mAP calculation...")
    validation_results = validate_video_sequence_from_dataset(
        model, dataset, metrics, max_objects, device, args.max_videos, 
        use_magnitude=use_magnitude, model_type=model_type
    )
    
    all_results[model_name] = validation_results
    
    # 3. Create comparison plots if baseline was also run
    if args.create_plots:
        print("\n📊 Creating mAP analysis plots...")
        
        # Aggregate static baseline maps across all videos
        from collections import defaultdict
        aggregated_static_baseline = defaultdict(list)
        aggregated_motion_baseline = defaultdict(list)
        for video_name, results in validation_results.items():
            static_maps = results.get('static_baseline_maps', {})
            for frame_idx, map_values in static_maps.items():
                aggregated_static_baseline[frame_idx].extend(map_values)
            
            motion_maps = results.get('motion_baseline_maps', {})
            for frame_idx, map_values in motion_maps.items():
                aggregated_motion_baseline[frame_idx].extend(map_values)
        
        # Create temporal dependency analysis plot (always)
        create_temporal_dependency_plot(validation_results, args.output_dir, model_name, 
                                       static_baseline_maps=dict(aggregated_static_baseline),
                                       motion_baseline_maps=dict(aggregated_motion_baseline))
        
        if args.compare_baseline and baseline_results:
            create_comparison_plots(validation_results, baseline_results, args.output_dir, model_name)
        else:
            create_map_analysis_plots(validation_results, args.output_dir)
    
    # Calculate overall statistics for deep learning model
    total_frames = sum(r['frames_processed'] for r in validation_results.values())
    avg_x_coverage = np.mean([r['motion_coverage']['x_component'] for r in validation_results.values()]) if validation_results else 0.0
    avg_y_coverage = np.mean([r['motion_coverage']['y_component'] for r in validation_results.values()]) if validation_results else 0.0
    y_working = any(r['motion_coverage']['y_working'] for r in validation_results.values()) if validation_results else False
    
    overall_map = {}
    for thresh in metrics.iou_thresholds:
        map_key = f'mAP@{thresh}'
        if validation_results:
            overall_map[map_key] = np.mean([r['avg_map'].get(map_key, 0) for r in validation_results.values()])
        else:
            overall_map[map_key] = 0.0
    
    # Save results
    final_results = {
        'videos': validation_results,
        'overall_stats': {
            'videos_processed': len(validation_results),
            'total_frames': total_frames,
            'motion_coverage': {
                'x_component': avg_x_coverage,
                'y_component': avg_y_coverage,
                'y_working': y_working
            },
            'overall_map': overall_map
        },
        'config': {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'max_videos': args.max_videos,
            'device': args.device,
            'use_magnitude': use_magnitude
        }
    }
    
    if baseline_results:
        final_results['baseline_comparison'] = baseline_results
    
    results_path = os.path.join(args.output_dir, 'motion_map_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✅ Validation completed!")
    print(f"📊 Videos processed: {len(validation_results)}")
    print(f"📊 Average motion coverage:")
    print(f"   X-component: {avg_x_coverage:.1f}%")
    print(f"   Y-component: {avg_y_coverage:.1f}%")
    print(f"📊 Y-component working: {y_working}")
    print(f"📊 Average mAP performance:")
    for thresh in metrics.iou_thresholds:
        print(f"   mAP@{thresh}: {overall_map[f'mAP@{thresh}']:.3f}")
    print(f"📊 Results saved to: {results_path}")
    
    # Print comparison if baseline was run
    if args.compare_baseline and baseline_results:
        print(f"\n{'='*70}")
        print(f"🏆 COMPARISON: Deep Learning vs Motion Vector Propagation")
        print(f"{'='*70}")
        
        # Calculate baseline overall mAP
        baseline_maps = [v['avg_map']['mAP@0.5'] for v in baseline_results.values()]
        baseline_overall = np.mean(baseline_maps) if baseline_maps else 0.0
        
        dl_map = overall_map['mAP@0.5']
        improvement = ((dl_map - baseline_overall) / baseline_overall * 100) if baseline_overall > 0 else 0
        
        print(f"\nmAP@0.5 Scores:")
        print(f"   Motion Vector Propagation: {baseline_overall:.4f}")
        print(f"   Deep Learning ({model_name}): {dl_map:.4f}")
        print(f"   Improvement: {improvement:+.2f}%")
        
        if improvement > 0:
            print(f"\n✅ Deep learning model performs {improvement:.1f}% better than baseline!")
        elif improvement < 0:
            print(f"\n⚠️  Deep learning model performs {abs(improvement):.1f}% worse than baseline!")
        else:
            print(f"\n➖ Performance is similar between methods")
    
    # Print per-video mAP summary with all three methods
    print(f"\n📋 Per-video mAP@0.5 summary:")
    print(f"{'Video':<35} {'Model':<10} {'Static':<10} {'Per-Box Motion':<15}")
    print(f"{'-'*70}")
    for video_name, results in validation_results.items():
        model_map = results['avg_map'].get('mAP@0.5', 0)
        static_map = results.get('static_baseline_avg', {}).get('mAP@0.5', 0)
        motion_map = results.get('motion_baseline_avg', {}).get('mAP@0.5', 0)
        short_name = video_name[:30] + '...' if len(video_name) > 30 else video_name
        print(f"   {short_name:<32} {model_map:<10.3f} {static_map:<10.3f} {motion_map:<15.3f}")

if __name__ == "__main__":
    main()
