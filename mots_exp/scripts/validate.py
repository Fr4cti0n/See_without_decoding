#!/usr/bin/env python3
"""
ID-Aware Multi-Object Tracker Validation Script
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid canvas issues
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import defaultdict

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'dataset'))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate ID-Aware Multi-Object Tracker")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-path", type=str, default="/home/aduche/Bureau/datasets/MOTS/videos/", help="MOTS dataset path")
    parser.add_argument("--max-sequences", type=int, default=5, help="Maximum sequences to process")
    parser.add_argument("--output-dir", type=str, default="outputs/validation", help="Output directory")
    parser.add_argument("--create-videos", action="store_true", help="Create visualization videos")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    return parser.parse_args()

class ValidationMetrics:
    """Compute validation metrics including mAP, tracking accuracy, etc."""
    
    def __init__(self, iou_thresholds=None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75, 0.9]
        self.reset()
    
    def reset(self):
        self.bbox_losses = []
        self.id_losses = []
        self.total_losses = []
        self.ap_scores = {thresh: [] for thresh in self.iou_thresholds}
        self.id_accuracy = []
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes."""
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
    
    def calculate_ap(self, pred_boxes, gt_boxes, valid_mask, iou_threshold):
        """Calculate Average Precision for given IoU threshold."""
        if not valid_mask.any():
            return 0.0
        
        pred_valid = pred_boxes[valid_mask]
        gt_valid = gt_boxes[valid_mask]
        
        if len(gt_valid) == 0:
            return 0.0
        
        # Calculate IoUs for all predictions
        ious = []
        for i in range(len(pred_valid)):
            if i < len(gt_valid):
                iou = self.calculate_iou(pred_valid[i], gt_valid[i])
                ious.append(iou)
        
        # Count true positives
        tp = sum(1 for iou in ious if iou >= iou_threshold)
        precision = tp / len(ious) if len(ious) > 0 else 0.0
        
        return precision
    
    def get_summary(self):
        """Get summary of all metrics."""
        summary = {
            'loss_metrics': {
                'avg_bbox_loss': np.mean(self.bbox_losses) if self.bbox_losses else 0.0,
                'avg_id_loss': np.mean(self.id_losses) if self.id_losses else 0.0,
                'avg_total_loss': np.mean(self.total_losses) if self.total_losses else 0.0,
            },
            'detection_metrics': {},
            'tracking_metrics': {
                'avg_id_accuracy': np.mean(self.id_accuracy) if self.id_accuracy else 0.0,
            }
        }
        
        # Add mAP metrics
        for thresh in self.iou_thresholds:
            if self.ap_scores[thresh]:
                summary['detection_metrics'][f'mAP@{thresh}'] = np.mean(self.ap_scores[thresh])
        
        return summary

def create_rgb_tracking_video(model, sequence_data, sequence_name, output_path, device, max_objects=20):
    """Create video with RGB P-frames background showing predictions (solid) and GT (dashed) through GOP sequence."""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1280, 720)
    fps = 5.0
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    model.eval()
    frames_processed = 0
    
    # Initialize tracking state
    gop_prev_boxes = None
    gop_object_ids = None
    gop_valid_mask = None
    
    try:
        with torch.no_grad():
            # Process all frames in the GOP sequence (48 frames)
            for frame_idx in range(min(len(sequence_data), 48)):
                try:
                    sample = sequence_data[frame_idx]
                    motion_vectors = sample.get('motion_vectors')
                    boxes_data = sample.get('boxes', torch.zeros(1, 4))
                    
                    # For frame 0, use I-frame, for subsequent frames use P-frames
                    if frame_idx == 0:
                        rgb_frame = sample.get('iframe', None)
                    else:
                        rgb_frame = sample.get('pframe', None)
                        if rgb_frame is None:
                            # Fallback to iframe if pframe not available
                            rgb_frame = sample.get('iframe', None)
                    
                    # Additional fallbacks
                    if rgb_frame is None:
                        rgb_frame = sample.get('frame', None)
                    if rgb_frame is None:
                        rgb_frame = sample.get('rgb_frame', None)
                    
                    if rgb_frame is None:
                        print(f"⚠️ No RGB frame available for frame {frame_idx}, skipping")
                        continue
                    
                    # Handle motion vector format - use Channel 0 with proper X/Y data
                    if isinstance(motion_vectors, torch.Tensor):
                        if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                            # Use Channel 0 which contains real X/Y motion data
                            motion_vectors = motion_vectors[0].permute(2, 0, 1)
                        elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                            # For 3D tensors, select proper channel structure
                            motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                        
                        if motion_vectors.shape[0] != 2:
                            motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else torch.zeros(2, 40, 40)
                    
                    motion_tensor = motion_vectors.unsqueeze(0).to(device)
                    
                    # Initialize or update bounding boxes
                    if frame_idx == 0:
                        # Initialize with real annotations for first frame
                        gop_prev_boxes = torch.zeros(1, max_objects, 4, device=device)
                        gop_object_ids = torch.zeros(1, max_objects, dtype=torch.long, device=device)
                        gop_valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
                        
                        if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                            boxes_tensor = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                            boxes_tensor = boxes_tensor[boxes_tensor.sum(dim=1) != 0]
                            
                            if len(boxes_tensor) > 0:
                                num_objects = min(len(boxes_tensor), max_objects)
                                gop_prev_boxes[0, :num_objects] = boxes_tensor[:num_objects, :4].to(device)
                                gop_object_ids[0, :num_objects] = torch.arange(1, num_objects + 1, device=device)
                                gop_valid_mask[0, :num_objects] = True
                    
                    # Model prediction
                    predicted_boxes, id_logits, attention_weights = model(
                        motion_tensor, gop_prev_boxes, gop_object_ids, gop_valid_mask
                    )
                    
                    # Create visualization with RGB background
                    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                    frame_type = "I-Frame" if frame_idx == 0 else "P-Frame"
                    fig.suptitle(f'{sequence_name} - {frame_type} RGB Tracking - Frame {frame_idx + 1}/48', fontsize=16)
                    
                    # Display RGB frame as background
                    if isinstance(rgb_frame, torch.Tensor):
                        rgb_np = rgb_frame.cpu().numpy()
                        
                        # Handle different tensor shapes: (B, H, W, C), (H, W, C), (C, H, W), (B, C, H, W)
                        if len(rgb_np.shape) == 4:  # Has batch dimension
                            rgb_np = rgb_np[0]  # Remove batch dimension
                        
                        if rgb_np.shape[0] == 3 and len(rgb_np.shape) == 3:  # CHW format
                            rgb_np = rgb_np.transpose(1, 2, 0)  # Convert to HWC
                        elif rgb_np.shape[-1] != 3 and len(rgb_np.shape) == 3:  # Might be HWC with wrong channel
                            if rgb_np.shape[0] == 3:  # Actually CHW
                                rgb_np = rgb_np.transpose(1, 2, 0)
                        
                        # Normalize to 0-255 range if needed
                        if rgb_np.max() <= 1.0:
                            rgb_np = (rgb_np * 255).astype(np.uint8)
                        else:
                            rgb_np = rgb_np.astype(np.uint8)
                    else:
                        rgb_np = np.zeros((640, 640, 3), dtype=np.uint8)
                    
                    ax.imshow(rgb_np)
                    
                    # Draw predicted boxes (solid lines)
                    valid_objects = gop_valid_mask[0].cpu().numpy()
                    pred_boxes_np = predicted_boxes[0].detach().cpu().numpy()
                    
                    h, w = rgb_np.shape[:2]
                    for obj_idx in range(len(valid_objects)):
                        if valid_objects[obj_idx]:
                            box = pred_boxes_np[obj_idx]
                            x_center, y_center, width, height = box
                            # Convert normalized coordinates to pixel coordinates
                            x_center_px = x_center * w
                            y_center_px = y_center * h
                            width_px = width * w
                            height_px = height * h
                            
                            x_min = x_center_px - width_px/2
                            y_min = y_center_px - height_px/2
                            
                            rect = plt.Rectangle((x_min, y_min), width_px, height_px,
                                               linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
                            ax.add_patch(rect)
                            ax.text(x_center_px, y_center_px - height_px/2 - 10, f'PRED-{obj_idx+1}', 
                                   ha='center', va='bottom', color='red', fontweight='bold', fontsize=10)
                    
                    # Draw ground truth boxes (dashed lines)
                    if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                        gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                        gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
                        
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_box.sum() != 0:
                                x_center, y_center, width, height = gt_box
                                # Convert normalized coordinates to pixel coordinates
                                x_center_px = x_center * w
                                y_center_px = y_center * h
                                width_px = width * w
                                height_px = height * h
                                
                                x_min = x_center_px - width_px/2
                                y_min = y_center_px - height_px/2
                                
                                rect = plt.Rectangle((x_min, y_min), width_px, height_px,
                                                   linewidth=3, edgecolor='green', facecolor='none', linestyle='--')
                                ax.add_patch(rect)
                                ax.text(x_center_px, y_center_px + height_px/2 + 10, f'GT-{gt_idx+1}', 
                                       ha='center', va='top', color='green', fontweight='bold', fontsize=10)
                    
                    ax.set_xlim(0, w)
                    ax.set_ylim(h, 0)
                    title_info = f'Frame {frame_idx+1}/48 - Active Objects: {gop_valid_mask[0].sum().item()} | RED=Predictions(solid) | GREEN=GroundTruth(dashed)'
                    ax.set_title(title_info)
                    
                    # Convert plot to video frame
                    plt.tight_layout()
                    fig.canvas.draw()
                    
                    buf = fig.canvas.buffer_rgba()
                    img = np.asarray(buf)
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_resized = cv2.resize(img_bgr, frame_size)
                    
                    out.write(img_resized)
                    plt.close(fig)
                    
                    # Update for next frame (temporal evolution)
                    gop_prev_boxes = predicted_boxes.detach().clone()
                    
                    frames_processed += 1
                    
                    if frame_idx == 0:
                        print(f"✅ Processed I-frame: {frame_idx+1}")
                    elif frame_idx % 20 == 0:
                        print(f"✅ Processed P-frame: {frame_idx+1}")
                    
                except Exception as e:
                    print(f"❌ Error processing RGB frame {frame_idx}: {e}")
                    continue
    
    finally:
        out.release()
    
    print(f"✅ RGB Video created: {output_path} ({frames_processed} frames)")
    return frames_processed > 0


def create_motion_vector_video(model, sequence_data, sequence_name, output_path, device, max_objects=20):
    """Create video with motion vector visualization showing predictions and GT annotations."""
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1280, 720)
    fps = 5.0
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    model.eval()
    frames_processed = 0
    
    # Initialize tracking state
    gop_prev_boxes = None
    gop_object_ids = None
    gop_valid_mask = None
    
    try:
        with torch.no_grad():
            for frame_idx in range(len(sequence_data)):
                try:
                    sample = sequence_data[frame_idx]
                    motion_vectors = sample.get('motion_vectors')
                    boxes_data = sample.get('boxes', torch.zeros(1, 4))
                    
                    # Handle motion vector format - use Channel 0 with proper X/Y data
                    if isinstance(motion_vectors, torch.Tensor):
                        if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                            # Use Channel 0 which contains real X/Y motion data
                            motion_vectors = motion_vectors[0].permute(2, 0, 1)
                        elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                            # For 3D tensors, select proper channel structure
                            motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                        
                        if motion_vectors.shape[0] != 2:
                            motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else torch.zeros(2, 40, 40)
                    
                    motion_tensor = motion_vectors.unsqueeze(0).to(device)
                    
                    # Initialize or update bounding boxes
                    if frame_idx == 0:
                        # Initialize with real annotations
                        gop_prev_boxes = torch.zeros(1, max_objects, 4, device=device)
                        gop_object_ids = torch.zeros(1, max_objects, dtype=torch.long, device=device)
                        gop_valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
                        
                        if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                            boxes_tensor = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                            boxes_tensor = boxes_tensor[boxes_tensor.sum(dim=1) != 0]
                            
                            if len(boxes_tensor) > 0:
                                num_objects = min(len(boxes_tensor), max_objects)
                                gop_prev_boxes[0, :num_objects] = boxes_tensor[:num_objects, :4].to(device)
                                gop_object_ids[0, :num_objects] = torch.arange(1, num_objects + 1, device=device)
                                gop_valid_mask[0, :num_objects] = True
                    
                    # Model prediction
                    predicted_boxes, id_logits, attention_weights = model(
                        motion_tensor, gop_prev_boxes, gop_object_ids, gop_valid_mask
                    )
                    
                    # Create motion vector visualization
                    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
                    fig.suptitle(f'{sequence_name} - Motion Vector Analysis - Frame {frame_idx + 1}/48', fontsize=16)
                    
                    # Plot 1: Motion magnitude
                    mv_x = motion_vectors[0].cpu().numpy()
                    mv_y = motion_vectors[1].cpu().numpy()
                    motion_magnitude = np.sqrt(mv_x**2 + mv_y**2)
                    
                    im1 = axes[0, 0].imshow(motion_magnitude, cmap='viridis')
                    axes[0, 0].set_title('Motion Magnitude')
                    plt.colorbar(im1, ax=axes[0, 0])
                    
                    # Plot 2: Motion vector field with arrows
                    y, x = np.mgrid[0:40:3, 0:40:3]  # Subsample for clarity
                    mv_x_sub = mv_x[::3, ::3]
                    mv_y_sub = mv_y[::3, ::3]
                    
                    axes[0, 1].quiver(x, y, mv_x_sub, mv_y_sub, 
                                    scale_units='xy', scale=0.5, color='blue', alpha=0.8)
                    axes[0, 1].set_title('Motion Vector Field')
                    axes[0, 1].set_aspect('equal')
                    axes[0, 1].set_xlim(0, 40)
                    axes[0, 1].set_ylim(40, 0)
                    
                    # Plot 3: Bounding boxes on motion field
                    motion_display = axes[1, 0].imshow(motion_magnitude, cmap='plasma', alpha=0.7)
                    
                    # Draw predicted boxes (solid red)
                    valid_objects = gop_valid_mask[0].cpu().numpy()
                    pred_boxes_np = predicted_boxes[0].detach().cpu().numpy()
                    
                    for obj_idx in range(len(valid_objects)):
                        if valid_objects[obj_idx]:
                            box = pred_boxes_np[obj_idx]
                            x_center, y_center, width, height = box
                            # Convert normalized coordinates to motion field coordinates (40x40)
                            x_center_mv = x_center * 40
                            y_center_mv = y_center * 40
                            width_mv = width * 40
                            height_mv = height * 40
                            
                            x_min = x_center_mv - width_mv/2
                            y_min = y_center_mv - height_mv/2
                            
                            rect = plt.Rectangle((x_min, y_min), width_mv, height_mv,
                                               linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
                            axes[1, 0].add_patch(rect)
                            axes[1, 0].text(x_center_mv, y_center_mv, f'P{obj_idx+1}', 
                                           ha='center', va='center', color='red', fontweight='bold', fontsize=8)
                    
                    # Draw ground truth boxes (dashed green)
                    if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                        gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                        gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
                        
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            if gt_box.sum() != 0:
                                x_center, y_center, width, height = gt_box
                                # Convert normalized coordinates to motion field coordinates (40x40)
                                x_center_mv = x_center * 40
                                y_center_mv = y_center * 40
                                width_mv = width * 40
                                height_mv = height * 40
                                
                                x_min = x_center_mv - width_mv/2
                                y_min = y_center_mv - height_mv/2
                                
                                rect = plt.Rectangle((x_min, y_min), width_mv, height_mv,
                                                   linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
                                axes[1, 0].add_patch(rect)
                                axes[1, 0].text(x_center_mv, y_center_mv + height_mv/2, f'GT{gt_idx+1}', 
                                               ha='center', va='bottom', color='green', fontweight='bold', fontsize=8)
                    
                    axes[1, 0].set_title(f'Motion Field + Boxes | RED=Pred(solid) | GREEN=GT(dashed)')
                    axes[1, 0].set_xlim(0, 40)
                    axes[1, 0].set_ylim(40, 0)
                    
                    # Plot 4: Statistics
                    axes[1, 1].axis('off')
                    stats_text = f"""Frame: {frame_idx + 1}/48
Active Objects: {gop_valid_mask[0].sum().item()}
Motion Range X: [{mv_x.min():.3f}, {mv_x.max():.3f}]
Motion Range Y: [{mv_y.min():.3f}, {mv_y.max():.3f}]
Motion Magnitude:
  Mean: {motion_magnitude.mean():.3f}
  Max:  {motion_magnitude.max():.3f}
  Std:  {motion_magnitude.std():.3f}
Model: ID-Aware Tracker"""
                    
                    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, 
                                   verticalalignment='center', fontfamily='monospace')
                    
                    # Convert plot to video frame
                    plt.tight_layout()
                    fig.canvas.draw()
                    
                    buf = fig.canvas.buffer_rgba()
                    img = np.asarray(buf)
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img_resized = cv2.resize(img_bgr, frame_size)
                    
                    out.write(img_resized)
                    plt.close(fig)
                    
                    # Update for next frame
                    gop_prev_boxes = predicted_boxes.detach().clone()
                    
                    frames_processed += 1
                    
                except Exception as e:
                    print(f"❌ Error processing motion frame {frame_idx}: {e}")
                    continue
    
    finally:
        out.release()
    
    print(f"✅ Motion Vector Video created: {output_path} ({frames_processed} frames)")
    return frames_processed > 0


def validate_model(args):
    """Main validation function."""
    print(f"🚀 Starting ID-Aware Tracker Validation")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Data: {args.data_path}")
    print(f"📁 Output: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load config and imports
    from configs.train_config import SequentialTrainConfig
    from models.id_aware_tracker import IDMultiObjectTracker
    from data.sequential_dataset import SequentialMOTSDataset
    
    config = checkpoint.get('config', SequentialTrainConfig())
    
    # Try to infer model parameters from the saved state
    model_state = checkpoint['model_state_dict']
    id_embedding_shape = model_state['id_encoder.embedding.weight'].shape[0]
    max_id_from_model = id_embedding_shape  # This is the max_id used during training
    max_objects_from_model = max_id_from_model // 10  # Assuming 10 IDs per object
    
    print(f"📊 Inferred max_id from model: {max_id_from_model}")
    print(f"📊 Inferred max_objects from model: {max_objects_from_model}")
    
    model = IDMultiObjectTracker(
        motion_shape=(2, 40, 40),
        hidden_dim=128,
        max_objects=max_objects_from_model,
        max_id=max_id_from_model
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Model info: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load validation dataset
    try:
        # Add dataset path for imports (same as training script)
        # From mots_exp/scripts/ go up two levels to reach dataset/
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
        
        # Import dataset factory
        from dataset.factory.dataset_factory import create_mots_dataset
        
        # Create validation dataset (use train data for validation)
        dataset = create_mots_dataset(
            dataset_type="mot17",
            resolution=640,
            mode="train",  # Use train data for validation
            load_iframe=True,   # Enable I-frames for RGB video creation
            load_pframe=True,   # Enable P-frames for RGB video creation
            load_motion_vectors=True,
            load_residuals=False,
            load_annotations=True,
            sequence_length=48  # GOP length
        )
        print(f"📊 Dataset loaded: {len(dataset)} samples")
        print(f"⚡ Limiting to 1 GOP for RGB testing (avoiding memory overload with I-frames + P-frames)")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Group samples by sequence - LIMIT TO 1 GOP FOR TESTING
    sequence_groups = defaultdict(list)
    max_sequences_to_load = 1  # Limit to 1 GOP when RGB frames are enabled
    processed_sequences = 0
    target_sequence = None  # Track which sequence we want to load completely
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if sample is not None and 'sequence_id' in sample:
            seq_id = sample['sequence_id']
            
            # If we haven't selected a target sequence yet, select this one
            if target_sequence is None:
                target_sequence = seq_id
                print(f"🔍 Selected target sequence: {seq_id}")
            
            # Only load frames from the target sequence
            if seq_id == target_sequence:
                if seq_id not in sequence_groups:
                    processed_sequences += 1
                sequence_groups[seq_id].append((idx, sample))
                
                # Debug: Show frame info
                if len(sequence_groups[seq_id]) <= 5 or len(sequence_groups[seq_id]) % 10 == 0:
                    frame_id = sample.get('frame_id', 'unknown')
                    has_iframe = sample.get('iframe') is not None
                    has_pframe = sample.get('pframe') is not None
                    print(f"  Frame {len(sequence_groups[seq_id])}: ID={frame_id}, I-frame={has_iframe}, P-frame={has_pframe}")
            
            # Stop after we've loaded one complete sequence (all its frames)
            elif processed_sequences >= max_sequences_to_load:
                break
    
    print(f"📊 Found {len(sequence_groups)} sequences for validation (limited to {max_sequences_to_load} for RGB testing)")
    
    # Debug: Show sequence sizes
    for seq_id, frames_list in sequence_groups.items():
        print(f"🔍 Sequence {seq_id}: {len(frames_list)} frames")
        if len(frames_list) > 0:
            first_sample = frames_list[0][1]
            last_sample = frames_list[-1][1]
            print(f"  First frame ID: {first_sample.get('frame_id', 'unknown')}")
            print(f"  Last frame ID: {last_sample.get('frame_id', 'unknown')}")
            print(f"  Sample keys: {list(first_sample.keys())}")
    
    # Initialize metrics
    metrics = ValidationMetrics()
    
    # Process each sequence
    validation_results = {
        'sequences': {},
        'overall_metrics': {},
        'config': {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'num_sequences': len(sequence_groups),
            'device': args.device
        }
    }
    
    processed_sequences = 0
    for seq_id, frames_list in tqdm(sequence_groups.items(), desc="Validating sequences"):
        if processed_sequences >= args.max_sequences:
            break
            
        print(f"\n🔍 Validating sequence: {seq_id}")
        
        # Sort frames by index
        frames_list.sort(key=lambda x: x[0])
        frames_data = [sample for _, sample in frames_list]
        
        # Create videos if requested
        if args.create_videos and frames_data:
            # Create RGB tracking video (solid predictions, dashed GT)
            rgb_video_path = os.path.join(args.output_dir, 'videos', f'{seq_id}_rgb_tracking.mp4')
            print(f"🎬 Creating RGB tracking video for {seq_id}")
            rgb_video_created = create_rgb_tracking_video(
                model, frames_data, seq_id, rgb_video_path, device, max_objects_from_model
            )
            
            # Create motion vector analysis video
            motion_video_path = os.path.join(args.output_dir, 'videos', f'{seq_id}_motion_analysis.mp4')
            print(f"🎬 Creating motion vector analysis video for {seq_id}")
            motion_video_created = create_motion_vector_video(
                model, frames_data, seq_id, motion_video_path, device, max_objects_from_model
            )
            
            validation_results['sequences'][seq_id] = {
                'frames_processed': len(frames_data),
                'rgb_video_created': rgb_video_created,
                'rgb_video_path': rgb_video_path if rgb_video_created else None,
                'motion_video_created': motion_video_created,
                'motion_video_path': motion_video_path if motion_video_created else None
            }
        
        processed_sequences += 1
    
    # Generate overall metrics
    overall_metrics = metrics.get_summary()
    validation_results['overall_metrics'] = overall_metrics
    
    # Save results
    results_path = os.path.join(args.output_dir, 'validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n✅ Validation completed!")
    print(f"📊 Results saved to: {results_path}")
    print(f"📈 Processed {processed_sequences} sequences")
    if args.create_videos:
        print(f"🎬 Videos created in: {args.output_dir}/videos/")

def main():
    args = parse_arguments()
    validate_model(args)

if __name__ == "__main__":
    main()