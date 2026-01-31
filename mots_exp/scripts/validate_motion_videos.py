#!/usr/bin/env python3
"""
Motion Vector Video Visualization Script
Creates visualization videos showing motion vectors and tracking performance
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from collections import defaultdict

# Add project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'dataset'))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Motion Vector Video Visualization")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-path", type=str, default="/home/aduche/Bureau/datasets/MOTS/videos/", help="MOTS dataset path")
    parser.add_argument("--resolution", type=int, default=960, choices=[640, 960], help="Video resolution (must match model training)")
    parser.add_argument("--max-videos", type=int, default=3, help="Maximum videos to process")
    parser.add_argument("--output-dir", type=str, default="outputs/motion_videos", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--video-type", type=str, choices=['motion', 'rgb', 'both'], default='both', 
                       help="Type of videos to create")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames per video")
    return parser.parse_args()

def load_model(model_path, device):
    """Load the trained model and detect type from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect model type
    if 'model_info' in checkpoint:
        # MV-Center model (standard or enhanced)
        model_info = checkpoint['model_info']
        
        # Check if enhanced model
        use_id_embedding = model_info.get('use_id_embedding', False)
        embedding_dim = model_info.get('embedding_dim', 128)
        
        if use_id_embedding:
            # Enhanced model with ID embeddings
            from mots_exp.models.mv_center.mv_center_memory_enhanced import MVCenterMemoryTrackerEnhanced
            
            model = MVCenterMemoryTrackerEnhanced(
                version=model_info.get('version', 'memory'),
                config=model_info.get('config', 'standard'),
                use_magnitude=model_info.get('use_magnitude', True),
                use_roi_align=model_info.get('use_roi_align', True),
                image_size=model_info.get('image_size', 960),
                use_id_embedding=use_id_embedding,
                embedding_dim=embedding_dim
            )
            print(f"✨ Loaded ENHANCED MV-Center model with ID embeddings (dim={embedding_dim})")
        else:
            # Standard MV-Center model
            from mots_exp.models.mv_center.mv_center_memory import MVCenterMemoryTracker
            
            model = MVCenterMemoryTracker(
                version=model_info.get('version', 'memory'),
                config=model_info.get('config', 'standard'),
                use_magnitude=model_info.get('use_magnitude', True),
                use_roi_align=model_info.get('use_roi_align', True),
                image_size=model_info.get('image_size', 960)
            )
            print(f"📊 Loaded standard MV-Center model: {model_info.get('version', 'memory')}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        max_objects = 100
        use_magnitude = model_info.get('use_magnitude', True)
        model_type = 'mv_center'
    elif 'config' in checkpoint:
        # ID-Tracker model
        from models.id_aware_tracker import IDMultiObjectTracker
        
        config = checkpoint['config']
        motion_shape = config.get('motion_shape', (2, 40, 40))
        max_objects = config.get('max_objects', 10)
        use_magnitude = motion_shape[0] == 3
        model = IDMultiObjectTracker(
            motion_shape=motion_shape,
            max_objects=max_objects,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model_type = 'id_tracker'
        print(f"📊 Loaded ID-Tracker model: max_objects={max_objects}, channels={motion_shape[0]}")
    else:
        # Legacy format - try to infer from state dict
        from models.id_aware_tracker import IDMultiObjectTracker
        model_state = checkpoint['model_state_dict']
        id_embedding_shape = model_state['id_encoder.embedding.weight'].shape[0]
        max_id = id_embedding_shape
        max_objects = max_id // 10  # Assuming 10 IDs per object
        
        print(f"📊 Inferred max_id: {max_id}, max_objects: {max_objects}")
        
        model = IDMultiObjectTracker(
            motion_shape=(2, 40, 40),
            hidden_dim=128,
            max_objects=max_objects,
            max_id=max_id
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        use_magnitude = False
        model_type = 'id_tracker'
    
    model.to(device)
    model.eval()
    
    print(f"📊 Model loaded: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"🎯 Model type: {model_type}")
    return model, max_objects, use_magnitude, model_type


def convert_mv_center_predictions_to_boxes(predictions, max_objects=100, score_threshold=0.01):
    """Convert MV-Center predictions (heatmaps) to bounding boxes"""
    all_boxes = []
    all_scores = []
    
    for level_name, level_preds in predictions.items():
        # Extract center heatmap and box parameters
        center_logits = level_preds['center'][0, 0]  # [H, W] - logits
        center_map = torch.sigmoid(center_logits)  # Apply sigmoid to get probabilities
        box_map = level_preds['box'][0]  # [4, H, W]
        
        h, w = center_map.shape
        stride = 640 / h  # P3: 64px, P4: 128px
        
        # Find peaks in center heatmap
        peak_mask = center_map > score_threshold
        peak_coords = torch.nonzero(peak_mask, as_tuple=False)
        
        if len(peak_coords) == 0:
            continue
        
        # Extract boxes at peak locations
        for y, x in peak_coords:
            score = center_map[y, x].item()
            
            # Get box parameters
            dx = box_map[0, y, x].item()
            dy = box_map[1, y, x].item()
            log_w = box_map[2, y, x].item()
            log_h = box_map[3, y, x].item()
            
            # Convert to absolute coordinates (normalized to [0, 1])
            cx = (x.item() + dx) * stride / 640
            cy = (y.item() + dy) * stride / 640
            box_w = np.exp(log_w) * stride / 640
            box_h = np.exp(log_h) * stride / 640
            
            # Convert to center format [cx, cy, w, h]
            all_boxes.append([cx, cy, box_w, box_h])
            all_scores.append(score)
    
    # Sort by score and take top-k
    if len(all_boxes) > 0:
        sorted_indices = np.argsort(all_scores)[::-1][:max_objects]
        boxes = torch.tensor([all_boxes[i] for i in sorted_indices])
        scores = torch.tensor([all_scores[i] for i in sorted_indices])
        
        # Pad to max_objects
        boxes_tensor = torch.zeros(max_objects, 4)
        valid_mask = torch.zeros(max_objects, dtype=torch.bool)
        boxes_tensor[:len(boxes)] = boxes
        valid_mask[:len(boxes)] = True
        
        return boxes_tensor, valid_mask, scores
    else:
        return torch.zeros(max_objects, 4), torch.zeros(max_objects, dtype=torch.bool), torch.tensor([])


def run_model_inference(model, model_type, motion_tensor, prev_boxes, object_ids, valid_mask, max_objects):
    """Unified inference for both model types"""
    if model_type == 'mv_center':
        # MV-Center: Detection-based, stateless
        predictions = model(motion_tensor)
        predicted_boxes, valid_mask_out, scores = convert_mv_center_predictions_to_boxes(predictions, max_objects)
        predicted_boxes = predicted_boxes.unsqueeze(0).to(motion_tensor.device)
        valid_mask_out = valid_mask_out.unsqueeze(0).to(motion_tensor.device)
        
        # Generate sequential IDs for detected boxes
        num_valid = valid_mask_out[0].sum().item()
        ids_out = torch.zeros(1, max_objects, dtype=torch.long, device=motion_tensor.device)
        ids_out[0, :num_valid] = torch.arange(1, num_valid + 1, device=motion_tensor.device)
        
        return predicted_boxes, ids_out, valid_mask_out
    else:
        # ID-Tracker: Tracking-based, stateful
        predicted_boxes = model(motion_tensor, prev_boxes, object_ids, valid_mask)
        return predicted_boxes, object_ids, valid_mask

def scan_for_video_sequences(data_path, max_videos, resolution=960):
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
        
        # Create dataset with motion vectors only
        dataset = create_mots_dataset(
            dataset_type="mot17",
            resolution=resolution,
            mode="train",  # Use train data for validation
            load_iframe=False,       # No RGB frames for motion videos
            load_pframe=False,       # No RGB frames for motion videos  
            load_motion_vectors=True,  # Only motion vectors
            load_residuals=False,
            load_annotations=True,   # Need annotations for visualization
            sequence_length=48  # GOP length
        )
        
        if dataset is None or len(dataset) == 0:
            print(f"❌ Dataset creation failed or empty")
            return None
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_memory_tracker_gop_inference(model, gop_frames, device):
    """
    Run inference on entire GOP using MVCenterMemoryTracker.
    Returns frame-by-frame predictions for visualization.
    """
    try:
        # Extract motion vectors and boxes for entire GOP
        motion_sequence = []
        iframe_boxes_list = []
        gt_boxes_list = []
        
        for frame_data in gop_frames:
            motion_vectors = frame_data.get('motion_vectors')
            boxes = frame_data.get('boxes', torch.zeros(0, 4))
            
            if motion_vectors is None:
                continue
                
            # Handle motion vector format [H, W, 2] -> [2, H, W]
            if isinstance(motion_vectors, torch.Tensor):
                if len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                    motion_vectors = motion_vectors.permute(2, 0, 1)
                elif len(motion_vectors.shape) == 4:
                    motion_vectors = motion_vectors[0].permute(2, 0, 1)
            
            motion_sequence.append(motion_vectors)
            gt_boxes_list.append(boxes)
            
            # First frame boxes as I-frame detections
            if len(iframe_boxes_list) == 0 and isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                iframe_boxes_list = boxes.view(-1, 4) if len(boxes.shape) > 1 else boxes.unsqueeze(0)
                iframe_boxes_list = iframe_boxes_list[iframe_boxes_list.sum(dim=1) != 0]
        
        if len(motion_sequence) == 0:
            return [], []
        
        # Stack motion vectors [T, 2, H, W]
        motion_batch = torch.stack(motion_sequence).to(device)
        iframe_boxes = iframe_boxes_list.to(device) if len(iframe_boxes_list) > 0 else torch.zeros(0, 4).to(device)
        
        # Run GOP inference
        with torch.no_grad():
            outputs = model.forward_gop(motion_batch, iframe_boxes)
            
            # Handle both standard and enhanced models
            if isinstance(outputs, tuple) and len(outputs) == 3:
                predictions_list, confidences_list, embeddings_list = outputs
            else:
                predictions_list, confidences_list = outputs
        
        return predictions_list, gt_boxes_list
        
    except Exception as e:
        print(f"❌ Error in GOP inference: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def create_gop_motion_video(model, gop_name, frames_list, output_path, device, max_objects, use_magnitude, model_type):
    """Create video with motion vector visualization from GOP frames."""
    
    if not frames_list:
        print(f"   ⚠️ No frames in GOP")
        return False
    
    # Extract frame data
    gop_frames = [sample for _, sample in frames_list]
    
    # Run GOP-based inference for MV-Center models
    if model_type == 'mv_center':
        predictions_list, gt_boxes_list = run_memory_tracker_gop_inference(model, gop_frames, device)
        
        if len(predictions_list) == 0:
            print(f"   ⚠️ No predictions from model")
            return False
    else:
        print(f"   ⚠️ Model type {model_type} not supported for video visualization yet")
        return False
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1920, 1080)  # Larger for better visibility
    fps = 5.0
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    frames_processed = 0
    
    try:
        with torch.no_grad():
            for frame_idx in range(len(gop_frames)):
                try:
                    sample = gop_frames[frame_idx]
                    motion_vectors = sample.get('motion_vectors')
                    
                    if motion_vectors is None:
                        continue
                    
                    # Handle motion vector format [H, W, 2] -> [2, H, W]
                    if isinstance(motion_vectors, torch.Tensor):
                        if len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                            motion_vectors = motion_vectors.permute(2, 0, 1)
                        elif len(motion_vectors.shape) == 4:
                            motion_vectors = motion_vectors[0].permute(2, 0, 1)
                    
                    mv_x, mv_y = motion_vectors[0].cpu(), motion_vectors[1].cpu()
                    
                    # Get predictions and GT for this frame
                    if frame_idx < len(predictions_list):
                        pred_boxes = predictions_list[frame_idx].cpu() if isinstance(predictions_list[frame_idx], torch.Tensor) else torch.zeros(0, 4)
                    else:
                        pred_boxes = torch.zeros(0, 4)
                    
                    if frame_idx < len(gt_boxes_list):
                        gt_boxes = gt_boxes_list[frame_idx]
                        if isinstance(gt_boxes, torch.Tensor) and gt_boxes.numel() > 0:
                            gt_boxes = gt_boxes.view(-1, 4) if len(gt_boxes.shape) > 1 else gt_boxes.unsqueeze(0)
                            gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0].cpu()
                        else:
                            gt_boxes = torch.zeros(0, 4)
                    else:
                        gt_boxes = torch.zeros(0, 4)
                    
                    # Create visualization
                    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
                    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
                    
                    fig.suptitle(f'{gop_name} - Frame {frame_idx}\nModel Predictions (Green) vs Ground Truth (Red)', 
                                fontsize=20, fontweight='bold')
                    
                    # Plot 1: Motion Magnitude
                    ax1 = fig.add_subplot(gs[0, 0])
                    motion_magnitude = torch.sqrt(mv_x**2 + mv_y**2).numpy()
                    im1 = ax1.imshow(motion_magnitude, cmap='viridis')
                    ax1.set_title(f'Motion Magnitude\nMax: {motion_magnitude.max():.1f}', fontsize=14)
                    plt.colorbar(im1, ax=ax1, fraction=0.046)
                    ax1.axis('off')
                    
                    # Plot 2: Motion Vector Field
                    ax2 = fig.add_subplot(gs[0, 1])
                    h, w = mv_x.shape
                    step = max(1, h // 15)
                    y, x = np.mgrid[0:h:step, 0:w:step]
                    mv_x_np = mv_x.numpy()[::step, ::step]
                    mv_y_np = mv_y.numpy()[::step, ::step]
                    
                    ax2.quiver(x, y, mv_x_np, mv_y_np, 
                              scale_units='xy', scale=1.0, color='blue', alpha=0.7, width=0.003)
                    ax2.set_title('Motion Vector Field', fontsize=14)
                    ax2.set_aspect('equal')
                    ax2.set_xlim(0, w)
                    ax2.set_ylim(h, 0)
                    ax2.axis('off')
                    
                    # Plot 3: Predictions on Motion Heatmap
                    ax3 = fig.add_subplot(gs[0, 2])
                    ax3.imshow(motion_magnitude, cmap='gray', alpha=0.5)
                    
                    # Draw GT boxes in RED
                    for box in gt_boxes:
                        cx, cy, w_box, h_box = box
                        x1, y1 = (cx - w_box/2) * w, (cy - h_box/2) * h
                        width, height = w_box * w, h_box * h
                        rect = plt.Rectangle((x1, y1), width, height, fill=False, color='red', linewidth=3, label='GT')
                        ax3.add_patch(rect)
                    
                    # Draw predictions in GREEN
                    for box in pred_boxes:
                        if box.sum() > 0:
                            cx, cy, w_box, h_box = box
                            x1, y1 = (cx - w_box/2) * w, (cy - h_box/2) * h
                            width, height = w_box * w, h_box * h
                            rect = plt.Rectangle((x1, y1), width, height, fill=False, color='lime', linewidth=2, linestyle='--', label='Pred')
                            ax3.add_patch(rect)
                    
                    ax3.set_title(f'Boxes: GT={len(gt_boxes)}, Pred={len([b for b in pred_boxes if b.sum() > 0])}', fontsize=14)
                    ax3.axis('off')
                    
                    # Plot 4: Box Size Visualization
                    ax4 = fig.add_subplot(gs[1, 0])
                    if len(pred_boxes) > 0 and pred_boxes[0].sum() > 0:
                        box_widths = [b[2].item() for b in pred_boxes if b.sum() > 0]
                        box_heights = [b[3].item() for b in pred_boxes if b.sum() > 0]
                        ax4.scatter(box_widths, box_heights, c='lime', s=100, alpha=0.6, label='Predictions')
                    if len(gt_boxes) > 0:
                        gt_widths = [b[2].item() for b in gt_boxes]
                        gt_heights = [b[3].item() for b in gt_boxes]
                        ax4.scatter(gt_widths, gt_heights, c='red', s=100, alpha=0.6, marker='x', label='Ground Truth')
                    ax4.set_xlabel('Width', fontsize=12)
                    ax4.set_ylabel('Height', fontsize=12)
                    ax4.set_title('Box Sizes (Normalized)', fontsize=14)
                    ax4.legend()
                    ax4.grid(True, alpha=0.3)
                    
                    # Plot 5: Center Positions
                    ax5 = fig.add_subplot(gs[1, 1])
                    ax5.set_xlim(0, 1)
                    ax5.set_ylim(0, 1)
                    if len(gt_boxes) > 0:
                        gt_centers = [(b[0].item(), b[1].item()) for b in gt_boxes]
                        ax5.scatter([c[0] for c in gt_centers], [c[1] for c in gt_centers], 
                                   c='red', s=150, alpha=0.6, marker='x', label='GT Centers')
                    if len(pred_boxes) > 0 and pred_boxes[0].sum() > 0:
                        pred_centers = [(b[0].item(), b[1].item()) for b in pred_boxes if b.sum() > 0]
                        ax5.scatter([c[0] for c in pred_centers], [c[1] for c in pred_centers], 
                                   c='lime', s=100, alpha=0.6, label='Pred Centers')
                    ax5.set_xlabel('X (normalized)', fontsize=12)
                    ax5.set_ylabel('Y (normalized)', fontsize=12)
                    ax5.set_title('Object Centers', fontsize=14)
                    ax5.legend()
                    ax5.grid(True, alpha=0.3)
                    ax5.invert_yaxis()
                    
                    # Plot 6: Statistics
                    ax6 = fig.add_subplot(gs[1, 2])
                    ax6.axis('off')
                    
                    stats_text = f"""
Frame Statistics:

Motion Vectors:
  • Mean magnitude: {motion_magnitude.mean():.2f}
  • Max magnitude: {motion_magnitude.max():.2f}
  • Active pixels: {(motion_magnitude > 1).sum()}

Ground Truth:
  • Objects: {len(gt_boxes)}
  
Predictions:
  • Objects: {len([b for b in pred_boxes if b.sum() > 0])}
  
Frame Type:
  • {"I-frame (GT)" if frame_idx == 0 else f"P-frame {frame_idx}"}
                    """
                    
                    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                            fontsize=12, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            family='monospace')
                    
                    # Convert to image
                    fig.canvas.draw()
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    
                    # Convert RGB to BGR for OpenCV
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Write frame
                    out.write(img_bgr)
                    frames_processed += 1
                    
                    plt.close(fig)
                    
                except Exception as e:
                    print(f"   ❌ Error processing frame {frame_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    finally:
        out.release()
    
    print(f"   ✅ Video created: {frames_processed} frames -> {output_path}")
    return frames_processed > 0


def create_motion_vector_video(model, sequence_name, data_path, output_path, device, max_objects, max_frames, use_magnitude, model_type):
    """Create video with motion vector visualization."""
    sequence_path = os.path.join(data_path, sequence_name)
    motion_dir = os.path.join(sequence_path, 'motion_vectors')
    annotations_dir = os.path.join(sequence_path, 'annotations')
    
    if not os.path.exists(motion_dir):
        print(f"   ⚠️ No motion vectors directory found")
        return False
    
    # Find motion vector files
    motion_files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])[:max_frames]
    
    if not motion_files:
        print(f"   ⚠️ No motion vector files found")
        return False
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1280, 720)
    fps = 5.0
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    frames_processed = 0
    
    # Initialize tracking state
    prev_boxes = None
    object_ids = None
    valid_mask = None
    
    try:
        with torch.no_grad():
            for frame_idx, motion_file in enumerate(motion_files):
                try:
                    # Load motion vectors
                    motion_path = os.path.join(motion_dir, motion_file)
                    motion_vectors = np.load(motion_path)
                    
                    # Handle motion vector format - use Channel 0 with proper X/Y data
                    if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                        motion_vectors = motion_vectors[0].transpose(2, 0, 1)
                    elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                        motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                    
                    if motion_vectors.shape[0] != 2:
                        motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else np.zeros((2, 40, 40))
                    
                    motion_tensor = torch.from_numpy(motion_vectors).float().unsqueeze(0).to(device)
                    
                    # Load annotations if available
                    annotation_file = motion_file.replace('.npy', '.json')
                    annotation_path = os.path.join(annotations_dir, annotation_file)
                    
                    gt_boxes = []
                    if os.path.exists(annotation_path):
                        with open(annotation_path, 'r') as f:
                            annotations = json.load(f)
                        
                        for obj in annotations.get('objects', []):
                            bbox = obj.get('bbox', [0, 0, 0, 0])
                            if len(bbox) == 4 and sum(bbox) > 0:
                                gt_boxes.append(bbox)
                    
                    gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros(0, 4)
                    
                    # Initialize or update tracking state
                    if frame_idx == 0 or prev_boxes is None:
                        prev_boxes = torch.zeros(1, max_objects, 4, device=device)
                        object_ids = torch.zeros(1, max_objects, dtype=torch.long, device=device)
                        valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
                        
                        if len(gt_boxes_tensor) > 0:
                            num_objects = min(len(gt_boxes_tensor), max_objects)
                            prev_boxes[0, :num_objects] = gt_boxes_tensor[:num_objects].to(device)
                            object_ids[0, :num_objects] = torch.arange(1, num_objects + 1, device=device)
                            valid_mask[0, :num_objects] = True
                    
                    # Model prediction
                    if frame_idx == 0:
                        # I-frame: use GT
                        predicted_boxes = prev_boxes.clone()
                        ids_out = object_ids
                        valid_out = valid_mask
                    else:
                        # P-frame: use model
                        predicted_boxes, ids_out, valid_out = run_model_inference(
                            model, model_type, motion_tensor, prev_boxes, object_ids, valid_mask, max_objects
                        )
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
                    fig.suptitle(f'{sequence_name} - Motion Vector Analysis - Frame {frame_idx + 1}', fontsize=16)
                    
                    # Plot 1: Motion magnitude
                    mv_x = motion_vectors[0]
                    mv_y = motion_vectors[1]
                    motion_magnitude = np.sqrt(mv_x**2 + mv_y**2)
                    
                    im1 = axes[0, 0].imshow(motion_magnitude, cmap='viridis')
                    axes[0, 0].set_title('Motion Magnitude')
                    plt.colorbar(im1, ax=axes[0, 0])
                    
                    # Plot 2: Motion vector field
                    y, x = np.mgrid[0:40:3, 0:40:3]
                    mv_x_sub = mv_x[::3, ::3]
                    mv_y_sub = mv_y[::3, ::3]
                    
                    axes[0, 1].quiver(x, y, mv_x_sub, mv_y_sub, 
                                    scale_units='xy', scale=0.5, color='blue', alpha=0.8)
                    axes[0, 1].set_title('Motion Vector Field')
                    axes[0, 1].set_aspect('equal')
                    axes[0, 1].set_xlim(0, 40)
                    axes[0, 1].set_ylim(40, 0)
                    
                    # Plot 3: Tracking overlay
                    axes[1, 0].imshow(motion_magnitude, cmap='plasma', alpha=0.7)
                    
                    # Draw predicted boxes (solid red)
                    valid_objects = valid_out[0].cpu().numpy()
                    pred_boxes_np = predicted_boxes[0].detach().cpu().numpy()
                    
                    for obj_idx in range(len(valid_objects)):
                        if valid_objects[obj_idx]:
                            box = pred_boxes_np[obj_idx]
                            x_center, y_center, width, height = box
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
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        x_center, y_center, width, height = gt_box
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
                    
                    axes[1, 0].set_title('Tracking Overlay')
                    axes[1, 0].set_xlim(0, 40)
                    axes[1, 0].set_ylim(40, 0)
                    
                    # Plot 4: Statistics
                    axes[1, 1].axis('off')
                    frame_type = "I-FRAME (GT)" if frame_idx == 0 else "P-FRAME (Model)"
                    stats_text = f"""Frame: {frame_idx + 1} ({frame_type})
Active Objects: {valid_out[0].sum().item()}
Ground Truth: {len(gt_boxes)}
Model Type: {model_type.upper()}

Motion Statistics:
  X Range: [{mv_x.min():.3f}, {mv_x.max():.3f}]
  Y Range: [{mv_y.min():.3f}, {mv_y.max():.3f}]
  Magnitude:
    Mean: {motion_magnitude.mean():.3f}
    Max:  {motion_magnitude.max():.3f}
    Std:  {motion_magnitude.std():.3f}

Legend:
  RED (solid): Predictions
  GREEN (dashed): Ground Truth"""
                    
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
                    
                    # Update tracking state
                    prev_boxes = predicted_boxes.detach().clone()
                    
                    frames_processed += 1
                    
                except Exception as e:
                    print(f"   ❌ Error processing frame {frame_idx}: {e}")
                    continue
    
    finally:
        out.release()
    
    print(f"   ✅ Motion video created: {frames_processed} frames")
    return frames_processed > 0

def create_rgb_tracking_video(model, sequence_name, data_path, output_path, device, max_objects, max_frames, use_magnitude, model_type):
    """Create video with RGB visualization (if RGB data is available)."""
    sequence_path = os.path.join(data_path, sequence_name)
    rgb_dir = os.path.join(sequence_path, 'rgb_frames')  # Assuming RGB frames are stored here
    motion_dir = os.path.join(sequence_path, 'motion_vectors')
    annotations_dir = os.path.join(sequence_path, 'annotations')
    
    if not os.path.exists(rgb_dir) or not os.path.exists(motion_dir):
        print(f"   ⚠️ RGB frames or motion vectors not found, skipping RGB video")
        return False
    
    # Find files
    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:max_frames]
    motion_files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])[:max_frames]
    
    if not rgb_files or not motion_files:
        print(f"   ⚠️ No RGB or motion files found")
        return False
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (1280, 720)
    fps = 5.0
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    frames_processed = 0
    
    # Initialize tracking state
    prev_boxes = None
    object_ids = None
    valid_mask = None
    
    try:
        with torch.no_grad():
            for frame_idx in range(min(len(rgb_files), len(motion_files))):
                try:
                    # Load RGB frame
                    rgb_path = os.path.join(rgb_dir, rgb_files[frame_idx])
                    rgb_frame = cv2.imread(rgb_path)
                    if rgb_frame is None:
                        continue
                    
                    # Load motion vectors
                    motion_path = os.path.join(motion_dir, motion_files[frame_idx])
                    motion_vectors = np.load(motion_path)
                    
                    # Handle motion vector format
                    if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                        motion_vectors = motion_vectors[0].transpose(2, 0, 1)
                    elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                        motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                    
                    if motion_vectors.shape[0] != 2:
                        motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else np.zeros((2, 40, 40))
                    
                    motion_tensor = torch.from_numpy(motion_vectors).float().unsqueeze(0).to(device)
                    
                    # Load annotations
                    annotation_file = motion_files[frame_idx].replace('.npy', '.json')
                    annotation_path = os.path.join(annotations_dir, annotation_file)
                    
                    gt_boxes = []
                    if os.path.exists(annotation_path):
                        with open(annotation_path, 'r') as f:
                            annotations = json.load(f)
                        
                        for obj in annotations.get('objects', []):
                            bbox = obj.get('bbox', [0, 0, 0, 0])
                            if len(bbox) == 4 and sum(bbox) > 0:
                                gt_boxes.append(bbox)
                    
                    gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros(0, 4)
                    
                    # Initialize tracking state
                    if frame_idx == 0 or prev_boxes is None:
                        prev_boxes = torch.zeros(1, max_objects, 4, device=device)
                        object_ids = torch.zeros(1, max_objects, dtype=torch.long, device=device)
                        valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
                        
                        if len(gt_boxes_tensor) > 0:
                            num_objects = min(len(gt_boxes_tensor), max_objects)
                            prev_boxes[0, :num_objects] = gt_boxes_tensor[:num_objects].to(device)
                            object_ids[0, :num_objects] = torch.arange(1, num_objects + 1, device=device)
                            valid_mask[0, :num_objects] = True
                    
                    # Model prediction
                    if frame_idx == 0:
                        # I-frame: use GT
                        predicted_boxes = prev_boxes.clone()
                        ids_out = object_ids
                        valid_out = valid_mask
                    else:
                        # P-frame: use model
                        predicted_boxes, ids_out, valid_out = run_model_inference(
                            model, model_type, motion_tensor, prev_boxes, object_ids, valid_mask, max_objects
                        )
                    
                    # Create visualization
                    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
                    fig.suptitle(f'{sequence_name} - RGB Tracking - Frame {frame_idx + 1}', fontsize=16)
                    
                    # Display RGB frame
                    rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
                    ax.imshow(rgb_frame_rgb)
                    
                    h, w = rgb_frame.shape[:2]
                    
                    # Draw predicted boxes (solid red)
                    valid_objects = valid_out[0].cpu().numpy()
                    pred_boxes_np = predicted_boxes[0].detach().cpu().numpy()
                    
                    for obj_idx in range(len(valid_objects)):
                        if valid_objects[obj_idx]:
                            box = pred_boxes_np[obj_idx]
                            x_center, y_center, width, height = box
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
                    
                    # Draw ground truth boxes (dashed green)
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        x_center, y_center, width, height = gt_box
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
                    ax.set_title(f'Frame {frame_idx+1} - RED=Predictions | GREEN=Ground Truth')
                    
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
                    
                    # Update tracking state
                    prev_boxes = predicted_boxes.detach().clone()
                    
                    frames_processed += 1
                    
                except Exception as e:
                    print(f"   ❌ Error processing RGB frame {frame_idx}: {e}")
                    continue
    
    finally:
        out.release()
    
    print(f"   ✅ RGB video created: {frames_processed} frames")
    return frames_processed > 0

def main():
    args = parse_arguments()
    
    print(f"🚀 Starting Motion Vector Video Visualization")
    print(f"📁 Model: {args.model_path}")
    print(f"📁 Data: {args.data_path}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🎬 Video type: {args.video_type}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device(args.device)
    try:
        model, max_objects, use_magnitude, model_type = load_model(args.model_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Scan for video sequences
    print(f"🔍 Scanning for video sequences in: {args.data_path}")
    dataset = scan_for_video_sequences(args.data_path, args.max_videos, args.resolution)
    
    if dataset is None:
        print(f"❌ No video sequences found in {args.data_path}")
        return
    
    print(f"📊 Found {len(dataset)} samples")
    
    # Group samples by video and GOP (same as validate_motion_map.py)
    video_gop_groups = defaultdict(lambda: defaultdict(list))
    
    print(f"🔄 Grouping {len(dataset)} samples by video and GOP...")
    
    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
            if sample is not None and 'sequence_id' in sample:
                seq_id = sample['sequence_id']
                
                # Extract video name and GOP index from sequence_id
                # Format: "MOT17-05-SDP_640x640_gop50_500frames_gop0"
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
                            continue
        except Exception as e:
            print(f"⚠️ Error loading sample {idx}: {e}")
            continue
    
    print(f"📊 Found {len(video_gop_groups)} videos with GOP sequences:")
    for video_name, gops in video_gop_groups.items():
        print(f"   {video_name}: {len(gops)} GOPs (indices: {sorted(gops.keys())[:5]}...)")
    
    # Limit to max_videos if specified
    if args.max_videos > 0:
        video_names = list(video_gop_groups.keys())[:args.max_videos]
        video_gop_groups = {name: video_gop_groups[name] for name in video_names}
        print(f"📊 Limited to {len(video_gop_groups)} videos for processing")
    
    # Process each video sequence - create one video per GOP
    results = {}
    
    print(f"\n🎬 Creating videos for {len(video_gop_groups)} videos...")
    
    for video_name, gop_dict in video_gop_groups.items():
        print(f"🎬 Processing: {video_name} with {len(gop_dict)} GOPs")
        
        # Process first GOP only (or first few GOPs if desired)
        gop_indices = sorted(gop_dict.keys())[:1]  # Just first GOP for now
        
        for gop_idx in gop_indices:
            frames_list = gop_dict[gop_idx]
            frames_list.sort(key=lambda x: x[0])  # Sort by sample index
            
            print(f"   🔄 GOP {gop_idx}: {len(frames_list)} frames")
            
            gop_name = f"{video_name}_gop{gop_idx}"
            sequence_results = {}
            
            # Create motion vector video
            if args.video_type in ['motion', 'both']:
                motion_output = os.path.join(args.output_dir, f'{gop_name}_motion_analysis.mp4')
                motion_success = create_gop_motion_video(
                    model, gop_name, frames_list, motion_output, 
                    device, max_objects, use_magnitude, model_type
                )
                sequence_results['motion_video'] = {
                    'created': motion_success,
                    'path': motion_output if motion_success else None
                }
            
            results[gop_name] = sequence_results
    
    # Save results summary
    summary = {
        'config': {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'video_type': args.video_type,
            'max_videos': args.max_videos,
            'max_frames': args.max_frames
        },
        'results': results
    }
    
    results_path = os.path.join(args.output_dir, 'video_creation_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n✅ Video creation completed!")
    print(f"📊 GOPs processed: {len(results)}")
    
    motion_count = sum(1 for r in results.values() if r.get('motion_video', {}).get('created', False))
    rgb_count = sum(1 for r in results.values() if r.get('rgb_video', {}).get('created', False))
    
    if args.video_type in ['motion', 'both']:
        print(f"🎬 Motion videos created: {motion_count}/{len(results)}")
    if args.video_type in ['rgb', 'both']:
        print(f"🎬 RGB videos created: {rgb_count}/{len(results)}")
    
    print(f"📊 Results saved to: {results_path}")
    print(f"🎬 Videos saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
