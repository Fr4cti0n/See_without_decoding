#!/usr/bin/env python3
"""
Memory Tracker Visualization Script
Creates videos showing predictions vs ground truth for memory-based LSTM tracker
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# Add project paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / 'dataset'))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Memory Tracker Visualization")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--max-gops", type=int, default=None, help="Maximum GOPs to visualize (None = all)")
    parser.add_argument("--data-path", type=str, default="/home/aduche/Bureau/datasets/MOTS/videos/", help="Path to video data")
    parser.add_argument("--output-dir", type=str, default="outputs/memory_tracker_videos", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--arrow-scale", type=float, default=2.0, help="Scale factor for motion vector arrows")
    return parser.parse_args()


def load_model(model_path, device):
    """Load memory tracker model from checkpoint."""
    print(f"📦 Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model info
    if 'model_info' in checkpoint:
        model_info = checkpoint['model_info']
    else:
        # Default parameters
        model_info = {
            'feature_dim': 128,
            'hidden_dim': 256,
            'max_objects': 100,
            'grid_size': 40,
            'image_size': 960,
            'use_roi_align': True,
            'roi_size': (7, 7)
        }
    
    # Extract parameters
    feature_dim = model_info.get('feature_dim', 128)
    hidden_dim = model_info.get('hidden_dim', 256)
    max_objects = model_info.get('max_objects', 100)
    grid_size = model_info.get('grid_size', 40)
    image_size = model_info.get('image_size', 960)
    use_roi_align = model_info.get('use_roi_align', True)
    roi_size = model_info.get('roi_size', (7, 7))
    
    # Check for model architecture type
    use_id_embedding = model_info.get('use_id_embedding', False)
    use_backbone = model_info.get('use_backbone', False)
    embedding_dim = model_info.get('embedding_dim', 128)
    
    if use_backbone:
        # Load full bbox model with CNN backbone
        from mots_exp.models.mv_center.mv_center_memory_fullbox import MVCenterMemoryTrackerFullBox
        
        model = MVCenterMemoryTrackerFullBox(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_objects=max_objects,
            grid_size=grid_size,
            image_size=image_size,
            use_backbone=use_backbone
        )
        print(f"   ✨ Full bbox model with CNN backbone (feature_dim={feature_dim}, hidden_dim={hidden_dim})")
    elif use_id_embedding:
        # Load enhanced model
        from mots_exp.models.mv_center.mv_center_memory_enhanced import MVCenterMemoryTrackerEnhanced
        
        model = MVCenterMemoryTrackerEnhanced(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_objects=max_objects,
            grid_size=grid_size,
            image_size=image_size,
            use_roi_align=use_roi_align,
            roi_size=roi_size,
            use_id_embedding=use_id_embedding,
            embedding_dim=embedding_dim
        )
        print(f"   ✨ Enhanced model with ID embeddings (dim={embedding_dim})")
    else:
        # Load standard model
        from mots_exp.models.mv_center.mv_center_memory import MVCenterMemoryTracker
        
        model = MVCenterMemoryTracker(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            max_objects=max_objects,
            grid_size=grid_size,
            image_size=image_size,
            use_roi_align=use_roi_align,
            roi_size=roi_size
        )
        print(f"   📊 Standard memory tracker model")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Image size: {image_size}")
    print(f"   Grid size: {grid_size}")
    print(f"   Use ROI Align: {use_roi_align}")
    
    return model, model_info


def load_dataset(resolution=960):
    """Load MOTS dataset with motion vectors and RGB frames."""
    print(f"🔍 Loading dataset at {resolution}×{resolution}...")
    
    from dataset.factory.dataset_factory import create_mots_dataset
    
    dataset = create_mots_dataset(
        dataset_type="mot17",
        resolution=resolution,
        mode="train",
        load_iframe=True,      # Load I-frames for reference
        load_pframe=True,      # Load P-frames (RGB images)
        load_motion_vectors=True,
        load_residuals=False,
        load_annotations=True,
        sequence_length=48
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    return dataset


def group_gops(dataset):
    """Group dataset samples by GOP (using indices only, loading minimal data)."""
    from collections import defaultdict
    gop_groups = defaultdict(list)
    
    print(f"📊 Grouping {len(dataset)} samples by GOP (loading metadata only)...")
    
    # We need to load samples but can skip heavy pframe loading by temporarily modifying dataset
    # Store original load_pframe flag
    original_load_pframe = dataset.data_loader.load_pframe if hasattr(dataset, 'data_loader') else True
    
    # Disable pframe loading temporarily for faster metadata extraction  
    if hasattr(dataset, 'data_loader'):
        dataset.data_loader.load_pframe = False
    
    try:
        for idx in range(len(dataset)):
            try:
                sample = dataset[idx]
                if sample is None or 'sequence_id' not in sample:
                    continue
                seq_id = sample['sequence_id']
                gop_groups[seq_id].append(idx)
            except Exception as e:
                print(f"Warning: Could not load sample {idx}: {e}")
                continue
    finally:
        # Restore original setting
        if hasattr(dataset, 'data_loader'):
            dataset.data_loader.load_pframe = original_load_pframe
    
    # Sort indices within each GOP
    for seq_id in gop_groups:
        gop_groups[seq_id].sort()
    
    print(f"✅ Found {len(gop_groups)} GOPs")
    return gop_groups


def draw_boxes_on_canvas(canvas, boxes, confidences, color, label, thickness=2):
    """Draw bounding boxes on canvas with confidence scores."""
    h, w = canvas.shape[:2]
    
    for i, box in enumerate(boxes):
        if len(box) != 4:
            continue
        
        # Boxes are in normalized [cx, cy, w, h] format
        cx, cy, bw, bh = box
        
        # Convert to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Ensure within bounds
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        
        # Draw box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence if available
        if confidences is not None and i < len(confidences):
            label_text = f"{label} {i+1}: {confidences[i]:.2f}"
        else:
            label_text = f"{label} {i+1}"
        cv2.putText(canvas, label_text, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def create_motion_visualization(motion_vectors, size=(640, 640), arrow_scale=2.0, step=16):
    """Create visualization of motion vectors as arrows."""
    canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    
    mv_x = motion_vectors[0].cpu().numpy()
    mv_y = motion_vectors[1].cpu().numpy()
    
    # Get dimensions
    h, w = mv_x.shape
    
    # Calculate scaling factors
    scale_x = size[0] / w
    scale_y = size[1] / h
    
    # Draw arrows on a grid
    for y in range(0, h, max(1, step // int(scale_y))):
        for x in range(0, w, max(1, step // int(scale_x))):
            dx = float(mv_x[y, x]) * arrow_scale
            dy = float(mv_y[y, x]) * arrow_scale
            
            # Only draw if motion is significant
            magnitude = np.sqrt(dx*dx + dy*dy)
            if magnitude > 0.5:
                # Start point in pixel coordinates
                start_x = int((x + 0.5) * scale_x)
                start_y = int((y + 0.5) * scale_y)
                start = (start_x, start_y)
                
                # End point
                end = (int(start_x + dx), int(start_y + dy))
                
                # Color based on magnitude (green to red)
                normalized_mag = min(magnitude / 20.0, 1.0)
                color = (
                    int(normalized_mag * 255),       # Red
                    int((1 - normalized_mag) * 255), # Green
                    0                                # Blue
                )
                
                cv2.arrowedLine(canvas, start, end, color, 1, tipLength=0.3)
    
    return canvas


def create_gop_video(model, gop_name, gop_indices, dataset, output_path, device, fps=5, arrow_scale=2.0):
    """Create visualization video for one GOP."""
    print(f"   🎬 Creating video for {gop_name}...")
    
    # Prepare GOP data
    motion_sequence = []
    gt_boxes_sequence = []
    rgb_frames = []  # Store RGB P-frames
    
    # Load samples from indices
    for idx in gop_indices:
        sample = dataset[idx]
        if sample is None:
            continue
            
        motion_vectors = sample.get('motion_vectors')
        boxes = sample.get('boxes', torch.zeros(0, 4))
        
        if motion_vectors is None:
            continue
        
        # Format motion vectors
        if isinstance(motion_vectors, torch.Tensor):
            if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                motion_vectors = motion_vectors[0].permute(2, 0, 1)
            elif len(motion_vectors.shape) == 3 and motion_vectors.shape[2] == 2:
                motion_vectors = motion_vectors.permute(2, 0, 1)
        
        motion_sequence.append(motion_vectors.to(device))
        
        # Format GT boxes
        if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
            gt_boxes = boxes.view(-1, 4)
            gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
        else:
            gt_boxes = torch.zeros(0, 4)
        gt_boxes_sequence.append(gt_boxes)
        
        # Get RGB P-frame
        pframe_bgr = None
        pframe = sample.get('pframe')
        
        if pframe is not None and isinstance(pframe, torch.Tensor):
            try:
                # Dataset returns frames as [1, H, W, 3] in RGB format, float32, range [0, 255]
                pframe_np = pframe.squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Convert RGB -> BGR for OpenCV
                if pframe_np.shape[-1] == 3:
                    pframe_bgr = cv2.cvtColor(pframe_np, cv2.COLOR_RGB2BGR)
                else:
                    pframe_bgr = pframe_np
                
                # Resize to standard size
                pframe_bgr = cv2.resize(pframe_bgr, (640, 640))
            except Exception as e:
                print(f"   ⚠️ Error loading P-frame: {e}")
                pframe_bgr = None
        
        # Try iframe as fallback if pframe failed
        if pframe_bgr is None:
            iframe = sample.get('iframe')
            if iframe is not None and isinstance(iframe, torch.Tensor):
                try:
                    # Dataset returns frames as [1, H, W, 3] in RGB format, float32, range [0, 255]
                    iframe_np = iframe.squeeze(0).cpu().numpy().astype(np.uint8)
                    
                    # Convert RGB -> BGR for OpenCV
                    if iframe_np.shape[-1] == 3:
                        pframe_bgr = cv2.cvtColor(iframe_np, cv2.COLOR_RGB2BGR)
                    else:
                        pframe_bgr = iframe_np
                    
                    pframe_bgr = cv2.resize(pframe_bgr, (640, 640))
                except Exception as e:
                    print(f"   ⚠️ Error loading I-frame: {e}")
                    pframe_bgr = None
        
        # Final fallback: create gray frame if no image available
        if pframe_bgr is None:
            pframe_bgr = np.ones((640, 640, 3), dtype=np.uint8) * 50  # Dark gray
        
        rgb_frames.append(pframe_bgr)
    
    if len(motion_sequence) == 0:
        print(f"   ⚠️ No valid frames")
        return False
    
    # Run GOP inference
    motion_batch = torch.stack(motion_sequence, dim=0)
    iframe_boxes = gt_boxes_sequence[0].to(device)
    
    model.reset()
    with torch.no_grad():
        outputs = model.forward_gop(motion_batch, iframe_boxes)
        
        # Handle both standard and enhanced models
        if isinstance(outputs, tuple) and len(outputs) == 3:
            predictions_list, confidences_list, embeddings_list = outputs
        else:
            predictions_list, confidences_list = outputs
    
    # Create video with 3-panel layout: RGB | Motion Arrows | Predictions
    frame_size = (1920, 640)  # 3 panels of 640x640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    for t in range(len(motion_sequence)):
        # Get motion vectors
        mv = motion_sequence[t].cpu()
        
        # Get GT boxes
        gt_boxes = gt_boxes_sequence[t].cpu().numpy()
        
        # Get predicted boxes
        pred_boxes = predictions_list[t].cpu().numpy() if isinstance(predictions_list[t], torch.Tensor) else predictions_list[t]
        pred_conf = confidences_list[t].cpu().numpy() if isinstance(confidences_list[t], torch.Tensor) else confidences_list[t]
        
        # Panel 1: RGB P-frame with annotations
        rgb_panel = rgb_frames[t].copy()
        draw_boxes_on_canvas(rgb_panel, gt_boxes, None, (0, 255, 0), "GT", thickness=2)
        draw_boxes_on_canvas(rgb_panel, pred_boxes, pred_conf, (0, 0, 255), "Pred", thickness=2)
        
        # Panel 2: Motion vector arrows WITH BOUNDING BOXES
        motion_panel = create_motion_visualization(mv, size=(640, 640), arrow_scale=arrow_scale)
        # Add bounding boxes on motion panel
        draw_boxes_on_canvas(motion_panel, gt_boxes, None, (0, 255, 0), "GT", thickness=2)
        draw_boxes_on_canvas(motion_panel, pred_boxes, pred_conf, (255, 0, 255), "Pred", thickness=2)  # Magenta for visibility
        
        # Panel 3: Predictions vs GT on black canvas
        pred_panel = np.zeros((640, 640, 3), dtype=np.uint8)
        draw_boxes_on_canvas(pred_panel, gt_boxes, None, (0, 255, 0), "GT", thickness=2)
        draw_boxes_on_canvas(pred_panel, pred_boxes, pred_conf, (0, 0, 255), "Pred", thickness=2)
        
        # Combine panels horizontally
        frame = np.hstack([rgb_panel, motion_panel, pred_panel])
        
        # Add text overlay
        text = f"Frame {t}/{len(motion_sequence)-1} | GT: {len(gt_boxes)} (Green) | Pred: {len(pred_boxes)} (Red/Magenta)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add panel labels
        cv2.putText(frame, "RGB P-Frame + Boxes", (180, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Motion Vectors + Boxes", (800, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Predictions vs GT", (1460, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"   ✅ Video saved: {output_path}")
    return True


def main():
    args = parse_arguments()
    
    print(f"🚀 Memory Tracker Visualization")
    print(f"=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, model_info = load_model(args.model_path, args.device)
    
    # Get resolution from model
    resolution = model_info.get('image_size', 960)
    
    # Load dataset with matching resolution
    dataset = load_dataset(resolution=resolution)
    
    # Group by GOP
    gop_groups = group_gops(dataset)
    
    # Group GOPs by video sequence (extract base sequence name from gop_name)
    from collections import defaultdict
    sequence_gops = defaultdict(list)
    for gop_name in gop_groups.keys():
        # Extract base sequence name (e.g., "MOT17-09-SDP_960x960_gop50_500frames_gop0" -> "MOT17-09-SDP")
        # The format is: <sequence>_<resolution>_gop<length>_<totalframes>_gop<number>
        parts = gop_name.split('_')
        # Find the sequence name (everything before resolution pattern)
        base_name = parts[0]  # e.g., "MOT17-09-SDP"
        sequence_gops[base_name].append(gop_name)
    
    print(f"📊 Found {len(sequence_gops)} unique video sequences with {len(gop_groups)} total GOPs")
    
    # Select GOPs to process
    gop_names = []
    if args.max_gops is None:
        # Process all GOPs
        gop_names = list(gop_groups.keys())
        print(f"📊 Processing ALL {len(gop_names)} GOPs")
    else:
        # Process max_gops GOPs from EACH video sequence
        for seq_name in sorted(sequence_gops.keys()):
            seq_gops = sorted(sequence_gops[seq_name])[:args.max_gops]
            gop_names.extend(seq_gops)
        print(f"📊 Processing {args.max_gops} GOP(s) from each of {len(sequence_gops)} video sequences = {len(gop_names)} videos total")
    
    for gop_name in tqdm(gop_names, desc="Creating videos"):
        gop_indices = gop_groups[gop_name]
        output_path = os.path.join(args.output_dir, f"{gop_name}_visualization.mp4")
        
        create_gop_video(model, gop_name, gop_indices, dataset, output_path, 
                        args.device, args.fps, args.arrow_scale)
    
    print(f"\n✅ Created {len(gop_names)} visualization videos in {args.output_dir}")


if __name__ == "__main__":
    main()
