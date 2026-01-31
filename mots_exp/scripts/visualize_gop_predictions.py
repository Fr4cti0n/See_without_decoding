"""
Visualize bounding box predictions across an entire GOP.
This helps debug what the model is actually predicting.
"""
import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dataset.factory.dataset_factory import DatasetFactory
from mots_exp.models.mv_center.mv_center_v1 import create_mv_center_v1
from models.id_aware_tracker import IDMultiObjectTracker


def load_model(model_path, device):
    """Load model and detect type from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Detect model type
    if 'model_info' in checkpoint:
        # MV-Center model
        model_info = checkpoint['model_info']
        input_channels = model_info.get('input_channels', 2)
        model = create_mv_center_v1(input_channels=input_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
        max_objects = 100
        use_magnitude = input_channels == 3
        model_type = 'mv_center'
        print(f"📊 Loaded MV-Center model: {model_info.get('version', 'v1')}, channels={input_channels}")
    elif 'config' in checkpoint:
        # ID-Tracker model
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
        raise ValueError("Unknown checkpoint format")
    
    model = model.to(device)
    model.eval()
    
    return model, max_objects, use_magnitude, model_type


def convert_mv_center_predictions_to_boxes(predictions, max_objects=100, score_threshold=0.1):
    """Convert MV-Center predictions (heatmaps) to bounding boxes"""
    all_boxes = []
    all_scores = []
    
    for level_name, level_preds in predictions.items():
        # Extract center heatmap and box parameters
        center_map = level_preds['center'][0, 0]  # [H, W]
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
            
            # Convert to [x1, y1, x2, y2] format
            x1 = cx - box_w / 2
            y1 = cy - box_h / 2
            x2 = cx + box_w / 2
            y2 = cy + box_h / 2
            
            all_boxes.append([x1, y1, x2, y2])
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


def visualize_gop(model, dataset, gop_samples, max_objects, device, use_magnitude, model_type, output_dir):
    """Visualize predictions for an entire GOP"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracking state
    prev_boxes = None
    object_ids = None
    valid_mask = None
    
    # Create figure with subplots
    num_frames = len(gop_samples)
    cols = 8
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    print(f"\n📊 Visualizing GOP with {num_frames} frames...")
    
    with torch.no_grad():
        for frame_idx, (_, sample) in enumerate(gop_samples):
            ax = axes[frame_idx]
            
            # Extract motion vectors
            motion_vectors = sample.get('motion_vectors')
            boxes_data = sample.get('boxes', torch.zeros(1, 4))
            
            if motion_vectors is None:
                ax.text(0.5, 0.5, 'No MV', ha='center', va='center')
                ax.axis('off')
                continue
            
            # Handle motion vector format
            if isinstance(motion_vectors, torch.Tensor):
                if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                    motion_vectors = motion_vectors[0].permute(2, 0, 1)
                elif len(motion_vectors.shape) == 3 and motion_vectors.shape[-1] == 2:
                    motion_vectors = motion_vectors[0] if motion_vectors.shape[0] > 1 else motion_vectors.squeeze(-1)
                
                if motion_vectors.shape[0] != 2:
                    motion_vectors = motion_vectors[:2] if motion_vectors.shape[0] > 2 else torch.zeros(2, 40, 40)
            
            # Process motion vectors
            mv_x, mv_y = motion_vectors[0], motion_vectors[1]
            
            if use_magnitude:
                mv_magnitude = torch.sqrt(mv_x**2 + mv_y**2)
                motion_vectors_final = torch.stack([mv_x, mv_y, mv_magnitude], dim=0)
            else:
                motion_vectors_final = torch.stack([mv_x, mv_y], dim=0)
            
            motion_tensor = motion_vectors_final.unsqueeze(0).to(device)
            
            # Prepare ground truth boxes
            if isinstance(boxes_data, torch.Tensor) and boxes_data.numel() > 0:
                gt_boxes = boxes_data.view(-1, 4) if len(boxes_data.shape) > 1 else boxes_data.unsqueeze(0)
                gt_boxes = gt_boxes[gt_boxes.sum(dim=1) != 0]
            else:
                gt_boxes = torch.zeros(0, 4)
            
            # Initialize tracking state for first frame
            if frame_idx == 0 or prev_boxes is None:
                prev_boxes = torch.zeros(1, max_objects, 4, device=device)
                object_ids = torch.zeros(1, max_objects, dtype=torch.long, device=device)
                valid_mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
                
                # Initialize with ground truth
                if len(gt_boxes) > 0:
                    num_objects = min(len(gt_boxes), max_objects)
                    prev_boxes[0, :num_objects] = gt_boxes[:num_objects].to(device)
                    object_ids[0, :num_objects] = torch.arange(1, num_objects + 1, device=device)
                    valid_mask[0, :num_objects] = True
            
            # Get predictions
            if frame_idx == 0:
                # I-frame: use GT
                predicted_boxes = prev_boxes.clone()
            else:
                # P-frame: use model
                predicted_boxes, _, _ = run_model_inference(
                    model, model_type, motion_tensor, prev_boxes, object_ids, valid_mask, max_objects
                )
            
            # Visualize motion vectors as background
            mv_magnitude_viz = torch.sqrt(mv_x**2 + mv_y**2).cpu().numpy()
            ax.imshow(mv_magnitude_viz, cmap='gray', alpha=0.5)
            
            # Draw GT boxes in green
            for box in gt_boxes:
                x1, y1, x2, y2 = box.cpu().numpy()
                # Convert from normalized to pixel coordinates (40x40 motion vector grid)
                x1, x2 = x1 * 40, x2 * 40
                y1, y2 = y1 * 40, y2 * 40
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', facecolor='none', label='GT')
                ax.add_patch(rect)
            
            # Draw predicted boxes in red
            pred_boxes_cpu = predicted_boxes[0].cpu()
            valid_pred_mask = valid_mask[0].cpu() if frame_idx == 0 else (pred_boxes_cpu.sum(dim=1) != 0)
            
            for idx, (box, is_valid) in enumerate(zip(pred_boxes_cpu, valid_pred_mask)):
                if is_valid:
                    x1, y1, x2, y2 = box.numpy()
                    # Convert from normalized to pixel coordinates
                    x1, x2 = x1 * 40, x2 * 40
                    y1, y2 = y1 * 40, y2 * 40
                    w, h = x2 - x1, y2 - y1
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label='Pred')
                    ax.add_patch(rect)
            
            # Set title
            frame_type = "I-frame (GT)" if frame_idx == 0 else "P-frame (Model)"
            num_pred = valid_mask[0].sum().item() if frame_idx == 0 else valid_pred_mask.sum().item()
            ax.set_title(f"Frame {frame_idx} ({frame_type})\nGT: {len(gt_boxes)}, Pred: {num_pred}", fontsize=10)
            ax.axis('off')
            
            # Update state
            prev_boxes = predicted_boxes.detach().clone()
            
            print(f"  Frame {frame_idx}: GT boxes={len(gt_boxes)}, Pred boxes={num_pred}")
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    # Add legend
    handles = [
        patches.Patch(color='green', label='Ground Truth'),
        patches.Patch(color='red', label='Prediction')
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=12)
    
    # Save figure
    output_path = os.path.join(output_dir, 'gop_visualization.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize GOP predictions')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset-dir', default='/home/aduche/Bureau/motion_sight_back_up/data/mots/MVextracted_data', 
                       help='Path to dataset')
    parser.add_argument('--output-dir', default='outputs/gop_visualization', help='Output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gop-idx', type=int, default=0, help='GOP index to visualize')
    parser.add_argument('--video-idx', type=int, default=0, help='Video index to use')
    
    args = parser.parse_args()
    
    # Load model
    print("📦 Loading model...")
    model, max_objects, use_magnitude, model_type = load_model(args.model_path, args.device)
    
    # Load dataset
    print("📦 Loading dataset...")
    dataset = DatasetFactory.create_dataset(
        dataset_type='mots',
        dataset_dir=args.dataset_dir,
        split='train',
        is_train=False
    )
    
    print(f"📊 Dataset loaded: {len(dataset)} samples")
    
    # Group samples by video and GOP
    from collections import defaultdict
    video_gop_groups = defaultdict(lambda: defaultdict(list))
    
    for idx in range(len(dataset)):
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
    
    # Select video and GOP
    video_names = sorted(video_gop_groups.keys())
    if args.video_idx >= len(video_names):
        print(f"❌ Video index {args.video_idx} out of range. Available: {len(video_names)}")
        return
    
    video_name = video_names[args.video_idx]
    gop_dict = video_gop_groups[video_name]
    
    if args.gop_idx not in gop_dict:
        print(f"❌ GOP index {args.gop_idx} not found. Available: {sorted(gop_dict.keys())}")
        return
    
    gop_samples = gop_dict[args.gop_idx]
    gop_samples.sort(key=lambda x: x[0])
    
    print(f"\n📊 Visualizing: {video_name}, GOP {args.gop_idx} ({len(gop_samples)} frames)")
    
    # Visualize
    visualize_gop(model, dataset, gop_samples, max_objects, args.device, use_magnitude, model_type, args.output_dir)


if __name__ == '__main__':
    main()
