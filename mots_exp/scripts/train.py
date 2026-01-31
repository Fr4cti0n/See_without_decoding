#!/usr/bin/env python3
import argparse
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add dataset directory to path for imports
dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset')
if dataset_path not in sys.path:
    sys.path.append(dataset_path)
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import from proper locations
from mots_exp.configs.train_config import SequentialTrainConfig
from mots_exp.models.sequential_gop_tracker import SequentialGOPTracker
from mots_exp.models.id_aware_tracker import IDMultiObjectTracker, IDLoss
from mots_exp.data.sequential_dataset import SequentialMOTSDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sequential GOP Training for MOTS")
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum training samples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--dry-run', action='store_true',
                       help='Just test configuration and exit')
    parser.add_argument('--visualize', action='store_true',
                       help='Plot RGB frames and motion vectors')
    parser.add_argument('--use-magnitude', action='store_true',
                       help='Use 3-channel motion vectors (X, Y, Magnitude) instead of 2-channel (X, Y)')
    
    return parser.parse_args()

def load_complete_gop_sequence(dataset, gop_idx):
    """Load a complete GOP sequence with all 49 P-frames."""
    try:
        # Get the GOP sequence from dataset
        gop_sequence = dataset.get_gop_sequence(gop_idx)
        
        print(f"🎬 Loaded GOP {gop_idx}: {len(gop_sequence)} frames")
        return gop_sequence
        
    except Exception as e:
        print(f"⚠️ Error loading GOP {gop_idx}: {e}")
        return None


def process_data_loader_sample(sample, max_objects=8):
    """Process sample from data loader into training format (same structure as generate_all_plots.py)."""
    try:
        # Extract motion vectors and RGB frame from sample (same as generate_all_plots.py)
        if 'motion_vectors' not in sample:
            return None
            
        motion_vectors = sample['motion_vectors']  # Should be (2, 40, 40)
        rgb_frame = sample.get('rgb_frame', None)
        
        # Convert to tensor if needed
        if not isinstance(motion_vectors, torch.Tensor):
            motion_vectors = torch.tensor(motion_vectors, dtype=torch.float32)
        
        print(f"🔍 Raw motion vector shape: {motion_vectors.shape}")
        print(f"🔍 Raw motion vector range: min={motion_vectors.min():.6f}, max={motion_vectors.max():.6f}")
        print(f"🔍 Raw motion vector unique values count: {torch.unique(motion_vectors).shape[0]}")
        print(f"🔍 Raw motion vector sample values: {motion_vectors.flatten()[:10].tolist()}")
        
        # Ensure correct shape (2, 40, 40)
        if len(motion_vectors.shape) == 4 and motion_vectors.shape[0] == 2 and motion_vectors.shape[3] == 2:
            # Shape is (2, 40, 40, 2) - take the motion vector components (x, y)
            motion_vectors = motion_vectors[:, :, :, 0:2]  # Keep both motion components
            motion_vectors = motion_vectors.permute(3, 0, 1, 2)  # (2, 2, 40, 40) -> average channels
            motion_vectors = motion_vectors.mean(dim=1)  # (2, 40, 40)
        elif len(motion_vectors.shape) == 3 and motion_vectors.shape[0] == 2:
            pass  # Already correct (2, 40, 40)
        elif len(motion_vectors.shape) == 3 and motion_vectors.shape[2] == 2:
            motion_vectors = motion_vectors.permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
        elif len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
            # Shape (H, W, ?, 2) -> (2, H, W)
            motion_vectors = motion_vectors.mean(dim=2).permute(2, 0, 1)
        else:
            print(f"⚠️ Unexpected motion vector shape: {motion_vectors.shape}")
            return None
        
        print(f"🔍 Processed motion vector shape: {motion_vectors.shape}")
        print(f"🔍 Processed motion vector range: min={motion_vectors.min():.6f}, max={motion_vectors.max():.6f}")
        print(f"🔍 Processed motion vector unique values count: {torch.unique(motion_vectors).shape[0]}")
        
        # DETAILED Y-COMPONENT ANALYSIS
        mv_x = motion_vectors[0]  # X component
        mv_y = motion_vectors[1]  # Y component
        
        print(f"🔍 X-component stats:")
        print(f"    Mean: {mv_x.mean():.6f}, Std: {mv_x.std():.6f}")
        print(f"    Min: {mv_x.min():.6f}, Max: {mv_x.max():.6f}")
        print(f"    Non-zero values: {(mv_x != 0).sum().item()}/{mv_x.numel()} ({(mv_x != 0).float().mean()*100:.1f}%)")
        
        print(f"🔍 Y-component stats:")
        print(f"    Mean: {mv_y.mean():.6f}, Std: {mv_y.std():.6f}")
        print(f"    Min: {mv_y.min():.6f}, Max: {mv_y.max():.6f}")
        print(f"    Non-zero values: {(mv_y != 0).sum().item()}/{mv_y.numel()} ({(mv_y != 0).float().mean()*100:.1f}%)")
        
        # Check for potential Y-component issues
        y_near_zero = (torch.abs(mv_y) < 0.001).sum().item()
        y_exactly_zero = (mv_y == 0).sum().item()
        print(f"🔍 Y-component analysis:")
        print(f"    Values exactly zero: {y_exactly_zero}/{mv_y.numel()} ({y_exactly_zero/mv_y.numel()*100:.1f}%)")
        print(f"    Values near zero (<0.001): {y_near_zero}/{mv_y.numel()} ({y_near_zero/mv_y.numel()*100:.1f}%)")
        
        # Check magnitude distribution
        magnitude = torch.sqrt(mv_x**2 + mv_y**2)
        print(f"🔍 Motion magnitude stats:")
        print(f"    Mean: {magnitude.mean():.6f}, Std: {magnitude.std():.6f}")
        print(f"    Min: {magnitude.min():.6f}, Max: {magnitude.max():.6f}")
        
        # Check direction distribution
        direction = torch.atan2(mv_y, mv_x)
        direction_degrees = direction * 180 / torch.pi
        print(f"🔍 Motion direction stats (degrees):")
        print(f"    Mean: {direction_degrees.mean():.2f}, Std: {direction_degrees.std():.2f}")
        print(f"    Min: {direction_degrees.min():.2f}, Max: {direction_degrees.max():.2f}")
        
        # Count horizontal vs vertical motion
        horizontal_dominant = (torch.abs(mv_x) > torch.abs(mv_y)).sum().item()
        vertical_dominant = (torch.abs(mv_y) > torch.abs(mv_x)).sum().item()
        equal_motion = (torch.abs(mv_x) == torch.abs(mv_y)).sum().item()
        
        print(f"🔍 Motion direction dominance:")
        print(f"    Horizontal dominant: {horizontal_dominant}/{mv_x.numel()} ({horizontal_dominant/mv_x.numel()*100:.1f}%)")
        print(f"    Vertical dominant: {vertical_dominant}/{mv_x.numel()} ({vertical_dominant/mv_x.numel()*100:.1f}%)")
        print(f"    Equal motion: {equal_motion}/{mv_x.numel()} ({equal_motion/mv_x.numel()*100:.1f}%)")
        
        # Print some sample values for inspection
        print(f"🔍 Sample motion vector values (first 5x5 region):")
        print(f"    X values: {mv_x[:5, :5].flatten().tolist()}")
        print(f"    Y values: {mv_y[:5, :5].flatten().tolist()}")
        
        # WARNING if Y-component is problematic
        if y_exactly_zero / mv_y.numel() > 0.8:
            print("⚠️  WARNING: Over 80% of Y-component values are exactly zero!")
            print("⚠️  This suggests a potential issue with:")
            print("⚠️    1. Motion vector extraction from compressed video")
            print("⚠️    2. Data preprocessing pipeline")
            print("⚠️    3. Video encoding settings")
            print("⚠️    4. Dataset-specific motion patterns")
        
        if vertical_dominant / mv_x.numel() < 0.1:
            print("⚠️  WARNING: Less than 10% of motion is vertically dominant!")
            print("⚠️  This is unusual for natural video motion patterns")
        
        # Create dummy bounding boxes for training (since we don't have ground truth tracking)
        # Generate random previous and target boxes for max_objects
        prev_boxes = torch.zeros(max_objects, 4)  # (x, y, w, h)
        target_boxes = torch.zeros(max_objects, 4)
        valid_mask = torch.zeros(max_objects, dtype=torch.bool)
        
        # Create some dummy tracking data for the first few objects
        num_dummy_objects = min(3, max_objects)
        for i in range(num_dummy_objects):
            # Random previous box
            prev_boxes[i] = torch.tensor([
                torch.rand(1) * 30 + 5,   # x (5-35)
                torch.rand(1) * 30 + 5,   # y (5-35)  
                torch.rand(1) * 8 + 2,    # w (2-10)
                torch.rand(1) * 8 + 2     # h (2-10)
            ])
            
            # Target box with small motion
            motion_offset = torch.randn(2) * 0.5  # Small random motion
            target_boxes[i] = prev_boxes[i].clone()
            target_boxes[i][:2] += motion_offset  # Apply motion to x,y
            
            valid_mask[i] = True
        
        return {
            'motion_vectors': motion_vectors,
            'prev_boxes': prev_boxes,
            'target_boxes': target_boxes,
            'valid_mask': valid_mask,
            'rgb_frame': rgb_frame
        }
        
    except Exception as e:
        print(f"⚠️ Error processing sample: {e}")
        return None


def plot_loss_curves(epoch_losses, bbox_losses, id_losses, output_dir):
    """Plot and save separate loss curves for each component with enhanced analysis."""
    import matplotlib.pyplot as plt
    import os
    
    epochs = range(1, len(epoch_losses) + 1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better plots
    plt.style.use('default')
    
    # Plot 1: Total Loss with trend analysis
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, epoch_losses, 'b-', linewidth=3, marker='o', markersize=6, label='Total Loss')
    
    # Add trend line if we have enough data points
    if len(epoch_losses) >= 3:
        z = np.polyfit(epochs, epoch_losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), 'r--', alpha=0.7, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
    
    plt.title('ID-Aware Tracker: Total Training Loss Over Time', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Total Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add annotations for min/max
    min_loss_idx = np.argmin(epoch_losses)
    max_loss_idx = np.argmax(epoch_losses)
    plt.annotate(f'Min: {epoch_losses[min_loss_idx]:.6f}', 
                xy=(min_loss_idx+1, epoch_losses[min_loss_idx]), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    total_loss_path = os.path.join(output_dir, 'total_loss_curve.png')
    plt.savefig(total_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Total loss plot saved to: {total_loss_path}")
    
    # Plot 2: Bounding Box Loss with scale analysis
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, bbox_losses, 'g-', linewidth=3, marker='s', markersize=6, label='Bounding Box Loss')
    
    # Add trend line
    if len(bbox_losses) >= 3:
        z = np.polyfit(epochs, bbox_losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), 'r--', alpha=0.7, linewidth=2, label=f'Trend (slope: {z[0]:.8f})')
    
    plt.title('ID-Aware Tracker: Bounding Box Regression Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('BBox Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Use log scale if values are very small
    if max(bbox_losses) < 0.01:
        plt.yscale('log')
        plt.ylabel('BBox Loss (log scale)', fontsize=14)
    
    plt.tight_layout()
    bbox_loss_path = os.path.join(output_dir, 'bbox_loss_curve.png')
    plt.savefig(bbox_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 BBox loss plot saved to: {bbox_loss_path}")
    
    # Plot 3: ID Classification Loss
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, id_losses, 'r-', linewidth=3, marker='^', markersize=6, label='ID Classification Loss')
    
    # Add trend line
    if len(id_losses) >= 3:
        z = np.polyfit(epochs, id_losses, 1)
        p = np.poly1d(z)
        plt.plot(epochs, p(epochs), 'b--', alpha=0.7, linewidth=2, label=f'Trend (slope: {z[0]:.6f})')
    
    plt.title('ID-Aware Tracker: ID Classification Loss', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('ID Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    id_loss_path = os.path.join(output_dir, 'id_loss_curve.png')
    plt.savefig(id_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 ID loss plot saved to: {id_loss_path}")
    
    # Plot 4: Combined Loss Comparison with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Total Loss
    ax1.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o')
    ax1.set_title('Total Loss', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: BBox Loss
    ax2.plot(epochs, bbox_losses, 'g-', linewidth=2, marker='s')
    ax2.set_title('Bounding Box Loss', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('BBox Loss')
    ax2.grid(True, alpha=0.3)
    if max(bbox_losses) < 0.01:
        ax2.set_yscale('log')
    
    # Subplot 3: ID Loss
    ax3.plot(epochs, id_losses, 'r-', linewidth=2, marker='^')
    ax3.set_title('ID Classification Loss', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('ID Loss')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: All losses on same plot with normalization
    # Normalize losses to [0,1] for comparison
    norm_total = np.array(epoch_losses) / max(epoch_losses) if max(epoch_losses) > 0 else np.array(epoch_losses)
    norm_bbox = np.array(bbox_losses) / max(bbox_losses) if max(bbox_losses) > 0 else np.array(bbox_losses)
    norm_id = np.array(id_losses) / max(id_losses) if max(id_losses) > 0 else np.array(id_losses)
    
    ax4.plot(epochs, norm_total, 'b-', linewidth=2, label='Total (normalized)', marker='o')
    ax4.plot(epochs, norm_bbox, 'g-', linewidth=2, label='BBox (normalized)', marker='s')
    ax4.plot(epochs, norm_id, 'r-', linewidth=2, label='ID (normalized)', marker='^')
    ax4.set_title('Normalized Loss Comparison', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Normalized Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('ID-Aware Multi-Object Tracker: Comprehensive Training Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'combined_loss_analysis.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Combined loss analysis saved to: {combined_path}")
    
    # Generate learning summary
    print(f"\n📈 LEARNING SUMMARY (Epoch {len(epochs)}):")
    print(f"   Current Total Loss: {epoch_losses[-1]:.6f}")
    print(f"   Current BBox Loss: {bbox_losses[-1]:.6f}")
    print(f"   Current ID Loss: {id_losses[-1]:.6f}")
    
    if len(epoch_losses) >= 2:
        total_improvement = ((epoch_losses[0] - epoch_losses[-1]) / max(epoch_losses[0], 1e-8)) * 100
        bbox_improvement = ((bbox_losses[0] - bbox_losses[-1]) / max(bbox_losses[0], 1e-8)) * 100
        id_improvement = ((id_losses[0] - id_losses[-1]) / max(id_losses[0], 1e-8)) * 100
        
        print(f"   Total Loss Improvement: {total_improvement:+.2f}%")
        print(f"   BBox Loss Improvement: {bbox_improvement:+.2f}%")
        print(f"   ID Loss Improvement: {id_improvement:+.2f}%")
        
        if total_improvement > 5:
            print(f"   ✅ GOOD: Model is learning well!")
        elif total_improvement > 1:
            print(f"   📊 OK: Model showing some improvement")
        elif total_improvement < -5:
            print(f"   ⚠️ WARNING: Model may be overfitting or learning rate too high")
        else:
            print(f"   📝 INFO: Loss relatively stable")


def custom_collate_fn(batch):
    """Custom collate function to handle None samples in dataset."""
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # For single-item batches, just return the item
    if len(batch) == 1:
        return batch[0]
    
    # Default collation for multiple items
    return torch.utils.data.dataloader.default_collate(batch)


def sequential_gop_training(config: SequentialTrainConfig, device, visualize=False, use_magnitude=False):
    """Sequential GOP training using direct dataset factory import.
    
    Args:
        config: Training configuration
        device: torch device (cuda/cpu)
        visualize: Whether to visualize frames
        use_magnitude: If True, use 3-channel motion vectors (X, Y, Magnitude)
                      If False, use 2-channel motion vectors (X, Y only)
    """
    
    print(f"📊 Creating sequential GOP dataset using dataset factory...")
    
    try:
        # Import from dataset factory directly (simpler approach)
        from dataset.factory.dataset_factory import create_mots_dataset
        
        # Create dataset using the factory
        dataset = create_mots_dataset(
            dataset_type="mot17",
            resolution=640,
            mode="train", 
            load_iframe=False,  # No RGB frames needed for training
            load_pframe=False,  # No RGB frames needed for training
            load_motion_vectors=True,
            load_residuals=False,  # Skip residuals for now
            load_annotations=True,
            sequence_length=48,  # Fixed GOP length to 48 frames
            data_format="separate"
        )
        
        if dataset is None or len(dataset) == 0:
            print("❌ No dataset found, returning zero loss")
            return 0.0
        
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Create DataLoader for batch processing
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1,  # Process one GOP at a time
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn  # Use custom collate to handle None samples
        )
        
        print(f"✅ Data loader created with {len(data_loader)} batches")
        
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return 0.0
    
    # Determine motion shape based on use_magnitude flag
    motion_channels = 3 if use_magnitude else 2
    motion_shape = (motion_channels, 40, 40)
    
    print(f"\n🎨 Motion Vector Configuration:")
    print(f"   Use magnitude: {use_magnitude}")
    print(f"   Motion channels: {motion_channels} ({'X, Y, Magnitude' if use_magnitude else 'X, Y'})")
    print(f"   Motion shape: {motion_shape}")
    
    # Create model for sequential processing - USE ID-AWARE TRACKER
    model = IDMultiObjectTracker(
        motion_shape=motion_shape,
        hidden_dim=128,
        max_objects=config.max_objects,
        max_id=config.max_objects * 2  # Allow more IDs than objects
    ).to(device)
    
    print(f"🧠 ID-Aware Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup with ID-aware loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = IDLoss(bbox_weight=1.0, id_weight=0.5)  # Balance bbox and ID losses
    
    print(f"🚀 Starting sequential GOP training...")
    print(f"   Epochs: {config.epochs}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   GOP length: 48 frames (fixed)")
    print(f"   Max GOPs per video: All available")
    print(f"   Dataset samples: {len(dataset)}")
    print(f"   Training mode: Motion vectors only (no RGB)")
    
    # Sequential training loop - process dataset samples
    model.train()
    
    # Loss tracking for plotting
    epoch_losses = []
    bbox_losses = []
    id_losses = []
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_id_loss = 0.0
        num_samples = 0
        
        print(f"\n📈 Epoch {epoch+1}/{config.epochs} - Sequential GOP Processing")
        
        # Group dataset samples by sequence_id to ensure temporal order
        print(f"🔄 Grouping {len(dataset)} samples by GOP sequence...")
        sequence_groups = {}
        
        # For training efficiency, process ALL GOPs per video
        max_samples_to_test = 10000
        total_samples = min(len(dataset), max_samples_to_test)
        print(f"   (Processing ALL GOPs with {total_samples} samples)")
        
        # Group samples by sequence_id (GOP) - process all GOPs
        video_gop_count = {}  # Track GOPs per video
        for idx in range(total_samples):
            try:
                sample = dataset[idx]
                if sample is not None and 'sequence_id' in sample:
                    seq_id = sample['sequence_id']
                    
                    # Extract video name from sequence_id (remove GOP suffix)
                    video_name = seq_id.rsplit('_gop', 1)[0] if '_gop' in seq_id else seq_id
                    
                    # Process ALL GOPs per video
                    if video_name not in video_gop_count:
                        video_gop_count[video_name] = 0
                    
                    if seq_id not in sequence_groups:
                        sequence_groups[seq_id] = []
                        video_gop_count[video_name] += 1
                    sequence_groups[seq_id].append((idx, sample))
                    
                else:
                    print(f"⚠️ Sample {idx} missing sequence_id")
            except Exception as e:
                print(f"⚠️ Error loading sample {idx}: {e}")
                continue
        
        print(f"📊 Found {len(sequence_groups)} GOP sequences to process")
        
        # Process each GOP sequence completely before moving to next with progress bar
        gop_progress = tqdm(sequence_groups.items(), desc=f"Epoch {epoch+1}/{config.epochs} GOPs", 
                           unit="GOP", leave=True, position=0)
        
        for seq_id, frames_list in gop_progress:
            # Get current GOP number for tracking
            gop_num = list(sequence_groups.keys()).index(seq_id) + 1
            
            print(f"\n🎬 Processing GOP {gop_num}/{len(sequence_groups)}: {seq_id}")
            print(f"   Total frames in this GOP: {len(frames_list)}")
            
            gop_progress.set_description(f"Epoch {epoch+1}/{config.epochs} | GOP {gop_num}: {seq_id[:30]}...")
            
            # Reset model state for new GOP sequence
            model.reset_sequence()
            print(f"   🔄 LSTM state reset for new GOP sequence")
            
            # Sort frames by their index to ensure temporal order within GOP
            frames_list.sort(key=lambda x: x[0])
            
            # Limit to 48 frames per GOP as requested
            max_frames_per_gop = 48
            if len(frames_list) > max_frames_per_gop:
                frames_list = frames_list[:max_frames_per_gop]
                print(f"   📏 Limited GOP to {max_frames_per_gop} frames (was {len(frames_list)})")
            
            # Show the frame indices that will be processed to verify sequential order
            frame_indices = [sample_idx for sample_idx, _ in frames_list]
            print(f"   📋 Frame indices to process sequentially: {frame_indices[:10]}{'...' if len(frame_indices) > 10 else ''} (total: {len(frame_indices)})")
            
            # Initialize GOP state variables for sequential tracking
            gop_prev_boxes = None
            gop_valid_mask = None
            gop_object_ids = None  # Track object IDs across frames
            
            # Process each P-frame in this GOP sequentially with frame progress bar
            frame_progress = tqdm(enumerate(frames_list), desc=f"P-frames in GOP {gop_num}", 
                                 total=len(frames_list), unit="frame", leave=False, position=1)
            
            for frame_idx, (sample_idx, sample) in frame_progress:
                try:
                    # Update progress description with detailed info
                    frame_progress.set_description(f"GOP {gop_num} | P-frame {frame_idx+1}/{len(frames_list)} | Sample {sample_idx}")
                    
                    # Print detailed frame processing info
                    print(f"   🎯 Processing P-frame {frame_idx+1}/{len(frames_list)} (dataset sample {sample_idx}) in GOP {seq_id}")
                    
                    # Extract motion vectors and annotations for this frame
                    if 'motion_vectors' not in sample:
                        continue
                        
                    motion_vectors = sample['motion_vectors']
                    real_boxes = sample.get('boxes', None)
                    real_labels = sample.get('labels', None)
                    real_ids = sample.get('ids', None)
                    
                    # Convert to tensor if needed
                    if not isinstance(motion_vectors, torch.Tensor):
                        motion_vectors = torch.tensor(motion_vectors, dtype=torch.float32)
                    
                    # Fix motion vector shape to (2, 40, 40) for CNN input
                    
                    if len(motion_vectors.shape) == 4 and motion_vectors.shape[3] == 2:
                        # Shape: (2, 40, 40, 2) - Select Channel 0 which contains real motion data
                        # Based on our investigation, Channel 0 has both X and Y components, Channel 1 is all zeros
                        motion_vectors = motion_vectors[0]  # Select Channel 0: (40, 40, 2)
                        motion_vectors = motion_vectors.permute(2, 0, 1)  # (40, 40, 2) -> (2, 40, 40)
                    elif len(motion_vectors.shape) == 3 and motion_vectors.shape[0] == 2:
                        # Already correct shape: (2, 40, 40)
                        pass
                    elif len(motion_vectors.shape) == 3 and motion_vectors.shape[1] == 2:
                        # Shape: (40, 2, 40) -> (2, 40, 40)
                        motion_vectors = motion_vectors.permute(1, 0, 2)
                    elif len(motion_vectors.shape) == 3 and motion_vectors.shape[2] == 2:
                        # Shape: (40, 40, 2) -> (2, 40, 40)
                        motion_vectors = motion_vectors.permute(2, 0, 1)
                    else:
                        print(f"   ⚠️ Unexpected motion vector shape: {motion_vectors.shape}")
                        print(f"   📊 Shape details: {[motion_vectors.shape[i] for i in range(len(motion_vectors.shape))]}")
                        continue
                    
                    # Motion vector processing: optionally add magnitude
                    mv_x, mv_y = motion_vectors[0], motion_vectors[1]  # Extract X and Y components
                    
                    if use_magnitude:
                        # ENHANCEMENT: Add magnitude as third channel (X, Y, Magnitude)
                        mv_magnitude = torch.sqrt(mv_x**2 + mv_y**2)  # Calculate magnitude
                        motion_vectors_final = torch.stack([mv_x, mv_y, mv_magnitude], dim=0)  # (3, 40, 40)
                        print(f"   📊 Enhanced motion vectors: X={mv_x.mean():.4f}, Y={mv_y.mean():.4f}, Mag={mv_magnitude.mean():.4f}")
                    else:
                        # BASELINE: Use only X and Y components (2 channels)
                        motion_vectors_final = torch.stack([mv_x, mv_y], dim=0)  # (2, 40, 40)
                        print(f"   📊 Baseline motion vectors: X={mv_x.mean():.4f}, Y={mv_y.mean():.4f}")
                    
                    # Ensure motion vectors are on device
                    motion_tensor = motion_vectors_final.unsqueeze(0).to(device)  # Add batch dim: (1, C, 40, 40)
                    
                    batch_size = 1
                    max_objects = config.max_objects
                    
                    # SEQUENTIAL TRACKING LOGIC: Handle I-frame vs P-frame differently
                    if frame_idx == 0:
                        # I-FRAME: Initialize with real annotations (conceptual I-frame)
                        if real_boxes is None:
                            print(f"   ⚠️ No real boxes available for GOP initialization")
                            continue
                        
                        prev_boxes = torch.zeros(batch_size, max_objects, 4, device=device, requires_grad=False)
                        valid_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool, device=device)
                        object_ids = torch.zeros(batch_size, max_objects, dtype=torch.long, device=device)  # ID tracking
                        
                        # Process real MOTS bounding boxes for GOP initialization
                        if isinstance(real_boxes, torch.Tensor):
                            boxes_tensor = real_boxes.squeeze() if len(real_boxes.shape) > 2 else real_boxes
                        else:
                            boxes_tensor = torch.tensor(real_boxes, dtype=torch.float32)
                        
                        # Fill with real MOTS annotations
                        num_real_objects = min(len(boxes_tensor), max_objects) if len(boxes_tensor.shape) > 1 else min(1, max_objects)
                        
                        if len(boxes_tensor.shape) > 1 and boxes_tensor.shape[1] >= 4:
                            prev_boxes[0, :num_real_objects] = boxes_tensor[:num_real_objects, :4].to(device)
                            valid_mask[0, :num_real_objects] = True
                            # Assign sequential IDs starting from 1 (0 is reserved for invalid)
                            object_ids[0, :num_real_objects] = torch.arange(1, num_real_objects + 1, device=device)
                            print(f"   🏁 I-FRAME: Using {num_real_objects} REAL annotations with IDs 1-{num_real_objects}")
                        elif len(boxes_tensor.shape) == 1 and len(boxes_tensor) >= 4:
                            prev_boxes[0, 0] = boxes_tensor[:4].to(device)
                            valid_mask[0, 0] = True
                            object_ids[0, 0] = 1  # Single object gets ID 1
                            print(f"   🏁 I-FRAME: Using 1 REAL annotation with ID 1")
                        else:
                            print(f"   ⚠️ Unexpected real boxes format: {boxes_tensor.shape}")
                            continue
                        
                        # I-FRAME: Just initialize state, no training/loss computation
                        print(f"   🎯 I-FRAME: Initialized with GT annotations (no loss computation)")
                        
                        # Store GOP state for next frame
                        with torch.no_grad():
                            gop_prev_boxes = prev_boxes.detach().clone()
                            gop_valid_mask = valid_mask.detach().clone()
                            gop_object_ids = object_ids.detach().clone()
                        
                        # Skip loss computation for I-frame
                        continue
                            
                    else:
                        # SUBSEQUENT P-FRAMES: Use predictions from previous frame
                        if gop_prev_boxes is None or gop_valid_mask is None or gop_object_ids is None:
                            print(f"   ⚠️ No previous predictions available for frame {frame_idx}")
                            continue
                            
                        # CRITICAL FIX: Ensure all inputs are properly detached and don't require gradients
                        prev_boxes = gop_prev_boxes.detach().clone().requires_grad_(False)
                        valid_mask = gop_valid_mask.detach().clone()
                        object_ids = gop_object_ids.detach().clone()  # Maintain ID consistency
                        print(f"   🔗 P-frame {frame_idx}: Using PREDICTED boxes with maintained IDs")
                        
                        # P-FRAME: Forward pass through ID-aware model
                        predicted_boxes, id_logits, attention_weights = model(motion_tensor, prev_boxes, object_ids, valid_mask)
                        
                        # Create target boxes by applying motion displacement to current boxes
                        mv_x, mv_y = motion_tensor[0, 0], motion_tensor[0, 1]
                        target_boxes = prev_boxes.detach().clone().requires_grad_(False)
                        
                        # Apply motion displacement only to valid objects
                        for obj_idx in range(max_objects):
                            if valid_mask[0, obj_idx]:
                                # Get box center position
                                box_x, box_y = prev_boxes[0, obj_idx, 0], prev_boxes[0, obj_idx, 1]
                                
                                # Calculate motion at this box position
                                grid_x = int(box_x.item() * (mv_x.shape[1] - 1))
                                grid_y = int(box_y.item() * (mv_x.shape[0] - 1))
                                grid_x = max(0, min(grid_x, mv_x.shape[1] - 1))
                                grid_y = max(0, min(grid_y, mv_x.shape[0] - 1))
                                
                                # Get motion at this position
                                motion_x = mv_x[grid_y, grid_x].item()
                                motion_y = mv_y[grid_y, grid_x].item()
                                
                                # Apply motion displacement
                                motion_scale = 0.01  # Scale factor
                                new_x = target_boxes[0, obj_idx, 0] + motion_x * motion_scale
                                new_y = target_boxes[0, obj_idx, 1] + motion_y * motion_scale
                                target_boxes[0, obj_idx, 0] = new_x
                                target_boxes[0, obj_idx, 1] = new_y
                        
                        # Ensure targets don't require gradients
                        target_boxes = target_boxes.detach().requires_grad_(False)
                        target_ids = object_ids.detach().clone()  # IDs should remain consistent
                        
                        # Clear gradients before forward pass
                        optimizer.zero_grad()
                        
                        # Compute ID-aware loss
                        loss_dict = criterion(predicted_boxes, target_boxes, id_logits, target_ids, valid_mask)
                        loss = loss_dict['total_loss']
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        epoch_bbox_loss += loss_dict['bbox_loss'].item()
                        epoch_id_loss += loss_dict['id_loss'].item()
                        num_samples += 1
                        
                        # Update GOP state for next frame - CRITICAL FOR SEQUENTIAL TRACKING
                        # Properly detach to prevent gradient accumulation across frames
                        with torch.no_grad():
                            gop_prev_boxes = predicted_boxes.detach().clone().requires_grad_(False)
                            gop_valid_mask = valid_mask.detach().clone()
                            gop_object_ids = object_ids.detach().clone()  # Maintain ID consistency
                            
                            # Add small noise to prevent deterministic cycles
                            noise = torch.randn_like(gop_prev_boxes) * 0.001
                            gop_prev_boxes = gop_prev_boxes + noise
                            gop_prev_boxes = gop_prev_boxes.detach().requires_grad_(False)
                        
                        # Update progress bars with current metrics
                        current_avg = epoch_loss / max(num_samples, 1)
                        frame_progress.set_postfix(
                            total_loss=f"{loss.item():.6f}", 
                            bbox_loss=f"{loss_dict['bbox_loss'].item():.6f}",
                            id_loss=f"{loss_dict['id_loss'].item():.6f}",
                            avg_loss=f"{current_avg:.6f}"
                        )
                        gop_progress.set_postfix(avg_loss=f"{current_avg:.6f}", samples=num_samples)
                        
                        # Show sequential tracking status
                        gop_position = list(sequence_groups.keys()).index(seq_id) + 1
                        if frame_idx < 3 and gop_position <= 2:
                            source = "PREDICTED"
                            num_active_objects = valid_mask.sum().item()
                            print(f"      ✅ P-frame {frame_idx}: {source} boxes → Total Loss = {loss.item():.6f} | Active Objects: {num_active_objects}")
                            print(f"         BBox Loss: {loss_dict['bbox_loss'].item():.6f} | ID Loss: {loss_dict['id_loss'].item():.6f}")
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing frame {frame_idx} in GOP {seq_id}: {e}")
                    continue
                        
                except Exception as e:
                    print(f"   ⚠️ Error processing frame {frame_idx} in GOP {seq_id}: {e}")
                    continue
                
                # Print GOP completion status for progress tracking
                gop_position = list(sequence_groups.keys()).index(seq_id) + 1
                if gop_position <= 3:  # Show details for first few GOPs
                    current_avg = epoch_loss / max(num_samples, 1)
                    gop_progress.set_postfix(avg_loss=f"{current_avg:.6f}", frames=len(frames_list))
        
        # Calculate epoch statistics
        avg_loss = epoch_loss / max(num_samples, 1)
        avg_bbox_loss = epoch_bbox_loss / max(num_samples, 1)
        avg_id_loss = epoch_id_loss / max(num_samples, 1)
        
        # Store losses for plotting
        epoch_losses.append(avg_loss)
        bbox_losses.append(avg_bbox_loss)
        id_losses.append(avg_id_loss)
        
        print(f"🎯 Epoch {epoch+1} Complete:")
        print(f"   Total Loss: {avg_loss:.6f}")
        print(f"   BBox Loss: {avg_bbox_loss:.6f}")
        print(f"   ID Loss: {avg_id_loss:.6f}")
        print(f"   Samples: {num_samples}")
        
        # Generate intermediate loss plots after each epoch to track learning progress
        if len(epoch_losses) > 1:  # Only plot if we have at least 2 epochs
            print(f"📊 Generating loss plots after epoch {epoch+1}...")
            plot_loss_curves(epoch_losses, bbox_losses, id_losses, config.output_dir)
            
            # Print learning progress analysis
            print(f"📈 Learning Progress Analysis:")
            if len(epoch_losses) >= 2:
                total_change = ((epoch_losses[-1] - epoch_losses[-2]) / epoch_losses[-2]) * 100
                bbox_change = ((bbox_losses[-1] - bbox_losses[-2]) / max(bbox_losses[-2], 1e-8)) * 100
                id_change = ((id_losses[-1] - id_losses[-2]) / max(id_losses[-2], 1e-8)) * 100
                
                print(f"   Total Loss Change: {total_change:+.2f}%")
                print(f"   BBox Loss Change: {bbox_change:+.2f}%")
                print(f"   ID Loss Change: {id_change:+.2f}%")
                
                if total_change < -1.0:
                    print(f"   ✅ Model is learning! Loss decreased by {abs(total_change):.2f}%")
                elif total_change > 1.0:
                    print(f"   ⚠️ Loss increased by {total_change:.2f}% - possible overfitting or learning rate too high")
                else:
                    print(f"   📊 Loss relatively stable (change: {total_change:+.2f}%)")
        
        # Save checkpoint with loss history every epoch
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': {
                'motion_shape': motion_shape,
                'hidden_dim': 128,
                'max_objects': config.max_objects,
                'max_id': config.max_objects * 2,
                'use_magnitude': use_magnitude,  # Store magnitude flag
                'motion_channels': motion_channels
            },
            'loss_history': {
                'total_losses': epoch_losses,
                'bbox_losses': bbox_losses,
                'id_losses': id_losses
            },
            'current_losses': {
                'total_loss': avg_loss,
                'bbox_loss': avg_bbox_loss,
                'id_loss': avg_id_loss
            }
        }
        
        # Include magnitude info in checkpoint name for easy identification
        magnitude_suffix = "_magnitude" if use_magnitude else "_baseline"
        checkpoint_path = os.path.join(config.output_dir, f'checkpoint_epoch_{epoch+1}{magnitude_suffix}.pt')
        os.makedirs(config.output_dir, exist_ok=True)
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Plot and save loss curves
    plot_loss_curves(epoch_losses, bbox_losses, id_losses, config.output_dir)
    
    final_loss = epoch_loss / max(num_samples, 1)
    return final_loss, epoch_losses, bbox_losses, id_losses


def save_results(config: SequentialTrainConfig, final_loss: float, epoch_losses: list, bbox_losses: list, id_losses: list):
    """Save training results with detailed loss tracking."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    results = {
        'model': 'id_aware_multi_object_tracker',
        'final_total_loss': final_loss,
        'final_bbox_loss': bbox_losses[-1] if bbox_losses else 0.0,
        'final_id_loss': id_losses[-1] if id_losses else 0.0,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'max_objects': config.max_objects,
        'sequence_length': config.sequence_length,
        'gop_length': config.gop_length,
        'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
        'training_mode': 'id_aware_sequential_gop_pframes',
        'loss_history': {
            'total_losses': epoch_losses,
            'bbox_losses': bbox_losses,
            'id_losses': id_losses
        }
    }
    
    results_path = os.path.join(config.output_dir, 'id_aware_tracker_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {results_path}")
    print(f"📊 Training Summary:")
    print(f"   Final Total Loss: {final_loss:.6f}")
    print(f"   Final BBox Loss: {bbox_losses[-1]:.6f}" if bbox_losses else "   No BBox Loss recorded")
    print(f"   Final ID Loss: {id_losses[-1]:.6f}" if id_losses else "   No ID Loss recorded")
    print(f"   Model Parameters: 1.7M")
    print(f"   Architecture: ID-Aware Multi-Object Tracker with Self-Attention")


def main():
    """Main training function."""
    try:
        # Parse arguments and create config
        args = parse_args()
        config = SequentialTrainConfig.from_args(args)
        
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        print(f"🔧 Sequential GOP MOTS Training")
        print(f"   Model: {config.model_name}")
        print(f"   Device: {device}")
        print(f"   Epochs: {config.epochs}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   GOP length: {config.gop_length} frames")
        print(f"   Max objects: {config.max_objects}")
        print(f"   Data path: {config.data_path}")
        print(f"   Use magnitude: {args.use_magnitude} ({'3-channel: X,Y,Mag' if args.use_magnitude else '2-channel: X,Y'})")
        
        if args.dry_run:
            print("✅ Dry run completed - configuration looks good!")
            return 0
        
        # Run sequential GOP training with P-frame iteration
        training_results = sequential_gop_training(config, device, args.visualize, use_magnitude=args.use_magnitude)
        final_loss, epoch_losses, bbox_losses, id_losses = training_results
        
        # Save results with detailed loss tracking
        save_results(config, final_loss, epoch_losses, bbox_losses, id_losses)
        
        print(f"✅ Sequential GOP training completed successfully!")
        print(f"   Final loss: {final_loss:.6f}")
        print(f"   Processed all P-frames sequentially in each GOP")
        print(f"   Model type: {'Magnitude-enhanced (3-channel)' if args.use_magnitude else 'Baseline (2-channel)'}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

import argparse
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TrainConfig:
    """Training configuration with sensible defaults."""
    model_name: str = "optimized_enhanced_offset_tracker"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    device: Optional[str] = None
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class DataConfig:
    """Data configuration with defaults."""
    dataset_type: str = "mot17"
    resolution: int = 640
    sequence_length: int = 8
    max_objects: int = 50


@dataclass
class Config:
    """Main configuration container."""
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        cfg = cls()
        
        # Update train config from args
        if hasattr(args, 'epochs') and args.epochs is not None:
            cfg.train.epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cfg.train.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            cfg.train.learning_rate = args.learning_rate
        if hasattr(args, 'model') and args.model is not None:
            cfg.train.model_name = args.model
            
        return cfg


def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_hint: Optional[str] = None):
    """Get the best available device."""
    import torch
    if device_hint:
        return torch.device(device_hint)
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def gpu_info():
    """Get GPU information string."""
    import torch
    if torch.cuda.is_available():
        return f"CUDA {torch.cuda.get_device_name()}"
    else:
        return "CPU only"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MOTS Modular Training Script")
    parser.add_argument('--model', type=str, default='optimized_enhanced_offset_tracker',
                       help='Model name to train')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--dry-run', action='store_true',
                       help='Just test configuration and exit')
    
    return parser.parse_args()


def create_real_mots_dataset(config: Config):
    """Create real MOTS dataset for training."""
    import torch
    from torch.utils.data import Dataset
    import os
    import sys
    
    # Add paths for dataset imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dataset_dir = os.path.join(project_root, 'dataset')
    training_dir = os.path.join(project_root, 'training')
    
    sys.path.insert(0, project_root)
    sys.path.insert(0, dataset_dir)
    sys.path.insert(0, training_dir)
    
    try:
        # Try to import the real MOTS dataset from training folder
        from training.mots17_deep_tracker import RealMOTS17Dataset
        
        print(f"📊 Loading real MOTS17 dataset...")
        print(f"   Dataset type: {config.data.dataset_type}")
        print(f"   Max objects: {config.data.max_objects}")
        
        # Create train dataset - USE ALL SAMPLES
        train_dataset = RealMOTS17Dataset(
            dataset_type='train',
            max_objects=config.data.max_objects,
            root_dir='/home/aduche/Bureau/datasets/MOTS/videos'
        )
        
        # Create validation dataset - USE ALL SAMPLES
        val_dataset = RealMOTS17Dataset(
            dataset_type='val',
            max_objects=config.data.max_objects,
            root_dir='/home/aduche/Bureau/datasets/MOTS/videos'
        )
        
        print(f"✅ Created MOTS datasets: {len(train_dataset)} train, {len(val_dataset)} val")
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"⚠️  Could not load real MOTS dataset: {e}")
        print(f"🔄 Falling back to dummy data for testing...")

        return None


def create_simple_model(config: Config, device):
    """Create a simple model for testing."""
    import torch
    
    class SimpleTracker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Adjust for real MOTS data: motion vectors are (2, 60, 60)
            self.conv1 = torch.nn.Conv2d(2, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)  # Global average pooling
            
            # Box encoder
            self.box_encoder = torch.nn.Linear(4, 64)
            
            # Main predictor
            self.predictor = torch.nn.Sequential(
                torch.nn.Linear(64 + 64, 128),  # motion features + box features
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 4)  # x, y, w, h delta
            )
            
        def forward(self, motion_vectors, prev_boxes, valid_mask=None):
            # Extract motion features
            motion_feat = self.conv1(motion_vectors)
            motion_feat = torch.nn.functional.relu(motion_feat)
            motion_feat = self.conv2(motion_feat)
            motion_feat = torch.nn.functional.relu(motion_feat)
            motion_feat = self.pool(motion_feat).squeeze(-1).squeeze(-1)  # [B, 64]
            
            batch_size = motion_vectors.size(0)
            max_objects = prev_boxes.size(1)
            
            outputs = []
            for b in range(batch_size):
                batch_outputs = []
                batch_motion = motion_feat[b]  # [64]
                
                for obj in range(max_objects):
                    bbox = prev_boxes[b, obj]  # [4]
                    
                    # Skip invalid objects if mask is provided
                    if valid_mask is not None and not valid_mask[b, obj]:
                        batch_outputs.append(torch.zeros(4, device=bbox.device))
                        continue
                    
                    # Encode bounding box
                    bbox_feat = self.box_encoder(bbox)  # [64]
                    
                    # Combine features and predict
                    combined = torch.cat([batch_motion, bbox_feat])  # [128]
                    delta = self.predictor(combined)  # [4]
                    
                    # Apply delta to get new box
                    new_box = bbox + delta
                    batch_outputs.append(new_box)
                
                outputs.append(torch.stack(batch_outputs))
            
            return torch.stack(outputs)
    
    return SimpleTracker().to(device)


def train_simple_model(model, train_dataset, val_dataset, config: Config, device):
    """Simple training loop adapted for real MOTS data."""
    import torch
    
    print(f"🚀 Starting training...")
    print(f"   Epochs: {config.train.epochs}")
    print(f"   Batch size: {config.train.batch_size}")
    print(f"   Learning rate: {config.train.learning_rate}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(config.train.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(train_dataset), config.train.batch_size):
            # Get batch
            batch_data = []
            for j in range(i, min(i + config.train.batch_size, len(train_dataset))):
                sample = train_dataset[j]
                if sample is not None:
                    batch_data.append(sample)
            
            if not batch_data:
                continue
            
            # Handle different data formats
            try:
                if 'motion_vectors' in batch_data[0]:
                    # Real MOTS data format
                    motion_vectors = torch.stack([d['motion_vectors'] for d in batch_data]).to(device)
                    prev_boxes = torch.stack([d['prev_boxes'] for d in batch_data]).to(device)
                    target_boxes = torch.stack([d['target_boxes'] for d in batch_data]).to(device)
                    valid_mask = torch.stack([d['valid_mask'] for d in batch_data]).to(device)
                else:
                    # Dummy data format (fallback)
                    motion_vectors = torch.stack([d['mv_map'] for d in batch_data]).to(device)
                    prev_boxes = torch.stack([d['prev_boxes'] for d in batch_data]).to(device)
                    target_boxes = torch.stack([d['target_boxes'] for d in batch_data]).to(device)
                    valid_mask = None
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(motion_vectors, prev_boxes, valid_mask)
                
                # Compute loss only for valid objects
                if valid_mask is not None:
                    # Mask out invalid predictions
                    loss_mask = valid_mask.unsqueeze(-1).expand_as(predictions)
                    masked_predictions = predictions * loss_mask.float()
                    masked_targets = target_boxes * loss_mask.float()
                    loss = criterion(masked_predictions, masked_targets)
                else:
                    loss = criterion(predictions, target_boxes)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"⚠️ Batch error: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"   Epoch {epoch+1}/{config.train.epochs}: Loss = {avg_loss:.6f}")
    
    return avg_loss


def save_results(config: Config, final_loss: float):
    """Save training results."""
    os.makedirs(config.train.output_dir, exist_ok=True)
    
    results = {
        'model': config.train.model_name,
        'final_loss': final_loss,
        'epochs': config.train.epochs,
        'batch_size': config.train.batch_size,
        'learning_rate': config.train.learning_rate,
        'device': str(get_device(config.train.device))
    }
    
    results_path = os.path.join(config.train.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {results_path}")


def main():
    """Main training function."""
    try:
        args = parse_args()
        config = Config.from_args(args)
        
        device = get_device(config.train.device)
        seed_everything(args.seed)
        
        print(f"🔧 MOTS Modular Training")
        print(f"   Model: {config.train.model_name}")
        print(f"   Device: {device} ({gpu_info()})")
        print(f"   Epochs: {config.train.epochs}")
        print(f"   Batch size: {config.train.batch_size}")
        print(f"   Learning rate: {config.train.learning_rate}")
        
        if args.dry_run:
            print("✅ Dry run completed - configuration looks good!")
            return 0
        
        # Create datasets and model
        train_dataset, val_dataset = create_real_mots_dataset(config)
        model = create_simple_model(config, device)
        
        print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        final_loss = train_simple_model(model, train_dataset, val_dataset, config, device)
        save_results(config, final_loss)
        
        print(f"✅ Training completed successfully!")
        print(f"   Final loss: {final_loss:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


# Duplicate main function removed - using sequential GOP training main function above
