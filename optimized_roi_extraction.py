"""
Optimized batched ROI extraction to replace the for-loop version.

This version processes ALL boxes at once using vectorized operations.
Expected speedup: 2-4x
"""

import torch
import torch.nn.functional as F


def extract_roi_motion_features_batched(mv_field, boxes, grid_size, image_size):
    """
    Extract motion features for ALL boxes at once (batched).
    
    Args:
        mv_field: [2, H, W] motion vector field
        boxes: [N, 4] normalized boxes [cx, cy, w, h] in [0,1]
        grid_size: Motion vector grid size (H or W)
        image_size: Image size in pixels
        
    Returns:
        roi_features: [N, 6] - [mean_vx, mean_vy, std_vx, std_vy, num_mvs, sparsity_ratio]
    """
    N = len(boxes)
    if N == 0:
        return torch.zeros(0, 6, device=boxes.device, dtype=boxes.dtype)
    
    H, W = mv_field.shape[1], mv_field.shape[2]
    device = boxes.device
    
    # Method 1: Grid-based approach (fastest for many boxes)
    # Create a grid of coordinates
    y_coords = torch.arange(H, device=device, dtype=boxes.dtype)
    x_coords = torch.arange(W, device=device, dtype=boxes.dtype)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W]
    
    # Normalize grid coordinates to [0, 1]
    y_grid_norm = (y_grid + 0.5) / H  # [H, W]
    x_grid_norm = (x_grid + 0.5) / W  # [H, W]
    
    # Expand for broadcasting: [1, H, W] and [N, 1, 1]
    y_grid_norm = y_grid_norm.unsqueeze(0)  # [1, H, W]
    x_grid_norm = x_grid_norm.unsqueeze(0)  # [1, H, W]
    
    # Extract box parameters
    cx = boxes[:, 0].unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
    cy = boxes[:, 1].unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
    w = boxes[:, 2].unsqueeze(-1).unsqueeze(-1)   # [N, 1, 1]
    h = boxes[:, 3].unsqueeze(-1).unsqueeze(-1)   # [N, 1, 1]
    
    # Compute which grid cells are inside each box
    # For each of N boxes, create a [H, W] mask
    inside_x = (x_grid_norm >= (cx - w/2)) & (x_grid_norm < (cx + w/2))  # [N, H, W]
    inside_y = (y_grid_norm >= (cy - h/2)) & (y_grid_norm < (cy + h/2))  # [N, H, W]
    inside_mask = inside_x & inside_y  # [N, H, W]
    
    # Expand motion vectors for batching
    mv_x = mv_field[0].unsqueeze(0)  # [1, H, W]
    mv_y = mv_field[1].unsqueeze(0)  # [1, H, W]
    
    # Apply mask to get MVs inside each box
    # Where mask is False, set to 0
    mv_x_masked = torch.where(inside_mask, mv_x, torch.tensor(0.0, device=device))  # [N, H, W]
    mv_y_masked = torch.where(inside_mask, mv_y, torch.tensor(0.0, device=device))  # [N, H, W]
    
    # Count cells inside each box
    num_cells = inside_mask.sum(dim=(1, 2)).float()  # [N]
    num_cells = torch.clamp(num_cells, min=1.0)  # Avoid division by zero
    
    # Compute mean (sum of masked values / count)
    mean_vx = mv_x_masked.sum(dim=(1, 2)) / num_cells  # [N]
    mean_vy = mv_y_masked.sum(dim=(1, 2)) / num_cells  # [N]
    
    # Compute std
    # std = sqrt(E[(x - mean)^2])
    diff_x = mv_x_masked - mean_vx.unsqueeze(-1).unsqueeze(-1)  # [N, H, W]
    diff_y = mv_y_masked - mean_vy.unsqueeze(-1).unsqueeze(-1)  # [N, H, W]
    
    # Only consider cells inside the box for std calculation
    diff_x_masked = torch.where(inside_mask, diff_x, torch.tensor(0.0, device=device))
    diff_y_masked = torch.where(inside_mask, diff_y, torch.tensor(0.0, device=device))
    
    var_x = (diff_x_masked ** 2).sum(dim=(1, 2)) / num_cells  # [N]
    var_y = (diff_y_masked ** 2).sum(dim=(1, 2)) / num_cells  # [N]
    
    std_vx = torch.sqrt(var_x + 1e-8)  # [N]
    std_vy = torch.sqrt(var_y + 1e-8)  # [N]
    
    # Compute sparsity (non-zero MVs)
    magnitude = torch.sqrt(mv_x**2 + mv_y**2)  # [1, H, W]
    non_zero_mask = (magnitude > 0.01) & inside_mask  # [N, H, W]
    num_non_zero = non_zero_mask.sum(dim=(1, 2)).float()  # [N]
    sparsity_ratio = num_non_zero / num_cells  # [N]
    
    # Stack features
    roi_features = torch.stack([
        mean_vx, mean_vy, std_vx, std_vy, num_non_zero, sparsity_ratio
    ], dim=1)  # [N, 6]
    
    return roi_features


# ============================================================================
# Alternative Method 2: ROI Align (if you want to use torchvision)
# ============================================================================

def extract_roi_motion_features_roialign(mv_field, boxes, grid_size, image_size, roi_size=(7, 7)):
    """
    Extract motion features using ROI Align (vectorized, very fast).
    
    This is the FASTEST method but requires torchvision.
    """
    from torchvision.ops import roi_align
    
    N = len(boxes)
    if N == 0:
        return torch.zeros(0, 6, device=boxes.device, dtype=boxes.dtype)
    
    # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w/2) * grid_size
    y1 = (cy - h/2) * grid_size
    x2 = (cx + w/2) * grid_size
    y2 = (cy + h/2) * grid_size
    
    # ROI format: [batch_idx, x1, y1, x2, y2]
    batch_idx = torch.zeros(N, 1, device=boxes.device)
    rois = torch.cat([batch_idx, x1.unsqueeze(1), y1.unsqueeze(1), 
                      x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)  # [N, 5]
    
    # Extract ROIs (all at once!)
    mv_field_batch = mv_field.unsqueeze(0)  # [1, 2, H, W]
    roi_mvs = roi_align(
        mv_field_batch,     # [1, 2, H, W]
        rois,               # [N, 5]
        output_size=roi_size,  # (7, 7)
        spatial_scale=1.0,
        sampling_ratio=2
    )  # [N, 2, 7, 7]
    
    # Compute statistics on extracted ROIs
    # Shape: [N, 2, 7, 7]
    mean_vx = roi_mvs[:, 0].mean(dim=(1, 2))  # [N]
    mean_vy = roi_mvs[:, 1].mean(dim=(1, 2))  # [N]
    std_vx = roi_mvs[:, 0].std(dim=(1, 2))    # [N]
    std_vy = roi_mvs[:, 1].std(dim=(1, 2))    # [N]
    
    # Compute sparsity
    magnitude = torch.sqrt(roi_mvs[:, 0]**2 + roi_mvs[:, 1]**2)  # [N, 7, 7]
    non_zero_mask = magnitude > 0.01
    num_non_zero = non_zero_mask.sum(dim=(1, 2)).float()  # [N]
    total_cells = torch.tensor(roi_size[0] * roi_size[1], device=boxes.device).float()
    sparsity_ratio = num_non_zero / total_cells  # [N]
    
    # Stack features
    roi_features = torch.stack([
        mean_vx, mean_vy, std_vx, std_vy, num_non_zero, sparsity_ratio
    ], dim=1)  # [N, 6]
    
    return roi_features


# ============================================================================
# Benchmark comparison
# ============================================================================

if __name__ == '__main__':
    import time
    
    device = 'cuda'
    N = 30  # number of objects
    H, W = 60, 60  # grid size
    
    # Create test data
    mv_field = torch.randn(2, H, W, device=device)
    boxes = torch.rand(N, 4, device=device)
    
    # Warmup
    for _ in range(10):
        _ = extract_roi_motion_features_batched(mv_field, boxes, H, 960)
    
    # Benchmark batched version
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        features = extract_roi_motion_features_batched(mv_field, boxes, H, 960)
    torch.cuda.synchronize()
    batched_time = time.time() - t0
    
    print(f"Batched version: {batched_time*1000:.2f} ms for 1000 iterations")
    print(f"Per call: {batched_time:.4f} ms")
    print(f"Features shape: {features.shape}")
    
    # Try ROI Align version if available
    try:
        # Warmup
        for _ in range(10):
            _ = extract_roi_motion_features_roialign(mv_field, boxes, H, 960)
        
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(1000):
            features = extract_roi_motion_features_roialign(mv_field, boxes, H, 960)
        torch.cuda.synchronize()
        roialign_time = time.time() - t0
        
        print(f"\nROI Align version: {roialign_time*1000:.2f} ms for 1000 iterations")
        print(f"Per call: {roialign_time:.4f} ms")
        print(f"Speedup vs batched: {batched_time/roialign_time:.2f}x")
    except ImportError:
        print("\nROI Align not available (need torchvision)")
