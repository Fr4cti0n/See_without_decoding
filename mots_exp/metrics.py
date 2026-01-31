import torch
from typing import Tuple

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between all pairs of boxes.
    boxes format: [cx, cy, w, h] (center format)
    Returns: IoU matrix of shape [N, M] where N=len(boxes1), M=len(boxes2)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return torch.zeros(len(boxes1), len(boxes2))
    
    # Convert to corner format
    b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2
    
    b2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
    b2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
    b2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
    b2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2
    
    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1.t())
    inter_y1 = torch.max(b1_y1, b2_y1.t())
    inter_x2 = torch.min(b1_x2, b2_x2.t())
    inter_y2 = torch.min(b1_y2, b2_y2.t())
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area.t() - inter_area + 1e-8
    
    return inter_area / union_area

def iou_diagonal(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Legacy function - assumes same number of boxes"""
    if len(pred_boxes) == 0:
        return torch.empty(0, device=target_boxes.device)
    px1 = pred_boxes[:,0]-pred_boxes[:,2]/2; py1 = pred_boxes[:,1]-pred_boxes[:,3]/2
    px2 = pred_boxes[:,0]+pred_boxes[:,2]/2; py2 = pred_boxes[:,1]+pred_boxes[:,3]/2
    tx1 = target_boxes[:,0]-target_boxes[:,2]/2; ty1 = target_boxes[:,1]-target_boxes[:,3]/2
    tx2 = target_boxes[:,0]+target_boxes[:,2]/2; ty2 = target_boxes[:,1]+target_boxes[:,3]/2
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    inter = torch.clamp(ix2-ix1, min=0)*torch.clamp(iy2-iy1, min=0)
    p_area = (px2-px1)*(py2-py1); t_area = (tx2-tx1)*(ty2-ty1)
    union = p_area + t_area - inter + 1e-8
    return inter / union

def simple_map(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, thresh: float = 0.5) -> float:
    """
    Compute mAP using greedy matching between predictions and targets.
    Handles different numbers of pred/target boxes.
    """
    if len(pred_boxes) == 0 or len(target_boxes) == 0:
        return 0.0
    
    # Compute IoU matrix [N_pred, N_target]
    iou_matrix = box_iou(pred_boxes, target_boxes)
    
    # Greedy matching: for each target, find best matching prediction
    matched_preds = set()
    tp = 0
    
    for tgt_idx in range(len(target_boxes)):
        # Get IoUs for this target with all predictions
        ious = iou_matrix[:, tgt_idx]
        
        # Find best prediction that hasn't been matched yet
        best_iou = 0.0
        best_pred_idx = -1
        for pred_idx in range(len(pred_boxes)):
            if pred_idx not in matched_preds and ious[pred_idx] > best_iou:
                best_iou = ious[pred_idx]
                best_pred_idx = pred_idx
        
        # If best match exceeds threshold, count as true positive
        if best_iou >= thresh:
            tp += 1
            matched_preds.add(best_pred_idx)
    
    fp = len(pred_boxes) - tp
    fn = len(target_boxes) - tp
    
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
