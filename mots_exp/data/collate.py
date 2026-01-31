import torch
from typing import List, Dict, Any

def collate_fn(batch: List[Dict[str, Any]]):
    valid = [b for b in batch if b is not None]
    if not valid:
        return None
    mv_maps = torch.stack([b['mv_map'] for b in valid])
    prev_boxes = [b['prev_boxes'] for b in valid]
    target_boxes = [b['target_boxes'] for b in valid]
    gop_indices = [b['gop_index'] for b in valid]
    return {
        'mv_map': mv_maps,
        'prev_boxes': prev_boxes,
        'target_boxes': target_boxes,
        'gop_indices': gop_indices,
    }
