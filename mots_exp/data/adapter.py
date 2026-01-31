"""MOTSDataAdapter extracted from legacy training script.
Converts raw MOTS tracking dataset samples into (mv_map, prev_boxes, target_boxes, gop_index).
Includes simple temporal memory blending.
"""
from __future__ import annotations
import random
import torch
from typing import Any, Dict, Optional

class MOTSDataAdapter:
    def __init__(self, resolution: int = 640, use_temporal_memory: bool = True, memory_ratio: float = 0.7):
        self.resolution = resolution
        self.use_temporal_memory = use_temporal_memory
        self.memory_ratio = memory_ratio
        self.sequence_memory: Dict[str, Dict[str, Any]] = {}

    def reset_sequence_memory(self, sequence_id: Optional[str] = None):
        if sequence_id is None:
            self.sequence_memory.clear()
        else:
            self.sequence_memory.pop(sequence_id, None)

    def update_sequence_memory(self, video_name: str, gop_index: int, predictions: Dict[int, torch.Tensor], track_ids):
        stored_track_ids = torch.tensor(list(predictions.keys()), dtype=torch.long)
        self.sequence_memory[video_name] = {
            'predictions': {k: v.clone() for k, v in predictions.items()},
            'track_ids': stored_track_ids,
            'last_gop_index': gop_index
        }

    def get_previous_predictions(self, video_name: str, current_track_ids: torch.Tensor):
        if video_name not in self.sequence_memory:
            return None
        seq = self.sequence_memory[video_name]
        prev_preds = seq['predictions']
        out = {}
        for tid in current_track_ids:
            tid_i = int(tid.item())
            if tid_i in prev_preds:
                out[tid_i] = prev_preds[tid_i]
        return out if out else None

    def convert_sample(self, sample: Dict[str, Any]):
        if sample is None:
            return None
        try:
            iframe_data = sample['iframe_data']
            pframe_data = sample['pframe_data']
            motion_vectors = sample['motion_vectors']
            sequence_info = sample['sequence_info']
        except KeyError:
            return None

        if not iframe_data or not pframe_data or not motion_vectors:
            return None

        video_name = sequence_info.get('video_name', 'unknown')
        gop_index = sequence_info.get('gop_index', 0)

        iframe_boxes = iframe_data['boxes']  # Tensor [N,4]
        iframe_ids = iframe_data['track_ids']  # Tensor [N]

        use_memory = self.use_temporal_memory and random.random() < self.memory_ratio
        if use_memory and len(iframe_ids) > 0:
            prev_preds = self.get_previous_predictions(video_name, iframe_ids)
            if prev_preds:
                blended = []
                alpha, beta = 0.7, 0.3
                for i, tid in enumerate(iframe_ids):
                    tid_i = int(tid.item())
                    base_box = iframe_boxes[i]
                    if tid_i in prev_preds:
                        blended.append(alpha * base_box + beta * prev_preds[tid_i])
                    else:
                        blended.append(base_box)
                iframe_boxes = torch.stack(blended)

        p_last = pframe_data[-1]
        if 'boxes' in p_last:
            pframe_boxes = p_last['boxes']
            pframe_ids = p_last['track_ids']
        else:
            # Fallback cannot reconstruct -> skip
            return None

        iframe_map = {int(t.item()): iframe_boxes[i] for i, t in enumerate(iframe_ids)}
        pframe_map = {int(t.item()): pframe_boxes[i] for i, t in enumerate(pframe_ids)}
        common = sorted(set(iframe_map.keys()) & set(pframe_map.keys()))
        if not common:
            return None
        prev_list = [iframe_map[c] for c in common]
        tgt_list = [pframe_map[c] for c in common]
        prev_boxes = torch.stack(prev_list)
        target_boxes = torch.stack(tgt_list)

        current_predictions = {cid: target_boxes[i] for i, cid in enumerate(common)}
        self.update_sequence_memory(video_name, gop_index, current_predictions, common)

        mv = motion_vectors[-1]
        if mv.shape == (2, 40, 40, 2):
            mv = mv.mean(dim=0).permute(2, 0, 1)
        elif mv.shape == (2, 40, 40):
            pass
        else:
            mv = torch.zeros((2, 40, 40), dtype=torch.float32)

        return {
            'mv_map': mv,
            'prev_boxes': prev_boxes,
            'target_boxes': target_boxes,
            'gop_index': gop_index
        }
