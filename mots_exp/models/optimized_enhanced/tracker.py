import torch
import torch.nn as nn
from .motion_encoder import LightweightMotionEncoder
from .object_interaction import EfficientObjectInteraction
from .offset_predictor import OptimizedOffsetPredictor

class OptimizedEnhancedOffsetTracker(nn.Module):
    def __init__(self, mv_channels: int = 2, base_channels: int = 32, object_feature_dim: int = 128, roi_size: int = 8, hidden_dim: int = 256, n_attention_heads: int = 4, image_size=(640, 640)):
        super().__init__()
        self.image_size = image_size
        self.object_feature_dim = object_feature_dim
        self.mv_encoder = LightweightMotionEncoder(mv_channels, base_channels)
        self.object_interaction = EfficientObjectInteraction(object_feature_dim, n_attention_heads, hidden_dim)
        self.offset_predictor = OptimizedOffsetPredictor(64, object_feature_dim, roi_size, hidden_dim)

    @staticmethod
    def _apply_offsets(boxes: torch.Tensor, offsets: torch.Tensor, confidences: torch.Tensor):
        if len(boxes) == 0:
            return boxes, torch.empty(0, device=boxes.device)
        weighted = offsets * confidences
        new_cx = boxes[:, 0] + weighted[:, 0]
        new_cy = boxes[:, 1] + weighted[:, 1]
        new_w = boxes[:, 2] + weighted[:, 2]
        new_h = boxes[:, 3] + weighted[:, 3]
        new_w = torch.clamp(new_w, 0.01, 0.9)
        new_h = torch.clamp(new_h, 0.01, 0.9)
        new_cx = torch.clamp(new_cx, new_w / 2, 1 - new_w / 2)
        new_cy = torch.clamp(new_cy, new_h / 2, 1 - new_h / 2)
        new_boxes = torch.stack([new_cx, new_cy, new_w, new_h], dim=1)
        scores = confidences.mean(1)
        return new_boxes, scores

    def forward(self, mv_map: torch.Tensor, prev_boxes: torch.Tensor):
        if len(prev_boxes) == 0:
            return {"predicted_offsets": torch.empty(0,4), "predicted_confidences": torch.empty(0,4), "updated_boxes": torch.empty(0,4), "confidence_scores": torch.empty(0), "mv_features": None}
        mv_features = self.mv_encoder(mv_map)
        init_feats = self.offset_predictor.extract_features(mv_features, mv_map, prev_boxes)
        if len(prev_boxes) > 1:
            h, w = self.image_size
            positions = torch.stack([prev_boxes[:,0]*w, prev_boxes[:,1]*h], dim=1)
            enhanced = self.object_interaction(init_feats, positions)
        else:
            enhanced = init_feats
        offsets, confidences = self.offset_predictor(mv_features, mv_map, prev_boxes, enhanced)
        updated_boxes, scores = self._apply_offsets(prev_boxes, offsets, confidences)
        return {"predicted_offsets": offsets, "predicted_confidences": confidences, "updated_boxes": updated_boxes, "confidence_scores": scores, "mv_features": mv_features, "enhanced_features": enhanced}
