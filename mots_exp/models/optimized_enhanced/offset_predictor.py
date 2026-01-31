import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

class OptimizedOffsetPredictor(nn.Module):
    def __init__(self, mv_feature_dim: int = 64, object_feature_dim: int = 128, roi_size: int = 8, hidden_dim: int = 256):
        super().__init__()
        self.roi_size = roi_size
        self.object_feature_dim = object_feature_dim
        self.roi_conv = nn.Sequential(
            nn.Conv2d(mv_feature_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(32 * 9, 64)
        )
        self.mv_stats_extractor = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 48)
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 32)
        )
        feature_input_dim = 64 + 48 + 32
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, object_feature_dim)
        )
        self.offset_head = nn.Sequential(
            nn.Linear(object_feature_dim, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(96, 4)
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(object_feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
        nn.init.normal_(self.offset_head[-1].weight, mean=0.0, std=0.2)
        nn.init.zeros_((self.offset_head[-1].bias))
        nn.init.normal_(self.confidence_head[-2].weight, mean=0.0, std=0.1)
        nn.init.constant_(self.confidence_head[-2].bias, 2.0)

    def extract_features(self, mv_features: torch.Tensor, mv_map: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if len(boxes) == 0:
            return torch.empty(0, self.object_feature_dim, device=boxes.device)
        h, w = mv_map.shape[-2:]
        cx = boxes[:, 0] * w
        cy = boxes[:, 1] * h
        bw = boxes[:, 2] * w
        bh = boxes[:, 3] * h
        x1 = cx - bw / 2; y1 = cy - bh / 2; x2 = cx + bw / 2; y2 = cy + bh / 2
        batch_idx = torch.zeros(len(boxes), 1, device=boxes.device)
        roi_boxes = torch.cat([batch_idx, x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)], dim=1)
        roi_features = roi_align(mv_features, roi_boxes, output_size=(self.roi_size, self.roi_size), spatial_scale=1.0, aligned=True)
        mv_roi = roi_align(mv_map, roi_boxes, output_size=(self.roi_size, self.roi_size), spatial_scale=1.0, aligned=True)
        roi_processed = self.roi_conv(roi_features)
        mv_u = mv_roi[:, 0].flatten(1); mv_v = mv_roi[:, 1].flatten(1)
        mean_u = mv_u.mean(1); mean_v = mv_v.mean(1)
        std_u = mv_u.std(1); std_v = mv_v.std(1)
        magnitude = torch.sqrt(mean_u**2 + mean_v**2)
        direction = torch.atan2(mean_v, mean_u)
        mv_stats = torch.stack([mean_u, mean_v, std_u, std_v, magnitude, direction], dim=1)
        mv_processed = self.mv_stats_extractor(mv_stats)
        aspect_ratio = bw / (bh + 1e-8)
        area = bw * bh / (w * h)
        geometry = torch.stack([bw / w, bh / h, aspect_ratio, area], dim=1)
        geo_processed = self.geometry_encoder(geometry)
        combined = torch.cat([roi_processed, mv_processed, geo_processed], dim=1)
        return self.feature_fusion(combined)

    def forward(self, mv_features: torch.Tensor, mv_map: torch.Tensor, boxes: torch.Tensor, enhanced_features: torch.Tensor | None = None):
        if len(boxes) == 0:
            device = mv_features.device
            return (torch.empty(0, 4, device=device), torch.empty(0, 4, device=device))
        object_features = enhanced_features if enhanced_features is not None else self.extract_features(mv_features, mv_map, boxes)
        offsets = self.offset_head(object_features)
        confidences = self.confidence_head(object_features)
        return offsets, confidences
