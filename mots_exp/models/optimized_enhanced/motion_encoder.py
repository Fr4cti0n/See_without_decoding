import torch
import torch.nn as nn
from .channel_attention import LightweightChannelAttention

class LightweightMotionEncoder(nn.Module):
    """Lightweight multi-scale motion encoder"""
    def __init__(self, input_channels: int = 2, base_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.attention = LightweightChannelAttention(base_channels * 2)
        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, mv_map: torch.Tensor) -> torch.Tensor:
        x = self.conv1(mv_map)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        fused = torch.cat([b1, b2], dim=1)
        fused = self.fusion(fused)
        attended = self.attention(fused)
        return self.output_proj(attended)
