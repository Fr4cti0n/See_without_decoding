import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientObjectInteraction(nn.Module):
    def __init__(self, feature_dim: int = 128, n_heads: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, feature_dim)
        )
        self.qkv_proj = nn.Linear(feature_dim, feature_dim * 3)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.scale = (self.head_dim ** -0.5)

    def forward(self, object_features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        n = object_features.size(0)
        if n <= 1:
            return object_features
        pos_encoding = self.pos_encoder(positions / 320.0)
        x = object_features + pos_encoding
        residual = x
        x = self.norm1(x)
        qkv = self.qkv_proj(x).reshape(n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).reshape(n, self.feature_dim)
        out = self.out_proj(out)
        x = residual + out
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x
