"""Sequential GOP model for MOTS tracking with temporal dependencies."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequentialGOPTracker(nn.Module):
    """Model designed for sequential GOP processing with LSTM for temporal modeling."""
    
    def __init__(self, motion_shape=(2, 40, 40), hidden_dim=128, max_objects=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        
        # Motion vector encoder
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_dim)
        )
        
        # Bounding box encoder
        self.bbox_encoder = nn.Linear(4, 64)
        
        # LSTM for temporal sequence modeling
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim + 64,  # motion features + bbox features
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Multi-object interaction via self-attention
        self.object_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output predictor for bbox deltas
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # x, y, w, h deltas
        )
        
        # Hidden states for sequence processing
        self.hidden_state = None
        
    def reset_sequence(self):
        """Reset LSTM hidden states for new GOP sequence."""
        self.hidden_state = None
        
    def forward(self, motion_vectors, prev_boxes, valid_mask=None):
        """
        Forward pass with temporal LSTM processing.
        
        Args:
            motion_vectors: [B, 2, H, W] motion vector field
            prev_boxes: [B, max_objects, 4] previous bounding boxes
            valid_mask: [B, max_objects] mask for valid objects
        
        Returns:
            [B, max_objects, 4] predicted bounding boxes
        """
        batch_size = motion_vectors.size(0)
        
        # Extract motion features
        motion_features = self.motion_encoder(motion_vectors)  # [B, 128]
        
        # Process each object with LSTM (detach to avoid gradient issues)
        predictions = []
        
        for b in range(batch_size):
            batch_predictions = []
            batch_motion = motion_features[b].unsqueeze(0)  # [1, 128]
            
            for obj_idx in range(self.max_objects):
                if valid_mask is not None and not valid_mask[b, obj_idx]:
                    # Invalid object - predict zeros
                    batch_predictions.append(torch.zeros(4, device=motion_vectors.device))
                    continue
                
                # Get current box
                current_box = prev_boxes[b, obj_idx].unsqueeze(0)  # [1, 4]
                
                # Encode box
                box_features = self.bbox_encoder(current_box)  # [1, 64]
                
                # Combine features
                combined = torch.cat([batch_motion, box_features], dim=1)  # [1, 192]
                
                # LSTM processing (detach hidden states to avoid gradient retention)
                if self.hidden_state is not None:
                    h, c = self.hidden_state
                    h = h.detach()  # Detach to prevent gradient flow
                    c = c.detach()  # Detach to prevent gradient flow
                    self.hidden_state = (h, c)
                
                lstm_out, self.hidden_state = self.temporal_lstm(combined, self.hidden_state)
                
                # Predict bbox delta
                bbox_delta = self.predictor(lstm_out.squeeze(0))  # [4]
                
                # Apply delta to get new box
                new_box = current_box.squeeze(0) + bbox_delta
                batch_predictions.append(new_box)
            
            predictions.append(torch.stack(batch_predictions))
        
        return torch.stack(predictions)
