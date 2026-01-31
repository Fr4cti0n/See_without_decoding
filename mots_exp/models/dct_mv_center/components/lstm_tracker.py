"""
LSTM-based Object Tracker
Tracks objects across frames using temporal modeling
"""

import torch
import torch.nn as nn


class LSTMTracker(nn.Module):
    """
    LSTM-based tracker for temporal object tracking
    
    Takes ROI features and global context, predicts:
    - Position updates (deltas)
    - Confidence scores
    """
    def __init__(
        self,
        roi_feature_dim=64,
        global_feature_dim=64,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.roi_feature_dim = roi_feature_dim
        self.global_feature_dim = global_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Combine ROI and global features
        input_dim = roi_feature_dim + global_feature_dim
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Position prediction head (delta from current position)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # (dx, dy) - center displacement
        )
        
        # Confidence/objectness score head (outputs raw logits for proper loss computation)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Raw logits (no sigmoid for BCE/Focal loss)
        )
        
        # Optional: Size prediction head (for box size changes)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # (dw, dh) - size changes
        )
    
    def forward(self, roi_features, global_features, hidden_state=None, return_logits=False):
        """
        Forward pass through LSTM tracker
        
        Args:
            roi_features: [N, roi_dim] - ROI features for tracked objects
            global_features: [global_dim] or [1, global_dim] - Global context
            hidden_state: Optional (h, c) tuple for LSTM state
            return_logits: If True, return raw logits; if False, return probabilities (default)
        
        Returns:
            position_deltas: [N, 2] - (dx, dy) center displacement
            confidences: [N, 1] - Objectness scores (logits or probabilities)
            size_deltas: [N, 2] - (dw, dh) size changes
            hidden_state: (h, c) tuple for next frame
        """
        N = roi_features.shape[0]
        
        # Expand global features to match number of objects
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)  # [1, global_dim]
        global_features = global_features.expand(N, -1)  # [N, global_dim]
        
        # Combine ROI and global features
        combined_features = torch.cat([roi_features, global_features], dim=1)  # [N, input_dim]
        
        # Add sequence dimension (treating each object as a sequence of length 1)
        combined_features = combined_features.unsqueeze(1)  # [N, 1, input_dim]
        
        # LSTM forward pass
        lstm_out, hidden_state = self.lstm(combined_features, hidden_state)  # [N, 1, hidden_dim]
        lstm_out = lstm_out.squeeze(1)  # [N, hidden_dim]
        
        # Predict outputs
        position_deltas = self.position_head(lstm_out)  # [N, 2]
        confidence_logits = self.confidence_head(lstm_out)  # [N, 1] raw logits
        size_deltas = self.size_head(lstm_out)          # [N, 2]
        
        # Apply sigmoid for inference if needed
        if return_logits:
            confidences = confidence_logits
        else:
            confidences = torch.sigmoid(confidence_logits)
        
        return position_deltas, confidences, size_deltas, hidden_state
    
    def forward_sequence(self, roi_features_seq, global_features_seq):
        """
        Process a sequence of frames
        
        Args:
            roi_features_seq: [T, N, roi_dim] - Sequence of ROI features
            global_features_seq: [T, global_dim] - Sequence of global features
        
        Returns:
            position_deltas: [T, N, 2]
            confidences: [T, N, 1]
            size_deltas: [T, N, 2]
        """
        T, N, _ = roi_features_seq.shape
        
        position_deltas_list = []
        confidences_list = []
        size_deltas_list = []
        
        hidden_state = None
        
        for t in range(T):
            roi_features = roi_features_seq[t]        # [N, roi_dim]
            global_features = global_features_seq[t]  # [global_dim]
            
            pos_delta, conf, size_delta, hidden_state = self.forward(
                roi_features, global_features, hidden_state
            )
            
            position_deltas_list.append(pos_delta)
            confidences_list.append(conf)
            size_deltas_list.append(size_delta)
        
        position_deltas = torch.stack(position_deltas_list, dim=0)  # [T, N, 2]
        confidences = torch.stack(confidences_list, dim=0)          # [T, N, 1]
        size_deltas = torch.stack(size_deltas_list, dim=0)          # [T, N, 2]
        
        return position_deltas, confidences, size_deltas
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state for LSTM
        
        Args:
            batch_size: Number of objects to track
            device: torch.device
        
        Returns:
            (h0, c0): Initial hidden and cell states
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)
