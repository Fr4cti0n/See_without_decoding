"""
ID-Aware Multi-Object Tracker with Positional Encoding
======================================================

A sophisticated multi-object tracker that:
1. Uses positional encoding for object IDs
2. Maintains object identities across frames
3. Handles object interactions via self-attention
4. Outputs both bounding boxes and ID predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IDPositionalEncoding(nn.Module):
    """Positional encoding for object IDs."""
    
    def __init__(self, d_model, max_id=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(max_id, d_model)
        
    def forward(self, object_ids):
        """
        Args:
            object_ids: [B, max_objects] object IDs
        Returns:
            [B, max_objects, d_model] positional encodings
        """
        return self.embedding(object_ids)


class IDLoss(nn.Module):
    """Combined loss for bounding box regression and ID classification."""
    
    def __init__(self, bbox_weight=1.0, id_weight=1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.id_weight = id_weight
        self.bbox_loss = nn.MSELoss()
        self.id_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore invalid IDs
        
    def forward(self, pred_boxes, target_boxes, pred_ids, target_ids, valid_mask):
        """
        Args:
            pred_boxes: [B, max_objects, 4] predicted boxes
            target_boxes: [B, max_objects, 4] target boxes
            pred_ids: [B, max_objects, max_id] ID logits
            target_ids: [B, max_objects] target IDs
            valid_mask: [B, max_objects] validity mask
        """
        # Bounding box loss (only for valid objects)
        if valid_mask.sum() > 0:
            loss_mask = valid_mask.unsqueeze(-1).expand_as(pred_boxes)
            masked_pred_boxes = pred_boxes * loss_mask.float()
            masked_target_boxes = target_boxes * loss_mask.float()
            bbox_loss = self.bbox_loss(masked_pred_boxes, masked_target_boxes)
        else:
            bbox_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        # ID classification loss (only for valid objects)
        valid_indices = valid_mask.view(-1)
        if valid_indices.sum() > 0:
            flat_pred_ids = pred_ids.view(-1, pred_ids.size(-1))[valid_indices]
            flat_target_ids = target_ids.view(-1)[valid_indices]
            id_loss = self.id_loss(flat_pred_ids, flat_target_ids)
        else:
            id_loss = torch.tensor(0.0, device=pred_ids.device)
        
        total_loss = self.bbox_weight * bbox_loss + self.id_weight * id_loss
        
        return {
            'total_loss': total_loss,
            'bbox_loss': bbox_loss,
            'id_loss': id_loss
        }


class IDMultiObjectTracker(nn.Module):
    """
    ID-aware multi-object tracker with positional encoding.
    
    Features:
    - Positional encoding for object IDs
    - Self-attention for object interactions
    - LSTM for temporal modeling
    - Parallel processing for speed
    - Identity preservation across frames
    """
    
    def __init__(self, motion_shape=(3, 40, 40), hidden_dim=128, max_objects=100, max_id=1000):  # Enhanced: X, Y, Magnitude
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.max_id = max_id
        
        # Motion vector encoder (enhanced for 3 channels: X, Y, Magnitude)
        self.motion_encoder = nn.Sequential(
            nn.Conv2d(motion_shape[0], 64, 3, padding=1),  # Now handles 3 input channels
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, hidden_dim)
        )
        
        # Bounding box encoder
        self.bbox_encoder = nn.Linear(4, hidden_dim // 2)
        
        # ID positional encoding
        self.id_encoder = IDPositionalEncoding(hidden_dim // 2, max_id)
        
        # Multi-head self-attention for object interactions
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # motion + bbox + id features
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # LSTM for temporal sequence modeling
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output prediction heads
        self.bbox_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # x, y, w, h deltas
        )
        
        self.id_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, max_id)  # ID classification logits
        )
        
        # Hidden states for temporal modeling
        self.hidden_state = None
        
    def reset_sequence(self):
        """Reset LSTM hidden states for new GOP sequence."""
        self.hidden_state = None
        
    def forward(self, motion_vectors, prev_boxes, object_ids, valid_mask):
        """
        Forward pass through ID-aware multi-object tracker.
        
        Args:
            motion_vectors: [B, 2, H, W] motion vector field
            prev_boxes: [B, max_objects, 4] previous bounding boxes (x, y, w, h)
            object_ids: [B, max_objects] object IDs (1-based, 0 for invalid)
            valid_mask: [B, max_objects] validity mask for active objects
            
        Returns:
            predicted_boxes: [B, max_objects, 4] predicted bounding boxes
            id_logits: [B, max_objects, max_id] ID classification logits
            attention_weights: attention weights for analysis
        """
        batch_size = motion_vectors.size(0)
        
        # Extract global motion features (shared across all objects)
        motion_features = self.motion_encoder(motion_vectors)  # [B, hidden_dim]
        
        # PARALLEL PROCESSING FOR SPEED
        
        # Encode all bounding boxes at once
        box_features = self.bbox_encoder(prev_boxes)  # [B, max_objects, hidden_dim//2]
        
        # Encode all object IDs with positional encoding
        id_features = self.id_encoder(object_ids)  # [B, max_objects, hidden_dim//2]
        
        # Expand motion features for all objects
        motion_expanded = motion_features.unsqueeze(1).expand(
            batch_size, self.max_objects, -1
        )  # [B, max_objects, hidden_dim]
        
        # Combine all features: motion + bbox + id
        combined_features = torch.cat([
            motion_expanded,  # [B, max_objects, hidden_dim]
            box_features,     # [B, max_objects, hidden_dim//2]
            id_features       # [B, max_objects, hidden_dim//2]
        ], dim=-1)  # [B, max_objects, hidden_dim * 2]
        
        # Apply validity mask to zero out invalid objects
        mask_expanded = valid_mask.unsqueeze(-1).expand_as(combined_features)
        combined_features = combined_features * mask_expanded.float()
        
        # Self-attention for object interactions (collision avoidance)
        attended_features, attention_weights = self.self_attention(
            combined_features, 
            combined_features, 
            combined_features,
            key_padding_mask=~valid_mask  # Mask out invalid objects
        )  # [B, max_objects, hidden_dim * 2]
        
        # LSTM temporal processing (process all objects in parallel)
        # Reshape for LSTM: [B * max_objects, 1, hidden_dim * 2]
        lstm_input = attended_features.reshape(-1, 1, self.hidden_dim * 2)
        
        # CRITICAL FIX: Properly detach hidden states to prevent gradient accumulation
        if self.hidden_state is not None:
            # Detach hidden states to break gradient connection from previous frames
            h0, c0 = self.hidden_state
            self.hidden_state = (h0.detach(), c0.detach())
        
        # Process through LSTM
        lstm_output, new_hidden_state = self.temporal_lstm(lstm_input, self.hidden_state)
        
        # Store detached hidden state for next frame
        self.hidden_state = (new_hidden_state[0].detach(), new_hidden_state[1].detach())
        
        lstm_output = lstm_output.squeeze(1)  # [B * max_objects, hidden_dim]
        
        # Reshape back to batch format
        lstm_output = lstm_output.reshape(batch_size, self.max_objects, -1)  # [B, max_objects, hidden_dim]
        
        # Apply validity mask to LSTM output
        lstm_mask = valid_mask.unsqueeze(-1).expand_as(lstm_output)
        lstm_output = lstm_output * lstm_mask.float()
        
        # PREDICTION HEADS
        
        # Predict bounding box deltas
        bbox_deltas = self.bbox_predictor(lstm_output)  # [B, max_objects, 4]
        predicted_boxes = prev_boxes + bbox_deltas  # Apply deltas to get new boxes
        
        # Predict object IDs (for identity consistency)
        id_logits = self.id_predictor(lstm_output)  # [B, max_objects, max_id]
        
        # Apply validity mask to outputs
        box_mask = valid_mask.unsqueeze(-1).expand_as(predicted_boxes)
        predicted_boxes = predicted_boxes * box_mask.float()
        
        id_mask = valid_mask.unsqueeze(-1).expand_as(id_logits)
        id_logits = id_logits * id_mask.float()
        
        return predicted_boxes, id_logits, attention_weights
    
    def predict_with_nms(self, motion_vectors, prev_boxes, object_ids, valid_mask, nms_threshold=0.5):
        """
        Prediction with Non-Maximum Suppression for collision handling.
        
        Args:
            Same as forward()
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Same as forward() but with NMS applied to resolve overlaps
        """
        predicted_boxes, id_logits, attention_weights = self.forward(
            motion_vectors, prev_boxes, object_ids, valid_mask
        )
        
        # Apply NMS to each batch independently
        batch_size = predicted_boxes.size(0)
        final_boxes = predicted_boxes.clone()
        final_ids = id_logits.clone()
        
        for b in range(batch_size):
            valid_indices = valid_mask[b]
            if valid_indices.sum() == 0:
                continue
                
            # Get valid boxes and their confidence scores
            valid_boxes = predicted_boxes[b][valid_indices]
            valid_id_logits = id_logits[b][valid_indices]
            
            # Use max ID probability as confidence score
            confidence_scores = torch.max(torch.softmax(valid_id_logits, dim=-1), dim=-1)[0]
            
            # Apply NMS
            keep_indices = self._nms(valid_boxes, confidence_scores, nms_threshold)
            
            # Update predictions - suppress overlapping detections
            suppressed_mask = torch.ones(valid_indices.sum(), dtype=torch.bool, device=valid_boxes.device)
            suppressed_mask[keep_indices] = False
            
            # Zero out suppressed detections
            if suppressed_mask.sum() > 0:
                global_suppressed_indices = torch.where(valid_indices)[0][suppressed_mask]
                final_boxes[b][global_suppressed_indices] = 0
                final_ids[b][global_suppressed_indices] = 0
                valid_mask[b][global_suppressed_indices] = False
        
        return final_boxes, final_ids, attention_weights
    
    def _nms(self, boxes, scores, threshold):
        """Simple NMS implementation."""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Convert to (x1, y1, x2, y2) format
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            # Compute IoU
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            
            w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
            h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            inds = torch.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return torch.tensor(keep, dtype=torch.long)


def create_id_aware_tracker(config):
    """Factory function to create ID-aware tracker with config."""
    return IDMultiObjectTracker(
        motion_shape=(2, 40, 40),
        hidden_dim=128,
        max_objects=config.max_objects,
        max_id=config.max_objects * 2  # Allow for more IDs than objects
    )
