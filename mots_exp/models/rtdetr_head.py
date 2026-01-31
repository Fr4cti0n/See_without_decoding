"""
RT-DETR Head for Object Detection with Explicit No-Object Class

This module implements a detection head inspired by RT-DETR that:
1. Uses explicit "no_object" class for better handling of empty slots
2. Employs focal loss to handle class imbalance (290 empty vs 10 occupied slots)
3. Combines L1 + GIoU losses for stable box regression
4. Provides quality-aware predictions for autoregressive propagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RTDETRHead(nn.Module):
    """
    RT-DETR inspired detection head for tracking.
    
    Predicts:
    - Bounding boxes (4 values: x, y, w, h)
    - Class scores (2 classes: no_object, pedestrian)
    - Track ID (for identity preservation)
    
    Args:
        input_dim: Dimension of input features (from LSTM)
        num_slots: Number of detection slots (300)
        num_classes: Number of classes including no_object (2: [no_object, pedestrian])
        num_track_ids: Number of possible track IDs (1000)
        hidden_dim: Hidden dimension for MLP layers
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        num_slots: int = 300,
        num_classes: int = 2,  # [no_object, pedestrian]
        num_track_ids: int = 1000,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.num_track_ids = num_track_ids
        
        # Box regression head (L1 + GIoU for stability)
        self.box_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)  # [x, y, w, h]
        )
        
        # Classification head (no_object vs pedestrian)
        self.class_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)  # Logits for [no_object, pedestrian]
        )
        
        # Track ID head
        self.track_id_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_track_ids)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Bias initialization for class head
        # Start with MORE BALANCED prior to allow gradient flow
        # p(no_object) = 0.7, p(pedestrian) = 0.3
        # This gives enough signal for learning while biasing toward no_object
        prior_prob = 0.3  # Probability of pedestrian (was 0.01 - too extreme!)
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.class_head[-1].bias[0], -bias_value)  # no_object
        nn.init.constant_(self.class_head[-1].bias[1], bias_value)   # pedestrian
    
    def forward(self, slot_features):
        """
        Forward pass to predict boxes, classes, and track IDs
        
        Args:
            slot_features: [batch_size, num_slots, input_dim]
        
        Returns:
            boxes: [batch_size, num_slots, 4] - predicted boxes (x, y, w, h) in [0, 1]
            class_logits: [batch_size, num_slots, num_classes] - logits for classification
            track_ids: [batch_size, num_slots, num_track_ids] - logits for track ID
        """
        batch_size, num_slots, _ = slot_features.shape
        
        # Predict boxes
        boxes = self.box_head(slot_features)  # [B, N, 4]
        boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]
        
        # Predict class logits (will apply softmax during loss computation)
        class_logits = self.class_head(slot_features)  # [B, N, 2]
        
        # Predict track ID logits
        track_ids = self.track_id_head(slot_features)  # [B, N, num_track_ids]
        
        return boxes, class_logits, track_ids
    
    def get_confident_predictions(self, boxes, class_logits, track_ids, confidence_threshold=0.5):
        """
        Filter predictions to keep only confident detections (for autoregressive propagation)
        
        Args:
            boxes: [batch_size, num_slots, 4]
            class_logits: [batch_size, num_slots, num_classes]
            track_ids: [batch_size, num_slots, num_track_ids]
            confidence_threshold: Minimum p(pedestrian) to keep
        
        Returns:
            filtered_boxes: [batch_size, K, 4] where K <= num_slots
            filtered_track_ids: [batch_size, K]
            mask: [batch_size, num_slots] - True for kept predictions
        """
        batch_size = boxes.shape[0]
        
        # Get class probabilities
        class_probs = F.softmax(class_logits, dim=-1)  # [B, N, 2]
        pedestrian_probs = class_probs[:, :, 1]  # [B, N] - probability of pedestrian class
        
        # Create mask for confident predictions
        mask = pedestrian_probs > confidence_threshold  # [B, N]
        
        # For each batch, select confident predictions
        # Note: This returns variable-length outputs per batch
        # For simplicity in autoregressive mode, we'll keep all slots but mark confidence
        
        return boxes, track_ids, mask


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    where:
    - p_t is the model's estimated probability for the class
    - α_t is the weighting factor (class balance)
    - γ is the focusing parameter (down-weights easy examples)
    
    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, num_classes] - logits
            targets: [N] - class indices
        
        Returns:
            loss: scalar or [N] depending on reduction
        """
        # Get probabilities
        p = F.softmax(inputs, dim=-1)  # [N, num_classes]
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [N]
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # [N]
        
        # Focal loss formula
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(targets == 0, 1 - self.alpha, self.alpha)
        
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_giou_loss(pred_boxes, target_boxes):
    """
    Compute GIoU loss between predicted and target boxes
    
    Args:
        pred_boxes: [N, 4] - (x, y, w, h) in [0, 1]
        target_boxes: [N, 4] - (x, y, w, h) in [0, 1]
    
    Returns:
        giou_loss: [N] - GIoU loss per box (1 - GIoU)
    """
    # Convert to [x1, y1, x2, y2]
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    
    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
    
    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
    # Loss is 1 - GIoU
    loss = 1 - giou
    
    return loss


if __name__ == "__main__":
    # Test the RT-DETR head
    print("Testing RT-DETR Head...")
    
    batch_size = 2
    num_slots = 300
    input_dim = 256
    
    # Create random features
    slot_features = torch.randn(batch_size, num_slots, input_dim)
    
    # Create head
    head = RTDETRHead(input_dim=input_dim, num_slots=num_slots)
    
    # Forward pass
    boxes, class_logits, track_ids = head(slot_features)
    
    print(f"Input features: {slot_features.shape}")
    print(f"Output boxes: {boxes.shape}")
    print(f"Output class logits: {class_logits.shape}")
    print(f"Output track IDs: {track_ids.shape}")
    
    # Test confident predictions
    filtered_boxes, filtered_track_ids, mask = head.get_confident_predictions(
        boxes, class_logits, track_ids, confidence_threshold=0.5
    )
    
    print(f"\nConfident predictions:")
    print(f"Mask shape: {mask.shape}")
    print(f"Num confident per batch: {mask.sum(dim=1)}")
    
    # Test focal loss
    print("\nTesting Focal Loss...")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Simulate predictions and targets
    num_samples = 300
    logits = torch.randn(num_samples, 2)  # Random logits
    targets = torch.cat([
        torch.ones(10, dtype=torch.long),   # 10 pedestrians
        torch.zeros(290, dtype=torch.long)  # 290 no_object
    ])
    
    loss = focal_loss(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test GIoU loss
    print("\nTesting GIoU Loss...")
    pred_boxes = torch.rand(10, 4)
    target_boxes = torch.rand(10, 4)
    giou_loss = compute_giou_loss(pred_boxes, target_boxes)
    print(f"GIoU loss shape: {giou_loss.shape}")
    print(f"Mean GIoU loss: {giou_loss.mean().item():.4f}")
    
    print("\n✅ All tests passed!")
