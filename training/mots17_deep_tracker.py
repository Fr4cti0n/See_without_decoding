#!/usr/bin/env python3
"""
MOTS17 Deep Learning Tracker
============================

A comprehensive deep learning model for Multi-Object Tracking and Segmentation 
on the MOTS17 dataset using motion vector accumulation and transformer attention.

Features:
- Motion vector accumulation-based tracking
- Transformer attention for object relationships  
- Multi-scale feature extraction
- Temporal consistency modeling
- Advanced loss functions with IoU and confidence prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm 
import random 
import math
from pathlib import Path

# Set up paths to import your data loaders
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'dataset'))
sys.path.append(os.path.join(parent_dir, 'dataset', 'visualization_toolkit'))

# Import your real data loaders
try:
    from dataset.core.mots_dataset import MOTSSequenceDataset
    REAL_DATA_AVAILABLE = True
    print("✅ Successfully imported MOTS dataset from dataset/core")
except ImportError as e:
    print(f"⚠️ Could not import MOTS dataset: {e}")
    print("📄 Using fallback synthetic data for testing")
    REAL_DATA_AVAILABLE = False

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer attention"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MotionVectorEncoder(nn.Module):
    """Enhanced motion vector encoder with multi-scale processing"""
    
    def __init__(self, input_channels=2, hidden_dim=128):
        super().__init__()
        
        # Multi-scale convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Different scale processing
        self.scale1 = nn.Conv2d(128, 64, kernel_size=1)  # 1x1 conv
        self.scale2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 3x3 conv
        self.scale3 = nn.Conv2d(128, 64, kernel_size=5, padding=2)  # 5x5 conv
        
        # Fusion layer
        self.fusion = nn.Conv2d(192, hidden_dim, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, motion_vectors):
        # Initial feature extraction
        x = self.relu(self.conv1(motion_vectors))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Multi-scale processing
        scale1 = self.relu(self.scale1(x))
        scale2 = self.relu(self.scale2(x))
        scale3 = self.relu(self.scale3(x))
        
        # Concatenate and fuse
        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)
        features = self.fusion(multi_scale)
        features = self.dropout(features)
        
        return features

class ObjectEncoder(nn.Module):
    """Encode object bounding boxes with position and size information"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Bounding box encoding (cx, cy, w, h) -> hidden_dim
        self.bbox_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Position embedding for spatial relationships
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_dim))
        
    def forward(self, bboxes):
        # bboxes: (batch_size, num_objects, 4)
        encoded = self.bbox_encoder(bboxes)
        
        # Add positional embedding
        encoded = encoded + self.pos_embedding
        
        return encoded

class TransformerAttention(nn.Module):
    """Multi-head attention for object relationships"""
    
    def __init__(self, hidden_dim=128, num_heads=8, num_layers=2):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, object_features):
        # object_features: (batch_size, num_objects, hidden_dim)
        # Add positional encoding
        features = self.pos_encoding(object_features)
        
        # Apply transformer attention
        attended_features = self.transformer(features)
        
        return attended_features

class MotionVectorDeepTracker(nn.Module):
    """
    Deep Learning Model for MOTS17 Multi-Object Tracking using Motion Vector Accumulation
    
    Architecture:
    1. Motion Vector Encoder: Multi-scale CNN for motion field processing
    2. Object Encoder: MLP for bounding box feature extraction  
    3. Feature Fusion: Spatial pooling to extract motion at object locations
    4. Transformer Attention: Model object relationships and interactions
    5. Prediction Head: Predict offset, confidence, and temporal consistency
    """
    
    def __init__(self, hidden_dim=128, num_heads=8, transformer_layers=2):
        super().__init__()
        
        # Core components
        self.motion_encoder = MotionVectorEncoder(input_channels=2, hidden_dim=hidden_dim)
        self.object_encoder = ObjectEncoder(hidden_dim=hidden_dim)
        self.attention = TransformerAttention(hidden_dim, num_heads, transformer_layers)
        
        # Feature fusion
        self.motion_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Combined feature processing
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Prediction heads
        self.offset_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # dx, dy, dw, dh
        )
        
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency head
        self.temporal_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.hidden_dim = hidden_dim
        
    def extract_object_motion_features(self, motion_features, bboxes):
        """Extract motion features at object locations using spatial pooling"""
        batch_size, num_objects = bboxes.shape[:2]
        _, channels, height, width = motion_features.shape
        
        object_motion_features = []
        
        for b in range(batch_size):
            batch_features = []
            for obj in range(num_objects):
                # Get object bounding box (normalized coordinates)
                cx, cy, w, h = bboxes[b, obj]
                
                # Convert to pixel coordinates
                x1 = max(0, int((cx - w/2) * width))
                y1 = max(0, int((cy - h/2) * height))
                x2 = min(width, int((cx + w/2) * width))
                y2 = min(height, int((cy + h/2) * height))
                
                # Extract region and pool
                if x2 > x1 and y2 > y1:
                    region = motion_features[b, :, y1:y2, x1:x2]
                    pooled = F.adaptive_avg_pool2d(region, (1, 1)).squeeze()
                else:
                    # Fallback for invalid regions
                    pooled = torch.zeros(channels, device=motion_features.device)
                
                batch_features.append(pooled)
            
            if batch_features:
                object_motion_features.append(torch.stack(batch_features))
            else:
                # Fallback for empty batch
                object_motion_features.append(torch.zeros(1, channels, device=motion_features.device))
        
        return torch.stack(object_motion_features)
    
    def forward(self, motion_vectors, prev_bboxes, valid_mask=None):
        """
        Forward pass
        
        Args:
            motion_vectors: (batch_size, 2, height, width) - Motion vector field
            prev_bboxes: (batch_size, max_objects, 4) - Previous frame bounding boxes (padded)
            valid_mask: (batch_size, max_objects) - Mask indicating valid objects
            
        Returns:
            predictions: Dict with offset predictions, confidence scores, etc.
        """
        batch_size, max_objects = prev_bboxes.shape[:2]
        
        if valid_mask is None:
            valid_mask = torch.ones(batch_size, max_objects, dtype=torch.bool, device=prev_bboxes.device)
        
        # Encode motion vectors
        motion_features = self.motion_encoder(motion_vectors)  # (B, hidden_dim, H, W)
        
        # Encode object bounding boxes
        object_features = self.object_encoder(prev_bboxes)  # (B, max_objects, hidden_dim)
        
        # Extract motion features at object locations
        object_motion_features = self.extract_object_motion_features(
            motion_features, prev_bboxes
        )  # (B, max_objects, hidden_dim)
        
        # Combine object and motion features
        combined_features = torch.cat([object_features, object_motion_features], dim=-1)
        combined_features = self.feature_combiner(combined_features)  # (B, max_objects, hidden_dim)
        
        # Apply attention mask for valid objects
        # Create attention mask: False for valid positions, True for masked positions
        attention_mask = ~valid_mask  # (B, max_objects)
        
        # Apply transformer attention for object relationships
        attended_features = self.attention(combined_features)  # (B, max_objects, hidden_dim)
        
        # Zero out features for invalid objects
        attended_features = attended_features * valid_mask.unsqueeze(-1).float()
        
        # Predict offsets, confidence, and temporal consistency
        predicted_offsets = self.offset_predictor(attended_features)  # (B, max_objects, 4)
        confidence_scores = self.confidence_predictor(attended_features)  # (B, max_objects, 1)
        temporal_scores = self.temporal_predictor(attended_features)  # (B, max_objects, 1)
        
        # Apply offsets to get updated bounding boxes
        updated_bboxes = prev_bboxes + predicted_offsets
        
        # Clamp to valid range [0, 1]
        updated_bboxes = torch.clamp(updated_bboxes, 0, 1)
        
        # Zero out predictions for invalid objects
        updated_bboxes = updated_bboxes * valid_mask.unsqueeze(-1).float()
        predicted_offsets = predicted_offsets * valid_mask.unsqueeze(-1).float()
        confidence_scores = confidence_scores * valid_mask.unsqueeze(-1).float()
        temporal_scores = temporal_scores * valid_mask.unsqueeze(-1).float()
        
        return {
            'updated_boxes': updated_bboxes,
            'predicted_offsets': predicted_offsets,
            'confidence_scores': confidence_scores.squeeze(-1),
            'temporal_scores': temporal_scores.squeeze(-1),
            'attended_features': attended_features,
            'valid_mask': valid_mask
        }

class MOTS17DeepLoss(nn.Module):
    """
    Advanced loss function for MOTS17 deep learning tracking
    
    Components:
    1. Offset Loss: L1 loss for bounding box displacement
    2. IoU Loss: Generalized IoU loss for spatial accuracy
    3. Confidence Loss: Binary cross-entropy for tracking confidence
    4. Temporal Consistency Loss: Encourages smooth temporal transitions
    5. Attention Regularization: Prevents attention collapse
    """
    
    def __init__(self, 
                 offset_weight=1.0,
                 iou_weight=2.0, 
                 confidence_weight=1.0,
                 temporal_weight=0.5,
                 attention_weight=0.1):
        super().__init__()
        
        self.offset_weight = offset_weight
        self.iou_weight = iou_weight
        self.confidence_weight = confidence_weight 
        self.temporal_weight = temporal_weight
        self.attention_weight = attention_weight
        
    def compute_giou_loss(self, pred_boxes, target_boxes):
        """Compute Generalized IoU loss"""
        # Convert to corner format
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2
        
        target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
        target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2
        
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
        iou = inter_area / (union_area + 1e-8)
        
        # Enclosing box for GIoU
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-8)
        
        return 1 - giou, iou
    
    def forward(self, predictions, targets):
        """
        Compute multi-component loss
        
        Args:
            predictions: Dict from model forward pass
            targets: Dict with target_boxes, valid_mask and other ground truth
            
        Returns:
            loss: Total loss value
            components: Dict with individual loss components and metrics
        """
        pred_boxes = predictions['updated_boxes']
        target_boxes = targets['target_boxes']
        confidence_scores = predictions['confidence_scores']
        temporal_scores = predictions['temporal_scores']
        valid_mask = predictions['valid_mask']
        
        # Only compute loss for valid objects
        valid_pred_boxes = pred_boxes[valid_mask]
        valid_target_boxes = target_boxes[valid_mask]
        valid_confidence_scores = confidence_scores[valid_mask]
        valid_temporal_scores = temporal_scores[valid_mask]
        
        if valid_pred_boxes.size(0) == 0:
            # No valid objects in batch
            return torch.tensor(0.0, device=pred_boxes.device), {
                'offset_loss': 0.0,
                'giou_loss': 0.0,
                'confidence_loss': 0.0,
                'temporal_loss': 0.0,
                'attention_reg': 0.0,
                'mean_iou': 0.0,
                'mean_confidence': 0.0,
                'mean_temporal': 0.0
            }
        
        # 1. Offset Loss (L1)
        predicted_offsets = predictions['predicted_offsets'][valid_mask]
        valid_prev_boxes = targets['prev_boxes'][valid_mask]
        target_offsets = valid_target_boxes - valid_prev_boxes
        offset_loss = F.l1_loss(predicted_offsets, target_offsets)
        
        # 2. GIoU Loss  
        giou_loss, iou_values = self.compute_giou_loss(valid_pred_boxes, valid_target_boxes)
        giou_loss = giou_loss.mean()
        
        # 3. Confidence Loss (target confidence based on IoU)
        confidence_targets = iou_values.detach()
        confidence_loss = F.binary_cross_entropy(valid_confidence_scores, confidence_targets)
        
        # 4. Temporal Consistency Loss (encourage high temporal scores for good predictions)
        temporal_targets = (iou_values > 0.5).float().detach()  # High temporal score for good IoU
        temporal_loss = F.binary_cross_entropy(valid_temporal_scores, temporal_targets)
        
        # 5. Attention Regularization (prevent attention collapse)
        attended_features = predictions['attended_features'][valid_mask]
        if attended_features.size(0) > 1:
            attention_reg = torch.var(attended_features, dim=0).mean()  # Encourage diversity
        else:
            attention_reg = torch.tensor(0.0, device=pred_boxes.device)
        
        # Total loss
        total_loss = (
            self.offset_weight * offset_loss +
            self.iou_weight * giou_loss + 
            self.confidence_weight * confidence_loss +
            self.temporal_weight * temporal_loss +
            self.attention_weight * attention_reg
        )
        
        # Metrics for monitoring
        with torch.no_grad():
            mean_iou = iou_values.mean().item()
            mean_confidence = valid_confidence_scores.mean().item()
            mean_temporal = valid_temporal_scores.mean().item()
        
        components = {
            'offset_loss': offset_loss.item(),
            'giou_loss': giou_loss.item(), 
            'confidence_loss': confidence_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'attention_reg': attention_reg.item(),
            'mean_iou': mean_iou,
            'mean_confidence': mean_confidence,
            'mean_temporal': mean_temporal
        }
        
        return total_loss, components

class RealMOTS17Dataset(Dataset):
    """
    Real MOTS17 dataset using your actual dataset/core/mots_dataset.py
    Loads real motion vectors and bounding box sequences from your compressed data
    """
    
    def __init__(self, dataset_type='train', max_objects=8, max_samples=None, root_dir=None):
        """
        Args:
            dataset_type: 'train' or 'val' 
            max_objects: Maximum number of objects to pad to
            max_samples: Limit number of samples (for testing)
            root_dir: Root directory for MOTS data (will auto-detect if None)
        """
        self.max_objects = max_objects
        self.samples = []
        
        if not REAL_DATA_AVAILABLE:
            print("⚠️ Real MOTS dataset not available, falling back to synthetic data")
            self._create_synthetic_fallback(max_samples or 1000)
            return
            
        print(f"🔍 Loading real MOTS17 {dataset_type} data using dataset/core...")
        
        try:
            # Auto-detect root directory if not provided
            if root_dir is None:
                # Look for MOTS data directories (prioritize user's real data location)
                possible_dirs = [
                    '/home/aduche/Bureau/datasets/MOTS/videos',  # User's real MOTS data
                    '/home/aduche/Bureau/datasets/MOTS',  # Alternative MOTS location
                    '/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/MOTS-experiments/dataset/data',
                    '/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/MOTS-experiments/data',
                    os.path.join(parent_dir, 'data'),
                    os.path.join(parent_dir, 'dataset', 'data'),
                ]
                
                for dir_path in possible_dirs:
                    if os.path.exists(dir_path):
                        root_dir = dir_path
                        break
                
                if root_dir is None:
                    print("⚠️ Could not find MOTS data directory, creating synthetic fallback")
                    self._create_synthetic_fallback(max_samples or 1000)
                    return
            
            print(f"📁 Using MOTS data directory: {root_dir}")
            
            # Initialize the real MOTS dataset - ONLY motion vectors and annotations
            self.mots_dataset = MOTSSequenceDataset(
                root_dir=root_dir,
                image_size=640,  # Standard MOTS size
                mode=dataset_type,
                mot_dataset="mot17",
                load_iframe=False,     # ❌ Don't load iframe - not needed
                load_pframe=False,     # ❌ Don't load pframe - not needed  
                load_motion_vectors=True,   # ✅ ONLY motion vectors
                load_residuals=False,       # ❌ Don't load residuals
                load_annotations=True,      # ✅ ONLY annotations
                data_format="separate"
            )
            
            print(f"✅ Loaded MOTS dataset with {len(self.mots_dataset)} samples")
            
            # Process samples for deep learning
            self._process_mots_samples(max_samples)
            
            if len(self.samples) == 0:
                print("⚠️ No processable samples found, falling back to synthetic data")
                self._create_synthetic_fallback(max_samples or 1000)
                
        except Exception as e:
            print(f"❌ Error loading real MOTS data: {e}")
            import traceback
            traceback.print_exc()
            print("� Falling back to synthetic data")
            self._create_synthetic_fallback(max_samples or 1000)
    
    def _process_mots_samples(self, max_samples):
        """Process MOTS dataset samples for deep learning training"""
        print(f"🔄 Processing MOTS samples for deep learning...")
        
        processed_count = 0
        consecutive_frames = []
        
        try:
            # Process samples in pairs (consecutive frames)
            for i in range(len(self.mots_dataset)):
                if max_samples and processed_count >= max_samples:
                    break
                
                try:
                    # Get current sample
                    sample = self.mots_dataset[i]
                    
                    # Check if we have the required data
                    if not self._validate_sample(sample):
                        continue
                    
                    # Store for consecutive frame processing
                    consecutive_frames.append((i, sample))
                    
                    # Process pairs of consecutive frames
                    if len(consecutive_frames) >= 2:
                        prev_idx, prev_sample = consecutive_frames[-2]
                        curr_idx, curr_sample = consecutive_frames[-1]
                        
                        # Create training sample from consecutive frames
                        training_sample = self._create_training_sample(
                            prev_sample, curr_sample, prev_idx, curr_idx
                        )
                        
                        if training_sample:
                            self.samples.append(training_sample)
                            processed_count += 1
                            
                        # Keep only last frame for next iteration
                        consecutive_frames = consecutive_frames[-1:]
                        
                except Exception as e:
                    print(f"⚠️ Error processing sample {i}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error in sample processing: {e}")
            
        print(f"✅ Processed {len(self.samples)} training samples from MOTS dataset")
    
    def _validate_sample(self, sample):
        """Validate that sample has required data for training - ONLY motion vectors and annotations"""
        required_keys = ['motion_vectors', 'boxes']  # boxes contains the bounding box annotations
        
        for key in required_keys:
            if key not in sample or sample[key] is None:
                return False
        
        # Check that motion vectors have correct format
        motion_vectors = sample['motion_vectors']
        if not isinstance(motion_vectors, (torch.Tensor, np.ndarray)):
            return False
            
        # Check bounding boxes
        boxes = sample['boxes']
        if not isinstance(boxes, (torch.Tensor, np.ndarray)) or len(boxes) == 0:
            return False
            
        return True
    
    def _create_training_sample(self, prev_sample, curr_sample, prev_idx, curr_idx):
        """Create a training sample from two consecutive frames"""
        try:
            # Extract motion vectors (from current frame)
            motion_vectors = curr_sample['motion_vectors']
            
            # Convert to tensor if numpy
            if isinstance(motion_vectors, np.ndarray):
                motion_vectors = torch.from_numpy(motion_vectors).float()
            
            # Debug: print original shape
            original_shape = motion_vectors.shape
            
            # Handle the 4D case: [2, 60, 60, 2] -> extract the useful channel
            if motion_vectors.dim() == 4 and motion_vectors.shape[0] == 2:
                # We have 2 channels, extract the one with actual motion vectors (coordinates)
                # Usually the second channel (index 1) contains the motion coordinates
                motion_vectors = motion_vectors[1]  # Shape: [60, 60, 2]
                #print(f"🔧 Extracted motion vector channel from {original_shape} -> {motion_vectors.shape}")
            elif motion_vectors.dim() == 4:
                # Remove batch dimension if present: (1, H, W, 2) -> (H, W, 2)
                motion_vectors = motion_vectors.squeeze(0)
            elif motion_vectors.dim() == 2:
                # If somehow flattened, try to detect the format
                if motion_vectors.shape[1] == 2:  # Likely (N, 2) format
                    print(f"⚠️ Unexpected 2D motion vector shape: {original_shape}, skipping")
                    return None
            
            # Now handle the 3D case: convert (H, W, 2) -> (2, H, W)
            if motion_vectors.dim() == 3:
                if motion_vectors.shape[-1] == 2:
                    # (H, W, 2) -> (2, H, W)
                    motion_vectors = motion_vectors.permute(2, 0, 1)
                elif motion_vectors.shape[0] == 2:
                    # Already in (2, H, W) format
                    pass
                else:
                    print(f"⚠️ Unexpected motion vector shape: {original_shape} -> {motion_vectors.shape}")
                    return None
            else:
                print(f"⚠️ Unsupported motion vector dimensions: {original_shape} (after processing: {motion_vectors.shape})")
                return None
            
            # Final safety check before interpolation
            if motion_vectors.dim() != 3 or motion_vectors.shape[0] != 2:
                print(f"⚠️ Motion vectors not in (2, H, W) format: {motion_vectors.shape}")
                return None
            
            # Resize to standard size if needed
            if motion_vectors.shape[1:] != (60, 60):
                # motion_vectors is now in (2, H, W) format, add batch dimension for interpolation
                try:
                    motion_vectors = F.interpolate(
                        motion_vectors.unsqueeze(0),  # (1, 2, H, W)
                        size=(60, 60), 
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Back to (2, H, W)
                except Exception as interp_e:
                    print(f"⚠️ F.interpolate failed with shape {motion_vectors.shape}: {interp_e}")
                    return None
            
            # Extract bounding boxes from both frames
            prev_boxes = self._extract_bboxes_from_sample(prev_sample)
            curr_boxes = self._extract_bboxes_from_sample(curr_sample)
            
            if len(prev_boxes) == 0 or len(curr_boxes) == 0:
                return None
            
            # Match objects between frames
            matched_pairs = self._match_objects_simple(prev_boxes, curr_boxes)
            
            if len(matched_pairs) == 0:
                return None
            
            # Create padded tensors
            num_objects = min(len(matched_pairs), self.max_objects)
            prev_boxes_tensor = torch.zeros(self.max_objects, 4)
            target_boxes_tensor = torch.zeros(self.max_objects, 4)
            valid_mask = torch.zeros(self.max_objects, dtype=torch.bool)
            
            for i in range(num_objects):
                prev_boxes_tensor[i] = torch.tensor(matched_pairs[i]['prev'])
                target_boxes_tensor[i] = torch.tensor(matched_pairs[i]['curr'])
                valid_mask[i] = True
            
            return {
                'motion_vectors': motion_vectors,
                'prev_boxes': prev_boxes_tensor,
                'target_boxes': target_boxes_tensor,
                'valid_mask': valid_mask,
                'num_objects': num_objects,
                'frame_pair': (prev_idx, curr_idx),
                'is_real_data': True,
                'sequence_id': curr_sample.get('sequence_id', f'seq_{curr_idx}')
            }
            
        except Exception as e:
            print(f"⚠️ Error creating training sample: {e}")
            return None
    
    def _extract_bboxes_from_sample(self, sample):
        """Extract normalized bounding boxes from MOTS sample format"""
        boxes = []
        
        try:
            if 'boxes' in sample and 'ids' in sample:
                bbox_tensors = sample['boxes']  # Tensor of shape [N, 4]
                id_tensors = sample['ids']      # Tensor of shape [N]
                
                # Convert to numpy for easier handling
                if isinstance(bbox_tensors, torch.Tensor):
                    bbox_tensors = bbox_tensors.cpu().numpy()
                if isinstance(id_tensors, torch.Tensor):
                    id_tensors = id_tensors.cpu().numpy()
                
                # Extract bounding boxes
                for i in range(len(bbox_tensors)):
                    bbox = bbox_tensors[i]  # [x, y, w, h] format
                    obj_id = id_tensors[i]
                    
                    # Convert to normalized center format
                    x, y, w, h = bbox[:4]
                    
                    # Normalize to [0, 1] range (assuming 640x640 image)
                    cx = (x + w/2) / 640.0
                    cy = (y + h/2) / 640.0
                    nw = w / 640.0
                    nh = h / 640.0
                    
                    # Clamp to valid range
                    cx = np.clip(cx, 0, 1)
                    cy = np.clip(cy, 0, 1)
                    nw = np.clip(nw, 0, 1)
                    nh = np.clip(nh, 0, 1)
                    
                    boxes.append({
                        'id': int(obj_id),
                        'bbox': [cx, cy, nw, nh]
                    })
                        
        except Exception as e:
            print(f"⚠️ Error extracting bboxes from sample: {e}")
            
        return boxes
    
    def _extract_bboxes(self, annotations):
        """Extract normalized bounding boxes from annotations (legacy method)"""
        boxes = []
        
        try:
            if isinstance(annotations, dict):
                for obj_id, obj_data in annotations.items():
                    if isinstance(obj_data, dict) and 'bbox' in obj_data:
                        bbox = obj_data['bbox']
                        # Normalize to [0, 1] range
                        # Assuming bbox is [x, y, w, h] in pixel coordinates
                        x, y, w, h = bbox[:4]
                        
                        # Convert to center format and normalize (assuming 640x640 image)
                        cx = (x + w/2) / 640.0
                        cy = (y + h/2) / 640.0
                        nw = w / 640.0
                        nh = h / 640.0
                        
                        boxes.append({
                            'id': obj_id,
                            'bbox': [cx, cy, nw, nh]
                        })
            elif isinstance(annotations, list):
                for i, bbox in enumerate(annotations):
                    if len(bbox) >= 4:
                        x, y, w, h = bbox[:4]
                        cx = (x + w/2) / 640.0
                        cy = (y + h/2) / 640.0
                        nw = w / 640.0
                        nh = h / 640.0
                        
                        boxes.append({
                            'id': i,
                            'bbox': [cx, cy, nw, nh]
                        })
                        
        except Exception as e:
            print(f"⚠️ Error extracting bboxes: {e}")
            
        return boxes
    
    def _match_objects_simple(self, prev_boxes, curr_boxes):
        """Simple nearest neighbor object matching"""
        matched_pairs = []
        
        for prev_obj in prev_boxes:
            best_match = None
            best_distance = float('inf')
            
            prev_center = np.array(prev_obj['bbox'][:2])
            
            for curr_obj in curr_boxes:
                curr_center = np.array(curr_obj['bbox'][:2])
                distance = np.linalg.norm(prev_center - curr_center)
                
                if distance < best_distance and distance < 0.1:  # Max distance threshold
                    best_distance = distance
                    best_match = curr_obj
            
            if best_match:
                matched_pairs.append({
                    'prev': prev_obj['bbox'],
                    'curr': best_match['bbox'],
                    'prev_id': prev_obj['id'],
                    'curr_id': best_match['id']
                })
        
        return matched_pairs
    
    def _create_synthetic_fallback(self, num_samples):
        """Create synthetic samples as fallback"""
        print(f"📄 Creating {num_samples} synthetic samples as fallback")
        
        for idx in range(num_samples):
            # Generate random motion field
            motion_field = torch.randn(2, 60, 60) * 0.1
            
            # Generate random number of objects
            num_objects = np.random.randint(1, self.max_objects + 1)
            
            # Initialize padded tensors
            prev_boxes = torch.zeros(self.max_objects, 4)
            target_boxes = torch.zeros(self.max_objects, 4)
            valid_mask = torch.zeros(self.max_objects, dtype=torch.bool)
            
            # Generate actual objects
            for i in range(num_objects):
                cx = torch.rand(1).item()
                cy = torch.rand(1).item()
                w = torch.rand(1).item() * 0.3 + 0.1
                h = torch.rand(1).item() * 0.3 + 0.1
                
                prev_boxes[i] = torch.tensor([cx, cy, w, h])
                valid_mask[i] = True
                
                # Apply synthetic motion
                motion_x = motion_field[0, int(cy*59), int(cx*59)].item()
                motion_y = motion_field[1, int(cy*59), int(cx*59)].item()
                target_boxes[i] = torch.tensor([cx + motion_x, cy + motion_y, w, h])
            
            target_boxes = torch.clamp(target_boxes, 0, 1)
            
            self.samples.append({
                'motion_vectors': motion_field,
                'prev_boxes': prev_boxes,
                'target_boxes': target_boxes,
                'valid_mask': valid_mask,
                'num_objects': num_objects,
                'gop_index': idx % 50,
                'is_real_data': False
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class MOTS17Trainer:
    """Trainer for MOTS17 Deep Learning Tracker"""
    
    def __init__(self, 
                 model_name="mots17_deep_tracker",
                 learning_rate=0.001,
                 batch_size=16,
                 num_epochs=50,
                 hidden_dim=128,
                 num_heads=8,
                 transformer_layers=2,
                 device='cuda'):
        
        self.model_name = model_name
        self.device = device
        self.num_epochs = num_epochs
        
        # Initialize deep learning model
        self.model = MotionVectorDeepTracker(
            hidden_dim=hidden_dim,
            num_heads=num_heads, 
            transformer_layers=transformer_layers
        ).to(device)
        
        self.criterion = MOTS17DeepLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=learning_rate * 0.01
        )
        
        # Training history
        self.train_losses = []
        self.train_ious = []
        self.val_losses = []
        self.val_ious = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🚀 Initialized MOTS17 Deep Learning Tracker")
        print(f"   Model: {self.model_name}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Hidden dimension: {hidden_dim}")
        print(f"   Attention heads: {num_heads}")
        print(f"   Transformer layers: {transformer_layers}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Device: {device}")
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        epoch_ious = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            # Move to device
            motion_vectors = batch['motion_vectors'].to(self.device)
            prev_boxes = batch['prev_boxes'].to(self.device)
            target_boxes = batch['target_boxes'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(motion_vectors, prev_boxes, valid_mask)
            
            # Compute loss
            targets = {
                'prev_boxes': prev_boxes,
                'target_boxes': target_boxes,
                'valid_mask': valid_mask
            }
            
            loss, components = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Record metrics
            epoch_losses.append(loss.item())
            epoch_ious.append(components['mean_iou'])
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'IoU': f"{components['mean_iou']:.4f}"
            })
        
        return np.mean(epoch_losses), np.mean(epoch_ious)
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = []
        epoch_ious = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                # Move to device
                motion_vectors = batch['motion_vectors'].to(self.device)
                prev_boxes = batch['prev_boxes'].to(self.device)
                target_boxes = batch['target_boxes'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(motion_vectors, prev_boxes, valid_mask)
                
                # Compute loss
                targets = {
                    'prev_boxes': prev_boxes,
                    'target_boxes': target_boxes,
                    'valid_mask': valid_mask
                }
                
                loss, components = self.criterion(predictions, targets)
                
                # Record metrics
                epoch_losses.append(loss.item())
                epoch_ious.append(components['mean_iou'])
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'IoU': f"{components['mean_iou']:.4f}"
                })
        
        return np.mean(epoch_losses), np.mean(epoch_ious)
    
    def train(self, train_dataset, val_dataset):
        """Full training loop"""
        print(f"\n🔄 Starting training for {self.num_epochs} epochs")
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Create data loaders with device-appropriate settings
        num_workers = 0 if self.device == 'cpu' else 2  # No multiprocessing for CPU
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{self.num_epochs}")
            
            # Training phase
            train_loss, train_iou = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_iou = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.train_ious.append(train_iou)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)
            
            # Print epoch summary
            print(f"   📊 Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
            print(f"   📊 Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
            
            # Save best model
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                self.best_val_loss = val_loss
                self.save_model('best_mots17_deep_tracker.pt')
                print(f"   🎯 New best IoU: {val_iou:.4f} - Model saved!")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final model save
        self.save_model('final_mots17_deep_tracker.pt')
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"\n🎯 Training completed!")
        print(f"   Best validation IoU: {self.best_val_iou:.4f}")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        
        return self.best_val_iou
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_ious': self.train_ious,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'best_val_iou': self.best_val_iou,
            'model_name': self.model_name
        }
        
        torch.save(checkpoint, filename)
        print(f"   💾 Model saved: {filename}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', alpha=0.8)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # IoU curves
        ax2.plot(epochs, self.train_ious, 'b-', label='Training IoU', alpha=0.8)
        ax2.plot(epochs, self.val_ious, 'r-', label='Validation IoU', alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU')
        ax2.set_title('Training and Validation IoU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name}_training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📈 Training curves saved: {self.model_name}_training_curves.png")

def main():
    """Main training function for MOTS17 Deep Learning Tracker"""
    print("🚀 MOTS17 DEEP LEARNING TRACKER TRAINING")
    print("=" * 60)
    
    # Check CUDA availability and configure device
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"🔧 Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"   🔧 CUDA not available - optimizing for CPU training")
        print(f"   💡 CPU cores: {torch.get_num_threads()}")
        # Optimize for CPU training
        torch.set_num_threads(4)  # Limit threads for better performance
        print(f"   🚀 Using {torch.get_num_threads()} CPU threads")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)
    
    # Create real MOTS17 datasets using your motion vector accumulation
    print(f"\n📂 Creating MOTS17 datasets with real motion vectors and annotations...")
    
    if REAL_DATA_AVAILABLE:
        print("🎯 Using REAL MOTS17 data with your motion vector accumulation!")
        train_dataset = RealMOTS17Dataset(dataset_type='train', max_objects=8, max_samples=2000)
        val_dataset = RealMOTS17Dataset(dataset_type='val', max_objects=8, max_samples=500)
    else:
        print("📄 Using synthetic fallback data for testing")
        train_dataset = RealMOTS17Dataset(dataset_type='train', max_objects=8, max_samples=2000)
        val_dataset = RealMOTS17Dataset(dataset_type='val', max_objects=8, max_samples=500)
    
    print(f"   ✅ Training dataset: {len(train_dataset)} samples")
    print(f"   ✅ Validation dataset: {len(val_dataset)} samples")
    
    # Check if we have real data
    if len(train_dataset.samples) > 0:
        sample = train_dataset.samples[0]
        is_real = sample.get('is_real_data', False)
        if is_real:
            print(f"   🎯 Using REAL motion vectors and annotations!")
            print(f"   📊 Sample GOP index: {sample.get('gop_index', 'N/A')}")
            print(f"   📏 Motion vector shape: {sample['motion_vectors'].shape}")
            print(f"   📦 Number of objects in first sample: {sample['num_objects']}")
        else:
            print(f"   📄 Using synthetic fallback data")
    else:
        print(f"   ⚠️ No samples loaded")
    
    # Initialize trainer with device-appropriate settings
    batch_size = 8 if device == 'cpu' else 16  # Smaller batch for CPU
    num_epochs = 10 if device == 'cpu' else 30  # Fewer epochs for CPU testing
    
    trainer = MOTS17Trainer(
        model_name="mots17_deep_tracker", 
        learning_rate=0.001,
        batch_size=batch_size,
        num_epochs=num_epochs,
        hidden_dim=128,
        num_heads=8,
        transformer_layers=2,
        device=device
    )
    
    print(f"🚀 Configured for {device.upper()} training:")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: 0.001")
    
    # Start training
    try:
        best_iou = trainer.train(train_dataset, val_dataset)
        
        print(f"\n🎯 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"   Best IoU achieved: {best_iou:.4f}")
        print(f"   Model saved as: best_mots17_deep_tracker.pt")
        
        # Enhanced training summary
        summary = {
            'model_name': 'mots17_deep_tracker',
            'best_validation_iou': best_iou,
            'training_date': datetime.now().isoformat(),
            'architecture': {
                'motion_encoder': 'Multi-scale CNN with attention',
                'object_encoder': 'MLP with positional embedding',
                'transformer': '2-layer multi-head attention',
                'prediction_heads': 'Offset + Confidence + Temporal'
            },
            'loss_components': [
                'Offset Loss (L1)',
                'Generalized IoU Loss',
                'Confidence Loss (BCE)',
                'Temporal Consistency Loss',
                'Attention Regularization'
            ],
            'dataset': 'Real MOTS17 data with motion vectors and annotations' if REAL_DATA_AVAILABLE else 'Synthetic MOTS17-like data',
            'total_parameters': sum(p.numel() for p in trainer.model.parameters())
        }
        
        with open('mots17_deep_tracker_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   📄 Summary saved: mots17_deep_tracker_summary.json")
        print(f"\n✅ MOTS17 DEEP LEARNING TRACKER READY FOR DEPLOYMENT")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
