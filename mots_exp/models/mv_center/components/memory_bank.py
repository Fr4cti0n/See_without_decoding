"""
Object Memory Bank for MV-Center Tracking

Stores and manages object states across frames in a GOP.
"""

import torch
import torch.nn as nn


class ObjectMemoryBank(nn.Module):
    """
    Memory bank to store object states across a GOP sequence.
    
    Stores:
    - positions: [N, 4] bounding boxes in [cx, cy, w, h] format
    - features: [N, D] learned feature embeddings
    - velocities: [N, 2] motion velocities [vx, vy]
    - active_mask: [N] boolean mask for active objects
    - object_ids: [N] unique object IDs
    """
    
    def __init__(self, max_objects=100, feature_dim=256):
        super().__init__()
        
        self.max_objects = max_objects
        self.feature_dim = feature_dim
        
        # Initialize storage buffers
        self.register_buffer('positions', torch.zeros(max_objects, 4))
        self.register_buffer('features', torch.zeros(max_objects, feature_dim))
        self.register_buffer('velocities', torch.zeros(max_objects, 2))
        self.register_buffer('active_mask', torch.zeros(max_objects, dtype=torch.bool))
        self.register_buffer('object_ids', torch.zeros(max_objects, dtype=torch.long))
        self.register_buffer('num_active', torch.tensor(0, dtype=torch.long))
    
    def reset(self):
        """Clear all stored objects."""
        self.positions.zero_()
        self.features.zero_()
        self.velocities.zero_()
        self.active_mask.zero_()
        self.object_ids.zero_()
        self.num_active.zero_()
    
    def init_from_boxes(self, boxes, object_ids=None, initial_features=None):
        """
        Initialize memory from I-frame bounding boxes.
        
        Args:
            boxes: [N, 4] tensor of boxes in [cx, cy, w, h] format
            object_ids: [N] tensor of object IDs (optional, will auto-assign)
            initial_features: [N, D] tensor of initial features (optional)
        """
        num_objects = min(len(boxes), self.max_objects)
        
        # Reset first
        self.reset()
        
        # Store boxes
        self.positions[:num_objects] = boxes[:num_objects]
        
        # Initialize velocities to zero (stationary)
        self.velocities[:num_objects] = 0.0
        
        # Set active mask
        self.active_mask[:num_objects] = True
        self.num_active = torch.tensor(num_objects, dtype=torch.long)
        
        # Assign IDs
        if object_ids is not None:
            self.object_ids[:num_objects] = object_ids[:num_objects]
        else:
            # Auto-assign sequential IDs
            self.object_ids[:num_objects] = torch.arange(num_objects, dtype=torch.long)
        
        # Store features if provided
        if initial_features is not None:
            self.features[:num_objects] = initial_features[:num_objects]
    
    def update_positions(self, indices, new_positions):
        """
        Update positions for specific objects.
        
        Args:
            indices: [K] tensor of object indices to update
            new_positions: [K, 4] tensor of new positions
        """
        # Use .data to avoid gradient tracking issues
        self.positions.data[indices] = new_positions.detach()
    
    def update_velocities(self, indices, new_velocities):
        """
        Update velocities for specific objects.
        
        Args:
            indices: [K] tensor of object indices to update
            new_velocities: [K, 2] tensor of new velocities
        """
        # Use .data to avoid gradient tracking issues
        self.velocities.data[indices] = new_velocities.detach()
    
    def update_features(self, indices, new_features):
        """
        Update features for specific objects.
        
        Args:
            indices: [K] tensor of object indices to update
            new_features: [K, D] tensor of new features
        """
        # Use .data to avoid gradient tracking issues
        self.features.data[indices] = new_features.detach()
    
    def get_active_objects(self):
        """
        Get all active object states.
        
        Returns:
            Dict with:
                - positions: [N_active, 4] (detached)
                - velocities: [N_active, 2] (detached)
                - features: [N_active, D] (detached)
                - ids: [N_active]
                - indices: [N_active] original indices in memory
        """
        active_indices = torch.where(self.active_mask)[0]
        
        return {
            'positions': self.positions[active_indices].clone(),  # Clone to avoid in-place issues
            'velocities': self.velocities[active_indices].clone(),
            'features': self.features[active_indices].clone(),
            'ids': self.object_ids[active_indices],
            'indices': active_indices
        }
    
    def deactivate_objects(self, indices):
        """Mark objects as inactive (e.g., left scene)."""
        self.active_mask[indices] = False
        self.num_active = self.active_mask.sum()
    
    def activate_objects(self, indices):
        """Mark objects as active (e.g., re-entered scene)."""
        self.active_mask[indices] = True
        self.num_active = self.active_mask.sum()
    
    def __len__(self):
        """Return number of active objects."""
        return self.num_active.item()
    
    def __repr__(self):
        return f"ObjectMemoryBank(max={self.max_objects}, active={len(self)}, feature_dim={self.feature_dim})"
