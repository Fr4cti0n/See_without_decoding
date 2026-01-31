"""Sequential dataset for GOP-based temporal MOTS training."""

import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import os
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

class SequentialMOTSDataset(Dataset):
    """Dataset for sequential GOP processing with temporal dependencies."""
    
    def __init__(self, data_path, max_samples=None, max_objects=8, sequence_length=8):
        self.data_path = data_path
        self.max_objects = max_objects
        self.sequence_length = sequence_length
        self.gop_sequences = []
        
        try:
            # Import the core data loading infrastructure (same as generate_all_plots.py)
            dataset_dir = current_dir.parent.parent / 'dataset'
            sys.path.insert(0, str(dataset_dir))
            
            # Change to dataset directory for relative imports to work
            original_cwd = os.getcwd()
            os.chdir(str(dataset_dir))
            
            from core.data_loader import MOTSDataLoaderFactory
            
            print(f"📊 Loading sequential MOTS dataset from {data_path}")
            
            # Use the same approach as generate_all_plots.py
            factory = MOTSDataLoaderFactory(verbose=False)
            sequences = factory.list_sequences(['MOT17'], ['640x640'])
            
            if not sequences:
                print("❌ No sequences found")
                os.chdir(original_cwd)  # Restore directory
                self.gop_sequences = []
                return
            
            # Use the first available sequence
            sequence_name = sequences[0]
            print(f"✅ Using sequence: {sequence_name}")
            
            # Create data loader (same as generate_all_plots.py)
            self.data_loader = factory.create_loader(sequence_name, ['MOT17'], ['640x640'])
            
            print(f"✅ Data loader created with {len(self.data_loader)} samples")
            
            # Restore original directory
            os.chdir(original_cwd)
            
            # Group samples into GOP sequences
            self._create_gop_sequences_from_data_loader()
            
            print(f"✅ Created {len(self.gop_sequences)} GOP sequences")
            
        except Exception as e:
            print(f"❌ Failed to load sequential MOTS dataset: {e}")
            import traceback
            traceback.print_exc()
            # Restore directory in case of error
            try:
                os.chdir(original_cwd)
            except:
                pass
            self.gop_sequences = []
            
    def _create_gop_sequences_from_data_loader(self):
        """Create GOP sequences from the data loader (same approach as generate_all_plots.py)."""
        # Process samples from data loader and group into GOP sequences
        sequence_groups = {}
        
        # Limit samples if specified
        max_samples_to_process = min(len(self.data_loader), self.max_samples or len(self.data_loader))
        
        for idx in range(max_samples_to_process):
            try:
                sample = self.data_loader[idx]
                if sample is None:
                    continue
                
                # Get sequence ID (same way as data loader uses)
                seq_id = sample.get('sequence_id', f'seq_{idx // 50}')  # Group by GOP size
                
                if seq_id not in sequence_groups:
                    sequence_groups[seq_id] = []
                    
                sequence_groups[seq_id].append(idx)
                
            except Exception as e:
                print(f"⚠️ Error processing sample {idx}: {e}")
                continue
        
        # Create sequential GOP groups  
        for seq_id, sample_indices in sequence_groups.items():
            # Sort indices to ensure temporal order
            sample_indices.sort()
            
            # Create sub-sequences of length sequence_length
            for i in range(0, len(sample_indices), self.sequence_length):
                subseq = sample_indices[i:i + self.sequence_length]
                if len(subseq) >= 2:  # Need at least 2 frames for temporal modeling
                    self.gop_sequences.append({
                        'sequence_id': seq_id,
                        'sample_indices': subseq,
                        'start_frame': i,
                        'length': len(subseq)
                    })
    
    def _create_gop_sequences(self):
        """Group individual samples into GOP sequences for temporal processing."""
        # Group samples by sequence ID
        sequence_groups = {}
        
        for idx, sample in enumerate(self.base_dataset.samples):
            if sample is None:
                continue
                
            # Try to get sequence info
            seq_id = getattr(sample, 'sequence_id', f'seq_{idx // 50}')  # Assume 50 frames per GOP
            
            if seq_id not in sequence_groups:
                sequence_groups[seq_id] = []
            sequence_groups[seq_id].append(idx)
        
        # Create sequential GOP groups
        for seq_id, sample_indices in sequence_groups.items():
            # Sort indices to ensure temporal order
            sample_indices.sort()
            
            # Create sub-sequences of length sequence_length
            for i in range(0, len(sample_indices), self.sequence_length):
                subseq = sample_indices[i:i + self.sequence_length]
                if len(subseq) >= 2:  # Need at least 2 frames for temporal modeling
                    self.gop_sequences.append({
                        'sequence_id': seq_id,
                        'sample_indices': subseq,
                        'start_frame': i,
                        'length': len(subseq)
                    })
    
    def __len__(self):
        return len(self.gop_sequences)
    
    def get_gop_sequence(self, gop_idx):
        """Get complete GOP sequence with all P-frames for sequential training."""
        if gop_idx >= len(self.gop_sequences):
            raise IndexError(f"GOP index {gop_idx} out of range (max: {len(self.gop_sequences)-1})")
        
        gop_sequence = []
        gop_info = self.gop_sequences[gop_idx]
        sample_indices = gop_info['sample_indices']
        
        print(f"🎬 Loading GOP {gop_idx} with {len(sample_indices)} P-frames")
        
        # Load all P-frames in the GOP sequentially using data loader
        for frame_idx, sample_idx in enumerate(sample_indices):
            try:
                # Get the sample directly from data loader (same as generate_all_plots.py)
                sample = self.data_loader[sample_idx]
                
                if sample is None:
                    continue
                
                # Process sample for deep learning using the same data structure
                processed_sample = self._process_data_loader_sample(sample)
                
                if processed_sample is not None:
                    # Add frame information
                    processed_sample['gop_idx'] = gop_idx
                    processed_sample['frame_idx'] = frame_idx
                    processed_sample['total_frames'] = len(sample_indices)
                    processed_sample['sequence_id'] = gop_info['sequence_id']
                    
                    gop_sequence.append(processed_sample)
                    
            except Exception as e:
                print(f"⚠️ Error loading P-frame {frame_idx} in GOP {gop_idx}: {e}")
                continue
        
        print(f"✅ Loaded {len(gop_sequence)} P-frames from GOP {gop_idx}")
        return gop_sequence
    
    def _process_data_loader_sample(self, sample):
        """Process data loader sample (same structure as generate_all_plots.py uses)."""
        try:
            if sample is None:
                return None
            
            # The data loader provides the same structure as used in generate_all_plots.py
            # Get motion vectors
            motion_vectors = sample.get('motion_vectors', None)
            
            if motion_vectors is None:
                return None
            
            # Convert to tensors if needed
            if isinstance(motion_vectors, np.ndarray):
                motion_vectors = torch.from_numpy(motion_vectors).float()
            elif not isinstance(motion_vectors, torch.Tensor):
                return None
            
            # Handle motion vector format - data loader provides proper format
            if len(motion_vectors.shape) == 4:  # [2, H, W, 2] format from data loader
                # Extract the motion coordinates (like generate_all_plots.py does)
                if motion_vectors.shape[0] == 2 and motion_vectors.shape[-1] == 2:
                    # Take the actual motion vectors
                    motion_vectors = motion_vectors[0, :, :, :]  # [H, W, 2] 
                    motion_vectors = motion_vectors.permute(2, 0, 1)  # [2, H, W]
            
            # Get bounding boxes from data loader format
            bounding_boxes = []
            if 'boxes' in sample and sample['boxes'] is not None:
                boxes = sample['boxes']
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                for box in boxes:
                    bounding_boxes.append({'bbox': box})
            
            # Process bounding boxes for training
            max_objects = self.max_objects
            prev_boxes = torch.zeros(max_objects, 4)
            target_boxes = torch.zeros(max_objects, 4) 
            valid_mask = torch.zeros(max_objects, dtype=torch.bool)
            
            num_objects = min(len(bounding_boxes), max_objects)
            
            for i in range(num_objects):
                bbox = bounding_boxes[i]
                if isinstance(bbox, dict) and 'bbox' in bbox:
                    box = bbox['bbox']
                    if len(box) >= 4:
                        prev_boxes[i] = torch.tensor(box[:4], dtype=torch.float32)
                        # Target with small motion for training
                        target_boxes[i] = prev_boxes[i] + torch.randn(4) * 0.1
                        valid_mask[i] = True
            
            # Get RGB frame from data loader (same as generate_all_plots.py)
            rgb_frame = None
            
            # Check different possible keys for RGB data
            for key in ['pframe', 'iframe', 'image', 'frame']:
                if key in sample and sample[key] is not None:
                    rgb_frame = sample[key]
                    break
            
            # Process RGB frame for visualization
            if rgb_frame is not None and isinstance(rgb_frame, torch.Tensor):
                rgb_frame = rgb_frame.cpu().numpy()
                
                # Handle tensor format
                if len(rgb_frame.shape) == 4:  # [B, H, W, C] 
                    rgb_frame = rgb_frame[0]  # Remove batch dimension
                
                if len(rgb_frame.shape) == 3:
                    if rgb_frame.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                        rgb_frame = rgb_frame.transpose(1, 2, 0)
                    elif rgb_frame.shape[-1] == 3:  # Already [H, W, C]
                        pass
                    
                # Normalize to 0-255 if needed
                if rgb_frame.max() <= 1.0:
                    rgb_frame = (rgb_frame * 255).astype(np.uint8)
                else:
                    rgb_frame = rgb_frame.astype(np.uint8)
            
            return {
                'motion_vectors': motion_vectors,
                'prev_boxes': prev_boxes,
                'target_boxes': target_boxes,
                'valid_mask': valid_mask,
                'num_objects': num_objects,
                'rgb_frame': rgb_frame  # Real RGB frame from data loader
            }
            
        except Exception as e:
            print(f"⚠️ Error processing data loader sample: {e}")
            return None
    
    def _process_sample_for_deep_learning(self, sample):
        """Process raw MOTS sample for deep learning format."""
        try:
            if sample is None:
                return None
            
            # Extract motion vectors and bounding boxes
            motion_vectors = sample.get('motion_vectors', None)
            bounding_boxes = sample.get('bounding_boxes', [])
            
            if motion_vectors is None:
                return None
            
            # Convert to tensors
            if isinstance(motion_vectors, np.ndarray):
                motion_vectors = torch.from_numpy(motion_vectors).float()
            elif not isinstance(motion_vectors, torch.Tensor):
                return None
            
            # Ensure correct motion vector format [2, H, W]
            if len(motion_vectors.shape) == 3 and motion_vectors.shape[0] == 2:
                pass  # Already correct format
            elif len(motion_vectors.shape) == 4:
                motion_vectors = motion_vectors.squeeze(0)  # Remove batch dimension
            else:
                print(f"⚠️ Unexpected motion vector shape: {motion_vectors.shape}")
                return None
            
            # Process bounding boxes
            max_objects = self.max_objects
            prev_boxes = torch.zeros(max_objects, 4)
            target_boxes = torch.zeros(max_objects, 4) 
            valid_mask = torch.zeros(max_objects, dtype=torch.bool)
            
            num_objects = min(len(bounding_boxes), max_objects)
            
            for i in range(num_objects):
                bbox = bounding_boxes[i]
                if isinstance(bbox, dict) and 'bbox' in bbox:
                    # Extract coordinates
                    box = bbox['bbox']
                    if len(box) >= 4:
                        prev_boxes[i] = torch.tensor(box[:4], dtype=torch.float32)
                        # For training, target is slightly modified previous box (simulating motion)
                        target_boxes[i] = prev_boxes[i] + torch.randn(4) * 0.1
                        valid_mask[i] = True
            
            # Try to get RGB frame for visualization (P-frame)
            rgb_frame = None
            if 'pframe' in sample and sample['pframe'] is not None:
                rgb_frame = sample['pframe']
            elif 'iframe' in sample and sample['iframe'] is not None:
                rgb_frame = sample['iframe']
            elif 'image' in sample and sample['image'] is not None:
                rgb_frame = sample['image']
            elif 'frame' in sample and sample['frame'] is not None:
                rgb_frame = sample['frame']
            
            # If RGB frame is a tensor, convert to numpy for visualization
            if rgb_frame is not None and isinstance(rgb_frame, torch.Tensor):
                rgb_frame = rgb_frame.cpu().numpy()
                # Handle different tensor formats
                if len(rgb_frame.shape) == 4:  # [B, H, W, C] or [B, C, H, W]
                    rgb_frame = rgb_frame[0]  # Remove batch dimension
                
                if len(rgb_frame.shape) == 3:
                    if rgb_frame.shape[0] == 3:  # [C, H, W] -> [H, W, C]
                        rgb_frame = rgb_frame.transpose(1, 2, 0)
                    elif rgb_frame.shape[-1] == 3:  # Already [H, W, C]
                        pass
                    
                # Normalize to 0-255 if needed
                if rgb_frame.max() <= 1.0:
                    rgb_frame = (rgb_frame * 255).astype(np.uint8)
                else:
                    rgb_frame = rgb_frame.astype(np.uint8)
            
            return {
                'motion_vectors': motion_vectors,
                'prev_boxes': prev_boxes,
                'target_boxes': target_boxes,
                'valid_mask': valid_mask,
                'num_objects': num_objects,
                'rgb_frame': rgb_frame  # For visualization
            }
            
        except Exception as e:
            print(f"⚠️ Error processing sample: {e}")
            return None
    
    def __getitem__(self, idx):
        """Get a sequential GOP sample."""
        try:
            gop_info = self.gop_sequences[idx]
            sample_indices = gop_info['sample_indices']
            
            # Get the first sample as reference
            first_sample = self.base_dataset[sample_indices[0]]
            if first_sample is None:
                return self._get_dummy_sequence(idx)
            
            # For now, return single frame but structured for sequence processing
            # Later this can be extended to return full sequences
            
            motion_vectors = first_sample.get('motion_vectors', torch.zeros(2, 60, 60))
            prev_boxes = first_sample.get('prev_boxes', torch.zeros(self.max_objects, 4))
            target_boxes = first_sample.get('target_boxes', torch.zeros(self.max_objects, 4))
            valid_mask = first_sample.get('valid_mask', torch.zeros(self.max_objects, dtype=torch.bool))
            
            # Ensure correct motion vector format
            if motion_vectors.dim() == 4:  # [2, H, W, 2] format
                motion_vectors = motion_vectors.mean(dim=0).permute(2, 0, 1)  # -> [2, H, W]
            elif motion_vectors.dim() == 3 and motion_vectors.shape[-1] == 2:
                motion_vectors = motion_vectors.permute(2, 0, 1)  # [H, W, 2] -> [2, H, W]
            
            # Resize to standard size if needed
            if motion_vectors.shape[1:] != (40, 40):
                motion_vectors = torch.nn.functional.interpolate(
                    motion_vectors.unsqueeze(0), 
                    size=(40, 40), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            return {
                'motion_vectors': motion_vectors.float(),
                'prev_boxes': prev_boxes.float(),
                'target_boxes': target_boxes.float(),
                'valid_mask': valid_mask,
                'sequence_id': gop_info['sequence_id'],
                'gop_frame': gop_info['start_frame'],
                'sequence_length': gop_info['length']
            }
            
        except Exception as e:
            print(f"⚠️ Error loading GOP sequence {idx}: {e}")
            return self._get_dummy_sequence(idx)
    
    def _get_dummy_sequence(self, idx):
        """Return dummy data for fallback."""
        return {
            'motion_vectors': torch.zeros(2, 40, 40),
            'prev_boxes': torch.zeros(self.max_objects, 4),
            'target_boxes': torch.zeros(self.max_objects, 4),
            'valid_mask': torch.zeros(self.max_objects, dtype=torch.bool),
            'sequence_id': f'dummy_{idx}',
            'gop_frame': 0,
            'sequence_length': 1
        }
