"""Training configuration for sequential GOP processing."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SequentialTrainConfig:
    """Configuration for sequential GOP training."""
    model_name: str = "id_aware_multi_object_tracker"
    epochs: int = 10  # Train for 10 epochs with detailed loss tracking
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: Optional[str] = None  # Auto-detect
    seed: int = 42
    
    # Sequential processing parameters
    gop_length: int = 48  # Full GOP length for complete P-frame processing
    max_objects: int = 100  # Increased to handle more objects without parameter explosion
    sequence_length: int = 8  # Number of frames to process in sequence
    
    # Data parameters
    data_path: str = "/home/aduche/Bureau/datasets/MOTS/videos"
    max_samples: int = 10  # Train on more GOPs for better results
    val_samples: int = 50
    
    # Output
    output_dir: str = "outputs"
    save_best: bool = True
    save_final: bool = True

    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments."""
        cfg = cls()
        
        if hasattr(args, 'epochs') and args.epochs is not None:
            cfg.epochs = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            cfg.learning_rate = args.learning_rate
        if hasattr(args, 'max_samples') and args.max_samples is not None:
            cfg.max_samples = args.max_samples
            
        return cfg
