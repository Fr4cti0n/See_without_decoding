from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class TrainConfig:
    model_name: str = "optimized_enhanced_offset_tracker"
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    device: Optional[str] = None  # Auto-select if None
    seed: int = 42
    num_workers: int = 0
    shuffle: bool = False  # Keep temporal ordering by default
    skip_verification: bool = True
    output_dir: str = "outputs"
    save_best: bool = True
    save_final: bool = True
    early_stop_patience: int = 200
    # Future extensions
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetConfig:
    dataset_type: str = "mot17"
    resolution: int = 640
    sequence_length: int = 8
    max_objects: int = 100
    use_temporal_memory: bool = True
    memory_ratio: float = 0.7

@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DatasetConfig = field(default_factory=DatasetConfig)

    @staticmethod
    def from_args(args: Any) -> "Config":
        cfg = Config()
        if hasattr(args, 'epochs') and args.epochs is not None:
            cfg.train.epochs = args.epochs
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cfg.train.batch_size = args.batch_size
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            cfg.train.learning_rate = args.learning_rate
        if hasattr(args, 'model') and args.model is not None:
            cfg.train.model_name = args.model
        return cfg
