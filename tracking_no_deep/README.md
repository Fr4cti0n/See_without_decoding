# MOTS Experiments Framework

## Overview
This is a complete experimental framework for Multiple Object Tracking and Segmentation (MOTS) research using motion vectors and temporal dependencies.

## Structure

```
MOTS-experiments/
├── __init__.py                    # Main package initialization
├── README.md                      # This file
├── training/                      # Training scripts and models
│   ├── __init__.py               # Training package initialization
│   ├── train_optimized_enhanced_offset_tracker.py  # Main training script
│   └── optimized_enhanced_offset_tracker.py        # Model definitions
└── dataset/                       # Dataset utilities
    ├── __init__.py               # Dataset package initialization
    ├── tracking_dataset.py       # GOP-based tracking dataset
    ├── dataset_mots.py          # Base MOTS dataset loader
    └── [other dataset utilities]
```

## Quick Start

### 1. Training the Model
```bash
cd MOTS-experiments/training
python train_optimized_enhanced_offset_tracker.py --epochs 100
```

### 2. Key Features
- **Temporal Memory System**: 70% prediction propagation through GOP sequences
- **Optimized Architecture**: 372K parameters (within 500K limit)
- **Multi-Scale Processing**: Motion vector encoding with object interactions
- **Real Data**: Uses actual H.264 motion vectors from MOTS dataset

### 3. Model Components
- **OptimizedEnhancedOffsetTracker**: Main model (372K params)
- **MOTSDataAdapter**: Data conversion with temporal memory
- **OptimizedEnhancedLoss**: Multi-component loss (IoU + offset + confidence)

### 4. Dataset Integration
- **MOTSTrackingDataset**: GOP sequence processing
- **Motion Vectors**: Real H.264 compression data
- **Annotations**: MOTS ground truth with temporal consistency

## Dependencies
The framework automatically handles import paths between training and dataset modules. All dependencies are resolved at runtime.

## Technical Details
- **GOP Structure**: I-frame supervision + P-frame prediction propagation
- **Resolution**: 640x640 processing with motion vector downsampling
- **Temporal Memory**: Video-level sequence tracking with 70% prediction ratio
- **Training Strategy**: Mixed GT/prediction training for stability

## Results
The system achieves temporal consistency through external training-level dependencies rather than internal model complexity, making it efficient for video compression scenarios.
