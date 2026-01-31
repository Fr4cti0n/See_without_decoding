#!/bin/bash

# Training Comparison Script
# This script trains two models for comparison:
# 1. Baseline model (2-channel: X, Y only)
# 2. Magnitude-enhanced model (3-channel: X, Y, Magnitude)

# Activate virtual environment
source /home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/YOLOv11-pt/YOLO/bin/activate

echo "========================================"
echo "  MOTION VECTOR MODEL COMPARISON"
echo "========================================"
echo ""
echo "This script will train TWO models:"
echo "  1️⃣  Baseline: 2-channel motion vectors (X, Y)"
echo "  2️⃣  Enhanced: 3-channel motion vectors (X, Y, Magnitude)"
echo ""
echo "Each model will be trained for 10 epochs"
echo ""
echo "Checkpoints will be saved as:"
echo "  - checkpoint_epoch_N_baseline.pt (for baseline)"
echo "  - checkpoint_epoch_N_magnitude.pt (for enhanced)"
echo ""
read -p "Press ENTER to start training..."

# Set common parameters
EPOCHS=10
LEARNING_RATE=0.001

echo ""
echo "========================================"
echo "  PHASE 1: Training BASELINE Model"
echo "  (2-channel: X, Y only)"
echo "========================================"
echo ""

python mots_exp/scripts/train.py \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE

if [ $? -ne 0 ]; then
    echo "❌ Baseline training failed!"
    exit 1
fi

echo ""
echo "✅ Baseline model training completed!"
echo ""
echo "========================================"
echo "  PHASE 2: Training MAGNITUDE Model"
echo "  (3-channel: X, Y, Magnitude)"
echo "========================================"
echo ""

python mots_exp/scripts/train.py \
    --epochs $EPOCHS \
    --learning-rate $LEARNING_RATE \
    --use-magnitude

if [ $? -ne 0 ]; then
    echo "❌ Magnitude model training failed!"
    exit 1
fi

echo ""
echo "✅ Both models trained successfully!"
echo ""
echo "========================================"
echo "  TRAINING COMPLETE"
echo "========================================"
echo ""
echo "📊 Checkpoint files created:"
ls -lh outputs/checkpoint_epoch_*_baseline.pt outputs/checkpoint_epoch_*_magnitude.pt 2>/dev/null
echo ""
echo "🔍 Next steps:"
echo "  1. Compare model sizes and parameter counts"
echo "  2. Run validation on both models"
echo "  3. Compare mAP scores to see magnitude benefit"
echo ""
echo "To validate the models, run:"
echo "  python mots_exp/scripts/validate_motion_map.py --model-path outputs/checkpoint_epoch_10_baseline.pt"
echo "  python mots_exp/scripts/validate_motion_map.py --model-path outputs/checkpoint_epoch_10_magnitude.pt"
echo ""
