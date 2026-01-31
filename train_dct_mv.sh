#!/bin/bash
# Training script for DCT-MV Model (proven working architecture)

echo "🚀 Training DCT-MV Model (Motion Vectors + DCT Residuals)"
echo "=============================================================="
echo ""
echo "Architecture:"
echo "  - Dual encoder: Motion vectors (64-dim) + DCT residuals (64-dim)"
echo "  - Combined features: 128-dim"
echo "  - LSTM tracker: 128 hidden dim"
echo "  - Proven stable training"
echo ""
echo "Dataset:"
echo "  - MOT15 + MOT17 + MOT20 (auto-loaded, combined)"
echo "  - Motion vectors + DCT residuals"
echo "  - GOP length: 47 frames"
echo "  - Resolution: 960×960"
echo ""
echo "=============================================================="
echo ""

# Training configuration
EPOCHS=30
BATCH_SIZE=8
LEARNING_RATE=1e-3
FEATURE_DIM=64
HIDDEN_DIM=128
MAX_OBJECTS=100
GOP_LENGTH=47
RESOLUTION=960

# Output directory
OUTPUT_DIR="experiments/dct_mv_30epochs"
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/training.log"

echo "Starting training..."
echo "Output: $OUTPUT_DIR"
echo ""

# Run training with DCT-MV model
python mots_exp/scripts/train_mv_center.py \
    --use-dct \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --feature-dim $FEATURE_DIM \
    --hidden-dim $HIDDEN_DIM \
    --max-objects $MAX_OBJECTS \
    --gop-length $GOP_LENGTH \
    --resolution $RESOLUTION \
    --output-dir "$OUTPUT_DIR" \
    --save-freq 10 \
    --log-freq 10 \
    --num-workers 4 \
    --device cuda \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "✅ Training complete!"
echo "=============================================================="
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo "📝 Log file: $LOG_FILE"
echo ""
echo "📊 Output files:"
echo "   - Training curves: $OUTPUT_DIR/training_curves.png"
echo "   - Best checkpoint: $OUTPUT_DIR/best_model.pt"
echo "   - Final checkpoint: $OUTPUT_DIR/checkpoint_epoch_${EPOCHS}.pt"
echo ""
echo "🚀 To resume training or run inference, use these checkpoints!"
