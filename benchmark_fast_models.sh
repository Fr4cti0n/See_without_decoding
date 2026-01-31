#!/bin/bash

# 🔄 RETRAIN IMPROVED FAST MV-ONLY MODEL
# =====================================
# This script retrains the MV-only model with the IMPROVED architecture that fixes
# the critical global pooling issue identified in the performance analysis.
#
# KEY CHANGES FROM OLD VERSION:
# 1. ✅ Uses ImprovedFastDCTMVTracker instead of FastDCTMVTracker
# 2. ✅ Box-aligned motion feature extraction (NOT global pooling!)
# 3. ✅ Trains on GOP-50 (matches evaluation, not GOP-10)
# 4. ✅ Per-box motion statistics (mean + variance + range)
# 5. ✅ Respects 16×16 macroblock structure
#
# EXPECTED IMPROVEMENT:
# - Old model (global pooling): 0.41 mAP on moving objects
# - Mean-VC baseline: 0.73 mAP on moving objects
# - New model (box-aligned): Should BEAT baseline (>0.75 mAP expected)

echo "🔄 RETRAINING IMPROVED FAST MV-ONLY MODEL"
echo "=========================================="
echo ""
echo "📋 Architecture Changes:"
echo "   ❌ OLD: Global pooling → destroys spatial information"
echo "   ✅ NEW: Box-aligned motion encoder → per-box features"
echo ""
echo "📋 Training Configuration:"
echo "   ✅ GOP length: 50 frames (matches evaluation)"
echo "   ✅ Per-box MV statistics: mean + std + range"
echo "   ✅ Respects 16×16 macroblock grid"
echo ""

# Configuration
OUTPUT_DIR="experiments/ablation_fast_improved/mv_only"
EPOCHS=50
BATCH_SIZE=8
LR=1e-3
GOP_LENGTH=50          # ✅ CRITICAL: Match evaluation (was 10 before!)
MAX_GOPS=70
MAX_VAL_GOPS=30

# Backup old model if it exists
OLD_MODEL_DIR="experiments/ablation_fast/mv_only"
if [ -d "$OLD_MODEL_DIR" ]; then
    BACKUP_DIR="${OLD_MODEL_DIR}_old_global_pooling_backup"
    if [ ! -d "$BACKUP_DIR" ]; then
        echo "💾 Backing up old model (global pooling version)..."
        cp -r "$OLD_MODEL_DIR" "$BACKUP_DIR"
        echo "   ✅ Backup saved to: $BACKUP_DIR"
    fi
fi

echo ""
echo "🎯 Starting Training..."
echo "   Output directory: $OUTPUT_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LR"
echo "   GOP length: $GOP_LENGTH frames"
echo "   Max training GOPs: $MAX_GOPS"
echo "   Max validation GOPs: $MAX_VAL_GOPS"
echo ""

# Activate virtual environment
VENV_PATH="/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/YOLOv11-pt/YOLO/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "🔧 Activating virtual environment..."
    source "$VENV_PATH"
    echo "   ✅ Virtual environment activated"
else
    echo "⚠️  Warning: Virtual environment not found at $VENV_PATH"
    echo "   Proceeding with system Python..."
fi

# Train the improved model
# NOTE: The training script will automatically use ImprovedFastDCTMVTracker
# when --use-fast is specified (we updated the model factory)
echo ""
echo "🚀 Training MV-only model with BOX-ALIGNED motion features..."
echo ""

python mots_exp/scripts/train_mv_center.py \
  --use-dct \
  --dct-coeffs 0 \
  --use-fast \
  --use-detection-loss \
  --box-weight 5.0 \
  --giou-weight 2.0 \
  --class-weight 2.0 \
  --learning-rate ${LR} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --gop-length ${GOP_LENGTH} \
  --max-gops ${MAX_GOPS} \
  --max-val-gops ${MAX_VAL_GOPS} \
  --device cuda \
  --output-dir ${OUTPUT_DIR} \
  --save-freq 5 \
  --log-freq 10

TRAIN_EXIT_CODE=$?

echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Model saved to: ${OUTPUT_DIR}/best_model.pt"
    echo ""
    echo "📈 Training results:"
    if [ -f "${OUTPUT_DIR}/training_results.json" ]; then
        python -c "
import json
with open('${OUTPUT_DIR}/training_results.json', 'r') as f:
    results = json.load(f)
print(f\"   Best validation mAP: {results.get('best_val_map', 'N/A'):.4f}\")
print(f\"   Best epoch: {results.get('best_epoch', 'N/A')}\")
print(f\"   Final loss: {results.get('final_loss', 'N/A'):.4f}\")
"
    fi
    echo ""
    echo "� Next steps:"
    echo "   1. Evaluate on test set: ./run_three_method_comparison.sh"
    echo "   2. Compare with old model performance:"
    echo "      - Old (global pooling): ~0.41 mAP on moving objects"
    echo "      - Mean-VC baseline: ~0.73 mAP on moving objects"
    echo "      - Expected improvement: >0.75 mAP on moving objects"
    echo ""
else
    echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
    echo ""
    echo "🔍 Check the error messages above for details"
    echo "   Common issues:"
    echo "   - CUDA out of memory → reduce batch size"
    echo "   - Data loading errors → check dataset paths"
    echo "   - Model architecture errors → verify ImprovedFastDCTMVTracker exists"
    exit $TRAIN_EXIT_CODE
fi

# Run evaluation on the new model
echo ""
echo "=========================================="
echo "📊 EVALUATION ON TEST SET"
echo "=========================================="
echo ""
echo "🔬 Running three-method comparison..."
echo "   This will compare:"
echo "   1. Static I-frame baseline"
echo "   2. Mean-VC baseline (autoregressive)"
echo "   3. NEW Improved MV model (box-aligned features)"
echo ""

# Update the comparison script to use the new model
COMPARISON_SCRIPT="run_three_method_comparison.sh"
if [ -f "$COMPARISON_SCRIPT" ]; then
    # Temporarily modify the comparison script to use new model path
    sed -i.bak "s|experiments/ablation_fast/mv_only|${OUTPUT_DIR}|g" "$COMPARISON_SCRIPT"
    
    # Run evaluation
    ./"$COMPARISON_SCRIPT"
    EVAL_EXIT_CODE=$?
    
    # Restore original comparison script
    mv "${COMPARISON_SCRIPT}.bak" "$COMPARISON_SCRIPT"
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Evaluation completed!"
        echo ""
        echo "� Results saved to: results/three_method_comparison.json"
        echo ""
        echo "🎯 Performance Comparison:"
        if [ -f "results/three_method_comparison.json" ]; then
            python -c "
import json
with open('results/three_method_comparison.json', 'r') as f:
    results = json.load(f)

for dataset, data in results.items():
    if dataset == 'MOT17':  # Show MOT17 as example
        print(f'\n{dataset} Results (Moving Objects):')
        print(f\"   I-frame:  {data['iframe']['moving_map']:.4f} ± {data['iframe']['moving_std']:.4f}\")
        print(f\"   Mean-VC:  {data['mean_vc']['moving_map']:.4f} ± {data['mean_vc']['moving_std']:.4f}\")
        print(f\"   MV Model: {data['mv_model']['moving_map']:.4f} ± {data['mv_model']['moving_std']:.4f}\")
        
        improvement = data['mv_model']['moving_map'] - data['mean_vc']['moving_map']
        if improvement > 0:
            print(f\"   ✅ IMPROVEMENT: +{improvement:.4f} ({improvement/data['mean_vc']['moving_map']*100:.1f}%)\")
        else:
            print(f\"   ⚠️  Still behind baseline: {improvement:.4f}\")
"
        fi
    else
        echo "⚠️  Evaluation results not found"
    fi
else
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
fi

echo ""
echo "=========================================="
echo "🎉 TRAINING & EVALUATION COMPLETE"
echo "=========================================="
echo ""
echo "📁 Model Location: ${OUTPUT_DIR}/best_model.pt"
echo "📁 Training Results: ${OUTPUT_DIR}/training_results.json"
echo "📁 Evaluation Results: results/three_method_comparison.json"
echo ""
echo "🔍 Architecture Improvements Applied:"
echo "   ✅ Removed global pooling (was destroying spatial info)"
echo "   ✅ Added box-aligned motion feature extraction"
echo "   ✅ Per-box MV statistics: mean + std + range"
echo "   ✅ GOP-50 training (matches evaluation length)"
echo "   ✅ Respects 16×16 macroblock structure"
echo ""
echo "📊 Expected Performance Gain:"
echo "   Old model (global pooling): ~0.41 mAP"
echo "   Mean-VC baseline: ~0.73 mAP"
echo "   Target (box-aligned): >0.75 mAP"
echo ""
echo "📝 If model still underperforms:"
echo "   1. Check training logs for convergence issues"
echo "   2. Verify BoxAlignedMotionEncoder is being used"
echo "   3. Monitor per-object tracking accuracy during validation"
echo "   4. Consider adjusting learning rate or batch size"
echo ""
