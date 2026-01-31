#!/bin/bash
# Example training scripts for Fast DCT-MV architectures

# Set base parameters
BASE_DIR="experiments/ablation_validation"
EPOCHS=50
BATCH_SIZE=8
LR=1e-3
MAX_GOPS=70
MAX_VAL_GOPS=30

echo "🚀 Fast Architecture Training Examples"
echo "========================================"

# 1. Standard model with MV+DCT-8 (baseline for comparison)
echo ""
echo "1️⃣  Training Standard Model (MV+DCT-8) - Baseline"
python mots_exp/scripts/train_mv_center.py \
  --use-dct \
  --dct-coeffs 8 \
  --use-parallel-heads \
  --use-attention \
  --attention-heads 4 \
  --use-detection-loss \
  --box-weight 5.0 \
  --giou-weight 2.0 \
  --class-weight 2.0 \
  --learning-rate ${LR} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --max-gops ${MAX_GOPS} \
  --max-val-gops ${MAX_VAL_GOPS} \
  --device cuda \
  --output-dir ${BASE_DIR}/mv_dct_8_standard

# 2. Fast model with MV+DCT-8
echo ""
echo "2️⃣  Training Fast Model (MV+DCT-8) - 2.9x faster"
python mots_exp/scripts/train_mv_center.py \
  --use-dct \
  --dct-coeffs 8 \
  --use-fast \
  --use-detection-loss \
  --box-weight 5.0 \
  --giou-weight 2.0 \
  --class-weight 2.0 \
  --learning-rate ${LR} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --max-gops ${MAX_GOPS} \
  --max-val-gops ${MAX_VAL_GOPS} \
  --device cuda \
  --output-dir ${BASE_DIR}/mv_dct_8_fast

# 3. Ultra-Fast model with MV+DCT-8
echo ""
echo "3️⃣  Training Ultra-Fast Model (MV+DCT-8) - 4.1x faster"
python mots_exp/scripts/train_mv_center.py \
  --use-dct \
  --dct-coeffs 8 \
  --use-ultra-fast \
  --use-detection-loss \
  --box-weight 5.0 \
  --giou-weight 2.0 \
  --class-weight 2.0 \
  --learning-rate 2e-3 \
  --epochs ${EPOCHS} \
  --batch-size 16 \
  --max-gops ${MAX_GOPS} \
  --max-val-gops ${MAX_VAL_GOPS} \
  --device cuda \
  --output-dir ${BASE_DIR}/mv_dct_8_ultra_fast

# 4. Fast model with MV-only
echo ""
echo "4️⃣  Training Fast Model (MV-only)"
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
  --max-gops ${MAX_GOPS} \
  --max-val-gops ${MAX_VAL_GOPS} \
  --device cuda \
  --output-dir ${BASE_DIR}/mv_only_fast

# 5. Fast model with DCT-only
echo ""
echo "5️⃣  Training Fast Model (DCT-8 only, no MV)"
python mots_exp/scripts/train_mv_center.py \
  --use-dct \
  --dct-coeffs 8 \
  --no-mv \
  --use-fast \
  --use-detection-loss \
  --box-weight 5.0 \
  --giou-weight 2.0 \
  --class-weight 2.0 \
  --learning-rate ${LR} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --max-gops ${MAX_GOPS} \
  --max-val-gops ${MAX_VAL_GOPS} \
  --device cuda \
  --output-dir ${BASE_DIR}/dct_8_fast

echo ""
echo "✅ Training scripts completed!"
echo ""
echo "📊 To compare models, use:"
echo "python mots_exp/scripts/compare_ablation_performance.py"
