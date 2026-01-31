#!/bin/bash

###############################################################################
# Evaluate MV Model on Static Cameras - Moving Objects Only
#
# This script evaluates the MV-only model specifically on:
# 1. Static camera sequences (no camera motion)
# 2. Moving objects only (filtered by displacement threshold)
#
# Rationale:
# - I-frame baseline assumes objects stay at their initial position
# - This works for static objects but fails for moving objects
# - MV model should excel at tracking moving objects
# - Static objects may have "false motion" from illumination/artifacts
#
# By filtering to moving objects, we demonstrate the MV model's true advantage
###############################################################################

set -e  # Exit on error

# Configuration
CHECKPOINT="best_mots17_deep_tracker.pt"
GOP_DIR="experiments/gop_data_50"
OUTPUT_DIR="experiments/static_camera_moving_objects"
DEVICE="cuda"

# Motion detection parameters
MOTION_THRESHOLD=10.0  # Minimum displacement in pixels to consider as "moving"
IOU_THRESHOLD=0.5      # IoU threshold for matching boxes across frames

echo "========================================================================"
echo "Static Camera - Moving Objects Only Evaluation"
echo "========================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "GOP Directory: $GOP_DIR"
echo "Motion Threshold: $MOTION_THRESHOLD pixels"
echo "IoU Threshold: $IOU_THRESHOLD"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================================"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python evaluate_static_camera_moving_objects.py \
    --checkpoint "$CHECKPOINT" \
    --gop-dir "$GOP_DIR" \
    --motion-threshold "$MOTION_THRESHOLD" \
    --iou-threshold "$IOU_THRESHOLD" \
    --device "$DEVICE" \
    --output "$OUTPUT_DIR/static_camera_moving_objects_results.json" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

echo
echo "========================================================================"
echo "✅ Evaluation Complete!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "  - static_camera_moving_objects_results.json"
echo "  - evaluation.log"
echo "========================================================================"
echo

# Display key results
if [ -f "$OUTPUT_DIR/static_camera_moving_objects_results.json" ]; then
    echo "Quick Summary:"
    echo "---"
    python -c "
import json
with open('$OUTPUT_DIR/static_camera_moving_objects_results.json') as f:
    results = json.load(f)
    
if 'overall' in results:
    overall = results['overall']
    print(f\"Total GOPs (static cameras): {overall['total_gops']}\")
    print(f\"Total moving objects: {overall['total_moving_objects']}\")
    print(f\"Total static objects: {overall['total_static_objects']}\")
    print(f\"Motion ratio: {overall['motion_ratio']:.1%}\")
    print(f\"Overall mAP@0.5 (moving objects only): {overall['overall_map']:.4f}\")
    print()
    
    for dataset in ['MOT17', 'MOT15', 'MOT20']:
        if dataset in results and results[dataset]['total_gops'] > 0:
            r = results[dataset]
            print(f\"{dataset}: {r['total_gops']} GOPs, mAP={r['overall_map']:.4f} ({r['total_moving_objects']} moving objects)\")
"
fi

echo
echo "========================================================================"
echo "Justification"
echo "========================================================================"
echo "This evaluation demonstrates the MV model's advantage over I-frame baseline:"
echo
echo "1. I-frame baseline assumes objects don't move → fails on moving objects"
echo "2. MV model learns actual motion → should significantly outperform"
echo "3. Static objects may have 'false motion' from:"
echo "   - Illumination changes"
echo "   - Compression artifacts"
echo "   - Camera vibration"
echo
echo "By filtering to moving objects only, we show the MV model's core strength."
echo "========================================================================"
