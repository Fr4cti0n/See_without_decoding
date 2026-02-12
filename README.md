# See Without Decoding (GitHub Repository)

This repository hosts the codebase for the paper "See Without Decoding." BAFE (Box-Aligned Feature Extraction) is a core component used in the paper, and the details below remain the primary reference for that module.

**Status:** Accepted to ESANN 2026.

# BAFE Model (Box-Aligned Feature Extraction)

## What is BAFE?
BAFE (Box-Aligned Feature Extraction) is the tracking/detection head in this project that extracts fixed-grid features aligned to predicted bounding boxes. It pairs motion-vector streams (MV) with DCT residual features to keep spatial structure without ROI pooling. Typical tensors used in this repo:
- BAFE-MV: N x 15 x 15 x 2 motion-vector crops
- BAFE-DCT: N x 30 x 30 x 32 DCT feature crops

## Key Ideas
- Extract fixed-size grids per box for both MV and DCT branches.
- Preserve spatial layout inside each box to stabilize motion reasoning.
- Lightweight extraction stage (~tens of thousands of params) intended for fast MOTS experiments.

## Where to look in this repo
- Architecture diagrams and generator: pdf_latex/generate_architecture_roi_clear.py and pdf_latex/bafe_architecture_fg0.tex
- End-to-end experiments and training scripts: mots_exp/scripts/ (e.g., train_mv_center.py) and root-level train_*.sh helpers.
- Evaluation helpers: evaluate_* scripts under the repo root.

## Quickstart (common pattern)
1) Activate the virtualenv used for these experiments (example):
```
source .venv/bin/activate
```
2) Run a training script that uses the BAFE extraction (adjust flags/paths as needed):
```
python mots_exp/scripts/train_mv_center.py \
  --data_root ./dataset \
  --exp_name bafe_mv_center_run
```
3) Evaluate a trained checkpoint with one of the evaluation scripts:
```
python evaluate_mv_only_by_dataset.py --checkpoint checkpoints/your_model.pt
```

## Tips
- Keep the MV and DCT crops synchronized; mismatched grid sizes will break the extraction stage.
- For speed/latency comparisons, see benchmark_fast_models.sh and benchmark_model_speed.py.

## Outputs
- Checkpoints land in checkpoints/ unless overridden.
- Logs are saved alongside training scripts (e.g., mv_center_training.log) for quick inspection.
