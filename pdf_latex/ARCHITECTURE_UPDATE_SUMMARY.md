# Architecture Diagram Update - esannV2 Notation

## Summary of Changes

Successfully regenerated the Fast DCT-MV architecture diagram and documentation using consistent mathematical notation from `esannV2.tex` Section 3.

## Files Created/Updated

### 1. Architecture Generation Script
**File**: `generate_architecture_esann.py`

**Changes**:
- Uses esannV2.tex notation from Section 3
- **Fixed overlapping issues**:
  - Increased spacing between Conv layers (from 0.5 to 1.0 units)
  - Increased spacing between major components (2.5-3.5 units)
  - Reduced box sizes slightly (35→30 for conv, 40→35 for inputs)
  - Removed buggy LSTM skip arrow that was displaying incorrectly
  - **Moved dimension labels to captions** (no more s_filer/n_filer overlap)
- Captions now include dimensions in parentheses: "Motion Vectors (2)", "Conv 3x3 (256)", etc.
- Better horizontal layout with no overlap

### 2. Architecture Diagram
**File**: `fast_architecture_esann_notation.tex` / `.pdf` (21KB)

**Improvements**:
- ✅ No overlapping text (Conv3x3 layers now properly spaced)
- ✅ Clean caption labels with dimensions in parentheses
- ✅ Removed buggy LSTM feedback arrow
- ✅ Better visual flow from left to right
- ✅ Proper spacing for readability
- ✅ **No dimension label overlap** - all info in captions

**Components shown**:
1. Motion Vectors input (2 channels)
2. DCT Residuals input (0-64 channels)
3. Concatenation layer
4. Conv 3×3 layers (2 layers, well-spaced)
5. Global Pool (Fast component - no ROI)
6. Feature vector
7. LSTM (Fast component - no attention)
8. Detection Head
9. Output (bounding boxes)

### 3. Documentation
**File**: `fast_model_documentation_esann.tex` / `.pdf` (234KB, 8 pages)

**Content**:
- **Section 1**: Introduction with motivation and contributions
- **Section 2**: Compressed Video Representation
  - MPEG-4 Part 2 structure
  - GOP notation: $\mathcal{G}^g = \{f_0^g, \dots, f_N^g\}$
  - I-frame: $f_0^g = \{\mathcal{DCT}(Y_0^g)\}$
  - P-frame: $f_n^g = \{\mathcal{DCT}(\Delta Y_n^g), MV_n^g\}$
  - Data efficiency table
  - Partial decompression benefits
- **Section 3**: Fast Architecture
  - Architecture comparison table (Standard vs Fast)
  - Input channel configurations (9 variants)
  - **Architecture diagram with notation table**
- **Section 4**: Training Configuration
  - Dataset table (MOT17/15/20)
  - Hyperparameters table
  - Loss function formula
- **Section 5**: Evaluation Results
  - Static camera performance (0.5800 mAP, +44.3%)
  - Moving camera performance (0.3945 mAP, +399.4%)
  - Performance summary table
- **Section 6**: Analysis
  - Key findings (MOT15 beats baseline!)
  - Per-dataset performance breakdown
  - Computational efficiency (6-12× speedup)
- **Section 7**: Limitations and Future Work
- **Section 8**: Conclusions

## Mathematical Notation (Consistent with esannV2.tex)

### Video Structure
- **GOP**: $\mathcal{G}^g = \{f_0^g, f_1^g, \dots, f_N^g\}$
- **I-frame**: $f_0^g = \{\mathcal{DCT}(Y_0^g)\}$
- **P-frame**: $f_n^g = \{\mathcal{DCT}(\Delta Y_n^g), MV_n^g\}$

### Features
- **Motion vectors**: $MV_n^g$ (2D block displacements)
- **DCT residuals**: $\mathcal{DCT}(\Delta Y_n^g)$ (frequency coefficients)
- **Feature vector**: $\mathbf{h}_n$ (encoded representation)
- **Predictions**: $\hat{\mathbf{b}}_n$ (bounding boxes at frame $n$)

### Loss Function
$$\mathcal{L} = \lambda_{cls} \mathcal{L}_{cls} + \lambda_{bbox} \mathcal{L}_{L1} + \lambda_{giou} \mathcal{L}_{GIoU}$$

## Key Results (from Documentation)

### Static Cameras (106 GOPs)
| Dataset | mAP | vs Mean MV | vs Static BL |
|---------|-----|------------|--------------|
| MOT17   | 0.7341 | +6.7%  | -11.4% |
| MOT15   | 0.4371 | +64.1% | **+2.5%** ✨ |
| MOT20   | 0.6747 | +58.7% | -4.0% |
| **Combined** | **0.5800** | **+44.3%** | **-3.8%** |

### Moving Cameras (94 GOPs)
| Dataset | mAP | vs Mean MV | vs Static BL |
|---------|-----|------------|--------------|
| MOT17   | 0.4304 | **+1410.1%** 🚀 | +63.3% |
| MOT15   | 0.3537 | +150.1% | +9.6% |
| **Combined** | **0.3945** | **+399.4%** | **+32.7%** |

**Breakthrough**: MOT15 learned model exceeds static I-frame baseline on static cameras!

## Visual Improvements

### Before (Issues)
- ❌ Conv3x3 layers overlapping
- ❌ Mathematical notation in captions causing display issues
- ❌ LSTM skip arrow buggy/ugly
- ❌ Tight spacing causing text collision
- ❌ Dimension labels overlapping with captions

### After (Fixed)
- ✅ Clean spacing between all components
- ✅ Simple English captions in diagram
- ✅ Mathematical notation in caption text only
- ✅ No buggy arrows
- ✅ Professional appearance
- ✅ Dimensions moved to captions (no overlap)

## Compilation Commands

```bash
# Generate architecture diagram
cd pdf_latex
python generate_architecture_esann.py
pdflatex fast_architecture_esann_notation.tex

# Compile documentation
pdflatex fast_model_documentation_esann.tex
pdflatex fast_model_documentation_esann.tex  # Second pass for references
```

## File Locations

All files in: `MOTS-experiments/pdf_latex/`

```
pdf_latex/
├── generate_architecture_esann.py              # Updated generator with fixes
├── fast_architecture_esann_notation.tex        # Generated architecture diagram
├── fast_architecture_esann_notation.pdf        # 40KB, clean layout
├── fast_model_documentation_esann.tex          # Full documentation
├── fast_model_documentation_esann.pdf          # 234KB, 8 pages
├── esannV2.tex                                 # Source of notation
└── README.md                                   # Usage guide
```

## Next Steps (Remaining TODO Items)

1. **Ablation study results tables** - Extract from training_results.json
2. **Baseline comparison visualizations** - pgfplots bar charts
3. **Frame-by-frame degradation plots** - mAP over GOP frames
4. **Training convergence plots** - Loss/mAP curves over epochs
5. **Detailed efficiency metrics table** - Parameters, memory, throughput

## Technical Notes

### Spacing Parameters Used
- Input spacing: 2.5 units
- Concat spacing: 3.0 units
- Conv1 spacing: 3.0 units
- Conv2 spacing: 1.0 units (was 0.5, **fixed**)
- Global pool spacing: 2.5 units
- Features spacing: 2.0 units
- LSTM spacing: 3.0 units
- Detection head spacing: 3.5 units
- Output spacing: 2.5 units

### Box Sizes
- Inputs: 35×35 (reduced from 40×40)
- Conv layers: 30×30 (reduced from 35×35)
- Global pool: 8×8 (reduced from 10×10)
- LSTM: 18×18 (reduced from 20×20)
- Detection head: 16×16 (reduced from 18×18)
- Output: 10×1 (reduced from 12×1)

### Caption Strategy
- **In PlotNeuralNet**: Simple English ("Motion Vectors", "Conv 3x3", "LSTM")
- **In LaTeX caption**: Full mathematical notation with table below diagram
- **Result**: Clean visual + complete mathematical description

---

**Status**: ✅ Architecture diagram and documentation successfully updated with esannV2 notation, all overlapping issues resolved, professional appearance achieved.
