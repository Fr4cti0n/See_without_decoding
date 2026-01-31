# Fast DCT-MV Model Documentation (LaTeX)

This folder contains all LaTeX-related files for generating the Fast DCT-MV model documentation PDF.

## 📁 Contents

### Main Documentation
- **`fast_model_documentation.tex`** - Main LaTeX document with all sections
- **`fast_model_documentation.pdf`** - Generated documentation (162KB, 6 pages)

### Architecture Diagram
- **`generate_fast_architecture.py`** - Python script to generate architecture diagram using PlotNeuralNet
- **`fast_dct_mv_architecture.tex`** - Generated TikZ architecture diagram
- **`fast_dct_mv_architecture.pdf`** - Standalone architecture diagram (42KB)

### Legacy/Alternative Scripts
- **`generate_architecture_diagrams.py`** - Original attempt with more complex diagram generation

## 🚀 Usage

### Regenerate Architecture Diagram

```bash
cd pdf_latex
python generate_fast_architecture.py
```

This will generate:
- `fast_dct_mv_architecture.tex` - TikZ diagram source
- `fast_dct_mv_architecture.pdf` - Compiled diagram

### Compile Documentation

```bash
cd pdf_latex
pdflatex fast_model_documentation.tex
pdflatex fast_model_documentation.tex  # Run twice for references
```

Output: `fast_model_documentation.pdf`

## 📊 Documentation Structure

The main documentation includes:

1. **Introduction** - Motivation and contributions
2. **Architecture** 
   - Fast vs Standard comparison table
   - PlotNeuralNet-generated architecture diagram
   - Input channel configurations
3. **Training Configuration**
   - Dataset information (MOT17/15/20)
   - Hyperparameters
   - Loss function details
4. **Ablation Study**
   - 9 model variants (MV-only, DCT-8/16/32/64, combined)
5. **Evaluation Results**
   - Static camera performance (0.5800 mAP, +44.3%)
   - Moving camera performance (0.3945 mAP, +399.4%)
   - Per-dataset breakdown
6. **Analysis** - Key findings and insights
7. **Conclusions** - Summary and future work

## 🎨 Architecture Diagram Features

The PlotNeuralNet-generated diagram shows:
- ✅ Motion Vector input (2 channels)
- ✅ Optional DCT Residual input (0-64 channels)
- ✅ Feature encoder (Conv 3×3 layers)
- ✅ **Global Pooling** (Fast - no ROI)
- ✅ **Simple LSTM** (Fast - no attention)
- ✅ Detection heads (Class + BBox)

## 📝 Key Results (MV-only Model)

### Static Cameras (106 GOPs)
- MOT17: 0.7341 mAP (+6.7% vs Mean MV)
- MOT15: 0.4371 mAP (+64.1% vs Mean MV, **exceeds static baseline!**)
- MOT20: 0.6747 mAP (+58.7% vs Mean MV)
- **Combined: 0.5800 mAP (+44.3%)**

### Moving Cameras (94 GOPs)
- MOT17: 0.4304 mAP (+1410.1% vs Mean MV!)
- MOT15: 0.3537 mAP (+150.1% vs Mean MV)
- **Combined: 0.3945 mAP (+399.4%)**

## 🔧 Dependencies

### For Architecture Diagram Generation
- Python 3
- PlotNeuralNet (cloned in parent directory: `../PlotNeuralNet`)

### For PDF Compilation
- LaTeX distribution (TeX Live, MiKTeX, etc.)
- Required packages:
  - tikz
  - pgfplots
  - booktabs
  - multirow
  - xcolor
  - hyperref
  - listings

## 📂 File Organization

```
pdf_latex/
├── README.md                              # This file
├── generate_fast_architecture.py          # Architecture diagram generator
├── generate_architecture_diagrams.py      # Legacy script
├── fast_dct_mv_architecture.tex          # Generated TikZ diagram
├── fast_dct_mv_architecture.pdf          # Compiled diagram
├── fast_model_documentation.tex          # Main documentation
├── fast_model_documentation.pdf          # Final PDF
└── *.aux, *.log, *.out                   # LaTeX auxiliary files
```

## 🔄 Workflow

1. **Generate architecture diagram** (if needed):
   ```bash
   python generate_fast_architecture.py
   ```

2. **Edit documentation** (if needed):
   Edit `fast_model_documentation.tex`

3. **Compile PDF**:
   ```bash
   pdflatex fast_model_documentation.tex
   pdflatex fast_model_documentation.tex  # For references
   ```

4. **View result**:
   Open `fast_model_documentation.pdf`

## 📌 Notes

- The architecture diagram is included as a PDF (`\includegraphics{fast_dct_mv_architecture.pdf}`)
- PlotNeuralNet must be in the parent directory (`../PlotNeuralNet`)
- Run pdflatex twice to resolve cross-references
- Some Unicode characters (✓, ✗, α, γ) may show warnings but compile successfully

## 🎯 TODO (from main project)

See parent directory `TODO list` for remaining documentation tasks:
- [ ] Document training parameters (extract from logs)
- [ ] Create ablation study results tables (9 variants)
- [ ] Add baseline comparison visualizations (pgfplots)
- [ ] Create frame-by-frame degradation plots
- [ ] Add training convergence plots
- [ ] Document efficiency metrics
- [ ] Add qualitative results section
- [ ] Expand conclusions and future work
