# Fast DCT-MV Architecture Diagram with Component Groups

## Overview
The architecture diagram (`fast_architecture_esann_notation.pdf`) now includes **colored background overlays** that group the three main processing stages, with parameter counts for each section.

## Component Groups

### 1. **MV Encoder** (Blue Background)
- **Location**: Left branch of the architecture
- **Components**:
  - `MV_n^g` input: 40×40×2 motion vectors
  - Conv layer: 40×40×32
  - Upsample: 40×40 → 80×80 (matches DCT resolution)
- **Parameters**: ~14K
- **Function**: Encodes and upsamples motion vector information from compressed video

### 2. **DCT Encoder** (Green Background)
- **Location**: Right branch of the architecture  
- **Components**:
  - `DCT(ΔY_n^g)` input: 80×80×64 DCT residuals
  - Conv layer: 80×80×32
- **Parameters**: ~18K–74K (depends on number of coefficients: 8–64)
- **Function**: Encodes DCT frequency coefficients from compressed video

### 3. **Fusion & Temporal** (Orange Background)
- **Location**: Right side after encoders merge
- **Components**:
  - Concat: 80×80×64 (32 MV + 32 DCT channels)
  - Conv 3×3: H×W×256 (×2 layers)
  - Global Pool: 1×1×256
  - Features: 256-dim
  - LSTM: 256 hidden units
  - Detection head: 128-dim
  - Output: {b̂_i} (N_det × 4 bounding boxes)
- **Parameters**: ~196K
- **Function**: Fuses encoded features, performs temporal modeling, and predicts bounding boxes

## Total Model Parameters

| Configuration | MV Encoder | DCT Encoder | Fusion & Temporal | **Total** |
|--------------|-----------|-------------|-------------------|-----------|
| MV-only      | 14K       | 0           | 196K             | **~210K** |
| MV + DCT-8   | 14K       | 18K         | 196K             | **~228K** |
| MV + DCT-16  | 14K       | 28K         | 196K             | **~238K** |
| MV + DCT-32  | 14K       | 46K         | 196K             | **~256K** |
| MV + DCT-64  | 14K       | 74K         | 196K             | **~284K** |

## Visualization Features

### Background Overlays
- **Purpose**: Clearly delineate the three processing stages
- **Style**: 
  - Rounded corners (15pt radius)
  - Semi-transparent fill (12% opacity)
  - Colored border (60% color intensity, 2.5pt width)
  - Drawn on background layer (behind architecture components)

### Labels
- **Group Names**: Bold, large font, white background with colored border
- **Parameter Counts**: Smaller font, white background
- **Position**: Top-left of each group overlay

### Coverage
- Overlays extend beyond component boxes to include:
  - Connection arrows between layers
  - Component labels and dimensions
  - Mathematical notation
- Padding: 2 units left/right, 4.5 units bottom, 6 units top

## Implementation Details

### TikZ Libraries Used
- `calc`: For coordinate calculations (positioning overlays)
- `backgrounds`: For drawing overlays behind architecture
- `positioning`: For relative component placement
- `3d`: For 3D box visualization

### Color Scheme
- **Blue** (`blue!12` fill, `blue!60!black` border): MV Encoder
- **Green** (`green!12` fill, `green!60!black` border): DCT Encoder  
- **Orange** (`orange!12` fill, `orange!60!black` border): Fusion & Temporal

## Usage

### Compilation
```bash
cd pdf_latex/
python generate_architecture_esann.py  # Generate .tex file
pdflatex fast_architecture_esann_notation.tex  # Compile to PDF
```

### Integration in Documentation
The diagram is included in `fast_model_documentation_esann.tex` as a full-page figure:

```latex
\begin{figure}[p]
\centering
\includegraphics[width=1.0\textwidth,height=0.9\textheight,keepaspectratio]
    {fast_architecture_esann_notation.pdf}
\caption{Fast DCT-MV Tracker Architecture...}
\end{figure}
```

## Architecture Flow

```
┌─────────────────────┐  ┌─────────────────────┐
│   MV Encoder (14K)  │  │  DCT Encoder (18-74K)│
│                     │  │                      │
│  MV → Conv → Upsample  │  DCT → Conv         │
│  40×40×2   80×80×32 │  │  80×80×64  80×80×32 │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           └────────┬───────────────┘
                    ↓
        ┌─────────────────────────────────────┐
        │   Fusion & Temporal (196K)          │
        │                                     │
        │  Concat → Conv → Pool → LSTM → Det │
        │  80×80×64   H×W×256   256   {b̂_i}  │
        └─────────────────────────────────────┘
```

## Notes

- The overlays are implemented as TikZ `scope` environments with `on background layer` option
- This ensures they appear behind all architecture components and connections
- The diagram matches the actual implementation in `fast_dct_mv_tracker.py` and `dct_mv_encoder.py`
- Mathematical notation follows esannV2.tex conventions for consistency with paper
