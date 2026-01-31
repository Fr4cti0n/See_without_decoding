#!/usr/bin/env python3
"""
Generate CLEAR BAFE (Box-Aligned Feature Extraction) Architecture Diagram

Key clarifications:
1. Boxes from (t-1) define WHERE to extract box-aligned features
2. MV and DCT from (t) provide actual data being extracted
3. Fixed-size grids: 15×15 for MV, 30×30 for DCT (from paper)
4. Dashed lines = "defines extraction region" (not data flow)
5. Solid arrows = data flow
6. All dimensions explicitly shown
"""

import sys
import os

# Add PlotNeuralNet to path (parent directory)
plot_neural_net_path = os.path.join(os.path.dirname(__file__), '..', 'PlotNeuralNet')
sys.path.append(plot_neural_net_path)

from pycore.tikzeng import *

def add_group_overlay(name, corner1, corner2, color, label, param_text, padding_x=2.5, padding_x_left=None, padding_x_right=None, padding_y_top=6.0, padding_y_bot=-5.0, padding_z=3.0, label_position="top", use_corner2_for_label=False, title_x_offset=None, center_title=False, use_left_alignment=False):
    """Add colored rectangle overlay to group components."""
    pad_left = padding_x_left if padding_x_left is not None else padding_x
    pad_right = padding_x_right if padding_x_right is not None else padding_x
    
    if label_position == "middle":
        label_node_anchor = "west"
        label_y_offset = "0.0"
        param_y_offset = "-2.5"
    else:  # "top"
        label_node_anchor = "north" if center_title else "north west"
        # Use left or right alignment based on parameter
        if use_left_alignment:
            label_corner = corner1
            label_anchor_type = "nearnorthwest"
            title_x = title_x_offset if title_x_offset is not None else (pad_left - 0.3)
        else:
            label_corner = corner2
            label_anchor_type = "nearnortheast"
            title_x = title_x_offset if title_x_offset is not None else -(pad_left - 0.3)
        
        label_y_offset = f"{padding_y_top - 1.2}"
        param_y_offset = f"{padding_y_top - 2.5}"
    
    # Use centered position if requested, otherwise left/right-aligned
    if center_title:
        label_pos = f"($({corner1}-nearsouthwest)!0.5!({corner2}-nearnortheast) + (0, {label_y_offset}, 0)$)"
        param_pos = f"($({corner1}-nearsouthwest)!0.5!({corner2}-nearnortheast) + (0, {param_y_offset}, 0)$)"
    else:
        label_pos = f"($({label_corner}-{label_anchor_type}) + ({title_x}, {label_y_offset}, 0)$)"
        param_pos = f"($({label_corner}-{label_anchor_type}) + ({title_x}, {param_y_offset}, 0)$)"
    
    return rf"""
% Group overlay: {label}
\begin{{scope}}[on background layer]
\fill[{color}!8, rounded corners=15pt] 
    ($({corner1}-nearsouthwest) + (-{pad_left}, {padding_y_bot}, -{padding_z})$) rectangle 
    ($({corner2}-nearnortheast) + ({pad_right}, {padding_y_top}, {padding_z})$);
\draw[{color}!70!black, line width=3pt, rounded corners=15pt] 
    ($({corner1}-nearsouthwest) + (-{pad_left}, {padding_y_bot}, -{padding_z})$) rectangle 
    ($({corner2}-nearnortheast) + ({pad_right}, {padding_y_top}, {padding_z})$);
\end{{scope}}
\node[anchor={label_node_anchor}, font=\bfseries\LARGE, text={color}!90!black, fill={color}!15, inner sep=5pt, rounded corners=5pt, draw={color}!70!black, line width=2pt] at 
    {label_pos} {{{label}}};
\node[anchor={label_node_anchor}, font=\large, text={color}!90!black, fill={color}!15, inner sep=4pt, rounded corners=4pt, draw={color}!50!black, line width=1pt] at 
    {param_pos} {{{param_text}}};
"""

def main():
    """Generate clearer ROI architecture with proper data flow."""
    
    arch = [
        to_head('../PlotNeuralNet'),
        r"\usepackage{amsmath}" + "\n",
        r"\usetikzlibrary{calc}" + "\n",
        r"\usetikzlibrary{backgrounds}" + "\n",
        r"\usetikzlibrary{decorations.pathreplacing}" + "\n",
        to_cor(),
        to_begin(),
        
        # ===== STAGE 1: INPUTS =====
        # Define invisible anchor points for connections (no boxes, just signals)
        r"\coordinate (dct_frame) at (2,6,0);",
        r"\coordinate (mv_frame) at (2,0,0);",
        r"\coordinate (boxes_prev) at (2,-6,0);",
        
        # ===== STAGE 2: BAFE (BOX-ALIGNED FEATURE EXTRACTION) =====
        # Note: Boxes define WHERE, MV/DCT provide WHAT
        
        # Add curly brace showing MV and DCT come from same frame
        r"""
\draw[decorate, decoration={brace, amplitude=15pt, raise=5pt}, line width=2pt, black!100] 
    ($(-0.2,-1.2,0) + (-1.8,0,0)$) -- ($(-0.2,6.9,0) + (-1.8,0,0)$) 
    node[midway, left=20pt, font=\LARGE, align=center, text=black] {P-Frame\\[1pt]$f_n^g$};
""",
        
        # MV Grid Extraction: Extract 15×15 grid per box (from paper)
        # Fixed grid: 15×15 for motion vectors (16-pixel macroblocks)
        # Place label to the left of coordinate
        r"\node[anchor=east, font=\LARGE, align=right, text=black] at ($(2,0,0) + (-0.3,0,0)$) {$MV_n^g$\\[1pt]$60{\times}60{\times}2$};",
        to_Conv("mv_roi", s_filer="", n_filer="", offset="(3.5,0,0)", to="(1.5,0,0)", 
                width=1.8, height=13, depth=8, caption=r"\Large BAFE-MV\\[2pt]$N{\times}15{\times}15{\times}2$"),
        r"\draw [connection] (mv_frame) -- (mv_roi-west);",
        
        # MV A-Features: Add label below arrow instead of box
        # Define a coordinate point where mv_stats would have been
        r"\coordinate (mv_stats) at ($(mv_roi-east) + (6.2,0,0)$);",
        r"\draw [connection] (mv_roi-east) -- node[below=2pt, font=\LARGE, align=center, text=black] {MV A-Features\\[1pt]$N{\times}128$} (mv_stats);",
        
        # DCT Conv Processing
        # Place label to the left of coordinate
        r"\node[anchor=east, font=\LARGE, align=right, text=black] at ($(2,6,0) + (-0.3,0,0)$) {$\Delta Y_n^g$\\[1pt]$120{\times}120{\times}64$};",
        to_Conv("dct_conv", s_filer="", n_filer="", offset="(2,0,0)", to="(1.75,6,0)", 
                width=2, height=13, depth=12, caption=r"\Large Conv $1{\times}1$ \\[2pt]"),
        r"\draw [connection] (dct_frame) -- (dct_conv-west);",
        
        # DCT Grid Extraction: Extract 30×30 grid per box (from paper)
        # Fixed grid: 30×30 for DCT (8×8 blocks, 64 channels)
        to_Conv("dct_roi", s_filer="", n_filer="", offset="(7,0,0)", to="(dct_conv-east)", 
                width=1.8, height=13, depth=8, caption=r"\Large BAFE-DCT\\[2pt]$N{\times}30{\times}30{\times}32$"),
        r"\draw [connection] (dct_conv-east) -- node[below=2pt, font=\LARGE, align=center, text=black] {Encoded DCT\\[1pt]$120{\times}120{\times}32$} (dct_roi-west);",
        
        # Dashed line for DCT box-aligned regions
        
        # DCT A-Features: Add label below arrow instead of box
        # Define a coordinate point where dct_stats would have been
        r"\coordinate (dct_stats) at ($(dct_roi-east) + (6.2,0,0)$);",
        r"\draw [connection] (dct_roi-east) -- node[below=2pt, font=\LARGE, align=center, text=black] {DCT A-Features\\[1pt]$N{\times}128$} (dct_stats);",
        
        # ===== STAGE 3: BOX EMBEDDING =====
        # Encode box coordinates (parallel path)
        # Place label to the left of coordinate
        r"\node[anchor=east, font=\LARGE, align=right, text=black] at ($(2,-6,0) + (-0.3,0,0)$) {$bb(n{-}1)$\\[1pt]$N{\times}5$};",
        to_Conv("box_enc", s_filer="", n_filer="", offset="(3.5,0,0)", to="(3.7,-6,0)", 
                width=1.8, height=10, depth=1, caption=r"\Large MLP\\[2pt]$5{\rightarrow}32$"),
        r"\draw [connection] (boxes_prev) -- (box_enc-west);",
        
        # ===== STAGE 4: FUSION =====
        # Concatenate: MV [N×128] + DCT [N×128] + Box [N×32] = [N×288]
        to_Conv("concat", s_filer="", n_filer="", offset="(1.5,-10.0,0)", to="($(dct_stats) + (1.5,0,0)$)", 
                width=2, height=13, depth=11, caption=r"\Large Concat\\$N{\times}288$"),
        r"\draw [connection] (mv_stats) -- ++(1.5,0) |- (concat-west);",
        r"\draw [connection] (dct_stats) -- ++(1.5,0) |- (concat-west);",
        r"\draw [connection] (box_enc-east) -- node[above=2pt, font=\LARGE, align=center, text=black] {Encoded Boxes\\[1pt]$N{\times}32$} ++(5.5,0) |- (concat-west);",
        
        # ===== STAGE 5: TEMPORAL MODELING =====
        to_SoftMax("lstm", s_filer="", offset="(2.0,0,0)", to="(concat-east)", 
                   width=2.5, height=13, depth=11, caption=r"\Large BiLSTM\\$N{\times}128$"),
        to_connection("concat", "lstm"),
        
        # ===== STAGE 6: PREDICTION HEADS =====
        # Refinement head with coordinate point before boxes output
        r"\coordinate (refinement) at ($(lstm-east) + (5.5,0,0)$);",
        r"\draw [connection] (lstm-east) -- node[below=2pt, font=\LARGE, align=center, text=black] {Refinements\\[1pt]$N{\times}5$} (refinement);",
        
        # ===== OUTPUT: Updated Boxes =====
        to_Conv("boxes_out", s_filer="", n_filer="", offset="(0,0,0)", to="(refinement)", 
                width=0.8, height=10, depth=1, caption=r"\Large ABoxes \\$bb(n)$\\$N{\times}5$"),
        r"\draw [connection] (refinement) -- (boxes_out-west);",
        
        # Red dashed arrow showing temporal loop: boxes(t) -> boxes(t-1) for next frame
        # Exit right with offset, go down, across, come back to coordinate point
        r"\draw[-Stealth, red!70!black, line width=2.5pt, dashed, opacity=0.8] (boxes_out-east) -- ++(2,0) |- ++(0,-5) -| (boxes_prev);",
        
        # ===== LOSS SECTION =====
        # Position loss equations at the top (without box border, moved right, total first)
        r"""
% Loss Section - Detection Objectives (no border)
\node[anchor=north west, text width=12.25cm, align=left, font=\Large] 
    at ($(0,0,0) + (20.0, 9.75, 0)$) {
    {\bfseries\LARGE Training Loss}
    
    \vspace{16pt}
    
    \textbf{Total Loss:}
    $\displaystyle \mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{box}} + \lambda_2 \mathcal{L}_{\text{GIoU}} + \lambda_3 \mathcal{L}_{\text{cls}}$
    
    \vspace{8pt}

    
    \textbf{ Box Regression (L1):} 
    $\displaystyle \mathcal{L}_{\text{box}}=\frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \text{L1}(\widehat{bb}_i, bb_i^*)$
    
    \vspace{6pt}
    
    \textbf{ GIoU Loss:} 
    $\displaystyle \mathcal{L}_{\text{GIoU}} = \frac{1}{N_{\text{pos}}} \sum_{i \in \text{pos}} \left(1 - \text{GIoU}(\widehat{bb}_i, bb_i^*)\right)$
    
    \vspace{6pt}
    
    \textbf{ Focal Loss:} 
    $\displaystyle \mathcal{L}_{\text{cls}} = -\frac{1}{N} \sum_{i=1}^{N} \alpha_i (1-p_i)^\gamma \log(p_i)$
};
""",
        
        # ===== OVERLAYS =====
        
        # Input Stage - using coordinate points
        r"""
% Input Stage overlay (using coordinates)
\begin{scope}[on background layer]
\fill[purple!8, rounded corners=15pt] 
    ($(0,-6,0) + (-6.5, -5.0, -2.5)$) rectangle 
    ($(1,6,0) + (2.5, 5.0, 2.5)$);
\draw[purple!70!black, line width=3pt, rounded corners=15pt] 
    ($(0,-6,0) + (-6.5, -5.0, -2.5)$) rectangle 
    ($(1,6,0) + (2.5, 5.0, 2.5)$);
\end{scope}
\node[anchor=north west, font=\bfseries\LARGE, text=purple!90!black, fill=purple!15, inner sep=5pt, rounded corners=5pt, draw=purple!70!black, line width=2pt] at 
    ($(-1.5,7,0) + (-1.0, 2.8, 0)$) {Inputs};
""",
        
        # BAFE Extraction Stage - using dct_roi as corner2 since dct_stats is now a coordinate
        r"""
% BAFE Extraction overlay
\begin{scope}[on background layer]
\fill[blue!8, rounded corners=15pt] 
    ($(mv_roi-nearsouthwest) + (-3, -9.5, -2.8)$) rectangle 
    ($(dct_roi-nearnortheast) + (9.5, 4.0, 2.8)$);
\draw[blue!70!black, line width=3pt, rounded corners=15pt] 
    ($(mv_roi-nearsouthwest) + (-3, -9.5, -2.8)$) rectangle 
    ($(dct_roi-nearnortheast) + (9.5, 4.0, 2.8)$);
\end{scope}
\node[anchor=north west, font=\bfseries\LARGE, text=blue!90!black, fill=blue!15, inner sep=5pt, rounded corners=5pt, draw=blue!70!black, line width=2pt] at 
    ($(dct_roi-nearnortheast) + (-5.5, 2.8, 0)$) {BAFE Extraction};
\node[anchor=north west, font=\large, text=blue!90!black, fill=blue!15, inner sep=4pt, rounded corners=4pt, draw=blue!50!black, line width=1pt] at 
    ($(dct_roi-nearnortheast) + (-5.5, 1.5, 0)$) {~70K params};
""",
        
        # Fusion & Temporal Stage - now includes boxes_out box
        r"""
% Fusion & Temporal overlay
\begin{scope}[on background layer]
\fill[orange!8, rounded corners=15pt] 
    ($(concat-nearsouthwest) + (-1.7, -5.5, -3.2)$) rectangle 
    ($(boxes_out-nearnortheast) + (4.0, 6.65, 3.2)$);
\draw[orange!70!black, line width=3pt, rounded corners=15pt] 
    ($(concat-nearsouthwest) + (-1.7, -5.5, -3.2)$) rectangle 
    ($(boxes_out-nearnortheast) + (4.0, 6.65, 3.2)$);
\end{scope}
\node[anchor=north west, font=\bfseries\LARGE, text=orange!90!black, fill=orange!15, inner sep=5pt, rounded corners=5pt, draw=orange!70!black, line width=2pt] at 
    ($(concat-nearnortheast) + (0, 5.4, 0)$) {Fusion \& Temporal};
\node[anchor=north west, font=\large, text=orange!90!black, fill=orange!15, inner sep=4pt, rounded corners=4pt, draw=orange!50!black, line width=1pt] at 
    ($(concat-nearnortheast) + (0, 4.1, 0)$) {~160K params (BiLSTM + heads)};
""",
        
        to_end()
    ]
    
    # Generate the LaTeX file
    output_file = "bafe_architecture.tex"
    to_generate(arch, output_file)
    
    print("=" * 70)
    print("✅ BAFE (Box-Aligned Feature Extraction) Architecture Generated!")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
    print(f"📝 Compile with: pdflatex {output_file}")
    print()
    print("🎯 Key Features:")
    print("  • BAFE: Box-Aligned Feature Extraction (not RoI pooling)")
    print("  • Boxes(t-1): Define WHERE to extract (spatial guidance)")
    print("  • MV(t), DCT(t): Actual data being extracted (WHAT)")
    print("  • Dashed lines: Spatial region definition (not data flow)")
    print("  • Solid arrows: Data flow")
    print("  • Larger fonts: \\Large for captions, \\LARGE for labels")
    print()
    print("📊 Grid Dimensions (from paper):")
    print("  • MV Grid: 15×15 (16-pixel macroblocks)")
    print("  • DCT Grid: 30×30 (8×8 blocks, 64 channels)")
    print("  • N: Number of boxes (variable, typically 5-50 per frame)")
    print()
    print("  • Boxes(t-1): N×4 (cx, cy, w, h)")
    print("  • MV(t): 60×60×2 (full frame)")
    print("  • DCT(t): 120×120×64 (full frame)")
    print("  • MV Grid: 60×60×2 → N×15×15×2 (fixed grid per box)")
    print("  • DCT Grid: 120×120×64 → N×30×30×32 (fixed grid per box)")
    print("  • MV Conv: N×15×15×2 → N×128 (2 Conv blocks, 64 channels)")
    print("  • DCT Conv: N×30×30×32 → N×128 (2 Conv blocks, 64 channels)")
    print("  • Box Enc: N×4 → N×32")
    print("  • Concat: N×288 (128 MV + 128 DCT + 32 box)")
    print("  • BiLSTM: N×288 → N×128")
    print("  • Output: N×4 boxes(t)")
    print()
    print("⚙️  Parameters (~230K total):")
    print("  • BAFE Extraction: ~70K params")
    print("    - MV Conv: 2 blocks, 64 channels")
    print("    - DCT Conv: 2 blocks, 64 channels")
    print("    - Same weights applied to all N boxes")
    print("  • Fusion & Temporal: ~160K params")
    print("    - BiLSTM + prediction heads")
    print("    - Processes all N boxes together")
    print()
    print("🔑 Key Difference from RoI Align:")
    print("  • RoI Align: Adaptive pooling to fixed size")
    print("  • BAFE: Fixed-size grid extraction (15×15 MV, 30×30 DCT)")
    print("  • BAFE preserves spatial structure within bounding box")
    print()

if __name__ == '__main__':
    main()
