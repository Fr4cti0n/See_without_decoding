#!/usr/bin/env python3
"""
Generate ROI-Based Architecture Diagram using PlotNeuralNet

This script generates the architecture with ROI-aligned feature extraction:
- Box-aligned motion feature extraction (per-object features)
- No global pooling (uses ROI-specific statistics instead)
- Each box gets different motion features based on its region
"""

import sys
import os

# Add PlotNeuralNet to path (parent directory)
plot_neural_net_path = os.path.join(os.path.dirname(__file__), '..', 'PlotNeuralNet')
sys.path.append(plot_neural_net_path)

from pycore.tikzeng import *

def add_group_overlay(name, corner1, corner2, color, label, param_text, padding_x=2.5, padding_x_left=None, padding_x_right=None, padding_y_top=6.0, padding_y_bot=-5.0, padding_z=3.0, label_position="top", use_corner2_for_label=False, title_x_offset=None):
    """
    Add a colored rectangle overlay to group architecture components.
    """
    # Support asymmetric padding
    pad_left = padding_x_left if padding_x_left is not None else padding_x
    pad_right = padding_x_right if padding_x_right is not None else padding_x
    
    # Calculate label positions based on label_position parameter
    if label_position == "middle":
        label_anchor_point = f"{corner1}-west"
        label_node_anchor = "west"
        label_y_offset = "0.0"
        param_y_offset = "-2.5"
    else:  # "top"
        label_node_anchor = "north west"
        label_corner = corner2 if use_corner2_for_label else corner2
        label_anchor_type = "nearnortheast" if use_corner2_for_label else "nearnortheast"
        label_y_offset = f"{padding_y_top - 1.2}" if not use_corner2_for_label else f"{padding_y_top - 1.0}"
        param_y_offset = f"{padding_y_top - 2.5}" if not use_corner2_for_label else f"{padding_y_top - 2.5}"
    
    # Calculate title X offset
    title_x = title_x_offset if title_x_offset is not None else -(pad_left - 0.3)
    
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
% Labels
\node[anchor={label_node_anchor}, font=\bfseries\Large, text={color}!90!black, fill={color}!15, inner sep=5pt, rounded corners=5pt, draw={color}!70!black, line width=2pt] at 
    ($({label_corner}-{label_anchor_type}) + ({title_x}, {label_y_offset}, {"0" if not use_corner2_for_label else padding_z})$) {{{label}}};
\node[anchor={label_node_anchor}, font=\normalsize, text={color}!90!black, fill={color}!15, inner sep=4pt, rounded corners=4pt, draw={color}!50!black, line width=1pt] at 
    ($({label_corner}-{label_anchor_type}) + ({title_x}, {param_y_offset}, {"0" if not use_corner2_for_label else padding_z})$) {{{param_text}}};
"""

def main():
    """
    Generate the ROI-based architecture diagram - CLEARER VERSION.
    
    Flow: Boxes(t-1) + MV(t) + DCT(t) → ROI Features → LSTM → Boxes(t)
    
    Key improvements:
    1. Linear left-to-right flow (easier to follow)
    2. Input boxes at the START (not hidden at bottom)
    3. Clear temporal progression: t-1 → processing → t
    4. Grouped by function: Inputs | ROI Extraction | Fusion | Output
    """
    
    arch = [
        to_head('../PlotNeuralNet'),
        r"\usetikzlibrary{calc}" + "\n",
        r"\usetikzlibrary{backgrounds}" + "\n",
        to_cor(),
        to_begin(),
        
        # ===== INPUTS (Left side - clearly grouped) =====
        # Start at X=0, with inputs stacked vertically
        
        # Top: Previous Boxes (t-1) - PRIMARY INPUT
        to_Conv("prev_boxes", s_filer="", n_filer="", offset="(0,6,0)", to="(0,0,0)", 
                width=1, height=12, depth=1, caption=r"Boxes $(t\!-\!1)$\\$N{\times}4$"),
        
        # Middle: Motion Vectors (t)
        to_Conv("mv_input", s_filer="", n_filer="", offset="(0,0,0)", to="(0,0,0)", 
                width=1.5, height=16, depth=10, caption=r"$MV_t$\\$60{\times}60{\times}2$"),
        
        # Bottom: DCT Residuals (t)
        to_Conv("dct_input", s_filer="", n_filer="", offset="(0,-6,0)", to="(0,0,0)", 
                width=2, height=18, depth=12, caption=r"DCT $(t)$\\$80{\times}80{\times}C$"),
        
        # ===== BOX ENCODING (prepare boxes for fusion) =====
        to_Conv("box_embed", s_filer="", n_filer="", offset="(3,0,0)", to="(prev_boxes-east)", 
                width=1.5, height=10, depth=1, caption=r"Box Enc\\$N{\times}32$"),
        to_connection("prev_boxes", "box_embed"),
        
        # ===== ROI FEATURE EXTRACTION =====
        # Extract features FROM motion vectors USING box regions
        
        # MV ROI Extraction
        to_Conv("roi_motion", s_filer="", n_filer="", offset="(3,0,0)", to="(mv_input-east)", 
                width=2, height=14, depth=12, caption=r"MV ROI\\Stats"),
        to_connection("mv_input", "roi_motion"),
        to_connection("prev_boxes", "roi_motion"),  # Boxes define ROI regions
        
        # Motion features per box
        to_Conv("motion_feat", s_filer="", n_filer="", offset="(2.5,0,0)", to="(roi_motion-east)", 
                width=1.5, height=10, depth=1, caption=r"$N{\times}64$"),
        to_connection("roi_motion", "motion_feat"),
        
        # DCT ROI Extraction
        to_Conv("roi_dct", s_filer="", n_filer="", offset="(3,0,0)", to="(dct_input-east)", 
                width=2, height=14, depth=12, caption=r"DCT ROI\\Stats"),
        to_connection("dct_input", "roi_dct"),
        to_connection("prev_boxes", "roi_dct"),  # Boxes define ROI regions
        
        # DCT features per box
        to_Conv("dct_feat", s_filer="", n_filer="", offset="(2.5,0,0)", to="(roi_dct-east)", 
                width=1.5, height=10, depth=1, caption=r"$N{\times}32$"),
        to_connection("roi_dct", "dct_feat"),
        
        # ===== FUSION =====
        # Concatenate all per-box features
        to_Conv("fusion", s_filer="", n_filer="", offset="(11,0,0)", to="(0,0,0)", 
                width=2, height=12, depth=12, caption=r"Concat\\$N{\times}128$"),
        to_connection("box_embed", "fusion"),
        to_connection("motion_feat", "fusion"),
        to_connection("dct_feat", "fusion"),
        
        # ===== TEMPORAL MODELING =====
        to_SoftMax("lstm", s_filer="", offset="(2.5,0,0)", to="(fusion-east)", 
                   width=2.5, height=12, depth=12, caption=r"BiLSTM\\$N{\times}128$"),
        to_connection("fusion", "lstm"),
        
        # ===== OUTPUT HEADS =====
        # Position deltas
        to_Conv("pos_head", s_filer="", n_filer="", offset="(2.5,0,0)", to="(lstm-east)", 
                width=1.5, height=10, depth=1, caption=r"$\Delta$pos\\$N{\times}2$"),
        to_connection("lstm", "pos_head"),
        
        # Size deltas (below position)
        to_Conv("size_head", s_filer="", n_filer="", offset="(0,-3,0)", to="(pos_head-east)", 
                width=1.5, height=10, depth=1, caption=r"$\Delta$size\\$N{\times}2$"),
        to_connection("lstm", "size_head"),
        
        # ===== OUTPUT: Updated Boxes (t) =====
        to_Conv("output_boxes", s_filer="", n_filer="", offset="(2.5,1.5,0)", to="(pos_head-east)", 
                width=1, height=12, depth=1, caption=r"Boxes $(t)$\\$N{\times}4$"),
        to_connection("pos_head", "output_boxes"),
        to_connection("size_head", "output_boxes"),
        
        # ===== GROUP OVERLAYS (clearer grouping) =====
        
        # Stage 1: INPUTS (all 3 inputs together)
        add_group_overlay(
            name="inputs_group",
            corner1="dct_input",  # Bottom
            corner2="prev_boxes",  # Top
            color="purple",
            label="Inputs $(t\!-\!1, t)$",
            param_text=r"Boxes + MV + DCT",
            padding_x=2.0,
            padding_y_top=4.0,
            padding_y_bot=-4.0,
            padding_z=2.5,
            label_position="top"
        ),
        
        # Stage 2: ROI EXTRACTION (MV + DCT branches)
        add_group_overlay(
            name="roi_extraction",
            corner1="roi_dct",  # Bottom  
            corner2="motion_feat",  # Top
            color="blue",
            label="ROI Extraction",
            param_text=r"32K params (box-aligned)",
            padding_x=2.5,
            padding_y_top=4.5,
            padding_y_bot=-4.5,
            padding_z=3.0,
            label_position="top"
        ),
        
        # Stage 3: FUSION & TEMPORAL
        add_group_overlay(
            name="fusion_temporal",
            corner1="fusion",
            corner2="output_boxes",
            color="orange",
            label=r"Fusion \& Temporal",
            param_text=r"196K params",
            padding_x=2.5,
            padding_y_top=4.5,
            padding_y_bot=-4.5,
            padding_z=3.5,
            label_position="top"
        ),
        
        to_end()
    ]
    
    # Generate the LaTeX file
    output_file = "roi_architecture.tex"
    to_generate(arch, output_file)
    
    print("=" * 70)
    print("✅ ROI-Based Architecture Diagram Generated (CLEAR VERSION)!")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
    print(f"📝 Compile with: pdflatex {output_file}")
    print(f"📊 Output PDF: {output_file.replace('.tex', '.pdf')}")
    print()
    print("🎯 CLEARER DESIGN:")
    print("  • Linear left-to-right flow (easy to follow)")
    print("  • Inputs grouped at START (not hidden)")
    print("  • Clear temporal: Boxes(t-1) → Processing → Boxes(t)")
    print("  • Autoregressive loop visible")
    print()
    print("📊 Architecture Flow:")
    print("  1️⃣  INPUTS: Boxes(t-1) + MV(t) + DCT(t)")
    print("  2️⃣  ROI EXTRACTION: Extract box-specific features")
    print("  3️⃣  FUSION: Combine box + MV + DCT features")
    print("  4️⃣  TEMPORAL: BiLSTM processes sequence")
    print("  5️⃣  OUTPUT: Predict deltas → Boxes(t)")
    print()
    print("🏗️ Key Improvement:")
    print("  Each box gets DIFFERENT motion features based on its region ✅")
    print()

if __name__ == '__main__':
    main()
