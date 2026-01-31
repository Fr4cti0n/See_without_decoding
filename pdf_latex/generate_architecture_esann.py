#!/usr/bin/env python3
"""
Generate Fast DCT-MV Architecture Diagram using PlotNeuralNet with esannV2.tex notation

This script generates a TikZ diagram of the Fast DCT-MV tracker architecture
using the mathematical notation from Section 3 of esannV2.tex:
- GOP: G^g = {f_0^g, ..., f_N^g}
- I-frame: f_0^g = {DCT(Y_0^g)}
- P-frame: f_n^g = {DCT(ΔY_n^g), MV_n^g}
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
    Labels can be placed at top or middle of the overlay.
    
    CRITICAL FIX: Use proper corner anchors instead of -west/-east!
    PlotNeuralNet defines these anchors for each Box:
      - {name}-nearsouthwest: bottom-left-front corner (0, -y/2, z/2)
      - {name}-nearnortheast: top-right-front cornnoer (LastEastx, y/2, z/2)
      - {name}-northwest: top-left center (0, y/2, 0)
      - {name}-west: middle-left center (0, 0, 0)
    
    BALANCED padding: SMALL vertical to prevent overlap, MODERATE horizontal
    
    Args:
        name: Unique identifier for the rectangle
        corner1: Name of first component (e.g., "mv_input")
        corner2: Name of last component (e.g., "mv_upsample")
        color: Fill color (e.g., "blue!10")
        label: Text label for the group
        param_text: Parameter count description
        padding_x: Horizontal padding (left/right) - MODERATE (default: 2.5)
        padding_x_left: Left padding override (if None, uses padding_x)
        padding_x_right: Right padding override (if None, uses padding_x)
        padding_y_top: Top padding - SMALL to avoid overlap (default: 6.0)
        padding_y_bot: Bottom padding - SMALL to avoid overlap (default: -5.0)
        padding_z: Depth padding (front/back in 3D) - SMALL (default: 3.0)
        label_position: "top" or "middle" - where to place labels (default: "top")
        title_x_offset: Manual X offset for title (if None, calculated from padding)
        padding_y_bot: Bottom padding - SMALL to avoid overlap (default: -5.0)
        padding_z: Depth padding (front/back in 3D) - SMALL (default: 3.0)
        label_position: "top" or "middle" - where to place labels (default: "top")
    
    Returns:
        TikZ code string
    """
    # Support asymmetric padding
    pad_left = padding_x_left if padding_x_left is not None else padding_x
    pad_right = padding_x_right if padding_x_right is not None else padding_x
    
    # Calculate label positions based on label_position parameter
    if label_position == "middle":
        # Middle-left positioning - using -west anchor (middle of left edge)
        # Use "west" anchor for labels so they're centered, not "north west"
        label_anchor_point = f"{corner1}-west"
        label_node_anchor = "west"  # Label's anchor point (center-left of text)
        label_y_offset = "0.0"  # Centered vertically
        param_y_offset = "-2.5"  # Below the main label
    else:  # "top"
        # Top-left positioning - position INSIDE overlay near top edge
        label_node_anchor = "north west"
        label_corner = corner2 if use_corner2_for_label else corner2
        label_anchor_type = "nearnortheast" if use_corner2_for_label else "nearnortheast"
        # Position just below overlay top edge (padding_y_top - small offset = inside)
        # Params positioned very close below title (minimal spacing)
        label_y_offset = f"{padding_y_top - 1.2}" if not use_corner2_for_label else f"{padding_y_top - 1.0}"
        param_y_offset = f"{padding_y_top - 2.5}" if not use_corner2_for_label else f"{padding_y_top - 2.5}"
    
    # Calculate title X offset: use manual override if provided, otherwise calculate from padding
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
% Labels at OVERLAY BOX top-left corner (INSIDE the box)
\node[anchor={label_node_anchor}, font=\bfseries\Large, text={color}!90!black, fill={color}!15, inner sep=5pt, rounded corners=5pt, draw={color}!70!black, line width=2pt] at 
    ($({label_corner}-{label_anchor_type}) + ({title_x}, {label_y_offset}, {"0" if not use_corner2_for_label else padding_z})$) {{{label}}};
\node[anchor={label_node_anchor}, font=\normalsize, text={color}!90!black, fill={color}!15, inner sep=4pt, rounded corners=4pt, draw={color}!50!black, line width=1pt] at 
    ($({label_corner}-{label_anchor_type}) + ({title_x}, {param_y_offset}, {"0" if not use_corner2_for_label else padding_z})$) {{{param_text}}};
"""

def main():
    """
    Generate the Fast DCT-MV architecture diagram with proper notation.
    
    Architecture flow:
    1. Parallel Inputs: Motion Vectors MV_n^g (2 ch) + DCT Residuals DCT(ΔY_n^g) (0-64 ch)
    2. Feature Encoder: Concatenation → Convolutional layers
    3. Global Pooling: Spatial aggregation (Fast - no ROI)
    4. LSTM: Temporal modeling (Fast - no attention)
    5. Detection Head: Multiple bounding box predictions {b̂_i}_{i=1}^{N_det}
    """
    
    arch = [
        to_head('../PlotNeuralNet'),  # Reference parent directory
        r"\usetikzlibrary{calc} % For coordinate calculations in overlays" + "\n",
        r"\usetikzlibrary{backgrounds} % For drawing overlays behind architecture" + "\n",
        to_cor(),
        to_begin(),
        
        # ===== INPUT BRANCHES (PARALLEL) =====
        # Motion Vectors Branch: MV_n^g (2D block displacements)
        # Input: 40×40×2 (lower spatial resolution than DCT)
        # MOVED DOWN to Y=-3 to create more vertical separation from Fusion
        # BETTER CENTERED: X=-1.5 (more centered, not too close to left edge)
        # Label NOT diagonal - using xlabel for horizontal positioning
        to_Conv("mv_input", s_filer="", n_filer="", offset="(-1.5,-3,0)", to="(0,3,0)", 
                width=1, height=16, depth=10, caption=r"$MV_n^g$\\$40{\times}40{\times}2$"),
        
        # MV processing: Conv layers at original resolution - MODERATE SPACING (1.5) to avoid notation overlap while fitting in layout
        to_Conv("mv_conv", s_filer="", n_filer="", offset="(1.5,0,0)", to="(mv_input-east)", 
                width=2, height=16, depth=10, caption=r"Conv\\$40{\times}40{\times}32$"),  # Reduced depth: 16→10, spacing: 1.5
        to_connection("mv_input", "mv_conv"),
        
        # MV Upsample: 40×40 → 80×80 to match DCT spatial resolution - MODERATE SPACING (1.5)
        # Uses nearest neighbor interpolation (scale_factor=2)
        to_Conv("mv_upsample", s_filer="", n_filer="", offset="(1.5,0,0)", to="(mv_conv-east)", 
                width=2, height=22, depth=14, caption=r"Upsample\\$80{\times}80{\times}32$"),  # Reduced depth: 22→14, spacing: 1.5
        to_connection("mv_conv", "mv_upsample"),
        
        # DCT Residuals Branch: DCT(ΔY_n^g) (frequency coefficients) - PARALLEL INPUT
        # Input: 80×80×64 (higher spatial resolution, more channels)
        # INCREASED vertical offset from 7 to 9 to create CLEAR GAP between MV and DCT layouts
        # Label NOT diagonal - using caption for horizontal positioning
        to_Conv("dct_input", s_filer="", n_filer="", offset="(0,9,0)", to="(0,3,0)", 
                width=2, height=22, depth=14, caption=r"$\mathcal{DCT}(\Delta Y_n^g)$\\$80{\times}80{\times}64$"),  # Y: 7→9 for clear gap
        
        # DCT processing: Conv layers at native resolution - TIGHT SPACING within group
        to_Conv("dct_conv", s_filer="", n_filer="", offset="(1.5,0,0)", to="(dct_input-east)", 
                width=2, height=22, depth=14, caption=r"Conv\\$80{\times}80{\times}32$"),  # Reduced depth: 22→14
        to_connection("dct_input", "dct_conv"),
        
                # ===== FEATURE FUSION ===== 
        # LARGE GAP before fusion to separate encoder groups from fusion group
        # Concat POSITIONED VERY HIGH at Y=7.0 to create MAXIMUM gap with MV
        # MV at Y=0, DCT at Y=9, so Y=7.0 is very close to DCT but far from MV
        # Positioned at X=8 for good horizontal separation from encoders
        # Concatenation of processed MV (upsampled) and DCT features
        # Both now at SAME spatial resolution (80×80) before concat
        # After concat: 80×80×64 (32 MV channels + 32 DCT channels)
        to_Conv("concat", s_filer="", n_filer="", offset="(8,7.0,0)", to="(0,0,0)",  # X=8, Y=7.0 (VERY HIGH)
                width=2.5, height=22, depth=22, caption=r"Concat\\$80{\times}80{\times}64$"),
        to_connection("mv_upsample", "concat"),
        to_connection("dct_conv", "concat"),
        
        # Convolutional processing of fused features - TIGHT SPACING within fusion group
        to_Conv("conv1", s_filer="", n_filer="", offset="(1.5,0,0)", to="(concat-east)", 
                width=3, height=22, depth=22, caption=r"Conv\\$H{\times}W{\times}256$"),
        to_connection("concat", "conv1"),
        
        # Second conv layer - TIGHT SPACING
        to_Conv("conv2", s_filer="", n_filer="", offset="(1.5,0,0)", to="(conv1-east)", 
                width=3, height=22, depth=22, caption=r"Conv\\$H{\times}W{\times}256$"),
        to_connection("conv1", "conv2"),
        
        # ===== TEMPORAL MODELING =====
        # Global Average Pooling: H×W×256 → 1×1×256 - TIGHT SPACING
        # Reduces spatial dimensions while preserving channel information
        to_Pool("global_pool", offset="(1.5,0,0)", to="(conv2-east)", 
                height=7, depth=7, opacity=0.5, caption=r"GAP\\$256$"),
        to_connection("conv2", "global_pool"),
        
        # Flatten to feature vector - TIGHT SPACING
        to_Conv("features", s_filer="", n_filer="", offset="(1.5,0,0)", to="(global_pool-east)", 
                width=1, height=12, depth=1, caption=r"256"),
        to_connection("global_pool", "features"),
        
        # ===== SEQUENTIAL PROCESSING =====
        # BiLSTM: Temporal aggregation of features across GOP
        # Maintains hidden state h_n for sequential predictions - TIGHT SPACING
        to_SoftMax("lstm", s_filer="", offset="(1.5,0,0)", to="(features-east)", 
                   width=2, height=12, depth=12, caption=r"BiLSTM\\256 hidden"),
        to_connection("features", "lstm"),
        
        # ===== DETECTION HEAD =====
        # Output bounding boxes for frame f_n^g - TIGHT SPACING
        # Intermediate features: 128
        to_Conv("bbox_head", s_filer="", n_filer="", offset="(1.5,0,0)", to="(lstm-east)", 
                width=1.5, height=11, depth=11, caption=r"Detection\\128-dim"),
        to_connection("lstm", "bbox_head"),
        
        # ===== OUTPUT =====
        # Predicted bounding boxes: multiple detections per frame - TIGHT SPACING
        # Each bbox: 4 values (x, y, w, h)
        to_Conv("bbox_out", s_filer="", n_filer="", offset="(1.5,0,0)", to="(bbox_head-east)", 
                width=0.5, height=10, depth=1, caption=r"$\{\hat{\mathbf{b}}_i\}$\\$N_{det}{\times}4$"),
        to_connection("bbox_head", "bbox_out"),
        
        # ===== GROUP OVERLAYS WITH PARAMETER COUNTS =====
        # Add colored rectangles to group different processing stages
        # INCREASED horizontal padding to move notations away from borders
        # Fusion components already centered at Y=4.5 (between MV Y=0 and DCT Y=9)
        
                # MV Processing Branch (blue overlay) - bottom branch at Y=-3
        # Covers 3 components: mv_input + mv_conv + mv_upsample
        # FIGURES BETTER CENTERED: Content starts at X=-1.5, spans 8.0 units (ends at 6.5)
        # Content span = 1 + 1.5 + 2 + 1.5 + 2 = 8.0 units (moderate spacing to avoid notation overlap)
        # PADDING: left=2.25, right=3.75 -> Total width: 8.0 + 2.25 + 3.75 = 14.0 units
        # Left edge at -3.75 (ALIGNED with DCT), TITLE manually offset to match DCT
        add_group_overlay(
            name="mv_group",
            corner1="mv_input",           
            corner2="mv_upsample",         
            color="blue",
            label="MV Encoder",
            param_text=r"14K params",
            padding_x_left=2.25,   # Moderate left padding to align edge at -3.75
            padding_x_right=3.75,  # Moderate right padding for 14.0 total width
            title_x_offset=-3.45,  # Manual override to align title with DCT
            padding_y_top=4.5,  # Space for title + params
            padding_y_bot=-5.5, # Bottom padding
            padding_z=3.0,      # Depth padding
            label_position="top"  # Label at top-left
        ),# DCT Processing Branch (green overlay) - top branch at Y=9 (increased from Y=7)
        # Covers 2 components: dct_input + dct_conv  
        # Content span = 2 + 1.5 + 2 = 5.5 units
        # SYMMETRIC PADDING: left=3.75, right=3.75 -> Total: 5.5 + 3.75 + 3.75 = 13.0
        # Left edge at -3.75 (original alignment)
        add_group_overlay(
            name="dct_group",
            corner1="dct_input",           
            corner2="dct_conv",            
            color="green", 
            label="DCT Encoder",
            param_text=r"18K--74K params (8--64 coeffs)",
            padding_x_left=3.75,  # Left padding: back to original
            padding_x_right=3.75, # Right padding: symmetric (original)
            padding_y_top=4.5,  # Matching MV vertical spacing
            padding_y_bot=-4.5, # Bottom padding
            padding_z=3.0,      # Depth padding
            label_position="top"  # Label at top-left
        ),
        
        # Fusion + Temporal Processing (orange overlay) - at Y=7.0
        # Covers ALL components: concat + conv1 + conv2 + GAP + features + lstm + bbox_head + bbox_out
        # Components at Y=7.0 (moved UP significantly from 5.5)
        # INCREASED padding to fully cover all figures and annotations AND title/params
        # Label using corner2 (original position that was working well)
        add_group_overlay(
            name="fusion_group",
            corner1="concat",              
            corner2="bbox_out",            
            color="orange",
            label=r"Fusion \& Temporal",
            param_text=r"196K params",
            padding_x=3.5,      # Horizontal padding for consistency
            padding_y_top=5.5,  # MUCH INCREASED to keep title INSIDE overlay (concat is tall!)
            padding_y_bot=-4.0, # INCREASED bottom padding to capture all annotations
            padding_z=4.0,      # Depth padding (larger for concat depth=22)
            label_position="top",  # Label at top-left
            use_corner2_for_label=True  # Use corner2 for label positioning (keeps original position)
        ),
        
        to_end()
    ]
    
    # Generate the LaTeX file
    output_file = "fast_architecture_esann_notation.tex"
    to_generate(arch, output_file)
    
    print("=" * 70)
    print("✅ Fast DCT-MV Architecture Diagram Generated (esannV2 notation)!")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
    print(f"📝 Compile with: pdflatex {output_file}")
    print(f"📊 Output PDF: {output_file.replace('.tex', '.pdf')}")
    print()
    print("🎨 Notation from esannV2.tex Section 3:")
    print("  • GOP: 𝒢^g = {f_0^g, ..., f_N^g}")
    print("  • I-frame: f_0^g = {DCT(Y_0^g)}")
    print("  • P-frame: f_n^g = {DCT(ΔY_n^g), MV_n^g}")
    print("  • Motion vectors: MV_n^g (2D block displacements)")
    print("  • DCT residuals: DCT(ΔY_n^g) (frequency coefficients)")
    print("  • Feature vector: h_n (encoded representation)")
    print("  • Output: {b̂_i}_{i=1}^{N_det} (set of predicted bounding boxes)")
    print()
    print("🏗️ Architecture Features:")
    print("  • Parallel input branches: MV_n^g and DCT(ΔY_n^g)")
    print("  • Global Pooling (Fast - no ROI)")
    print("  • Simple LSTM (Fast - no attention)")
    print("  • Multiple bounding box predictions per frame")
    print("  • Direct codec-domain processing")
    print()

if __name__ == '__main__':
    main()
