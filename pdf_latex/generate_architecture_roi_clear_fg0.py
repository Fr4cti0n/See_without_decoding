#!/usr/bin/env python3
"""
Generate CLEAR BAFE Architecture Diagram with Compressed Reference RT-DETR (enlarged)

Differences vs. `generate_architecture_roi_clear.py`:
- Entire diagram rendered at 2× scale for higher resolution figures
- Adds an upper pathway processing the compressed reference frame $f_0^g$ using RT-DETR
- RT-DETR warm-starts the previous-boxes input with initial detections before the temporal block
- Keeps original diagram unchanged by writing to a new LaTeX output file
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
    """Generate enlarged ROI architecture with original-frame branch."""
    
    arch = [
        to_head('../PlotNeuralNet'),
        r"\usepackage{amsmath}" + "\n",
        r"\usetikzlibrary{calc}" + "\n",
        r"\usetikzlibrary{backgrounds}" + "\n",
        r"\usetikzlibrary{decorations.pathreplacing}" + "\n",
        to_cor(),
        to_begin(),
        r"\begin{scope}[scale=2.0]" + "\n",
        
        # ===== STAGE 1: INPUTS =====
        r"\coordinate (orig_frame) at (0,12,0);",
        r"\coordinate (dct_frame) at (0,6,0);",
        r"\coordinate (mv_frame) at (0,0,0);",
        r"\coordinate (boxes_prev) at (0,-6,0);",
        
        # P-Frame brace
        r"""
\draw[decorate, decoration={brace, amplitude=18pt, raise=6pt}, line width=2.5pt, black!100] 
    ($(-2.5,-1.2,0) + (-0.7,0,0)$) -- ($(-0.2,6.9,0) + (-3.0,0,0)$) 
    node[midway, left=24pt, font=\LARGE, align=center, text=black] {P-Frame\\[1pt]$f_n^g$};
""",
        r"""
\draw[decorate, decoration={brace, amplitude=18pt, raise=6pt}, line width=2.5pt, teal!80!black] 
    ($(-2.5,7.8,0) + (-0.7,0,0)$) -- ($(-0.2,13.5,0) + (-3.0,0,0)$) 
    node[midway, left=24pt, font=\LARGE, align=center, text=teal!90!black] {Reference\\[1pt]$f_0^g$};
""",
        
        # ===== COMPRESSED REFERENCE (RT-DETR) PATHWAY =====
        r"\node[anchor=east, font=\huge, align=right, text=black] at ($(2,12,0) + (-2.5,0,0)$) {\textbf{Compressed $Y_0^g$}\\[1pt]$120{\times}120{\times}64$};",
        to_Conv("ref_y_conv", s_filer="", n_filer="", offset="(4.0,0,0)", to="(1.75,12,0)",
            width=3.2, height=20, depth=12, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=6.0cm] at ($(ref_y_conv-south) + (0,-1.5,0)$) {\textbf{Conv $s{=}2$}\\[2pt]$60{\times}60{\times}64$};",
        r"\draw [connection] (orig_frame) -- (ref_y_conv-west);",
        r"\coordinate (cbcr_frame) at (0,9,0);",
        r"\node[anchor=east, font=\huge, align=right, text=black] at ($(cbcr_frame) + (0.1,0,0)$) {\textbf{Compressed $Cb/Cr$}\\[1pt]$60{\times}60{\times}64{\times}2$};",
        to_Conv("ref_concat", s_filer="", n_filer="", offset="(9.0,0,0)", to="(ref_y_conv-east)",
            width=3.0, height=20, depth=10, caption=""),
        r"\node[font=\huge, align=center, text=black, text width=6.0cm] at ($(ref_concat-south) + (0,-1.5,0)$) {\textbf{Concat}\\[2pt]$60{\times}60{\times}64{\times}3$};",
        r"\draw [connection] (ref_y_conv-east) -- node[below=28pt, font=\huge, align=center, text=black] {$Y$ Features\\[1pt]$60{\times}60{\times}64$} (ref_concat-west);",
        r"\draw [connection] (cbcr_frame) -- ++(7.3,0) node[pos=0.65, right=44pt, font=\huge, align=center, text=black] {} |- (ref_concat-west);",
        to_Conv("rtdetr_backbone", s_filer="", n_filer="", offset="(9.0,0,0)", to="(ref_concat-east)",
            width=3.4, height=20, depth=12, caption=""),
        r"\node[font=\huge, align=center, text=black, text width=6.4cm] at ($(rtdetr_backbone-south) + (0,-1.5,0)$) {\textbf{RT-DETR Backbone}\\[2pt]$60{\times}60{\times}256$};",
        r"\draw [connection] (ref_concat-east) -- node[below=28pt, font=\huge, align=center, text=black] {Compressed Features\\[1pt]$60{\times}60{\times}64{\times}3$} (rtdetr_backbone-west);",
        to_Conv("rtdetr_encoder", s_filer="", n_filer="", offset="(9.0,0,0)", to="(rtdetr_backbone-east)",
            width=3.0, height=20, depth=12, caption=""),
        r"\node[font=\huge, align=center, text=black, text width=6.0cm] at ($(rtdetr_encoder-south) + (0,-1.5,0)$) {\textbf{RT-DETR Encoder}\\[2pt]$N{\times}256$};",
        r"\draw [connection] (rtdetr_backbone-east) -- node[below=28pt, font=\huge, align=center, text=black] {Flattened Tokens\\[1pt]$N{\times}256$} (rtdetr_encoder-west);",
        to_Conv("rtdetr_decoder", s_filer="", n_filer="", offset="(8.6,0,0)", to="(rtdetr_encoder-east)",
            width=2.8, height=20, depth=10, caption=""),
        r"\node[font=\huge, align=center, text=black, text width=6.4cm] at ($(rtdetr_decoder-south) + (0,-1.6,0)$) {\textbf{RT-DETR Heads}\\[2pt]$N{\times}256$};",
        r"\draw [connection] (rtdetr_encoder-east) -- node[below=28pt, font=\huge, align=center, text=black] {Decoder Queries\\[1pt]$N{\times}256$} (rtdetr_decoder-west);",
        to_Conv("rtdetr_boxes", s_filer="", n_filer="", offset="(9.0,0,0)", to="(rtdetr_decoder-east)",
            width=1.6, height=16, depth=2, caption=""),
        r"\node[font=\huge, align=center, text=black, text width=5.6cm] at ($(rtdetr_boxes-south) + (0,-1.4,0)$) {\textbf{Initial Boxes}\\$bb_0(n)$\\$N{\times}5$};",
        r"\draw [connection] (rtdetr_decoder-east) -- node[below=28pt, font=\huge, align=center, text=black] {RT-DETR Boxes\\[1pt]$N{\times}5$} (rtdetr_boxes-west);",
        
        # ===== BAFE (existing) =====
        r"\node[anchor=east, font=\LARGE, align=right, text=black] at ($(2,6,0) + (-2.3,0,0)$) {\textbf{$\Delta Y_n^g$}\\[1pt]$120{\times}120{\times}64$};",
        to_Conv("dct_conv", s_filer="", n_filer="", offset="(2.0,0,0)", to="(1.75,6,0)",
            width=2.6, height=20, depth=14, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=5.6cm] at ($(dct_conv-south) + (0,-1.4,0)$) {\textbf{Conv $1{\times}1$}\\[2pt]$120{\times}120{\times}32$};",
        r"\draw [connection] (dct_frame) -- (dct_conv-west);",
        to_Conv("dct_roi", s_filer="", n_filer="", offset="(7.0,0,0)", to="(dct_conv-east)",
            width=2.6, height=20, depth=9, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=6.0cm] at ($(dct_roi-south) + (0,-1.4,0)$) {\textbf{BAFE-DCT}\\[2pt]$N{\times}30{\times}30{\times}32$};",
        r"\draw [connection] (dct_conv-east) -- node[below=28pt, font=\huge, align=center, text=black] {Encoded DCT\\[1pt]$120{\times}120{\times}32$} (dct_roi-west);",
        r"\coordinate (dct_stats) at ($(dct_roi-east) + (6.4,0,0)$);",
        r"\draw [connection] (dct_roi-east) -- node[below=28pt, font=\huge, align=center, text=black] {DCT A-Features\\[1pt]$N{\times}128$} (dct_stats);",
        
        r"\node[anchor=east, font=\LARGE, align=right, text=black] at ($(2,0,0) + (-2.3,0,0)$) {\textbf{$MV_n^g$}\\[1pt]$60{\times}60{\times}2$};",
        to_Conv("mv_roi", s_filer="", n_filer="", offset="(3.5,0,0)", to="(1.5,0,0)",
            width=2.6, height=20, depth=9, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=6.0cm] at ($(mv_roi-south) + (0,-1.4,0)$) {\textbf{BAFE-MV}\\[2pt]$N{\times}15{\times}15{\times}2$};",
        r"\draw [connection] (mv_frame) -- (mv_roi-west);",
        r"\coordinate (mv_stats) at ($(mv_roi-east) + (6.4,0,0)$);",
        r"\draw [connection] (mv_roi-east) -- node[below=28pt, font=\huge, align=center, text=black] {MV A-Features\\[1pt]$N{\times}128$} (mv_stats);",

        r"\draw[-Stealth, dashed, line width=2.8pt, draw=green!70!black] (rtdetr_boxes-east) -- ++(6.0,0) |- ($(boxes_prev) + (6.0,-1.0,0)$) -| (boxes_prev);",
        r"\node[anchor=west, font=\huge, text=green!60!black, fill=green!15, inner sep=4pt, rounded corners=4pt, draw=green!40!black, line width=1.4pt] at ($(boxes_prev) + (1,-0.65,0)$) {Warm-start $bb(n{-}1)$ input};",
        
        # ===== BOX EMBEDDING =====
        r"\node[anchor=east, font=\LARGE, align=right, text=black] at ($(2,-6,0) + (-0.3,0,0)$) {\textbf{$bb(n{-}1)$}\\[1pt]$N{\times}5$};",
            to_Conv("box_enc", s_filer="", n_filer="", offset="(3.5,0,0)", to="(3.7,-6,0)",
                width=2.4, height=16, depth=2, caption=""),
            r"\node[font=\LARGE, align=center, text=black, text width=5.0cm] at ($(box_enc-south) + (0,-1.2,0)$) {\textbf{MLP}\\$5{\rightarrow}32$};",
        r"\draw [connection] (boxes_prev) -- (box_enc-west);",
        
        # ===== FUSION =====
        to_Conv("concat", s_filer="", n_filer="", offset="(1.8,-10.5,0)", to="($(dct_stats) + (3.5,0,0)$)",
            width=2.8, height=20, depth=13, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=5.6cm] at ($(concat-south) + (0,-1.4,0)$) {\textbf{Concat}\\$N{\times}288$};",
        r"\draw [connection] (mv_stats) -- ++(1.5,0) |- (concat-west);",
        r"\draw [connection] (dct_stats) -- ++(1.5,0) |- (concat-west);",
        r"\draw [connection] (box_enc-east) -- node[above=28pt, font=\huge, align=center, text=black] {Encoded Boxes\\[1pt]$N{\times}32$} ++(5.5,0) |- (concat-west);",
        
        to_SoftMax("lstm", s_filer="", offset="(2.2,0,0)", to="(concat-east)",
                   width=3.0, height=20, depth=13, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=6.0cm] at ($(lstm-south) + (0,-1.4,0)$) {\textbf{BiLSTM}\\$N{\times}128$};",
        to_connection("concat", "lstm"),
        
        r"\coordinate (refinement) at ($(lstm-east) + (6.0,0,0)$);",
        r"\draw [connection] (lstm-east) -- node[below=28pt, font=\huge, align=center, text=black] {Refinements\\[1pt]$N{\times}5$} (refinement);",
        to_Conv("boxes_out", s_filer="", n_filer="", offset="(0,0,0)", to="(refinement)",
                width=1.0, height=16, depth=2, caption=""),
        r"\node[font=\LARGE, align=center, text=black, text width=5.4cm] at ($(boxes_out-south) + (0,-1.3,0)$) {\textbf{ABoxes}\\$bb(n)$\\$N{\times}5$};",
        r"\draw [connection] (refinement) -- (boxes_out-west);",
        r"\draw[-Stealth, red!70!black, line width=2.8pt, dashed, opacity=0.8] (boxes_out-east) -- ++(2.4,0) node[pos=0.55, right=10pt, font=\bfseries\Large, text=red!75!black] {Sequentially updates $bb(n{-}1)$} |- ++(0,-5.0) -| (boxes_prev);",
        
        # ===== OVERLAYS =====
        r"""
% Input Stage overlay (extended for orig frame)
\begin{scope}[on background layer]
\fill[purple!8, rounded corners=15pt] 
    ($(0,-6,0) + (-6.3, -2.5, -2.5)$) rectangle 
    ($(1,12,0) + (0.3, 2.90, 2.5)$);
\draw[purple!70!black, line width=3pt, rounded corners=15pt] 
    ($(0,-6,0) + (-6.3, -2.5, -2.5)$) rectangle 
    ($(1,12,0) + (0.3, 2.90, 2.5)$);
\end{scope}
\node[anchor=north west, font=\bfseries\LARGE, text=purple!90!black, fill=purple!15, inner sep=5pt, rounded corners=5pt, draw=purple!70!black, line width=2pt] at 
    ($(-1.5,10.5,0) + (-3.5, 3.0, 0)$) {Inputs};
""",
        r"""
% Compressed Reference Processing overlay
\begin{scope}[on background layer]
\fill[teal!8, rounded corners=15pt] 
    ($(ref_y_conv-nearsouthwest) + (-4.2, -3.4, -3.2)$) rectangle 
    ($(rtdetr_boxes-nearnortheast) + (7.6, 2.4, 3.2)$);
\draw[teal!70!black, line width=3pt, rounded corners=15pt] 
    ($(ref_y_conv-nearsouthwest) + (-4.2, -3.4, -3.2)$) rectangle 
    ($(rtdetr_boxes-nearnortheast) + (7.6, 2.4, 3.2)$);
\end{scope}
\node[anchor=north west, font=\bfseries\huge, text=teal!90!black, fill=teal!15, inner sep=5pt, rounded corners=5pt, draw=teal!70!black, line width=2pt] at 
    ($(rtdetr_boxes-nearnortheast) + (-0.8, 1.0, 0)$) {Compressed Reference RT-DETR};
\node[anchor=north west, font=\bfseries\LARGE, text=teal!90!black, fill=teal!15, inner sep=4pt, rounded corners=4pt, draw=teal!50!black, line width=1pt] at 
    ($(rtdetr_boxes-nearnortheast) + (-0.8, 0.0, 0)$) {\textbf{Warm-starts Initial boxes}};
\node[anchor=north west, font=\bfseries\LARGE, text=teal!90!black, fill=teal!15, inner sep=4pt, rounded corners=4pt, draw=teal!50!black, line width=1pt] at 
    ($(rtdetr_boxes-nearnortheast) + (0.8, -1.2, 0)$) {\textbf{~28M params}};
""",
        r"""
% BAFE Extraction overlay (MV + DCT)
\begin{scope}[on background layer]
\fill[blue!8, rounded corners=15pt] 
    ($(mv_roi-nearsouthwest) + (-3.9, -7.7, -3.4)$) rectangle 
    ($(dct_roi-nearnortheast) + (8.6, 2.9, 3.4)$);
\draw[blue!70!black, line width=3pt, rounded corners=15pt] 
    ($(mv_roi-nearsouthwest) + (-3.9, -7.7, -3.4)$) rectangle 
    ($(dct_roi-nearnortheast) + (8.6, 2.9, 3.4)$);
\end{scope}
\node[anchor=north west, font=\bfseries\huge, text=blue!90!black, fill=blue!15, inner sep=5pt, rounded corners=5pt, draw=blue!70!black, line width=2pt] at 
    ($(dct_roi-nearnortheast) + (-5.5, 1.3, 0)$) {\textbf{BAFE Extraction}};
\node[anchor=north west, font=\bfseries\LARGE, text=blue!90!black, fill=blue!15, inner sep=4pt, rounded corners=4pt, draw=blue!50!black, line width=1pt] at 
    ($(dct_roi-nearnortheast) + (-6.0, 0.5, 0)$) {\textbf{~70K params}};
""",
        r"""
% Fusion & Temporal overlay (updated concat size)
\begin{scope}[on background layer]
\fill[orange!8, rounded corners=15pt] 
    ($(concat-nearsouthwest) + (-4.6, -8.35, -3.5)$) rectangle 
    ($(boxes_out-nearnortheast) + (10.15, 7.1, 3.5)$);
\draw[orange!70!black, line width=3pt, rounded corners=15pt] 
    ($(concat-nearsouthwest) + (-4.6, -8.35, -3.5)$) rectangle 
    ($(boxes_out-nearnortheast) + (10.15, 7.1, 3.5)$);
\end{scope}
\node[anchor=north west, font=\bfseries\huge, text=orange!90!black, fill=orange!15, inner sep=5pt, rounded corners=5pt, draw=orange!70!black, line width=2pt] at 
    ($(concat-nearnortheast) + (0, 5.6, 0)$) {Fusion \& Temporal};
\node[anchor=north west, font=\bfseries\LARGE, text=orange!90!black, fill=orange!15, inner sep=4pt, rounded corners=4pt, draw=orange!50!black, line width=1pt] at 
    ($(concat-nearnortheast) + (0, 4.2, 0)$) {\textbf{~220K params}};
""",
        
        r"\end{scope}",
        to_end()
    ]
    
    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, "bafe_architecture_fg0.tex")
    to_generate(arch, output_file)
    
    print("=" * 70)
    print("✅ BAFE Architecture with Compressed Reference RT-DETR Generated!")
    print("=" * 70)
    rel_output = os.path.relpath(output_file, start=os.getcwd())
    print(f"📄 Output file: {rel_output}")
    print(f"📝 Compile with: pdflatex {rel_output}")
    print()
    print("🎯 Enhancements:")
    print("  • Diagram rendered at 2× scale for publication-quality exports")
    print("  • Compressed reference pathway now follows standard RT-DETR")
    print("  • RT-DETR outputs initial boxes to warm-start the $bb(n{-}1)$ input")
    print()
    print("📊 Updated Feature Dimensions:")
    print("  • RT-DETR branch → initial bounding boxes (N×5) for warm-start")
    print("  • Concat now aggregates MV + DCT + box embeddings → N×288")
    print()
    print("⚙️  Parameter Overview:")
    print("  • Compressed Reference RT-DETR: standard backbone/encoder/heads")
    print("  • MV/DCT BAFE ≈ 70K params")
    print("  • Fusion & Temporal ≈ 220K params")
    print()
    print("🔁 Temporal Loop remains identical to base architecture")
    print()

if __name__ == '__main__':
    main()
