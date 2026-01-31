#!/usr/bin/env python3
"""
Generate LaTeX Architecture Diagram
Produces LaTeX/TikZ code to visualize the Fast DCT-MV Tracker architecture.
"""

import argparse
from pathlib import Path


def generate_architecture_latex(output_file, architecture_type='fast'):
    """
    Generate LaTeX code for architecture visualization.
    
    Args:
        output_file: Path to output .tex file
        architecture_type: 'fast' or 'standard'
    """
    print(f"📐 Generating {architecture_type} architecture LaTeX diagram...")
    
    with open(output_file, 'w') as f:
        f.write("% Fast DCT-MV Tracker Architecture Diagram\n")
        f.write("% Requires: \\usepackage{tikz}\n")
        f.write("% Requires: \\usetikzlibrary{positioning,shapes.geometric,arrows.meta,calc}\n\n")
        
        if architecture_type == 'fast':
            # Fast Architecture (Global Pooling + 1-layer LSTM)
            f.write("\\begin{figure*}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\begin{tikzpicture}[node distance=1.8cm,\n")
            f.write("    box/.style={rectangle, draw, minimum width=2.8cm, minimum height=1cm, align=center},\n")
            f.write("    input/.style={box, fill=blue!20},\n")
            f.write("    process/.style={box, fill=green!20},\n")
            f.write("    lstm/.style={box, fill=orange!20, minimum width=2.8cm},\n")
            f.write("    output/.style={box, fill=red!20},\n")
            f.write("    shape/.style={font=\\tiny\\ttfamily, text=blue!70!black},\n")
            f.write("    arrow/.style={-Stealth, thick}]\n\n")
            
            # Input branches
            f.write("    % Input branches\n")
            f.write("    \\node[input] (mv) {Motion Vectors\\\\\\textbf{(MV)}};\n")
            f.write("    \\node[shape, below=0.05cm of mv] {$[B,H,W,2]$};\n\n")
            
            f.write("    \\node[input, right=of mv] (dct) {DCT Coefficients\\\\\\textbf{(DCT)}};\n")
            f.write("    \\node[shape, below=0.05cm of dct] {$[B,H,W,C_{dct}]$};\n\n")
            
            f.write("    \\node[input, right=of dct] (iframe) {I-frame Boxes\\\\\\textbf{($B_0$)}};\n")
            f.write("    \\node[shape, below=0.05cm of iframe] {$[B,N,4]$};\n\n")
            
            # Encoders
            f.write("    % Encoder branches\n")
            f.write("    \\node[process, below=2cm of mv] (mv_conv) {MV Encoder\\\\Conv2d+ReLU};\n")
            f.write("    \\node[shape, below=0.05cm of mv_conv] {$[B,64,H,W]$};\n\n")
            
            f.write("    \\node[process, below=2cm of dct] (dct_conv) {DCT Encoder\\\\Conv2d+ReLU};\n")
            f.write("    \\node[shape, below=0.05cm of dct_conv] {$[B,C_{dct},H,W]$};\n\n")
            
            # Global pooling
            f.write("    % Global pooling\n")
            f.write("    \\node[process, below=2.3cm of mv_conv] (mv_pool) {Global Avg\\\\Pool};\n")
            f.write("    \\node[shape, below=0.05cm of mv_pool] {$[B,64]$};\n\n")
            
            f.write("    \\node[process, below=2.3cm of dct_conv] (dct_pool) {Global Avg\\\\Pool};\n")
            f.write("    \\node[shape, below=0.05cm of dct_pool] {$[B,C_{dct}]$};\n\n")
            
            # Concatenation
            f.write("    % Feature concatenation\n")
            f.write("    \\node[process, below=2.5cm of $(mv_pool)!0.5!(dct_pool)$] (concat) {Concatenate};\n")
            f.write("    \\node[shape, below=0.05cm of concat] {$[B,64+C_{dct}]$};\n\n")
            
            # Box embedding
            f.write("    % Box embedding branch\n")
            f.write("    \\node[process, below=2cm of iframe] (box_embed) {Box Embed\\\\Linear(4→256)};\n")
            f.write("    \\node[shape, below=0.05cm of box_embed] {$[B,N,256]$};\n\n")
            
            # Expand & concat
            f.write("    % Expand global context\n")
            f.write("    \\node[process, below=2.3cm of concat] (expand) {Expand\\&Concat};\n")
            f.write("    \\node[shape, below=0.05cm of expand] {$[B,N,256+64+C_{dct}]$};\n\n")
            
            # LSTM
            f.write("    % LSTM processing\n")
            f.write("    \\node[lstm, below=2.5cm of $(box_embed)!0.5!(expand)$] (lstm) {1-Layer LSTM\\\\Hidden=256};\n")
            f.write("    \\node[shape, below=0.05cm of lstm] {$[B,N,256]$};\n\n")
            
            # Prediction heads
            f.write("    % Prediction heads\n")
            f.write("    \\node[output, below=2.3cm of lstm, xshift=-2cm] (bbox_head) {BBox Head\\\\Linear(256→4)};\n")
            f.write("    \\node[shape, below=0.05cm of bbox_head] {$[B,N,4]$};\n\n")
            
            f.write("    \\node[output, below=2.3cm of lstm, xshift=2cm] (conf_head) {Conf Head\\\\Linear+Sigmoid};\n")
            f.write("    \\node[shape, below=0.05cm of conf_head] {$[B,N,1]$};\n\n")
            
            # Final output
            f.write("    % Final output\n")
            f.write("    \\node[output, below=2.5cm of $(bbox_head)!0.5!(conf_head)$] (final) {Predicted Boxes\\\\$B_t = B_0 + \\Delta_{box}$};\n")
            f.write("    \\node[shape, below=0.05cm of final] {$[B,N,4]$};\n\n")
            
            # Arrows
            f.write("    % Flow arrows\n")
            f.write("    \\draw[arrow] (mv) -- (mv_conv);\n")
            f.write("    \\draw[arrow] (dct) -- (dct_conv);\n")
            f.write("    \\draw[arrow] (iframe) -- (box_embed);\n")
            f.write("    \\draw[arrow] (mv_conv) -- (mv_pool);\n")
            f.write("    \\draw[arrow] (dct_conv) -- (dct_pool);\n")
            f.write("    \\draw[arrow] (mv_pool) -- (concat);\n")
            f.write("    \\draw[arrow] (dct_pool) -- (concat);\n")
            f.write("    \\draw[arrow] (concat) -- (expand);\n")
            f.write("    \\draw[arrow] (box_embed) -- (lstm);\n")
            f.write("    \\draw[arrow] (expand) -- node[left, font=\\tiny] {context} (lstm);\n")
            f.write("    \\draw[arrow] (lstm) -- (bbox_head);\n")
            f.write("    \\draw[arrow] (lstm) -- (conf_head);\n")
            f.write("    \\draw[arrow] (bbox_head) -- (final);\n")
            f.write("    \\draw[arrow] (conf_head) -- (final);\n\n")
            
            f.write("\\end{tikzpicture}\n")
            f.write("\\caption{\\textbf{Fast Architecture with Tensor Shapes.} ")
            f.write("$B$=batch, $N$=objects, $H \\times W$=spatial (60$\\times$60), ")
            f.write("$C_{dct}$=8/16/32/64 DCT channels. ")
            f.write("Global pooling eliminates ROI operations: 7.8$\\times$ fewer parameters (212K--258K).}\n")
            f.write("\\label{fig:fast_architecture}\n")
            f.write("\\end{figure*}\n\n")
            
        else:
            # Standard Architecture (with ROI Pooling + Attention)
            f.write("\\begin{figure*}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\begin{tikzpicture}[node distance=1.8cm,\n")
            f.write("    box/.style={rectangle, draw, minimum width=2.8cm, minimum height=1cm, align=center},\n")
            f.write("    input/.style={box, fill=blue!20},\n")
            f.write("    process/.style={box, fill=green!20},\n")
            f.write("    attention/.style={box, fill=purple!20},\n")
            f.write("    lstm/.style={box, fill=orange!20, minimum width=2.8cm},\n")
            f.write("    output/.style={box, fill=red!20},\n")
            f.write("    shape/.style={font=\\tiny\\ttfamily, text=blue!70!black},\n")
            f.write("    arrow/.style={-Stealth, thick}]\n\n")
            
            # Input branches
            f.write("    % Input branches\n")
            f.write("    \\node[input] (mv) {Motion Vectors\\\\\\textbf{(MV)}};\n")
            f.write("    \\node[shape, below=0.05cm of mv] {$[B,H,W,2]$};\n\n")
            
            f.write("    \\node[input, right=of mv] (dct) {DCT Coefficients\\\\\\textbf{(DCT)}};\n")
            f.write("    \\node[shape, below=0.05cm of dct] {$[B,H,W,C_{dct}]$};\n\n")
            
            f.write("    \\node[input, right=of dct] (iframe) {I-frame Boxes\\\\\\textbf{($B_0$)}};\n")
            f.write("    \\node[shape, below=0.05cm of iframe] {$[B,N,4]$};\n\n")
            
            # Encoders
            f.write("    % Encoder branches\n")
            f.write("    \\node[process, below=2cm of mv] (mv_conv) {MV Encoder\\\\Conv2d+ReLU};\n")
            f.write("    \\node[shape, below=0.05cm of mv_conv] {$[B,64,H,W]$};\n\n")
            
            f.write("    \\node[process, below=2cm of dct] (dct_conv) {DCT Encoder\\\\Conv2d+ReLU};\n")
            f.write("    \\node[shape, below=0.05cm of dct_conv] {$[B,C_{dct},H,W]$};\n\n")
            
            # ROI pooling
            f.write("    % ROI pooling\n")
            f.write("    \\node[process, below=2.3cm of mv_conv] (mv_roi) {ROI Align\\\\(per-object)};\n")
            f.write("    \\node[shape, below=0.05cm of mv_roi] {$[B,N,64,7,7]$};\n\n")
            
            f.write("    \\node[process, below=2.3cm of dct_conv] (dct_roi) {ROI Align\\\\(per-object)};\n")
            f.write("    \\node[shape, below=0.05cm of dct_roi] {$[B,N,C_{dct},7,7]$};\n\n")
            
            # Concatenation & Flatten
            f.write("    % Concatenation\n")
            f.write("    \\node[process, below=2.5cm of $(mv_roi)!0.5!(dct_roi)$] (concat) {Concat\\&Flatten};\n")
            f.write("    \\node[shape, below=0.05cm of concat] {$[B,N,D]$};\n\n")
            
            # Attention
            f.write("    % Attention mechanism\n")
            f.write("    \\node[attention, below=2.3cm of concat] (attention) {Multi-head\\\\Self-Attention};\n")
            f.write("    \\node[shape, below=0.05cm of attention] {$[B,N,D]$};\n\n")
            
            # Box embedding
            f.write("    % Box embedding\n")
            f.write("    \\node[process, below=2cm of iframe] (box_embed) {Box Embed\\\\Linear(4→512)};\n")
            f.write("    \\node[shape, below=0.05cm of box_embed] {$[B,N,512]$};\n\n")
            
            # Concat box + attention
            f.write("    % Combine features\n")
            f.write("    \\node[process, below=4.8cm of box_embed] (combine) {Concatenate};\n")
            f.write("    \\node[shape, below=0.05cm of combine] {$[B,N,D+512]$};\n\n")
            
            # LSTM
            f.write("    % LSTM\n")
            f.write("    \\node[lstm, below=2.3cm of combine] (lstm) {2-Layer LSTM\\\\Hidden=512};\n")
            f.write("    \\node[shape, below=0.05cm of lstm] {$[B,N,512]$};\n\n")
            
            # Prediction heads
            f.write("    % Prediction heads\n")
            f.write("    \\node[output, below=2.3cm of lstm, xshift=-2cm] (bbox_head) {BBox Head\\\\Linear(512→4)};\n")
            f.write("    \\node[shape, below=0.05cm of bbox_head] {$[B,N,4]$};\n\n")
            
            f.write("    \\node[output, below=2.3cm of lstm, xshift=2cm] (conf_head) {Conf Head\\\\Linear+Sigmoid};\n")
            f.write("    \\node[shape, below=0.05cm of conf_head] {$[B,N,1]$};\n\n")
            
            # Final output
            f.write("    % Final output\n")
            f.write("    \\node[output, below=2.5cm of $(bbox_head)!0.5!(conf_head)$] (final) {Predicted Boxes\\\\$B_t = B_0 + \\Delta_{box}$};\n")
            f.write("    \\node[shape, below=0.05cm of final] {$[B,N,4]$};\n\n")
            
            # Arrows
            f.write("    % Flow arrows\n")
            f.write("    \\draw[arrow] (mv) -- (mv_conv);\n")
            f.write("    \\draw[arrow] (dct) -- (dct_conv);\n")
            f.write("    \\draw[arrow] (iframe) -- (box_embed);\n")
            f.write("    \\draw[arrow] (mv_conv) -- (mv_roi);\n")
            f.write("    \\draw[arrow] (dct_conv) -- (dct_roi);\n")
            f.write("    \\draw[arrow] (mv_roi) -- (concat);\n")
            f.write("    \\draw[arrow] (dct_roi) -- (concat);\n")
            f.write("    \\draw[arrow] (concat) -- (attention);\n")
            f.write("    \\draw[arrow] (box_embed) -- (combine);\n")
            f.write("    \\draw[arrow] (attention) -- node[right, font=\\tiny] {attended} (combine);\n")
            f.write("    \\draw[arrow] (combine) -- (lstm);\n")
            f.write("    \\draw[arrow] (lstm) -- (bbox_head);\n")
            f.write("    \\draw[arrow] (lstm) -- (conf_head);\n")
            f.write("    \\draw[arrow] (bbox_head) -- (final);\n")
            f.write("    \\draw[arrow] (conf_head) -- (final);\n\n")
            
            f.write("\\end{tikzpicture}\n")
            f.write("\\caption{\\textbf{Standard Architecture with Tensor Shapes.} ")
            f.write("ROI Align extracts per-object features (7$\\times$7 spatial), ")
            f.write("multi-head attention models object relationships. ")
            f.write("Total: 1.9M parameters (7.8$\\times$ more than Fast).}\n")
            f.write("\\label{fig:standard_architecture}\n")
            f.write("\\end{figure*}\n\n")
        
        # Add comparison table
        f.write("\n% Architecture Comparison Table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Architecture Comparison: Fast vs Standard}\n")
        f.write("\\label{tab:architecture_comparison}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Component & Standard & Fast \\\\\n")
        f.write("\\midrule\n")
        f.write("Feature Extraction & ROI Align & Global Avg Pool \\\\\n")
        f.write("Attention Mechanism & Multi-head & None \\\\\n")
        f.write("LSTM Layers & 2 & 1 \\\\\n")
        f.write("Hidden Dim & 512 & 256 \\\\\n")
        f.write("\\midrule\n")
        f.write("Total Parameters & 1.9M & 212K--258K \\\\\n")
        f.write("Parameter Reduction & 1$\\times$ & \\textbf{7.8$\\times$} \\\\\n")
        f.write("\\midrule\n")
        f.write("FPS (GPU) & $\\sim$800 & $\\sim$1,600 \\\\\n")
        f.write("Speedup & 1$\\times$ & \\textbf{2$\\times$} \\\\\n")
        f.write("\\midrule\n")
        f.write("mAP @0.5 (Full GOP) & 0.52 & 0.49 \\\\\n")
        f.write("mAP @0.5 (6P) & --- & 0.77--0.81 \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Add variant table
        f.write("% Input Modality Variants\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Fast Architecture Variants: Input Modality Ablation}\n")
        f.write("\\label{tab:architecture_variants}\n")
        f.write("\\begin{tabular}{llcc}\n")
        f.write("\\toprule\n")
        f.write("Variant & Inputs & Channels & Parameters \\\\\n")
        f.write("\\midrule\n")
        f.write("MV-only & Motion vectors only & $C_{mv}$ & 212K \\\\\n")
        f.write("DCT-8 & DCT (8 coeffs) only & $C_{dct}=8$ & 215K \\\\\n")
        f.write("DCT-16 & DCT (16 coeffs) only & $C_{dct}=16$ & 220K \\\\\n")
        f.write("DCT-32 & DCT (32 coeffs) only & $C_{dct}=32$ & 229K \\\\\n")
        f.write("DCT-64 & DCT (64 coeffs) only & $C_{dct}=64$ & 247K \\\\\n")
        f.write("\\midrule\n")
        f.write("MV+DCT-8 & MV + DCT (8 coeffs) & $C_{mv}+8$ & 223K \\\\\n")
        f.write("MV+DCT-16 & MV + DCT (16 coeffs) & $C_{mv}+16$ & 228K \\\\\n")
        f.write("MV+DCT-32 & MV + DCT (32 coeffs) & $C_{mv}+32$ & 237K \\\\\n")
        f.write("MV+DCT-64 & MV + DCT (64 coeffs) & $C_{mv}+64$ & 258K \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"   ✅ LaTeX architecture diagram exported to: {output_file}")
    print(f"   📐 Generated:")
    print(f"      - TikZ architecture diagram")
    print(f"      - Architecture comparison table")
    print(f"      - Input modality variants table")
    print(f"\n   📝 Usage in LaTeX:")
    print(f"      \\input{{{output_file.name}}}")
    print(f"\n   ⚠️  Required packages:")
    print(f"      \\usepackage{{tikz}}")
    print(f"      \\usetikzlibrary{{positioning,shapes.geometric,arrows.meta,calc}}")


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX architecture diagram')
    parser.add_argument('--output', type=str, default='architecture_diagram.tex',
                       help='Output .tex file (default: architecture_diagram.tex)')
    parser.add_argument('--architecture', type=str, choices=['fast', 'standard'], default='fast',
                       help='Architecture type to generate (default: fast)')
    parser.add_argument('--output-dir', type=str, default='experiments/ablation_fast',
                       help='Output directory (default: experiments/ablation_fast)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / args.output
    
    # Generate LaTeX
    generate_architecture_latex(output_file, args.architecture)
    
    print(f"\n✨ Architecture diagram generated successfully!")


if __name__ == '__main__':
    main()
