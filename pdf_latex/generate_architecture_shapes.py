#!/usr/bin/env python3
"""
Generate LaTeX Architecture Diagram with Detailed Tensor Shapes
"""

from pathlib import Path


def generate_architecture_with_shapes(output_file):
    """Generate architecture diagram with complete tensor shape annotations."""
    
    print(f"📐 Generating architecture diagram with tensor shapes...")
    
    with open(output_file, 'w') as f:
        f.write("% Fast DCT-MV Tracker Architecture with Tensor Shapes\n")
        f.write("% Requires: \\usepackage{tikz}\n")
        f.write("% Requires: \\usetikzlibrary{positioning,shapes.geometric,arrows.meta,calc}\n\n")
        
        # Architecture with detailed shapes
        f.write("\\begin{figure*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\begin{tikzpicture}[node distance=1.8cm,\n")
        f.write("    box/.style={rectangle, draw, minimum width=3cm, minimum height=1cm, align=center},\n")
        f.write("    input/.style={box, fill=blue!20},\n")
        f.write("    process/.style={box, fill=green!20},\n")
        f.write("    lstm/.style={box, fill=orange!20, minimum width=3cm},\n")
        f.write("    output/.style={box, fill=red!20},\n")
        f.write("    shape/.style={font=\\scriptsize\\ttfamily, text=blue!70!black},\n")
        f.write("    arrow/.style={-Stealth, thick}]\n\n")
        
        # Input branches with shapes
        f.write("    % Input branches\n")
        f.write("    \\node[input] (mv) {Motion Vectors\\\\\\textbf{(MV)}};\n")
        f.write("    \\node[shape, below=0.1cm of mv] {$[B, H, W, 2]$};\n\n")
        
        f.write("    \\node[input, right=2.5cm of mv] (dct) {DCT Coefficients\\\\\\textbf{(DCT)}};\n")
        f.write("    \\node[shape, below=0.1cm of dct] {$[B, H, W, C_{dct}]$};\n\n")
        
        f.write("    \\node[input, right=2.5cm of dct] (iframe) {I-frame Boxes\\\\\\textbf{($B_0$)}};\n")
        f.write("    \\node[shape, below=0.1cm of iframe] {$[B, N, 4]$};\n\n")
        
        # Encoders
        f.write("    % Encoder branches\n")
        f.write("    \\node[process, below=2cm of mv] (mv_conv) {MV Encoder\\\\Conv2d + ReLU};\n")
        f.write("    \\node[shape, below=0.1cm of mv_conv] {$[B, C_{mv}, H, W]$};\n\n")
        
        f.write("    \\node[process, below=2cm of dct] (dct_conv) {DCT Encoder\\\\Conv2d + ReLU};\n")
        f.write("    \\node[shape, below=0.1cm of dct_conv] {$[B, C_{dct}, H, W]$};\n\n")
        
        # Global pooling
        f.write("    % Global pooling\n")
        f.write("    \\node[process, below=2.2cm of mv_conv] (mv_pool) {Global Avg Pool};\n")
        f.write("    \\node[shape, below=0.1cm of mv_pool] {$[B, C_{mv}]$};\n\n")
        
        f.write("    \\node[process, below=2.2cm of dct_conv] (dct_pool) {Global Avg Pool};\n")
        f.write("    \\node[shape, below=0.1cm of dct_pool] {$[B, C_{dct}]$};\n\n")
        
        # Concatenation
        f.write("    % Feature concatenation\n")
        f.write("    \\node[process, below=2.5cm of $(mv_pool)!0.5!(dct_pool)$] (concat) {Concatenate};\n")
        f.write("    \\node[shape, below=0.1cm of concat] {$[B, C_{mv}+C_{dct}]$};\n\n")
        
        # Box embedding
        f.write("    % Box embedding branch\n")
        f.write("    \\node[process, below=2cm of iframe] (box_embed) {Box Embedding\\\\Linear};\n")
        f.write("    \\node[shape, below=0.1cm of box_embed] {$[B, N, D]$};\n\n")
        
        # Expand global context
        f.write("    % Expand global context for broadcasting\n")
        f.write("    \\node[process, below=2.2cm of concat] (expand) {Expand\\\\$\\&$ Concat};\n")
        f.write("    \\node[shape, below=0.1cm of expand] {$[B, N, D+C_{mv}+C_{dct}]$};\n\n")
        
        # LSTM
        f.write("    % LSTM processing\n")
        f.write("    \\node[lstm, below=2.5cm of $(box_embed)!0.5!(expand)$] (lstm) {1-Layer LSTM\\\\(per-object)};\n")
        f.write("    \\node[shape, below=0.1cm of lstm] {$[B, N, H]$};\n\n")
        
        # Prediction heads
        f.write("    % Prediction heads\n")
        f.write("    \\node[output, below=2.2cm of lstm, xshift=-2.5cm] (bbox_head) {BBox Head\\\\Linear};\n")
        f.write("    \\node[shape, below=0.1cm of bbox_head] {$[B, N, 4]$};\n\n")
        
        f.write("    \\node[output, below=2.2cm of lstm, xshift=2.5cm] (conf_head) {Confidence Head\\\\Linear + Sigmoid};\n")
        f.write("    \\node[shape, below=0.1cm of conf_head] {$[B, N, 1]$};\n\n")
        
        # Final output
        f.write("    % Final output\n")
        f.write("    \\node[output, below=2.5cm of $(bbox_head)!0.5!(conf_head)$] (final) {Predicted Boxes\\\\$B_t = B_0 + \\Delta_{box}$};\n")
        f.write("    \\node[shape, below=0.1cm of final] {$[B, N, 4]$};\n\n")
        
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
        f.write("    \\draw[arrow] (expand) -- node[left, font=\\scriptsize] {global context} (lstm);\n")
        f.write("    \\draw[arrow] (lstm) -- (bbox_head);\n")
        f.write("    \\draw[arrow] (lstm) -- (conf_head);\n")
        f.write("    \\draw[arrow] (bbox_head) -- (final);\n")
        f.write("    \\draw[arrow] (conf_head) -- (final);\n\n")
        
        f.write("\\end{tikzpicture}\n")
        f.write("\\caption{Fast DCT-MV Tracker Architecture with Tensor Shapes. ")
        f.write("$B$=batch size, $N$=number of objects, $H \\times W$=spatial dimensions (e.g., 60$\\times$60), ")
        f.write("$C_{mv}$=MV channels (e.g., 64), $C_{dct}$=DCT channels (8/16/32/64), ")
        f.write("$D$=embedding dimension (256), $H$=LSTM hidden size (256).}\n")
        f.write("\\label{fig:fast_architecture_shapes}\n")
        f.write("\\end{figure*}\n\n")
        
        # Shape legend table
        f.write("% Tensor Shape Legend\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Tensor Shape Notation and Typical Values}\n")
        f.write("\\label{tab:tensor_shapes}\n")
        f.write("\\begin{tabular}{clc}\n")
        f.write("\\toprule\n")
        f.write("Symbol & Description & Typical Value \\\\\n")
        f.write("\\midrule\n")
        f.write("$B$ & Batch size & 1--8 \\\\\n")
        f.write("$N$ & Number of objects per frame & 1--50 \\\\\n")
        f.write("$H \\times W$ & Spatial feature map dimensions & 60$\\times$60 \\\\\n")
        f.write("$C_{mv}$ & MV encoder output channels & 64 \\\\\n")
        f.write("$C_{dct}$ & DCT coefficient channels & 8, 16, 32, 64 \\\\\n")
        f.write("$D$ & Box embedding dimension & 256 \\\\\n")
        f.write("$H$ & LSTM hidden state size & 256 \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # Detailed flow table
        f.write("% Detailed Data Flow with Shapes\n")
        f.write("\\begin{table*}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Complete Data Flow Through Architecture}\n")
        f.write("\\label{tab:data_flow}\n")
        f.write("\\begin{tabular}{llll}\n")
        f.write("\\toprule\n")
        f.write("Stage & Layer & Input Shape & Output Shape \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{Motion Vector Branch}} \\\\\n")
        f.write("Input & Motion vectors & --- & $[B, H, W, 2]$ \\\\\n")
        f.write("Encoder & Conv2d(in=2, out=$C_{mv}$, k=3) & $[B, 2, H, W]$ & $[B, C_{mv}, H, W]$ \\\\\n")
        f.write(" & ReLU & $[B, C_{mv}, H, W]$ & $[B, C_{mv}, H, W]$ \\\\\n")
        f.write("Pooling & Global Avg Pool & $[B, C_{mv}, H, W]$ & $[B, C_{mv}]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{DCT Coefficient Branch}} \\\\\n")
        f.write("Input & DCT coefficients & --- & $[B, H, W, C_{dct}]$ \\\\\n")
        f.write("Encoder & Conv2d(in=$C_{dct}$, out=$C_{dct}$, k=3) & $[B, C_{dct}, H, W]$ & $[B, C_{dct}, H, W]$ \\\\\n")
        f.write(" & ReLU & $[B, C_{dct}, H, W]$ & $[B, C_{dct}, H, W]$ \\\\\n")
        f.write("Pooling & Global Avg Pool & $[B, C_{dct}, H, W]$ & $[B, C_{dct}]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{Global Context Fusion}} \\\\\n")
        f.write("Fusion & Concatenate MV + DCT & $[B, C_{mv}], [B, C_{dct}]$ & $[B, C_{mv}+C_{dct}]$ \\\\\n")
        f.write(" & Expand for N objects & $[B, C_{mv}+C_{dct}]$ & $[B, N, C_{mv}+C_{dct}]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{Box Embedding Branch}} \\\\\n")
        f.write("Input & I-frame boxes & --- & $[B, N, 4]$ \\\\\n")
        f.write("Embedding & Linear(4 $\\rightarrow$ D) & $[B, N, 4]$ & $[B, N, D]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{Temporal Modeling}} \\\\\n")
        f.write("Fusion & Concatenate box + context & $[B, N, D], [B, N, C_{mv}+C_{dct}]$ & $[B, N, D+C_{mv}+C_{dct}]$ \\\\\n")
        f.write("LSTM & 1-Layer LSTM(hidden=H) & $[B, N, D+C_{mv}+C_{dct}]$ & $[B, N, H]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{Prediction Heads}} \\\\\n")
        f.write("BBox Head & Linear(H $\\rightarrow$ 4) & $[B, N, H]$ & $[B, N, 4]$ \\\\\n")
        f.write("Conf Head & Linear(H $\\rightarrow$ 1) + Sigmoid & $[B, N, H]$ & $[B, N, 1]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{4}{l}{\\textbf{Final Output}} \\\\\n")
        f.write("Output & $B_t = B_0 + \\Delta_{box}$ & $[B, N, 4], [B, N, 4]$ & $[B, N, 4]$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n\n")
        
        # Example instantiation
        f.write("% Example with Concrete Values\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Example: MV+DCT-32 Variant (Batch=1, 10 Objects)}\n")
        f.write("\\label{tab:example_shapes}\n")
        f.write("\\begin{tabular}{llc}\n")
        f.write("\\toprule\n")
        f.write("Stage & Tensor & Shape \\\\\n")
        f.write("\\midrule\n")
        f.write("Input MV & Motion vectors & $[1, 60, 60, 2]$ \\\\\n")
        f.write("Input DCT & DCT-32 coefficients & $[1, 60, 60, 32]$ \\\\\n")
        f.write("Input Boxes & I-frame detections & $[1, 10, 4]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("MV Encoded & After Conv2d + ReLU & $[1, 64, 60, 60]$ \\\\\n")
        f.write("DCT Encoded & After Conv2d + ReLU & $[1, 32, 60, 60]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("MV Pooled & Global avg pool & $[1, 64]$ \\\\\n")
        f.write("DCT Pooled & Global avg pool & $[1, 32]$ \\\\\n")
        f.write("Global Context & Concatenated & $[1, 96]$ \\\\\n")
        f.write("Expanded Context & Broadcast to N & $[1, 10, 96]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("Box Embedding & After linear & $[1, 10, 256]$ \\\\\n")
        f.write("LSTM Input & Box + context & $[1, 10, 352]$ \\\\\n")
        f.write("LSTM Output & Hidden states & $[1, 10, 256]$ \\\\\n")
        f.write("\\midrule\n")
        f.write("BBox Deltas & Regression head & $[1, 10, 4]$ \\\\\n")
        f.write("Confidence & Classification head & $[1, 10, 1]$ \\\\\n")
        f.write("Final Boxes & $B_0 + \\Delta$ & $[1, 10, 4]$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"   ✅ Architecture with tensor shapes exported to: {output_file}")
    print(f"   📊 Generated:")
    print(f"      - TikZ diagram with shape annotations")
    print(f"      - Tensor shape legend table")
    print(f"      - Complete data flow table")
    print(f"      - Concrete example (MV+DCT-32, 10 objects)")


def main():
    output_dir = Path('experiments/ablation_fast')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / 'architecture_with_shapes.tex'
    
    generate_architecture_with_shapes(output_file)
    
    print(f"\n✨ Architecture diagram with shapes generated successfully!")
    print(f"📝 Include in LaTeX: \\input{{architecture_with_shapes.tex}}")


if __name__ == '__main__':
    main()
