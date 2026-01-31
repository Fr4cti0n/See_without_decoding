#!/usr/bin/env python3
"""
Generate Fast DCT-MV Architecture Diagram using PlotNeuralNet
Based on: https://github.com/HarisIqbal88/PlotNeuralNet

This script generates a TikZ diagram of the Fast DCT-MV tracker architecture
showing the flow from input (MV/DCT) through global pooling and LSTM to predictions.
"""

import sys
import os

# Add PlotNeuralNet to path
plot_neural_net_path = os.path.join(os.path.dirname(__file__), 'PlotNeuralNet')
sys.path.append(plot_neural_net_path)

# Import PlotNeuralNet components
from pycore.tikzeng import *
from pycore.blocks import *

def create_fast_dct_mv_architecture():
    """
    Generate the Fast DCT-MV architecture diagram.
    
    Architecture flow:
    1. Input: Motion Vectors (2 channels) + Optional DCT Residuals (0-64 channels)
    2. Global Pooling: Spatial aggregation across entire frame
    3. LSTM: Temporal modeling with hidden state
    4. Output: Bounding box predictions (class, bbox, confidence)
    """
    
    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),
        
        # Title
        to_text("Fast DCT-MV Object Tracker", size=20, y=6),
        
        # ===== INPUT LAYER =====
        to_input('../examples/fcn8s/cat.jpg', to='(-3,0,0)', width=8, height=8),
        to_text("Input Frame", size=10, y=-1.5),
        
        # Motion Vectors Input
        to_Conv(
            name="mv_input",
            s_filer="",
            n_filer=2,
            offset="(0,0,0)",
            to="(0,0,0)",
            width=2,
            height=40,
            depth=40,
            caption="Motion Vectors"
        ),
        to_text("2 channels\\\\(x, y)", size=8, y=-6),
        
        # DCT Residuals Input (optional)
        to_Conv(
            name="dct_input",
            s_filer="",
            n_filer="0-64",
            offset="(2,0,0)",
            to="(mv_input-east)",
            width=4,
            height=40,
            depth=40,
            caption="DCT Residuals"
        ),
        to_text("0-64 coefficients\\\\(optional)", size=8, y=-6),
        
        # Connection arrow
        to_connection("mv_input", "dct_input"),
        
        # ===== ENCODER LAYER =====
        to_text("Feature Encoder", size=12, y=8, x=5),
        
        # Concatenation
        to_Conv(
            name="concat",
            s_filer="",
            n_filer="2-66",
            offset="(3,0,0)",
            to="(dct_input-east)",
            width=3,
            height=40,
            depth=40,
            caption="Concat"
        ),
        to_connection("dct_input", "concat"),
        
        # Conv layer
        to_Conv(
            name="conv1",
            s_filer=256,
            n_filer=256,
            offset="(2,0,0)",
            to="(concat-east)",
            width=6,
            height=35,
            depth=35,
            caption="Conv 3×3"
        ),
        to_connection("concat", "conv1"),
        
        # ===== GLOBAL POOLING =====
        to_text("Global Pooling", size=12, y=8, x=11),
        
        to_Pool(
            name="global_pool",
            offset="(2,0,0)",
            to="(conv1-east)",
            width=2,
            height=10,
            depth=10,
            opacity=0.5,
            caption="Global\\\\Avg Pool"
        ),
        to_connection("conv1", "global_pool"),
        
        # Feature vector
        to_Conv(
            name="features",
            s_filer="",
            n_filer=256,
            offset="(2,0,0)",
            to="(global_pool-east)",
            width=1,
            height=25,
            depth=1,
            caption="Features"
        ),
        to_connection("global_pool", "features"),
        
        # ===== LSTM LAYER =====
        to_text("Temporal Modeling", size=12, y=8, x=17),
        
        to_SoftMax(
            name="lstm",
            s_filer=256,
            offset="(3,0,0)",
            to="(features-east)",
            width=3,
            height=25,
            depth=25,
            caption="LSTM"
        ),
        to_connection("features", "lstm"),
        
        # Hidden state feedback
        to_skip(
            of='lstm',
            to='lstm',
            pos=2.5,
            style="dashed"
        ),
        to_text("Hidden State", size=8, y=4, x=17),
        
        # ===== OUTPUT LAYER =====
        to_text("Detection Head", size=12, y=8, x=22),
        
        # Classification branch
        to_Conv(
            name="cls_head",
            s_filer=128,
            n_filer=128,
            offset="(3,2,0)",
            to="(lstm-east)",
            width=2,
            height=15,
            depth=15,
            caption="Class\\\\Head"
        ),
        to_connection("lstm", "cls_head"),
        
        # Bounding box branch
        to_Conv(
            name="bbox_head",
            s_filer=128,
            n_filer=128,
            offset="(0,-4,0)",
            to="(cls_head-south)",
            width=2,
            height=15,
            depth=15,
            caption="BBox\\\\Head"
        ),
        to_connection("lstm", "bbox_head"),
        
        # Final outputs
        to_Conv(
            name="cls_out",
            s_filer="",
            n_filer=1,
            offset="(2,0,0)",
            to="(cls_head-east)",
            width=1,
            height=8,
            depth=1,
            caption="Class"
        ),
        to_connection("cls_head", "cls_out"),
        
        to_Conv(
            name="bbox_out",
            s_filer="",
            n_filer=4,
            offset="(2,0,0)",
            to="(bbox_head-east)",
            width=1,
            height=12,
            depth=1,
            caption="BBox"
        ),
        to_connection("bbox_head", "bbox_out"),
        
        # ===== ANNOTATIONS =====
        # Mark Fast components
        to_text("\\textbf{Fast Components:}", size=10, y=-8, x=0),
        to_text("• Global Pooling (not ROI)", size=8, y=-9, x=0),
        to_text("• Simple LSTM (no attention)", size=8, y=-10, x=0),
        to_text("• 2-3× faster than standard", size=8, y=-11, x=0),
        
        # Mark ablation options
        to_text("\\textbf{Ablation Variants:}", size=10, y=-8, x=12),
        to_text("• MV-only: 2 input channels", size=8, y=-9, x=12),
        to_text("• DCT-8/16/32/64: 8-64 DCT coeffs", size=8, y=-10, x=12),
        to_text("• MV+DCT: Combined features", size=8, y=-11, x=12),
        
        to_end()
    ]
    
    # Generate the LaTeX file
    output_file = "fast_dct_mv_architecture.tex"
    
    with open(output_file, 'w') as f:
        for item in arch:
            f.write(item)
    
    print(f"✅ Architecture diagram generated: {output_file}")
    print(f"📝 Compile with: pdflatex {output_file}")
    print(f"📊 Output will be: fast_dct_mv_architecture.pdf")
    
    return output_file


def create_architecture_comparison():
    """
    Generate a side-by-side comparison of Standard vs Fast architecture.
    """
    
    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),
        
        # Title
        to_text("Architecture Comparison: Standard vs Fast", size=20, y=8),
        
        # ===== STANDARD ARCHITECTURE (LEFT) =====
        to_text("\\textbf{Standard Architecture}", size=14, y=6, x=-8),
        
        # Standard: ROI Pooling
        to_Conv(
            name="std_input",
            s_filer=256,
            n_filer=256,
            offset="(-12,0,0)",
            to="(0,0,0)",
            width=4,
            height=30,
            depth=30,
            caption="Features"
        ),
        
        to_Pool(
            name="roi_pool",
            offset="(2,0,0)",
            to="(std_input-east)",
            width=3,
            height=20,
            depth=20,
            opacity=0.5,
            caption="ROI\\\\Pooling"
        ),
        to_connection("std_input", "roi_pool"),
        
        # Standard: Attention LSTM
        to_SoftMax(
            name="attn_lstm",
            s_filer=256,
            offset="(2,0,0)",
            to="(roi_pool-east)",
            width=4,
            height=25,
            depth=25,
            caption="Attention\\\\LSTM"
        ),
        to_connection("roi_pool", "attn_lstm"),
        
        # Standard metrics
        to_text("Speed: 1.0×", size=9, y=-4, x=-8),
        to_text("Memory: High", size=9, y=-5, x=-8),
        to_text("Per-object: ✓", size=9, y=-6, x=-8),
        
        # ===== FAST ARCHITECTURE (RIGHT) =====
        to_text("\\textbf{Fast Architecture}", size=14, y=6, x=8),
        
        # Fast: Global Pooling
        to_Conv(
            name="fast_input",
            s_filer=256,
            n_filer=256,
            offset="(4,0,0)",
            to="(0,0,0)",
            width=4,
            height=30,
            depth=30,
            caption="Features"
        ),
        
        to_Pool(
            name="global_pool",
            offset="(2,0,0)",
            to="(fast_input-east)",
            width=2,
            height=10,
            depth=10,
            opacity=0.5,
            caption="Global\\\\Pool"
        ),
        to_connection("fast_input", "global_pool"),
        
        # Fast: Simple LSTM
        to_SoftMax(
            name="simple_lstm",
            s_filer=256,
            offset="(2,0,0)",
            to="(global_pool-east)",
            width=3,
            height=20,
            depth=20,
            caption="Simple\\\\LSTM"
        ),
        to_connection("global_pool", "simple_lstm"),
        
        # Fast metrics
        to_text("Speed: 2-3×", size=9, y=-4, x=8),
        to_text("Memory: Low", size=9, y=-5, x=8),
        to_text("Per-object: ✗", size=9, y=-6, x=8),
        
        to_end()
    ]
    
    output_file = "architecture_comparison.tex"
    
    with open(output_file, 'w') as f:
        for item in arch:
            f.write(item)
    
    print(f"✅ Comparison diagram generated: {output_file}")
    
    return output_file


if __name__ == '__main__':
    print("=" * 60)
    print("Fast DCT-MV Architecture Diagram Generator")
    print("Using PlotNeuralNet: https://github.com/HarisIqbal88/PlotNeuralNet")
    print("=" * 60)
    print()
    
    # Check if PlotNeuralNet is available
    try:
        from pycore.tikzeng import *
        print("✅ PlotNeuralNet found!")
    except ImportError:
        print("❌ PlotNeuralNet not found!")
        print()
        print("Please install PlotNeuralNet:")
        print("  cd /path/to/MOTS-experiments")
        print("  git clone https://github.com/HarisIqbal88/PlotNeuralNet.git")
        print()
        sys.exit(1)
    
    print("\n📊 Generating architecture diagrams...\n")
    
    # Generate main architecture
    arch_file = create_fast_dct_mv_architecture()
    
    print()
    
    # Generate comparison
    comp_file = create_architecture_comparison()
    
    print()
    print("=" * 60)
    print("✅ Generation complete!")
    print("=" * 60)
    print()
    print("📝 To compile:")
    print(f"  pdflatex {arch_file}")
    print(f"  pdflatex {comp_file}")
    print()
    print("📄 Include in your LaTeX document:")
    print(f"  \\input{{{arch_file}}}")
    print(f"  \\input{{{comp_file}}}")
    print()
