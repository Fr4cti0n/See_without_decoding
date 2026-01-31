#!/usr/bin/env python3
"""
Generate Fast DCT-MV Architecture Diagram using PlotNeuralNet
Based on: https://github.com/HarisIqbal88/PlotNeuralNet

This script generates a TikZ diagram of the Fast DCT-MV tracker architecture.
"""

import sys
import os

# Add PlotNeuralNet to path (parent directory)
plot_neural_net_path = os.path.join(os.path.dirname(__file__), '..', 'PlotNeuralNet')
sys.path.append(plot_neural_net_path)

from pycore.tikzeng import *

def main():
    """
    Generate the Fast DCT-MV architecture diagram.
    
    Architecture flow:
    1. Input: Motion Vectors (2 channels) + Optional DCT Residuals (0-64 channels)
    2. Feature Encoder: Conv layers
    3. Global Pooling: Spatial aggregation
    4. LSTM: Temporal modeling
    5. Detection Head: Class + BBox predictions
    """
    
    arch = [
        to_head('../PlotNeuralNet'),  # Reference parent directory
        to_cor(),
        to_begin(),
        
        # ===== INPUT LAYERS =====
        # Motion Vectors Input (2 channels: x, y)
        to_Conv("mv_input", s_filer="", n_filer=2, offset="(0,0,0)", to="(0,0,0)", 
                width=1, height=40, depth=40, caption="MV Input (2ch)"),
        
        # DCT Residuals Input (0-64 channels, optional)
        to_Conv("dct_input", s_filer="", n_filer="0-64", offset="(1.5,0,0)", to="(mv_input-east)", 
                width=2, height=40, depth=40, caption="DCT Residuals"),
        to_connection("mv_input", "dct_input"),
        
        # ===== FEATURE ENCODER =====
        # Concatenation
        to_Conv("concat", s_filer="", n_filer="2-66", offset="(2,0,0)", to="(dct_input-east)", 
                width=2, height=40, depth=40, caption="Concat"),
        to_connection("dct_input", "concat"),
        
        # Conv layer 3x3
        to_Conv("conv1", s_filer=256, n_filer=256, offset="(2,0,0)", to="(concat-east)", 
                width=4, height=35, depth=35, caption="Conv 3×3"),
        to_connection("concat", "conv1"),
        
        # Conv layer 3x3
        to_Conv("conv2", s_filer=256, n_filer=256, offset="(0.5,0,0)", to="(conv1-east)", 
                width=4, height=35, depth=35, caption="Conv 3×3"),
        to_connection("conv1", "conv2"),
        
        # ===== GLOBAL POOLING (Fast Architecture) =====
        # Global average pooling - key Fast component!
        to_Pool("global_pool", offset="(2,0,0)", to="(conv2-east)", 
                width=1, height=10, depth=10, opacity=0.5, caption="Global Pool"),
        to_connection("conv2", "global_pool"),
        
        # Feature vector after pooling
        to_Conv("features", s_filer="", n_filer=256, offset="(1.5,0,0)", to="(global_pool-east)", 
                width=0.5, height=20, depth=1, caption="Features"),
        to_connection("global_pool", "features"),
        
        # ===== LSTM LAYER (Simple LSTM - Fast Architecture) =====
        # Simple LSTM without attention - key Fast component!
        to_SoftMax("lstm", s_filer=256, offset="(2,0,0)", to="(features-east)", 
                   width=2, height=20, depth=20, opacity=0.8, caption="LSTM"),
        to_connection("features", "lstm"),
        
        # Hidden state feedback (self-connection)
        to_skip("lstm", "lstm", pos=1.8),
        
        # ===== DETECTION HEAD =====
        # Classification head
        to_Conv("cls_head", s_filer=128, n_filer=128, offset="(3,2,0)", to="(lstm-east)", 
                width=1.5, height=15, depth=15, caption="Class Head"),
        to_connection("lstm", "cls_head"),
        
        # Bounding box head
        to_Conv("bbox_head", s_filer=128, n_filer=128, offset="(0,-4,0)", to="(cls_head-south)", 
                width=1.5, height=15, depth=15, caption="BBox Head"),
        to_connection("lstm", "bbox_head"),
        
        # ===== OUTPUTS =====
        # Class output
        to_Conv("cls_out", s_filer="", n_filer=1, offset="(1.5,0,0)", to="(cls_head-east)", 
                width=0.5, height=8, depth=1, caption="Class"),
        to_connection("cls_head", "cls_out"),
        
        # BBox output (4 values: x, y, w, h)
        to_Conv("bbox_out", s_filer="", n_filer=4, offset="(1.5,0,0)", to="(bbox_head-east)", 
                width=0.5, height=12, depth=1, caption="BBox Output"),
        to_connection("bbox_head", "bbox_out"),
        
        to_end()
    ]
    
    # Generate the LaTeX file
    output_file = "fast_dct_mv_architecture.tex"
    to_generate(arch, output_file)
    
    print("=" * 60)
    print("✅ Fast DCT-MV Architecture Diagram Generated!")
    print("=" * 60)
    print(f"📄 Output file: {output_file}")
    print(f"📝 Compile with: bash PlotNeuralNet/tikzmake.sh {output_file.replace('.tex', '')}")
    print(f"    or manually: pdflatex {output_file}")
    print(f"📊 Output PDF: {output_file.replace('.tex', '.pdf')}")
    print()
    print("🎨 Key Features Shown:")
    print("  • Motion Vector input (2 channels)")
    print("  • Optional DCT residual input (0-64 channels)")
    print("  • Feature encoder (Conv layers)")
    print("  • Global Pooling (Fast architecture - no ROI)")
    print("  • Simple LSTM (Fast architecture - no attention)")
    print("  • Detection heads (class + bounding box)")
    print()

if __name__ == '__main__':
    main()
