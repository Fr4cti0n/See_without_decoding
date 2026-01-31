#!/bin/bash

# Generate Architecture Diagram in LaTeX
# Creates TikZ diagram and comparison tables for publication

echo "📐 Generating Architecture LaTeX Diagram"
echo "========================================"
echo ""

OUTPUT_DIR="experiments/ablation_fast"
OUTPUT_FILE="architecture_diagram.tex"
ARCHITECTURE="fast"  # Options: 'fast' or 'standard'

# Run the generator
python generate_architecture_latex.py \
    --output "$OUTPUT_FILE" \
    --architecture "$ARCHITECTURE" \
    --output-dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Architecture diagram generated successfully!"
    echo "📄 Output: $OUTPUT_DIR/$OUTPUT_FILE"
    echo ""
    echo "📝 To use in your LaTeX document:"
    echo "   \\input{$OUTPUT_FILE}"
    echo ""
    echo "⚠️  Required LaTeX packages:"
    echo "   \\usepackage{tikz}"
    echo "   \\usepackage{booktabs}"
    echo "   \\usetikzlibrary{positioning,shapes.geometric,arrows.meta,calc}"
else
    echo ""
    echo "❌ Error: Failed to generate architecture diagram"
    exit 1
fi
