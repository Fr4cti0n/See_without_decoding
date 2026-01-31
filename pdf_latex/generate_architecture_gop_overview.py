#!/usr/bin/env python3
"""Generate a high-level GOP processing overview diagram.

The diagram summarizes how compressed I-frames and P-frames flow through two
specialized detectors (RT-DETRv2Comp and RecMamba) before fusing detections into
joint GOP-level outputs.
"""

import os
import sys

# Add PlotNeuralNet utilities to path (parent directory).
plot_neural_net_path = os.path.join(os.path.dirname(__file__), '..', 'PlotNeuralNet')
sys.path.append(plot_neural_net_path)

from pycore.tikzeng import to_begin, to_end, to_head, to_cor, to_generate, to_Conv

def main() -> None:
    arch = [
        to_head('../PlotNeuralNet'),
        r"\usetikzlibrary{calc}" + "\n",
        r"\usetikzlibrary{positioning}" + "\n",
        r"\usetikzlibrary{backgrounds}" + "\n",
        r"\usetikzlibrary{decorations.pathreplacing}" + "\n",
        r"\usetikzlibrary{fit}" + "\n",
        r"\usetikzlibrary{arrows.meta}" + "\n",
        to_cor(),
        to_begin(),
        r"\begin{scope}[scale=1.45]" + "\n",

        # ================= INPUT GOP =================
        r"\node[inner sep=0pt, rounded corners=5pt, draw=gray!60!black, line width=0.9pt]"
        r"      (rgb_sample_0) at (-20.0,15.3,0) {\includegraphics[width=3.1cm]{assets/avion_rgb/frame_0000.png}};",
        r"\node[inner sep=0pt, rounded corners=5pt, draw=gray!60!black, line width=0.9pt]"
        r"      (rgb_sample_1) at ($(rgb_sample_0.south) + (0,-1.65,0)$) {\includegraphics[width=3.1cm]{assets/avion_rgb/frame_0004.png}};",
        r"\node[font=\Huge, text=gray!70!black]"
        r"      (rgb_sample_dots) at ($(rgb_sample_1.south) + (0,-1.35,0)$) {\vdots};",
        r"\node[inner sep=0pt, rounded corners=5pt, draw=gray!60!black, line width=0.9pt]"
        r"      (rgb_sample_last) at ($(rgb_sample_dots.south) + (0,-1.45,0)$) {\includegraphics[width=3.1cm]{assets/avion_rgb/frame_0011.png}};",

        r"\node[font=\small\bfseries, text=gray!60!black] at ($(rgb_sample_0.south) + (0,-0.32)$) {Frame $I_0$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(rgb_sample_1.south) + (0,-0.32)$) {Frame $P_1$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(rgb_sample_last.south) + (0,-0.32)$) {Frame $P_{11}$};",

        r"\node[draw=black!70!gray, line width=1.6pt, rounded corners=10pt,"
        r"      inner xsep=14pt, inner ysep=22pt, fit={(rgb_sample_0) (rgb_sample_1) (rgb_sample_last) (rgb_sample_dots)}] (gop_block) {};",
        r"\node[anchor=north, font=\Large\bfseries, text=gray!70!black]"
        r"      at ($(gop_block.north) + (0,0.60,0)$) {Group of Pictures};",

        r"\node[draw=black!70!gray, line width=1.6pt, fill=black!6, rounded corners=8pt,"
        r"      minimum width=4.4cm, minimum height=2.6cm, align=center, font=\Large,"
        r"      name=camera_src] at ($(rgb_sample_1.west) + (-2.8,-1.25,0)$) {\textbf{Camera feed}\\Raw sensor frames\\{\large GOP$_n$: I$_0$ + 11 P}};",

        r"\node[draw=black!70!gray, line width=1.6pt, fill=black!6, rounded corners=8pt,"
        r"      minimum width=4.4cm, minimum height=2.4cm, align=center, font=\Large,"
        r"      name=codec] at ($(camera_src.east) + (6.3,-0.0,0)$) {\textbf{Compression codec}\\Mpeg4 Part2 \\ Encoding};",

        # Flow arrows: camera to frames with a single diagonal then horizontal; frames straight into codec
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=purple!75!black]"
        r"    (camera_src.east) -- ($(rgb_sample_0.west)+(-0.35,0)$) -- (rgb_sample_0.west);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=purple!75!black]"
        r"    (rgb_sample_0.east) -- ($(rgb_sample_0.east)+(0.2,0)$) -- (codec.west);",

        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!70!black]"
        r"    (camera_src.east) -- ($(rgb_sample_1.west)+(-0.35,0)$) -- (rgb_sample_1.west);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!70!black]"
        r"    (rgb_sample_1.east) -- ($(rgb_sample_1.east)+(0.2,0)$) -- (codec.west);",

        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!70!black]"
        r"    (camera_src.east) -- ($(rgb_sample_last.west)+(-0.35,0)$) -- (rgb_sample_last.west);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!70!black]"
        r"    (rgb_sample_last.east) -- ($(rgb_sample_last.east)+(0.2,0)$) -- (codec.west);",

        # Camera feed arrows removed per request
        "\n% Codec outputs (motion vectors and residuals)\n",
        r"\path let \p1 = (rgb_sample_1), \p2 = (codec.east) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=blue!60!black, line width=0.9pt]"
        r"      (mv_sample_0) at (\x2+2.2cm,\y1) {\includegraphics[width=3.0cm]{assets/avion_rgb/motion_0004.png}};",
        r"\node[font=\Huge, text=blue!60!black]"
        r"      (mv_sample_dots) at ($(mv_sample_0.south) + (0,-1.30)$) {\vdots};",
        r"\path let \p1 = (rgb_sample_last), \p2 = (codec.east) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=blue!60!black, line width=0.9pt]"
        r"      (mv_sample_1) at (\x2+2.2cm,\y1) {\includegraphics[width=3.0cm]{assets/avion_rgb/motion_0011.png}};",

        r"\path let \p1 = (rgb_sample_0), \p2 = (codec.south east) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=purple!70!black, line width=0.9pt]"
        r"      (res_sample_initial) at (\x2+4.3cm,\y1) {\includegraphics[width=3.0cm]{assets/avion_rgb/residual_0000.png}};",
        r"\path let \p1 = (rgb_sample_1), \p2 = (codec.south east) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=purple!70!black, line width=0.9pt]"
        r"      (res_sample_0) at (\x2+4.3cm,\y1) {\includegraphics[width=3.0cm]{assets/avion_rgb/residual_0004.png}};",
        r"\node[font=\Huge, text=purple!60!black]"
        r"      (res_sample_dots) at ($(res_sample_0.south) + (0,-1.30)$) {\vdots};",
        r"\path let \p1 = (rgb_sample_last), \p2 = (codec.south east) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=purple!70!black, line width=0.9pt]"
        r"      (res_sample_1) at (\x2+4.3cm,\y1) {\includegraphics[width=3.0cm]{assets/avion_rgb/residual_0011.png}};",

        r"\node[font=\small\bfseries, text=gray!60!black] at ($(mv_sample_0.south) + (0,-0.32)$) {Motion vectors $P_1$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(mv_sample_1.south) + (0,-0.32)$) {Motion vectors $P_{11}$};",
        r"\node[font=\small\bfseries, text=gray!60!black, align=center] at ($(res_sample_initial.south) + (0,-0.32)$) {Spatial encoding $I_0$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(res_sample_0.south) + (0,-0.32)$) {Residuals $P_1$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(res_sample_1.south) + (0,-0.32)$) {Residuals $P_{11}$};",

        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!70!black]"
        r"    (codec.east) -- ($(mv_sample_0.west)+(-0.35,0)$) -- (mv_sample_0.west);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!70!black]"
        r"    (codec.east) -- ($(mv_sample_1.west)+(-0.35,0)$) -- (mv_sample_1.west);",

        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=purple!75!black]"
        r"    (codec.east) -- ($(res_sample_initial.west)+(-2.45,0)$) -- (res_sample_initial.west);",

        r"\node[draw=black!70!gray, line width=1.6pt, rounded corners=10pt,"
        r"      inner xsep=14pt, inner ysep=22pt, fit={ (mv_sample_0) (mv_sample_1) (mv_sample_dots) (res_sample_initial) (res_sample_0) (res_sample_dots) (res_sample_1)}]"
        r"      (codec_block) {};",
        r"\node[anchor=north, font=\Large\bfseries, text=gray!70!black]"
        r"      at ($(codec_block.north) + (0,0.6,0)$) {Compressed Data};",


        r"\node[draw=purple!80!black, line width=1.4pt, fill=purple!6, rounded corners=7pt,"
        r"      minimum width=5.4cm, minimum height=2.6cm, align=center, font=\Large,"
        r"      name=spectra_i] at ($(res_sample_initial.east) + (3.,0,0)$) {\textbf{SpectraDet-I}\\Compressed spatial detector};",
        r"\node[draw=blue!80!black, line width=1.3pt, fill=blue!6, rounded corners=7pt,"
        r"      minimum width=5.4cm, minimum height=2.4cm, align=center, font=\Large,"
        r"      name=spectra_p1] at ($(res_sample_0.east) + (3.2,0,0)$) {\textbf{SpectraDet-P}\\Residual + motion detector};",
        r"\node[draw=blue!80!black, line width=1.3pt, fill=blue!6, rounded corners=7pt,"
        r"      minimum width=5.4cm, minimum height=2.4cm, align=center, font=\Large,"
        r"      name=spectra_p11] at ($(res_sample_1.east) + (3.2,0,0)$) {\textbf{SpectraDet-P}\\Residual + motion detector};",
        r"\node[font=\Huge, text=blue!60!black]"
        r"      (spectra_p_dots) at ($(spectra_p1.south)!0.50!(spectra_p11.north)$) {\vdots};",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=purple!75!black]"
        r"    ($(res_sample_initial.east)$) -- ($(spectra_i.west) $);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!75!black]"
        r"    ($(res_sample_0.east) $) -- ($(spectra_p1.west) $);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!75!black]"
        r"    ($(res_sample_1.east) $) -- ($(spectra_p11.west) $);",
        r"\node[draw=black!70!gray, line width=1.6pt, rounded corners=10pt,"
        r"      inner xsep=14pt, inner ysep=24pt, fit={(spectra_i) (spectra_p1) (spectra_p_dots) (spectra_p11)}] (spectra_block) {};",
        r"\node[anchor=north, font=\Large\bfseries, text=gray!70!black]"
        r"      at ($(spectra_block.north) + (0,0.60,0)$) {SpectraFlow};",

        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!65!black, dash pattern=on 2pt off 2pt on 1pt off 2pt]"
        r"    ($(spectra_i.east)$) -- ($(spectra_i.east)+(1.1,0)$) -- ($(spectra_i.east)+(1.1,-1.4)$) -- ($(spectra_p1.west)+(-0.3,1.3)$) -- ($(spectra_p1.west)+(-0.3,0)$) -- ($(spectra_p1.west)$);",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!65!black, dash pattern=on 2pt off 2pt on 1pt off 2pt]"
        r"    ($(spectra_p1.east)$) -- ($(spectra_p1.east)+(1.1,0)$) -- ($(spectra_p1.east)+(1.1,-1.8)$) -- ($(spectra_p_dots.west)+(0.0,0.25)$) --($(spectra_p_dots.north)$)"
        r"    node[pos=0.55, above=2pt, text=gray!60!black, font=\small\bfseries]{P$_{2}$ boxes};",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!65!black, dash pattern=on 2pt off 2pt on 1pt off 2pt]"
        r"    ($(spectra_p_dots.south)$) -- ($(spectra_p_dots.south)$) -- ($(spectra_p11.west)+(-0.4,1.8)$) -- ($(spectra_p11.west)+(-0.4,0)$) -- ($(spectra_p11.west)$)"
        r"    node[pos=6.25, below=-65pt, text=gray!60!black, font=\small\bfseries]{P$_{10}$ boxes};",

        # Bounding-box overlays column
        r"\coordinate (bbox_x_anchor) at ($(spectra_i)+(6.0,0)$);",
        r"\path let \p1 = (bbox_x_anchor), \p2 = (spectra_i) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=red!70!black, line width=1.0pt]"
        r"      (bbox_rgb_0) at (\x1,\y2) {\includegraphics[width=3.1cm]{assets/avion_rgb/frame_0000.png}};",
        r"\path let \p1 = (bbox_x_anchor), \p2 = (spectra_p1) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=red!70!black, line width=1.0pt]"
        r"      (bbox_rgb_1) at (\x1,\y2) {\includegraphics[width=3.1cm]{assets/avion_rgb/frame_0004.png}};",
        r"\node[font=\Huge, text=gray!60!black]"
        r"      (bbox_rgb_dots) at ($(bbox_rgb_1.south) + (0,-1.30)$) {\vdots};",
        r"\path let \p1 = (bbox_x_anchor), \p2 = (spectra_p11) in"
        r"    node[inner sep=0pt, rounded corners=5pt, draw=red!70!black, line width=1.0pt]"
        r"      (bbox_rgb_last) at (\x1,\y2) {\includegraphics[width=3.1cm]{assets/avion_rgb/frame_0011.png}};",

        r"\draw[red!80!black, line width=1.1pt, rounded corners=2pt]"
        r"    ($(bbox_rgb_0.south west)+(0.55,0.35)$) rectangle ($(bbox_rgb_0.south west)+(1.80,2.05)$);",
        r"\draw[red!80!black, line width=1.1pt, rounded corners=2pt]"
        r"    ($(bbox_rgb_1.south west)+(0.50,0.35)$) rectangle ($(bbox_rgb_1.south west)+(1.75,2.05)$);",
        r"\draw[red!80!black, line width=1.1pt, rounded corners=2pt]"
        r"    ($(bbox_rgb_last.south west)+(0.50,0.35)$) rectangle ($(bbox_rgb_last.south west)+(1.75,2.05)$);",

        r"\node[font=\small\bfseries, text=gray!60!black] at ($(bbox_rgb_0.south) + (0,-0.32)$) {Detected $I_0$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(bbox_rgb_1.south) + (0,-0.32)$) {Detected $P_1$};",
        r"\node[font=\small\bfseries, text=gray!60!black] at ($(bbox_rgb_last.south) + (0,-0.32)$) {Detected $P_{11}$};",

        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!65!black, dash pattern=on 2pt off 2pt on 1pt off 2pt]"
        r"    ($(spectra_i.east)+(0.2,0)$) -- ($(bbox_rgb_0.west)+(-0.4,0)$)"
        r"    node[pos=0.42, above=4pt, text=gray!60!black, font=\small\bfseries]{I$_0$ boxes};",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!65!black, dash pattern=on 2pt off 2pt on 1pt off 2pt]"
        r"    ($(spectra_p1.east)+(0.2,0)$) -- ($(bbox_rgb_1.west)+(-0.4,0)$)"
        r"    node[pos=0.50, above=4pt, text=gray!60!black, font=\small\bfseries]{P$_1$ boxes};",
        r"\draw[-{Stealth[length=8pt,width=7pt]}, thick, draw=blue!65!black, dash pattern=on 2pt off 2pt on 1pt off 2pt]"
        r"    ($(spectra_p11.east)+(0.2,0)$) -- ($(bbox_rgb_last.west)+(-0.4,0)$)"
        r"    node[pos=0.50, above=4pt, text=gray!60!black, font=\small\bfseries]{P$_{11}$ boxes};",

        r"\node[draw=black!70!gray, line width=1.6pt, rounded corners=10pt,"
        r"      inner xsep=12pt, inner ysep=18pt, fit={(bbox_rgb_0) (bbox_rgb_1) (bbox_rgb_dots) (bbox_rgb_last)}] (detected_block) {};",
        r"\node[anchor=north, font=\Large\bfseries, text=gray!70!black]"
        r"      at ($(detected_block.north) + (0,0.60,0)$) {Detected Boxes};",

        
        r"\end{scope}",
        to_end(),
    ]

    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, "gop_overview_architecture.tex")
    to_generate(arch, output_file)

    print("=" * 70)
    print("✅ GOP overview diagram generated!")
    print("=" * 70)
    rel_output = os.path.relpath(output_file, start=os.getcwd())
    print(f"📄 Output file: {rel_output}")
    print(f"📝 Compile with: pdflatex {rel_output}")
    print()
    print("Diagram highlights:")
    print("  • Shows GOP structure (I + 11 P) and their dedicated detectors")
    print("  • Emphasizes compressed-domain processing for both frame types")
    print("  • Captures joint fusion delivering scene-level outputs")

if __name__ == '__main__':
    main()
