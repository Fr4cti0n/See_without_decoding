#!/usr/bin/env python3
"""
Motion Block Estimation Algorithm Visualization

This script creates a visual explanation of how motion vectors are estimated
in video compression by:
1. Extracting two consecutive frames from a video
2. Zooming into a specific region
3. Drawing a 16×16 pixel grid to show macroblocks
4. (To be completed) Showing motion vector estimation process
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))


class MotionBlockVisualization:
    """
    Visualize motion block estimation algorithm for video compression.
    """
    
    def __init__(self, video_path, output_dir="output_schema"):
        """
        Initialize the visualization.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save output images
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"📹 Loading video: {self.video_path.name}")
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"   Resolution: {self.width}×{self.height}")
        print(f"   FPS: {self.fps:.2f}")
        print(f"   Total frames: {self.total_frames}")
        print()
    
    def extract_consecutive_frames(self, frame_idx=50):
        """
        Extract two consecutive frames from the video.
        
        Args:
            frame_idx: Index of the first frame (default: 50)
        
        Returns:
            frame1, frame2: Two consecutive frames (RGB format)
        """
        print(f"📸 Extracting frames {frame_idx} and {frame_idx + 1}...")
        
        # Set to first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read first frame
        ret1, frame1 = self.cap.read()
        if not ret1:
            raise ValueError(f"Cannot read frame {frame_idx}")
        
        # Read second frame
        ret2, frame2 = self.cap.read()
        if not ret2:
            raise ValueError(f"Cannot read frame {frame_idx + 1}")
        
        # Convert BGR to RGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        print(f"   ✅ Frames extracted successfully")
        print()
        
        return frame1, frame2
    
    def zoom_region(self, frame, x, y, width, height):
        """
        Extract and zoom into a specific region of the frame.
        
        Args:
            frame: Input frame
            x, y: Top-left corner of region
            width, height: Region dimensions
        
        Returns:
            Zoomed region
        """
        # Ensure coordinates are within bounds
        x = max(0, min(x, frame.shape[1] - width))
        y = max(0, min(y, frame.shape[0] - height))
        
        # Extract region
        region = frame[y:y+height, x:x+width].copy()
        
        return region
    
    def draw_grid(self, frame, block_size=16, color=(255, 0, 0), thickness=1):
        """
        Draw a grid of blocks over the frame.
        
        Args:
            frame: Input frame
            block_size: Size of each block (default: 16×16 pixels)
            color: Grid line color (RGB)
            thickness: Line thickness
        
        Returns:
            Frame with grid overlay
        """
        frame_with_grid = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw vertical lines
        for x in range(0, width, block_size):
            cv2.line(frame_with_grid, (x, 0), (x, height), color, thickness)
        
        # Draw horizontal lines
        for y in range(0, height, block_size):
            cv2.line(frame_with_grid, (0, y), (width, y), color, thickness)
        
        # Draw border
        cv2.rectangle(frame_with_grid, (0, 0), (width-1, height-1), color, thickness)
        
        return frame_with_grid
    
    def highlight_blocks(self, frame, block_size=16, target_block=None, 
                        source_block=None, search_zone=True, alpha=0.25):
        """
        Highlight specific blocks with transparent colors.
        
        Args:
            frame: Input frame (with or without grid)
            block_size: Size of each block (default: 16×16 pixels)
            target_block: Tuple (bx, by) - block coordinates for target (frame t)
            source_block: Tuple (bx, by) - block coordinates for source (frame t+1)
            search_zone: If True, highlight search zone around source block
            alpha: Transparency level (0=transparent, 1=opaque, default: 0.25)
        
        Returns:
            Frame with highlighted blocks
        """
        frame_highlighted = frame.copy()
        overlay = frame.copy()
        
        # Highlight target block (frame t) - Blue color
        if target_block is not None:
            bx, by = target_block
            x1 = bx * block_size
            y1 = by * block_size
            x2 = x1 + block_size
            y2 = y1 + block_size
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 120, 255), -1)  # Blue fill
            cv2.rectangle(frame_highlighted, (x1, y1), (x2, y2), (0, 80, 200), 1)  # Blue border (thinner)
        
        # Highlight source block (frame t+1) - Darker Green color
        if source_block is not None:
            bx, by = source_block
            x1 = bx * block_size
            y1 = by * block_size
            x2 = x1 + block_size
            y2 = y1 + block_size
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 100, 0), -1)  # Darker green fill
            cv2.rectangle(frame_highlighted, (x1, y1), (x2, y2), (0, 80, 0), 1)  # Darker green border (thinner)
            
            # Highlight search zone (all blocks touching the source block)
            if search_zone:
                # Search zone includes all 8 neighboring blocks
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip source block itself
                        
                        nbx = bx + dx
                        nby = by + dy
                        
                        # Check bounds
                        height, width = frame.shape[:2]
                        if (0 <= nbx * block_size < width and 
                            0 <= nby * block_size < height):
                            nx1 = nbx * block_size
                            ny1 = nby * block_size
                            nx2 = nx1 + block_size
                            ny2 = ny1 + block_size
                            
                            # Yellow/orange for search zone
                            cv2.rectangle(overlay, (nx1, ny1), (nx2, ny2), 
                                        (255, 200, 0), -1)  # Orange fill
                            cv2.rectangle(frame_highlighted, (nx1, ny1), (nx2, ny2), 
                                        (255, 165, 0), 1)  # Orange border (thinner)
        
        # Blend overlay with original frame for transparency
        cv2.addWeighted(overlay, alpha, frame_highlighted, 1 - alpha, 0, frame_highlighted)
        
        return frame_highlighted
    
    def draw_motion_arrow(self, frame, source_block, target_block, block_size=16, 
                         color=(255, 0, 255), thickness=2):
        """
        Draw an arrow from source block to target block.
        
        Args:
            frame: Input frame
            source_block: Tuple (bx, by) - source block coordinates
            target_block: Tuple (bx, by) - target block coordinates
            block_size: Size of each block
            color: Arrow color (BGR format)
            thickness: Arrow thickness
        
        Returns:
            Frame with arrow
        """
        frame_with_arrow = frame.copy()
        
        # Calculate center of source block
        source_cx = int((source_block[0] + 0.5) * block_size)
        source_cy = int((source_block[1] + 0.5) * block_size)
        
        # Calculate center of target block
        target_cx = int((target_block[0] + 0.5) * block_size)
        target_cy = int((target_block[1] + 0.5) * block_size)
        
        # Draw arrow from source to target
        cv2.arrowedLine(frame_with_arrow, (source_cx, source_cy), (target_cx, target_cy),
                       color, thickness, tipLength=0.3)
        
        return frame_with_arrow
    
    def create_block_difference_visualization(self, frame1_zoom, frame2_zoom, 
                                             target_block, source_block, block_size=16):
        """
        Create a mathematical subtraction visualization: Target - Source = Residual → DCT
        
        Args:
            frame1_zoom: Zoomed frame t
            frame2_zoom: Zoomed frame t+1
            target_block: Target block coordinates (bx, by)
            source_block: Source block coordinates (bx, by)
            block_size: Size of blocks
        
        Returns:
            Visualization image showing mathematical subtraction and DCT encoding
        """
        # Extract target block from frame t
        tx, ty = target_block
        target_x1, target_y1 = tx * block_size, ty * block_size
        target_x2, target_y2 = target_x1 + block_size, target_y1 + block_size
        target_patch = frame1_zoom[target_y1:target_y2, target_x1:target_x2].copy()
        
        # Extract source block from frame t+1
        sx, sy = source_block
        source_x1, source_y1 = sx * block_size, sy * block_size
        source_x2, source_y2 = source_x1 + block_size, source_y1 + block_size
        source_patch = frame2_zoom[source_y1:source_y2, source_x1:source_x2].copy()
        
        # Calculate residual (signed difference, not absolute)
        target_gray = cv2.cvtColor(target_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        source_gray = cv2.cvtColor(source_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
        residual = target_gray - source_gray
        
        # Apply 2D-DCT to the residual
        dct_coeffs = cv2.dct(residual)
        
        # Visualize DCT coefficients (log scale for better visibility)
        dct_vis = np.log(np.abs(dct_coeffs) + 1)
        dct_vis = (dct_vis / dct_vis.max() * 255).astype(np.uint8)
        # Apply purple colormap
        dct_vis_colored = cv2.applyColorMap(dct_vis, cv2.COLORMAP_JET)
        dct_vis_colored = cv2.cvtColor(dct_vis_colored, cv2.COLOR_BGR2RGB)
        # Apply purple tint to match theme
        purple_tint = np.array([128, 0, 128], dtype=np.float32)
        dct_vis_colored = (dct_vis_colored.astype(np.float32) * 0.6 + purple_tint * 0.4).clip(0, 255).astype(np.uint8)
        
        # Normalize residual for visualization (scale to 0-255)
        residual_vis = ((residual + 255) / 2).clip(0, 255).astype(np.uint8)
        residual_vis_colored = cv2.cvtColor(residual_vis, cv2.COLOR_GRAY2RGB)
        
        # Create blocks at top - increased scale for better visibility
        scale = 15  # Increased significantly for larger blocks in the middle section
        target_large = cv2.resize(target_patch, (block_size*scale, block_size*scale), 
                                 interpolation=cv2.INTER_NEAREST)
        source_large = cv2.resize(source_patch, (block_size*scale, block_size*scale), 
                                 interpolation=cv2.INTER_NEAREST)
        residual_large = cv2.resize(residual_vis_colored, (block_size*scale, block_size*scale), 
                                   interpolation=cv2.INTER_NEAREST)
        dct_large = cv2.resize(dct_vis_colored, (block_size*scale, block_size*scale), 
                              interpolation=cv2.INTER_NEAREST)
        
        # Calculate motion vector
        mv_x = target_block[0] - source_block[0]
        mv_y = target_block[1] - source_block[1]
        
        # Add labels below each block
        label_height = 35  # Increased from 25 for larger labels
        label_font_scale = 0.4
        label_thickness = 1
        
        def add_label(img, text, bg_color=(255, 255, 255)):
            """Add a label below an image."""
            h, w = img.shape[:2]
            label_img = np.ones((label_height, w, 3), dtype=np.uint8)
            label_img[:] = bg_color
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (label_height + text_size[1]) // 2
            cv2.putText(label_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       label_font_scale, (0, 0, 0), label_thickness)
            return np.vstack([img, label_img])
        
        # Add labels to each block (using ESANN paper notation)
        def add_latex_label(img, latex_text, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
            """Add a LaTeX-rendered label below an image."""
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            h, w = img.shape[:2]
            
            # Create figure with higher DPI for crisp rendering
            dpi = 200  # Increased from 150 for sharper text
            fig_width = w / dpi
            fig_height = label_height / dpi
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
            ax.text(0.5, 0.5, latex_text, ha='center', va='center', 
                   fontsize=9, fontfamily='serif', antialiased=True,
                   color=np.array(text_color)/255.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            # Render to array
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            label_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel
            plt.close(fig)
            
            # Ensure exact dimensions
            if label_array.shape[0] != label_height or label_array.shape[1] != w:
                label_array = cv2.resize(label_array, (w, label_height), interpolation=cv2.INTER_LINEAR)
            
            return np.vstack([img, label_array])
        
        target_labeled = add_latex_label(target_large, r"$\mathbf{Target\ Block}$", text_color=(0, 0, 255))  # Blue
        source_labeled = add_latex_label(source_large, r"$\mathbf{Source\ Block}$", text_color=(0, 100, 0))  # Darker green
        residual_labeled = add_latex_label(residual_large, r"$\mathbf{\Delta Y_n^g}$")
        
        # Create motion vector visualization with arrow (light orange background)
        mv_img = np.ones((block_size*scale, block_size*scale, 3), dtype=np.uint8)
        mv_img[:] = [255, 235, 215]  # Light orange/peach background to match theme
        
        # Draw arrow representing motion vector
        center_x = (block_size*scale) // 2
        center_y = (block_size*scale) // 2
        arrow_scale = 35  # INCREASED from 20 to 35 for much larger arrow
        end_x = center_x + int(mv_x * arrow_scale)
        end_y = center_y + int(mv_y * arrow_scale)
        
        # Draw the arrow (orange color to match theme) with antialiasing - thicker for better visibility
        cv2.arrowedLine(mv_img, (center_x, center_y), (end_x, end_y),
                       (255, 140, 0), 6, tipLength=0.4, line_type=cv2.LINE_AA)
        
        # Add coordinate text with LARGER font size
        coord_text = f"({mv_x},{mv_y})"
        text_size = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = (block_size*scale - text_size[0]) // 2
        text_y = block_size*scale - 15
        cv2.putText(mv_img, coord_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.9, (80, 80, 80), 2, cv2.LINE_AA)
        
        mv_labeled = add_latex_label(mv_img, r"$\mathbf{MV_n^g}$")
        
        # Create visualization with extracted blocks at top, equation in middle, outputs at bottom
        
        # Step 1: Show extracted blocks side by side at top
        h_block = target_labeled.shape[0]
        w_block = target_labeled.shape[1]
        
        # Add spacing between blocks
        spacing_between = 30
        spacing_img = np.ones((h_block, spacing_between, 3), dtype=np.uint8) * 255
        
        # Top row: extracted blocks
        extracted_blocks_row = np.hstack([target_labeled, spacing_img, source_labeled])
        
        # Step 2: Add arrows pointing down from blocks toward the center (minus symbol)
        arrow_height = 60
        arrow_section = np.ones((arrow_height, extracted_blocks_row.shape[1], 3), dtype=np.uint8) * 255
        
        # Calculate center position where title and minus will be
        center_of_row = extracted_blocks_row.shape[1] // 2
        
        # Draw diagonal arrows converging toward center (pointing to title area)
        arrow_y_start = 5
        arrow_y_end = arrow_height - 5  # Point closer to title
        
        # Add horizontal spacing between arrow endpoints (8 pixels apart)
        arrow_spacing = 8
        
        # Arrow from target block (points slightly left of center toward title)
        target_center_x = w_block // 2
        target_arrow_end_x = center_of_row - arrow_spacing // 2
        cv2.arrowedLine(arrow_section, (target_center_x, arrow_y_start), (target_arrow_end_x, arrow_y_end),
                       (100, 100, 100), 2, tipLength=0.2, line_type=cv2.LINE_AA)
        
        # Arrow from source block (points slightly right of center toward title)
        source_center_x = w_block + spacing_between + w_block // 2
        source_arrow_end_x = center_of_row + arrow_spacing // 2
        cv2.arrowedLine(arrow_section, (source_center_x, arrow_y_start), (source_arrow_end_x, arrow_y_end),
                       (100, 100, 100), 2, tipLength=0.2, line_type=cv2.LINE_AA)
        
        # Step 2.5: Add title and subtitles
        def add_center_title(width, title_text):
            """Add a centered LaTeX title (single line, bold)."""
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            title_height = 50  # Height for single title line
            dpi = 200
            min_width = 800
            actual_width = max(width, min_width)
            fig_width = actual_width / dpi
            fig_height = title_height / dpi
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.text(0.5, 0.5, title_text, ha='center', va='center', 
                   fontsize=12, fontfamily='serif', antialiased=True, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            title_array = np.asarray(buf)[:, :, :3]
            plt.close(fig)
            
            # Ensure exact dimensions
            if title_array.shape[0] != title_height or title_array.shape[1] != actual_width:
                title_array = cv2.resize(title_array, (actual_width, title_height), interpolation=cv2.INTER_LINEAR)
            
            return title_array
        
        # Title/subtitles are rendered at the figure level in save_visualization now,
        # so we don't create image strips for them inside this function.
        
        # Step 3: Create equation section - just the minus symbol (no blocks)
        # Create a section with just the circled minus symbol - LARGER SIZE
        symbol_height = 90  # Increased from 60 for larger symbol
        symbol_width = 90   # Width to make it square
        minus_img = np.ones((symbol_height, symbol_width, 3), dtype=np.uint8) * 255
        minus_y = symbol_height // 2
        center_x = symbol_width // 2
        # Draw circle around minus (radius 30 - larger)
        cv2.circle(minus_img, (center_x, minus_y), 30, (0, 0, 0), 3)
        # Draw horizontal minus line (longer but not touching circle border)
        cv2.line(minus_img, (center_x-15, minus_y), (center_x+15, minus_y), (0, 0, 0), 4)
        
        # Center the minus symbol in the row
        equation_row = minus_img
        equation_width = equation_row.shape[1]
        
        # Match width with extracted_blocks_row by centering
        extracted_width = extracted_blocks_row.shape[1]
        if equation_width < extracted_width:
            pad_eq = (extracted_width - equation_width) // 2
            pad_left = np.ones((symbol_height, pad_eq, 3), dtype=np.uint8) * 255
            pad_right = np.ones((symbol_height, extracted_width - equation_width - pad_eq, 3), dtype=np.uint8) * 255
            equation_row = np.hstack([pad_left, equation_row, pad_right])
        
        # Step 4: Show outputs below equation (LARGER SIZE)
        # Create output blocks with borders
        border_thickness = 2
        
        # Use larger scale for output blocks to make them more visible
        scale_output = 14  # Increased from 6 to 8 for larger outputs
        
        # Use residual instead of DCT
        residual_output = cv2.resize(residual_vis_colored, (block_size*scale_output, block_size*scale_output), 
                               interpolation=cv2.INTER_NEAREST)
        residual_labeled_out = add_latex_label(residual_output, r"$\mathbf{\Delta Y_n^g}$")
        # Add horizontal padding to make image wider for title display
        horizontal_padding = 30  # Add 30 pixels on each side
        residual_labeled_padded = cv2.copyMakeBorder(residual_labeled_out, 0, 0, 
                                                horizontal_padding, horizontal_padding,
                                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # Gray border for residual
        residual_border_color = (128, 128, 128)  # Gray
        residual_bordered = cv2.copyMakeBorder(residual_labeled_padded, border_thickness, border_thickness, 
                                         border_thickness, border_thickness,
                                         cv2.BORDER_CONSTANT, value=residual_border_color)
        
        mv_output = cv2.resize(mv_img, (block_size*scale_output, block_size*scale_output), 
                              interpolation=cv2.INTER_NEAREST)
        mv_labeled_out = add_latex_label(mv_output, r"$\mathbf{MV_n^g}$")
        # Add horizontal padding to make image wider for title display
        mv_labeled_padded = cv2.copyMakeBorder(mv_labeled_out, 0, 0, 
                                               horizontal_padding, horizontal_padding,
                                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # Orange border for MV
        mv_border_color = (255, 140, 0)  # Orange
        mv_bordered = cv2.copyMakeBorder(mv_labeled_padded, border_thickness, border_thickness, 
                                        border_thickness, border_thickness,
                                        cv2.BORDER_CONSTANT, value=mv_border_color)
        
        # Create two separate output sections with labels using LaTeX
        def add_section_label(bordered_img, latex_text, text_color=(0, 0, 0), fontsize=6):
            """Add a LaTeX-rendered section label above an image."""
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            section_height = 75  # Increased from 60 to 75 to avoid overlap with gray arrows
            w = bordered_img.shape[1]            # Create figure with higher DPI for crisp rendering
            dpi = 200  # Increased from 150 for sharper text
            fig_width = w / dpi
            fig_height = section_height / dpi
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
            ax.text(0.5, 0.5, latex_text, ha='center', va='center', 
                   fontsize=fontsize, fontfamily='serif', antialiased=True,
                   color=np.array(text_color)/255.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            label_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel
            plt.close(fig)
            
            # Ensure exact dimensions
            if label_array.shape[0] != section_height or label_array.shape[1] != w:
                label_array = cv2.resize(label_array, (w, section_height), interpolation=cv2.INTER_LINEAR)
            
            return np.vstack([label_array, bordered_img])
        
        # Residual output section with gray color - Changed to "Residuals" (plural)
        residual_section = add_section_label(residual_bordered, r"$\mathbf{Residuals}$", (128, 128, 128), 9)
        
        # MV output section with orange color - REMOVE "Output 2:"
        mv_section = add_section_label(mv_bordered, r"$\mathbf{Motion\ Vectors}$", (255, 140, 0), 9)
        
        # Add brackets around each output to distinguish them
        def add_bracket(img, bracket_color=(0, 0, 0)):
            """Add left and right brackets around an image."""
            h, w = img.shape[:2]
            bracket_width = 50  # Space on each side
            bracket_thickness = 2
            bracket_offset = 25  # Distance between bracket and image (increased from 5 to 25)
            
            # Create canvas with brackets
            canvas_width = w + 2 * bracket_width
            canvas = np.ones((h, canvas_width, 3), dtype=np.uint8) * 255
            
            # Place image in center
            canvas[:, bracket_width:bracket_width+w] = img
            
            # Draw left bracket [ (further from image)
            cv2.line(canvas, (bracket_width-bracket_offset, 5), (bracket_width-bracket_offset, h-5), bracket_color, bracket_thickness)
            cv2.line(canvas, (bracket_width-bracket_offset, 5), (bracket_width-bracket_offset+3, 5), bracket_color, bracket_thickness)
            cv2.line(canvas, (bracket_width-bracket_offset, h-5), (bracket_width-bracket_offset+3, h-5), bracket_color, bracket_thickness)
            
            # Draw right bracket ] (further from image)
            cv2.line(canvas, (bracket_width+w+bracket_offset, 5), (bracket_width+w+bracket_offset, h-5), bracket_color, bracket_thickness)
            cv2.line(canvas, (bracket_width+w+bracket_offset, 5), (bracket_width+w+bracket_offset-3, 5), bracket_color, bracket_thickness)
            cv2.line(canvas, (bracket_width+w+bracket_offset, h-5), (bracket_width+w+bracket_offset-3, h-5), bracket_color, bracket_thickness)
            
            return canvas
        
        # Add brackets to both outputs with matching colors
        residual_section_bracketed = add_bracket(residual_section, (128, 128, 128))  # Gray for Residual
        mv_section_bracketed = add_bracket(mv_section, (255, 140, 0))  # Orange for MV
        
        # Stack outputs horizontally with spacing
        output_spacing = 40
        output_space_img = np.ones((residual_section_bracketed.shape[0], output_spacing, 3), dtype=np.uint8) * 255
        outputs_row = np.hstack([residual_section_bracketed, output_space_img, mv_section_bracketed])
        
        # Center outputs
        output_width = outputs_row.shape[1]
        if output_width < extracted_width:
            pad_out = (extracted_width - output_width) // 2
            pad_left_out = np.ones((outputs_row.shape[0], pad_out, 3), dtype=np.uint8) * 255
            pad_right_out = np.ones((outputs_row.shape[0], extracted_width - output_width - pad_out, 3), dtype=np.uint8) * 255
            outputs_row = np.hstack([pad_left_out, outputs_row, pad_right_out])
        
        # Make all rows same width before stacking - CENTER each element
        max_width = max(extracted_width, equation_row.shape[1], outputs_row.shape[1])

        # Center equation row if needed
        if equation_row.shape[1] < max_width:
            pad_width = max_width - equation_row.shape[1]
            pad_left = np.ones((equation_row.shape[0], pad_width // 2, 3), dtype=np.uint8) * 255
            pad_right = np.ones((equation_row.shape[0], pad_width - pad_width // 2, 3), dtype=np.uint8) * 255
            equation_row = np.hstack([pad_left, equation_row, pad_right])

        # Center outputs row if needed
        if outputs_row.shape[1] < max_width:
            pad_width = max_width - outputs_row.shape[1]
            pad_left = np.ones((outputs_row.shape[0], pad_width // 2, 3), dtype=np.uint8) * 255
            pad_right = np.ones((outputs_row.shape[0], pad_width - pad_width // 2, 3), dtype=np.uint8) * 255
            outputs_row = np.hstack([pad_left, outputs_row, pad_right])

        # Center extracted blocks if needed
        if extracted_width < max_width:
            pad_width = max_width - extracted_width
            pad_left = np.ones((extracted_blocks_row.shape[0], pad_width // 2, 3), dtype=np.uint8) * 255
            pad_right = np.ones((extracted_blocks_row.shape[0], pad_width - pad_width // 2, 3), dtype=np.uint8) * 255
            extracted_blocks_row = np.hstack([pad_left, extracted_blocks_row, pad_right])

        # Center arrow section to match width
        if arrow_section.shape[1] < max_width:
            pad_width = max_width - arrow_section.shape[1]
            pad_left = np.ones((arrow_section.shape[0], pad_width // 2, 3), dtype=np.uint8) * 255
            pad_right = np.ones((arrow_section.shape[0], pad_width - pad_width // 2, 3), dtype=np.uint8) * 255
            arrow_section = np.hstack([pad_left, arrow_section, pad_right])
        
        # Add spacing between sections
        v_spacing = 15
        v_space1 = np.ones((v_spacing, max_width, 3), dtype=np.uint8) * 255
        
        # Create arrow section from minus to outputs - INCREASED HEIGHT to prevent label cut-off
        arrow_to_outputs_height = 130
        arrow_to_outputs_section = np.ones((arrow_to_outputs_height, max_width, 3), dtype=np.uint8) * 255
        
        # Calculate positions for arrows to point to each output
        # Find center of minus symbol (which is centered in equation_row)
        minus_center_x = max_width // 2
        
        # Calculate exact positions of the two outputs after centering
        # outputs_row structure: [padding] + dct_section_bracketed + output_spacing + mv_section_bracketed + [padding]
        output_section_width = outputs_row.shape[1]
        output_start_x = (max_width - output_section_width) // 2
        
        # Residual section is first, calculate its center
        residual_width = residual_section_bracketed.shape[1]
        residual_center_x = output_start_x + residual_width // 2
        
        # MV section is after Residual + spacing, calculate its center
        mv_start = output_start_x + residual_width + 40  # 40 is output_spacing
        mv_width = mv_section_bracketed.shape[1]
        mv_center_x = mv_start + mv_width // 2
        
        # Draw arrows from minus center to each output (diagonal arrows) - symmetrical with matching colors
        arrow_y_start = 10
        arrow_y_end = arrow_to_outputs_height - 10
        
        # Arrow to Residual output (left) - Gray color to match Residual theme
        cv2.arrowedLine(arrow_to_outputs_section, (minus_center_x, arrow_y_start), 
                       (residual_center_x, arrow_y_end), (128, 128, 128), 3, tipLength=0.15, line_type=cv2.LINE_AA)
        
        # Arrow to MV output (right) - Orange color to match MV theme
        cv2.arrowedLine(arrow_to_outputs_section, (minus_center_x, arrow_y_start), 
                       (mv_center_x, arrow_y_end), (255, 140, 0), 3, tipLength=0.15, line_type=cv2.LINE_AA)
        
        # Add LaTeX labels on arrows
        def add_arrow_label(img, latex_text, center_x, center_y, text_color=(80, 80, 80)):
            """Add a LaTeX-rendered label at specified position on the image."""
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            # Create a small figure for the text - INCREASED SIZE
            label_width = 280
            label_height = 50
            dpi = 150
            fig_width = label_width / dpi
            fig_height = label_height / dpi
            
            fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.text(0.5, 0.5, latex_text, ha='center', va='center', 
                   fontsize=13, fontfamily='serif', antialiased=True,
                   color=np.array(text_color)/255.0)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            fig.patch.set_facecolor('white')
            fig.patch.set_alpha(0.0)
            
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            label_array = np.asarray(buf)
            plt.close(fig)
            
            # Resize to exact dimensions
            label_array = cv2.resize(label_array, (label_width, label_height), interpolation=cv2.INTER_LINEAR)
            
            # Extract alpha channel and RGB
            alpha = label_array[:, :, 3] / 255.0
            label_rgb = label_array[:, :, :3]
            
            # Calculate position to overlay
            y1 = max(0, center_y - label_height // 2)
            y2 = min(img.shape[0], y1 + label_height)
            x1 = max(0, center_x - label_width // 2)
            x2 = min(img.shape[1], x1 + label_width)
            
            # Crop label if needed
            label_y1 = 0 if y1 >= 0 else -y1
            label_y2 = label_height if y2 <= img.shape[0] else label_height - (y2 - img.shape[0])
            label_x1 = 0 if x1 >= 0 else -x1
            label_x2 = label_width if x2 <= img.shape[1] else label_width - (x2 - img.shape[1])
            
            # Blend the label onto the image
            for c in range(3):
                img[y1:y2, x1:x2, c] = (
                    alpha[label_y1:label_y2, label_x1:label_x2] * label_rgb[label_y1:label_y2, label_x1:label_x2, c] +
                    (1 - alpha[label_y1:label_y2, label_x1:label_x2]) * img[y1:y2, x1:x2, c]
                )
        
        # Add label "Pixels Subtraction" near the left arrow (to Residual) - Gray color
        # Shifted MORE to the left horizontally to avoid overlap
        label1_x = (minus_center_x + residual_center_x) // 2 - 120
        label1_y = 35  # Keep at previous y value
        add_arrow_label(arrow_to_outputs_section, r"$\mathbf{Pixels\ Subtraction}$", label1_x, label1_y, text_color=(128, 128, 128))
        
        # Add label "Motion Estimation" near the right arrow (to MV) - Orange color
        # Shifted MORE to the right horizontally to avoid overlap
        label2_x = (minus_center_x + mv_center_x) // 2 + 120
        label2_y = 35  # Keep at previous y value
        add_arrow_label(arrow_to_outputs_section, r"$\mathbf{Motion\ Estimation}$", label2_x, label2_y, text_color=(255, 140, 0))
        
        # Stack all sections vertically:
        # 1. Extracted blocks
        # 2. Arrows pointing down to title
        # 3. Small spacing
        # 4. Title "Block Motion Estimation Algorithm"
        # 5. Subtitle "Candidates Block Selection"
        # 6. Small spacing
        # 7. Subtitle "Motion Estimation"
        # 8. Small spacing
        # 9. Minus symbol
        # 10. Arrows pointing to outputs
        # 11. Outputs
        v_space_title = np.ones((10, max_width, 3), dtype=np.uint8) * 255
        # compact spacing used between title/blocks and the equation
        equation_viz = np.vstack([
            extracted_blocks_row,
            arrow_section,
            v_space_title,
            equation_row,
            arrow_to_outputs_section,
            outputs_row
        ])
        
        # Return the equation visualization without the table
        return equation_viz, target_patch, source_patch, residual
    
    def visualize_two_frames_with_grid(
        self, 
        frame_idx=50, 
        zoom_x=400, 
        zoom_y=300, 
        zoom_width=160, 
        zoom_height=160,
        block_size=16,
        target_block=None,
        source_block=None,
        show_search_zone=True,
        alpha=0.25
    ):
        """
        Create visualization showing two consecutive frames with grid overlay.
        
        Args:
            frame_idx: Index of first frame
            zoom_x, zoom_y: Top-left corner of zoom region
            zoom_width, zoom_height: Dimensions of zoom region
            block_size: Block size for grid (default: 16)
            target_block: Tuple (bx, by) for target block in frame t (optional)
            source_block: Tuple (bx, by) for source block in frame t+1 (optional)
            show_search_zone: If True, show search zone around source block
            alpha: Transparency level for block highlighting (0-1, default: 0.25)
        
        Returns:
            Dictionary with visualizations
        """
        # Extract frames
        frame1, frame2 = self.extract_consecutive_frames(frame_idx)
        
        # Zoom into region
        print(f"🔍 Zooming into region: ({zoom_x}, {zoom_y}) → ({zoom_x+zoom_width}, {zoom_y+zoom_height})")
        zoom1 = self.zoom_region(frame1, zoom_x, zoom_y, zoom_width, zoom_height)
        zoom2 = self.zoom_region(frame2, zoom_x, zoom_y, zoom_width, zoom_height)
        
        # Draw grid
        print(f"📐 Drawing {block_size}×{block_size} pixel grid...")
        zoom1_grid = self.draw_grid(zoom1, block_size=block_size)
        zoom2_grid = self.draw_grid(zoom2, block_size=block_size)
        
        # Add block highlighting if requested
        zoom1_grid_highlighted = None
        zoom2_grid_highlighted = None
        
        if target_block is not None or source_block is not None:
            print(f"🎨 Adding block highlighting...")
            
            # Highlight target block in frame t (blue)
            if target_block is not None:
                print(f"   🎯 Target block at ({target_block[0]}, {target_block[1]}) - Blue")
                zoom1_grid_highlighted = self.highlight_blocks(
                    zoom1_grid,
                    block_size=block_size,
                    target_block=target_block,
                    source_block=None,
                    search_zone=False,
                    alpha=alpha
                )
            else:
                zoom1_grid_highlighted = zoom1_grid.copy()
            
            # Highlight only source block in frame t+1 (green, no orange search zone)
            if source_block is not None:
                print(f"   📍 Source block at ({source_block[0]}, {source_block[1]}) - Green")
                
                # Highlight only source block (green), no search zone
                zoom2_grid_highlighted = self.highlight_blocks(
                    zoom2_grid,
                    block_size=block_size,
                    target_block=None,  # Don't show target block on right frame
                    source_block=source_block,
                    search_zone=False,  # Disable search zone
                    alpha=alpha
                )
                
            else:
                zoom2_grid_highlighted = zoom2_grid.copy()
        
        # Create block difference visualization if both blocks specified
        block_diff_viz = None
        if target_block is not None and source_block is not None:
            print(f"   📊 Creating block difference visualization...")
            block_diff_viz, _, _, _ = self.create_block_difference_visualization(
                zoom1, zoom2, target_block, source_block, block_size
            )
        
        # Also draw zoom region rectangle on full frames
        frame1_marked = frame1.copy()
        frame2_marked = frame2.copy()
        cv2.rectangle(frame1_marked, (zoom_x, zoom_y), 
                     (zoom_x + zoom_width, zoom_y + zoom_height), 
                     (0, 255, 0), 3)
        cv2.rectangle(frame2_marked, (zoom_x, zoom_y), 
                     (zoom_x + zoom_width, zoom_y + zoom_height), 
                     (0, 255, 0), 3)
        
        print(f"   ✅ Grid overlay complete")
        print()
        
        results = {
            'frame1_full': frame1_marked,
            'frame2_full': frame2_marked,
            'frame1_zoom': zoom1,
            'frame2_zoom': zoom2,
            'frame1_zoom_grid': zoom1_grid,
            'frame2_zoom_grid': zoom2_grid,
            'zoom_coords': (zoom_x, zoom_y, zoom_width, zoom_height),
            'block_size': block_size
        }
        
        # Add highlighted versions if they exist
        if zoom1_grid_highlighted is not None:
            results['frame1_zoom_grid_highlighted'] = zoom1_grid_highlighted
        if zoom2_grid_highlighted is not None:
            results['frame2_zoom_grid_highlighted'] = zoom2_grid_highlighted
        if block_diff_viz is not None:
            results['block_difference'] = block_diff_viz
        
        # Add block information for arrow drawing
        results['target_block'] = target_block
        results['source_block'] = source_block
        
        return results
    
    def save_visualization(self, results, prefix="motion_blocks"):
        """
        Save all visualization images.
        
        Args:
            results: Dictionary from visualize_two_frames_with_grid()
            prefix: Filename prefix
        """
        print(f"💾 Saving visualizations to: {self.output_dir}/")
        
        # Save full frames with zoom region marked
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(results['frame1_full'])
        plt.title(f"Frame t (Full)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(results['frame2_full'])
        plt.title(f"Frame t+1 (Full)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        full_path = self.output_dir / f"{prefix}_full_frames.png"
        plt.savefig(full_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {full_path.name}")
        
        # Save zoomed regions without grid
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(results['frame1_zoom'])
        plt.title(f"Frame t (Zoom)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(results['frame2_zoom'])
        plt.title(f"Frame t+1 (Zoom)", fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        zoom_path = self.output_dir / f"{prefix}_zoomed.png"
        plt.savefig(zoom_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {zoom_path.name}")
        
        # Save zoomed regions WITH grid
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(results['frame1_zoom_grid'])
        plt.title(f"Frame t (Zoom + {results['block_size']}×{results['block_size']} Grid)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(results['frame2_zoom_grid'])
        plt.title(f"Frame t+1 (Zoom + {results['block_size']}×{results['block_size']} Grid)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        grid_path = self.output_dir / f"{prefix}_with_grid.png"
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {grid_path.name}")
        
        # Save zoomed regions WITH grid AND block highlighting (if available)
        if 'frame1_zoom_grid_highlighted' in results or 'frame2_zoom_grid_highlighted' in results:
            # Check if we have block difference visualization for 3-panel layout
            if 'block_difference' in results:
                # Create 3-panel layout with reduced spacing: Frame t | Block Difference | Frame t+1
                fig = plt.figure(figsize=(20, 7))

                # Add figure-level title (no subtitle)
                fig.suptitle(r"$\mathbf{Block\ Motion\ Estimation\ Algorithm}$", fontsize=20, fontweight='bold', y=0.96)

                # Use gridspec for tighter control over spacing
                import matplotlib.gridspec as gridspec
                gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.02)  # Very tight horizontal spacing
                
                # Left panel: Frame f_{n-1}^g with target block (SWAPPED - Target is previous frame)
                ax1 = fig.add_subplot(gs[0, 0])
                frame1_to_show = results.get('frame1_zoom_grid_highlighted', results['frame1_zoom_grid'])
                ax1.imshow(frame1_to_show)
                ax1.set_title(r"$\mathbf{Frame\ f_{n-1}^g}$" + "\n" + r"$\mathbf{Target\ Block}$", 
                         fontsize=16, fontweight='bold')
                ax1.axis('off')
                
                # Middle panel: Block motion estimation algorithm
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(results['block_difference'])
                # Middle panel does not repeat the main title (it's shown at the figure level)
                ax2.axis('off')
                
                # Right panel: Frame f_n^g with source block (SWAPPED - Source is current frame)
                ax3 = fig.add_subplot(gs[0, 2])
                frame2_to_show = results.get('frame2_zoom_grid_highlighted', results['frame2_zoom_grid'])
                ax3.imshow(frame2_to_show)
                ax3.set_title(r"$\mathbf{Frame\ f_n^g}$" + "\n" + r"$\mathbf{Source\ Block}$", 
                         fontsize=16, fontweight='bold')
                ax3.axis('off')
                
                # Add dashed arrows from actual block positions to middle panel
                from matplotlib.patches import FancyArrowPatch
                
                # Get positions of the panels
                bbox1 = ax1.get_position()
                bbox2 = ax2.get_position()
                bbox3 = ax3.get_position()
                
                # Calculate block positions in normalized coordinates
                # Get image dimensions and block positions
                img_height, img_width = frame1_to_show.shape[:2]
                block_size = results['block_size']
                target_block = results.get('target_block')
                source_block = results.get('source_block')
                
                # Target block position in left panel (in pixels)
                if target_block is not None:
                    target_center_x = (target_block[0] + 0.5) * block_size
                    target_center_y = (target_block[1] + 0.5) * block_size
                    
                    # Convert to normalized figure coordinates
                    target_x_norm = bbox1.x0 + (target_center_x / img_width) * bbox1.width
                    target_y_norm = bbox1.y0 + (1 - target_center_y / img_height) * bbox1.height
                    
                    # Get the middle panel image to calculate exact block position
                    middle_img = results['block_difference']
                    middle_height, middle_width = middle_img.shape[:2]
                    
                    # Calculate exact position of target block in equation section
                    # The equation blocks are horizontally aligned with extracted blocks
                    # Target block is on the LEFT side
                    # Point to the LEFT border at the MIDDLE vertical position
                    
                    # Horizontal: LEFT BORDER of the target block (accounting for centering)
                    # With centered content, target block is further from left edge
                    target_block_left_x = bbox2.x0  + bbox2.width * 0.32  # Left border with centering offset
                    
                    # Vertical: Middle of the EXTRACTED BLOCKS AT THE TOP (the large labeled blocks)
                    # Structure: extracted_blocks_row (TOP) -> arrow_section -> v_space1 -> equation_row
                    # Point to the extracted blocks at the top (close to y0, so small multiplier)
                    # y0 is top, y1 is bottom, so y1 - small*height = near top
                    target_block_middle_y = bbox2.y1 - bbox2.height * 0.12  # Middle of extracted blocks AT TOP
                    
                    # Arrow from target block to left middle border of equation target block
                    arrow1 = FancyArrowPatch(
                        (target_x_norm, target_y_norm),  # From target block center
                        (target_block_left_x, target_block_middle_y),  # To left middle border
                        transform=fig.transFigure,
                        arrowstyle='->',
                        linestyle='--',
                        linewidth=2,
                        color='blue',
                        alpha=0.7,
                        mutation_scale=20
                    )
                    fig.patches.append(arrow1)
                
                # Source block position in right panel (in pixels)
                if source_block is not None:
                    source_center_x = (source_block[0] + 0.5) * block_size
                    source_center_y = (source_block[1] + 0.5) * block_size
                    
                    # Convert to normalized figure coordinates
                    source_x_norm = bbox3.x0 + (source_center_x / img_width) * bbox3.width
                    source_y_norm = bbox3.y0 + (1 - source_center_y / img_height) * bbox3.height
                    
                    # Get the middle panel image to calculate exact block position
                    middle_img = results['block_difference']
                    middle_height, middle_width = middle_img.shape[:2]
                    
                    # Calculate exact position of source block in equation section
                    # The equation blocks are horizontally aligned with extracted blocks
                    # Source block is on the RIGHT side
                    # Point to the RIGHT border at the MIDDLE vertical position
                    
                    # Horizontal: RIGHT BORDER of the source block (accounting for centering)
                    # With centered content, source block is before the right edge
                    source_block_right_x = bbox2.x0 + bbox2.width * 0.68  # Right border with centering offset
                    
                    # Vertical: Middle of the EXTRACTED BLOCKS AT THE TOP (same vertical position as target)
                    source_block_middle_y = bbox2.y1 - bbox2.height * 0.12  # Middle of extracted blocks AT TOP
                    
                    # Arrow from source block to right middle border of equation source block
                    arrow2 = FancyArrowPatch(
                        (source_x_norm, source_y_norm),  # From source block center
                        (source_block_right_x, source_block_middle_y),  # To right middle border
                        transform=fig.transFigure,
                        arrowstyle='->',
                        linestyle='--',
                        linewidth=2,
                        color='green',
                        alpha=0.7,
                        mutation_scale=20
                    )
                    fig.patches.append(arrow2)
                highlighted_path = self.output_dir / f"{prefix}_with_grid_highlighted.png"
                plt.savefig(highlighted_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved: {highlighted_path.name} (3-panel with block difference)")
            else:
                # Original 2-panel layout
                plt.figure(figsize=(15, 7))
                plt.subplot(1, 2, 1)
                frame1_to_show = results.get('frame1_zoom_grid_highlighted', results['frame1_zoom_grid'])
                plt.imshow(frame1_to_show)
                plt.title(f"Frame t - Target Block (Blue)", 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                frame2_to_show = results.get('frame2_zoom_grid_highlighted', results['frame2_zoom_grid'])
                plt.imshow(frame2_to_show)
                plt.title(f"Frame t+1 - Source Block (Green) + Search Zone (Orange)", 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                
                plt.tight_layout()
                highlighted_path = self.output_dir / f"{prefix}_with_grid_highlighted.png"
                plt.savefig(highlighted_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved: {highlighted_path.name}")
        
        # Save individual frames (for further processing)
        cv2.imwrite(str(self.output_dir / f"{prefix}_frame_t.png"), 
                   cv2.cvtColor(results['frame1_zoom'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.output_dir / f"{prefix}_frame_t_plus_1.png"), 
                   cv2.cvtColor(results['frame2_zoom'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.output_dir / f"{prefix}_frame_t_grid.png"), 
                   cv2.cvtColor(results['frame1_zoom_grid'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(self.output_dir / f"{prefix}_frame_t_plus_1_grid.png"), 
                   cv2.cvtColor(results['frame2_zoom_grid'], cv2.COLOR_RGB2BGR))
        
        print(f"   ✅ Saved individual frames")
        print()
    
    def close(self):
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()


def main():
    """
    Main function to create motion block visualization.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create motion block estimation visualization from video'
    )
    parser.add_argument('--video', type=str, 
                       default='MOT17-04-SDP_960x960_gop50_500frames.mp4',
                       help='Video filename (default: MOT17-04-SDP_960x960_gop50_500frames.mp4)')
    parser.add_argument('--video-dir', type=str,
                       default='/home/aduche/Bureau/motion_sight_back_up/R-Yolov1/yolo-v3/Machine-Learning-Collection/ML/Pytorch/object_detection/YoloV11/yolov_11_compressed/MOTS-experiments/data/MOTS',
                       help='Directory containing the video')
    parser.add_argument('--frame-idx', type=int, default=50,
                       help='Frame index to extract (default: 50)')
    parser.add_argument('--zoom-x', type=int, default=400,
                       help='X coordinate of zoom region (default: 400)')
    parser.add_argument('--zoom-y', type=int, default=300,
                       help='Y coordinate of zoom region (default: 300)')
    parser.add_argument('--zoom-width', type=int, default=160,
                       help='Width of zoom region (default: 160)')
    parser.add_argument('--zoom-height', type=int, default=160,
                       help='Height of zoom region (default: 160)')
    parser.add_argument('--block-size', type=int, default=16,
                       help='Block size for grid (default: 16)')
    parser.add_argument('--output-dir', type=str, default='output_schema',
                       help='Output directory for visualizations')
    parser.add_argument('--target-x', type=int, default=None,
                       help='X coordinate of target block (in grid units)')
    parser.add_argument('--target-y', type=int, default=None,
                       help='Y coordinate of target block (in grid units)')
    parser.add_argument('--source-x', type=int, default=None,
                       help='X coordinate of source block (in grid units)')
    parser.add_argument('--source-y', type=int, default=None,
                       help='Y coordinate of source block (in grid units)')
    parser.add_argument('--no-search-zone', action='store_true',
                       help='Disable search zone visualization')
    parser.add_argument('--alpha', type=float, default=0.25,
                       help='Transparency level for block highlighting (0-1, default: 0.25)')
    
    args = parser.parse_args()
    
    # Construct video path
    video_path = Path(args.video_dir) / args.video
    
    # Parse block coordinates
    target_block = None
    if args.target_x is not None and args.target_y is not None:
        target_block = (args.target_x, args.target_y)
    
    source_block = None
    if args.source_x is not None and args.source_y is not None:
        source_block = (args.source_x, args.source_y)
    
    print("=" * 80)
    print("🎬 MOTION BLOCK ESTIMATION VISUALIZATION")
    print("=" * 80)
    print()
    
    if target_block or source_block:
        print("🎨 Block highlighting enabled:")
        if target_block:
            print(f"   🎯 Target block: ({target_block[0]}, {target_block[1]}) - Blue")
        if source_block:
            print(f"   📍 Source block: ({source_block[0]}, {source_block[1]}) - Green")
            if not args.no_search_zone:
                print(f"   🔍 Search zone (8 neighbors) - Orange")
        print()
    
    try:
        # Create visualizer
        viz = MotionBlockVisualization(video_path, output_dir=args.output_dir)
        
        # Generate visualization
        results = viz.visualize_two_frames_with_grid(
            frame_idx=args.frame_idx,
            zoom_x=args.zoom_x,
            zoom_y=args.zoom_y,
            zoom_width=args.zoom_width,
            zoom_height=args.zoom_height,
            block_size=args.block_size,
            target_block=target_block,
            source_block=source_block,
            show_search_zone=not args.no_search_zone,
            alpha=args.alpha
        )
        
        # Save outputs
        viz.save_visualization(results)
        
        # Close video
        viz.close()
        
        print("=" * 80)
        print("✅ VISUALIZATION COMPLETE!")
        print("=" * 80)
        print()
        print(f"📁 Output directory: {Path(args.output_dir).absolute()}")
        print(f"   - motion_blocks_full_frames.png (full view with zoom region)")
        print(f"   - motion_blocks_zoomed.png (zoomed regions without grid)")
        print(f"   - motion_blocks_with_grid.png (zoomed with {args.block_size}×{args.block_size} grid)")
        print(f"   - Individual frame files for further processing")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
