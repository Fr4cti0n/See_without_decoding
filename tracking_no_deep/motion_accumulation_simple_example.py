#!/usr/bin/env python3
"""
Simple Example: Motion Vector Accumulation Step-by-Step

This example shows exactly how motion vectors are summed to get displacement at time t.
"""

import numpy as np
import matplotlib.pyplot as plt

def motion_vector_accumulation_example():
    """Step-by-step example of motion vector accumulation."""
    
    print("🎯 MOTION VECTOR ACCUMULATION: SIMPLE EXAMPLE")
    print("=" * 60)
    
    print("""
📊 SCENARIO: Tracking a person walking across a video
- Video has 6 frames (1 I-frame + 5 P-frames)
- Person starts at position (100, 200) pixels
- Each P-frame contains motion vectors showing movement
""")
    
    # Example motion vectors for 5 P-frames
    motion_vectors = [
        np.array([3.0, -1.0]),   # Frame 1: Move 3 pixels right, 1 pixel up
        np.array([2.5, -0.5]),   # Frame 2: Move 2.5 pixels right, 0.5 pixels up
        np.array([4.0, -2.0]),   # Frame 3: Move 4 pixels right, 2 pixels up
        np.array([2.0, -1.5]),   # Frame 4: Move 2 pixels right, 1.5 pixels up
        np.array([3.5, -0.8])    # Frame 5: Move 3.5 pixels right, 0.8 pixels up
    ]
    
    # Initial position from I-frame object detection
    initial_position = np.array([100.0, 200.0])
    
    print("🔧 STEP-BY-STEP CALCULATION:")
    print("=" * 40)
    
    # Track positions through accumulation
    positions = [initial_position.copy()]
    accumulated_displacement = np.array([0.0, 0.0])
    
    print(f"Frame 0 (I-frame):")
    print(f"   Position: {initial_position}")
    print(f"   Accumulated displacement: {accumulated_displacement}")
    print()
    
    for frame_idx, motion_vec in enumerate(motion_vectors, 1):
        # THIS IS THE KEY: Add current motion to total accumulated displacement
        accumulated_displacement += motion_vec
        
        # Calculate new position = initial + accumulated displacement
        new_position = initial_position + accumulated_displacement
        positions.append(new_position.copy())
        
        print(f"Frame {frame_idx} (P-frame):")
        print(f"   Motion vector: {motion_vec}")
        print(f"   Accumulated displacement: {accumulated_displacement}")
        print(f"   New position: {new_position}")
        print(f"   Formula: position = initial + accumulated_displacement")
        print(f"   Formula: {new_position} = {initial_position} + {accumulated_displacement}")
        print()
    
    print("📊 MATHEMATICAL FORMULATION:")
    print("=" * 40)
    print("""
For any frame t:
   accumulated_displacement[t] = Σ(motion_vector[i]) for i = 0 to t-1
   position[t] = initial_position + accumulated_displacement[t]

Explicitly:
   Frame 0: position = initial_position + 0
   Frame 1: position = initial_position + MV₀
   Frame 2: position = initial_position + MV₀ + MV₁
   Frame 3: position = initial_position + MV₀ + MV₁ + MV₂
   Frame t: position = initial_position + Σ(MV₀...MVₜ₋₁)
""")
    
    print("🧮 NUMERICAL EXAMPLE:")
    print("=" * 25)
    
    print("Initial position: (100.0, 200.0)")
    print()
    
    accumulated_disp = np.array([0.0, 0.0])
    for i, mv in enumerate(motion_vectors):
        accumulated_disp += mv
        pos = initial_position + accumulated_disp
        
        print(f"Frame {i+1}:")
        print(f"   Motion vector: ({mv[0]:4.1f}, {mv[1]:4.1f})")
        print(f"   Sum so far:    ({accumulated_disp[0]:4.1f}, {accumulated_disp[1]:4.1f})")
        print(f"   Position:      ({pos[0]:4.1f}, {pos[1]:4.1f})")
        print()
    
    print("🔍 WHY ACCUMULATION WORKS:")
    print("=" * 30)
    print("""
1. TEMPORAL CONSISTENCY:
   Each motion vector shows displacement from previous frame
   Summing them gives total displacement from start
   
2. RELATIVE MOVEMENT:
   Motion vectors are RELATIVE displacements
   To get absolute position, add to starting point
   
3. COORDINATE SYSTEM:
   All motion vectors in same coordinate system
   Addition is straightforward vector arithmetic
""")
    
    print("🎬 CODE IMPLEMENTATION:")
    print("=" * 25)
    print("""
```python
# Initialize tracking
initial_pos = detect_object_in_iframe()  # From object detection
accumulated_displacement = np.array([0.0, 0.0])
positions = [initial_pos.copy()]

# For each P-frame
for frame_idx in range(1, num_frames):
    # Extract motion vector at object location
    motion_vec = extract_motion_at_position(motion_field, current_pos)
    
    # KEY STEP: Accumulate motion
    accumulated_displacement += motion_vec
    
    # Calculate new position
    new_pos = initial_pos + accumulated_displacement
    positions.append(new_pos)
    
    # Update bounding box around new position
    bbox = create_bbox_around_center(new_pos, object_size)
```
""")
    
    print("⚠️  IMPORTANT DETAILS:")
    print("=" * 20)
    print("""
1. MOTION VECTOR EXTRACTION:
   - Object position → macroblock coordinates: mb_x = pos_x // 16
   - Extract motion: motion_vec = motion_field[mb_y, mb_x]
   
2. COORDINATE CLAMPING:
   - Keep positions within frame bounds: [0, 959] for 960x960 video
   - Prevents objects from moving outside visible area
   
3. SMOOTHING:
   - Apply Gaussian filter to motion field before extraction
   - Reduces noise in motion vectors
   
4. MULTI-LAYER COMBINATION:
   - Combine multiple motion layers: 0.6 * layer0 + 0.4 * layer1
   - Provides more robust motion estimation
""")
    
    # Visual representation
    create_visual_example(positions, motion_vectors, initial_position)
    
    return positions, motion_vectors

def create_visual_example(positions, motion_vectors, initial_pos):
    """Create a visual representation of motion accumulation."""
    
    print("\n📈 VISUAL REPRESENTATION:")
    print("=" * 30)
    
    # Convert to numpy arrays for easier handling
    positions = np.array(positions)
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    
    # Create ASCII visualization
    print("Motion Path (X-axis: horizontal, Y-axis: vertical)")
    print("=" * 50)
    
    # Normalize coordinates for ASCII display
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Simple path display
    print(f"Start: ({initial_pos[0]:.1f}, {initial_pos[1]:.1f}) 🟦")
    
    for i, (pos, mv) in enumerate(zip(positions[1:], motion_vectors), 1):
        arrow = "→" if mv[0] > 0 else "←" if mv[0] < 0 else "↕"
        if abs(mv[1]) > abs(mv[0]):
            arrow = "↑" if mv[1] < 0 else "↓"
        
        print(f"Frame {i}: ({pos[0]:.1f}, {pos[1]:.1f}) {arrow} [MV: ({mv[0]:+.1f}, {mv[1]:+.1f})]")
    
    print(f"End: ({positions[-1][0]:.1f}, {positions[-1][1]:.1f}) 🟩")
    print()
    
    # Show total displacement
    total_displacement = positions[-1] - positions[0]
    total_distance = np.linalg.norm(total_displacement)
    
    print(f"📊 SUMMARY STATISTICS:")
    print(f"   Total displacement: ({total_displacement[0]:.1f}, {total_displacement[1]:.1f})")
    print(f"   Total distance moved: {total_distance:.1f} pixels")
    print(f"   Average motion per frame: {total_distance/len(motion_vectors):.1f} pixels")
    
    # Verify our calculation
    manual_sum = np.sum(motion_vectors, axis=0)
    print(f"   Manual sum of motion vectors: ({manual_sum[0]:.1f}, {manual_sum[1]:.1f})")
    print(f"   ✓ Matches total displacement: {np.allclose(total_displacement, manual_sum)}")

def demonstrate_accumulation_properties():
    """Demonstrate key properties of motion vector accumulation."""
    
    print("\n🔬 ACCUMULATION PROPERTIES:")
    print("=" * 35)
    
    # Property 1: Associativity
    mv1 = np.array([2.0, -1.0])
    mv2 = np.array([3.0, 1.5])
    mv3 = np.array([1.5, -0.5])
    
    # Different groupings give same result
    result1 = (mv1 + mv2) + mv3
    result2 = mv1 + (mv2 + mv3)
    result3 = mv1 + mv2 + mv3
    
    print("1. ASSOCIATIVITY:")
    print(f"   (MV₁ + MV₂) + MV₃ = {result1}")
    print(f"   MV₁ + (MV₂ + MV₃) = {result2}")
    print(f"   MV₁ + MV₂ + MV₃ = {result3}")
    print(f"   All equal: {np.allclose(result1, result2) and np.allclose(result2, result3)}")
    print()
    
    # Property 2: Commutativity (order doesn't matter for final sum)
    order1 = mv1 + mv2 + mv3
    order2 = mv3 + mv1 + mv2
    order3 = mv2 + mv3 + mv1
    
    print("2. COMMUTATIVITY (final sum):")
    print(f"   MV₁ + MV₂ + MV₃ = {order1}")
    print(f"   MV₃ + MV₁ + MV₂ = {order2}")
    print(f"   MV₂ + MV₃ + MV₁ = {order3}")
    print(f"   All equal: {np.allclose(order1, order2) and np.allclose(order2, order3)}")
    print()
    
    # Property 3: Incremental accumulation
    print("3. INCREMENTAL ACCUMULATION:")
    accumulated = np.array([0.0, 0.0])
    motions = [mv1, mv2, mv3]
    
    for i, mv in enumerate(motions):
        accumulated += mv
        print(f"   After frame {i+1}: accumulated = {accumulated}")
    
    print(f"   Final accumulated = {accumulated}")
    print(f"   Direct sum = {np.sum(motions, axis=0)}")
    print(f"   Match: {np.allclose(accumulated, np.sum(motions, axis=0))}")

def show_real_world_analogy():
    """Show real-world analogy for motion vector accumulation."""
    
    print("\n🌍 REAL-WORLD ANALOGY:")
    print("=" * 25)
    
    print("""
Think of motion vector accumulation like tracking someone's walk:

🚶 WALKING ANALOGY:
   - You start at your house (initial position)
   - Step 1: Walk 3 blocks east, 1 block north
   - Step 2: Walk 2 blocks east, 1 block south  
   - Step 3: Walk 4 blocks east, 2 blocks north
   
   To find where you are after Step 3:
   ✅ CORRECT: Add all steps to starting position
      Final position = house + step1 + step2 + step3
   
   ❌ WRONG: Only use the last step
      This would put you near the house, not at your actual location!

🎯 MOTION VECTOR ANALOGY:
   - Initial position = house location
   - Motion vectors = walking steps  
   - Accumulated displacement = sum of all steps
   - Current position = house + sum of all steps
   
This is exactly what motion vector accumulation does:
   position[t] = initial_position + Σ(motion_vectors[0...t-1])
""")

if __name__ == "__main__":
    # Run the example
    positions, motion_vectors = motion_vector_accumulation_example()
    demonstrate_accumulation_properties()
    show_real_world_analogy()
    
    print("\n✅ CONCLUSION:")
    print("=" * 15)
    print("""
Motion vector accumulation is simply adding up all the motion vectors
from the start to get the total displacement at any time t.

Key formula: position[t] = initial_position + Σ(motion_vectors[0...t-1])

This gives us the object's current location by tracking all its movement
from the initial detection in the I-frame.
""")
