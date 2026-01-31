"""
Profile encoder performance for DCT-only vs MV+DCT
Microbenchmark to isolate the speed difference
"""

import torch
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mots_exp.models.dct_mv_center.components.dct_mv_encoder import SpatiallyAlignedDCTMVEncoder


def benchmark_encoder(encoder, mv_input, dct_input, num_iters=1000, warmup=100):
    """Benchmark encoder forward pass"""
    encoder.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = encoder(mv_input, dct_input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_iters):
            _ = encoder(mv_input, dct_input)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed / num_iters * 1000  # ms per iteration


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 1
    num_iters = 1000
    warmup = 100
    
    print(f"\nBenchmark settings:")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iters}")
    print(f"  Warmup: {warmup}")
    
    # ========================================================================
    # TEST 1: DCT-only encoder (DCT-8)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: DCT-only encoder (8 channels)")
    print("="*80)
    
    encoder_dct = SpatiallyAlignedDCTMVEncoder(
        num_dct_coeffs=8,
        mv_channels=0,  # Disabled
        dct_channels=8,
        mv_feature_dim=32,
        dct_feature_dim=32,
        fused_feature_dim=64
    ).to(device)
    
    # Create dummy input
    dct_input = torch.randn(batch_size, 120, 120, 8).to(device)
    
    # Benchmark
    time_dct = benchmark_encoder(encoder_dct, None, dct_input, num_iters, warmup)
    
    print(f"\nResults:")
    print(f"  Time per iteration: {time_dct:.3f} ms")
    print(f"  Throughput: {1000 / time_dct:.1f} iterations/sec")
    
    # ========================================================================
    # TEST 2: MV+DCT encoder (MV + DCT-8)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: MV+DCT encoder (2 MV + 8 DCT channels)")
    print("="*80)
    
    encoder_mv_dct = SpatiallyAlignedDCTMVEncoder(
        num_dct_coeffs=8,
        mv_channels=2,  # Enabled
        dct_channels=8,
        mv_feature_dim=32,
        dct_feature_dim=32,
        fused_feature_dim=64
    ).to(device)
    
    # Create dummy inputs
    mv_input = torch.randn(batch_size, 2, 60, 60).to(device)
    dct_input = torch.randn(batch_size, 120, 120, 8).to(device)
    
    # Benchmark
    time_mv_dct = benchmark_encoder(encoder_mv_dct, mv_input, dct_input, num_iters, warmup)
    
    print(f"\nResults:")
    print(f"  Time per iteration: {time_mv_dct:.3f} ms")
    print(f"  Throughput: {1000 / time_mv_dct:.1f} iterations/sec")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    speedup = time_dct / time_mv_dct
    print(f"\nDCT-only:  {time_dct:.3f} ms/iter")
    print(f"MV+DCT:    {time_mv_dct:.3f} ms/iter")
    print(f"Speedup:   {speedup:.2f}x {'(MV+DCT FASTER!)' if speedup > 1 else '(DCT-only faster)'}")
    
    if speedup > 1:
        print(f"\n⚠️  PARADOX CONFIRMED: MV+DCT is {speedup:.2f}x faster despite processing MORE data!")
    else:
        print(f"\n✅ Expected: DCT-only is {1/speedup:.2f}x faster (processes less data)")
    
    # Memory usage
    print(f"\nMemory allocated:")
    print(f"  DCT-only:  {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  Peak:      {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    
    # ========================================================================
    # TEST 3: Breakdown by component
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Component-level breakdown")
    print("="*80)
    
    # Test DCT encoder alone
    print("\n[DCT-only] DCT encoder:")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            dct_perm = dct_input.permute(0, 3, 1, 2)
            _ = encoder_dct.dct_encoder(dct_perm)
        torch.cuda.synchronize()
        time_dct_encoder = (time.time() - start) / num_iters * 1000
    print(f"  Time: {time_dct_encoder:.3f} ms/iter")
    
    # Test fusion alone (DCT-only)
    print("\n[DCT-only] Fusion layer:")
    with torch.no_grad():
        dct_perm = dct_input.permute(0, 3, 1, 2)
        dct_features = encoder_dct.dct_encoder(dct_perm)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = encoder_dct.fusion(dct_features)
        torch.cuda.synchronize()
        time_dct_fusion = (time.time() - start) / num_iters * 1000
    print(f"  Time: {time_dct_fusion:.3f} ms/iter")
    
    print(f"\n[DCT-only] Total accounted: {time_dct_encoder + time_dct_fusion:.3f} ms")
    print(f"[DCT-only] Measured total: {time_dct:.3f} ms")
    print(f"[DCT-only] Overhead: {time_dct - (time_dct_encoder + time_dct_fusion):.3f} ms")
    
    # Test MV+DCT components
    print("\n[MV+DCT] MV encoder:")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            mv_feat = encoder_mv_dct.mv_encoder(mv_input)
            mv_up = encoder_mv_dct.mv_upsample(mv_feat)
            _ = encoder_mv_dct.mv_refine(mv_up)
        torch.cuda.synchronize()
        time_mv_encoder = (time.time() - start) / num_iters * 1000
    print(f"  Time: {time_mv_encoder:.3f} ms/iter")
    
    print("\n[MV+DCT] DCT encoder:")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            dct_perm = dct_input.permute(0, 3, 1, 2)
            _ = encoder_mv_dct.dct_encoder(dct_perm)
        torch.cuda.synchronize()
        time_mv_dct_encoder = (time.time() - start) / num_iters * 1000
    print(f"  Time: {time_mv_dct_encoder:.3f} ms/iter")
    
    print("\n[MV+DCT] Fusion layer:")
    with torch.no_grad():
        mv_feat = encoder_mv_dct.mv_encoder(mv_input)
        mv_up = encoder_mv_dct.mv_upsample(mv_feat)
        mv_refined = encoder_mv_dct.mv_refine(mv_up)
        dct_perm = dct_input.permute(0, 3, 1, 2)
        dct_features = encoder_mv_dct.dct_encoder(dct_perm)
        combined = torch.cat([mv_refined, dct_features], dim=1)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = encoder_mv_dct.fusion(combined)
        torch.cuda.synchronize()
        time_mv_dct_fusion = (time.time() - start) / num_iters * 1000
    print(f"  Time: {time_mv_dct_fusion:.3f} ms/iter")
    
    print(f"\n[MV+DCT] Total accounted: {time_mv_encoder + time_mv_dct_encoder + time_mv_dct_fusion:.3f} ms")
    print(f"[MV+DCT] Measured total: {time_mv_dct:.3f} ms")
    print(f"[MV+DCT] Overhead: {time_mv_dct - (time_mv_encoder + time_mv_dct_encoder + time_mv_dct_fusion):.3f} ms")
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    print(f"\nComponent comparison:")
    print(f"  DCT encoder (DCT-only):  {time_dct_encoder:.3f} ms")
    print(f"  DCT encoder (MV+DCT):    {time_mv_dct_encoder:.3f} ms")
    print(f"  Fusion (DCT-only 32→64): {time_dct_fusion:.3f} ms")
    print(f"  Fusion (MV+DCT 64→64):   {time_mv_dct_fusion:.3f} ms")
    print(f"  MV branch (only MV+DCT): {time_mv_encoder:.3f} ms")
    
    print(f"\nKey finding:")
    if time_dct_fusion > time_mv_dct_fusion:
        diff = time_dct_fusion - time_mv_dct_fusion
        print(f"  ⚠️  DCT-only fusion is {diff:.3f} ms SLOWER ({time_dct_fusion/time_mv_dct_fusion:.2f}x)")
        print(f"  This explains the paradox: 32→64 expansion is less efficient than 64→64!")
    else:
        print(f"  DCT-only fusion is actually faster, paradox unexplained")


if __name__ == "__main__":
    main()
