"""
Highly optimized matrix multiplication kernel for RTX 5090 using OpenAI Triton.

This implementation leverages:
- Tensor Core operations via tl.dot() for FP16/BF16
- Efficient tiling strategies optimized for RTX 5090's Blackwell architecture
- Grouped program IDs for better L2 cache utilization
- Coalesced memory access patterns
- Autotuning support for optimal performance

Optimized tile sizes for RTX 5090:
- 170 SMs with 128 CUDA cores each
- 680 Tensor Cores (4 per SM, 5th generation)
- Enhanced memory hierarchy in Blackwell architecture
"""

import triton
import triton.language as tl
import torch
import numpy as np
import sys
import os

# Debug info
print(f"DEBUG: Python executable: {sys.executable}")
print(f"DEBUG: CWD: {os.getcwd()}")
print(f"DEBUG: Torch version: {torch.__version__}")
try:
    print(f"DEBUG: CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"DEBUG: CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"DEBUG: CUDA check error: {e}")

from typing import Optional, Tuple, Union

# ============================================================================
# Tile Size Constants Optimized for RTX 5090 (Blackwell Architecture)
# ============================================================================

# Block-level tile sizes
# These are optimized for RTX 5090's memory hierarchy and Tensor Core capabilities
BLOCK_M = 128  # Tile size in M dimension per program instance
BLOCK_N = 128  # Tile size in N dimension per program instance
BLOCK_K = 64   # Tile size in K dimension (reduction dimension)
GROUP_SIZE_M = 8  # Group size for better L2 cache utilization

# ============================================================================
# Core Matrix Multiplication Kernel
# ============================================================================

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    # Stride information
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Highly optimized matrix multiplication kernel for RTX 5090.
    
    Computes C = A @ B where:
    - A: [M, K] matrix
    - B: [K, N] matrix
    - C: [M, N] matrix
    
    This kernel uses:
    - Grouped program IDs for better L2 cache utilization
    - Tiled computation over K dimension
    - Tensor Core operations via tl.dot() for FP16/BF16
    - Efficient memory access patterns with proper masking
    
    Args:
        A, B, C: Pointers to input/output matrices
        M, N, K: Matrix dimensions
        stride_*: Stride information for each matrix dimension
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile sizes
        GROUP_SIZE_M: Group size for program ID grouping
    """
    # Compute program IDs with grouping for better cache locality
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator (use float32 for precision)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Tiled computation over K dimension
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_blocks * BLOCK_SIZE_K, BLOCK_SIZE_K):
        # Compute pointers for A and B tiles
        a_ptrs = A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak)
        b_ptrs = B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn)
        
        # Load A and B tiles with proper masking for boundary conditions
        # Mask ensures we don't read out of bounds
        a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k) < K)
        b_mask = ((offs_k[:, None] + k) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Matrix multiplication using Tensor Cores (for FP16/BF16)
        # tl.dot() automatically uses Tensor Cores when available
        acc += tl.dot(a, b)
    
    # Write back results with proper masking
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ============================================================================
# Main Matrix Multiplication Function
# ============================================================================

def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
    group_size_m: int = GROUP_SIZE_M,
) -> torch.Tensor:
    """
    Highly optimized matrix multiplication for RTX 5090 using Triton.
    
    Computes C = A @ B where:
    - A: [M, K] matrix
    - B: [K, N] matrix
    - C: [M, N] matrix
    
    Supports FP16 and BF16 data types with automatic Tensor Core utilization.
    
    Args:
        a: Input matrix A [M, K]
        b: Input matrix B [K, N]
        block_m: Block tile size in M dimension (default: 128)
        block_n: Block tile size in N dimension (default: 128)
        block_k: Block tile size in K dimension (default: 64)
        group_size_m: Group size for program ID grouping (default: 8)
    
    Returns:
        Output matrix C [M, N] with same dtype as input
    
    Raises:
        ValueError: If matrix dimensions are incompatible
    """
    # Validate inputs
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Inputs must be 2D matrices")
    
    M, K = a.shape
    K_b, N = b.shape
    
    if K != K_b:
        raise ValueError(
            f"Incompatible matrix dimensions: A is [{M}, {K}], B is [{K_b}, {N}]"
        )
    
    # Ensure tensors are on GPU and contiguous
    if not a.is_cuda:
        a = a.cuda()
    if not b.is_cuda:
        b = b.cuda()
    a = a.contiguous()
    b = b.contiguous()
    
    # Allocate output tensor
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    
    # Calculate grid dimensions
    grid_m = triton.cdiv(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    grid = (grid_m * grid_n,)
    
    # Launch kernel
    matmul_kernel[grid](
        a, b, c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_size_m,
    )
    
    return c


# ============================================================================
# Autotuned Matrix Multiplication
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8},
            num_stages=3,
            num_warps=8,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_autotuned(
    A, B, C,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Autotuned version of matmul_kernel - same implementation as above."""
    # Compute program IDs with grouping
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Tiled computation over K dimension
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(0, num_k_blocks * BLOCK_SIZE_K, BLOCK_SIZE_K):
        a_ptrs = A + (offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak)
        b_ptrs = B + ((offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn)
        
        a_mask = (offs_m[:, None] < M) & ((offs_k[None, :] + k) < K)
        b_mask = ((offs_k[:, None] + k) < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    # Write back results
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def matmul_autotuned(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    Autotuned version of matrix multiplication.
    
    Automatically selects optimal tile sizes based on matrix dimensions.
    
    Args:
        a: Input matrix A [M, K]
        b: Input matrix B [K, N]
    
    Returns:
        Output matrix C [M, N] with same dtype as input
    """
    # Validate inputs
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Inputs must be 2D matrices")
    
    M, K = a.shape
    K_b, N = b.shape
    
    if K != K_b:
        raise ValueError(
            f"Incompatible matrix dimensions: A is [{M}, {K}], B is [{K_b}, {N}]"
        )
    
    # Ensure tensors are on GPU and contiguous
    if not a.is_cuda:
        a = a.cuda()
    if not b.is_cuda:
        b = b.cuda()
    a = a.contiguous()
    b = b.contiguous()
    
    # Allocate output tensor
    dtype = a.dtype
    c = torch.empty((M, N), device=a.device, dtype=dtype)
    
    # Calculate grid dimensions
    # Use minimum block size from autotune configs to ensure we have enough program IDs
    # The autotune will select the optimal block size, but we need enough grid points
    min_block_m = 64  # Minimum BLOCK_SIZE_M in autotune configs
    min_block_n = 64  # Minimum BLOCK_SIZE_N in autotune configs
    max_group_size_m = 8  # Maximum GROUP_SIZE_M in autotune configs
    grid_m = triton.cdiv(M, min_block_m)
    grid_n = triton.cdiv(N, min_block_n)
    # Grid size accounts for grouping: each group processes GROUP_SIZE_M blocks in M dimension
    grid = (grid_m * grid_n,)
    
    # Launch autotuned kernel
    matmul_kernel_autotuned[grid](
        a, b, c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
    )
    
    return c


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_matmul(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    dtype: torch.dtype = torch.float16,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    verbose: bool = True,
    use_autotuned: bool = False,
) -> bool:
    """
    Validate the optimized matmul against PyTorch reference implementation.
    
    Args:
        M, N, K: Matrix dimensions to test
        dtype: Data type to test (torch.float16 or torch.bfloat16)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        verbose: Whether to print validation results
        use_autotuned: Whether to use autotuned version
    
    Returns:
        True if validation passes, False otherwise
    """
    # Generate random test matrices
    torch.manual_seed(42)
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    
    # Reference implementation (PyTorch)
    c_ref = torch.matmul(a, b)
    
    # Optimized implementation
    try:
        if use_autotuned:
            c_opt = matmul_autotuned(a, b)
        else:
            c_opt = matmul(a, b)
        
        # Compare results
        # Convert to float32 for comparison to avoid precision issues
        c_ref_f32 = c_ref.float()
        c_opt_f32 = c_opt.float()
        
        max_diff = torch.max(torch.abs(c_ref_f32 - c_opt_f32)).item()
        mean_diff = torch.mean(torch.abs(c_ref_f32 - c_opt_f32)).item()
        
        # Check if results match within tolerance
        matches = torch.allclose(c_ref_f32, c_opt_f32, rtol=rtol, atol=atol)
        
        if verbose:
            print(f"Validation Test: [{M}, {K}] @ [{K}, {N}] = [{M}, {N}]")
            print(f"  Data type: {dtype}")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")
            print(f"  Match: {'✓ PASS' if matches else '✗ FAIL'}")
        
        return bool(matches)
    
    except Exception as e:
        if verbose:
            print(f"Validation failed with error: {e}")
        return False


def run_validation_suite(
    verbose: bool = True,
    use_autotuned: bool = False,
) -> bool:
    """
    Run a suite of validation tests with various matrix sizes.
    
    Args:
        verbose: Whether to print validation results
        use_autotuned: Whether to use autotuned version
    
    Returns:
        True if all tests pass, False otherwise
    """
    test_cases = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    
    dtypes = [torch.float16, torch.bfloat16]
    
    all_passed = True
    for M, N, K in test_cases:
        for dtype in dtypes:
            passed = validate_matmul(
                M, N, K, dtype=dtype, verbose=verbose, use_autotuned=use_autotuned
            )
            all_passed = all_passed and passed
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        print(f"{'='*50}")
    
    return all_passed


# ============================================================================
# Benchmarking Utilities
# ============================================================================

def benchmark_matmul(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype = torch.float16,
    num_warmup: int = 10,
    num_iterations: int = 100,
    compare_with_torch: bool = True,
    use_autotuned: bool = False,
) -> dict:
    """
    Benchmark the optimized matmul implementation.
    
    Args:
        M, N, K: Matrix dimensions
        dtype: Data type to benchmark
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        compare_with_torch: Whether to also benchmark PyTorch reference
        use_autotuned: Whether to use autotuned version
    
    Returns:
        Dictionary containing benchmark results:
        - 'optimized_time': Average time for optimized implementation (seconds)
        - 'optimized_tflops': Throughput in TFLOPS
        - 'torch_time': Average time for PyTorch reference (seconds, if compare_with_torch=True)
        - 'torch_tflops': Throughput in TFLOPS (if compare_with_torch=True)
        - 'speedup': Speedup factor (if compare_with_torch=True)
    """
    # Generate test matrices
    torch.manual_seed(42)
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    
    # Calculate number of operations (2 * M * N * K for matmul)
    num_ops = 2 * M * N * K
    
    # Warmup
    for _ in range(num_warmup):
        if use_autotuned:
            _ = matmul_autotuned(a, b)
        else:
            _ = matmul(a, b)
        torch.cuda.synchronize()
    
    # Benchmark optimized implementation
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_iterations):
        if use_autotuned:
            c_opt = matmul_autotuned(a, b)
        else:
            c_opt = matmul(a, b)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_time_opt = start_event.elapsed_time(end_event) / num_iterations / 1000.0  # Convert ms to seconds
    tflops_opt = (num_ops / 1e12) / avg_time_opt
    
    results = {
        'optimized_time': avg_time_opt,
        'optimized_tflops': tflops_opt,
        'matrix_size': f"{M}x{K} @ {K}x{N} = {M}x{N}",
        'dtype': str(dtype),
    }
    
    # Benchmark PyTorch reference if requested
    if compare_with_torch:
        # Warmup
        for _ in range(num_warmup):
            _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        
        # Time PyTorch reference
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_iterations):
            c_torch = torch.matmul(a, b)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_time_torch = start_event.elapsed_time(end_event) / num_iterations / 1000.0
        tflops_torch = (num_ops / 1e12) / avg_time_torch
        
        results['torch_time'] = avg_time_torch
        results['torch_tflops'] = tflops_torch
        results['speedup'] = avg_time_torch / avg_time_opt
    
    return results


def print_benchmark_results(results: dict):
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {results['matrix_size']}")
    print(f"Data Type: {results['dtype']}")
    print(f"{'='*60}")
    print(f"Optimized Implementation:")
    print(f"  Time:       {results['optimized_time']*1000:.3f} ms")
    print(f"  Throughput: {results['optimized_tflops']:.2f} TFLOPS")
    
    if 'torch_time' in results:
        print(f"\nPyTorch Reference Implementation:")
        print(f"  Time:       {results['torch_time']*1000:.3f} ms")
        print(f"  Throughput: {results['torch_tflops']:.2f} TFLOPS")
        print(f"\nSpeedup: {results['speedup']:.2f}x")
    print(f"{'='*60}\n")


def run_benchmark_suite(
    sizes: Optional[list] = None,
    dtype: torch.dtype = torch.float16,
    compare_with_torch: bool = True,
    use_autotuned: bool = False,
) -> list:
    """
    Run benchmarks for multiple matrix sizes.
    
    Args:
        sizes: List of (M, N, K) tuples to benchmark. If None, uses default sizes.
        dtype: Data type to benchmark
        compare_with_torch: Whether to compare with PyTorch reference
        use_autotuned: Whether to use autotuned version
    
    Returns:
        List of benchmark result dictionaries
    """
    if sizes is None:
        sizes = [
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
        ]
    
    all_results = []
    for M, N, K in sizes:
        print(f"Benchmarking {M}x{K} @ {K}x{N} = {M}x{N}...")
        results = benchmark_matmul(
            M, N, K,
            dtype=dtype,
            compare_with_torch=compare_with_torch,
            use_autotuned=use_autotuned,
        )
        print_benchmark_results(results)
        all_results.append(results)
    
    return all_results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("RTX 5090 Optimized Matrix Multiplication (Triton)")
    print("=" * 60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print(f"DEBUG: Torch CUDA is_available() returned False")
        print(f"DEBUG: Torch version: {torch.__version__}")
        print("ERROR: CUDA is not available. This implementation requires a CUDA-capable GPU.")
        exit(1)
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print()
    
    # Run validation tests
    print("Running validation tests...")
    print("-" * 60)
    validation_passed = run_validation_suite(use_autotuned=False)
    
    if validation_passed:
        # Run benchmarks
        print("\nRunning benchmarks (FP16)...")
        print("-" * 60)
        run_benchmark_suite(dtype=torch.float16, use_autotuned=False)
        
        print("\nRunning benchmarks (BF16)...")
        print("-" * 60)
        run_benchmark_suite(dtype=torch.bfloat16, use_autotuned=False)
        
        # Optionally test autotuned version
        print("\nTesting autotuned version...")
        print("-" * 60)
        print("Note: First run will be slower due to autotuning overhead")
        validation_autotuned = run_validation_suite(use_autotuned=True, verbose=False)
        if validation_autotuned:
            print("Autotuned version validation: ✓ PASSED")
            run_benchmark_suite(
                sizes=[(1024, 1024, 1024), (2048, 2048, 2048)],
                dtype=torch.float16,
                use_autotuned=True,
            )
        else:
            print("Autotuned version validation: ✗ FAILED")
    else:
        print("\nSkipping benchmarks due to validation failures.")
