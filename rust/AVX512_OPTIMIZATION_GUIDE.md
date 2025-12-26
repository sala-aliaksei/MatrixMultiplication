# AVX-512 Optimization Guide for Zen5 Matrix Multiplication

## Overview

This document describes the AVX-512 optimized kernel implementation for Zen5 architecture and how to view/analyze the generated assembly code.

## Key Optimizations

### 1. **Register Usage**

The AVX-512 kernel uses the full 512-bit ZMM registers:

#### For f64 (double precision):
- **8 doubles per ZMM register** (512 bits / 64 bits = 8)
- **Nr = 24** (3 ZMM registers for B matrix: zmm24, zmm25, zmm26)
- **Mr = 8** (8 rows)
- **24 ZMM accumulators** (zmm0-zmm23): 8 rows × 3 register-width columns

#### For f32 (single precision):
- **16 floats per ZMM register** (512 bits / 32 bits = 16)
- **Nr = 48** (3 ZMM registers for B matrix)
- **Mr = 8** (8 rows)
- **24 ZMM accumulators**: 8 rows × 3 register-width columns

### 2. **Key AVX-512 Instructions Used**

```assembly
# Initialize accumulators to zero
vpxord zmm0, zmm0, zmm0              # Zero out accumulator

# Load B matrix (3 registers, 24 f64 elements)
vmovupd zmm24, ZMMWORD PTR [rdx-0x80]  # Load 8 doubles from B
vmovupd zmm25, ZMMWORD PTR [rdx-0x40]  # Load next 8 doubles
vmovupd zmm26, ZMMWORD PTR [rdx]       # Load final 8 doubles

# Broadcast A matrix element to all lanes
vbroadcastsd zmm27, QWORD PTR [rdi+rax*8]  # Broadcast A[i] to all 8 lanes

# Fused Multiply-Add (FMA)
vfmadd231pd zmm23, zmm27, zmm24      # zmm23 = zmm23 + (zmm27 * zmm24)
```

### 3. **Performance Characteristics**

**Scalar vs AVX-512 Comparison:**

| Metric | Scalar | AVX-512 (f64) | Speedup |
|--------|--------|---------------|---------|
| SIMD Width | 1 | 8 | 8x |
| Instructions per iteration | ~7 | ~4 | ~1.75x |
| **Theoretical Peak** | - | **~14x** | Combined |
| Memory Bandwidth | Limited | Efficient | 8x wider |

**Key advantages:**
- **FMA instructions**: 2 FLOPs per instruction (multiply + add)
- **Wide SIMD**: Process 8 f64 or 16 f32 simultaneously
- **Reduced loop overhead**: Fewer iterations needed
- **Better cache utilization**: Vectorized loads/stores

### 4. **Assembly Analysis**

#### Initialization (Lines d43b-d465):
```assembly
vpxord zmm16, zmm16, zmm16  # Zero out 24 accumulator registers
vpxord zmm17, zmm17, zmm17  # (zmm0-zmm23)
...
```

#### Main Computation Loop (Lines d470-d5c0):
```assembly
# Load B matrix once per K iteration
vmovupd zmm24, [rdx-0x80]   # B[k][0:8]
vmovupd zmm25, [rdx-0x40]   # B[k][8:16]
vmovupd zmm26, [rdx]        # B[k][16:24]

# Process all 8 rows (unrolled)
vbroadcastsd zmm27, [rdi+rax*8]         # A[0][k]
vfmadd231pd zmm23, zmm27, zmm24         # C[0][0:8] += A[0][k] * B[k][0:8]
vfmadd231pd zmm22, zmm27, zmm25         # C[0][8:16] += ...
vfmadd231pd zmm21, zmm26, zmm27         # C[0][16:24] += ...

vbroadcastsd zmm27, [rdi+rax*8+0x8]    # A[1][k]
vfmadd231pd zmm20, zmm27, zmm24         # C[1][0:8] += ...
# ... continue for all 8 rows
```

#### Store Results (After main loop):
```assembly
# Load existing C values, add accumulators, store back
vmovupd zmm_temp, [c_ptr]
vaddpd zmm_temp, zmm_temp, zmm_acc
vmovupd [c_ptr], zmm_temp
```

## How to View Assembly

### Method 1: Using objdump (Recommended)
```bash
# Compile with AVX-512 enabled
RUSTFLAGS="-C target-feature=+avx512f" cargo build --example show_avx512_asm --release

# View assembly
objdump -d -M intel -C target/release/examples/show_avx512_asm | \
  grep -A200 "zen5_packed_kernel_f64_avx512" | less
```

### Method 2: Generate .s files
```bash
# Generate assembly with Intel syntax
RUSTFLAGS="-C target-feature=+avx512f" cargo rustc --example show_avx512_asm --release -- \
  --emit asm -C "llvm-args=-x86-asm-syntax=intel"

# View the file
cat target/release/examples/show_avx512_asm-*.s
```

### Method 3: cargo-asm
```bash
cargo install cargo-show-asm
cargo asm --example show_avx512_asm --rust zen5_packed_kernel_f64_avx512
```

### Method 4: Compiler Explorer (Godbolt)
Visit https://rust.godbolt.org/ and paste the kernel code with:
```rust
#![feature(stdsimd)]
// Add compiler flags: -C target-feature=+avx512f -C opt-level=3
```

## Performance Tuning Tips

### 1. **Compiler Flags**
```toml
# In Cargo.toml or .cargo/config.toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1

[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=native"]  # Or specifically: znver5
```

### 2. **CPU Features**
```bash
# Check available CPU features
rustc --print target-features

# Enable specific features
RUSTFLAGS="-C target-feature=+avx512f,+avx512dq,+avx512cd,+avx512bw,+avx512vl"
```

### 3. **Benchmarking**
```rust
// Use criterion for accurate benchmarks
cargo bench --bench mm_benchmark
```

### 4. **Prefetching**
The kernel could be further optimized with software prefetching:
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;

unsafe {
    _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
}
```

## Verification

Run tests to verify correctness:
```bash
cargo test --lib zen5_avx512_kernel
```

## Comparing with C++ Implementation

The Rust AVX-512 kernel matches the C++ implementation structure:
- Uses same register blocking (Mr=8, Nr=24 for f64)
- Same FMA pattern (broadcast A, multiply with B, accumulate to C)
- Similar memory access patterns

**Key differences:**
- Rust uses explicit `unsafe` blocks and `#[target_feature]` attributes
- C++ uses `std::experimental::simd` (portable SIMD abstraction)
- Rust intrinsics are more explicit but give finer control

## Expected Performance

On AMD Zen5 (9950X):
- **Theoretical Peak (f64)**: ~600 GFLOPS per core
- **Expected with kernel**: ~400-500 GFLOPS (66-83% peak)
- **Memory bound threshold**: ~3072x3072 matrices

Factors affecting performance:
1. Cache hit rates (L1/L2/L3)
2. Memory bandwidth (for large matrices)
3. Thread scheduling and affinity
4. TLB misses for very large matrices

## Next Steps

1. ✅ Created AVX-512 kernel with explicit intrinsics
2. ✅ Verified assembly generation shows ZMM registers
3. ⏳ Integrate kernel into main `matmul.rs`
4. ⏳ Add benchmarks comparing scalar vs AVX-512
5. ⏳ Tune for Zen5 specific optimizations (prefetching, alignment)

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [AMD Optimization Manual](https://www.amd.com/content/dam/amd/en/documents/processor-tech-docs/software-optimization-guides/57647.zip)
- [Rust std::arch documentation](https://doc.rust-lang.org/core/arch/index.html)
