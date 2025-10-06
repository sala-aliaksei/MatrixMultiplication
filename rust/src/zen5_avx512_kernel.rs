// AVX-512 optimized kernel for Zen5
// This module provides high-performance matrix multiplication kernels using AVX-512 intrinsics

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX-512 optimized kernel for f64 (double precision)
///
/// # Arguments
/// * `a` - Packed A micro-panel (Mr x Kc), column-major within micro-tiles
/// * `b` - Packed B micro-panel (Kc x Nr), row-major within micro-tiles  
/// * `c` - Output C tile (Mr x Nr), accumulates results
/// * `ldc` - Leading dimension of C (typically N, the full matrix width)
/// * `nr` - Number of columns in micro-tile (must be divisible by 8 for AVX-512)
/// * `mr` - Number of rows in micro-tile (typically 8)
/// * `kc` - K dimension of micro-tile
///
/// # Performance Notes
/// - Uses 3 ZMM registers for B (3 * 8 = 24 elements)
/// - Uses 1 ZMM register for broadcasting A
/// - Uses 24 ZMM registers for accumulators (3 * 8)
/// - Optimized for Zen5 architecture with AVX-512
#[target_feature(enable = "avx512f")]
#[inline(never)]
pub unsafe fn zen5_packed_kernel_f64_avx512<const NR: usize, const MR: usize>(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    kc: usize,
) {
    // Number of AVX-512 registers needed for one row (8 f64 per register)
    const SIMD_WIDTH: usize = 8;
    let nrs = NR / SIMD_WIDTH; // Number of SIMD registers per row

    debug_assert_eq!(NR % SIMD_WIDTH, 0, "Nr must be divisible by 8 for AVX-512");
    debug_assert_eq!(MR, 8, "Mr should be 8 for optimal performance");
    debug_assert_eq!(nrs, 3, "Expected 3 registers (24 elements) for Nr");

    // Accumulator registers: Mr x Nrs = 8 x 3 = 24 registers
    let mut acc: [[__m512d; 3]; 8] = [[_mm512_setzero_pd(); 3]; 8];

    // Main computation loop over K
    let mut a_ptr = 0;
    let mut b_ptr = 0;

    for _ in 0..kc {
        // Load B registers (3 registers, 24 elements total)
        let b0 = _mm512_loadu_pd(b.as_ptr().add(b_ptr));
        let b1 = _mm512_loadu_pd(b.as_ptr().add(b_ptr + 8));
        let b2 = _mm512_loadu_pd(b.as_ptr().add(b_ptr + 16));

        // Process all Mr rows with the same B values
        // Unrolled loop for 8 rows
        let a0 = _mm512_set1_pd(a[a_ptr]);
        acc[0][0] = _mm512_fmadd_pd(a0, b0, acc[0][0]);
        acc[0][1] = _mm512_fmadd_pd(a0, b1, acc[0][1]);
        acc[0][2] = _mm512_fmadd_pd(a0, b2, acc[0][2]);

        let a1 = _mm512_set1_pd(a[a_ptr + 1]);
        acc[1][0] = _mm512_fmadd_pd(a1, b0, acc[1][0]);
        acc[1][1] = _mm512_fmadd_pd(a1, b1, acc[1][1]);
        acc[1][2] = _mm512_fmadd_pd(a1, b2, acc[1][2]);

        let a2 = _mm512_set1_pd(a[a_ptr + 2]);
        acc[2][0] = _mm512_fmadd_pd(a2, b0, acc[2][0]);
        acc[2][1] = _mm512_fmadd_pd(a2, b1, acc[2][1]);
        acc[2][2] = _mm512_fmadd_pd(a2, b2, acc[2][2]);

        let a3 = _mm512_set1_pd(a[a_ptr + 3]);
        acc[3][0] = _mm512_fmadd_pd(a3, b0, acc[3][0]);
        acc[3][1] = _mm512_fmadd_pd(a3, b1, acc[3][1]);
        acc[3][2] = _mm512_fmadd_pd(a3, b2, acc[3][2]);

        let a4 = _mm512_set1_pd(a[a_ptr + 4]);
        acc[4][0] = _mm512_fmadd_pd(a4, b0, acc[4][0]);
        acc[4][1] = _mm512_fmadd_pd(a4, b1, acc[4][1]);
        acc[4][2] = _mm512_fmadd_pd(a4, b2, acc[4][2]);

        let a5 = _mm512_set1_pd(a[a_ptr + 5]);
        acc[5][0] = _mm512_fmadd_pd(a5, b0, acc[5][0]);
        acc[5][1] = _mm512_fmadd_pd(a5, b1, acc[5][1]);
        acc[5][2] = _mm512_fmadd_pd(a5, b2, acc[5][2]);

        let a6 = _mm512_set1_pd(a[a_ptr + 6]);
        acc[6][0] = _mm512_fmadd_pd(a6, b0, acc[6][0]);
        acc[6][1] = _mm512_fmadd_pd(a6, b1, acc[6][1]);
        acc[6][2] = _mm512_fmadd_pd(a6, b2, acc[6][2]);

        let a7 = _mm512_set1_pd(a[a_ptr + 7]);
        acc[7][0] = _mm512_fmadd_pd(a7, b0, acc[7][0]);
        acc[7][1] = _mm512_fmadd_pd(a7, b1, acc[7][1]);
        acc[7][2] = _mm512_fmadd_pd(a7, b2, acc[7][2]);

        a_ptr += MR;
        b_ptr += NR;
    }

    // Store results back to C with accumulation (C += results)
    for i in 0..MR {
        let c_row_offset = i * ldc;

        // Load existing C values, add accumulators, and store back
        let c0 = _mm512_loadu_pd(c.as_ptr().add(c_row_offset));
        let c1 = _mm512_loadu_pd(c.as_ptr().add(c_row_offset + 8));
        let c2 = _mm512_loadu_pd(c.as_ptr().add(c_row_offset + 16));

        let c0_new = _mm512_add_pd(c0, acc[i][0]);
        let c1_new = _mm512_add_pd(c1, acc[i][1]);
        let c2_new = _mm512_add_pd(c2, acc[i][2]);

        _mm512_storeu_pd(c.as_mut_ptr().add(c_row_offset), c0_new);
        _mm512_storeu_pd(c.as_mut_ptr().add(c_row_offset + 8), c1_new);
        _mm512_storeu_pd(c.as_mut_ptr().add(c_row_offset + 16), c2_new);
    }
}

/// AVX-512 optimized kernel for f32 (single precision)
///
/// # Arguments
/// * `a` - Packed A micro-panel (Mr x Kc), column-major within micro-tiles
/// * `b` - Packed B micro-panel (Kc x Nr), row-major within micro-tiles
/// * `c` - Output C tile (Mr x Nr), accumulates results
/// * `ldc` - Leading dimension of C
/// * `nr` - Number of columns in micro-tile (must be divisible by 16 for AVX-512)
/// * `mr` - Number of rows in micro-tile (typically 8)
/// * `kc` - K dimension of micro-tile
///
/// # Performance Notes
/// - Uses 3 ZMM registers for B (3 * 16 = 48 elements)
/// - Uses 1 ZMM register for broadcasting A
/// - Uses 24 ZMM registers for accumulators (3 * 8)
#[target_feature(enable = "avx512f")]
#[inline(never)]
pub unsafe fn zen5_packed_kernel_f32_avx512(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    nr: usize,
    mr: usize,
    kc: usize,
) {
    // Number of AVX-512 registers needed for one row (16 f32 per register)
    const SIMD_WIDTH: usize = 16;
    let nrs = nr / SIMD_WIDTH; // Number of SIMD registers per row

    debug_assert_eq!(nr % SIMD_WIDTH, 0, "Nr must be divisible by 16 for AVX-512");
    debug_assert_eq!(mr, 8, "Mr should be 8 for optimal performance");
    debug_assert_eq!(nrs, 3, "Expected 3 registers (48 elements) for Nr");

    // Accumulator registers: Mr x Nrs = 8 x 3 = 24 registers
    let mut acc: [[__m512; 3]; 8] = [[_mm512_setzero_ps(); 3]; 8];

    // Main computation loop over K
    let mut a_ptr = 0;
    let mut b_ptr = 0;

    for _ in 0..kc {
        // Load B registers (3 registers, 48 elements total)
        let b0 = _mm512_loadu_ps(b.as_ptr().add(b_ptr));
        let b1 = _mm512_loadu_ps(b.as_ptr().add(b_ptr + 16));
        let b2 = _mm512_loadu_ps(b.as_ptr().add(b_ptr + 32));

        // Process all Mr rows with the same B values
        // Unrolled loop for 8 rows
        let a0 = _mm512_set1_ps(a[a_ptr]);
        acc[0][0] = _mm512_fmadd_ps(a0, b0, acc[0][0]);
        acc[0][1] = _mm512_fmadd_ps(a0, b1, acc[0][1]);
        acc[0][2] = _mm512_fmadd_ps(a0, b2, acc[0][2]);

        let a1 = _mm512_set1_ps(a[a_ptr + 1]);
        acc[1][0] = _mm512_fmadd_ps(a1, b0, acc[1][0]);
        acc[1][1] = _mm512_fmadd_ps(a1, b1, acc[1][1]);
        acc[1][2] = _mm512_fmadd_ps(a1, b2, acc[1][2]);

        let a2 = _mm512_set1_ps(a[a_ptr + 2]);
        acc[2][0] = _mm512_fmadd_ps(a2, b0, acc[2][0]);
        acc[2][1] = _mm512_fmadd_ps(a2, b1, acc[2][1]);
        acc[2][2] = _mm512_fmadd_ps(a2, b2, acc[2][2]);

        let a3 = _mm512_set1_ps(a[a_ptr + 3]);
        acc[3][0] = _mm512_fmadd_ps(a3, b0, acc[3][0]);
        acc[3][1] = _mm512_fmadd_ps(a3, b1, acc[3][1]);
        acc[3][2] = _mm512_fmadd_ps(a3, b2, acc[3][2]);

        let a4 = _mm512_set1_ps(a[a_ptr + 4]);
        acc[4][0] = _mm512_fmadd_ps(a4, b0, acc[4][0]);
        acc[4][1] = _mm512_fmadd_ps(a4, b1, acc[4][1]);
        acc[4][2] = _mm512_fmadd_ps(a4, b2, acc[4][2]);

        let a5 = _mm512_set1_ps(a[a_ptr + 5]);
        acc[5][0] = _mm512_fmadd_ps(a5, b0, acc[5][0]);
        acc[5][1] = _mm512_fmadd_ps(a5, b1, acc[5][1]);
        acc[5][2] = _mm512_fmadd_ps(a5, b2, acc[5][2]);

        let a6 = _mm512_set1_ps(a[a_ptr + 6]);
        acc[6][0] = _mm512_fmadd_ps(a6, b0, acc[6][0]);
        acc[6][1] = _mm512_fmadd_ps(a6, b1, acc[6][1]);
        acc[6][2] = _mm512_fmadd_ps(a6, b2, acc[6][2]);

        let a7 = _mm512_set1_ps(a[a_ptr + 7]);
        acc[7][0] = _mm512_fmadd_ps(a7, b0, acc[7][0]);
        acc[7][1] = _mm512_fmadd_ps(a7, b1, acc[7][1]);
        acc[7][2] = _mm512_fmadd_ps(a7, b2, acc[7][2]);

        a_ptr += mr;
        b_ptr += nr;
    }

    // Store results back to C with accumulation (C += results)
    for i in 0..mr {
        let c_row_offset = i * ldc;

        // Load existing C values, add accumulators, and store back
        let c0 = _mm512_loadu_ps(c.as_ptr().add(c_row_offset));
        let c1 = _mm512_loadu_ps(c.as_ptr().add(c_row_offset + 16));
        let c2 = _mm512_loadu_ps(c.as_ptr().add(c_row_offset + 32));

        let c0_new = _mm512_add_ps(c0, acc[i][0]);
        let c1_new = _mm512_add_ps(c1, acc[i][1]);
        let c2_new = _mm512_add_ps(c2, acc[i][2]);

        _mm512_storeu_ps(c.as_mut_ptr().add(c_row_offset), c0_new);
        _mm512_storeu_ps(c.as_mut_ptr().add(c_row_offset + 16), c1_new);
        _mm512_storeu_ps(c.as_mut_ptr().add(c_row_offset + 32), c2_new);
    }
}

/// Generic wrapper that uses AVX-512 if available, falls back to scalar
#[inline]
pub fn zen5_packed_kernel_f64<const NR: usize, const MR: usize>(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    kc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                zen5_packed_kernel_f64_avx512::<NR, MR>(a, b, c, ldc, kc);
            }
            return;
        }
    }

    // Fallback scalar implementation
    zen5_packed_kernel_f64_scalar(a, b, c, ldc, NR, MR, kc);
}

/// Generic wrapper that uses AVX-512 if available, falls back to scalar
#[inline]
pub fn zen5_packed_kernel_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    nr: usize,
    mr: usize,
    kc: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                zen5_packed_kernel_f32_avx512(a, b, c, ldc, nr, mr, kc);
            }
            return;
        }
    }

    // Fallback scalar implementation
    zen5_packed_kernel_f32_scalar(a, b, c, ldc, nr, mr, kc);
}

/// Scalar fallback implementation for f64
#[inline(never)]
pub fn zen5_packed_kernel_f64_scalar(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    nr: usize,
    mr: usize,
    kc: usize,
) {
    for i in 0..mr {
        for j in 0..nr {
            let mut sum = 0.0f64;
            for k in 0..kc {
                if i * kc + k < a.len() && k * nr + j < b.len() {
                    sum += a[i * kc + k] * b[k * nr + j];
                }
            }
            if i * ldc + j < c.len() {
                c[i * ldc + j] += sum;
            }
        }
    }
}

/// Scalar fallback implementation for f32
#[inline(never)]
pub fn zen5_packed_kernel_f32_scalar(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    ldc: usize,
    nr: usize,
    mr: usize,
    kc: usize,
) {
    for i in 0..mr {
        for j in 0..nr {
            let mut sum = 0.0f32;
            for k in 0..kc {
                if i * kc + k < a.len() && k * nr + j < b.len() {
                    sum += a[i * kc + k] * b[k * nr + j];
                }
            }
            if i * ldc + j < c.len() {
                c[i * ldc + j] += sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_f64_kernel() {
        const MR: usize = 8;
        const NR: usize = 24;
        const KC: usize = 96;

        let a = vec![1.0; MR * KC];
        let b = vec![2.0; KC * NR];
        let mut c = vec![0.0; MR * NR];

        zen5_packed_kernel_f64(&a, &b, &mut c, NR, NR, MR, KC);

        // Each element should be KC * (1.0 * 2.0) = 192.0
        for &val in &c {
            assert!((val - 192.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_avx512_f32_kernel() {
        const MR: usize = 8;
        const NR: usize = 48;
        const KC: usize = 96;

        let a = vec![1.0; MR * KC];
        let b = vec![2.0; KC * NR];
        let mut c = vec![0.0; MR * NR];

        zen5_packed_kernel_f32(&a, &b, &mut c, NR, NR, MR, KC);

        // Each element should be KC * (1.0 * 2.0) = 192.0
        for &val in &c {
            assert!((val - 192.0).abs() < 1e-5);
        }
    }
}
