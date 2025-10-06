use wide::f64x8;

/// Portable SIMD kernel using the `wide` crate (f64)
///
/// Arguments mirror the AVX-512 kernel layout:
/// - `a`: Packed A micro-panel (Mr x Kc), column-major within micro-tiles
/// - `b`: Packed B micro-panel (Kc x Nr), row-major within micro-tiles
/// - `c`: Output C tile (Mr x Nr), accumulates results (row-major with leading dimension `ldc`)
/// - `ldc`: Leading dimension of C (typically N)
/// - `kc`: K dimension of the micro-tile
///
/// Notes:
/// - Uses `wide::f64x8` as the vector type. `NR` must be divisible by its lane count.
/// - Assumes `MR == 8` for optimal blocking (matching the AVX-512 kernel), but works for any `MR` >= 1.
#[inline(always)]
pub fn zen5_packed_kernel_f64_wide<const NR: usize, const MR: usize>(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    kc: usize,
) {
    type V = f64x8;
    // Derive lane count by size to avoid depending on specific API versions
    let lanes: usize = std::mem::size_of::<V>() / std::mem::size_of::<f64>();

    // Number of vector chunks per row of C
    let nrs: usize = NR / lanes;

    // Accumulators: MR rows x NRS vector columns
    let mut acc: Vec<V> = vec![V::splat(0.0); MR * nrs];

    // Main K loop
    let mut a_ptr: usize = 0;
    let mut b_ptr: usize = 0;
    // Exactly 3 B registers (24 columns total for f64x8)
    let mut b_regs: [V; 3] = [V::splat(0.0); 3];
    for _ in 0..kc {
        // Load B registers (safe slice copies)
        for j in 0..3 {
            let start = b_ptr + j * lanes;
            let mut tmp = [0.0f64; 8];
            tmp.copy_from_slice(&b[start..start + lanes]);
            b_regs[j] = V::from(tmp);
        }

        // Accumulate for all MR rows using 3 B regs
        for i in 0..MR {
            let a_val = a[a_ptr + i];
            let a_broadcast = V::splat(a_val);
            let base = i * nrs;
            acc[base + 0] = b_regs[0] * a_broadcast + acc[base + 0];
            acc[base + 1] = b_regs[1] * a_broadcast + acc[base + 1];
            acc[base + 2] = b_regs[2] * a_broadcast + acc[base + 2];
        }

        a_ptr += MR;
        b_ptr += NR;
    }

    // Store back to C with accumulation: C += acc
    for i in 0..MR {
        let c_row_offset = i * ldc;
        let base = i * nrs;
        // Exactly 3 vector segments per row
        let mut c0 = [0.0f64; 8];
        let mut c1 = [0.0f64; 8];
        let mut c2 = [0.0f64; 8];
        c0.copy_from_slice(&c[c_row_offset + 0 * lanes..c_row_offset + 1 * lanes]);
        c1.copy_from_slice(&c[c_row_offset + 1 * lanes..c_row_offset + 2 * lanes]);
        c2.copy_from_slice(&c[c_row_offset + 2 * lanes..c_row_offset + 3 * lanes]);

        let c0v = V::from(c0);
        let c1v = V::from(c1);
        let c2v = V::from(c2);
        let s0: [f64; 8] = (c0v + acc[base + 0]).into();
        let s1: [f64; 8] = (c1v + acc[base + 1]).into();
        let s2: [f64; 8] = (c2v + acc[base + 2]).into();
        c[c_row_offset + 0 * lanes..c_row_offset + 1 * lanes].copy_from_slice(&s0);
        c[c_row_offset + 1 * lanes..c_row_offset + 2 * lanes].copy_from_slice(&s1);
        c[c_row_offset + 2 * lanes..c_row_offset + 3 * lanes].copy_from_slice(&s2);
    }
}
