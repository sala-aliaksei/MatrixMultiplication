use rayon::prelude::*;
use std::arch::x86_64::*;

/// Multiplies matrices A and B, storing the result in C.
/// A is an m x k matrix, B is a k x n matrix, and C is an m x n matrix.
pub fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    assert!(a.len() == m * k, "Matrix A dimensions do not match.");
    assert!(b.len() == k * n, "Matrix B dimensions do not match.");
    assert!(c.len() == m * n, "Matrix C dimensions do not match.");

    // Initialize C to zero
    c.par_iter_mut().for_each(|elem| *elem = 0.0);

    // Parallelize over rows of C
    c.par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, c_row)| {
            for p in 0..k {
                let a_ip = a[i * k + p];
                let a_vec = unsafe { _mm256_set1_pd(a_ip) };

                let b_row = &b[p * n..(p + 1) * n];

                let mut j = 0;
                // Process 4 elements at a time using AVX2 for f64
                while j + 4 <= n {
                    unsafe {
                        // Prefetch next cache lines of B and C
                        _mm_prefetch(
                            b_row[j..].as_ptr() as *const i8,
                            _MM_HINT_T0,
                        );
                        _mm_prefetch(
                            c_row[j..].as_ptr() as *const i8,
                            _MM_HINT_T0,
                        );

                        let b_vec = _mm256_loadu_pd(b_row[j..].as_ptr());
                        let c_vec = _mm256_loadu_pd(c_row[j..].as_ptr());
                        let result = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
                        _mm256_storeu_pd(c_row[j..].as_mut_ptr(), result);
                    }
                    j += 4;
                }
                // Handle remaining elements
                while j < n {
                    c_row[j] += a_ip * b_row[j];
                    j += 1;
                }
            }
        });
}

fn main() {
    let N = 256* 8 * 4 * 2;
    let m = N; // Rows in A and C
    let k = N; // Columns in A and rows in B
    let n = N; // Columns in B and C

    let a = vec![1.0_f64; m * k]; // Initialize matrix A with double-precision floats
    let b = vec![1.0_f64; k * n]; // Initialize matrix B
    let mut c = vec![0.0_f64; m * n]; // Initialize matrix C

    matmul(&a, &b, &mut c, m, k, n);

    // c now contains the result of A * B
    // Optionally, print or verify the result
}