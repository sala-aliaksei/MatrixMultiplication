use rayon::prelude::*;
use std::arch::x86_64::*;
use matrix_mul::matmul::mat_mul_zen5_mt_blocking_f64;
use matrix_mul::matmul::Matrix;
use matrix_mul::matmul::golden_matmul;



/// Multiplies matrices A and B, storing the result in C.
/// A is an m x k matrix, B is a k x n matrix, and C is an m x n matrix.
pub fn matmul_impl(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
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
    let n = 3072;// Columns in B and C
    let m = n; // Rows in A and C
    let k = n; // Columns in A and rows in B


    let a = Matrix::new(m, k);
    let b = Matrix::new(k, n);
    let mut c = Matrix::new(m, n);
    let mut c_golden = Matrix::new(m, n);

    // print execution time
    let start = std::time::Instant::now();
    golden_matmul(&a, &b, &mut c_golden);
    let end = std::time::Instant::now();
    println!("Execution time: {:?}", end - start);

    let start = std::time::Instant::now();
    mat_mul_zen5_mt_blocking_f64(&a, &b, &mut c);
    let end = std::time::Instant::now();
    println!("Execution time: {:?}", end - start);


    assert!(c.eq(&c_golden), "Matrices c and c_golden are not equal!");
    println!("c and c_golden are equal");
}