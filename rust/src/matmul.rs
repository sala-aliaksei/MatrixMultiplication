use crate::wide_kernel::zen5_packed_kernel_f64_wide;
use crate::zen5_avx512_kernel::{zen5_packed_kernel_f32, zen5_packed_kernel_f64};

// Constants for different types - similar to MatMulZen5Config<T>
pub trait MatMulConfig {
    const NC: usize;
    const MC: usize;
    const KC: usize;
}

// Configuration for f64 (double)
pub struct ConfigF64;
impl MatMulConfig for ConfigF64 {
    const NC: usize = 96;
    const MC: usize = 96;
    const KC: usize = 96 * 2 * 2;
}

// pub struct ConfigF64;
// impl MatMulConfig for ConfigF64 {
//     const NC: usize = 4;
//     const MC: usize = 4;
//     const KC: usize = 4;
// }

// Configuration for f32 (float)
pub struct ConfigF32;
impl MatMulConfig for ConfigF32 {
    const NC: usize = 96;
    const MC: usize = 96;
    const KC: usize = 96 * 2 * 2;
}

// Matrix struct similar to C++ Matrix<T>
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T: Default + Clone> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }
}

// Placeholder for the kernel function
fn zen5_packed_kernel<const MR: usize, const NR: usize>(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    ldc: usize,
    kc: usize,
) {
    // println!("a: {:?}", a);
    // println!("b: {:?}", b);
    // println!("c: {:?}", c);
    // println!("ldc: {:?}", ldc);
    // println!("nr: {:?}", nr);
    // println!("mr: {:?}", mr);
    // println!("kc: {:?}", kc);

    for k in 0..kc {
        for i in 0..MR {
            for j in 0..NR {
                c[i * ldc + j] += a[k * MR + i] * b[k * NR + j];
            }
        }
    }
}

// Placeholder for reordering functions (would need actual implementations)
fn reorder_matrix_by_row<T: Clone>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    kc: usize,
    nc: usize,
    kr: usize,
    nr: usize,
) {
    let mut idx = 0;

    // DON'T REORDER LOOPS
    // Process columns in groups of 4
    for j in (0..nc).step_by(nr) {
        for i in (0..kc).step_by(kr) {
            for ic in 0..kr {
                for jc in 0..nr {
                    dst[idx] = src[(i + ic) * src_stride + j + jc].clone();
                    idx += 1;
                }
            }
        }
    }
}

fn reorder_matrix_by_col<T: Clone>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    mc: usize,
    kc: usize,
    mr: usize,
    kr: usize,
) {
    let mut idx = 0;

    for i in (0..mc).step_by(mr) {
        for j in (0..kc).step_by(kr) {
            for jc in 0..kr {
                for ic in 0..mr {
                    let col = (i + ic) * src_stride;
                    dst[idx] = src[col + j + jc].clone();
                    idx += 1;
                }
            }
        }
    }
}

fn split_into_chunks_mut<T>(slice: &mut [T], chunk_size: usize) -> Vec<&mut [T]> {
    assert!(chunk_size > 0);
    let mut chunks = Vec::new();
    let mut rest = slice;

    while !rest.is_empty() {
        let len = rest.len().min(chunk_size);
        let (head, tail) = rest.split_at_mut(len);
        chunks.push(head);
        rest = tail;
    }

    chunks
}

// Main matrix multiplication function
pub fn mat_mul_zen5_mt_blocking<C>(a: &Matrix<f64>, b: &Matrix<f64>, c: &mut Matrix<f64>)
where
    C: MatMulConfig,
{
    // Register configuration constants
    //const NUM_OF_REGS: usize = 32;
    //const BREGS_CNT: usize = 3;
    //const AREGS_CNT: usize = 1;

    // For simplicity, assuming 8 elements per register for f64, 16 for f32
    //let num_of_elems_in_reg: usize = if std::mem::size_of::<T>() == 8 { 8 } else { 16 };
    // let nr = BREGS_CNT * num_of_elems_in_reg;

    // Get configuration constants
    // let nc = C::NC;
    // let mc = C::MC;
    // let kc = C::KC;

    const NC: usize = 96;
    const MC: usize = 96;
    const KC: usize = 96 * 2;
    const MR: usize = 8;
    const KR: usize = 1;
    const NR: usize = 24;
    // const MR: usize = 2;
    // const KR: usize = 1;
    // const NR: usize = 2;

    // Matrix dimensions
    let n = b.cols;
    let k = a.cols;
    let m = a.rows;

    // Assertions (in debug mode)
    debug_assert_eq!(n % NC, 0, "N % Nc must be 0");
    debug_assert_eq!(k % KC, 0, "K % Kc must be 0");
    debug_assert_eq!(m % MC, 0, "M % Mc must be 0");
    debug_assert_eq!(MC % MR, 0, "Mc % Mr must be 0");
    debug_assert_eq!(NC % NR, 0, "Nc % Nr must be 0");
    debug_assert_eq!(KC % KR, 0, "Kc % Kr must be 0");

    // Fixed thread grid 4x8 → 32 threads
    const GRID_I: usize = 4;
    const GRID_J: usize = 4;
    const NUM_THREADS: usize = GRID_I * GRID_J;

    // Square-chunking in block units
    let total_iblocks = m / MC;
    let total_jblocks = n / NC;

    let iblocks_per_thread = total_iblocks / GRID_I;
    let jblocks_per_thread = total_jblocks / GRID_J;

    let c_ptr = c.data.as_mut_ptr() as usize;
    crossbeam::scope(|s| {
        for t in 0..NUM_THREADS {
            s.spawn(move |_| {
                let ti = t / GRID_J;
                let tj = t % GRID_J;

                let ibegin = ti * iblocks_per_thread * MC;
                let iend = ibegin + iblocks_per_thread * MC;
                let jbegin = tj * jblocks_per_thread * NC;
                let jend = jbegin + jblocks_per_thread * NC;
                for j_block in (jbegin..jend).step_by(NC) {
                    for k_block in (0..k).step_by(KC) {
                        let mut buf_b = [f64::default(); NC * KC];
                        reorder_matrix_by_row(
                            &b.data[n * k_block + j_block..],
                            n,
                            &mut buf_b,
                            KC,
                            NC,
                            KR,
                            NR,
                        );
                        for i_block in (ibegin..iend).step_by(MC) {
                            let mut buf_a = [f64::default(); MC * KC];

                            reorder_matrix_by_col(
                                &a.data[k * i_block + k_block..],
                                k,
                                &mut buf_a,
                                MC,
                                KC,
                                MR,
                                KR,
                            );

                            for j in (0..NC).step_by(NR) {
                                for i in (0..MC).step_by(MR) {
                                    let ac0 = &buf_a[KC * i..];
                                    let mut c_temp = [f64::default(); MR * NR];
                                    let bc1 = &buf_b[KC * j..];

                                    //zen5_packed_kernel::<MR, NR>(ac0, bc1, &mut c_temp, NR, kc);
                                    //zen5_packed_kernel_f64::<NR, MR>(ac0, bc1, &mut c_temp, NR, KC);
                                    zen5_packed_kernel_f64_wide::<NR, MR>(
                                        ac0,
                                        bc1,
                                        &mut c_temp,
                                        NR,
                                        KC,
                                    );

                                    for di in 0..MR {
                                        for dj in 0..NR {
                                            let c_offset =
                                                (i_block + i + di) * n + (j_block + j + dj);

                                            unsafe {
                                                let c_ptr_mut = (c_ptr as *mut f64).add(c_offset);
                                                std::ptr::write(
                                                    c_ptr_mut,
                                                    c_temp[di * NR + dj]
                                                        + std::ptr::read(c_ptr_mut),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        }
    })
    .unwrap();
}

// --------

/// Safe reference implementation of matrix multiplication (C = A * B)
/// A is an m x k matrix, B is a k x n matrix, and C is an m x n matrix.
pub fn golden_matmul<T>(a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>)
where
    T: Clone
        + Default
        + std::ops::AddAssign
        + std::ops::Mul<Output = T>
        + Copy
        + std::cmp::PartialEq,
{
    let m = a.rows;
    let k = a.cols;
    let n = b.cols;

    assert_eq!(
        b.rows, k,
        "Matrix dimensions incompatible for multiplication"
    );
    assert_eq!(c.rows, m, "Output matrix C has incorrect number of rows");
    assert_eq!(c.cols, n, "Output matrix C has incorrect number of columns");

    // Initialize C to zero
    for i in 0..m {
        for j in 0..n {
            c.data[i * n + j] = T::default();
        }
    }

    // Standard triple-nested loop matrix multiplication
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c.data[i * n + j] += a.data[i * k + p] * b.data[p * n + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m: Matrix<f64> = Matrix {
            data: vec![0.0; 16],
            rows: 4,
            cols: 4,
        };
        assert_eq!(m.rows, 4);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 16);
    }

    #[test]
    fn test_golden_matmul() {
        let m = 3072;
        let n = m;
        let k = m;
        let a = Matrix {
            data: vec![0.0; m * k],
            rows: m,
            cols: k,
        };
        let b = Matrix {
            data: vec![0.0; k * n],
            rows: k,
            cols: n,
        };
        let mut c = Matrix {
            data: vec![0.0; m * n],
            rows: m,
            cols: n,
        };
        let mut c_golden = Matrix {
            data: vec![0.0; m * n],
            rows: m,
            cols: n,
        };
        mat_mul_zen5_mt_blocking(&a, &b, &mut c);
        golden_matmul(&a, &b, &mut c_golden);
        assert!(
            c.data == c_golden.data,
            "Matrices c and c_golden are not equal!"
        );
    }
}

// let chunks = split_into_chunks_mut(&mut c.data, nc);

// let mut chunks_per_thread: Vec<Vec<&mut [T]>> = Vec::with_capacity(NUM_THREADS);
// for (row_idx, row) in chunks.into_iter().enumerate() {
//     let get_idx = || {
//         // row_idx -> thread_idx
//         (row_idx / jblocks_per_thread)
//     };

//     chunks_per_thread[row_idx % NUM_THREADS].push(row);
// }
