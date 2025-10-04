
use std::sync::Arc;
use std::thread;
use std::cmp::min;

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
    const KC: usize = 96 * 2;
}

// Configuration for f32 (float)
pub struct ConfigF32;
impl MatMulConfig for ConfigF32 {
    const NC: usize = 96 * 2;
    const MC: usize = 96;
    const KC: usize = 96 * 2 * 2;
}

// Matrix struct similar to C++ Matrix<T>
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default + std::cmp::PartialEq> Matrix<T> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Matrix { data, rows, cols }
    }

    #[inline]
    pub fn row(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn col(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.data[row * self.cols + col]
    }

    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        &mut self.data[row * self.cols + col]
    }

    pub fn eq(&self, other: &Self) -> bool {
        // vector comparison 
        if self.data.len() != other.data.len() {
            return false;
        }
        for i in 0..self.data.len() {
            if self.data[i] != other.data[i] {
                return false;
            }
        }
        true
    }
}

// Helper function to map thread ID to core ID (from C++ map_thread_id_to_core_id)
#[inline]
fn map_thread_id_to_core_id(n: usize) -> usize {
    if (n & 1) == 0 {
        n >> 1  // n / 2 for even
    } else {
        16 + (n >> 1)  // 16 + (n - 1) / 2 for odd
    }
}

// Platform-specific CPU affinity setting
#[cfg(target_os = "linux")]
fn set_thread_affinity(core_id: usize) {
    use libc::{cpu_set_t, CPU_SET, CPU_ZERO, pthread_self, pthread_setaffinity_np};
    
    unsafe {
        let mut cpuset: cpu_set_t = std::mem::zeroed();
        CPU_ZERO(&mut cpuset);
        CPU_SET(core_id, &mut cpuset);
        pthread_setaffinity_np(
            pthread_self(),
            std::mem::size_of::<cpu_set_t>(),
            &cpuset as *const _,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn set_thread_affinity(_core_id: usize) {
    // No-op on non-Linux platforms
}

// Placeholder for reordering functions (would need actual implementations)
fn reorder_row_major_matrix_avx<T: Clone>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    kc: usize,
    nc: usize,
    _kr: usize,
    _nr: usize,
) {
    // Simplified placeholder - actual implementation would do AVX-optimized reordering
    for k in 0..kc {
        for n in 0..nc {
            if k < src.len() / src_stride && n < src_stride {
                dst[k * nc + n] = src[k * src_stride + n].clone();
            }
        }
    }
}

fn reorder_col_order_matrix<T: Clone>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    mc: usize,
    kc: usize,
    _mr: usize,
    _kr: usize,
) {
    // Simplified placeholder - actual implementation would do optimized reordering
    for m in 0..mc {
        for k in 0..kc {
            if m < src.len() / src_stride && k < src_stride {
                dst[m * kc + k] = src[m * src_stride + k].clone();
            }
        }
    }
}

// Placeholder for the kernel function
fn zen5_packed_kernel<T: Clone + Default + std::ops::AddAssign + std::ops::Mul<Output = T>>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    ldc: usize,
    nr: usize,
    mr: usize,
    kc: usize,
) where T: Copy {
    for i in 0..mr {
        for j in 0..nr {
            let mut sum = T::default();
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

// Main matrix multiplication function
pub fn mat_mul_zen5_mt_blocking<T, C>(a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>)
where
    T: Clone + Default + Send + Sync + std::ops::AddAssign + std::ops::Mul<Output = T> + Copy + 'static + std::cmp::PartialEq,
    C: MatMulConfig,
{
    // Get configuration constants
    let nc = C::NC;
    let mc = C::MC;
    let kc = C::KC;
    
    // Register configuration constants
    //const NUM_OF_REGS: usize = 32;
    const BREGS_CNT: usize = 3;
    //const AREGS_CNT: usize = 1;
    
    // For simplicity, assuming 8 elements per register for f64, 16 for f32
    let num_of_elems_in_reg = if std::mem::size_of::<T>() == 8 { 8 } else { 16 };
    
    let nr = BREGS_CNT * num_of_elems_in_reg;
    const MR: usize = 8;
    const KR: usize = 1;
    
    // Matrix dimensions
    let n = b.col();
    let k = a.col();
    let m = a.row();
    
    // Assertions (in debug mode)
    debug_assert_eq!(n % nc, 0, "N % Nc must be 0");
    debug_assert_eq!(k % kc, 0, "K % Kc must be 0");
    debug_assert_eq!(m % mc, 0, "M % Mc must be 0");
    debug_assert_eq!(mc % MR, 0, "Mc % Mr must be 0");
    debug_assert_eq!(nc % nr, 0, "Nc % Nr must be 0");
    debug_assert_eq!(kc % KR, 0, "Kc % Kr must be 0");
    
    // Fixed thread grid 4x8 → 32 threads
    const GRID_I: usize = 4;
    const GRID_J: usize = 8;
    const NUM_THREADS: usize = GRID_I * GRID_J;
    
    // Allocate buffer for all threads
    // let buffer_size = NUM_THREADS * kc * (mc + nc);
    // let buffer: Vec<T> = vec![T::default(); buffer_size];
    //let buffer = Arc::new(buffer);
    
    // Square-chunking in block units

    let total_iblocks = m / mc;
    let total_jblocks = n / nc;

    let iblocks_per_thread = total_iblocks / GRID_I;
    let jblocks_per_thread = total_jblocks / GRID_J;
    
    // Wrap matrices in Arc for thread sharing
    let a_arc = Arc::new(a.data().to_vec());
    let b_arc = Arc::new(b.data().to_vec());
    let c_arc = Arc::new(std::sync::Mutex::new(c.data_mut().to_vec()));
    
    // Create thread pool
    let mut handles = vec![];
    
    for t in 0..NUM_THREADS {
        
        let a_clone = Arc::clone(&a_arc);
        let b_clone = Arc::clone(&b_arc);
        let c_clone = Arc::clone(&c_arc);
        
        let handle = thread::spawn(move || {
            // Set CPU affinity
            let core_id = map_thread_id_to_core_id(t);
            set_thread_affinity(core_id);
            

            
            // Thread's grid coordinates
            let ti = t / GRID_J;
            let tj = t % GRID_J;
            
            // Process chunks assigned to this thread
            for chi in (ti..iblocks_per_thread).step_by(GRID_I) {
                for chj in (tj..jblocks_per_thread).step_by(GRID_J) {
                    let ibegin = chi;
                    let iend = min(ibegin, total_iblocks);
                    let jbegin = chj;
                    let jend = min(jbegin, total_jblocks);
                    
                    for k_block in (0..k).step_by(kc) {
                        // Process each j_block in the chunk
                        for jb in jbegin..jend {
                            let j_block = jb * nc;
                            
                            // Create local mutable buffers for this iteration
                            let mut buf_b = vec![T::default(); nc * kc];
                            
                            // Reorder B matrix block
                            reorder_row_major_matrix_avx(
                                &b_clone[n * k_block + j_block..],
                                n,
                                &mut buf_b,
                                kc,
                                nc,
                                KR,
                                nr,
                            );
                            
                            for ib in ibegin..iend {
                                let i_block = ib * mc;
                                
                                // Create local mutable buffer for A
                                let mut buf_a = vec![T::default(); mc * kc];
                                
                                // Reorder A matrix block
                                reorder_col_order_matrix(
                                    &a_clone[k * i_block + k_block..],
                                    k,
                                    &mut buf_a,
                                    mc,
                                    kc,
                                    MR,
                                    KR,
                                );
                                
                                // Process the micro-tiles
                                for j in (0..nc).step_by(nr) {
                                    let bc1 = &buf_b[kc * j..];
                                    
                                    for i in (0..mc).step_by(MR) {
                                        // Calculate C position
                                        let c_offset = n * i_block + j + n * i + j_block;
                                        let ac0 = &buf_a[kc * i..];
                                        
                                        // Create a temporary buffer for the kernel output
                                        let mut c_temp = vec![T::default(); MR * nr];
                                        
                                        // Call the kernel
                                        zen5_packed_kernel(
                                            ac0,
                                            bc1,
                                            &mut c_temp,
                                            n,
                                            nr,
                                            MR,
                                            kc,
                                        );
                                        
                                        // Add results to C matrix (with lock)
                                        {
                                            let mut c_guard = c_clone.lock().unwrap();
                                            for di in 0..MR {
                                                for dj in 0..nr {
                                                    let idx = c_offset + di * n + dj;
                                                    if idx < c_guard.len() {
                                                        c_guard[idx] += c_temp[di * nr + dj];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Copy results back to C matrix
    let c_guard = c_arc.lock().unwrap();
    c.data_mut().copy_from_slice(&c_guard);
}

// Convenience functions for specific types
pub fn mat_mul_zen5_mt_blocking_f64(a: &Matrix<f64>, b: &Matrix<f64>, c: &mut Matrix<f64>) {
    mat_mul_zen5_mt_blocking::<f64, ConfigF64>(a, b, c);
}

pub fn mat_mul_zen5_mt_blocking_f32(a: &Matrix<f32>, b: &Matrix<f32>, c: &mut Matrix<f32>) {
    mat_mul_zen5_mt_blocking::<f32, ConfigF32>(a, b, c);
}

/// Safe reference implementation of matrix multiplication (C = A * B)
/// A is an m x k matrix, B is a k x n matrix, and C is an m x n matrix.
pub fn golden_matmul<T>(a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>)
where
    T: Clone + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + Copy + std::cmp::PartialEq,
{
    let m = a.row();
    let k = a.col();
    let n = b.col();

    assert_eq!(b.row(), k, "Matrix dimensions incompatible for multiplication");
    assert_eq!(c.row(), m, "Output matrix C has incorrect number of rows");
    assert_eq!(c.col(), n, "Output matrix C has incorrect number of columns");

    // Initialize C to zero
    for i in 0..m {
        for j in 0..n {
            *c.get_mut(i, j) = T::default();
        }
    }

    // Standard triple-nested loop matrix multiplication
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                *c.get_mut(i, j) += *a.get(i, p) * *b.get(p, j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m: Matrix<f64> = Matrix::new(4, 4);
        assert_eq!(m.row(), 4);
        assert_eq!(m.col(), 4);
        assert_eq!(m.data().len(), 16);
    }

    #[test]
    fn test_map_thread_id_to_core_id() {
        assert_eq!(map_thread_id_to_core_id(0), 0);
        assert_eq!(map_thread_id_to_core_id(1), 16);
        assert_eq!(map_thread_id_to_core_id(2), 1);
        assert_eq!(map_thread_id_to_core_id(3), 17);
    }

    #[test]
    fn test_golden_matmul() {
        let m = 3072;
        let n = m;
        let k = m;
        let a = Matrix::new(m, k);
        let b = Matrix::new(k, n);
        let mut c = Matrix::new(m, n);
        let mut c_golden = Matrix::new(m, n);
        mat_mul_zen5_mt_blocking_f64(&a, &b, &mut c);
        golden_matmul(&a, &b, &mut c_golden);
        assert!(c.eq(&c_golden), "Matrices c and c_golden are not equal!");
    }
}
