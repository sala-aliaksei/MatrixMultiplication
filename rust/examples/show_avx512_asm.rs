// Example to view AVX-512 assembly for zen5_packed_kernel
// Compile with: cargo rustc --example show_avx512_asm --release -- --emit asm -C "llvm-args=-x86-asm-syntax=intel" -C target-feature=+avx512f
// View assembly: grep -A200 "zen5_packed_kernel_f64_avx512" target/release/examples/show_avx512_asm-*.s

use matrix_mul::zen5_avx512_kernel::{zen5_packed_kernel_f32, zen5_packed_kernel_f64};

fn main() {
    // Test f64 kernel
    const MR: usize = 8;
    const NR_F64: usize = 24; // 3 * 8 for AVX-512
    const NR_F32: usize = 48; // 3 * 16 for AVX-512
    const KC: usize = 96;

    // F64 test
    let a_f64 = vec![1.0f64; MR * KC];
    let b_f64 = vec![2.0f64; KC * NR_F64];
    let mut c_f64 = vec![0.0f64; MR * NR_F64];

    zen5_packed_kernel_f64(&a_f64, &b_f64, &mut c_f64, NR_F64, NR_F64, MR, KC);

    println!("F64 result: {:.2}", c_f64.iter().sum::<f64>());

    // F32 test
    let a_f32 = vec![1.0f32; MR * KC];
    let b_f32 = vec![2.0f32; KC * NR_F32];
    let mut c_f32 = vec![0.0f32; MR * NR_F32];

    zen5_packed_kernel_f32(&a_f32, &b_f32, &mut c_f32, NR_F32, NR_F32, MR, KC);

    println!("F32 result: {:.2}", c_f32.iter().sum::<f32>());
}
