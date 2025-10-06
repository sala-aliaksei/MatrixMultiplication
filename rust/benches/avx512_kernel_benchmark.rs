use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use matrix_mul::zen5_avx512_kernel::*;

// Kernel configuration for f64
const MR_F64: usize = 8;
const NR_F64: usize = 24; // 3 * 8 for AVX-512
const KC_SMALL: usize = 96;
const KC_MEDIUM: usize = 96;
const KC_LARGE: usize = 96;

// Kernel configuration for f32
const MR_F32: usize = 8;
const NR_F32: usize = 48; // 3 * 16 for AVX-512

/// Benchmark f64 kernels with varying KC sizes
fn bench_f64_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("zen5_kernel_f64");

    for kc in [KC_SMALL, KC_MEDIUM, KC_LARGE] {
        // Calculate throughput: 2 * MR * NR * KC FLOPs (multiply + add)
        let flops = (2 * MR_F64 * NR_F64 * kc) as u64;
        group.throughput(Throughput::Elements(flops));

        // Prepare data
        let a = vec![1.0f64; MR_F64 * kc];
        let b = vec![2.0f64; kc * NR_F64];
        let mut c_avx512 = vec![0.0f64; MR_F64 * NR_F64];
        let mut c_scalar = vec![0.0f64; MR_F64 * NR_F64];

        // Benchmark AVX-512 implementation
        group.bench_with_input(BenchmarkId::new("avx512", kc), &kc, |bencher, _| {
            bencher.iter(|| {
                zen5_packed_kernel_f64(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_avx512),
                    NR_F64,
                    NR_F64,
                    MR_F64,
                    kc,
                )
            });
        });

        // Benchmark scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", kc), &kc, |bencher, _| {
            bencher.iter(|| {
                zen5_packed_kernel_f64_scalar(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_scalar),
                    NR_F64,
                    NR_F64,
                    MR_F64,
                    kc,
                )
            });
        });
    }

    group.finish();
}

/// Benchmark f32 kernels with varying KC sizes
fn bench_f32_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("zen5_kernel_f32");

    for kc in [KC_SMALL, KC_MEDIUM, KC_LARGE] {
        // Calculate throughput: 2 * MR * NR * KC FLOPs (multiply + add)
        let flops = (2 * MR_F32 * NR_F32 * kc) as u64;
        group.throughput(Throughput::Elements(flops));

        // Prepare data
        let a = vec![1.0f32; MR_F32 * kc];
        let b = vec![2.0f32; kc * NR_F32];
        let mut c_avx512 = vec![0.0f32; MR_F32 * NR_F32];
        let mut c_scalar = vec![0.0f32; MR_F32 * NR_F32];

        // Benchmark AVX-512 implementation
        group.bench_with_input(BenchmarkId::new("avx512", kc), &kc, |bencher, _| {
            bencher.iter(|| {
                zen5_packed_kernel_f32(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_avx512),
                    NR_F32,
                    NR_F32,
                    MR_F32,
                    kc,
                )
            });
        });

        // Benchmark scalar implementation
        group.bench_with_input(BenchmarkId::new("scalar", kc), &kc, |bencher, _| {
            bencher.iter(|| {
                zen5_packed_kernel_f32_scalar(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c_scalar),
                    NR_F32,
                    NR_F32,
                    MR_F32,
                    kc,
                )
            });
        });
    }

    group.finish();
}

/// Direct comparison: AVX-512 vs Scalar for f64 with standard KC
fn bench_f64_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("zen5_f64_comparison");

    const KC: usize = KC_MEDIUM;
    let flops = (2 * MR_F64 * NR_F64 * KC) as u64;
    group.throughput(Throughput::Elements(flops));

    let a = vec![1.0f64; MR_F64 * KC];
    let b = vec![2.0f64; KC * NR_F64];
    let mut c = vec![0.0f64; MR_F64 * NR_F64];

    group.bench_function("avx512_direct", |bencher| {
        bencher.iter(|| {
            zen5_packed_kernel_f64(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                NR_F64,
                NR_F64,
                MR_F64,
                KC,
            )
        });
    });

    group.bench_function("scalar_direct", |bencher| {
        bencher.iter(|| {
            zen5_packed_kernel_f64_scalar(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                NR_F64,
                NR_F64,
                MR_F64,
                KC,
            )
        });
    });

    group.finish();
}

/// Direct comparison: AVX-512 vs Scalar for f32 with standard KC
fn bench_f32_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("zen5_f32_comparison");

    const KC: usize = KC_MEDIUM;
    let flops = (2 * MR_F32 * NR_F32 * KC) as u64;
    group.throughput(Throughput::Elements(flops));

    let a = vec![1.0f32; MR_F32 * KC];
    let b = vec![2.0f32; KC * NR_F32];
    let mut c = vec![0.0f32; MR_F32 * NR_F32];

    group.bench_function("avx512_direct", |bencher| {
        bencher.iter(|| {
            zen5_packed_kernel_f32(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                NR_F32,
                NR_F32,
                MR_F32,
                KC,
            )
        });
    });

    group.bench_function("scalar_direct", |bencher| {
        bencher.iter(|| {
            zen5_packed_kernel_f32_scalar(
                black_box(&a),
                black_box(&b),
                black_box(&mut c),
                NR_F32,
                NR_F32,
                MR_F32,
                KC,
            )
        });
    });

    group.finish();
}

/// Benchmark only the AVX-512 intrinsic functions (when available)
#[cfg(target_arch = "x86_64")]
fn bench_avx512_intrinsics(c: &mut Criterion) {
    if !is_x86_feature_detected!("avx512f") {
        eprintln!("Warning: AVX-512 not available, skipping intrinsic benchmarks");
        return;
    }

    let mut group = c.benchmark_group("zen5_avx512_intrinsics");

    const KC: usize = KC_MEDIUM;

    // F64 intrinsics
    let a_f64 = vec![1.0f64; MR_F64 * KC];
    let b_f64 = vec![2.0f64; KC * NR_F64];
    let mut c_f64 = vec![0.0f64; MR_F64 * NR_F64];

    let flops_f64 = (2 * MR_F64 * NR_F64 * KC) as u64;
    group.throughput(Throughput::Elements(flops_f64));

    group.bench_function("f64_intrinsics", |bencher| {
        bencher.iter(|| unsafe {
            zen5_packed_kernel_f64_avx512(
                black_box(&a_f64),
                black_box(&b_f64),
                black_box(&mut c_f64),
                NR_F64,
                NR_F64,
                MR_F64,
                KC,
            )
        });
    });

    // F32 intrinsics
    let a_f32 = vec![1.0f32; MR_F32 * KC];
    let b_f32 = vec![2.0f32; KC * NR_F32];
    let mut c_f32 = vec![0.0f32; MR_F32 * NR_F32];

    let flops_f32 = (2 * MR_F32 * NR_F32 * KC) as u64;
    group.throughput(Throughput::Elements(flops_f32));

    group.bench_function("f32_intrinsics", |bencher| {
        bencher.iter(|| unsafe {
            zen5_packed_kernel_f32_avx512(
                black_box(&a_f32),
                black_box(&b_f32),
                black_box(&mut c_f32),
                NR_F32,
                NR_F32,
                MR_F32,
                KC,
            )
        });
    });

    group.finish();
}

#[cfg(not(target_arch = "x86_64"))]
fn bench_avx512_intrinsics(_c: &mut Criterion) {
    eprintln!("Skipping AVX-512 intrinsic benchmarks on non-x86_64 platform");
}

/// Benchmark memory bandwidth impact with different KC sizes
fn bench_memory_bandwidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("zen5_memory_bandwidth");

    // Test with increasingly large KC to show cache effects
    for kc in [96, 192, 384, 768, 1536] {
        let a = vec![1.0f64; MR_F64 * kc];
        let b = vec![2.0f64; kc * NR_F64];
        let mut c = vec![0.0f64; MR_F64 * NR_F64];

        // Calculate memory traffic (bytes)
        let bytes = ((MR_F64 * kc + kc * NR_F64 + MR_F64 * NR_F64) * 8) as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::from_parameter(kc), &kc, |bencher, &kc| {
            bencher.iter(|| {
                zen5_packed_kernel_f64(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut c),
                    NR_F64,
                    NR_F64,
                    MR_F64,
                    kc,
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_f64_kernels,
    bench_f32_kernels,
    bench_f64_comparison,
    bench_f32_comparison,
    bench_avx512_intrinsics,
    bench_memory_bandwidth,
);
criterion_main!(benches);
