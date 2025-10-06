use criterion::{criterion_group, criterion_main, Criterion};
use matrix_mul::matmul::mat_mul_zen5_mt_blocking_f64;
use matrix_mul::matmul::Matrix;

fn criterion_benchmark(cr: &mut Criterion) {
    let a = Matrix::new(3072, 3072);
    let b = Matrix::new(3072, 3072);
    let mut c = Matrix::new(3072, 3072);

    cr.bench_function("matmul", |bencher| {
        bencher.iter(|| mat_mul_zen5_mt_blocking_f64(&a, &b, &mut c))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
