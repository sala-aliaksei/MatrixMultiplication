use matrix_mul::matmul::{golden_matmul, mat_mul_zen5_mt_blocking, ConfigF64, Matrix};

#[target_feature(enable = "avx512f")]
unsafe fn foo() {}

fn main() {
    unsafe {
        foo();
    }

    let n = 3072; // Columns in B and C
    let m = n; // Rows in A and C
    let k = n; // Columns in A and rows in B

    let mut a = Matrix::new(m, k);
    let mut b = Matrix::new(k, n);
    let mut c = Matrix::new(m, n);
    let mut c_golden = Matrix::new(m, n);

    for i in 0..m {
        for j in 0..n {
            a.data[i * k + j] = rand::random();
            b.data[k * j + i] = rand::random();
            c.data[i * n + j] = 0.0;
        }
    }

    // print execution time
    let start = std::time::Instant::now();
    golden_matmul(&a, &b, &mut c_golden);
    let end = std::time::Instant::now();
    println!("Execution time: {:?}", end - start);

    let start = std::time::Instant::now();
    mat_mul_zen5_mt_blocking::<ConfigF64>(&a, &b, &mut c);
    let end = std::time::Instant::now();
    println!("Execution time: {:?}", end - start);

    // println!("c: {:?}", c.data);
    // println!("c_golden: {:?}", c_golden.data);

    // compare c and c_golden with a tolerance of 1e-3
    for i in 0..n {
        for j in 0..n {
            assert!((c.data[i * n + j] - c_golden.data[i * n + j]).abs() < 1e-3);
        }
    }
    println!("c and c_golden are equal");
}

// print a and b as 2d arrays of  nxn dimensions
// println!("a: ");
// for i in 0..n {
//     for j in 0..n {
//         print!("{} ", a.data[i * n + j]);
//     }
//     println!();
// }
// println!("\nb: ");
// for i in 0..n {
//     for j in 0..n {
//         print!("{} ", b.data[i * n + j]);
//     }
//     println!();
// }
