
---

# MatrixMultiplication Benchmarking Project

## Overview

This project is designed to explore and benchmark various matrix multiplication algorithms in C++. The goal is to implement custom matrix multiplication functions, optimize them using different techniques, and compare their performance against industry-standard libraries such as Intel MKL, Eigen, and OpenBLAS.



## Features

- **Custom Implementations**: Basic matrix multiplication using nested loops.
- **Optimized Implementations**: Techniques like loop unrolling, cache optimization, SIMD (Single Instruction, Multiple Data) instructions, and multi-threading.
- **Benchmarking**: Performance comparison with industry-standard libraries such as Intel MKL, Eigen, and OpenBLAS.
- **Unit Tests**: Ensure correctness of implementations.

## Getting Started

### Prerequisites

- C++20 or higher
- CMake 3.5 or higher
- [GoogleTest](https://github.com/google/googletest) (for unit tests)
- [GoogleBenchmark](https://github.com/google/benchmark)
- [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS)
- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html) (Optional)
- [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html) (Optional)


### Build Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MatrixMultiplication.git
   cd MatrixMultiplication
   ```

2. Create a build directory, run CMake and build the project:

   ```bash
   ./cmake-build.sh
   cmake --build ./build/
   ```

3. Run the benchmarks:

   ```bash
   ./build/BM_Matmul
   ```

4. Run the unit tests:

   ```bash
   ./build/UT_Matmul
   ```

## Optimization Techniques

### Basic Implementation
- **Naive Algorithm**: Basic matrix multiplication using three nested loops.

### Optimizations
- **Loop Unrolling**: Reduces the overhead of loop control and improves instruction-level parallelism.
- **Cache Optimization**: Minimizes cache misses by accessing matrix elements in a cache-friendly manner. See [Strassen algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm)
- **SIMD Instructions**: Uses processor-specific instructions to perform operations on multiple data points simultaneously.
- **Multi-threading**: Leverages multiple CPU cores to perform matrix multiplication in parallel.

## Benchmarking

The benchmarking is conducted by multiplying matrices of varying sizes and comparing the execution time of the custom implementations against the following libraries:
- **Eigen**: A high-performance library for linear algebra.
- **Intel MKL**: A highly optimized math library for Intel processors.
- **OpenBLAS**: An open-source implementation of BLAS (Basic Linear Algebra Subprograms).


## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or bug fixes.

### To Do
- [ ] Further optimize SIMD and multi-threaded implementations.
- [ ] Expand benchmarking to include GPU-based libraries, [std::linalg](https://en.cppreference.com/w/cpp/numeric/linalg).
- [ ] Add support for additional matrix formats (e.g., sparse matrices).

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

Special thanks to the developers of Eigen, Intel MKL, and OpenBLAS for providing high-quality linear algebra libraries.

---

This `README.md` provides an overview, setup instructions, and additional context to help users and contributors understand and work with the project.

