#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <ctime>

void checkCudaError(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << ": CUBLAS error" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 3072; // Matrix size N x N
    const size_t matrixSize = N * N * sizeof(double);

    // Host matrices
    double *h_A = new double[N * N];
    double *h_B = new double[N * N];
    double *h_C = new double[N * N];

    // Initialize host matrices with random values
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
        h_B[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Device matrices
    double *d_A, *d_B, *d_C;

    // Allocate device memory
    checkCudaError(cudaMalloc((void **)&d_A, matrixSize), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void **)&d_B, matrixSize), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void **)&d_C, matrixSize), "Failed to allocate device memory for C");

    // Copy matrices to device
    checkCudaError(cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice), "Failed to copy B to device");

    // cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    const double alpha = 1.0;
    const double beta = 0.0;

    checkCublasError(
        cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,  // Transpose options
            N, N, N,                  // Dimensions
            &alpha,                   // Scaling factor alpha
            d_A, N,                   // Matrix A
            d_B, N,                   // Matrix B
            &beta,                    // Scaling factor beta
            d_C, N                    // Result matrix C
        ),
        "Failed to perform DGEMM operation"
    );

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost), "Failed to copy C to host");

    // Print a small portion of the result to verify correctness
    std::cout << "Result matrix C (first 5x5 block):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

