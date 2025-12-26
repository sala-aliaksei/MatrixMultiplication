#include "matmulGpuBlas.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>



namespace{

void checkCudaError(cudaError_t status, const char* msg)
{
    if (status != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkCublasError(cublasStatus_t status, const char* msg)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << msg << ": CUBLAS error" << std::endl;
        exit(EXIT_FAILURE);
    }
}
}


namespace mm::gpu{
void matmulGpuBlas(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C)
{
    const int    N          = A.col(); // Matrix size N x N

    const size_t matrixSize = N*N * sizeof(float);


    // Device matrices
    float *d_A, *d_B, *d_C;

    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_A, matrixSize),
                    "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, matrixSize),
                    "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, matrixSize),
                    "Failed to allocate device memory for C");

    // Copy matrices to device
    checkCudaError(cudaMemcpy(d_A, A.data(), matrixSize, cudaMemcpyHostToDevice),
                    "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, B.data(), matrixSize, cudaMemcpyHostToDevice),
                    "Failed to copy B to device");

    // cuBLAS handle
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    const float alpha = 1.0;
    const float beta  = 0.0;
    {
        checkCublasError(cublasSgemm(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N, // Transpose options
                                        N,
                                        N,
                                        N,      // Dimensions
                                        &alpha, // Scaling factor alpha
                                        d_A,
                                        N, // Matrix A
                                        d_B,
                                        N,     // Matrix B
                                        &beta, // Scaling factor beta
                                        d_C,
                                        N // Result matrix C
                                        ),
                            "Failed to perform DGEMM operation");
    }

    // Copy result back to host
    checkCudaError(cudaMemcpy(C.data(), d_C, matrixSize, cudaMemcpyDeviceToHost),
                    "Failed to copy C to host");

    // Print a small portion of the result to verify correctness
    // std::cout << "Result matrix C (first 5x5 block):" << std::endl;
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         std::cout << h_C[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    


}
}