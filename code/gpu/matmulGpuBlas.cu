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

__global__ void transposeKernel(const float* input, float* output, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < N)
    {
        output[row * N + col] = input[col * N + row];
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
    // cuBLAS uses column-major, but our Matrix class uses row-major.
    // For row-major C = A * B:
    // - cuBLAS interprets row-major data as column-major (transposed)
    // - We compute C_col = B_col * A_col = B_row^T * A_row^T = (A_row * B_row)^T
    // - The result C_col needs to be transposed to get C_row = A_row * B_row
    const float alpha = 1.0;
    const float beta  = 0.0;
    
    // Allocate temporary buffer for C_col
    float *d_C_col;
    checkCudaError(cudaMalloc((void**)&d_C_col, matrixSize),
                    "Failed to allocate temporary device memory");
    
    // Compute C_col = B_col * A_col (swapped, no transpose)
    checkCublasError(cublasSgemm(handle,
                                    CUBLAS_OP_N,  // No transpose on B
                                    CUBLAS_OP_N,  // No transpose on A
                                    N,
                                    N,
                                    N,      // Dimensions
                                    &alpha, // Scaling factor alpha
                                    d_B,    // Swapped: B first
                                    N,      // Leading dimension of B
                                    d_A,    // Swapped: A second
                                    N,      // Leading dimension of A
                                    &beta,  // Scaling factor beta
                                    d_C_col,
                                    N       // Leading dimension of result
                                    ),
                        "Failed to perform SGEMM operation");
    
    // Transpose C_col to get C_row
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C_col, d_C, N);
    checkCudaError(cudaGetLastError(), "Transpose kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Transpose kernel execution failed");
    
    checkCudaError(cudaFree(d_C_col), "Failed to free temporary memory");

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