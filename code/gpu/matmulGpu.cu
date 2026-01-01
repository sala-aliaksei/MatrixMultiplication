#include "matmulGpu.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <string>
#include <vector>




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


__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < width && col < width)
    {
        float res = 0.0;
        for (int k = 0; k < width; ++k)
        {
            res += d_M[row * width + k] * d_N[k * width + col];
        }
        d_P[row * width + col] = res;
    }
}
}


namespace mm::gpu{

void matmulGpu(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C)
{
    int width = A.col();
    size_t size = width * width * sizeof(float);
    float *d_M, *d_N, *d_P;
    
    // Allocate device memory
    checkCudaError(cudaMalloc((void**)&d_M, size),
        "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_N, size),
        "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_P, size),
        "Failed to allocate device memory for C");


    checkCudaError(cudaMemcpy(d_M, A.data(), size, cudaMemcpyHostToDevice),
        "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_N, B.data(), size, cudaMemcpyHostToDevice),
        "Failed to copy B to device");

    // Kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, width);
    
    // Check for kernel launch errors and synchronize
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    checkCudaError(cudaMemcpy(C.data(), d_P, size, cudaMemcpyDeviceToHost),
        "Failed to copy C from device");

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}
}