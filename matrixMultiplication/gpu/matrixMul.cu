
#include <iostream>
#include <cuda_runtime.h>

#include <chrono>
#include <string>
#include <vector>

class Profiler
{
public:
    Profiler();
    Profiler(std::string name);
    ~Profiler();

private:
    std::chrono::steady_clock::time_point _start;
    std::string                           _name;
};

#define PROFILE(NAME) Profiler p_##__LINE__{NAME}

Profiler::Profiler()
{
    _start = std::chrono::steady_clock::now();
    _name  = "deafult";
}

Profiler::Profiler(std::string name)
  : _start(std::chrono::steady_clock::now())
  , _name(std::move(name))
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    //std::cout << "[Profiling] Start " << _name << std::endl;
}

Profiler::~Profiler()
{
    std::cout << "[Profiling] " << _name << ". Took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - _start)
                   .count()
              << " ms" << std::endl;
}

// Kernel function is defined here or in another cu file
__global__ void matrixMulKernel(float *d_M, float *d_N, float *d_P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < width && col < width) {
        float Pvalue = 0.0;
        for (int k = 0; k < width; ++k) {
            Pvalue += d_M[row * width + k] * d_N[k * width + col];
        }
        d_P[row * width + col] = Pvalue;
    }
}


int main() {
    PROFILE("Matrix mul");
    int width = 3000; // Example size, can be changed

    size_t size = width * width * sizeof(float);

    float *h_M, *h_N, *h_P;
    float *d_M, *d_N, *d_P;

    // Allocate host memory
    h_M = (float *)malloc(size);
    h_N = (float *)malloc(size);
    h_P = (float *)malloc(size);

    // Initialize host matrices
    // ... (Code to initialize h_M and h_N)

    // Allocate device memory
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_N, size);
    cudaMalloc((void **)&d_P, size);

    // Copy matrices from host to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_M, d_N, d_P, width);

    // Copy result back to host
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}
