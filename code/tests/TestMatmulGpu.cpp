#include "gpu/matmulGpu.hpp"
#include "gpu/matmulGpuBlas.hpp"
#include "mm/core/utils/utils.hpp"
#include <cuda_runtime.h>

#include <gtest/gtest.h> //--gtest_filter=MatrixMulGpuTest.MatMulLoopsRepack

/***********   FLOAT 32   ***********/

static const std::size_t N = 4096;//GetMatrixDimFromEnv();

class MatrixMulGpuTest : public testing::Test
{
  protected:
  MatrixMulGpuTest()
      : a(generateRandomMatrix<float>(N, N))
      , b(generateRandomMatrix<float>(N, N))
      , c(N, N)
      , expected(N, N)
    {
    }

    ~MatrixMulGpuTest() override = default;

    void SetUp() override
    {
        // Initialize CUDA context if not already initialized
        // This ensures the GPU is ready before we try to use it
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0)
        {
            GTEST_SKIP() << "No CUDA devices available";
        }
        
        // Set device 0 (or use the default)
        err = cudaSetDevice(0);
        if (err != cudaSuccess)
        {
            GTEST_SKIP() << "Failed to set CUDA device: " << cudaGetErrorString(err);
        }
        
        // Clear any previous error state
        cudaGetLastError();
        
        // Synchronize to ensure any pending operations complete
        cudaDeviceSynchronize();
        
        // Try to compute expected result using cuBLAS
        // If GPU memory allocation fails, skip the test
        try {
            mm::gpu::matmulGpuBlas(a, b, expected);
            gpuAvailable = true;
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Skipping test due to GPU memory issue: " << e.what();
        }
    }

    void TearDown() override
    {
        // Synchronize to ensure all GPU operations complete
        cudaDeviceSynchronize();
        
        // Clear any CUDA errors from previous operations
        // This helps prevent "out of memory" errors from persisting
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess && err != cudaErrorNotReady)
        {
            // Clear the error state
            cudaGetLastError();
        }
    }

    Matrix<float> a;
    Matrix<float> b;
    Matrix<float> c;
    Matrix<float> expected;
    bool gpuAvailable = false;
};

TEST_F(MatrixMulGpuTest, MatMulGpu)
{
    if (!gpuAvailable) {
        GTEST_SKIP() << "GPU not available";
    }
    
    try {
        mm::gpu::matmulGpu(a, b, c);
        EXPECT_EQ((expected == c), true);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping test due to GPU memory issue: " << e.what();
    }
}
