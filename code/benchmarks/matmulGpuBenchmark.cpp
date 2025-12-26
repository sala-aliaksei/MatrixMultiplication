#include "gpu/matmulGpu.hpp"
#include "gpu/matmulGpuBlas.hpp"
#include "mm/core/utils/utils.hpp"
#include "benchmark_utils.hpp"

#include <benchmark/benchmark.h>

int main(int argc, char** argv)
{
    int matrix_dim = 4096;//GetMatrixDimFromEnv();

    // TODO: First run will compile the kernel so discard first run result
    REGISTER_FLOAT(mm::gpu::matmulGpu, matrix_dim);
    REGISTER_FLOAT(mm::gpu::matmulGpuBlas, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
