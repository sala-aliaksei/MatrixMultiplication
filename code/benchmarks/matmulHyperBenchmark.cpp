
#include "mm/matmul/matMulHyper.hpp"
#include "mm/core/utils/utils.hpp"
#include "benchmark_utils.hpp"
#include <benchmark/benchmark.h>

int main(int argc, char** argv)
{
    int matrix_dim = GetMatrixDimFromEnv();

    REGISTER_DOUBLE(mm::hyper::matMulHyper, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
