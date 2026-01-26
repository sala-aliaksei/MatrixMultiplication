
#include "mm/matmul/matMulZen5.hpp"
#include "mm/core/utils/utils.hpp"
#include "benchmark_utils.hpp"
#include <benchmark/benchmark.h>
#include "mm/matmul/matMulStrassen.hpp"

int main(int argc, char** argv)
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    int matrix_dim = GetMatrixDimFromEnv();

    // Zen5
    REGISTER_DOUBLE_MT(mm::strassen::matMulStrassen, matrix_dim);
    REGISTER_DOUBLE_MT(mm::zen5::matMulZen5, matrix_dim);
    REGISTER_DOUBLE_MT(mm::zen5::matMulZen5MTBlocking, matrix_dim);
    REGISTER_DOUBLE_MT(mm::zen5::matMulZen5MTBlockingTails, matrix_dim);
    REGISTER_DOUBLE_MT(mm::zen5::matMulZen5Padding, matrix_dim);
    REGISTER_DOUBLE_MT(mm::zen5::matMulZen5MTBlockingSpan, matrix_dim);
    // REGISTER_DOUBLE(mm::zen5::matMulZen5MTBlockingL1, matrix_dim);
    //  REGISTER_DOUBLE_RANGE(mm::zen5::matMulZen5MTBlockingTails, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
