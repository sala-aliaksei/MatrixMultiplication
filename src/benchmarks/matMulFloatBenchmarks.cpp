

#include "mm/tpi/matMulOpenBlas.hpp"
#include "mm/matmul/matMulZen5.hpp"

#include "benchmark_utils.hpp"

int main(int argc, char** argv)
{
    int matrix_dim = GetMatrixDimFromEnv();

    REGISTER_FLOAT_RANGE(mm::tpi::matrixMulOpenBlas, matrix_dim);

    REGISTER_FLOAT(mm::zen5::matMulZen5, matrix_dim);
    REGISTER_FLOAT(mm::zen5::matMulZen5MTBlocking, matrix_dim);

    REGISTER_FLOAT_RANGE(mm::zen5::matMulZen5MTBlockingTails, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}