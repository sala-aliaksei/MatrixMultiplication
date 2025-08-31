

#include "mm/genai/matMulClaude.hpp"
#include "mm/genai/matMulGpt.hpp"

#include "mm/tpi/matMulBlis.hpp"
#include "mm/tpi/matMulEigen.hpp"
#include "mm/tpi/matMulOpenBlas.hpp"

#include "mm/matmul/matMul.hpp"
#include "mm/matmul/matMulAutotune.hpp"
// #include "mm/matmul/matMulColOpt.hpp"
#include "mm/matmul/matMulLoops.hpp"
#include "mm/matmul/matMulPadding.hpp"
#include "mm/matmul/matMulRegOpt.hpp"
#include "mm/matmul/matMulSimd.hpp"

#include "benchmark_utils.hpp"
#include <benchmark/benchmark.h>

constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

static void BM_MatrixMulParam_Eigen(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initEigenMatrix(N, N, N);

    for (auto _ : state)
    {
        matrixMulEigen(matrices);
    }
}

//////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    int matrix_dim = GetMatrixDimFromEnv();

    // TPI
    REGISTER(BM_MatrixMulParam_Eigen, matrix_dim);
    REGISTER_DOUBLE(matmulBlis, matrix_dim);

    REGISTER_DOUBLE_RANGE(mm::tpi::matrixMulOpenBlas, matrix_dim);

    // GenAI
    REGISTER_DOUBLE(matMulClaude, matrix_dim);
    REGISTER_DOUBLE(gpt_matrix_multiply, matrix_dim);

// Matmul
#ifdef ENABLE_NAIVE_BENCHMARKS
    REGISTER_DOUBLE(mm::matMul_Naive, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Naive_Order, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Naive_Order_KIJ, matrix_dim);
#endif
    REGISTER_DOUBLE(mm::matMul_Naive_Tile, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Simd, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_AddRegs, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_AddRegs_Unroll, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache_Regs, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache_Regs_UnrollRW, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache_Regs_Unroll, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache_Regs_Unroll_BPack, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache_Regs_Unroll_MT, matrix_dim);
    REGISTER_DOUBLE(mm::matMul_Avx_Cache_Regs_Unroll_BPack_MT, matrix_dim);

    REGISTER_DOUBLE(matMulRegOpt, matrix_dim);

    REGISTER_DOUBLE(matMulLoopsRepack, matrix_dim);
    // REGISTER_DOUBLE(BM_MatMulLoopRepackIKJ, matrix_dim); // slower
    REGISTER_DOUBLE(matMulLoopsBPacked, matrix_dim);

    //
    REGISTER_DOUBLE(matMulPadding, matrix_dim);
    REGISTER_DOUBLE(matMulAutotune, matrix_dim);
    REGISTER_DOUBLE(matMulSimd, matrix_dim);

    REGISTER_DOUBLE(mm::matMul_Tails, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
