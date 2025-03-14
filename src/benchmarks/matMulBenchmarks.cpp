

#include "mm/genai/matMulClaude.hpp"
#include "mm/genai/matMulGpt.hpp"

#include "mm/tpi/matMulOpenBlas.hpp"
#include "mm/tpi/matMulBlis.hpp"
#include "mm/tpi/matMulEigen.hpp"

#include "mm/matmul/matMul.hpp"
#include "mm/matmul/matMulRegOpt.hpp"
#include "mm/matmul/matMulColOpt.hpp"
#include "mm/matmul/matMulLoops.hpp"
#include "mm/matmul/matMulPadding.hpp"
#include "mm/matmul/matMulAutotune.hpp"
#include "mm/matmul/matMulSimd.hpp"

// #include "mm/matmul/cmatrix.h"

// #define ENABLE_NAIVE_BENCHMARKS

#include <benchmark/benchmark.h>

constexpr std::size_t NN        = 4 * 720;
constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : NN;
}

static void BM_MatrixMulOpenBLAS(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);
    for (auto _ : state)
    {
        matrixMulOpenBlas(matrices);
    }
}

static void BM_MatrixMulBLIS(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);
    for (auto _ : state)
    {
        matmulBlis(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_GPT(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Eigen(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initEigenMatrix(N, N, N);

    for (auto _ : state)
    {
        matrixMulEigen(matrices);
    }
}

static void BM_MatMulClaude(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        multiply_matrices_optimized(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulRegOpt(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulRegOpt(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulColOpt(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulColOpt(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulLoop(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulLoops(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulLoopRepack(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulLoopsRepack(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulLoopRepackV2(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulLoopsRepackV2(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulLoopIKJ(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulLoopsIKJ(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulLoopBPacked(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulLoopsBPacked(matrices.a, matrices.b, matrices.c);
    }
}

//   CPPNow implementation

static void BM_CN_MatMulNaive(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Naive(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMulNaive_Block(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Naive_Block(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMulNaive_Order_KIJ(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Naive_Order_KIJ(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMulNaive_Order(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Naive_Order(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Simd(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Simd(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_AddRegs(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_AddRegs(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_AddRegsV2(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_AddRegsV2(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_AddRegs_Unroll(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_AddRegs_Unroll(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache_Regs(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache_Regs_Unroll(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_UnrollRW(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache_Regs_UnrollRW(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache_Regs_Unroll_BPack(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache_Regs_Unroll_MT(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        cppnow::matMul_Avx_Cache_Regs_Unroll_BPack_MT(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulPadding(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulPadding(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulAutotune(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulAutotune(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulSimd(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulSimd(matrices.a, matrices.b, matrices.c);
    }
}
//////////////////////////////////////////////////////////////////////////////

// BM_CN_MatMulNaive/2880                               152089 ms       151374 ms            1
// BM_CN_MatMulNaive_Order/2880                          11938 ms        11889 ms            1
// BM_CN_MatMulNaive_Block/2880                           3908 ms         3892 ms            1
// BM_CN_MatMul_Simd/2880                                 7372 ms         7349 ms            1
// BM_CN_MatMul_Avx/2880                                  3158 ms         3147 ms            1
// BM_CN_MatMul_Avx_AddRegs/2880                          2663 ms         2653 ms            1
// BM_CN_MatMul_Avx_AddRegsV2/2880                        2454 ms         2444 ms            1
// BM_CN_MatMul_Avx_AddRegs_Unroll/2880                   6701 ms         6637 ms            1
// BM_CN_MatMul_Avx_Cache/2880                            3769 ms         3756 ms            1
// BM_CN_MatMul_Avx_Cache_Regs/2880                       2751 ms         2740 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_Unroll/2880                1375 ms         1369 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack/2880          1503 ms         1496 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT/2880              440 ms          436 ms            2
// BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT/2880        496 ms          455 ms            2
// BM_MatMulLoopRepackMT/2880                              374 ms          372 ms            2
// BM_MatMulLoopRepack/2880                                1370 ms         1370 ms            1

// BM_CN_MatMul_Avx/2880                                  3200 ms         3185 ms            1
// BM_CN_MatMul_Avx_AddRegs/2880                          2552 ms         2542 ms            1
// BM_CN_MatMul_Avx_AddRegsV2/2880                        2552 ms         2542 ms            1
// BM_CN_MatMul_Avx_AddRegs_Unroll/2880                   6933 ms         6868 ms            1
// BM_CN_MatMul_Avx_Cache/2880                            2709 ms         2698 ms            1
// BM_CN_MatMul_Avx_Cache_Regs/2880                       2705 ms         2695 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_UnrollRW/2880              1407 ms         1400 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_Unroll/2880                1429 ms         1423 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack/2880          1540 ms         1533 ms            1
// BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT/2880              466 ms          460 ms            2
// BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT/2880        477 ms          473 ms            2
// BM_MatMulRegOpt/2880                                    398 ms          390 ms            2
// BM_MatMulLoopRepack/2880                                369 ms          365 ms            2
// BM_MatMulLoopRepackV2/2880                              415 ms          412 ms            2
// BM_MatMulLoopBPacked/2880                               472 ms          467 ms            2
// BM_MatrixMulOpenBLAS/2880                               330 ms          327 ms            2

// BENCHMARK_MAIN();

#define REGISTER(NAME, DIM) benchmark::RegisterBenchmark(#NAME, NAME)->Arg(DIM);

int main(int argc, char** argv)
{
    int matrix_dim = GetMatrixDimFromEnv();

    // TPI
    REGISTER(BM_MatrixMulParam_Eigen, matrix_dim);
    REGISTER(BM_MatrixMulOpenBLAS, matrix_dim);
    REGISTER(BM_MatrixMulBLIS, matrix_dim);

    // GenAI
    REGISTER(BM_MatMulClaude, matrix_dim);
    REGISTER(BM_MatrixMulParam_GPT, matrix_dim);

// Matmul
#ifdef ENABLE_NAIVE_BENCHMARKS
    REGISTER(BM_CN_MatMulNaive, matrix_dim);
    REGISTER(BM_CN_MatMulNaive_Block, matrix_dim);
    REGISTER(BM_CN_MatMulNaive_Order, matrix_dim);
    REGISTER(BM_CN_MatMulNaive_Order_KIJ, matrix_dim);
#endif
    REGISTER(BM_CN_MatMul_Simd, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_AddRegs, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_AddRegsV2, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_AddRegs_Unroll, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_UnrollRW, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT, matrix_dim);

    REGISTER(BM_MatMulRegOpt, matrix_dim);
    REGISTER(BM_MatMulColOpt, matrix_dim);

    REGISTER(BM_MatMulLoopRepack, matrix_dim);
    REGISTER(BM_MatMulLoopRepackV2, matrix_dim); // slower
    REGISTER(BM_MatMulLoopBPacked, matrix_dim);

    //
    REGISTER(BM_MatMulPadding, matrix_dim);
    REGISTER(BM_MatMulAutotune, matrix_dim);
    REGISTER(BM_MatMulSimd, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}

// static void BM_MatrixMulParam_MT_VT_BL(benchmark::State& state)
//{
//     std::size_t      N        = state.range(0);
//     auto             matrices = initMatrix(N, N, N);
//     DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});

//    for (auto _ : state)
//    {
//        mul(matrices.a, matrices.b, matrices.c);
//    }
//}

// static void BM_MatrixMulParam_MT_VT_BL_TP(benchmark::State& state)
//{
//     std::size_t      N        = state.range(0);
//     auto             matrices = initMatrix(N, N, N);
//     DynamicMatrixMul mul(MatrixMulConfig{true, true, true, true});

//    for (auto _ : state)
//    {
//        mul(matrices.a, matrices.b, matrices.c);
//    }
//}

// static void BM_MatrixMulParam_Naive(benchmark::State& state)
//{
//     std::size_t      N        = state.range(0);
//     auto             matrices = initMatrix(N, N, N);
//     DynamicMatrixMul mul(MatrixMulConfig{false, false, false, false});

//    for (auto _ : state)
//    {
//        mul(matrices.a, matrices.b, matrices.c);
//    }
//}

// static void BM_MatrixMulParam_Naive_MT(benchmark::State& state)
//{
//     std::size_t      N        = state.range(0);
//     auto             matrices = initMatrix(N, N, N);
//     DynamicMatrixMul mul(MatrixMulConfig{true, false, false, false});

//    for (auto _ : state)
//    {
//        mul(matrices.a, matrices.b, matrices.c);
//    }
//}

// static void BM_MatrixMulParam_Naive_TP(benchmark::State& state)
//{
//     std::size_t      N        = state.range(0);
//     auto             matrices = initMatrix(N, N, N);
//     DynamicMatrixMul mul(MatrixMulConfig{false, false, true, false});

//    for (auto _ : state)
//    {
//        mul(matrices.a, matrices.b, matrices.c);
//    }
//}
