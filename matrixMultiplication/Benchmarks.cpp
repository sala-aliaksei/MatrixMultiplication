#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulGpt.hpp"
#include "matrixMultiplication/matrix/MatrixMulOpenBlas.hpp"
#include "matrixMultiplication/matrix/MatrixMulEigen.hpp"
#include "matrixMultiplication/matrix/claudeMatMul.hpp"
#include "matrixMultiplication/matrix/matMulRegOpt.hpp"
#include "matrixMultiplication/matrix/matMulColOpt.hpp"
#include "matrixMultiplication/matrix/matMulLoops.hpp"

#include "matrixMultiplication/matrix/cmatrix.h"

#include "matrix/disasm.hpp"

#include <benchmark/benchmark.h>

constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

// Provide matrix size to benchmarks
// constexpr std::size_t NN = 4 * 864;
constexpr std::size_t NN = 4 * 720;
// constexpr std::size_t NN = 4 * 768;

int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : NN; // Default to 2880 if not set
}

//---------------------------------------------------------------------
// Benchmark                           Time             CPU   Iterations
//---------------------------------------------------------------------
// BM_MatrixMulParam_MT_VT_BL        658 ms          638 ms            1
// BM_MatrixMulOpenBLAS              409 ms          407 ms            2

// When work only one thread (matmul was broken)
// constexpr std::size_t M = 768 * 4;
// constexpr std::size_t N = 768 * 4;
// Benchmark                           Time             CPU   Iterations
//---------------------------------------------------------------------
// BM_MatrixMulParam_MT_VT_BL        556 ms          552 ms            1
// BM_MatrixMulOpenBLAS              407 ms          404 ms            2

static void BM_MatrixMulOpenBLAS(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);
    for (auto _ : state)
    {
        matrixMulOpenBlas(matrices);
    }
}

static void BM_MatrixMulParam_MT_VT_BL(benchmark::State& state)
{
    std::size_t      N        = state.range(0);
    auto             matrices = initMatrix(N, N, N);
    DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_MT_VT_BL_TP(benchmark::State& state)
{
    std::size_t      N        = state.range(0);
    auto             matrices = initMatrix(N, N, N);
    DynamicMatrixMul mul(MatrixMulConfig{true, true, true, true});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive(benchmark::State& state)
{
    std::size_t      N        = state.range(0);
    auto             matrices = initMatrix(N, N, N);
    DynamicMatrixMul mul(MatrixMulConfig{false, false, false, false});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive_MT(benchmark::State& state)
{
    std::size_t      N        = state.range(0);
    auto             matrices = initMatrix(N, N, N);
    DynamicMatrixMul mul(MatrixMulConfig{true, false, false, false});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive_TP(benchmark::State& state)
{
    std::size_t      N        = state.range(0);
    auto             matrices = initMatrix(N, N, N);
    DynamicMatrixMul mul(MatrixMulConfig{false, false, true, false});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
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
//////////////////////////////////////////////////////////////////////////////

#define ENABLE_TEST_CN

//// Naive
#ifdef ENABLE_TEST_NAIVE
BENCHMARK(BM_MatrixMulParam_Naive)->Arg(NN);
BENCHMARK(BM_MatrixMulParam_Naive_TP)->Arg(NN);
BENCHMARK(BM_MatrixMulParam_Naive_MT)->Arg(NN);

#endif

#ifdef ENABLE_TEST_CN
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

// BENCHMARK(BM_CN_MatMulNaive)->Arg(NN);
// BENCHMARK(BM_CN_MatMulNaive_Order)->Arg(NN);
// BENCHMARK(BM_CN_MatMulNaive_Block)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Simd)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_AddRegs)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_AddRegsV2)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_AddRegs_Unroll)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_Cache)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_Cache_Regs)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_Cache_Regs_Unroll)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack)->Arg(NN);

// BENCHMARK(BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT)->Arg(NN);
// BENCHMARK(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT)->Arg(NN);

#endif

// Multithreads
#ifdef ENABLE_TEST_OPT
BENCHMARK(BM_MatrixMulOpenBLAS)->Arg(NN);
BENCHMARK(BM_MatMulRegOpt)->Arg(NN);
BENCHMARK(BM_MatMulLoopRepack)->Arg(NN);
BENCHMARK(BM_MatMulLoopBPacked)->Arg(NN);
BENCHMARK(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack)->Arg(NN);
#endif
// BENCHMARK(BM_MatMulColOpt)->Arg(NN);
//  BENCHMARK(BM_MatrixMulParam_MT_VT_BL_TP);
// BENCHMARK(BM_MatMulLoop)->Arg(NN); // slow
// BENCHMARK(BM_MatMulLoopIKJ)->Arg(NN); // slow
// BENCHMARK(BM_MatrixMulParam_MT_VT_BL)->Arg(NN); //slow

// Others
// BENCHMARK(BM_MatMulClaude);
// BENCHMARK(BM_MatrixMulParam_GPT);
// BENCHMARK(BM_MatrixMulParam_Eigen);

// TODO:
// Single threaded
// BM_MatrixMul_VT_BL_TP
// BM_MatrixMul_VT_BL
// BM_MatrixMul_BL_TP
// BM_MatrixMul_BL

// testMatrixMulStdBlas

// BENCHMARK_MAIN();

#define REGISTER(NAME, DIM) benchmark::RegisterBenchmark(#NAME, NAME)->Arg(DIM);

int main(int argc, char** argv)
{
    int matrix_dim = GetMatrixDimFromEnv();

#ifdef ENABLE_TEST_CN
    REGISTER(BM_MatrixMulOpenBLAS, matrix_dim);

    //    REGISTER(BM_CN_MatMulNaive, matrix_dim);
    //    REGISTER(BM_CN_MatMulNaive, matrix_dim);
    //    REGISTER(BM_CN_MatMulNaive_Order, matrix_dim);
    //    REGISTER(BM_CN_MatMulNaive_Block, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Simd, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_AddRegs, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_AddRegsV2, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_AddRegs_Unroll, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_Cache, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_Cache_Regs, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_UnrollRW, matrix_dim);

    //    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll, matrix_dim);
    //    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT, matrix_dim);

    //    REGISTER(BM_MatrixMulParam_GPT, matrix_dim);

#endif
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
