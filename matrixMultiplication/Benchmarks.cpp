#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulGpt.hpp"
#include "matrixMultiplication/matrix/MatrixMulOpenBlas.hpp"
#include "matrixMultiplication/matrix/MatrixMulEigen.hpp"
#include "matrixMultiplication/matrix/claudeMatMul.hpp"
#include "matrixMultiplication/matrix/matMulRegOpt.hpp"
#include "matrixMultiplication/matrix/cmatrix.h"

#include "matrix/disasm.hpp"

#include <benchmark/benchmark.h>

constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

// Provide matrix size to benchmarks
constexpr std::size_t M = 768 * 4; // 256 * 8; // * 8 * 4; //* 8 * 4 * 2;
constexpr std::size_t N = 768 * 4; // 256 * 8; // * 8 * 4; //* 8 * 4 * 2;
constexpr std::size_t K = 768 * 4;

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

static void BM_MatrixMulOpenBLAS_TP(benchmark::State& state)
{
    auto set = initMatrix(M, N, K);
    for (auto _ : state)
    {
        matrixMulOpenBlas_TP(set);
    }
}

static void BM_MatrixMulOpenBLAS(benchmark::State& state)
{
    auto set = initMatrix(M, N, K);
    for (auto _ : state)
    {
        matrixMulOpenBlas(set);
    }
}

static void BM_MatrixMulParam_MT_VT_BL(benchmark::State& state)
{
    auto             matrices = initMatrix(M, N, K);
    DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_MT_VT_BL_TP(benchmark::State& state)
{
    auto             matrices = initMatrix(M, N, K);
    DynamicMatrixMul mul(MatrixMulConfig{true, true, true, true});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive(benchmark::State& state)
{
    auto             matrices = initMatrix(M, N, K);
    DynamicMatrixMul mul(MatrixMulConfig{false, false, false, false});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive_MT(benchmark::State& state)
{
    auto             matrices = initMatrix(M, N, K);
    DynamicMatrixMul mul(MatrixMulConfig{true, false, false, false});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive_TP(benchmark::State& state)
{
    auto             matrices = initMatrix(M, N, K);
    DynamicMatrixMul mul(MatrixMulConfig{false, false, true, false});

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_GPT(benchmark::State& state)
{
    auto matrices = initMatrix(M, N, K);

    for (auto _ : state)
    {
        gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_GPT_v2(benchmark::State& state)
{
    auto matrices = initMatrix(M, N, K);

    for (auto _ : state)
    {
        matrix_multiply(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Eigen(benchmark::State& state)
{
    auto matrices = initEigenMatrix(M, N, K);

    for (auto _ : state)
    {
        matrixMulEigen(matrices);
    }
}

static void BM_MatMul_Skylake(benchmark::State& state)
{
    auto matrices = initMatrix(M, N, K);

    for (auto _ : state)
    {
        blasMatMul(M, N, K, 1.0, matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_MatMulClaude(benchmark::State& state)
{
    auto matrices = initMatrix(M, N, K);

    for (auto _ : state)
    {
        multiply_matrices_optimized(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatMulRegOpt(benchmark::State& state)
{
    auto matrices = initMatrix(M, N, K);

    for (auto _ : state)
    {
        matMulRegOpt(matrices.a, matrices.b, matrices.c);
    }
}
//////////////////////////////////////////////////////////////////////////////

//// Naive
#ifdef ENABLE_TEST_NAIVE
BENCHMARK(BM_MatrixMulParam_Naive);
BENCHMARK(BM_MatrixMulParam_Naive_TP);
BENCHMARK(BM_MatrixMulParam_Naive_MT);
#endif

// Multithreads
BENCHMARK(BM_MatrixMulParam_MT_VT_BL);
BENCHMARK(BM_MatMulRegOpt);
// BENCHMARK(BM_MatrixMulParam_MT_VT_BL_TP);

//// OpenBLAS
// BENCHMARK(BM_MatrixMulOpenBLAS_TP);
BENCHMARK(BM_MatrixMulOpenBLAS);
// BENCHMARK(BM_MatMul_Skylake);

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

BENCHMARK_MAIN();
