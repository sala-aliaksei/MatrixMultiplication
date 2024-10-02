#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulFunctions.hpp"
#include "matrixMultiplication/matrix/MatrixMulGpt.hpp"
#include "matrixMultiplication/matrix/MatrixMulOpenBlas.hpp"
#include "matrixMultiplication/matrix/MatrixMulEigen.hpp"
#include "matrixMultiplication/matrix/cmatrix.h"

#include <benchmark/benchmark.h>

constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

// TODO: Pass runtimer args: num_of_thread, block_size, etc...
// Provide matrix size to benchmarks

static void BM_MatrixMulOpenBLAS_TP(benchmark::State& state)
{
    auto set = initMatrix();
    for (auto _ : state)
    {
        matrixMulOpenBlas_TP(set);
    }
}

static void BM_MatrixMulOpenBLAS(benchmark::State& state)
{
    auto set = initMatrix();
    for (auto _ : state)
    {
        matrixMulOpenBlas(set);
    }
}

static void BM_MatrixMul_MT_VT_BL(benchmark::State& state)
{
    auto setFinal = initMatrix();
    for (auto _ : state)
    {
        matrixMul_MT_VT_BL(setFinal);
    }
}

static void BM_MatrixMulNaiveTransposed(benchmark::State& state)
{
    auto setNaiveTransposed = initMatrix();
    for (auto _ : state)
    {
        matrixMul_Naive_TP(setNaiveTransposed);
    }
}

static void BM_MatrixMulNaive(benchmark::State& state)
{
    auto setNaive = initMatrix();
    for (auto _ : state)
    {
        matrixMul_Naive(setNaive);
    }
}

static void BM_MatrixMul_MT_VT_BL_TP(benchmark::State& state)
{
    auto matrices = initMatrix();
    for (auto _ : state)
    {
        matrixMul_MT_VT_BL_TP(matrices);
    }
}

static void BM_MatrixMulParam_MT_VT_BL_TP(benchmark::State& state)
{
    auto             matrices = initMatrix();
    DynamicMatrixMul mul(std::thread::hardware_concurrency(), 8, true, true);

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive(benchmark::State& state)
{
    auto             matrices = initMatrix();
    DynamicMatrixMul mul(1, 1, false, false);

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Naive_TP(benchmark::State& state)
{
    auto             matrices = initMatrix();
    DynamicMatrixMul mul(1, 1, true, false);

    for (auto _ : state)
    {
        mul(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_GPT(benchmark::State& state)
{
    auto matrices = initMatrix();

    for (auto _ : state)
    {
        gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_GPT_v2(benchmark::State& state)
{
    auto matrices = initMatrix();

    for (auto _ : state)
    {
        matrix_multiply(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_MatrixMulParam_Eigen(benchmark::State& state)
{
    auto matrices = initEigenMatrix();

    for (auto _ : state)
    {
        matrixMulEigen(matrices);
    }
}

//////////////////////////////////////////////////////////////////////////////

//// Naive
BENCHMARK(BM_MatrixMulNaive);
BENCHMARK(BM_MatrixMulNaiveTransposed);
BENCHMARK(BM_MatrixMulParam_Naive);
BENCHMARK(BM_MatrixMulParam_Naive_TP);

// Multithreads
BENCHMARK(BM_MatrixMul_MT_VT_BL);
BENCHMARK(BM_MatrixMulParam_MT_VT_BL_TP);
BENCHMARK(BM_MatrixMul_MT_VT_BL_TP);

//// OpenBLAS
BENCHMARK(BM_MatrixMulOpenBLAS_TP);
BENCHMARK(BM_MatrixMulOpenBLAS);

// Others
BENCHMARK(BM_MatrixMulParam_GPT);
BENCHMARK(BM_MatrixMulParam_Eigen);

// TODO:
// Single threaded
// BM_MatrixMul_VT_BL_TP
// BM_MatrixMul_VT_BL
// BM_MatrixMul_BL_TP
// BM_MatrixMul_BL

// testMatrixMulStdBlas

BENCHMARK_MAIN();
