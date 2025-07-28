

#include "mm/genai/matMulClaude.hpp"
#include "mm/genai/matMulGpt.hpp"

#include "mm/tpi/matMulBlis.hpp"
#include "mm/tpi/matMulEigen.hpp"
#include "mm/tpi/matMulOpenBlas.hpp"

#include "mm/matmul/matMul.hpp"
#include "mm/matmul/matMulAutotune.hpp"
#include "mm/matmul/matMulColOpt.hpp"
#include "mm/matmul/matMulLoops.hpp"
#include "mm/matmul/matMulPadding.hpp"
#include "mm/matmul/matMulRegOpt.hpp"
#include "mm/matmul/matMulSimd.hpp"

// #include "mm/matmul/cmatrix.h"

// #define ENABLE_NAIVE_BENCHMARKS

#include <benchmark/benchmark.h>

// Most of the implementations don't handle matrix tailes, so we need to
// hardcode the size

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

static void BM_MatMulLoopRepackIKJ(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        matMulLoopsRepackIKJ(matrices.a, matrices.b, matrices.c);
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

//   mm implementation

static void BM_CN_MatMulNaive(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Naive(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMulNaive_Block(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Naive_Tile(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMulNaive_Order_KIJ(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Naive_Order_KIJ(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMulNaive_Order(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Naive_Order(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Simd(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Simd(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_AddRegs(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_AddRegs(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_AddRegs_Unroll(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_AddRegs_Unroll(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache_Regs(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache_Regs_Unroll(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_UnrollRW(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache_Regs_UnrollRW(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache_Regs_Unroll_BPack(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache_Regs_Unroll_MT(matrices.a, matrices.b, matrices.c);
    }
}

static void BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Avx_Cache_Regs_Unroll_BPack_MT(matrices.a, matrices.b, matrices.c);
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

static void BM_MatMulTailed(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initMatrix(N, N, N);

    for (auto _ : state)
    {
        mm::matMul_Tails(matrices.a, matrices.b, matrices.c);
    }
}
//////////////////////////////////////////////////////////////////////////////

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
    REGISTER(BM_CN_MatMulNaive_Order, matrix_dim);
    REGISTER(BM_CN_MatMulNaive_Order_KIJ, matrix_dim);
#endif
    REGISTER(BM_CN_MatMulNaive_Block, matrix_dim);
    REGISTER(BM_CN_MatMul_Simd, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_AddRegs, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_AddRegs_Unroll, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_UnrollRW, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT, matrix_dim);
    REGISTER(BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT, matrix_dim);

    REGISTER(BM_MatMulRegOpt, matrix_dim);

    REGISTER(BM_MatMulLoopRepack, matrix_dim);
    // REGISTER(BM_MatMulLoopRepackIKJ, matrix_dim); // slower
    REGISTER(BM_MatMulLoopBPacked, matrix_dim);

    //
    REGISTER(BM_MatMulPadding, matrix_dim);
    REGISTER(BM_MatMulAutotune, matrix_dim);
    REGISTER(BM_MatMulSimd, matrix_dim);

    REGISTER(BM_MatMulTailed, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
