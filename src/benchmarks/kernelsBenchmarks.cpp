#include "mm/core/Matrix.hpp"
#include "mm/core/kernels.hpp"
#include <benchmark/benchmark.h>

constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;
#define REGISTER(NAME, DIM) benchmark::RegisterBenchmark(#NAME, NAME)->Arg(DIM);

constexpr std::size_t NN = 4 * 720;
constexpr std::size_t M  = NN;
constexpr std::size_t N  = NN; // + 8;
constexpr std::size_t K  = NN;

constexpr int Mc = 20;
constexpr int Nc = 720;
constexpr int Kc = 80;

constexpr int Nr = 8;
constexpr int Mr = 4;

static void BM_CppGenericKern(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::cpp_generic_ukern<12, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N, K);
    }
}

static void BM_PackedKernelGeneric12x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::cpp_packed_kernel<12, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_GenericKernel8x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::cpp_generic_ukern<8, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N, K);
    }
}

static void BM_PackedKernel8x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::packed_ukernel8x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_PackedKernel8x4Aregs(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::packed_ukernel8x4_more_a_regs<Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_PackedKernelGeneric8x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::cpp_packed_kernel<8, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_PackedKernelGeneric4x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::cpp_packed_kernel<4, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_PackedKernel4x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::packed_ukernel4x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

////

static void BM_GenericKernel2x4(benchmark::State& state)
{
    constexpr int Nr = 2;
    constexpr int Mr = 4;

    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        kernels::cpp_generic_ukern<Nr, Mr, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr, K);
    }
}

static void BM_PackedKernel2x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        kernels::packed_ukernel2x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernelGeneric2x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        kernels::cpp_packed_kernel<2, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernel6x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        kernels::packed_ukernel6x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernel1x4(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        kernels::packed_ukernel1x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernel1x4_Simd(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        kernels::packed_ukernel1x4_simd<Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

int main(int argc, char** argv)
{
    int matrix_dim = NN;

    REGISTER(BM_CppGenericKern, matrix_dim);
    REGISTER(BM_PackedKernelGeneric12x4, matrix_dim);
    REGISTER(BM_GenericKernel8x4, matrix_dim);
    REGISTER(BM_PackedKernel8x4, matrix_dim);
    REGISTER(BM_PackedKernel8x4Aregs, matrix_dim);
    REGISTER(BM_PackedKernelGeneric8x4, matrix_dim);

    REGISTER(BM_PackedKernelGeneric4x4, matrix_dim);
    REGISTER(BM_PackedKernel4x4, matrix_dim);

    REGISTER(BM_GenericKernel2x4, matrix_dim);
    REGISTER(BM_PackedKernel2x4, matrix_dim);
    REGISTER(BM_PackedKernelGeneric2x4, matrix_dim);
    REGISTER(BM_PackedKernel6x4, matrix_dim);

    REGISTER(BM_PackedKernel1x4, matrix_dim);
    REGISTER(BM_PackedKernel1x4_Simd, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
