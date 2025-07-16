#include "mm/core/reorderMatrix.hpp"

#include <benchmark/benchmark.h>

#include <immintrin.h>

// constexpr int NN = 1536; // L3 Cache size
constexpr int NN = 2 * 2048;

static void BM_NaiveColumnReorder(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<double>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr int Mc = 24;
    constexpr int Nc = 96;
    //    constexpr int Mc = 180;
    //    constexpr int Nc = 720;

    constexpr int Mr = 4;
    constexpr int Nr = 12;

    std::vector<double> buf(Mc * Nc);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        //
        reorderColOrderMatrix<Mc, Nc, Mr, Nr>(arr.data(), N, buf.data());
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_NaiveRowReorder(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<double>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr int Mc = 180;
    constexpr int Nc = 720;

    constexpr int Mr = 4;
    constexpr int Nr = 4;

    std::vector<double> buf(Mc * Nc);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        //
        reorderRowMajorMatrix<Mc, Nc, Mr, Nr>(arr.data(), N, buf.data());
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_BlasRowReorder(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<double>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr int Mc = 180;
    constexpr int Nc = 720;

    constexpr int Mr = 4;
    constexpr int Nr = 4;

    std::vector<double> buf(Mc * Nc);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        //
        blasReorderRowOrder4x4(Mc, Nc, arr.data(), N, buf.data());
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_BlasColReorder(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<double>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr int Mc = 180;
    constexpr int Nc = 720;

    constexpr int Mr = 8;
    constexpr int Nr = 8;

    std::vector<double> buf(Mc * Nc);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        //
        blasReorderColOrder8x8(Mc, Nc, arr.data(), N, buf.data());
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

BENCHMARK(BM_BlasRowReorder)->Arg(NN);
BENCHMARK(BM_NaiveRowReorder)->Arg(NN);

BENCHMARK(BM_BlasColReorder)->Arg(NN);
BENCHMARK(BM_NaiveColumnReorder)->Arg(NN);

BENCHMARK_MAIN();
