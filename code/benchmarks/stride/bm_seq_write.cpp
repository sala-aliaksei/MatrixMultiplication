
#include <benchmark/benchmark.h>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <pthread.h>
#include "mm/core/utils/cpu.hpp"
#include "tracy_utils/tracy_cache_miss_counter.hpp"
#include <tracy/Tracy.hpp>
#include <cstring>

// constexpr int NN = 1536; // L3 Cache size
constexpr int NN = 2 * 2048;

static void BM_RowStride(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);

    benchmark::DoNotOptimize(arr);
    for (auto _ : state)
    {
        for (int cnt = 0; cnt < N; cnt++)
        {
            for (int i = 0, stride = 0; i < N; i++, stride++)
            {
                arr[stride] += 1;
            }
        }
    }
    benchmark::ClobberMemory(); // Prevent reordering across iterations
}

static void BM_ColStride(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        for (int cnt = 0; cnt < N; cnt++)
        {
            for (int i = 0, stride = 0; i < N; i++, stride += N)
            {
                // No impact
                //_mm_prefetch(arr.data() + 16, prefetch_type);
                arr[stride] += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_ColStridePrefetch(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        for (int cnt = 0; cnt < N; cnt++)
        {
            for (int i = 0, stride = 0; i < N; i++, stride += N)
            {

                _mm_prefetch(arr.data() + stride + N, prefetch_type);
                _mm_prefetch(arr.data() + stride + 2 * N, prefetch_type);
                _mm_prefetch(arr.data() + stride + 3 * N, prefetch_type);
                _mm_prefetch(arr.data() + stride + 4 * N, prefetch_type);
                _mm_prefetch(arr.data() + stride + 5 * N, prefetch_type);
                _mm_prefetch(arr.data() + stride + 6 * N, prefetch_type);
                _mm_prefetch(arr.data() + stride + 7 * N, prefetch_type);

                arr[stride] += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_RowOrderMattrix(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);

    constexpr auto prefetch_type = _MM_HINT_T2;

    benchmark::DoNotOptimize(arr);
    //    for (auto _ : state)
    //    {
    //        for (int i = 0; i < N; i++)
    //        {
    //            //            _mm_prefetch(arr.data() + (i + 1) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 2) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 3) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 4) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 5) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 6) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 7) * N, prefetch_type);
    //            //            _mm_prefetch(arr.data() + (i + 8) * N, prefetch_type);
    //            for (int j = 0; j < N; j++)
    //            {
    //                arr[i * N + j] += 1;
    //            }
    //        }
    //        benchmark::ClobberMemory(); // Prevent reordering across iterations
    //    }

    for (auto _ : state)
    {
        int* a = arr.data();
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++, ++a)
            {
                *a += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_ColOrderMattrix(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {

                arr[i + j * N] += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_ColOrderMatrixPrefetch(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {

        for (int i = 0; i < N; i++)
        {

            for (int j = 0; j < N; j++)
            {
                _mm_prefetch(arr.data() + (j + 1) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 2) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 3) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 4) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 5) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 6) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 7) * N + i, prefetch_type);
                _mm_prefetch(arr.data() + (j + 8) * N + i, prefetch_type);

                arr[i + j * N] += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_ColOrderMatrixPrefetchBadRefactor(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        for (int i = 0; i < N; i++)
        {
            int* a = &arr[i];
            for (int j = 0; j < N; j++, a += N)
            {
                _mm_prefetch(a + (j + 1) * N, prefetch_type);
                _mm_prefetch(a + (j + 2) * N, prefetch_type);
                _mm_prefetch(a + (j + 3) * N, prefetch_type);
                _mm_prefetch(a + (j + 4) * N, prefetch_type);
                _mm_prefetch(a + (j + 5) * N, prefetch_type);
                _mm_prefetch(a + (j + 6) * N, prefetch_type);
                _mm_prefetch(a + (j + 7) * N, prefetch_type);
                _mm_prefetch(a + (j + 8) * N, prefetch_type);

                *a += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_ColOrderMatrixPrefetchRefactor(benchmark::State& state)
{
    std::size_t N   = state.range(0);
    auto        arr = std::vector<int>(N * N);
    benchmark::DoNotOptimize(arr);

    constexpr auto prefetch_type = _MM_HINT_T0;
    for (auto _ : state)
    {
        for (int i = 0; i < N; i++)
        {
            int* a = &arr[i];
            for (int j = 0, stride = 0; j < N; j++, stride += N)
            {
                _mm_prefetch(a + (j + 1) * N, prefetch_type);
                _mm_prefetch(a + (j + 2) * N, prefetch_type);
                _mm_prefetch(a + (j + 3) * N, prefetch_type);
                _mm_prefetch(a + (j + 4) * N, prefetch_type);
                _mm_prefetch(a + (j + 5) * N, prefetch_type);
                _mm_prefetch(a + (j + 6) * N, prefetch_type);
                _mm_prefetch(a + (j + 7) * N, prefetch_type);
                _mm_prefetch(a + (j + 8) * N, prefetch_type);

                *(a + stride) += 1;
            }
        }
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
}

static void BM_WriteToArray(benchmark::State& state)
{
    int       thread_idx = state.thread_index();
    int       core_id    = map_thread_id_to_core_id(thread_idx);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    tracy_utils::CacheMissTracer l1_cache_miss_tracer(tracy_utils::Metric::L1DataCacheMissRate);
    tracy_utils::CacheMissTracer l2_cache_miss_tracer(tracy_utils::Metric::L2CacheMissRate);
    tracy_utils::CacheMissTracer llc_cache_miss_tracer(tracy_utils::Metric::LLCCacheMissRate);

    std::size_t         N = state.range(0);
    std::vector<double> a(N / sizeof(double));
    auto                size = a.size();

    static std::size_t last_memBytes = 0;
    if (N != last_memBytes)
    {
        last_memBytes = N;
        // Mark a new frame/zone in Tracy when the argument changes
        FrameMarkNamed("ArgChange");
        char msg[64];
        std::snprintf(msg, sizeof(msg), "Argument Changed: %zu", last_memBytes);
        TracyMessage(msg, std::strlen(msg));
    }

    benchmark::DoNotOptimize(a);
    for (auto _ : state)
    {
        {
            // ZoneScoped;
            for (int i = 0; i < size; i++)
            {
                a[i] = i;
            }
        }

        benchmark::ClobberMemory();

        // l1_cache_miss_tracer.update();
        // l2_cache_miss_tracer.update();
        // llc_cache_miss_tracer.update();
    }
    // add column to benchamrk results: operations per second
    state.counters["MemBW"] =
      benchmark::Counter(N * state.iterations(), benchmark::Counter::kIsRate);
    state.counters["OPS"] = benchmark::Counter(N * state.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_WriteToArray)
  ->Arg(4 * 1024)
  ->Arg(8 * 1024)
  ->Arg(16 * 1024)
  ->Arg(24 * 1024)
  ->Arg(48 * 1024) // L1 Cache size (9950x)
  ->Arg(52 * 1024)
  ->Arg(64 * 1024)
  ->Arg(72 * 1024)
  ->Arg(84 * 1024)
  ->Arg(96 * 1024)
  ->Arg(256 * 1024)
  ->Arg(384 * 1024)
  ->Arg(512 * 1024)
  ->Arg(640 * 1024)
  ->Arg(768 * 1024)
  ->Arg(896 * 1024)
  ->Arg(1024 * 1024) // L2 Cache size(9950x)
  ->Arg((256 + 1024) * 1024)
  ->Arg((512 + 1024) * 1024)
  ->Arg((768 + 1024) * 1024)
  ->Arg(2 * 1024 * 1024)
  ->Arg(4 * 1024 * 1024) // L3 Cache size(9950x)
  ->Arg(8 * 1024 * 1024)
  ->Arg(16 * 1024 * 1024)
  ->Threads(1);

// BENCHMARK(BM_ColOrderMatrixPrefetchRefactor)->Arg(NN);
//    BENCHMARK(BM_ColOrderMatrixPrefetchBadRefactor)->Arg(NN);

// BENCHMARK(BM_RowOrderMattrix)->Arg(NN);
// BENCHMARK(BM_ColOrderMattrix)->Arg(NN);
// BENCHMARK(BM_ColOrderMatrixPrefetch)->Arg(NN);

// BENCHMARK(BM_RowStride)->Arg(NN);
// BENCHMARK(BM_ColStride)->Arg(NN);
// BENCHMARK(BM_ColStridePrefetch)->Arg(NN);
BENCHMARK_MAIN();
