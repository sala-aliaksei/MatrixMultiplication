#include "mm/core/Matrix.hpp"
#include "mm/core/experimental_kernels.hpp"
#include "mm/core/zen5kernels.hpp"
#include "mm/core/utils/cpu.hpp"
#include "mm/matmul/zen5_constants.hpp"
#include "tracy_utils/tracy_cache_miss_counter.hpp"
#include <benchmark/benchmark.h>
#include <pthread.h>



constexpr int Kc = 80;


static void BM_CppGenericKern(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    std::size_t K        = N;
    auto        matrices = initDoubleMatrix(N, N, N);

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
    auto        matrices = initDoubleMatrix(N, N, N);

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
    std::size_t K        = N;
    auto        matrices = initDoubleMatrix(N, N, N);

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
    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        // Nr*Mr + Nr*Kc + Mr*Kc = Nr*Mr + Kc(Nr+Mr)
        kernels::packed_ukernel8x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_PackedKernel8x4Aregs(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(N, N, N);

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
    auto        matrices = initDoubleMatrix(N, N, N);

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
    auto        matrices = initDoubleMatrix(N, N, N);

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
    auto        matrices = initDoubleMatrix(N, N, N);

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
    std::size_t K        = N;

    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        kernels::cpp_generic_ukern<Nr, Mr, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr, K);
    }
}

static void BM_PackedKernel2x4(benchmark::State& state)
{
    constexpr int Nr = 2;
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        kernels::packed_ukernel2x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernelGeneric2x4(benchmark::State& state)
{
    constexpr int Nr = 2;
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        kernels::cpp_packed_kernel<2, 4, Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernel6x4(benchmark::State& state)
{
    constexpr int Nr = 6;
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        kernels::packed_ukernel6x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernel1x4(benchmark::State& state)
{
    constexpr int Nr = 1;
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        kernels::packed_ukernel1x4<Kc>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nr);
    }
}

static void BM_PackedKernel1x4_Simd(benchmark::State& state)
{
    std::size_t N        = state.range(0);
    auto        matrices = initDoubleMatrix(N, N, N);

    for (auto _ : state)
    {
        kernels::packed_ukernel1x4_simd<Kc>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), N);
    }
}

static void BM_NaiveBlockDynamicKc(benchmark::State& state)
{
    std::size_t memBytes = state.range(0);
    constexpr std::size_t Mr = mm::constants::MatMulZen5Config<double>::Mr;
    constexpr std::size_t Nr = mm::constants::MatMulZen5Config<double>::Nr;
    constexpr std::size_t Nc = mm::constants::MatMulZen5Config<double>::Nc;
    constexpr std::size_t Mc = mm::constants::MatMulZen5Config<double>::Mc;


    int Kc = memBytes / (Mr + Nr) / sizeof(double);

    auto          matrices = initDoubleMatrix(Mc, Nc, Kc);

    for (auto _ : state)
    {
        kernels::naive_block_dkc<Nr, Mr>(matrices.a.data(), matrices.b.data(), matrices.c.data(), Nc, Kc);
        benchmark::ClobberMemory();
    }

    std::size_t bytes_read = (Kc * Mr + Kc * Nr + Mr * Nr) * sizeof(double);
    std::size_t bytes_written = Mr * Nr * sizeof(double);
    std::size_t total_bytes = bytes_read + bytes_written;

    state.counters["FLOPS"] = benchmark::Counter(2*Kc * Mr * Nr * state.iterations(), benchmark::Counter::kIsRate);
    state.counters["MemBW"] = benchmark::Counter(total_bytes * state.iterations(), benchmark::Counter::kIsRate);
}


static void BM_PackedKernelZen5(benchmark::State& state)
{
    // Pin thread to specific core
    int       thread_idx = state.thread_index();
    int       core_id    = map_thread_id_to_core_id(thread_idx);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    std::size_t memBytes = state.range(0);
    constexpr std::size_t Nc = 96;
    constexpr std::size_t Mc = 96;
    constexpr std::size_t Mr = 8;
    constexpr std::size_t Nr = 24;
    int Kc = memBytes / (Mr + Nr) / sizeof(double);

    auto          matrices = initDoubleMatrix(Mc,Nc,Kc);

    static std::size_t last_memBytes = 0;
    if (memBytes != last_memBytes) {
        last_memBytes = memBytes;
        // Mark a new frame/zone in Tracy when the argument changes
        FrameMarkNamed("ArgChange");
        char msg[64];
        std::snprintf(msg, sizeof(msg), "Argument Changed: %zu", last_memBytes);
        TracyMessage(msg, std::strlen(msg));
    }

    tracy_utils::CacheMissTracer l1_cache_miss_tracer(tracy_utils::Metric::L1DataCacheMissRate);
    tracy_utils::CacheMissTracer l2_cache_miss_tracer(tracy_utils::Metric::L2CacheMissRate);
    tracy_utils::CacheMissTracer llc_cache_miss_tracer(tracy_utils::Metric::LLCCacheMissRate);
    tracy_utils::CacheMissTracer instr_tracer(tracy_utils::Metric::InstructionsRetired);
    for (auto _ : state)
    {
        ZoneScoped;
        kernels::zen5_packed_kernel<Nr, Mr>(
          matrices.a.data(), matrices.b.data(), matrices.c.data(), Nc, Kc);

          l1_cache_miss_tracer.update();
          l2_cache_miss_tracer.update();
          llc_cache_miss_tracer.update();
          instr_tracer.update();
          
        benchmark::ClobberMemory();
    }

    std::size_t bytes_read = (Kc * Mr + Kc * Nr + Mr * Nr) * sizeof(double);
    std::size_t bytes_written = Mr * Nr * sizeof(double);
    std::size_t total_bytes = bytes_read + bytes_written;

    state.counters["FLOPS"] = benchmark::Counter(2*Kc * Mr * Nr * state.iterations(), benchmark::Counter::kIsRate);
    state.counters["MemBW"] = benchmark::Counter(total_bytes * state.iterations(), benchmark::Counter::kIsRate);
}


// BENCHMARK(BM_NaiveBlockDynamicKc)
// ->Arg(1 * 1024)
// ->Arg(2 * 1024)
//   ->Arg(4 * 1024)
//   ->Arg(8 * 1024)
//   ->Arg(16 * 1024)
//   ->Arg(24 * 1024)
//   ->Arg(48 * 1024) // L1 Cache size (9950x)
//   ->Arg(52 * 1024)
//   ->Arg(64 * 1024)
//   ->Arg(72 * 1024)
//   ->Arg(84 * 1024)
//   ->Arg(96 * 1024)
//   ->Arg(256 * 1024)
//   ->Arg(384 * 1024)
//   ->Arg(512 * 1024)
//   ->Arg(640 * 1024)
//   ->Arg(768 * 1024)
//   ->Arg(896 * 1024)
//   ->Arg(1024 * 1024) // L2 Cache size(9950x)
//   ->Arg((256 + 1024) * 1024)
//   ->Arg((512 + 1024) * 1024)
//   ->Arg((768 + 1024) * 1024)
//   ->Arg(2 * 1024 * 1024)
//   ->Arg(4 * 1024 * 1024) // L3 Cache size(9950x)
//   ->Arg(8 * 1024 * 1024)
//   ->Arg(16 * 1024 * 1024)
//   ->Arg(20 * 1024 * 1024)
//   ->Arg(24 * 1024 * 1024)
//   ->ThreadPerCpu();
//   //->Threads(16);

BENCHMARK(BM_PackedKernelZen5)
->Arg(1 * 1024)
->Arg(2 * 1024)
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
  ->Arg(20 * 1024 * 1024)
  ->Arg(24 * 1024 * 1024)
//   ->ThreadPerCpu();
  ->Threads(1);

  BENCHMARK_MAIN();