#include "mm/core/Shape.hpp"

#include <mm/core/utils/utils.hpp>
#include <mm/core/Matrix.hpp>

#include <benchmark/benchmark.h>
#include <mdspan>
#include <vector>
#include <thread>
#include <barrier>
#include <cstddef>

constexpr int GRID_I = 4;
constexpr int GRID_J = 8;

constexpr std::size_t Kc = 128;
constexpr std::size_t Mc = 512;
constexpr std::size_t Nc = 768;
constexpr std::size_t Mr = 8;
constexpr std::size_t Nr = 24;

constexpr std::size_t l1_cache_size_per_core = 48 * 1024;            // 48KB per core
constexpr std::size_t l2_cache_size_per_core = 1024 * 1024 + 8 * 10; // 1MB
constexpr std::size_t l3_cache_size_per_cpu  = 64 * 1024 * 1024;     // 64MB per cpu

static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");

template<typename T, std::size_t SizeBytes>
struct Cache
{
    static constexpr auto capacity = SizeBytes / sizeof(T);

    std::size_t ofs = 0;
    static_assert(SizeBytes % sizeof(T) == 0, "Size must be divisible by sizeof(T)");

    using aligned_vector = std::vector<T>;
    aligned_vector buf;

    Cache()
      : buf(capacity)
    {
    }

    template<Shape2D shape>
    std::mdspan<T, std::extents<std::size_t, shape[0], shape[1]>> allocate_tile()
    {
        auto old_ofs = ofs;
        ofs += shape[0] * shape[1];
        if (ofs >= capacity)
        {
            using namespace std::string_literals;
            throw std::runtime_error("Cache is full with capacity: "s + std::to_string(capacity)
                                     + " and exceeded by: " + std::to_string(ofs - capacity));
        }
        // std::cout << "allocated tile at " << capacity << " with ofs " << old_ofs << " with shape
        // "
        //           << shape[0] << "x" << shape[1] << " and size: " << ofs - old_ofs << std::endl;

        return std::mdspan<T, std::extents<std::size_t, shape[0], shape[1]>>(
          buf.data() + old_ofs, shape[0], shape[1]);
    }
    // TODO: Must allocate and retund mdspan
};

template<typename T, typename extend>
void load_tile(const Matrix<T>& M, std::mdspan<T, extend> tile, Shape2D shape)
{
    for (int i = 0; i < tile.extent(0); i++)
    {
        for (int j = 0; j < tile.extent(1); j++)
        {
            tile[i, j] = M(shape[0] + i, shape[1] + j);
        }
    }
}

template<typename T, typename tile_extend, typename utile_extend>
void copy_to_utile(std::mdspan<T, tile_extend>  tile,
                   std::mdspan<T, utile_extend> utile,
                   int                          row,
                   int                          col)
{
    for (int i = 0; i < utile.extent(0); i++)
    {
        for (int j = 0; j < utile.extent(1); j++)
        {
            utile[i, j] = tile[row + i, col + j];
        }
    }
}

template<typename T>
void matmul(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C)
{
    auto N = B.col();
    auto K = A.col();
    auto M = A.row();

    std::barrier sync_point(2);

    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");
    massert(N % Nc == 0, "N % Nc == 0");
    massert(N % GRID_J == 0, "N % GRID_J == 0");
    massert(M % GRID_I == 0, "M % GRID_I == 0");

    Cache<T, l1_cache_size_per_core> l1_cache;
    Cache<T, l2_cache_size_per_core> l2_cache;
    Cache<T, l3_cache_size_per_cpu>  l3_cache;

    auto b_utile = l1_cache.template allocate_tile<Shape2D{Kc, Nr}>();
    auto b_tile  = l3_cache.template allocate_tile<Shape2D{Kc, Nc}>();

    auto worker = [&](int t)
    {
        auto a_utile = l1_cache.template allocate_tile<Shape2D{Mr, Kc}>();
        auto a_tile  = l2_cache.template allocate_tile<Shape2D{Mc, Kc}>();

        // 'lscpu -e' to check core_id
        auto      core_id = t % 16;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        int i_ofst = core_id / GRID_J;
        int j_ofst = core_id % GRID_J;

        for (int j = j_ofst; j < j_ofst + N / GRID_J; j += Nc)
        {
            for (int k = 0; k < K; k += Kc)
            {
                if (t == 0)
                {
                    load_tile(B, b_tile, Shape2D{k, j});
                }
                else
                {
                    std::this_thread::yield();
                }
                sync_point.arrive_and_wait();

                for (int i = i_ofst; i < i_ofst + M / GRID_I; i += Mc)
                {
                    // Load tiles into L2
                    load_tile(A, a_tile, Shape2D{i, k});
                    for (int jj = 0; jj < Nc; jj += Nr)
                    {
                        for (int ii = 0; ii < Mc; ii += Mr)
                        {
                            // Load utiles into L1
                            copy_to_utile(a_tile, a_utile, ii, 0);
                            if (t == 0)
                            {
                                copy_to_utile(b_tile, b_utile, 0, jj);
                            }
                            else
                            {
                                std::this_thread::yield();
                            }
                            sync_point.arrive_and_wait();

                            for (int kk = 0; kk < Kc; ++kk)
                            {
                                for (int iir = 0; iir < Mr; ++iir)
                                {
                                    for (int jjr = 0; jjr < Nr; ++jjr)
                                    {
                                        C(i + ii + iir, j + jj + jjr) +=
                                          a_utile[iir, kk] * b_utile[jjr, kk];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto worker_thread = std::jthread(worker, 0);
    worker(16);
}

template<typename T>
void matmulNaive(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C)
{
    auto N = B.col();
    auto K = A.col();
    auto M = A.row();

    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");
    massert(N % Nc == 0, "N % Nc == 0");
    massert(N % GRID_J == 0, "N % GRID_J == 0");
    massert(M % GRID_I == 0, "M % GRID_I == 0");

    Cache<T, l1_cache_size_per_core> l1_cache;
    Cache<T, l2_cache_size_per_core> l2_cache;
    Cache<T, l3_cache_size_per_cpu>  l3_cache;

    auto a1_utile = l1_cache.template allocate_tile<Shape2D{Mr, Kc}>();
    auto a2_utile = l1_cache.template allocate_tile<Shape2D{Mr, Kc}>();
    auto b_utile  = l1_cache.template allocate_tile<Shape2D{Kc, Nr}>();
    auto a1_tile  = l2_cache.template allocate_tile<Shape2D{Mc, Kc}>();
    auto a2_tile  = l2_cache.template allocate_tile<Shape2D{Mc, Kc}>();
    auto b_tile   = l3_cache.template allocate_tile<Shape2D{Kc, Nc}>();

    auto worker = [&](int t)
    {
        // 'lscpu -e' to check core_id
        auto      core_id = t % 16;
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        auto i_ofst = t / GRID_J;
        auto j_ofst = t % GRID_J;

        for (int j = j_ofst; j < N / GRID_J; j += Nc)
        {
            for (int k = 0; k < K; k += Kc)
            {

                for (int i = i_ofst; i < M / GRID_I; i += Mc)
                {

                    for (int jj = 0; jj < Nc; jj += Nr)
                    {
                        for (int ii = 0; ii < Mc; ii += Mr)
                        {

                            // Compute within utile (register blocking)
                            for (int kk = 0; kk < Kc; ++kk)
                            { // inner k loop
                                for (int iir = 0; iir < Mr; ++iir)
                                {
                                    for (int jjr = 0; jjr < Nr; ++jjr)
                                    {
                                        C(i + ii + iir, j + jj + jjr) +=
                                          A(i + ii + iir, k + kk) * B(k + kk, j + jj + jjr);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto worker_thread = std::jthread(worker, 0);
    worker(16);
}

template<typename T>
void matmulMemCompute(Matrix<T>& A, Matrix<T>& B, Matrix<T>& C)
{
    auto N = B.col();
    auto K = A.col();
    auto M = A.row();

    std::barrier sync_point(2);

    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");
    massert(N % Nc == 0, "N % Nc == 0");
    massert(N % GRID_J == 0, "N % GRID_J == 0");
    massert(M % GRID_I == 0, "M % GRID_I == 0");

    Cache<T, l1_cache_size_per_core> l1_cache;
    Cache<T, l2_cache_size_per_core> l2_cache;
    Cache<T, l3_cache_size_per_cpu>  l3_cache;

    auto b_utile = l1_cache.template allocate_tile<Shape2D{Kc, Nr}>();
    auto b_tile  = l3_cache.template allocate_tile<Shape2D{Kc, Nc}>();

    auto worker = [&]<int t>()
    {
        auto a_utile = l1_cache.template allocate_tile<Shape2D{Mr, Kc}>();
        auto a_tile  = l2_cache.template allocate_tile<Shape2D{Mc, Kc}>();

        // 'lscpu -e' to check core_id
        constexpr auto core_id = t % 16;
        cpu_set_t      cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        constexpr int i_ofst = core_id / GRID_J;
        constexpr int j_ofst = core_id % GRID_J;

        for (int j = j_ofst; j < j_ofst + N / GRID_J; j += Nc)
        {
            for (int k = 0; k < K; k += Kc)
            {
                if constexpr (t == 0)
                {
                    load_tile(B, b_tile, Shape2D{k, j});
                }

                sync_point.arrive_and_wait();

                for (int i = i_ofst; i < i_ofst + M / GRID_I; i += Mc)
                {
                    // Load tiles into L2
                    load_tile(A, a_tile, Shape2D{i, k});
                    for (int jj = 0; jj < Nc; jj += Nr)
                    {
                        for (int ii = 0; ii < Mc; ii += Mr)
                        {
                            // Load utiles into L1
                            copy_to_utile(a_tile, a_utile, ii, 0);
                            if constexpr (t == 0)
                            {
                                copy_to_utile(b_tile, b_utile, 0, jj);
                            }
                            sync_point.arrive_and_wait();

                            for (int kk = 0; kk < Kc; ++kk)
                            {
                                for (int iir = 0; iir < Mr; ++iir)
                                {
                                    for (int jjr = 0; jjr < Nr; ++jjr)
                                    {
                                        C(i + ii + iir, j + jj + jjr) +=
                                          a_utile[iir, kk] * b_utile[jjr, kk];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    auto worker_thread = std::jthread(worker, 0);
    worker(16);
}

static void BM_Naive(benchmark::State& state)
{
    std::size_t N = state.range(0);
    std::size_t K = state.range(0);
    std::size_t M = state.range(0);

    auto A = Matrix<double>(M, K);
    auto B = Matrix<double>(K, N);
    auto C = Matrix<double>(M, N);

    for (auto _ : state)
    {
        matmulNaive(A, B, C);
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
    state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * N * N * N * state.iterations(), benchmark::Counter::kIsRate);
}

static void BM_Cache(benchmark::State& state)
{
    std::size_t N = state.range(0);
    std::size_t K = state.range(0);
    std::size_t M = state.range(0);

    auto A = Matrix<double>(M, K);
    auto B = Matrix<double>(K, N);
    auto C = Matrix<double>(M, N);

    for (auto _ : state)
    {
        matmul(A, B, C);
        benchmark::ClobberMemory(); // Prevent reordering across iterations
    }
    state.counters["GFLOPS"] =
      benchmark::Counter(2.0 * N * N * N * state.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Cache)->Arg(6 * 1024);
BENCHMARK(BM_Naive)->Arg(6 * 1024);
//->Arg(3 * 1024)->Arg(6 * 1024)->Arg(12 * 1024);
//    ->Arg(24 * 1024)
//    ->Arg(48 * 1024)
//    ->Arg(72 * 1024)
//    ->Arg(96 * 1024)
//    ->Arg(512 * 1024)
//    ->Arg(1024 * 1024)
//    ->Arg(32 * 1024 * 1024)
//    ->Arg(64 * 1024 * 1024);

BENCHMARK_MAIN();
