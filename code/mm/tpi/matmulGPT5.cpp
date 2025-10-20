// matmul.cpp
// High-performance dense GEMM (row-major) for AMD Zen 5 with AVX-512 microkernels.
// - FP32/FP64 via templates (compile-time dtype)
// - 3-level blocking with packing (Mc, Nc, Kc)
// - Microkernel: MR=8 rows x NV=2 vectors across columns
//   - FP32 vector width W=16 → 32-column micro-tile
//   - FP64 vector width W=8  → 16-column micro-tile
// - Parallel outer loops over (Mc, Nc) tiles, per-thread packed buffers
// - Linux thread affinity (prefer spreading across physical cores)
// - Optional non-temporal final stores for C to reduce cache thrash
// - Scalar fallback if AVX-512 unavailable
// Build example:
// g++ -O3 -march=x86-64-v4 -mavx512f -mavx512dq -ffast-math matmul.cpp -o matmul
// With OpenMP:
// g++ -O3 -march=x86-64-v4 -mavx512f -mavx512dq -ffast-math -DUSE_OPENMP -fopenmp matmul.cpp -o
// matmul

/*
PROMT

Write production-quality C++20 code that implements a high-performance dense matrix multiplication C
= A × B optimized for AMD Zen 5 (Ryzen 9 9950X, 16 cores / 32 threads). Deliver a self-contained
project (single .cpp is fine) with:

# Requirements
- Data types: FP32 and FP64; choose at compile time (e.g., via template).
- Layout: row-major for A, B, C; support leading stride. Dimensions M×K times K×N → M×N, not assumed
to be multiples of tile sizes.
- SIMD: Use AVX-512 intrinsics for the inner microkernels:
  - FP32: __m512, loads/stores, FMA (vmulps/vaddps or fused).
  - FP64: __m512d equivalents.
  - Align loads/stores where possible; fall back gracefully for tails with masks.
- Blocking/packing:
  - 3-level blocking tuned for Zen 5 cache: (Mc, Nc, Kc) for L3/L2/L1.
  - Microkernel sizes (Mr×Nr) selected for register usage without spilling.
  - Pack A (Mc×Kc) and B (Kc×Nc) into contiguous, aligned buffers
  - Software prefetching into L1/L2 when beneficial.
  - Consider non-temporal stores for large N when write-back thrashes caches (make it toggleable).
- Parallelism:
  - Use std::jthread or OpenMP (toggle via macro) to parallelize outer loops over (Mc, Nc).
  - Implement thread pinning/affinity on Linux: pin one worker per hardware thread, prefer spreading
across cores before SMT. Expose an option to use only physical cores (16) or all hw threads (32).
  - NUMA: assume single socket, but keep memory allocation thread-friendly; allocate packed buffers
per thread to avoid false sharing.
- Memory:
  - 64B-aligned allocations (posix_memalign or aligned_alloc).
  - Avoid false sharing (pad per-thread accumulators/metadata to cache line).
- Correctness:
  - Reference naive (scalar) matmul for verification.
  - Compare max relative/absolute error for FP32/FP64; print a short report.
- CLI / config:
  - Command-line flags for M,N,K; threads; use_all_hw_threads (yes/no); block sizes; use_nt_store;
dtype=float|double.
  - Sensible defaults good for 3072×3072.
- Benchmarking:
  - Warmups + timed runs; report Gflop/s and time.
  - Optional: sweep threads and print a small table (1..32).
- Code hygiene:
  - Separate kernels/microkernels into static inline functions.
  - Minimal dependencies; compile with: g++/clang++ -O3 -march=x86-64-v4 -mavx512f -mavx512dq
-ffast-math -fopenmp (if OpenMP).
  - Guard intrinsics behind #ifdef __AVX512F__ and provide scalar fallback.
  - No UB; handle tails on M/N/K cleanly with masked ops.

# Deliverables
1) matmul.cpp implementing:
   - packA/packB
   - avx512 microkernel (Mr×Nr) for FP32 and FP64
   - blocked/packed GEMM driver with parallel outer loops
   - thread affinity helper (sched_setaffinity)
   - naive reference, correctness checker
   - benchmark harness (warmup + N runs), printing Gflop/s
2) Brief comments explaining chosen tile sizes and register usage.

# Tuning targets (reasonable starting points; make constants configurable)
- L1-resident microkernel: e.g., Mr=8 (FP64) or 16 (FP32), Nr=8..16 depending on registers.
- Kc sized to keep A/B panels hot in L1/L2.
- Mc,Nc sized for L2/L3 on Zen 5; provide defaults and allow overrides via CLI.

# Example run
./matmul --M 3072 --N 3072 --K 3072 --threads 32 --dtype double --use_nt_store 1

Print:
- Correctness summary (max abs/rel error)
- Best/avg Gflop/s, total time
- Config used (tile sizes, threads, NT stores on/off, dtype)

Provide the full compilable source in one code block.

*/

#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <unistd.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif

// ----------------------------------------------
// Utility: aligned allocation + RAII
// ----------------------------------------------
template<typename T>
struct AlignedDeleter
{
    void operator()(T* p) const noexcept
    {
        if (p)
            std::free(p);
    }
};

template<typename T>
using AlignedPtr = std::unique_ptr<T, AlignedDeleter<T>>;

template<typename T>
AlignedPtr<T> aligned_alloc_count(std::size_t count, std::size_t alignment = 64)
{
    void* p = nullptr;
#if defined(_ISOC11_SOURCE)
    p =
      std::aligned_alloc(alignment, ((count * sizeof(T) + alignment - 1) / alignment) * alignment);
    if (!p)
        return AlignedPtr<T>(nullptr);
#else
    if (posix_memalign(&p, alignment, count * sizeof(T)) != 0)
    {
        return AlignedPtr<T>(nullptr);
    }
#endif
    return AlignedPtr<T>(reinterpret_cast<T*>(p));
}

// ----------------------------------------------
// CPU affinity (Linux): pin threads to CPUs.
// Prefer spreading across physical cores first.
// Assumes symmetric HT: physical cores = cpus/2.
// ----------------------------------------------
static inline int get_online_cpu_count()
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 1;
}

static inline int pick_cpu_for_thread(int thread_idx, int num_threads, bool use_all_hw_threads)
{
    const int num_cpus = get_online_cpu_count();
    if (num_cpus <= 0)
        return 0;
    if (use_all_hw_threads)
    {
        // Spread across all hw threads first, then wrap.
        return thread_idx % num_cpus;
    }
    else
    {
        // Prefer physical cores: even CPUs assumed as first thread of each core.
        const int phys = std::max(1, num_cpus / 2);
        int       cpu  = (thread_idx % phys) * 2;
        if (cpu >= num_cpus)
            cpu = cpu % num_cpus;
        return cpu;
    }
}

static inline void pin_current_thread_to_cpu(int cpu_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    (void)sched_setaffinity(0, sizeof(cpu_set_t), &cpuset); // ignore errors gracefully
}

// ----------------------------------------------
// Config
// ----------------------------------------------
struct Config
{
    int M = 3072;
    int N = 3072;
    int K = 3072;

    int  threads            = std::max(1u, std::thread::hardware_concurrency());
    bool use_all_hw_threads = true; // true: 32 on 9950X; false: 16 (even CPUs)

    // Blocking
    int Mc = 256;
    int Nc = 512;
    int Kc = 512;

    // Microkernel shapes (fixed MR=8, NV=2 tuned for register usage)
    int MR = 8;
    int NV = 2; // number of vectors across N

    // Stores/preload
    bool use_nt_store      = false;
    int  prefetch_distance = 64; // bytes ahead (approx) for T0

    // Benchmarking
    int  warmup        = 2;
    int  iters         = 5;
    bool sweep_threads = false; // optional sweep 1..threads

    // Dtype
    enum DType
    {
        FLOAT32,
        FLOAT64
    } dtype = FLOAT64;

    // Correctness
    bool verify = true; // naive reference
};

// ----------------------------------------------
// CLI parsing (simple)
// ----------------------------------------------
static inline bool parse_bool(const std::string& s)
{
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static Config parse_cli(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i)
    {
        std::string key  = argv[i];
        auto        next = [&](int& dst)
        {
            if (i + 1 < argc)
                dst = std::atoi(argv[++i]);
        };
        auto nextb = [&](bool& dst)
        {
            if (i + 1 < argc)
                dst = parse_bool(argv[++i]);
        };
        if (key == "--M")
            next(cfg.M);
        else if (key == "--N")
            next(cfg.N);
        else if (key == "--K")
            next(cfg.K);
        else if (key == "--threads")
            next(cfg.threads);
        else if (key == "--use_all_hw_threads")
            nextb(cfg.use_all_hw_threads);
        else if (key == "--Mc")
            next(cfg.Mc);
        else if (key == "--Nc")
            next(cfg.Nc);
        else if (key == "--Kc")
            next(cfg.Kc);
        else if (key == "--MR")
            next(cfg.MR);
        else if (key == "--NV")
            next(cfg.NV);
        else if (key == "--use_nt_store")
            nextb(cfg.use_nt_store);
        else if (key == "--prefetch_distance")
            next(cfg.prefetch_distance);
        else if (key == "--warmup")
            next(cfg.warmup);
        else if (key == "--iters")
            next(cfg.iters);
        else if (key == "--sweep")
            nextb(cfg.sweep_threads);
        else if (key == "--verify")
            nextb(cfg.verify);
        else if (key == "--dtype")
        {
            if (i + 1 < argc)
            {
                std::string v = argv[++i];
                if (v == "float" || v == "fp32")
                    cfg.dtype = Config::FLOAT32;
                else if (v == "double" || v == "fp64")
                    cfg.dtype = Config::FLOAT64;
                else
                    std::cerr << "Unknown dtype: " << v << " (use float|double)\n";
            }
        }
    }
    cfg.MR = 8; // fixed MR to match microkernel specialization
    cfg.NV = 2; // fixed NV
    return cfg;
}

// ----------------------------------------------
// SIMD traits for float/double
// ----------------------------------------------
template<typename T>
struct SimdTraits
{
};

#ifdef __AVX512F__
template<>
struct SimdTraits<float>
{
    using Vec              = __m512;
    using Mask             = __mmask16;
    static constexpr int W = 16;
    static inline Vec    set1(float x)
    {
        return _mm512_set1_ps(x);
    }
    static inline Vec loadu(const float* p)
    {
        return _mm512_loadu_ps(p);
    }
    static inline void storeu(float* p, Vec v)
    {
        _mm512_storeu_ps(p, v);
    }
    static inline void mask_storeu(float* p, Mask m, Vec v)
    {
        _mm512_mask_storeu_ps(p, m, v);
    }
    static inline Vec fmadd(Vec a, Vec b, Vec c)
    {
        return _mm512_fmadd_ps(a, b, c);
    }
    static inline void stream(float* p, Vec v)
    {
        _mm512_stream_ps(p, v);
    }
    static inline void prefetch_T0(const void* p)
    {
        _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
    }
    static inline void prefetch_T1(const void* p)
    {
        _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T1);
    }
    static inline Mask tail_mask(int n)
    {
        return static_cast<Mask>((n >= W) ? 0xFFFF : ((n <= 0) ? 0 : ((1u << n) - 1u)));
    }
};
template<>
struct SimdTraits<double>
{
    using Vec              = __m512d;
    using Mask             = __mmask8;
    static constexpr int W = 8;
    static inline Vec    set1(double x)
    {
        return _mm512_set1_pd(x);
    }
    static inline Vec loadu(const double* p)
    {
        return _mm512_loadu_pd(p);
    }
    static inline void storeu(double* p, Vec v)
    {
        _mm512_storeu_pd(p, v);
    }
    static inline void mask_storeu(double* p, Mask m, Vec v)
    {
        _mm512_mask_storeu_pd(p, m, v);
    }
    static inline Vec fmadd(Vec a, Vec b, Vec c)
    {
        return _mm512_fmadd_pd(a, b, c);
    }
    static inline void stream(double* p, Vec v)
    {
        _mm512_stream_pd(p, v);
    }
    static inline void prefetch_T0(const void* p)
    {
        _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
    }
    static inline void prefetch_T1(const void* p)
    {
        _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T1);
    }
    static inline Mask tail_mask(int n)
    {
        return static_cast<Mask>((n >= W) ? 0xFF : ((n <= 0) ? 0 : ((1u << n) - 1u)));
    }
};
#else
// Scalar fallback
template<>
struct SimdTraits<float>
{
    using Vec              = float;
    using Mask             = uint16_t;
    static constexpr int W = 16;
    static inline Vec    set1(float x)
    {
        return x;
    }
    static inline Vec loadu(const float* p)
    {
        return *p;
    }
    static inline void storeu(float* p, Vec v)
    {
        *p = v;
    }
    static inline void mask_storeu(float* p, Mask, Vec v)
    {
        *p = v;
    }
    static inline Vec fmadd(Vec a, Vec b, Vec c)
    {
        return a * b + c;
    }
    static inline void stream(float* p, Vec v)
    {
        *p = v;
    }
    static inline void prefetch_T0(const void*) {}
    static inline void prefetch_T1(const void*) {}
    static inline Mask tail_mask(int)
    {
        return 0;
    }
};
template<>
struct SimdTraits<double>
{
    using Vec              = double;
    using Mask             = uint8_t;
    static constexpr int W = 8;
    static inline Vec    set1(double x)
    {
        return x;
    }
    static inline Vec loadu(const double* p)
    {
        return *p;
    }
    static inline void storeu(double* p, Vec v)
    {
        *p = v;
    }
    static inline void mask_storeu(double* p, Mask, Vec v)
    {
        *p = v;
    }
    static inline Vec fmadd(Vec a, Vec b, Vec c)
    {
        return a * b + c;
    }
    static inline void stream(double* p, Vec v)
    {
        *p = v;
    }
    static inline void prefetch_T0(const void*) {}
    static inline void prefetch_T1(const void*) {}
    static inline Mask tail_mask(int)
    {
        return 0;
    }
};
#endif

// ----------------------------------------------
// Packing
// A_pack layout: [k=0..Kc-1][r=0..Mc-1] → contiguous
// B_pack layout: [k=0..Kc-1][c=0..Nc-1] → contiguous
// This favors microkernel access: for each k, A rows are contiguous,
// and B row k has contiguous vectors across columns.
// ----------------------------------------------
template<typename T>
static inline void
pack_A_panel(const T* A, int lda, int i0, int k0, int Mc_eff, int Kc_eff, T* A_pack)
{
    for (int p = 0; p < Kc_eff; ++p)
    {
        const T* arow = A + (k0 + p); // column index in A
        T*       dst  = A_pack + p * Mc_eff;
        for (int r = 0; r < Mc_eff; ++r)
        {
            dst[r] = arow[(i0 + r) * lda];
        }
    }
}

template<typename T>
static inline void
pack_B_panel(const T* B, int ldb, int k0, int j0, int Kc_eff, int Nc_eff, T* B_pack)
{
    for (int p = 0; p < Kc_eff; ++p)
    {
        const T* brow = B + (k0 + p) * ldb + j0;
        T*       dst  = B_pack + p * Nc_eff;
        std::memcpy(dst, brow, sizeof(T) * Nc_eff);
    }
}

// ----------------------------------------------
// Microkernel: MR rows x (NV * W) columns, with masked tails.
// Accumulates onto existing C, using streaming stores only on final K-block.
// ----------------------------------------------
template<typename T>
static inline void microkernel_block(const T* A_pack,
                                     int      a_row_stride, // equals Mc_eff
                                     const T* B_pack,
                                     int      b_row_stride, // equals Nc_eff
                                     T*       C,
                                     int      ldc,
                                     int      kc_eff,
                                     int      mr_eff,        // rows <= MR
                                     int      num_full_vecs, // vectors fully covered
                                     int  tail_elems, // 0..W-1 elements in the last partial vector
                                     bool use_nt_store,
                                     bool is_final_k_block,
                                     int  prefetch_distance_bytes)
{

    using ST        = SimdTraits<T>;
    using Vec       = typename ST::Vec;
    using Mask      = typename ST::Mask;
    constexpr int W = ST::W;

#ifdef __AVX512F__
    // Accumulators: [row][vec]
    Vec       acc[8 * 4]; // upper bound if MR<=8 and NV<=4
    const int NV = num_full_vecs + (tail_elems > 0 ? 1 : 0);
    // Load C into accumulators
    for (int r = 0; r < mr_eff; ++r)
    {
        T*  cptr = C + r * ldc;
        int v    = 0;
        for (; v < num_full_vecs; ++v)
        {
            acc[r * NV + v] = ST::loadu(cptr + v * W);
        }
        if (tail_elems > 0)
        {
            Mask m = ST::tail_mask(tail_elems);
            // masked load emulated via loadu+mask if needed: we can load and rely on mask store
            // later.
            Vec tmp         = ST::loadu(cptr + v * W);
            acc[r * NV + v] = tmp;
        }
    }

    // K loop
    for (int p = 0; p < kc_eff; ++p)
    {
        const T* arow = A_pack + p * a_row_stride;
        const T* brow = B_pack + p * b_row_stride;

        // Prefetch next rows
        if (prefetch_distance_bytes > 0)
        {
            const char* pfA = reinterpret_cast<const char*>(arow) + prefetch_distance_bytes;
            const char* pfB = reinterpret_cast<const char*>(brow) + prefetch_distance_bytes;
            ST::prefetch_T0(pfA);
            ST::prefetch_T0(pfB);
        }

        // Load B vectors for this k
        Vec bvecs[4]; // NV<=4
        int v = 0;
        for (; v < num_full_vecs; ++v)
        {
            bvecs[v] = ST::loadu(brow + v * W);
        }
        Vec btail{};
        if (tail_elems > 0)
        {
            // Load full, but writes will be masked
            btail = ST::loadu(brow + v * W);
        }

        // For each row, broadcast A scalar and FMA with B vectors
        for (int r = 0; r < mr_eff; ++r)
        {
            Vec ab  = ST::set1(arow[r]);
            int idx = r * NV;
            int v2  = 0;
            for (; v2 < num_full_vecs; ++v2)
            {
                acc[idx + v2] = ST::fmadd(bvecs[v2], ab, acc[idx + v2]);
            }
            if (tail_elems > 0)
            {
                acc[idx + v2] = ST::fmadd(btail, ab, acc[idx + v2]);
            }
        }
    }

    // Store back to C
    for (int r = 0; r < mr_eff; ++r)
    {
        T*  cptr = C + r * ldc;
        int v    = 0;
        for (; v < num_full_vecs; ++v)
        {
            if (use_nt_store && is_final_k_block)
            {
                // Only stream when aligned and full-width to avoid penalties
                if (((reinterpret_cast<std::uintptr_t>(cptr + v * W)) & 63u) == 0u)
                {
                    ST::stream(cptr + v * W, acc[r * (num_full_vecs + (tail_elems ? 1 : 0)) + v]);
                }
                else
                {
                    ST::storeu(cptr + v * W, acc[r * (num_full_vecs + (tail_elems ? 1 : 0)) + v]);
                }
            }
            else
            {
                ST::storeu(cptr + v * W, acc[r * (num_full_vecs + (tail_elems ? 1 : 0)) + v]);
            }
        }
        if (tail_elems > 0)
        {
            Mask m = ST::tail_mask(tail_elems);
            ST::mask_storeu(cptr + v * W, m, acc[r * (num_full_vecs + 1) + v]);
        }
    }
#else
    // Scalar fallback: small inner loops
    for (int r = 0; r < mr_eff; ++r)
    {
        for (int v = 0; v < num_full_vecs * W + tail_elems; ++v)
        {
            // C[r, v] += sum_k A[r,k] * B[k,v]
            T sum = 0;
            for (int p = 0; p < kc_eff; ++p)
            {
                sum += A_pack[p * a_row_stride + r] * B_pack[p * b_row_stride + v];
            }
            C[r * ldc + v] += sum;
        }
    }
    (void)use_nt_store;
    (void)is_final_k_block;
    (void)prefetch_distance_bytes;
#endif
}

// ----------------------------------------------
// GEMM blocked/packed driver
// C[M×N] += A[M×K] * B[K×N]
// ----------------------------------------------
template<typename T>
struct ThreadLocalBuffers
{
    AlignedPtr<T> A_pack;
    AlignedPtr<T> B_pack;
    int           A_pack_elems = 0;
    int           B_pack_elems = 0;
    // Padding to avoid false sharing
    char pad[64];
};

template<typename T>
static void
gemm_blocked_packed(const T* A, int lda, const T* B, int ldb, T* C, int ldc, const Config& cfg)
{

    using ST        = SimdTraits<T>;
    constexpr int W = ST::W;

    const int M = cfg.M, N = cfg.N, K = cfg.K;
    const int Mc = cfg.Mc, Nc = cfg.Nc, Kc = cfg.Kc;
    const int MR       = cfg.MR;
    const int NV_fixed = cfg.NV;

    auto worker_fn = [&](int thread_idx, int num_threads)
    {
        // Pin thread
        int cpu = pick_cpu_for_thread(thread_idx, num_threads, cfg.use_all_hw_threads);
        pin_current_thread_to_cpu(cpu);

        // Allocate per-thread packing buffers (max tile size)
        ThreadLocalBuffers<T> tlb;
        tlb.A_pack_elems = Mc * Kc;
        tlb.B_pack_elems = Kc * Nc;
        tlb.A_pack       = aligned_alloc_count<T>(tlb.A_pack_elems, 64);
        tlb.B_pack       = aligned_alloc_count<T>(tlb.B_pack_elems, 64);
        T* A_pack        = tlb.A_pack.get();
        T* B_pack        = tlb.B_pack.get();

        // 2D tiling over (i0, j0). Simple static partition: interleave tiles among threads.
        int num_i       = (M + Mc - 1) / Mc;
        int num_j       = (N + Nc - 1) / Nc;
        int total_tiles = num_i * num_j;

        for (int tile = thread_idx; tile < total_tiles; tile += num_threads)
        {
            int tj = tile % num_j;
            int ti = tile / num_j;
            int i0 = ti * Mc;
            int j0 = tj * Nc;

            int Mc_eff = std::min(Mc, M - i0);
            int Nc_eff = std::min(Nc, N - j0);

            // Iterate over K panels
            for (int k0 = 0; k0 < K; k0 += Kc)
            {
                int Kc_eff = std::min(Kc, K - k0);
                // Pack B [Kc_eff x Nc_eff]
                pack_B_panel(B, ldb, k0, j0, Kc_eff, Nc_eff, B_pack);

                // For A micro-panels across Mc_eff
                for (int ii = 0; ii < Mc_eff; ii += MR)
                {
                    int mr_eff = std::min(MR, Mc_eff - ii);
                    // Pack A for [mr_eff x Kc_eff]: A_pack layout expects full Mc_eff.
                    // Packing for whole Mc_eff once can be faster; here, pack full Mc_eff per Kc
                    // and reuse across micro-rows. Simplicity: pack full Mc_eff once per Kc for
                    // this tile.
                    if (ii == 0)
                    {
                        pack_A_panel(A, lda, i0, k0, Mc_eff, Kc_eff, A_pack);
                    }

                    // N micro-tiles across NV vectors
                    for (int jj = 0; jj < Nc_eff; jj += NV_fixed * W)
                    {
                        int remaining_cols = Nc_eff - jj;
                        int num_full_vecs  = std::min(NV_fixed, remaining_cols / W);
                        int tail_elems     = 0;
                        if (num_full_vecs < NV_fixed)
                        {
                            tail_elems = remaining_cols - num_full_vecs * W;
                        }

                        // Pointers for this micro-tile
                        const T* A_pack_block = A_pack + ii; // base row offset within packed A
                        const T* B_pack_block = B_pack + jj; // base col offset within packed B
                        T*       C_block      = C + (i0 + ii) * ldc + (j0 + jj);

                        // Accumulate across the Kc_eff chunk in one microkernel call
                        bool is_final_k = (k0 + Kc_eff >= K);
                        microkernel_block<T>(A_pack_block,
                                             /*a_row_stride=*/Mc_eff,
                                             B_pack_block,
                                             /*b_row_stride=*/Nc_eff,
                                             C_block,
                                             ldc,
                                             Kc_eff,
                                             mr_eff,
                                             num_full_vecs,
                                             tail_elems,
                                             cfg.use_nt_store,
                                             is_final_k,
                                             cfg.prefetch_distance);
                    } // jj
                }     // ii
            }         // k0
        }             // tiles
    };

#ifdef USE_OPENMP
    // OpenMP parallel region
    const int T = std::max(1, cfg.threads);
#pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        // Pin each OpenMP worker
        int cpu = pick_cpu_for_thread(tid, nt, cfg.use_all_hw_threads);
        pin_current_thread_to_cpu(cpu);
        worker_fn(tid, nt);
    }
#else
    // std::jthread workers
    const int                 TH = std::max(1, cfg.threads);
    std::vector<std::jthread> threads;
    threads.reserve(TH);
    for (int t = 0; t < TH; ++t)
    {
        threads.emplace_back([&, t]() { worker_fn(t, TH); });
    }
    // join on destructor
#endif
}

// ----------------------------------------------
// Naive reference matmul (row-major): C = A*B + C
// ----------------------------------------------
template<typename T>
static void naive_gemm(const T* A, int lda, const T* B, int ldb, T* C, int ldc, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            T        a    = A[i * lda + k];
            const T* brow = B + k * ldb;
            T*       crow = C + i * ldc;
            for (int j = 0; j < N; ++j)
            {
                crow[j] += a * brow[j];
            }
        }
    }
}

// ----------------------------------------------
// Correctness check: max abs/rel error
// ----------------------------------------------
template<typename T>
static std::pair<T, T> max_abs_rel_error(const T* ref, const T* got, int M, int N)
{
    T max_abs = T(0), max_rel = T(0);
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            T r    = ref[i * N + j];
            T g    = got[i * N + j];
            T diff = std::abs(r - g);
            T rel  = diff / std::max<T>(T(1e-12), std::abs(r));
            if (diff > max_abs)
                max_abs = diff;
            if (rel > max_rel)
                max_rel = rel;
        }
    }
    return {max_abs, max_rel};
}

// ----------------------------------------------
// Benchmark harness
// ----------------------------------------------
template<typename T>
static void run_benchmark(const Config& cfg)
{
    const int M = cfg.M, N = cfg.N, K = cfg.K;
    const int lda = K, ldb = N, ldc = N;

    auto A     = aligned_alloc_count<T>(static_cast<std::size_t>(M) * K, 64);
    auto B     = aligned_alloc_count<T>(static_cast<std::size_t>(K) * N, 64);
    auto C     = aligned_alloc_count<T>(static_cast<std::size_t>(M) * N, 64);
    auto C_ref = aligned_alloc_count<T>(static_cast<std::size_t>(M) * N, 64);

    if (!A || !B || !C || !C_ref)
    {
        std::cerr << "Allocation failed\n";
        return;
    }

    // Init random
    std::mt19937                           rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < M * K; ++i)
        A.get()[i] = static_cast<T>(dist(rng));
    for (int i = 0; i < K * N; ++i)
        B.get()[i] = static_cast<T>(dist(rng));

    // Optional correctness
    if (cfg.verify)
    {
        std::fill(C_ref.get(), C_ref.get() + M * N, T(0));
        naive_gemm<T>(A.get(), lda, B.get(), ldb, C_ref.get(), ldc, M, N, K);
    }

    // Warmup + timed runs
    auto do_run = [&](int threads)
    {
        Config local  = cfg;
        local.threads = threads;
        std::fill(C.get(), C.get() + M * N, T(0));
        // Warmups
        for (int w = 0; w < cfg.warmup; ++w)
        {
            gemm_blocked_packed<T>(A.get(), lda, B.get(), ldb, C.get(), ldc, local);
        }
        // Timed
        double best = 1e100, sum = 0.0;
        for (int it = 0; it < cfg.iters; ++it)
        {
            std::fill(C.get(), C.get() + M * N, T(0));
            auto t0 = std::chrono::high_resolution_clock::now();
            gemm_blocked_packed<T>(A.get(), lda, B.get(), ldb, C.get(), ldc, local);
            auto   t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            best      = std::min(best, ms);
            sum += ms;
        }
        double avg         = sum / cfg.iters;
        double flops       = 2.0 * double(M) * double(N) * double(K);
        double gflops_best = flops / (best / 1000.0) / 1e9;
        double gflops_avg  = flops / (avg / 1000.0) / 1e9;
        return std::make_tuple(best, avg, gflops_best, gflops_avg);
    };

    if (cfg.sweep_threads)
    {
        std::cout << "Thread sweep (1.." << cfg.threads << ")\n";
        std::cout << "threads,best_ms,avg_ms,best_gflops,avg_gflops\n";
        for (int t = 1; t <= cfg.threads; ++t)
        {
            auto [best, avg, gbest, gavg] = do_run(t);
            std::cout << t << "," << best << "," << avg << "," << gbest << "," << gavg << "\n";
        }
    }
    else
    {
        auto [best, avg, gbest, gavg] = do_run(cfg.threads);
        // Correctness
        double max_abs = 0, max_rel = 0;
        if (cfg.verify)
        {
            auto errs = max_abs_rel_error<T>(C_ref.get(), C.get(), M, N);
            max_abs   = errs.first;
            max_rel   = errs.second;
        }

        std::cout << "Correctness (max abs, max rel): ";
        if (cfg.verify)
            std::cout << max_abs << ", " << max_rel << "\n";
        else
            std::cout << "skipped\n";

        std::cout << "Best Gflop/s: " << gbest << "  Avg Gflop/s: " << gavg
                  << "  Best time (ms): " << best << "  Avg time (ms): " << avg << "\n";
        std::cout << "Config: "
                  << "M=" << M << " N=" << N << " K=" << K << " threads=" << cfg.threads
                  << " dtype=" << (std::is_same<T, float>::value ? "float" : "double")
                  << " Mc=" << cfg.Mc << " Nc=" << cfg.Nc << " Kc=" << cfg.Kc << " MR=" << cfg.MR
                  << " NV=" << cfg.NV << " nt_store=" << (cfg.use_nt_store ? "on" : "off")
#ifdef USE_OPENMP
                  << " parallel=OpenMP"
#else
                  << " parallel=std::jthread"
#endif
                  << "\n";
    }
}

// ----------------------------------------------
// Main
// ----------------------------------------------
int main(int argc, char** argv)
{
    Config cfg = parse_cli(argc, argv);

    if (cfg.M <= 0 || cfg.N <= 0 || cfg.K <= 0)
    {
        std::cerr << "Invalid dimensions\n";
        return 1;
    }
    if (cfg.Mc <= 0 || cfg.Nc <= 0 || cfg.Kc <= 0)
    {
        std::cerr << "Invalid block sizes\n";
        return 1;
    }

    std::cout << "AMD Zen 5 GEMM (AVX-512) — "
              << "Use all HW threads: " << (cfg.use_all_hw_threads ? "yes" : "no") << "\n";

    if (cfg.dtype == Config::FLOAT32)
    {
        run_benchmark<float>(cfg);
    }
    else
    {
        run_benchmark<double>(cfg);
    }
    return 0;
}