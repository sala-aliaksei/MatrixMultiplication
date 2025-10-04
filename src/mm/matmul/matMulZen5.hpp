#pragma once

#include <mm/core/Matrix.hpp>
#include <mm/core/zen5kernels.hpp>
#include <mm/core/reorderMatrix.hpp>
#include <mm/core/bf16kernel.hpp>
#include <mm/core/utils/utils.hpp>
#include <mm/core/layout.hpp>
#include <mm/core/utils/algorithms.hpp>

#include <mdspan>
#include <thread>
#include <algorithm>
#include <omp.h>
#include <array>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace mm::zen5
{
constexpr int PAGE_SIZE = 4096;

template<typename T>
struct MatMulZen5Config;

template<>
struct MatMulZen5Config<double>
{
    // static constexpr int Nc = 96;
    // static constexpr int Mc = 96;
    // static constexpr int Kc = 96 * 2 * 2;
    //
    static constexpr int Nc = 96;
    static constexpr int Mc = 96;
    static constexpr int Kc = 96 * 2;
};

template<>
struct MatMulZen5Config<float>
{
    static constexpr int Nc = 96 * 2;
    static constexpr int Mc = 96;
    static constexpr int Kc = 96 * 2 * 2;
};

template<>
struct MatMulZen5Config<std::bfloat16_t>
{
    static constexpr int Nc = 96 * 2;
    static constexpr int Mc = 96;
    static constexpr int Kc = 96 * 2 * 2;
};

template<typename T>
void matMulZen5(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    constexpr int Nc = 96; // 3072/32=96
    constexpr int Mc = 96;
    constexpr int Kc = 96 * 2 * 2;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;

    constexpr int Kr = 1;
    constexpr int Nr = bregs_cnt * num_of_elems_in_reg;
    constexpr int Mr{8}; //{(num_of_regs - aregs_cnt - bregs_cnt) / bregs_cnt};

    //  creg_cnt = (bregs_cnt + aregs_cnt) * Mr
    //          -------------------------
    //          | breg1 | breg2 | breg3 |
    //          -------------------------
    // cregs    | breg1 | breg2 | breg3 |   areg
    //          .........................
    //          | breg1 | breg2 | breg3 |
    //          -------------------------

    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    auto num_threads = std::thread::hardware_concurrency();
    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const int N = static_cast<int>(B.col());
    const int K = static_cast<int>(A.col());
    const int M = static_cast<int>(A.row());

    //

    massert(N % Nc == 0, "N % Nc == 0");
    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");
    massert(N % num_threads == 0, "N % num_threads == 0");
    massert((N / num_threads) % Nc == 0, "(N/num_threads) % Nc == 0");

    std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>> buffer(num_threads * Kc
                                                                             * (Mc + Nc));

#pragma omp parallel for num_threads(num_threads)
    for (int j_block = 0; j_block < N; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        T*         buf = buffer.data() + ofs;

        for (int k_block = 0; k_block < K; k_block += Kc)
        {
            reorderRowMajorMatrixAVX<Kc, Nc, Kr, Nr>(
              B.data() + N * k_block + j_block, N, buf + Mc * Kc);

            for (int i_block = 0; i_block < M; i_block += Mc)
            {
                // all threads should access same memory
                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K, buf);

                for (int j = 0; j < Nc; j += Nr)
                {
                    const T* Bc1 = buf + Mc * Kc + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        T*       Cc0 = C.data() + N * i_block + j + N * i + j_block;
                        const T* Ac0 = buf + Kc * i;

                        if constexpr (std::is_same_v<T, bf16_std>)
                        {
                            kernels::zen5_packed_kernel_bf16<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        }
                        else
                        {
                            kernels::zen5_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        }
                    }
                }
            }
        }
    }
}

constexpr int map_thread_id_to_core_id(int n)
{
    // // Check if the input is within the specified range [0, 31].
    // if (n < 0 || n > 31) {
    //     // Return an error code or handle as appropriate for out-of-range input.
    //     return -1;
    // }

    if ((n & 1) == 0)
    {                  // If n is even (least significant bit is 0)
        return n >> 1; // Equivalent to n / 2
    }
    else
    {                         // If n is odd
        return 16 + (n >> 1); // Equivalent to 16 + (n - 1) / 2
    }
}

template<typename T>
void matMulZen5MTBlocking(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{

    constexpr int Nc = MatMulZen5Config<T>::Nc;
    constexpr int Mc = MatMulZen5Config<T>::Mc;
    constexpr int Kc = MatMulZen5Config<T>::Kc;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;

    constexpr int Nr{bregs_cnt * num_of_elems_in_reg};
    constexpr int Mr{8};
    constexpr int Kr{1};

    static_assert(Mc % Mr == 0, "invalid Mc cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid Nc cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid Kc cache/reg size of the block");

    const int N = static_cast<int>(B.col());
    const int K = static_cast<int>(A.col());
    const int M = static_cast<int>(A.row());

    massert(N % Nc == 0, "N % Nc == 0");
    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");

    // Fixed thread grid 4x8 → 32 threads
    constexpr int      GRID_I      = 4;
    constexpr int      GRID_J      = 8;
    constexpr unsigned num_threads = GRID_I * GRID_J;

    std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>> buffer(num_threads * Kc
                                                                             * (Mc + Nc));

    // Square-chunking in block units
    constexpr int ChunkBlocks = 2; // tiles per side in a chunk (Mc/Nc multiples)
    const int     blocksI     = M / Mc;
    const int     blocksJ     = N / Nc;
    const int     chunkRows   = (blocksI + ChunkBlocks - 1) / ChunkBlocks;
    const int     chunkCols   = (blocksJ + ChunkBlocks - 1) / ChunkBlocks;

    std::vector<std::jthread> workers;
    workers.reserve(num_threads);

    for (unsigned t = 0; t < num_threads; ++t)
    {
        workers.emplace_back(
          [&, t]()
          {
              auto      core_id = map_thread_id_to_core_id(t);
              cpu_set_t cpuset;
              CPU_ZERO(&cpuset);
              CPU_SET(core_id, &cpuset);
              (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

              const std::size_t ofs = t * Kc * (Mc + Nc);
              T* const          buf = buffer.data() + ofs;

              // Thread's grid coords
              const int ti = static_cast<int>(t) / GRID_J; // 0..GRID_I-1
              const int tj = static_cast<int>(t) % GRID_J; // 0..GRID_J-1

              // Each thread walks its subset of chunks with strides GRID_I/GRID_J
              for (int chi = ti; chi < chunkRows; chi += GRID_I)
              {
                  for (int chj = tj; chj < chunkCols; chj += GRID_J)
                  {
                      const int ibegin = chi * ChunkBlocks;
                      const int iend   = std::min(ibegin + ChunkBlocks, blocksI);
                      const int jbegin = chj * ChunkBlocks;
                      const int jend   = std::min(jbegin + ChunkBlocks, blocksJ);

                      for (int k_block = 0; k_block < K; k_block += Kc)
                      {
                          // For each j_block in the chunk, pack B once and reuse across all i_block
                          for (int jb = jbegin; jb < jend; ++jb)
                          {
                              const int j_block = jb * Nc;

                              reorderRowMajorMatrixAVX<Kc, Nc, Kr, Nr>(
                                B.data() + N * k_block + j_block, N, buf + Mc * Kc);

                              for (int ib = ibegin; ib < iend; ++ib)
                              {
                                  const int i_block = ib * Mc;

                                  reorderColOrderMatrix<Mc, Kc, Mr, Kr>(
                                    A.data() + K * i_block + k_block, K, buf);

                                  for (int j = 0; j < Nc; j += Nr)
                                  {
                                      const T* Bc1 = buf + Mc * Kc + Kc * j;
                                      for (int i = 0; i < Mc; i += Mr)
                                      {
                                          T* Cc0 = C.data() + N * i_block + j + N * i + j_block;
                                          const T* Ac0 = buf + Kc * i;
                                          if constexpr (std::is_same_v<T, std::bfloat16_t>)
                                          {
                                              kernels::zen5_packed_kernel_bf16<Nr, Mr, Kc>(
                                                Ac0, Bc1, Cc0, N);
                                          }
                                          else
                                          {
                                              kernels::zen5_packed_kernel<Nr, Mr, Kc>(
                                                Ac0, Bc1, Cc0, N);
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          });
    } // jthreads auto-join on destruction
}

template<typename T>
void matMulZen5MTBlockingTails(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    constexpr int Nc = 96;
    constexpr int Mc = 96;
    constexpr int Kc = 96 * 2;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;

    constexpr int Kr = 1;
    constexpr int Nr = bregs_cnt * num_of_elems_in_reg; // 24
    constexpr int Mr{8}; //{(num_of_regs - aregs_cnt - bregs_cnt) / bregs_cnt};

    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    auto num_threads = std::thread::hardware_concurrency();
    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const int N = static_cast<int>(B.col());
    const int K = static_cast<int>(A.col());
    const int M = static_cast<int>(A.row());

    //
    // Minimal per-tail padding during repacking enables arbitrary M/N/K
    const std::size_t per_thread_buf_elems =
      static_cast<std::size_t>(Kc) * (Mc + Nc) + static_cast<std::size_t>(Mc) * Nc;
    std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>> buffer(
      num_threads * per_thread_buf_elems);

    // Grid threading like matMulZen5MTBlocking
    constexpr int      GRID_I       = 4;
    constexpr int      GRID_J       = 8;
    constexpr unsigned grid_threads = GRID_I * GRID_J;

    // Square-chunking in block units; use ceil for tail tiles
    constexpr int ChunkBlocks = 2;
    const int     blocksI     = (M + Mc - 1) / Mc;
    const int     blocksJ     = (N + Nc - 1) / Nc;
    const int     chunkRows   = (blocksI + ChunkBlocks - 1) / ChunkBlocks;
    const int     chunkCols   = (blocksJ + ChunkBlocks - 1) / ChunkBlocks;

    auto worker_fn = [&](unsigned t)
    {
        const std::size_t ofs  = static_cast<std::size_t>(t) * per_thread_buf_elems;
        T* const          buf  = buffer.data() + ofs;
        T* const          bufA = buf;
        T* const          bufB = buf + Mc * Kc;
        T* const          bufC = buf + static_cast<std::size_t>(Kc) * (Mc + Nc);

        const int ti = static_cast<int>(t) / GRID_J;
        const int tj = static_cast<int>(t) % GRID_J;

        for (int chi = ti; chi < chunkRows; chi += GRID_I)
        {
            for (int chj = tj; chj < chunkCols; chj += GRID_J)
            {
                const int ibegin = chi * ChunkBlocks;
                const int iend   = std::min(ibegin + ChunkBlocks, blocksI);
                const int jbegin = chj * ChunkBlocks;
                const int jend   = std::min(jbegin + ChunkBlocks, blocksJ);

                for (int jb = jbegin; jb < jend; ++jb)
                {
                    const int j_block   = jb * Nc;
                    const int N_blk     = std::min(Nc, N - j_block);
                    const int N_blk_pad = blockWithPadding(N_blk, Nr);
                    for (int k_block = 0; k_block < K; k_block += Kc)
                    {
                        const int K_blk = std::min(Kc, K - k_block);

                        reorderRowMajorMatrixPadded<Kc, Nc, Kr, Nr>(
                          B.data() + static_cast<std::size_t>(N) * k_block + j_block,
                          N,
                          bufB,
                          K_blk,
                          N_blk);

                        for (int ib = ibegin; ib < iend; ++ib)
                        {
                            const int i_block   = ib * Mc;
                            const int M_blk     = std::min(Mc, M - i_block);
                            const int M_blk_pad = blockWithPadding(M_blk, Mr);

                            reorderColOrderMatrixPadded<Mc, Kc, Mr, Kr>(
                              A.data() + static_cast<std::size_t>(K) * i_block + k_block,
                              K,
                              bufA,
                              M_blk,
                              K_blk);

                            const std::size_t c_tile_elems =
                              static_cast<std::size_t>(M_blk_pad) * N_blk_pad;
                            std::fill(bufC, bufC + c_tile_elems, T(0));

                            for (int j = 0; j < N_blk_pad; j += Nr)
                            {
                                const T* Bc1 = bufB + Kc * j;
                                for (int i = 0; i < M_blk_pad; i += Mr)
                                {
                                    T* Cc0 = bufC + static_cast<std::size_t>(N_blk_pad) * i + j;
                                    const T* Ac0 = bufA + Kc * i;

                                    if constexpr (std::is_same_v<T, std::bfloat16_t>)
                                    {
                                        kernels::zen5_packed_kernel_bf16<Nr, Mr, Kc>(
                                          Ac0, Bc1, Cc0, N_blk_pad);
                                    }
                                    else
                                    {
                                        kernels::zen5_packed_kernel<Nr, Mr, Kc>(
                                          Ac0, Bc1, Cc0, N_blk_pad);
                                    }
                                }
                            }

                            for (int ii = 0; ii < M_blk; ++ii)
                            {
                                T* c_row =
                                  C.data() + static_cast<std::size_t>(N) * (i_block + ii) + j_block;
                                const T* tile_row = bufC + static_cast<std::size_t>(N_blk_pad) * ii;
                                for (int jj = 0; jj < N_blk; ++jj)
                                {
                                    c_row[jj] += tile_row[jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<std::jthread> workers;
    workers.reserve(grid_threads - 1);
    for (unsigned t = 0; t + 1 < grid_threads; ++t)
    {
        workers.emplace_back([&, t]() { worker_fn(t); });
    }
    worker_fn(grid_threads - 1);
}

template<typename T>
void matMulZen5MTBlockingSpan(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) noexcept
{
    using namespace mm::core;
    constexpr int Nc = MatMulZen5Config<T>::Nc;
    constexpr int Mc = MatMulZen5Config<T>::Mc;
    constexpr int Kc = MatMulZen5Config<T>::Kc;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;

    constexpr int Nr{bregs_cnt * num_of_elems_in_reg};
    constexpr int Mr{8};
    constexpr int Kr{1};

    static_assert(Mc % Mr == 0, "invalid Mc cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid Nc cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid Kc cache/reg size of the block");

    const int N = static_cast<int>(B.col());
    const int K = static_cast<int>(A.col());
    const int M = static_cast<int>(A.row());

    massert(N % Nc == 0, "N % Nc == 0");
    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");

    // Fixed thread grid 4x8 → 32 threads
    constexpr int      GRID_I      = 4;
    constexpr int      GRID_J      = 8;
    constexpr unsigned num_threads = GRID_I * GRID_J;

    constexpr auto tiles_size = Kc * (Mr + Nr);

    using b_tile_ext_t = std::extents<std::size_t, Nc / Nr, Nr * Kc>;
    using a_tile_ext_t = std::extents<std::size_t, Mc / Mr, Mr * Kc>;

    // static_assert(
    //     b_tile.static_extent(1)
    //       == b_utile.static_extent(1) * b_utile.static_extent(0),
    //     "b_tile.static_extent(1) != b_utile.static_extent(1) * "
    //     "b_utile.static_extent(0)");

    //   static_assert(
    //     a_tile.static_extent(1)
    //       == a_utile.static_extent(1) * a_utile.static_extent(0),
    //     "a_tile.static_extent(1) != a_utile.static_extent(1) * "
    //     "b_utile.static_extent(0)");

    // static_assert(b_utile.static_extent(0) == Kc, "b_utile.static_extent(0) != Kc");
    // static_assert(b_utile.static_extent(1) == Nr, "b_utile.static_extent(1) != Nr");
    // static_assert(a_utile.static_extent(0) == Kc, "a_utile.static_extent(0) != Kc");
    // static_assert(a_utile.static_extent(1) == Mr, "a_utile.static_extent(1) != Mr");

    std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>> buffer(num_threads
                                                                             * tiles_size);

    // Square-chunking in block units
    constexpr int ChunkBlocks = 2; // tiles per side in a chunk (Mc/Nc multiples)
    const int     blocksI     = M / Mc;
    const int     blocksJ     = N / Nc;
    const int     chunkRows   = (blocksI + ChunkBlocks - 1) / ChunkBlocks;
    const int     chunkCols   = (blocksJ + ChunkBlocks - 1) / ChunkBlocks;

    std::vector<std::jthread> workers;
    workers.reserve(num_threads);

    // static_for<0, num_threads>(
    //   [&]<int t>()
    for (int t = 0; t < num_threads; ++t)
    {
        workers.emplace_back(
          [&, t]()
          {
              // core id will be the same for threads which share resources
              auto core_id = map_thread_id_to_core_id(t);

              cpu_set_t cpuset;
              CPU_ZERO(&cpuset);
              CPU_SET(core_id, &cpuset);
              (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

              T* const buf = buffer.data() + t * tiles_size;

              std::mdspan<T, std::extents<std::size_t, Kc, Mr>> a_utile(buf, Kc, Mr);
              std::mdspan<T, std::extents<std::size_t, Kc, Nr>> b_utile(
                buf + a_utile.size(), Kc, Nr);

              // Thread's grid coords
              int ti = static_cast<int>(t) / GRID_J; // 0..GRID_I-1
              int tj = static_cast<int>(t) % GRID_J; // 0..GRID_J-1

              for (int chi = ti; chi < chunkRows; chi += GRID_I)
              {
                  for (int chj = tj; chj < chunkCols; chj += GRID_J)
                  {
                      const int ibegin = chi * ChunkBlocks;
                      const int iend   = std::min(ibegin + ChunkBlocks, blocksI);
                      const int jbegin = chj * ChunkBlocks;
                      const int jend   = std::min(jbegin + ChunkBlocks, blocksJ);

                      for (int k_block = 0; k_block < K; k_block += Kc)
                      {
                          for (int jb = jbegin; jb < jend; ++jb)
                          {
                              const int j_block = jb * Nc;

                              typename layout_blocked_colmajor<Nr>::mapping b_mapping(
                                b_tile_ext_t{}, M, N, k_block, j_block);
                              std::mdspan b_tile(B.data(), b_mapping);

                              for (int ib = ibegin; ib < iend; ++ib)
                              {
                                  const int i_block = ib * Mc;

                                  typename layout_microtile_colorder<Mr, Nr>::mapping a_mapping(
                                    a_tile_ext_t{}, M, N, i_block, k_block);
                                  std::mdspan a_tile(A.data(), a_mapping);

                                  for (int j = 0; j < Nc; j += Nr)
                                  {
                                      int tile_idx{0};
                                      for (int kdx = 0; kdx < b_utile.static_extent(0); kdx++)
                                      {
                                          for (int jdx = 0; jdx < b_utile.static_extent(1); jdx++)
                                          {
                                              b_utile[kdx, jdx] = b_tile[j / Nr, tile_idx++];
                                          }
                                      }

                                      for (int i = 0; i < Mc; i += Mr)
                                      {
                                          // Retrive next a_utile
                                          int tile_idx{0};
                                          // int tile_col;
                                          for (int kdx = 0; kdx < a_utile.static_extent(0); kdx++)
                                          {
                                              for (int idx = 0; idx < a_utile.static_extent(1);
                                                   idx++)
                                              {
                                                  a_utile[kdx, idx] = a_tile[i / Mr, tile_idx++];
                                              }
                                          }

                                          T* Cc0 = &C(i_block + i, j_block + j);
                                          kernels::zen5_mdspan_kernel(a_utile, b_utile, Cc0, N);
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          });
    }
    //});
}

template<typename T>
void matMulZen5MTBlockingL1(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) noexcept
{
    using namespace mm::core;
    constexpr std::size_t Nc = MatMulZen5Config<T>::Nc;
    constexpr std::size_t Mc = MatMulZen5Config<T>::Mc;
    constexpr std::size_t Kc = MatMulZen5Config<T>::Kc;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;

    constexpr int Nr{bregs_cnt * num_of_elems_in_reg};
    constexpr int Mr{8};
    constexpr int Kr{1};

    static_assert(Mc % Mr == 0, "invalid Mc cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid Nc cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid Kc cache/reg size of the block");

    const int N = static_cast<int>(B.col());
    const int K = static_cast<int>(A.col());
    const int M = static_cast<int>(A.row());

    massert(N % Nc == 0, "N % Nc == 0");
    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");

    // Fixed thread grid 4x8 → 32 threads
    constexpr int      GRID_I      = 4;
    constexpr int      GRID_J      = 8;
    constexpr unsigned num_threads = GRID_I * GRID_J;

    constexpr auto tiles_size = Kc * (Mr + Nr);

    std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>> buffer(num_threads
                                                                             * tiles_size);

    // Square-chunking in block units
    constexpr int ChunkBlocks = 2; // tiles per side in a chunk (Mc/Nc multiples)
    const int     blocksI     = M / Mc;
    const int     blocksJ     = N / Nc;
    const int     chunkRows   = (blocksI + ChunkBlocks - 1) / ChunkBlocks;
    const int     chunkCols   = (blocksJ + ChunkBlocks - 1) / ChunkBlocks;

    std::vector<std::jthread> workers;
    workers.reserve(num_threads);

    // static_for<0, num_threads>(
    //   [&]<int t>()
    for (int t = 0; t < num_threads; ++t)
    {
        workers.emplace_back(
          [&, t]()
          {
              // core id will be the same for threads which share resources
              auto core_id = map_thread_id_to_core_id(t);

              cpu_set_t cpuset;
              CPU_ZERO(&cpuset);
              CPU_SET(core_id, &cpuset);
              (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

              T* const buf = buffer.data() + t * tiles_size;

              std::mdspan<T, std::extents<std::size_t, Kc, Mr>> a_utile(buf, Kc, Mr);
              std::mdspan<T, std::extents<std::size_t, Kc, Nr>> b_utile(
                buf + a_utile.size(), Kc, Nr);

              // Thread's grid coords
              int ti = static_cast<int>(t) / GRID_J; // 0..GRID_I-1
              int tj = static_cast<int>(t) % GRID_J; // 0..GRID_J-1

              for (int chi = ti; chi < chunkRows; chi += GRID_I)
              {
                  for (int chj = tj; chj < chunkCols; chj += GRID_J)
                  {
                      const int ibegin = chi * ChunkBlocks;
                      const int iend   = std::min(ibegin + ChunkBlocks, blocksI);
                      const int jbegin = chj * ChunkBlocks;
                      const int jend   = std::min(jbegin + ChunkBlocks, blocksJ);

                      for (int k_block = 0; k_block < K; k_block += Kc)
                      {
                          for (int jb = jbegin; jb < jend; ++jb)
                          {
                              const int j_block = jb * Nc;
                              auto      b_tile  = core::submatrix<Kc, Nc>(B, k_block, j_block);

                              for (int ib = ibegin; ib < iend; ++ib)
                              {
                                  const int i_block = ib * Mc;
                                  auto      a_tile  = core::submatrix<Mc, Kc>(A, i_block, k_block);

                                  for (int j = 0; j < Nc; j += Nr)
                                  {
                                      initBTile(b_tile, b_utile);
                                      for (int i = 0; i < Mc; i += Mr)
                                      {
                                          // Retrive next a_utile
                                          initATile(a_tile, a_utile);
                                          T* Cc0 = &C(i_block + i, j_block + j);
                                          kernels::zen5_mdspan_kernel(a_utile, b_utile, Cc0, N);
                                      }
                                  }
                              }
                          }
                      }
                  }
              }
          });
    }
    //});
}

} // namespace mm::zen5