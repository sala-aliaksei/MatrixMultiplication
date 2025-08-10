#pragma once

#include <mm/core/Matrix.hpp>
#include <mm/core/kernels.hpp>
#include <mm/core/reorderMatrix.hpp>
#include <mm/core/bf16kernel.hpp>

#include <thread>
#include <omp.h>

namespace mm::zen5
{
constexpr int PAGE_SIZE = 4096;

#define massert(x, msg) \
    (bool((x)) == true ? void(0) : throw std::runtime_error("Assertion failed: " #x " " msg))

struct MatMulZen5Config
{
    static constexpr int Nc = 96; // 3072/32=96
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
    constexpr int Nr = bregs_cnt * num_of_elems_in_reg; // 24
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

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

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

                        if constexpr (std::is_same_v<T, std::bfloat16_t>)
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

template<typename T>
void matMulZen5MTBlocking(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{

    constexpr int Nc = std::is_same_v<T, std::bfloat16_t> ? 96 * 2 : 96; // 3072/32=96
    constexpr int Mc = 96;
    constexpr int Kc = 96;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;
    constexpr int  Nr{bregs_cnt * num_of_elems_in_reg};
    constexpr int  Mr{8};
    constexpr int  Kr{1};

    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

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
              const std::size_t ofs = static_cast<std::size_t>(t) * Kc * (Mc + Nc);
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

} // namespace mm::zen5