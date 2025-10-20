#include <mm/core/Matrix.hpp>
#include <mm/core/utils/utils.hpp>
#include <mm/core/kernels.hpp>

#include <thread>
#include <vector>
#include <algorithm>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace mm::cacheAware
{
constexpr int PAGE_SIZE = 4096;

template<typename T>
void matMulZen5CacheEvict(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{

    constexpr int Nc = 2 * 96;
    constexpr int Mc = 2 * 96;
    constexpr int Kc = 2 * 96;

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

    std::vector<T> buffer(num_threads * Kc * (Mc + Nc));

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
#ifdef __linux__
              unsigned hw_threads = std::thread::hardware_concurrency();
              if (hw_threads == 0)
              {
                  hw_threads = 1;
              }
              const unsigned core_id = static_cast<unsigned>(t % hw_threads);
              cpu_set_t      cpuset;
              CPU_ZERO(&cpuset);
              CPU_SET(core_id, &cpuset);
              (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
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
                              auto      bbuf    = buf + Mc * Kc;

                              reorderRowMajorMatrixAVX<Kc, Nc, Kr, Nr>(
                                B.data() + N * k_block + j_block, N, buf + Mc * Kc);

                              for (int ib = ibegin; ib < iend; ++ib)
                              {
                                  const int i_block = ib * Mc;

                                  auto abuf = buf;

                                  reorderColOrderMatrix<Mc, Kc, Mr, Kr>(
                                    A.data() + K * i_block + k_block, K, buf);

                                  for (int j = 0; j < Nc; j += Nr)
                                  {
                                      const T* Bc1 = buf + Mc * Kc + Kc * j;
                                      for (int i = 0; i < Mc; i += Mr)
                                      {
                                          T* Cc0 = C.data() + N * i_block + j + N * i + j_block;
                                          const T* Ac0 = buf + Kc * i;

                                          kernels::zen5_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
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
} // namespace mm::cacheAware