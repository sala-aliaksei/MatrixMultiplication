#include "mm/matmul/zen5_constants.hpp"

#include "mm/core/reorderMatrix.hpp"
#include "mm/core/utils/cpu.hpp"
#include "mm/core/utils/utils.hpp"
#include "mm/core/utils/algorithms.hpp"
#include <mm/core/Matrix.hpp>
#include <mm/core/zen5kernels.hpp>
#include <mm/core/kernels.hpp>
#include <thread>
#include <barrier>

#include <tracy/Tracy.hpp>

namespace mm::hyper
{

template<typename T>
void matMulZen5MTBlockingH(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    using namespace mm::constants;
    constexpr int Nc = MatMulZen5Config<T>::Nc;
    constexpr int Mc = MatMulZen5Config<T>::Mc;
    constexpr int Kc = MatMulZen5Config<T>::Kc;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = number_of_elems_in_512_reg_v<T>;

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

    // Fixed thread grid 4x4 → 16 tiles. 2 threads per tile (HT) -> 32 threads
    constexpr int      GRID_I      = 4;
    constexpr int      GRID_J      = 4;
    constexpr unsigned num_threads = 32;
    constexpr auto     num_cores   = 16;

    constexpr auto elems_per_core = 2 * Kc * (Mc + Nc);
    std::vector<T> buffer(num_cores * elems_per_core);

    std::vector<std::jthread> workers;
    workers.reserve(num_threads);

    std::array<std::barrier<>, num_cores> core_barriers = {std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2),
                                                           std::barrier(2)};

    for (unsigned t = 0; t < num_threads; ++t)
    {
        workers.emplace_back(
          [&, t]()
          {
              using namespace std::string_literals;
              auto      core_id    = map_thread_id_to_core_id(t);
              auto      thread_idx = core_id / 16;
              cpu_set_t cpuset;
              CPU_ZERO(&cpuset);
              CPU_SET(core_id, &cpuset);
              (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

              auto core = core_id % 16;
              tracy::SetThreadName(
                ("core"s + std::to_string(core) + "_thread_"s + std::to_string(t)).c_str());

              // Map threads to grid: pairs of threads (HT siblings) share a tile
              const int pair_idx = static_cast<int>(t) / 2;
              const int ti       = pair_idx / GRID_J; // 0..GRID_I-1
              const int tj       = pair_idx % GRID_J; // 0..GRID_J-1

              auto ibegin = ti * M / GRID_I; // flatten i index
              auto iend   = (ti + 1) * M / GRID_I;

              auto jbegin = tj * N / GRID_J; // flatted j index
              auto jend   = (tj + 1) * N / GRID_J;

              auto j_begin_per_core = (thread_idx == 0) ? jbegin : (jbegin + jend) / 2;
              auto j_end_per_core   = (thread_idx == 0) ? (jbegin + jend) / 2 : jend;

              auto& barrier = core_barriers[core];

              const std::size_t ofs = core * elems_per_core;
              T* const          buf = buffer.data() + ofs;

              for (int j_block = j_begin_per_core; j_block < j_end_per_core; j_block += Nc)
              {
                  for (int k_block = 0; k_block < K; k_block += Kc)
                  {
                      T* Bc2 = buf + 2 * Mc * Kc + Nc * Kc * thread_idx;
                      {
                          ZoneScopedN("[Data] Repacking B tile");
                          reorderRowMajorMatrixAVX<Kc, Nc, Kr, Nr>(
                            B.data() + N * k_block + j_block, N, Bc2);
                      }

                      for (int i_block = ibegin; i_block < iend; i_block += 2 * Mc)
                      {
                          T* Ac20 = buf;
                          T* Ac21 = buf + Mc * Kc;

                          if (thread_idx == 0)
                          {
                              ZoneScopedN("[Data] Repacking A0 tile");
                              reorderColOrderMatrix<Mc, Kc, Mr, Kr>(
                                A.data() + K * (i_block) + k_block, K, Ac20);
                          }
                          else
                          {
                              ZoneScopedN("[Data] Repacking A1 tile");
                              reorderColOrderMatrix<Mc, Kc, Mr, Kr>(
                                A.data() + K * (i_block + Mc) + k_block, K, Ac21);
                          }

                          {
                              ZoneScopedN("[Barrier] Wait for A tiles");
                              barrier.arrive_and_wait();
                          }
                          ZoneScopedN("[Compute] Copmute McxNc block");
                          for (int j = 0; j < Nc; j += Nr)
                          {
                              const T* Bc1 = Bc2 + Kc * j;
                              for (int i = 0; i < Mc; i += Mr)
                              {
                                  T*       Cc0 = C.data() + N * (i_block + i) + j + j_block;
                                  const T* Ac0 = Ac20 + Kc * i;
                                  xkernels::zen5_packed_kernel<Nr, Mr>(Ac0, Bc1, Cc0, N, Kc);
                              }
                          }

                          // if (i_block + Mc < iend)
                          {
                              for (int j = 0; j < Nc; j += Nr)
                              {
                                  const T* Bc1 = Bc2 + Kc * j;
                                  for (int i = 0; i < Mc; i += Mr)
                                  {
                                      T* Cc0 = C.data() + N * (i_block + Mc + i) + j + j_block;
                                      const T* Ac0 = Ac21 + Kc * i;

                                      xkernels::zen5_packed_kernel<Nr, Mr>(Ac0, Bc1, Cc0, N, Kc);
                                  }
                              }
                          }
                          {
                              ZoneScopedN("[Barrier] Wait for compute");
                              barrier.arrive_and_wait();
                          }
                      }
                  }
              }
          });

    } // jthreads auto-join on destruction
}

template<typename T>
void matMulHyper(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    using namespace mm::constants;

    constexpr std::size_t Nc = MatMulZen5Config<T>::Nc;
    constexpr std::size_t Mc = MatMulZen5Config<T>::Mc;
    constexpr std::size_t Kc = MatMulZen5Config<T>::Kc;
    constexpr std::size_t Nr = MatMulZen5Config<T>::Nr;
    constexpr std::size_t Mr = MatMulZen5Config<T>::Mr;
    constexpr std::size_t Kr = MatMulZen5Config<T>::Kr;

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
    constexpr int GRID_I = 4;
    constexpr int GRID_J = 4;

    constexpr unsigned num_physical_cores = GRID_I * GRID_J;
    constexpr unsigned num_threads        = 2 * num_physical_cores;

    constexpr auto tiles_size = 2 * Kc * (Mr + Nr);

    std::vector<T> buffer(num_physical_cores * tiles_size);

    const int total_iblocks_per_thread = M / Mc;
    const int total_jblocks_per_thread = N / Nc;

    const int iblocks_per_thread = total_iblocks_per_thread / GRID_I;
    const int jblocks_per_thread = total_jblocks_per_thread / GRID_J;

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    std::array<std::barrier<>, num_physical_cores> core_barriers = {std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2),
                                                                    std::barrier(2)};

    constexpr auto is_data_core_id = [](int core_id) constexpr
    { return core_id < num_physical_cores; };

    auto gemm_fn = [&]<int t>()
    {
        // core id will be the same for threads which share resources
        constexpr auto logical_core_id  = map_thread_id_to_core_id(t);
        constexpr auto physical_core_id = logical_core_id % num_physical_cores;

        // two threads are sharing the buf with same physical_core_id
        T* const buf     = buffer.data() + physical_core_id * tiles_size;
        auto&    barrier = core_barriers[physical_core_id];

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(logical_core_id, &cpuset);
        (void)pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        std::mdspan<T, std::extents<std::size_t, Kc, Mr>> a0_utile(buf, Kc, Mr);
        std::mdspan<T, std::extents<std::size_t, Kc, Mr>> a1_utile(
          a0_utile.data_handle() + a0_utile.size(), Kc, Mr);
        std::mdspan<T, std::extents<std::size_t, Kc, Nr>> b0_utile(
          a1_utile.data_handle() + a1_utile.size(), Kc, Nr);
        std::mdspan<T, std::extents<std::size_t, Kc, Nr>> b1_utile(
          b0_utile.data_handle() + b0_utile.size(), Kc, Nr);

        std::array<std::mdspan<T, std::extents<std::size_t, Kc, Mr>>, 2> a_utiles = {a0_utile,
                                                                                     a1_utile};
        std::array<std::mdspan<T, std::extents<std::size_t, Kc, Nr>>, 2> b_utiles = {b0_utile,
                                                                                     b1_utile};

        auto a_data_idx  = 0;
        auto b_data_idx  = 0;
        auto compute_idx = 0;
        T*   aptr        = nullptr;
        T*   bptr        = nullptr;

        // Thread's grid coords
        const int ti = static_cast<int>(physical_core_id) / GRID_J; // 0..GRID_I-1
        const int tj = static_cast<int>(physical_core_id) % GRID_J; // 0..GRID_J-1

        const int ibegin = ti * iblocks_per_thread * Mc;
        const int iend   = ibegin + iblocks_per_thread * Mc;
        const int jbegin = tj * jblocks_per_thread * Nc;
        const int jend   = jbegin + jblocks_per_thread * Nc;

        for (int j_block = jbegin; j_block < jend; j_block += Nc)
        {
            if constexpr (is_data_core_id(logical_core_id))
            {
                ZoneScopedN("[Data]j_block");
            }
            else
            {
                ZoneScopedN("[Compute]j_block");
            }
            for (int k_block = 0; k_block < K; k_block += Kc)
            {
                if constexpr (is_data_core_id(logical_core_id))
                {
                    ZoneScopedN("[Data]k_block");
                }
                else
                {
                    ZoneScopedN("[Compute]k_block");
                }
                auto b_tile = &B(k_block, j_block);
                for (int i_block = ibegin; i_block < iend; i_block += Mc)
                {
                    if constexpr (is_data_core_id(logical_core_id))
                    {
                        ZoneScopedN("[Data]i_block");
                    }
                    else
                    {
                        ZoneScopedN("[Compute]i_block");
                    }
                    auto a_tile = &A(i_block, k_block);
                    for (int j = 0; j < Nc; j += Nr)
                    {
                        if constexpr (is_data_core_id(logical_core_id))
                        {
                            ZoneScopedN("[Data] j");
                        }
                        else
                        {
                            ZoneScopedN("[Compute] j");
                        }
                        bptr = b_utiles[b_data_idx].data_handle();
                        if constexpr (is_data_core_id(logical_core_id))
                        {
                            ZoneScopedN("[Data]repacking b_tile");
                            for (int idx = 0, kl = 0; kl < Kc; kl++)
                            {
                                for (int jl = 0; jl < Nr; jl++)
                                {
                                    bptr[idx++] = b_tile[kl * N + j + jl];
                                }
                            }
                            // FrameMark;
                        }
                        b_data_idx = (b_data_idx + 1) % 2;

                        for (int i = 0; i < Mc; i += Mr)
                        {
                            if constexpr (is_data_core_id(logical_core_id))
                            {
                                ZoneScopedN("[Data] i");
                            }
                            else
                            {
                                ZoneScopedN("[Compute] i");
                            }
                            aptr = a_utiles[a_data_idx].data_handle();
                            if constexpr (is_data_core_id(logical_core_id))
                            {
                                ZoneScopedN("[Data]repacking a_tile");
                                for (int idx = 0, kl = 0; kl < Kc; kl++)
                                {
                                    for (int il = 0; il < Mr; il++)
                                    {
                                        aptr[idx++] = a_tile[(il + i) * N + kl];
                                    }
                                }
                            }
                            a_data_idx = (a_data_idx + 1) % 2;
                            barrier.arrive_and_wait();

                            if constexpr (!is_data_core_id(logical_core_id))
                            {
                                ZoneScopedN("[Compute]compute");
                                auto* compute_abuf = aptr;
                                auto* compute_bbuf = bptr;
                                auto  Cc0          = &C(i_block + i, j_block + j);
                                // kernels::zen5_packed_kernel<Nr, Mr, Kc>(compute_abuf,
                                kernels::naive_block<Nr, Mr, Kc>(
                                  compute_abuf, compute_bbuf, Cc0, N);
                                // FrameMark;
                            }
                            // FrameMark;
                        }
                        // FrameMark;
                    }
                    // FrameMark;
                }
                // FrameMark;
            }
            // FrameMark;
        }
    };

    static_for<num_threads>([&]<int t>()
                            { workers.emplace_back([&] { gemm_fn.template operator()<t>(); }); });

    for (auto& worker : workers)
    {
        worker.join();
    }
}
} // namespace mm::hyper

// for (int t = 0; t < num_threads; ++t)
// {
//     workers.emplace_back(gemm_fn, t);
// }

// FAST (38ms)

// THE SLOWEST (537 ms), bit here we didn't use std::jthread!!!
// [&]<std::size_t... I>(std::index_sequence<I...>)
// { (..., gemm_fn.template operator()<I>()); }(std::make_index_sequence<num_threads>{});

// doesn't compile, You can’t form a pointer-to-member from an object:
//  static_for<num_threads>([&]<int t>() { workers.emplace_back(gemm_fn.template operator()<t>);
//  });

// SLOW (130ms)
// static_for<num_threads>(
//   [&]<int t>()
//   {
//       workers.emplace_back(
//         [&]
//         {
//             gemm_fn.template operator()<t>(); // or <t>(A,B,C) if it takes args
//         });
//   });