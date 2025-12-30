#include "mm/matmul/zen5_constants.hpp"

#include "mm/core/reorderMatrix.hpp"
#include "mm/core/layout.hpp"
#include "mm/core/utils/cpu.hpp"
#include "mm/core/utils/utils.hpp"
#include "mm/core/utils/algorithms.hpp"
#include <mm/core/Matrix.hpp>
#include <mm/core/zen5kernels.hpp>
#include <thread>
#include <barrier>

namespace mm::hyper
{

template<typename T>
void matMulHyper(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    using namespace mm::constants;
    using namespace mm::core;

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
            for (int k_block = 0; k_block < K; k_block += Kc)
            {
                auto b_tile = &B(k_block, j_block);
                for (int i_block = ibegin; i_block < iend; i_block += Mc)
                {
                    auto a_tile = &A(i_block, k_block);
                    for (int j = 0; j < Nc; j += Nr)
                    {

                        bptr = b_utiles[b_data_idx].data_handle();
                        if constexpr (is_data_core_id(logical_core_id))
                        {
                            for (int idx = 0, kl = 0; kl < Kc; kl++)
                            {
                                for (int jl = 0; jl < Nr; jl++)
                                {
                                    bptr[idx++] = b_tile[kl * N + j + jl];
                                }
                            }
                        }
                        b_data_idx = (b_data_idx + 1) % 2;

                        for (int i = 0; i < Mc; i += Mr)
                        {
                            aptr = a_utiles[a_data_idx].data_handle();
                            if constexpr (is_data_core_id(logical_core_id))
                            {
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
                                auto* compute_abuf = aptr;
                                auto* compute_bbuf = bptr;
                                auto  Cc0          = &C(i_block + i, j_block + j);
                                // kernels::zen5_packed_kernel<Nr, Mr, Kc>(compute_abuf,
                                kernels::naive_block<Nr, Mr, Kc>(
                                  compute_abuf, compute_bbuf, Cc0, N);
                            }
                        }
                    }
                }
            }
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