#include "matMulZen5.hpp"

#include "mm/core/kernels.hpp"
#include "mm/core/reorderMatrix.hpp"

#include <thread>
#include <omp.h>

namespace mm::zen5
{
constexpr int PAGE_SIZE = 4096;

#define massert(x, msg) \
    (bool((x)) == true ? void(0) : throw std::runtime_error("Assertion failed: " #x " " msg))

void matMulZen5(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    constexpr int Nc = 96;
    constexpr int Mc = 24;
    constexpr int Kc = 96;

    constexpr auto num_of_regs         = 32;
    constexpr auto bregs_cnt           = 3;
    constexpr auto aregs_cnt           = 1;
    constexpr auto num_of_elems_in_reg = 8; // for 512bit regs

    constexpr int Nr = bregs_cnt * num_of_elems_in_reg; // 24
    constexpr int Mr = num_of_regs / (aregs_cnt + bregs_cnt);

    // consider to increase to improve repack perf
    // Kr = 1, no need for padding over k dim
    constexpr int Kr = 1;

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

    std::vector<double, boost::alignment::aligned_allocator<double, PAGE_SIZE>> buffer(
      num_threads * Kc * (Mc + Nc));

#pragma omp parallel for num_threads(num_threads)
    for (int j_block = 0; j_block < N; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

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
                    const double* Bc1 = buf + Mc * Kc + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = C.data() + N * i_block + j + N * i + j_block;
                        const double* Ac0 = buf + Kc * i;

                        kernels::zen5_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }
        }
    }
}
} // namespace mm::zen5