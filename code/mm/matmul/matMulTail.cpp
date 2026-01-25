#include "mm/core/utils/utils.hpp"
#include "mm/core/reorderMatrix.hpp"
#include "mm/matmul/matMulTail.hpp"

#include "mm/core/kernels.hpp"

#include <thread>

#include "omp.h"

constexpr unsigned long PAGE_SIZE = 4096;

void matMulSimd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    constexpr int Nc = 180; // 180(best for hawswell)
    constexpr int Mc = 20;
    constexpr int Kc = 80;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    constexpr int Kr = 1;

    auto num_threads = 16; // std::thread::hardware_concurrency();
    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    massert(N % Nc == 0, "N % Nc == 0");
    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");
    massert(N % num_threads == 0, "N % num_threads == 0");
    massert((N / num_threads) % Nc == 0, "(N/num_threads) % Nc == 0");

    std::vector<double> buffer(num_threads * Kc * (Mc + Nc));

#pragma omp parallel for num_threads(num_threads)
    for (int j_block = 0; j_block < N; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        for (int k_block = 0; k_block < K; k_block += Kc)
        {
            // reorderRowMajorMatrixAVX<Kc, Nc, Kr, Nr>(
            //   B.data() + N * k_block + j_block, N, buf + Mc * Kc);
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(
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

                        xkernels::cpp_packed_kernel<Nr, Mr>(Ac0, Bc1, Cc0, N, Kc);
                    }
                }
            }
        }
    }
}

/////       TAILS
///
///

void matMulSimdTails(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    // NEW BEST
    constexpr int Nc = 180;
    constexpr int Mc = 20;
    constexpr int Kc = 80;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    // consider to increase to improve repack perf
    // Kr = 1, no need for padding over k dim
    constexpr int Kr = 1;

    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    auto num_threads = std::thread::hardware_concurrency();

    std::vector<double> buffer(num_threads * Kc * (Mc + Nc));

    // tail is only in last block
    int dNc = N % Nc;
    int jl  = N - dNc;

#pragma omp parallel for num_threads(num_threads)
    for (int j_block = 0; j_block < jl; j_block += Nc)
    {
        auto       tid   = omp_get_thread_num();
        const auto ofs   = tid * Kc * (Mc + Nc);
        double*    a_buf = buffer.data() + ofs;
        double*    b_buf = a_buf + Mc * Kc;

        // For dDEBUG:
        //        constexpr int dKc = 3;

        int dKc   = K % Kc;
        int klast = K - dKc;

        for (int k_block = 0; k_block < klast; k_block += Kc)
        {
            // I can guarantee the we always within the block and no padding needed
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(B.data() + N * k_block + j_block, N, b_buf);

            int dMc   = M % Mc;
            int ilast = M - dMc;
            for (int i_block = 0; i_block < ilast; i_block += Mc)
            {
                // Can be access out of bound if i+Mc > M. No
                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K, a_buf);

                for (int j = 0; j < Nc; j += Nr)
                {
                    const double* Bc1 = b_buf + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = C.data() + N * (i_block + i) + j_block + j;
                        const double* Ac0 = a_buf + Kc * i;

                        // TODO: deduce args from span?
                        xkernels::cpp_packed_kernel<Nr, Mr>(Ac0, Bc1, Cc0, N, Kc);
                    }
                }
            }

            const double* Ac1 = A.data() + k_block + ilast * K;
            double*       Cc1 = C.data() + j_block + ilast * N;

            handleItail<Nr, Kr, Nc, Kc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, dMc);
        }

        // TODO: Choose Ktails properly
        handleKtail<Mr, Nr, Kr, Mc, Nc, 20, 10, 4, 2, 1>(a_buf,
                                                         b_buf,
                                                         A.data() + klast,
                                                         B.data() + N * klast + j_block,
                                                         C.data() + j_block,
                                                         M,
                                                         N,
                                                         K,
                                                         dKc);

        // TODO: Can recalc b_buf address to be cllsoer to a_buf
    }

    // TODO: Add multithreading

    handleJtail<Mr, Kr, Mc, Kc, 12, 8, 4, 2, 1>(
      buffer.data(), A.data(), &B(0, jl), &C(0, jl), M, K, N, dNc);
}
