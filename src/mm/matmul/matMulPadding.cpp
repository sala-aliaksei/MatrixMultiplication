#include "mm/matmul/matMulPadding.hpp"
#include "mm/core/reorderMatrix.hpp"

// #include <experimental/simd>

#include "omp.h"

__attribute__((always_inline)) static inline void load_inc_store_double(double* __restrict ptr,
                                                                        __m256d increment)
{
    // Load 4 double-precision values (256 bits) from memory into an AVX register
    __m256d vector = _mm256_load_pd(ptr);

    // Add the increment to the loaded vector
    __m256d result = _mm256_add_pd(vector, increment);

    // Store the result back to memory
    _mm256_store_pd(ptr, result);
    //    _mm256_stream_pd(ptr, result);
}

// NOTE: If Kc will be runtime arg? (Perf will drop)
// u - micro
// p - packed
template<int Nr, int Mr, int Kc>
static void upkernel(const double* __restrict ma,
                     const double* __restrict b,
                     double* __restrict mc,
                     int N)
{
    // constexpr int Kc = 240;
    double* c = mc;

    __m256d r00;
    __m256d r01;
    __m256d r02;
    __m256d r10;
    __m256d r11;
    __m256d r12;
    __m256d r20;
    __m256d r21;
    __m256d r22;
    __m256d r30;
    __m256d r31;
    __m256d r32;

    r00 = _mm256_xor_pd(r00, r00);
    r01 = _mm256_xor_pd(r01, r01);
    r02 = _mm256_xor_pd(r02, r02);
    r10 = _mm256_xor_pd(r10, r10);
    r11 = _mm256_xor_pd(r11, r11);
    r12 = _mm256_xor_pd(r12, r12);
    r20 = _mm256_xor_pd(r20, r20);
    r21 = _mm256_xor_pd(r21, r21);
    r22 = _mm256_xor_pd(r22, r22);
    r30 = _mm256_xor_pd(r30, r30);
    r31 = _mm256_xor_pd(r31, r31);
    r32 = _mm256_xor_pd(r32, r32);

    const double* a = ma;

    static_assert(Kc % 2 == 0, "Kc must me even for manual unrolling");
    for (int k = 0; k < Kc; k += 2, b += 2 * Nr, a += 2 * Mr)
    {
        //_mm_prefetch(b + 8, _MM_HINT_T0);
        __m256d b0 = _mm256_loadu_pd(&b[0]);
        __m256d b1 = _mm256_loadu_pd(&b[4]);
        __m256d b2 = _mm256_loadu_pd(&b[8]);

        __m256d a0 = _mm256_broadcast_sd(&a[0]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        a0 = _mm256_broadcast_sd(&a[1]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        a0 = _mm256_broadcast_sd(&a[2]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        a0 = _mm256_broadcast_sd(&a[3]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);

        // iter with prefetech

        b0 = _mm256_loadu_pd(&b[12]);
        b1 = _mm256_loadu_pd(&b[16]);
        b2 = _mm256_loadu_pd(&b[20]);

        a0 = _mm256_broadcast_sd(&a[4]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        a0 = _mm256_broadcast_sd(&a[5]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        a0 = _mm256_broadcast_sd(&a[6]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        //_mm_prefetch(a + 8, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[7]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);
    }

    //        _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);

    c += N;

    //    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    //    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
}

void matMulPadding(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    // BEST
    //    constexpr int Nc = 720;
    //    constexpr int Mc = 180;
    //    constexpr int Kc = 240;

    // NEW BEST
    //    constexpr int Nc = 720;
    //    constexpr int Mc = 20;
    //    constexpr int Kc = 80;

    constexpr int Nc = 720;
    constexpr int Mc = 20;
    constexpr int Kc = 80;

    //    constexpr int Nc = 3072;
    //    constexpr int Mc = 144;
    //    constexpr int Kc = 256;

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

    // TODO: Buffer should fit padding data as well.
    // We don't need it for outerloop J; since jtail will be handled separatly
    //

    std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(4 * Kc
                                                                                  * (Mc + Nc));

    // TODO: rewrite to while(N/Nc) and handle tail properly

    int N_max = N - (N % Nc);

#pragma omp parallel for
    for (int j_block = 0; j_block < N_max; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        double*       Cc4 = &C(0, j_block);
        const double* Bc4 = &B(0, j_block);

        int K_max = K - (K % Kc);

        for (int k_block = 0; k_block < K_max; k_block += Kc)
        {
            const double* Ac3  = &A(0, k_block);
            const double* Bcc3 = Bc4 + N * k_block;

            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(Bcc3, N, buf + Mc * Kc);
            const double* Bc3 = (buf + Mc * Kc);

            int M_max = M - (M % Mc);
            for (int i_block = 0; i_block < M_max; i_block += Mc)
            {
                double*       Cc2  = Cc4 + N * i_block;
                const double* Acc2 = Ac3 + K * i_block;

                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(Acc2, K, buf);
                const double* Ac2 = buf;

                for (int j = 0; j < Nc; j += Nr)
                {
                    const double* Bc1 = Bc3 + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = Cc2 + j + N * i;
                        const double* Ac0 = Ac2 + Kc * i;

                        // TODO: If Kc will be runtime arg? (Perf will drop)

                        // We don't need to fill Kdim with zeroes (if Kr==1)
                        // calc Kp and pass it instead
                        upkernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }

            // i tail
            // reorderRowMajorPaddingMatrix
            // reorderColOrderPaddingMatrix
            int Mlb = M - M_max;
            {
                //                double*       Cc2  = Cc4 + N * M_max;
                //                const double* Acc2 = Ac3 + K * M_max;

                //                // M_LAST_BLOCK_SIZE < Mc
                //                // apply only for corner subblock of cache block
                //                reorderColOrderPaddingMatrix<Mc, Kc, Mr, Kr>(Acc2, K, buf, Mlb,
                //                Kc); const double* Ac2 = buf;

                //                for (int j = 0; j < Nc; j += Nr)
                //                {
                //                    const double* Bc1 = Bc3 + Kc * j;
                //                    for (int i = 0; i < Mc; i += Mr)
                //                    {
                //                        double*       Cc0 = Cc2 + j + N * i;
                //                        const double* Ac0 = Ac2 + Kc * i;

                //                        // We don't need to fill Kdim with zeroes (only if Kr==1)
                //                        // calc Kp and pass it instead

                //                        // we need bufer for Cc0 [NrxMr] size
                //                        // We handle only cases where we need to padd i dim,
                //                        // Kernel should't have extra logic to handle corner cases

                //                        upkernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                //                        // in that case we don't need to do load/inc/store with
                //                        tmp buf?
                //                    }
                //                }
            }
        }

        // TODO: Handle k tail
        int K_LAST_BLOCK = K - K_max;
        {
            // Kc speedup perf, using dynamic value drop perf, that is way tail handled separatly
        }
    }
    // TODO: Hanle j tail
    int N_LAST_BLOCK = N - N_max;
    {
    }

    // TODO: Replace to padding version, buffer must fit padding elems (padding only for tails)
    // TODO: What if I don't need the whole KcxNc block? (tail case)
}
