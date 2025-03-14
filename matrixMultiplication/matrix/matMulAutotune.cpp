#include "matMulAutotune.hpp"
#include "reorderMatrix.hpp"

#include "omp.h"

#include <immintrin.h>

// BEST
// constexpr int Mc = 180;
// constexpr int Kc = 240;
// constexpr int Nc = 720;
#ifdef N_CACHE_SIZE
constexpr int Nc = N_CACHE_SIZE;
#else
constexpr int Nc = 720;
#endif

#ifdef M_CACHE_SIZE
constexpr int Mc = M_CACHE_SIZE;
#else
constexpr int Mc = 180;
#endif

#ifdef K_CACHE_SIZE
constexpr int Kc = K_CACHE_SIZE;
#else
constexpr int Kc = 240;
#endif

/// utils

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

template<int Nr, int Mr, int Kc>
static void upkernel(const double* __restrict ma,
                     const double* __restrict b,
                     double* __restrict mc,
                     int N)
{
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

    //                _mm_prefetch(b + 8, _MM_HINT_NTA);
    //                _mm_prefetch(b + 16, _MM_HINT_NTA);
    //                _mm_prefetch(b + 24, _MM_HINT_NTA);
    // _mm_prefetch(a + 8, _MM_HINT_NTA);

    //    _mm_prefetch(a + 8, _MM_HINT_T0);
    //    _mm_prefetch(b + 8, _MM_HINT_T0);
    //    _mm_prefetch(b + 16, _MM_HINT_T0);
    //    _mm_prefetch(b + 24, _MM_HINT_T0);
    //    _mm_prefetch(b + 64, _MM_HINT_T0);
    //    _mm_prefetch(b + 64 * 2, _MM_HINT_T0);
    //    _mm_prefetch(b + 64 * 3, _MM_HINT_T0);

    for (int k = 0; k < Kc; k += 2, b += 2 * Nr, a += 2 * Mr)
    {
        _mm_prefetch(b + 8, _MM_HINT_T0);
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

        _mm_prefetch(a + 8, _MM_HINT_T0);
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

void matMulAutotune(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // BEST
    // constexpr int Mc = 180;
    // constexpr int Kc = 240;
    // constexpr int Nc = 720;

    constexpr int Nr = 12;
    constexpr int Mr = 4;
    constexpr int Kr = 1; // consider to increase to improve repack perf

    std::cout << "Dimension : " << A.row() << " " << A.col() << std::endl;
    std::cout << "Nc: " << Nc << " Mc: " << Mc << " Kc: " << Kc << std::endl;

    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(4 * Kc
                                                                                  * (Mc + Nc));

#pragma omp parallel for
    for (int j = 0; j < N; j += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        double*       Cc4 = &C(0, j);
        const double* Bc4 = &B(0, j);

        for (int k = 0; k < K; k += Kc)
        {
            const double* Ac3  = &A(0, k);
            const double* Bcc3 = Bc4 + N * k;

            double* Bc3 = (buf + Mc * Kc);
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(Bcc3, N, Bc3);

            for (int i = 0; i < M; i += Mc)
            {
                double*       Cc2  = Cc4 + N * i;
                const double* Acc2 = Ac3 + K * i;

                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(Acc2, K, buf);
                const double* Ac2 = buf;

                for (int jb = 0; jb < Nc; jb += Nr)
                {
                    double*       Cc1 = Cc2 + jb;
                    const double* Bc1 = Bc3 + Kc * jb;

                    for (int ib = 0; ib < Mc; ib += Mr)
                    {
                        double*       Cc0 = Cc1 + N * ib;
                        const double* Ac0 = Ac2 + Kc * ib;

                        upkernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }
        }
    }

    // DONE
}
