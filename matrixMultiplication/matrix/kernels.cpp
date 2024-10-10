
#include "matrixMultiplication/matrix/kernels.hpp"

#include <immintrin.h>
#include <array>

/*****************     KERNELS     *******************/
namespace kernels
{

void kernelMulMatrix_BL_NV(double*           res,
                           const double*     mul1,
                           const double*     mul2,
                           const std::size_t block_size,
                           const std::size_t j_size,
                           const std::size_t k_size)
{
    const double *a, *b;

    int i2{}, k2{}, j2{};

    double* r;
    for (i2 = 0, r = res, a = mul1; i2 < block_size; ++i2, r += j_size, a += j_size)
        for (k2 = 0, b = mul2; k2 < block_size; ++k2, b += k_size)
            for (j2 = 0; j2 < block_size; ++j2)
                r[j2] += a[k2] * b[j2];
}

void kernelMulMatrix_TP_BL_NV(double*           r,
                              const double*     a,
                              const double*     mul2,
                              const std::size_t block_size,
                              const std::size_t j_size,
                              const std::size_t k_size)
{
    const double* b;

    for (auto i = 0; i < block_size; ++i, r += j_size, a += k_size)
    {
        b = mul2;
        for (auto j = 0; j < block_size; ++j, b += k_size)
        {
            double t = 0;
            for (auto k = 0; k < block_size; ++k)
            {
                t += a[k] * b[k];
            }
            r[j] += t;
        }
    }
}

static double sumElemFromReg(__m256d rk)
{
    std::array<double, 4> arrK{0, 0, 0, 0};
    _mm256_storeu_pd(arrK.data(), rk);
    return arrK[0] + arrK[1] + arrK[2] + arrK[3];
}

void kernelMulMatrix_VT_BL_TP(double*           r,
                              const double*     a,
                              const double*     mul2,
                              const std::size_t block_size,
                              const std::size_t j_size,
                              const std::size_t k_size)
{
    // TODO: change constants to vars

    const double* b;
    for (auto i = 0; i < block_size; ++i, r += j_size, a += k_size)
    {
        b = mul2;
        for (auto j = 0; j < block_size; ++j, b += k_size)
        {
            //_mm_prefetch(&b[N], _MM_HINT_NTA);

            __m256d rk = _mm256_setzero_pd();
            for (auto k = 0; k < block_size; k += 4)
            {
                __m256d m1 = _mm256_loadu_pd(&a[k]);
                __m256d m2 = _mm256_loadu_pd(&b[k]);

                rk = _mm256_fmadd_pd(m2, m1, rk);
            }

            r[j] += sumElemFromReg(rk);
        }
    }
}

// a[0:3]*b[0:3] = r[0:3]
//

static void mulMatrix_128VL_BL(double*           rres,
                               const double*     rmul1,
                               const double*     m_mul2,
                               const std::size_t block_size,
                               const std::size_t j_size,
                               const std::size_t k_size)
{
    const double* rmul2 = m_mul2;
    for (int i2 = 0; i2 < block_size; ++i2, rres += j_size, rmul1 += k_size)
    {
        _mm_prefetch(&rmul1[block_size], _MM_HINT_NTA);
        rmul2 = m_mul2;

        __m128d r20 = _mm_load_pd(&rres[0]);
        __m128d r21 = _mm_load_pd(&rres[2]);
        __m128d r22 = _mm_load_pd(&rres[4]);
        __m128d r23 = _mm_load_pd(&rres[6]);

        for (int k2 = 0; k2 < block_size; ++k2, rmul2 += j_size)
        {
            __m128d m20 = _mm_load_pd(&rmul2[0]);
            __m128d m21 = _mm_load_pd(&rmul2[2]);
            __m128d m22 = _mm_load_pd(&rmul2[4]);
            __m128d m23 = _mm_load_pd(&rmul2[6]);
            __m128d m1d = _mm_load_sd(&rmul1[k2]);
            m1d         = _mm_unpacklo_pd(m1d, m1d);

            r20 = _mm_fmadd_pd(m20, m1d, r20);
            r21 = _mm_fmadd_pd(m21, m1d, r21);
            r22 = _mm_fmadd_pd(m22, m1d, r22);
            r23 = _mm_fmadd_pd(m23, m1d, r23);
        }
        _mm_store_pd(&rres[0], r20);
        _mm_store_pd(&rres[2], r21);
        _mm_store_pd(&rres[4], r22);
        _mm_store_pd(&rres[6], r23);
    }
}

static void mulMatrix_256VL_BL(double*           rres,
                               const double*     rmul1,
                               const double*     m_mul2,
                               const std::size_t block_size,
                               const std::size_t j_size,
                               const std::size_t k_size)
{

    const double* rmul2 = m_mul2;
    for (int i2 = 0; i2 < block_size; ++i2, rres += j_size, rmul1 += j_size)
    {
        // _mm_prefetch(&rres[N], _MM_HINT_NTA);
        // _mm_prefetch(&rmul1[N], _MM_HINT_NTA);

        _mm_prefetch(&rres[block_size], _MM_HINT_T0);
        _mm_prefetch(&rmul1[block_size], _MM_HINT_T0);

        rmul2 = m_mul2;

        __m256d r20 = _mm256_loadu_pd(&rres[0]);
        __m256d r22 = _mm256_loadu_pd(&rres[4]);

        for (int k2 = 0; k2 < block_size; ++k2, rmul2 += k_size)
        {
            __m256d m20 = _mm256_loadu_pd(&rmul2[0]);
            __m256d m22 = _mm256_loadu_pd(&rmul2[4]);
            __m256d m1d = _mm256_broadcast_sd(&rmul1[k2]);
            r20         = _mm256_add_pd(r20, _mm256_mul_pd(m20, m1d));
            r22         = _mm256_add_pd(r22, _mm256_mul_pd(m22, m1d));
        }
        _mm256_storeu_pd(&rres[0], r20);
        _mm256_storeu_pd(&rres[4], r22);
    }
}

void kernelMulMatrix_VT_BL(double*           c,
                           const double*     a,
                           const double*     b,
                           const std::size_t block_size,
                           const std::size_t j_size,
                           const std::size_t k_size)
{

#if defined(__x86_64__)
#ifdef __AVX2__
    mulMatrix_256VL_BL(c, a, b, block_size, j_size, k_size);
#elif __SSE2__
    mulMatrix_128VL_BL(c, a, b, block_size, j_size, k_size);
#else
#error "Manual vectorization is not supported for current cpu!"
#endif
#elif defined(__ARM_ARCH)
#error "Manual vectorization is not supported for ARM arch!"
#endif
}

} // namespace kernels
