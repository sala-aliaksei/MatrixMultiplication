
#include "matrixMultiplication/matrix/kernels.hpp"

#include <immintrin.h>
#include <array>
#include <stdexcept>

/*****************     KERNELS     *******************/
namespace kernels
{

template<int block_size>
void mulMatrix_x(double*           c,
                 const double*     a,
                 const double*     mb,
                 const std::size_t j_size,
                 const std::size_t k_size)
{
    //    size_t i_max = std::min(ii + block_size, end_row);
    //    size_t j_max = std::min(jj + block_size, C.col());
    //    size_t k_max = std::min(kk + block_size, A.col());

    const double* b = mb;
    for (int i2 = 0; i2 < block_size; ++i2, c += j_size, a += j_size)
    {
        b = mb;

        std::array<__m256d, block_size / 4> res;
        std::array<__m256d, block_size / 4> breg;

        for (int j2 = 0, idx = 0; j2 < block_size; j2 += 4, ++idx)
        {
            res[idx]  = _mm256_loadu_pd(&c[j2]);
            breg[idx] = _mm256_loadu_pd(&b[j2]);
        }

        _mm_prefetch(&a[64 / sizeof(double)], _MM_HINT_NTA); // prefetch next cache line
        for (int k2 = 0; k2 < block_size; ++k2, b += k_size)
        {
            __m256d m1d = _mm256_broadcast_sd(&a[k2]);

            for (int j2 = 0, idx = 0; j2 < block_size; j2 += 4, ++idx)
            {
                //__m256d m2 = _mm256_loadu_pd(&b[j2]);;
                res[idx] = _mm256_add_pd(res[idx], _mm256_mul_pd(breg[idx], m1d));
            }
        }

        for (int j2 = 0, idx = 0; j2 < block_size; j2 += 4, ++idx)
        {
            _mm256_storeu_pd(&c[j2], res[idx]);
        }
    }
}

static double sumElemFromReg(__m256d rk)
{
    std::array<double, 4> arrK{0, 0, 0, 0};
    _mm256_storeu_pd(arrK.data(), rk);
    return arrK[0] + arrK[1] + arrK[2] + arrK[3];
}

void mulMatrix_256VL_BL_v2(double*           c,
                           const double*     a,
                           const double*     mb,
                           const std::size_t j_size,
                           const std::size_t k_size)
{
    // kernel must be inlined, block can be arg?
    constexpr std::size_t block_size = 8;
    // analyze data dependencies

    const double* b = mb;
    for (int i2 = 0; i2 < block_size; ++i2, c += j_size, a += j_size)
    {
        // _mm_prefetch(&rres[N], _MM_HINT_NTA);
        // _mm_prefetch(&rmul1[N], _MM_HINT_NTA);

        b = mb;

        __m256d r20 = _mm256_loadu_pd(&c[0]);
        __m256d r22 = _mm256_loadu_pd(&c[4]);

        // #pragma GCC unroll 2
        for (int k2 = 0; k2 < block_size; ++k2, b += k_size)
        {
            __m256d m1d = _mm256_broadcast_sd(&a[k2]);

            r20 = _mm256_add_pd(r20, _mm256_mul_pd(_mm256_loadu_pd(&b[0]), m1d));

            r22 = _mm256_add_pd(r22, _mm256_mul_pd(_mm256_loadu_pd(&b[4]), m1d));
        }
        _mm256_storeu_pd(&c[0], r20);
        _mm256_storeu_pd(&c[4], r22);
    }
}

static void mulMatrix_256VL_BL(double*           c,
                               const double*     a,
                               const double*     mb,
                               const std::size_t block_size,
                               const std::size_t j_size,
                               const std::size_t k_size)
{
    // kernel must be inlined, block can be arg?
    // const std::size_t block_size = 4;
    // analyze data dependencies

    if (block_size == 8)
    {
        mulMatrix_x<8>(c, a, mb, j_size, k_size);
    }
    else if (block_size == 16)
    {
        mulMatrix_x<16>(c, a, mb, j_size, k_size);
    }
    else if (block_size == 24)
    {
        mulMatrix_x<24>(c, a, mb, j_size, k_size);
    }
    else if (block_size == 32)
    {
        mulMatrix_x<32>(c, a, mb, j_size, k_size);
    }
    else if (block_size == 64)
    {
        mulMatrix_x<64>(c, a, mb, j_size, k_size);
    }
    else
    {
        throw std::runtime_error("Unsupported block size");
    }

    //    const double* b = mb;
    //    for (int i2 = 0; i2 < block_size; ++i2, c += j_size, a += j_size)
    //    {
    //        b = mb;

    //        for (int k2 = 0; k2 < block_size; ++k2, b += k_size)
    //        {
    //            __m256d m1d = _mm256_broadcast_sd(&a[k2]);

    //            for (int j2 = 0; j2 < block_size; j2 += 4)
    //            {
    //                __m256d b_reg = _mm256_loadu_pd(&b[j2]);
    //                __m256d c_reg = _mm256_loadu_pd(&c[j2]);
    //                c_reg         = _mm256_add_pd(c_reg, _mm256_mul_pd(b_reg, m1d));
    //                _mm256_storeu_pd(&c[j2], c_reg);
    //            }
    //        }
    //    }
}

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
