
#include "mm/core/kernels.hpp"
#include <math.h>

#include <immintrin.h>
#include <array>
#include <stdexcept>

#include <iostream>

// 1. thread coarsening
// 2.

/*****************     KERNELS     *******************/

// TODO: Block size should be private info
// looks like b[32x32] fit into l1 cache
// constexpr std::size_t block_size_i = 48; // 4;
// constexpr std::size_t block_size_j = 48; // 12;
// constexpr std::size_t block_size_k = 48; // 8;
// TODO: Remove from header, otherwise ODR since vars are defined in so lib
constexpr inline std::size_t block_size_i = 32;
constexpr inline std::size_t block_size_j = 48;
constexpr inline std::size_t block_size_k = 64;

static_assert(block_size_j % 4 == 0, "invalid block_size_j");

namespace kernels
{

// KERNEL4x12_I
// 12 - y reg cnt,  block_size_j = 48 -> 48/4=12

// template<int M, int N>
// std::array<double, M * N> packMatrix(const double* b, int col)
//{
//     int idx = 0;

//    std::array<double, M * N> b_packed;
//    for (int k = 0; k < M; k++)
//    {
//        for (int j = 0; j < N; j++)
//        {
//            b_packed[idx++] = b[k * col + j];
//        }
//    }
//    return b_packed;
//}

template<int M, int N>
std::array<double, M * N> packMatrix(const double* b, int col)
{
    int idx = 0;

    std::array<double, M * N> b_packed;
    for (int k = 0; k < M; k += block_size_k)
    {
        for (int j = 0; j < N; j += block_size_j)
        {
            for (int k2 = 0; k2 < block_size_k; k2++)
            {
                for (int j2 = 0; j2 < block_size_j; j2++)
                {
                    b_packed[(k + k2) * N + j + j2] = b[(k + k2) * col + j + j2];
                }
            }
        }
    }
    return b_packed;
}

void matmul_NV(double* __restrict c,
               const double* __restrict a,
               const double* __restrict mb,
               const std::size_t M,
               const std::size_t N,
               const std::size_t K)
{
    //
    const double* b = mb;
    for (int i2 = 0; i2 < M; ++i2, c += N, a += K)
    {
        b = mb;
        for (int k2 = 0; k2 < K; ++k2, b += N)
        {
            for (int j2 = 0; j2 < N; ++j2)
            {
                c[j2] += a[k2] * b[j2];
            }
        }
    }
}

void matmul_TP_NV(double* __restrict c,
                  const double* __restrict a,
                  const double* __restrict mb,
                  const std::size_t i_size,
                  const std::size_t j_size,
                  const std::size_t k_size)
{
    const double* b = mb;
    for (int i2 = 0; i2 < i_size; ++i2, c += j_size, a += k_size)
    {
        b = mb;
        for (int j2 = 0; j2 < j_size; ++j2)
        {
            for (int k2 = 0; k2 < k_size; ++k2, b += k_size)
            {
                c[j2] += a[k2] * b[k2];
            }
        }
    }
}

void kernelMulMatrix_BL_NV(double* __restrict c,
                           const double* __restrict a,
                           const double* __restrict mb,
                           const std::size_t i_size,
                           const std::size_t j_size,
                           const std::size_t k_size,
                           const std::size_t ii,
                           const std::size_t jj,
                           const std::size_t kk)
{
    const double* b = mb;
    for (int i2 = 0; i2 < block_size_i; ++i2, c += j_size, a += k_size)
    {
        b = mb;
        for (int k2 = 0; k2 < block_size_k; ++k2, b += j_size)
        {
            for (int j2 = 0; j2 < block_size_j; ++j2)
            {
                c[j2] = std::fma(a[k2], b[j2], c[j2]);
                // c[j2] += a[k2] * b[j2];
            }
        }
    }
}

static inline void load_inc_store_double(double* ptr, __m256d increment)
{
    // Load 4 double-precision values (256 bits) from memory into an AVX register
    __m256d vector = _mm256_load_pd(ptr);

    // Add the increment to the loaded vector
    __m256d result = _mm256_add_pd(vector, increment);

    // Store the result back to memory
    _mm256_store_pd(ptr, result);
}

// reach openblas perf
// add tail calculation
// add usage of x register if block size is less than 256bit
template<int block_size_i, int block_size_j, int block_size_k>
static void mulMatrix_x(double* __restrict c,
                        const double* __restrict ma,
                        const double* __restrict mb,
                        const std::size_t i_size,
                        const std::size_t j_size,
                        const std::size_t k_size,
                        const std::size_t ii,
                        const std::size_t jj,
                        const std::size_t kk)
{
    // block_J - amount of b,c registers
    // block_K - amount of b registers
    // amount of y registers - 16, do you take into account threads, how many reg per thread
    // availabe?

    constexpr int doubles_in_jblock{block_size_j / 4};

    // const auto packed_a = packMatrix<block_size_i, block_size_k>(ma, k_size);
    const auto packed_b = packMatrix<block_size_k, block_size_j>(mb, j_size);

    const double* a = ma; // packed_a.data();
    const double* b = packed_b.data();

    for (int i2 = 0; i2 < block_size_i; ++i2, c += j_size, a += k_size)
    {
        b = packed_b.data();

        std::array<__m256d, doubles_in_jblock> cache;
        std::array<__m256d, doubles_in_jblock> res;
        for (int j = 0, idx = 0; j < block_size_j; j += 4, ++idx)
        {
            cache[idx] = _mm256_loadu_pd(&c[j]);
            res[idx]   = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j2]);
        }

        //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

        for (int k = 0; k < block_size_k; ++k, b += block_size_j)
        {
            __m256d areg = _mm256_broadcast_sd(&a[k]);

            for (int j2 = 0, idx = 0; j2 < block_size_j; j2 += 4, ++idx)
            {
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j2]), res[idx]);
            }
        }

        for (int j = 0, idx = 0; j < block_size_j; j += 4, ++idx)
        {
            __m256d result = _mm256_add_pd(cache[idx], res[idx]);
            _mm256_store_pd(&c[j], result);

            // load_inc_store_double(&c[j2], res[idx]);
        }
    }
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

static void mulMatrix_256VL_BL(double* __restrict c,
                               const double* __restrict a,
                               const double* __restrict b,
                               const std::size_t i_size,
                               const std::size_t j_size,
                               const std::size_t k_size,
                               const std::size_t ii,
                               const std::size_t jj,
                               const std::size_t kk)
{
    kernelMulMatrix_BL_NV(c, a, b, i_size, j_size, k_size, ii, jj, kk);

    //    size_t i_max = std::min(ii + block_size_i, i_size);
    //    size_t j_max = std::min(jj + block_size_j, j_size);
    //    size_t k_max = std::min(kk + block_size_k, k_size);

    //    auto vi = i_size;
    //    auto vj = j_size;
    //    auto vk = k_size;

    //    mulMatrix_x<block_size_i, block_size_j, block_size_k>(
    //      c, a, b, i_size, j_size, k_size, ii, jj, kk);
}

void kernelMulMatrix_TP_BL_NV(double*           r,
                              const double*     a,
                              const double*     mul2,
                              const std::size_t j_size,
                              const std::size_t k_size)
{
    const double* b;

    for (auto i = 0; i < block_size_i; ++i, r += j_size, a += k_size)
    {
        b = mul2;
        for (auto j = 0; j < block_size_j; ++j, b += k_size)
        {
            double t = 0;
            for (auto k = 0; k < block_size_k; ++k)
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
                              const std::size_t j_size,
                              const std::size_t k_size)
{
    // TODO: change constants to vars

    const double* b;
    for (auto i = 0; i < block_size_i; ++i, r += j_size, a += k_size)
    {
        b = mul2;
        for (auto j = 0; j < block_size_j; ++j, b += k_size)
        {
            //_mm_prefetch(&b[N], _MM_HINT_NTA);

            __m256d rk = _mm256_setzero_pd();
            for (auto k = 0; k < block_size_k; k += 4)
            {
                __m256d m1 = _mm256_loadu_pd(&a[k]);
                __m256d m2 = _mm256_loadu_pd(&b[k]);

                rk = _mm256_fmadd_pd(m2, m1, rk);
            }

            r[j] += sumElemFromReg(rk);
        }
    }
}

// TODO: Broken, unrol j loop for fixed block size
static void mulMatrix_128VL_BL(double*           rres,
                               const double*     rmul1,
                               const double*     m_mul2,
                               const std::size_t j_size,
                               const std::size_t k_size)
{
    // hardhoded blockszie for j == 8
    const double* rmul2 = m_mul2;
    for (int i2 = 0; i2 < block_size_i; ++i2, rres += j_size, rmul1 += k_size)
    {
        _mm_prefetch(&rmul1[block_size_i], _MM_HINT_NTA);
        rmul2 = m_mul2;

        __m128d r20 = _mm_load_pd(&rres[0]);
        __m128d r21 = _mm_load_pd(&rres[2]);
        __m128d r22 = _mm_load_pd(&rres[4]);
        __m128d r23 = _mm_load_pd(&rres[6]);

        for (int k2 = 0; k2 < block_size_k; k2 += 4, rmul2 += j_size)
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

void kernelMulMatrix_VT_BL(double* __restrict c,
                           const double* __restrict a,
                           const double* __restrict b,
                           const std::size_t i_size,
                           const std::size_t j_size,
                           const std::size_t k_size,
                           const std::size_t ii,
                           const std::size_t jj,
                           const std::size_t kk)
{

#if defined(__x86_64__)
#ifdef __AVX2__
    mulMatrix_256VL_BL(c, a, b, i_size, j_size, k_size, ii, jj, kk);
#elif __SSE2__
    mulMatrix_128VL_BL(c, a, b, j_size, k_size);
#else
#error "Vectorization is not implemented for current cpu arch!"
#endif
#elif defined(__ARM_ARCH)
#error "Manual vectorization is not implemented for ARM arch!"
#endif
}

} // namespace kernels
