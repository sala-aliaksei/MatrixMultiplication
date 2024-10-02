
#include "matrixMultiplication/matrix/MatrixMulFunctions.hpp"
#include "matrixMultiplication/utils/utils.hpp"

#include <immintrin.h>
#include <future>
#include <new>
#include <vector>
#include <numeric>

constexpr auto        nthreads   = 4u;
constexpr auto        SM         = 32; //(64 / sizeof(double));
constexpr std::size_t BLOCK_SIZE = SM;

/*****************     KERNELS     *******************/

static void kernelMulMatrix_BL_NV(double* m_res, const double* m_mul1, const double* m_mul2)
{
    const double *a, *b;
    double*       r;
    int           i2{}, k2{}, j2{};
    for (i2 = 0, r = m_res, a = m_mul1; i2 < SM; ++i2, r += N, a += N)
        for (k2 = 0, b = m_mul2; k2 < SM; ++k2, b += N)
            for (j2 = 0; j2 < SM; ++j2)
                r[j2] += a[k2] * b[j2];
}

static void kernelMulMatrix_TP_BL_NV(double* r, const double* a, const double* m_mul2)
{
    const double* b;

    for (auto i = 0; i < SM; ++i, r += N, a += N)
    {
        b = m_mul2;
        for (auto j = 0; j < SM; ++j, b += N)
        {
            double t = 0;
            for (auto k = 0; k < SM; ++k)
            {
                t += a[k] * b[k];
            }
            r[j] += t;
        }
    }
}

static double sumElemFromReg(__m256d c_vec)
{
    alignas(32) double c_array[4];
    _mm256_store_pd(c_array, c_vec);
    return c_array[0] + c_array[1] + c_array[2] + c_array[3];
}

static void kernelMulMatrix_VT_BL_TP(double* r, const double* a, const double* m_mul2)
{
    const double* b;
    for (auto i = 0; i < SM; ++i, r += N, a += N)
    {
        b = m_mul2;
        for (auto j = 0; j < SM; ++j, b += N)
        {
            __m256d rk = _mm256_setzero_pd();
            for (auto k = 0; k < SM; k += 4)
            {
                __m256d m1 = _mm256_loadu_pd(&a[k]);
                __m256d m2 = _mm256_loadu_pd(&b[k]);

                rk = _mm256_fmadd_pd(m2, m1, rk);
            }

            r[j] += sumElemFromReg(rk);
        }
    }
}

static void mulMatrix_128VL_BL(double* rres, const double* rmul1, const double* m_mul2)
{
    const double* rmul2 = m_mul2;
    for (int i2 = 0; i2 < SM; ++i2, rres += N, rmul1 += N)
    {
        _mm_prefetch(&rmul1[SM], _MM_HINT_NTA);
        rmul2 = m_mul2;

        __m128d r20 = _mm_load_pd(&rres[0]);
        __m128d r21 = _mm_load_pd(&rres[2]);
        __m128d r22 = _mm_load_pd(&rres[4]);
        __m128d r23 = _mm_load_pd(&rres[6]);

        for (int k2 = 0; k2 < SM; ++k2, rmul2 += N)
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

static void mulMatrix_256VL_BL(double* rres, const double* rmul1, const double* m_mul2)
{

    const double* rmul2 = m_mul2;
    for (int i2 = 0; i2 < SM; ++i2, rres += N, rmul1 += N)
    {
        _mm_prefetch(&rres[SM], _MM_HINT_T0);
        _mm_prefetch(&rmul1[SM], _MM_HINT_T0);

        rmul2 = m_mul2;

        __m256d r20 = _mm256_loadu_pd(&rres[0]);
        __m256d r22 = _mm256_loadu_pd(&rres[4]);

        for (int k2 = 0; k2 < SM; ++k2, rmul2 += N)
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

static void kernelMulMatrix_VT_BL(double*       c,
                                  const double* a,
                                  const double* b,
                                  std::size_t   block_size)
{

#if defined(__x86_64__)
#ifdef __AVX2__
    mulMatrix_256VL_BL(c, a, b);
#elif __SSE2__
    mulMatrix_128VL_BL(c, a, b);
#else
#error "Manual vectorization is not supported for current cpu"
#endif
#elif defined(__ARM_ARCH) // TODO: Verify macro
    kernelMulMatrix_BL_NV(c, a, b);
#endif
}

//////////////////// END OF NAIVE

/***********************   Multithreaded Optimized   *********************/

static void mulPerThread(double* m_res, double* m_mul1, double* m_mul2, std::size_t t)
{
    constexpr std::size_t step = int(N / (8 * 4)) * 8;

    std::size_t start = t * step;
    std::size_t last  = t == 3 ? N : (t + 1) * step;
    for (int i = start; i < last; i += SM)
    {
        for (int j = 0; j < N; j += SM)
        {
            for (int k = 0; k < N; k += SM)
            {
                kernelMulMatrix_VT_BL(
                  &m_res[i * N + j], &m_mul1[i * N + k], &m_mul2[k * N + j], SM);
            }
        }
    }
}
static void mulTransposedPerThread(double* m_res, double* m_mul1, double* m_mul2, std::size_t t)
{
    constexpr std::size_t step = int(N / (8 * 4)) * 8;

    std::size_t start = t * step;
    std::size_t last  = t == 3 ? N : (t + 1) * step;
    for (int i = start; i < last; i += 1)
    {
        for (int j = 0; j < N; j += 1)
        {
            __m256d rk = _mm256_setzero_pd();
            for (int k = 0; k < N; k += SM)
            {
                auto* a = &m_mul1[i * N + k];
                auto* b = &m_mul2[j * N + k];

                for (auto kk = 0; kk < SM; kk += 4)
                {
                    __m256d m1 = _mm256_loadu_pd(&a[kk]);
                    __m256d m2 = _mm256_loadu_pd(&b[kk]);

                    rk = _mm256_fmadd_pd(m2, m1, rk);
                }
            }
            auto* r = &m_res[i * N + j];
            *r += sumElemFromReg(rk);
        }
    }
}

static void mulTransposedPerThreadOld(double*     m_res,
                                      double*     m_mul1,
                                      double*     m_mul2,
                                      std::size_t thread_id)
{
    constexpr std::size_t step  = N / 4;
    std::size_t           start = thread_id * step;
    std::size_t           last  = thread_id == 3 ? N : (thread_id + 1) * step;

    std::size_t block_size = SM;
    for (int i = start; i < last; i += block_size)
    {
        for (int j = 0; j < N; j += block_size)
        {
            for (int k = 0; k < N; k += block_size)
            {
                kernelMulMatrix_VT_BL_TP(&m_res[i * N + j], &m_mul1[i * N + k], &m_mul2[j * N + k]);
            }
        }
    }
}

/********************************************************************************************/

/////////////     NAIVE     ///////////////////

void matrixMul_Naive(MatrixSet& ms)
{
    auto res = ms.c.data();
    auto a   = ms.a.data();
    auto b   = ms.b.data();

    for (auto i = 0; i < N; ++i)
        for (auto j = 0; j < N; ++j)
            for (auto k = 0; k < N; ++k)
            {
                res[i * N + j] += a[i * N + k] * b[k * N + j];
            }
}

void matrixMul_Naive_TP(MatrixSet& set)
{
    auto res = set.c.data();
    auto a   = set.a.data();

    auto transposed = transpose(set.b);
    auto b          = transposed.data();

    for (auto i = 0; i < N; ++i)
        for (auto j = 0; j < N; ++j)
            for (auto k = 0; k < N; ++k)
            {
                res[i * N + j] += a[i * N + k] * b[j * N + k];
            }
}

void matrixMul_MT_VT_BL(MatrixSet& ms)
{
    std::vector<std::future<void>> fret(std::thread::hardware_concurrency());
    for (int tid = 0; tid < fret.size(); ++tid)
    {
        fret[tid] = std::async(mulPerThread, ms.c.data(), ms.a.data(), ms.b.data(), tid);
    }

    fret.resize(0); // wait all threads
}

void matrixMul_MT_VT_BL_TP(MatrixSet& ms)
{
    auto transposed = transpose(ms.b);

    std::vector<std::future<void>> fret(std::thread::hardware_concurrency());
    for (int tid = 0; tid < fret.size(); ++tid)
    {
        fret[tid] =
          std::async(mulTransposedPerThread, ms.c.data(), ms.a.data(), transposed.data(), tid);
    }

    fret.resize(0);
}
