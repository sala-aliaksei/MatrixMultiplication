#include "MatrixMul.hpp"
#include <cstring> // memcpy
#include <cmath>
#include <array>

#include <immintrin.h>

namespace cppnow
{

template<int I, int J>
std::array<double, I * J> packMatrix(const double* b, int j_size)
{

    constexpr int istep{I / 2};
    constexpr int jstep{J / 2};
    static_assert(I % 2 == 0, "");
    static_assert(J % 2 == 0, "");

    std::array<double, I * J> b_packed;
    for (int i = 0; i < I; i += istep)
    {

        for (int j = 0; j < J; j += jstep)
        {
            for (int i2 = 0; i2 < istep; i2++)
            {
                //_mm_prefetch(&b[(i + i2) * j_size + j], _MM_HINT_NTA);
                std::memcpy(&b_packed[(i + i2) * J] + j, &b[(i + i2) * j_size + j], jstep * 8);
            }
        }
    }

    return b_packed;
}

__attribute__((always_inline)) static inline void load_inc_store_double(double* __restrict ptr,
                                                                        __m256d increment)
{
    // Load 4 double-precision values (256 bits) from memory into an AVX register
    __m256d vector = _mm256_load_pd(ptr);

    // Add the increment to the loaded vector
    __m256d result = _mm256_add_pd(vector, increment);

    // Store the result back to memory
    _mm256_store_pd(ptr, result);
    //_mm256_stream_pd(ptr, result);
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_naive(double* __restrict c,
                      const double* __restrict a,
                      const double* __restrict bb,
                      int N,
                      int K)
{
    for (int i = 0; i < Mr; ++i, c += N, a += K)
    {
        const double* b = bb;
        for (int k = 0; k < Kc; ++k, b += N)
        {
            __m256d m1d = _mm256_broadcast_sd(&a[k]);
            for (int j = 0; j < Nr; j += 4)
            {
                __m256d r20 = _mm256_loadu_pd(&c[j]);
                r20         = _mm256_fmadd_pd(m1d, _mm256_loadu_pd(&b[j]), r20);
                _mm256_storeu_pd(&c[j], r20);
            }
        }
    }
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs(double* __restrict c,
                     const double* __restrict a,
                     const double* __restrict bb,
                     int N,
                     int K)
{
    constexpr int REG_CNT{3};
    for (int i = 0; i < Mr; ++i, c += N, a += K)
    {
        const double* b = bb;

        std::array<__m256d, REG_CNT> res;
        for (int j = 0, idx = 0; j < Nr; j += 4, ++idx)
        {
            res[idx] = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j]);
        }

        //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

        for (int k = 0; k < Kc; ++k, b += N)
        {
            __m256d areg = _mm256_broadcast_sd(&a[k]);
            for (int j = 0, idx = 0; j < Nr; j += 4, ++idx)
            {
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
        }

        for (int j = 0, idx = 0; j < Nr; j += 4, ++idx)
        {
            __m256d cr     = _mm256_loadu_pd(&c[j]);
            __m256d result = _mm256_add_pd(cr, res[idx]);
            _mm256_store_pd(&c[j], result);
        }
    }
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_v2(double* __restrict c,
                        const double* __restrict ma,
                        const double* __restrict mb,
                        int N,
                        int K)
{
    constexpr int REG_CNT{3};

    const double* b = mb;

    std::array<__m256d, REG_CNT> res;
    for (int j = 0, idx = 0; j < Nr; j += 4, ++idx)
    {
        res[idx] = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j]);
    }

    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

    for (int k = 0; k < Kc; ++k, b += N)
    {
        const double* a = ma;
        for (int i = 0; i < Mr; ++i, a += K)
        {
            __m256d areg = _mm256_broadcast_sd(&a[k]);
            for (int j = 0, idx = 0; j < Nr; j += 4, ++idx)
            {
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
        }
    }

    for (int j = 0, idx = 0; j < Nr; j += 4, ++idx)
    {
        __m256d cr     = _mm256_loadu_pd(&c[j]);
        __m256d result = _mm256_add_pd(cr, res[idx]);
        _mm256_store_pd(&c[j], result);
    }
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_unroll(double* __restrict c,
                            const double* __restrict aa,
                            const double* __restrict b,
                            int N,
                            int K)
{
    __m256d r00 = _mm256_setzero_pd();
    __m256d r01 = _mm256_setzero_pd();
    __m256d r02 = _mm256_setzero_pd();

    __m256d r10 = _mm256_setzero_pd();
    __m256d r11 = _mm256_setzero_pd();
    __m256d r12 = _mm256_setzero_pd();

    __m256d r20 = _mm256_setzero_pd();
    __m256d r21 = _mm256_setzero_pd();
    __m256d r22 = _mm256_setzero_pd();

    __m256d r30 = _mm256_setzero_pd();
    __m256d r31 = _mm256_setzero_pd();
    __m256d r32 = _mm256_setzero_pd();

    const double* a = aa;
    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        a = aa;

        __m256d b0 = _mm256_loadu_pd(&b[0]);
        __m256d b1 = _mm256_loadu_pd(&b[4]);
        __m256d b2 = _mm256_loadu_pd(&b[8]);

        __m256d a0 = _mm256_broadcast_sd(&a[k2]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);
    }

    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_unroll_rw(double* __restrict mc,
                               const double* __restrict aa,
                               const double* __restrict b,
                               int N,
                               int K)
{
    double* c = mc;
    _mm_prefetch(c + N, _MM_HINT_NTA);
    __m256d r00 = _mm256_load_pd(&c[0]);
    __m256d r01 = _mm256_load_pd(&c[4]);
    __m256d r02 = _mm256_load_pd(&c[8]);

    c += N;
    _mm_prefetch(c + N, _MM_HINT_NTA);
    __m256d r10 = _mm256_load_pd(&c[0]);
    __m256d r11 = _mm256_load_pd(&c[4]);
    __m256d r12 = _mm256_load_pd(&c[8]);

    c += N;
    _mm_prefetch(c + N, _MM_HINT_NTA);
    __m256d r20 = _mm256_load_pd(&c[0]);
    __m256d r21 = _mm256_load_pd(&c[4]);
    __m256d r22 = _mm256_load_pd(&c[8]);

    c += N;
    __m256d r30 = _mm256_load_pd(&c[0]);
    __m256d r31 = _mm256_load_pd(&c[4]);
    __m256d r32 = _mm256_load_pd(&c[8]);

    const double* a = aa;
    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        a = aa;

        __m256d b0 = _mm256_loadu_pd(&b[0]);
        __m256d b1 = _mm256_loadu_pd(&b[4]);
        __m256d b2 = _mm256_loadu_pd(&b[8]);

        __m256d a0 = _mm256_broadcast_sd(&a[k2]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);
    }

    c = mc;
    //_mm_prefetch(c + N, _MM_HINT_NTA);

    _mm256_store_pd(&c[0], r00);
    _mm256_store_pd(&c[4], r01);
    _mm256_store_pd(&c[8], r02);
    c += N;

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    _mm256_store_pd(&c[0], r10);
    _mm256_store_pd(&c[4], r11);
    _mm256_store_pd(&c[8], r12);
    c += N;

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    _mm256_store_pd(&c[0], r20);
    _mm256_store_pd(&c[4], r21);
    _mm256_store_pd(&c[8], r22);
    c += N;

    _mm256_store_pd(&c[0], r30);
    _mm256_store_pd(&c[4], r31);
    _mm256_store_pd(&c[8], r32);
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_unroll_kr(double* __restrict c,
                               const double* __restrict ma,
                               const double* __restrict mb,
                               int N,
                               int K)
{
    constexpr int Kr = 8;

    __m256d r00 = _mm256_setzero_pd();
    __m256d r01 = _mm256_setzero_pd();
    __m256d r02 = _mm256_setzero_pd();

    __m256d r10 = _mm256_setzero_pd();
    __m256d r11 = _mm256_setzero_pd();
    __m256d r12 = _mm256_setzero_pd();

    __m256d r20 = _mm256_setzero_pd();
    __m256d r21 = _mm256_setzero_pd();
    __m256d r22 = _mm256_setzero_pd();

    __m256d r30 = _mm256_setzero_pd();
    __m256d r31 = _mm256_setzero_pd();
    __m256d r32 = _mm256_setzero_pd();

    for (int k = 0; k < Kc; k += Kr)
    {

        const double* b = &mb[k * N];

        const double* a = &ma[k];
        for (int k2 = 0; k2 < Kr; k2++, b += N)
        {
            a = &ma[k];

            __m256d b0 = _mm256_loadu_pd(&b[0]);
            __m256d b1 = _mm256_loadu_pd(&b[4]);
            __m256d b2 = _mm256_loadu_pd(&b[8]);

            __m256d a0 = _mm256_broadcast_sd(&a[k2]);

            r00 = _mm256_fmadd_pd(a0, b0, r00);
            r01 = _mm256_fmadd_pd(a0, b1, r01);
            r02 = _mm256_fmadd_pd(a0, b2, r02);

            a += K;
            a0 = _mm256_broadcast_sd(&a[k2]);

            r10 = _mm256_fmadd_pd(a0, b0, r10);
            r11 = _mm256_fmadd_pd(a0, b1, r11);
            r12 = _mm256_fmadd_pd(a0, b2, r12);

            a += K;
            a0 = _mm256_broadcast_sd(&a[k2]);

            r20 = _mm256_fmadd_pd(a0, b0, r20);
            r21 = _mm256_fmadd_pd(a0, b1, r21);
            r22 = _mm256_fmadd_pd(a0, b2, r22);

            a += K;
            a0 = _mm256_broadcast_sd(&a[k2]);

            r30 = _mm256_fmadd_pd(a0, b0, r30);
            r31 = _mm256_fmadd_pd(a0, b1, r31);
            r32 = _mm256_fmadd_pd(a0, b2, r32);
        }
    }

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);
    c += N;

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_unroll_bpack(double* __restrict c,
                                  const double* __restrict aa,
                                  const double* __restrict b,
                                  int N,
                                  int K)
{
    __m256d r00 = _mm256_setzero_pd();
    __m256d r01 = _mm256_setzero_pd();
    __m256d r02 = _mm256_setzero_pd();

    __m256d r10 = _mm256_setzero_pd();
    __m256d r11 = _mm256_setzero_pd();
    __m256d r12 = _mm256_setzero_pd();

    __m256d r20 = _mm256_setzero_pd();
    __m256d r21 = _mm256_setzero_pd();
    __m256d r22 = _mm256_setzero_pd();

    __m256d r30 = _mm256_setzero_pd();
    __m256d r31 = _mm256_setzero_pd();
    __m256d r32 = _mm256_setzero_pd();

    constexpr auto b_cols = Nc;

    const double* a = aa;
    for (int k2 = 0; k2 < Kc; ++k2, b += b_cols)
    {
        a = aa;

        __m256d b0 = _mm256_loadu_pd(&b[0]);
        _mm_prefetch(b + 8, _MM_HINT_T0);
        __m256d b1 = _mm256_loadu_pd(&b[4]);

        __m256d a0 = _mm256_broadcast_sd(&a[k2]);
        __m256d b2 = _mm256_loadu_pd(&b[8]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        a += K;
        a0 = _mm256_broadcast_sd(&a[k2]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);
    }

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);
    c += N;

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    //_mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_unroll_bpack_kr(double* __restrict c,
                                     const double* __restrict ma,
                                     const double* __restrict mb,
                                     int N,
                                     int K)
{
    constexpr int Kr  = 8;
    __m256d       r00 = _mm256_setzero_pd();
    __m256d       r01 = _mm256_setzero_pd();
    __m256d       r02 = _mm256_setzero_pd();

    __m256d r10 = _mm256_setzero_pd();
    __m256d r11 = _mm256_setzero_pd();
    __m256d r12 = _mm256_setzero_pd();

    __m256d r20 = _mm256_setzero_pd();
    __m256d r21 = _mm256_setzero_pd();
    __m256d r22 = _mm256_setzero_pd();

    __m256d r30 = _mm256_setzero_pd();
    __m256d r31 = _mm256_setzero_pd();
    __m256d r32 = _mm256_setzero_pd();

    constexpr auto b_cols = Nc;

    for (int k = 0; k < Kc; k += Kr)
    {

        const double* b = &mb[k * b_cols];
        const double* a = &ma[k];

        for (int k2 = 0; k2 < Kr; k2++, b += b_cols)
        {
            a = &ma[k];

            __m256d b0 = _mm256_loadu_pd(&b[0]);
            __m256d b1 = _mm256_loadu_pd(&b[4]);
            __m256d b2 = _mm256_loadu_pd(&b[8]);

            __m256d a0 = _mm256_broadcast_sd(&a[k2]);

            r00 = _mm256_fmadd_pd(a0, b0, r00);
            r01 = _mm256_fmadd_pd(a0, b1, r01);
            r02 = _mm256_fmadd_pd(a0, b2, r02);

            a += K;
            a0 = _mm256_broadcast_sd(&a[k2]);

            r10 = _mm256_fmadd_pd(a0, b0, r10);
            r11 = _mm256_fmadd_pd(a0, b1, r11);
            r12 = _mm256_fmadd_pd(a0, b2, r12);

            a += K;
            a0 = _mm256_broadcast_sd(&a[k2]);

            r20 = _mm256_fmadd_pd(a0, b0, r20);
            r21 = _mm256_fmadd_pd(a0, b1, r21);
            r22 = _mm256_fmadd_pd(a0, b2, r22);

            a += K;
            a0 = _mm256_broadcast_sd(&a[k2]);

            r30 = _mm256_fmadd_pd(a0, b0, r30);
            r31 = _mm256_fmadd_pd(a0, b1, r31);
            r32 = _mm256_fmadd_pd(a0, b2, r32);
        }
    }

    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
}

/////////////////////////////////////////////////////////////////////////////////////////////

void matMulNaive(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < K; ++k)
            {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

void matMul_Naive(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    for (int i = 0; i < i_size; ++i, c += j_size, a += k_size)
    {
        for (int j = 0; j < j_size; ++j)
        {
            b = B.data();
            for (int k = 0; k < k_size; ++k, b += j_size)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

// TODO: if j_size > L3, what is the impact
void matMul_Naive_Order(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    for (int i = 0; i < i_size; ++i, c += j_size, a += k_size)
    {
        b = B.data();
        for (int k = 0; k < k_size; ++k, b += j_size)
        {
            for (int j = 0; j < j_size; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

void matMul_Naive_Order_KIJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    for (int k = 0; k < k_size; ++k, b += j_size)
    {
        a = A.data();
        c = C.data();
        for (int i = 0; i < i_size; ++i, c += j_size, a += k_size)
        {
            for (int j = 0; j < j_size; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

void matMul_Naive_Block(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    if ((i_size % 64 != 0) || (k_size % 64 != 0) || (j_size % 64 != 0))
    {
        throw std::runtime_error("size % BLOCK != 0");
    }

    constexpr auto BLOCK = 64;

    for (int ib = 0; ib < i_size; ib += BLOCK)
    {
        for (int kb = 0; kb < k_size; kb += BLOCK)
        {
            for (int jb = 0; jb < j_size; jb += BLOCK)
            {
                double*       c  = &mc[ib * j_size + jb];
                const double* a  = &ma[ib * k_size + kb];
                const double* bb = &mb[kb * j_size + jb];

                for (int i = 0; i < BLOCK; ++i, c += j_size, a += k_size)
                {
                    const double* b = bb;
                    for (int k = 0; k < BLOCK; ++k, b += j_size)
                    {
                        for (int j = 0; j < BLOCK; ++j)
                        {
                            // c[j] = std::fma(a[k], b[j], c[j]);
                            c[j] += a[k] * b[j];
                        }
                    }
                }
            }
        }
    }
}

void matMul_Simd_Global() {}

void matMul_Simd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    if ((i_size % 64 != 0) || (k_size % 64 != 0) || (j_size % 64 != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    constexpr auto BLOCK = 64;

    for (int ib = 0; ib < i_size; ib += BLOCK)
    {
        for (int kb = 0; kb < k_size; kb += BLOCK)
        {
            for (int jb = 0; jb < j_size; jb += BLOCK)
            {
                double*       c  = &mc[ib * j_size + jb];
                const double* a  = &ma[ib * k_size + kb];
                const double* bb = &mb[kb * j_size + jb];

                for (int i = 0; i < BLOCK; ++i, c += j_size, a += k_size)
                {
                    const double* b = bb;
                    for (int k = 0; k < BLOCK; ++k, b += j_size)
                    {
                        __m128d m1d = _mm_load_sd(&a[k]);
                        m1d         = _mm_unpacklo_pd(m1d, m1d);
                        for (int j = 0; j < BLOCK; j += 2)
                        {
                            __m128d m2 = _mm_load_pd(&b[j]);
                            __m128d r2 = _mm_load_pd(&c[j]);
                            _mm_store_pd(&c[j], _mm_add_pd(_mm_mul_pd(m2, m1d), r2));
                        }
                    }
                }
            }
        }
    }
}

void matMul_Avx(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    if ((i_size % 64 != 0) || (k_size % 64 != 0) || (j_size % 64 != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    constexpr auto BLOCK = 64;

    for (int ib = 0; ib < i_size; ib += BLOCK)
    {
        for (int kb = 0; kb < k_size; kb += BLOCK)
        {
            for (int jb = 0; jb < j_size; jb += BLOCK)
            {
                double*       c  = &mc[ib * j_size + jb];
                const double* bb = &mb[kb * j_size + jb];
                const double* a  = &ma[ib * k_size + kb];

                for (int i = 0; i < BLOCK; ++i, c += j_size, a += k_size)
                {
                    const double* b = bb;

                    for (int k = 0; k < BLOCK; ++k, b += j_size)
                    {
                        __m256d m1d = _mm256_broadcast_sd(&a[k]);
                        for (int j = 0; j < BLOCK; j += 4)
                        {
                            __m256d r20 = _mm256_loadu_pd(&c[j]);
                            r20         = _mm256_fmadd_pd(m1d, _mm256_loadu_pd(&b[j]), r20);
                            _mm256_storeu_pd(&c[j], r20);
                        }
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];
                const double* b2 = &mb[kb * N + jb];
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];
                        cppnow::avx_naive<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache_Regs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];
                const double* b2 = &mb[kb * N + jb];
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {

                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];

                        cppnow::avx_regs<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                        // cppnow::avx_regs_v2<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
    //        for (int ib = 0; ib < i_size; ib += Mc)
    //        {
    //            for (int kb = 0; kb < k_size; kb += Kc)
    //            {

    //                for (int jb = 0; jb < j_size; jb += Nc)
    //                {
    //                    double*       c  = &mc[ib * j_size + jb];
    //                    const double* a  = &ma[ib * k_size + kb];
    //                    const double* bb = &mb[kb * j_size + jb];

    //                    for (int i = 0; i < Mc; ++i, c += j_size, a += k_size)
    //                    {
    //                        constexpr int REG_CNT{64 / 4};

    //                        const double* b = bb;

    //                        std::array<__m256d, REG_CNT> cache;
    //                        std::array<__m256d, REG_CNT> res;
    //                        for (int j = 0, idx = 0; j < Nc; j += 4, ++idx)
    //                        {
    //                            cache[idx] = _mm256_loadu_pd(&c[j]);
    //                            res[idx]   = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j2]);
    //                        }

    //                        //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

    //                        for (int k = 0; k < Kc; ++k, b += j_size)
    //                        {
    //                            __m256d areg = _mm256_broadcast_sd(&a[k]);
    //                            for (int j = 0, idx = 0; j < Nc; j += 4, ++idx)
    //                            {
    //                                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]),
    //                                res[idx]);
    //                            }
    //                        }

    //                        for (int j = 0, idx = 0; j < Nc; j += 4, ++idx)
    //                        {
    //                            __m256d result = _mm256_add_pd(cache[idx], res[idx]);
    //                            _mm256_store_pd(&c[j], result);
    //                        }
    //                    }
    //                }
    //            }
    //        }
}

void matMul_Avx_Cache_Regs_Unroll(const Matrix<double>& A,
                                  const Matrix<double>& B,
                                  Matrix<double>&       C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];
                const double* b2 = &mb[kb * N + jb];
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {

                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];
                        cppnow::avx_regs_unroll<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache_Regs_UnrollRW(const Matrix<double>& A,
                                    const Matrix<double>& B,
                                    Matrix<double>&       C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];
                const double* b2 = &mb[kb * N + jb];
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {

                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];
                        cppnow::avx_regs_unroll_rw<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache_Regs_Unroll_BPack(const Matrix<double>& A,
                                        const Matrix<double>& B,
                                        Matrix<double>&       C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];

                auto          arr = packMatrix<Kc, Nc>(&mb[kb * N + jb], N);
                const double* b2  = arr.data();

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];
                        cppnow::avx_regs_unroll_bpack<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache_Regs_Unroll_MT(const Matrix<double>& A,
                                     const Matrix<double>& B,
                                     Matrix<double>&       C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

#pragma omp parallel for
    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];
                const double* b2 = &mb[kb * N + jb];
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {

                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];
                        cppnow::avx_regs_unroll_kr<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache_Regs_Unroll_BPack_MT(const Matrix<double>& A,
                                           const Matrix<double>& B,
                                           Matrix<double>&       C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

#pragma omp parallel for
    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* a2 = &ma[ib * K + kb];

                auto          arr = packMatrix<Kc, Nc>(&mb[kb * N + jb], N);
                const double* b2  = arr.data();

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* b = &b2[j2];
                        double*       c = &c2[i2 * N + j2];
                        const double* a = &a2[i2 * K];
                        cppnow::avx_regs_unroll_bpack<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

// void matMul_Avx_Cache_Regs_Unroll(const Matrix<double>& A,
//                                   const Matrix<double>& B,
//                                   Matrix<double>&       C)
//{
//     auto i_size = A.row();
//     auto k_size = A.col();
//     auto j_size = B.col();

//    const double* ma = A.data();
//    const double* mb = B.data();
//    double*       mc = C.data();

//    constexpr auto Mc = 180;
//    constexpr auto Nc = 48;
//    constexpr auto Kc = 96;

//    constexpr int REG_CNT{64 / 4};

//    if ((i_size % Mc != 0) || (k_size % Kc != 0) || (j_size % Nc != 0))
//    {
//        throw std::runtime_error("size % BLOKC != 0");
//    }

//    for (int ib = 0; ib < i_size; ib += Mc)
//    {
//        for (int kb = 0; kb < k_size; kb += Kc)
//        {

//            for (int jb = 0; jb < j_size; jb += Nc)
//            {
//                double*       c  = &mc[ib * j_size + jb];
//                const double* a  = &ma[ib * k_size + kb];
//                const double* bb = &mb[kb * j_size + jb];

//                for (int i = 0; i < Mc; ++i, c += j_size, a += k_size)
//                {
//                    const double* b = bb;

//                    std::array<__m256d, REG_CNT> cache;
//                    std::array<__m256d, REG_CNT> res;
//                    for (int j = 0, idx = 0; j < Nc; j += 4, ++idx)
//                    {
//                        cache[idx] = _mm256_loadu_pd(&c[j]);
//                        res[idx]   = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j2]);
//                    }

//                    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

//                    for (int k = 0; k < Kc; ++k, b += j_size)
//                    {
//                        __m256d areg = _mm256_broadcast_sd(&a[k]);
//                        for (int j = 0, idx = 0; j < Nc; j += 4, ++idx)
//                        {
//                            res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
//                        }
//                    }

//                    for (int j = 0, idx = 0; j < Nc; j += 4, ++idx)
//                    {
//                        __m256d result = _mm256_add_pd(cache[idx], res[idx]);
//                        _mm256_store_pd(&c[j], result);
//                    }
//                }
//            }
//        }
//    }
//}

void matMul_Avx_AddRegs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // it is not unroll, rename function
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    if ((i_size % 64 != 0) || (k_size % 64 != 0) || (j_size % 64 != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    constexpr auto BLOCK = 64;
    constexpr int  REG_CNT{BLOCK / 4};

    for (int ib = 0; ib < i_size; ib += BLOCK)
    {
        for (int kb = 0; kb < k_size; kb += BLOCK)
        {
            for (int jb = 0; jb < j_size; jb += BLOCK)
            {
                double*       c  = &mc[ib * j_size + jb];
                const double* a  = &ma[ib * k_size + kb];
                const double* bb = &mb[kb * j_size + jb];

                for (int i = 0; i < BLOCK; ++i, c += j_size, a += k_size)
                {
                    const double* b = bb;

                    std::array<__m256d, REG_CNT> cache;
                    std::array<__m256d, REG_CNT> res;
                    for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                    {
                        res[idx] = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j2]);
                    }

                    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

                    for (int k = 0; k < BLOCK; ++k, b += j_size)
                    {
                        __m256d areg = _mm256_broadcast_sd(&a[k]);
                        for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                        {
                            res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
                        }
                    }

                    for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                    {
                        __m256d creg   = _mm256_loadu_pd(&c[j]);
                        __m256d result = _mm256_add_pd(creg, res[idx]);
                        _mm256_store_pd(&c[j], result);
                    }
                }
            }
        }
    }
}

void matMul_Avx_AddRegsV2(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // it is not unroll, rename function
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    if ((i_size % 64 != 0) || (k_size % 64 != 0) || (j_size % 64 != 0))
    {
        throw std::runtime_error("size % BLOKC != 0");
    }

    constexpr auto BLOCK = 64;
    constexpr int  REG_CNT{BLOCK / 4};

    for (int ib = 0; ib < i_size; ib += BLOCK)
    {
        for (int kb = 0; kb < k_size; kb += BLOCK)
        {
            for (int jb = 0; jb < j_size; jb += BLOCK)
            {
                double*       c  = &mc[ib * j_size + jb];
                const double* a  = &ma[ib * k_size + kb];
                const double* bb = &mb[kb * j_size + jb];

                for (int i = 0; i < BLOCK; ++i, c += j_size, a += k_size)
                {
                    const double* b = bb;

                    std::array<__m256d, REG_CNT> cache;
                    std::array<__m256d, REG_CNT> res;
                    for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                    {
                        res[idx] = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j2]);
                    }

                    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

                    for (int k = 0; k < BLOCK; ++k, b += j_size)
                    {
                        __m256d areg = _mm256_broadcast_sd(&a[k]);
                        for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                        {
                            res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
                        }
                    }

                    for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                    {
                        __m256d creg   = _mm256_loadu_pd(&c[j]);
                        __m256d result = _mm256_add_pd(creg, res[idx]);
                        _mm256_store_pd(&c[j], result);
                    }
                }
            }
        }
    }
}

void matMul_Avx_AddRegs_Unroll(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // + remove k loop
    auto i_size = A.row();
    auto k_size = A.col();
    auto j_size = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr auto IBLOCK = 4;
    constexpr auto JBLOCK = 12;

    if ((i_size % IBLOCK != 0) || (j_size % JBLOCK != 0))
    {
        throw std::runtime_error("size % BLOCK != 0");
    }

    auto a_cols = k_size;
    auto b_cols = j_size;

    for (int ib = 0; ib < i_size; ib += IBLOCK)
    {
        for (int jb = 0; jb < j_size; jb += JBLOCK)
        {
            double*       c  = &mc[ib * j_size + jb];
            const double* aa = &ma[ib * k_size];
            const double* b  = &mb[jb];

            __m256d r00 = _mm256_setzero_pd();
            __m256d r01 = _mm256_setzero_pd();
            __m256d r02 = _mm256_setzero_pd();

            __m256d r10 = _mm256_setzero_pd();
            __m256d r11 = _mm256_setzero_pd();
            __m256d r12 = _mm256_setzero_pd();

            __m256d r20 = _mm256_setzero_pd();
            __m256d r21 = _mm256_setzero_pd();
            __m256d r22 = _mm256_setzero_pd();

            __m256d r30 = _mm256_setzero_pd();
            __m256d r31 = _mm256_setzero_pd();
            __m256d r32 = _mm256_setzero_pd();

            const double* a = aa;
            for (int k2 = 0; k2 < k_size; ++k2, b += b_cols)
            {
                a = aa;

                __m256d b0 = _mm256_loadu_pd(&b[0]);
                __m256d b1 = _mm256_loadu_pd(&b[4]);
                __m256d b2 = _mm256_loadu_pd(&b[8]);

                __m256d a0 = _mm256_broadcast_sd(&a[k2]);

                r00 = _mm256_fmadd_pd(a0, b0, r00);
                r01 = _mm256_fmadd_pd(a0, b1, r01);
                r02 = _mm256_fmadd_pd(a0, b2, r02);

                a += a_cols;
                a0 = _mm256_broadcast_sd(&a[k2]);

                r10 = _mm256_fmadd_pd(a0, b0, r10);
                r11 = _mm256_fmadd_pd(a0, b1, r11);
                r12 = _mm256_fmadd_pd(a0, b2, r12);

                a += a_cols;
                a0 = _mm256_broadcast_sd(&a[k2]);

                r20 = _mm256_fmadd_pd(a0, b0, r20);
                r21 = _mm256_fmadd_pd(a0, b1, r21);
                r22 = _mm256_fmadd_pd(a0, b2, r22);

                a += a_cols;
                a0 = _mm256_broadcast_sd(&a[k2]);

                r30 = _mm256_fmadd_pd(a0, b0, r30);
                r31 = _mm256_fmadd_pd(a0, b1, r31);
                r32 = _mm256_fmadd_pd(a0, b2, r32);
            }

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r00);
            load_inc_store_double(&c[4], r01);
            load_inc_store_double(&c[8], r02);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r10);
            load_inc_store_double(&c[4], r11);
            load_inc_store_double(&c[8], r12);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r20);
            load_inc_store_double(&c[4], r21);
            load_inc_store_double(&c[8], r22);
            c += j_size;

            load_inc_store_double(&c[0], r30);
            load_inc_store_double(&c[4], r31);
            load_inc_store_double(&c[8], r32);
        }
    }
}

void matMul_Avx_Unroll_Cache_Regs(const Matrix<double>& A,
                                  const Matrix<double>& B,
                                  Matrix<double>&       C)
{
}

void matMul_Avx_Unroll_Cache_Regs_Rename(const Matrix<double>& A,
                                         const Matrix<double>& B,
                                         Matrix<double>&       C)
{
}

void matMul_Avx_Unroll_Cache_Regs_Rename_PackB(const Matrix<double>& A,
                                               const Matrix<double>& B,
                                               Matrix<double>&       C)
{
}

void matMul_Avx_Unroll_Cache_Regs_Rename_ReorderAB(const Matrix<double>& A,
                                                   const Matrix<double>& B,
                                                   Matrix<double>&       C)
{
}

void matMul_Avx_Unroll_Cache_Regs_Rename_ReorderAB_Multithreads(const Matrix<double>& A,
                                                                const Matrix<double>& B,
                                                                Matrix<double>&       C)
{
}

} // namespace cppnow
