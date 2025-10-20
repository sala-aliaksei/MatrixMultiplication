#pragma once
#include <array>
#include <immintrin.h>
namespace ikernels
{
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

template<int BLOCK>
inline void naive_block(const double* __restrict a,
                        const double* __restrict mb,
                        double* __restrict c,
                        int N,
                        int K)
{
    for (int i = 0; i < BLOCK; ++i, c += N, a += K)
    {
        const double* b = mb;
        for (int k = 0; k < BLOCK; ++k, b += N)
        {
            for (int j = 0; j < BLOCK; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

template<int BLOCK>
inline void simd_block(const double* __restrict a,
                       const double* __restrict mb,
                       double* __restrict c,
                       int N,
                       int K)
{
    for (int i = 0; i < BLOCK; ++i, c += N, a += K)
    {
        const double* b = mb;
        for (int k = 0; k < BLOCK; ++k, b += N)
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

template<int BLOCK>
inline void avx_block(const double* __restrict a,
                      const double* __restrict mb,
                      double* __restrict c,
                      int N,
                      int K)
{
    for (int i = 0; i < BLOCK; ++i, c += N, a += K)
    {
        const double* b = mb;

        for (int k = 0; k < BLOCK; ++k, b += N)
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

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_naive(double* __restrict c,
                      const double* __restrict a,
                      const double* __restrict mb,
                      int N,
                      int K)
{
    for (int i = 0; i < Mr; ++i, c += N, a += K)
    {
        const double* b = mb;
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
                     const double* __restrict ma,
                     const double* __restrict mb,
                     int N,
                     int K)
{
    constexpr int CREG_CNT{Mr * Nr / 4};

    const double* b = mb;

    std::array<__m256d, CREG_CNT> res;
    for (int idx = 0; idx < CREG_CNT; ++idx)
    {
        res[idx] = _mm256_setzero_pd();
    }

    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

    for (int k = 0; k < Kc; ++k, b += N)
    {
        const double* a = ma;

        int idx = 0;
        for (int i = 0; i < Mr; ++i, a += K)
        {
            __m256d areg = _mm256_broadcast_sd(&a[k]);
            for (int j = 0; j < Nr; j += 4, ++idx)
            {
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < Mr; ++i, c += N)
    {
        for (int j = 0; j < Nr; j += 4, ++idx)
        {
            __m256d cr     = _mm256_loadu_pd(&c[j]);
            __m256d result = _mm256_add_pd(cr, res[idx]);
            _mm256_store_pd(&c[j], result);
        }
    }
}

template<int Nr, int Mr, int Kc, int Nc>
inline void avx_regs_unroll(double* __restrict c,
                            const double* __restrict ma,
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

    const double* a = ma;
    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        a = ma;

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
inline void avx_regs_v2_bpack(double* __restrict c,
                              const double* __restrict ma,
                              const double* __restrict mb,
                              int N,
                              int K)
{
    constexpr int CREG_CNT{Mr * Nr / 4};

    const double* b = mb;

    std::array<__m256d, CREG_CNT> res;
    for (int idx = 0; idx < CREG_CNT; ++idx)
    {
        res[idx] = _mm256_setzero_pd();
    }

    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

    for (int k = 0; k < Kc; ++k, b += Nc)
    {
        const double* a = ma;

        int idx = 0;
        for (int i = 0; i < Mr; ++i, a += K)
        {
            __m256d areg = _mm256_broadcast_sd(&a[k]);
            for (int j = 0; j < Nr; j += 4, ++idx)
            {
                res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < Mr; ++i, c += N)
    {
        for (int j = 0; j < Nr; j += 4, ++idx)
        {
            __m256d cr     = _mm256_loadu_pd(&c[j]);
            __m256d result = _mm256_add_pd(cr, res[idx]);
            _mm256_store_pd(&c[j], result);
        }
    }
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
} // namespace ikernels
