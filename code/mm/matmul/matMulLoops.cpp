#include "mm/core/ikernels.hpp"
#include "mm/core/reorderMatrix.hpp"
#include "mm/matmul/matMulLoops.hpp"

#include "omp.h"

#include <thread>
#include <immintrin.h>

namespace
{

//////////////////////     KERNELS

template<int Nr, int Mr, int Kc>
static void upkernel_loops_v2(const double* __restrict ma,
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

    //    _mm_prefetch(a + 8, _MM_HINT_T0);
    //    _mm_prefetch(b + 8, _MM_HINT_T0);
    //    _mm_prefetch(b + 16, _MM_HINT_T0);
    //    _mm_prefetch(b + 24, _MM_HINT_T0);
    //    _mm_prefetch(b + 64, _MM_HINT_T0);
    //    _mm_prefetch(b + 64 * 2, _MM_HINT_T0);
    //    _mm_prefetch(b + 64 * 3, _MM_HINT_T0);

    {
        _mm_prefetch(b + 8, _MM_HINT_T0);
        __m256d b0 = _mm256_loadu_pd(&b[0]);
        __m256d b1 = _mm256_loadu_pd(&b[4]);
        __m256d b2 = _mm256_loadu_pd(&b[8]);

        __m256d a0 = _mm256_broadcast_sd(&a[0]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        _mm_prefetch(b + 8 * 2, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[1]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        _mm_prefetch(b + 8 * 3, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[2]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        _mm_prefetch(b + 8 * 4, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[3]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);

        // iter with prefetech

        b0 = _mm256_loadu_pd(&b[12]);
        b1 = _mm256_loadu_pd(&b[16]);
        b2 = _mm256_loadu_pd(&b[20]);

        _mm_prefetch(b + 8 * 5, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[4]);

        r00 = _mm256_fmadd_pd(a0, b0, r00);
        r01 = _mm256_fmadd_pd(a0, b1, r01);
        r02 = _mm256_fmadd_pd(a0, b2, r02);

        _mm_prefetch(b + 8 * 6, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[5]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        a0 = _mm256_broadcast_sd(&a[6]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        _mm_prefetch(b + 8 * 7, _MM_HINT_T0);
        a0 = _mm256_broadcast_sd(&a[7]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);

        b += 2 * Nr;
        a += 2 * Mr;
    }

    for (int k = 2; k < Kc - 2; k += 2, b += 2 * Nr, a += 2 * Mr)
    {
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

        a0 = _mm256_broadcast_sd(&a[7]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);
    }

    {
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

        ikernels::load_inc_store_double(&c[0], r00);
        ikernels::load_inc_store_double(&c[4], r01);
        ikernels::load_inc_store_double(&c[8], r02);

        c += N;
        _mm_prefetch(c, _MM_HINT_NTA);

        a0 = _mm256_broadcast_sd(&a[5]);

        r10 = _mm256_fmadd_pd(a0, b0, r10);
        r11 = _mm256_fmadd_pd(a0, b1, r11);
        r12 = _mm256_fmadd_pd(a0, b2, r12);

        ikernels::load_inc_store_double(&c[0], r10);
        ikernels::load_inc_store_double(&c[4], r11);
        ikernels::load_inc_store_double(&c[8], r12);
        c += N;
        _mm_prefetch(c, _MM_HINT_NTA);

        a0 = _mm256_broadcast_sd(&a[6]);

        r20 = _mm256_fmadd_pd(a0, b0, r20);
        r21 = _mm256_fmadd_pd(a0, b1, r21);
        r22 = _mm256_fmadd_pd(a0, b2, r22);

        ikernels::load_inc_store_double(&c[0], r20);
        ikernels::load_inc_store_double(&c[4], r21);
        ikernels::load_inc_store_double(&c[8], r22);
        c += N;
        _mm_prefetch(c, _MM_HINT_NTA);

        a0 = _mm256_broadcast_sd(&a[7]);

        r30 = _mm256_fmadd_pd(a0, b0, r30);
        r31 = _mm256_fmadd_pd(a0, b1, r31);
        r32 = _mm256_fmadd_pd(a0, b2, r32);

        ikernels::load_inc_store_double(&c[0], r30);
        ikernels::load_inc_store_double(&c[4], r31);
        ikernels::load_inc_store_double(&c[8], r32);
    }

    //        _mm_prefetch(c + N, _MM_HINT_NTA);

    //    _mm_prefetch(c + N, _MM_HINT_NTA);

    //    _mm_prefetch(c + N, _MM_HINT_NTA);
}

template<int Nr, int Mr, int Kc>
static void upkernel_loops(const double* __restrict ma,
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

    // _mm_prefetch(b + 8, _MM_HINT_NTA);
    // _mm_prefetch(b + 16, _MM_HINT_NTA);
    // _mm_prefetch(b + 24, _MM_HINT_NTA);
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

    ikernels::load_inc_store_double(&c[0], r00);
    ikernels::load_inc_store_double(&c[4], r01);
    ikernels::load_inc_store_double(&c[8], r02);

    c += N;

    //    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r10);
    ikernels::load_inc_store_double(&c[4], r11);
    ikernels::load_inc_store_double(&c[8], r12);
    c += N;

    //    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r20);
    ikernels::load_inc_store_double(&c[4], r21);
    ikernels::load_inc_store_double(&c[8], r22);
    c += N;

    ikernels::load_inc_store_double(&c[0], r30);
    ikernels::load_inc_store_double(&c[4], r31);
    ikernels::load_inc_store_double(&c[8], r32);
}

template<int Nr, int Mr, int Kc>
inline void pack_ukernel_arr(const double* __restrict a,
                             const double* __restrict b,
                             double* __restrict c,
                             int N)
{
    constexpr int CREG_CNT{Mr * Nr / 4};

    std::array<__m256d, CREG_CNT> res;
    for (int idx = 0; idx < CREG_CNT; ++idx)
    {
        res[idx] = _mm256_setzero_pd();
    }

    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        int idx = 0;
        for (int i = 0; i < Mr; ++i)
        {
            __m256d areg = _mm256_broadcast_sd(&a[i]);
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

//////////////////

template<int Nr, int Mr, int Kc>
static void ukernel(const double* const __restrict ma,
                    const double* __restrict b,
                    double* __restrict c,
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

    // no repacking
    const auto a_cols = K;

    // no repacking
    const auto b_cols = N;

    const double* a = ma;
    for (int k2 = 0; k2 < Kc; ++k2, b += b_cols)
    {
        a = ma;

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

    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r00);
    ikernels::load_inc_store_double(&c[4], r01);
    ikernels::load_inc_store_double(&c[8], r02);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r10);
    ikernels::load_inc_store_double(&c[4], r11);
    ikernels::load_inc_store_double(&c[8], r12);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r20);
    ikernels::load_inc_store_double(&c[4], r21);
    ikernels::load_inc_store_double(&c[8], r22);
    c += N;

    ikernels::load_inc_store_double(&c[0], r30);
    ikernels::load_inc_store_double(&c[4], r31);
    ikernels::load_inc_store_double(&c[8], r32);
}

template<int Nr, int Mr, int Kc, int Nc>
static void ukernelBpacked(const double* const __restrict ma,
                           const double* __restrict b,
                           double* __restrict c,
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

    // no repacking
    const auto a_cols = K;

    // no repacking
    constexpr auto b_cols = Nc;

    const double* a = ma;
    for (int k2 = 0; k2 < Kc; ++k2, b += b_cols)
    {
        a = ma;

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

    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r00);
    ikernels::load_inc_store_double(&c[4], r01);
    ikernels::load_inc_store_double(&c[8], r02);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r10);
    ikernels::load_inc_store_double(&c[4], r11);
    ikernels::load_inc_store_double(&c[8], r12);
    c += N;

    _mm_prefetch(c + N, _MM_HINT_NTA);

    ikernels::load_inc_store_double(&c[0], r20);
    ikernels::load_inc_store_double(&c[4], r21);
    ikernels::load_inc_store_double(&c[8], r22);
    c += N;

    ikernels::load_inc_store_double(&c[0], r30);
    ikernels::load_inc_store_double(&c[4], r31);
    ikernels::load_inc_store_double(&c[8], r32);
}
} // namespace
//////////////////////////      MATMUL     ///////////////////////////////////

void matMulLoops(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    constexpr int Mc = 180;
    constexpr int Kc = 240;
    constexpr int Nc = 720;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

#pragma omp parallel for
    for (int j = 0; j < N; j += Nc)
    {
        double*       Cc4 = &C(0, j);
        const double* Bc4 = &B(0, j);

        for (int k = 0; k < K; k += Kc)
        {
            const double* Ac3 = &A(0, k);
            const double* Bc3 = Bc4 + N * k;

            for (int i = 0; i < M; i += Mc)
            {
                double*       Cc2 = Cc4 + N * i;
                const double* Ac2 = Ac3 + K * i;

                for (int jb = 0; jb < Nc; jb += Nr)
                {
                    double*       Cc1 = Cc2 + jb;
                    const double* Bc1 = Bc3 + jb;
                    for (int ib = 0; ib < Mc; ib += Mr)
                    {
                        double*       Cc0 = Cc1 + N * ib;
                        const double* Ac0 = Ac2 + K * ib;

                        ukernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N, K);
                    }
                }
            }
        }
    }

    // DONE
}

void matMulLoopsBPacked(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    constexpr int Mc = 180;
    constexpr int Kc = 48;
    constexpr int Nc = 96;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    static_assert(Nc % Nr == 0, "Invalid N divion");
    static_assert(Mc % Mr == 0, "Invalid N divion");

    const auto    N = B.col();
    const auto    K = A.col();
    const auto    M = A.row();
    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

#pragma omp parallel for
    for (int i = 0; i < M; i += Mc)
    {
        for (int k = 0; k < K; k += Kc)
        {
            for (int j = 0; j < N; j += Nc)
            {
                auto          arr = packMatrix<Kc, Nc>(&B(k, 0) + j, N);
                const double* Bc2 = arr.data();

                for (int ib = 0; ib < Mc; ib += Mr)
                {
                    for (int jb = 0; jb < Nc; jb += Nr)
                    {
                        ukernelBpacked<Nr, Mr, Kc, Nc>(
                          &A(i, 0) + k + ib * K, Bc2 + jb, &C(i, 0) + j + ib * N + jb, N, K);
                    }
                }
            }
        }
    }

    // DONE
}

void matMulLoopsIKJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    /*BEST
    constexpr int Mc = 180;
    constexpr int Kc = 48;
    constexpr int Nc = 96;
    */

    constexpr int Mc = 180;
    constexpr int Kc = 48;
    constexpr int Nc = 96;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    static_assert(Nc % Nr == 0, "Invalid N divion");
    static_assert(Mc % Mr == 0, "Invalid N divion");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

#pragma omp parallel for
    for (int i = 0; i < M; i += Mc)
    {
        double*       Cc4 = &C(i, 0);
        const double* Ac4 = &A(i, 0);

        for (int k = 0; k < K; k += Kc)
        {
            const double* Ac3 = Ac4 + k;
            const double* Bc3 = &B(k, 0);

            for (int j = 0; j < N; j += Nc)
            {
                double*       Cc2 = Cc4 + j;
                const double* Bc2 = Bc3 + j;

                for (int ib = 0; ib < Mc; ib += Mr)
                {
                    double*       Cc1 = Cc2 + ib * N;
                    const double* Ac1 = Ac3 + ib * K;
                    for (int jb = 0; jb < Nc; jb += Nr)
                    {
                        double*       Cc0 = Cc1 + jb;
                        const double* Bc0 = Bc2 + jb;

                        ukernel<Nr, Mr, Kc>(Ac1, Bc0, Cc0, N, K);
                    }
                }
            }
        }
    }

    // DONE
}

void matMulLoopsRepack(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // BEST

    constexpr int Mc = 24;
    constexpr int Kc = 96;
    constexpr int Nc = 720;

    constexpr int Nr = 12;
    constexpr int Mr = 4;
    constexpr int Kr = 1; // consider to increase to improve repack perf

    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    std::vector<double> buffer(4 * Kc * (Mc + Nc));

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

                        // upkernel_loops<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        upkernel_loops_v2<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // pack_ukernel_arr<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }
        }
    }

    // DONE
}

void matMulLoopsRepackIKJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    constexpr int Mc = 180; // 720 / 4;
    constexpr int Kc = 96;
    constexpr int Nc = 24;

    constexpr int Nr = 12;
    constexpr int Mr = 4;
    constexpr int Kr = 1; // consider to increase to improve repack perf

    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    auto num_threads = std::thread::hardware_concurrency(); // std::thread::hardware_concurrency();
    num_threads      = num_threads > 4 ? 16 : num_threads;

    std::vector<double> buffer(num_threads * Kc * (Mc + Nc));

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i += Mc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        double*       Cc4 = &C(i, 0);
        const double* Ac4 = &A(i, 0);

        for (int k = 0; k < K; k += Kc)
        {
            const double* Ac3 = Ac4 + k;

            reorderColOrderMatrix<Mc, Kc, Mr, Kr>(Ac3, K, buf);
            const double* Ac2 = buf;

            for (int j = 0; j < N; j += Nc)
            {
                double*       Cc3 = Cc4 + j;
                const double* Bc4 = &B(k, j);

                reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(Bc4, N, buf + Mc * Kc);
                const double* Bc3 = (buf + Mc * Kc);

                for (int jb = 0; jb < Nc; jb += Nr)
                {
                    double*       Cc1 = Cc3 + jb;
                    const double* Bc1 = Bc3 + Kc * jb;

                    for (int ib = 0; ib < Mc; ib += Mr)
                    {
                        double*       Cc0 = Cc1 + N * ib;
                        const double* Ac0 = Ac2 + Kc * ib;

                        upkernel_loops<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        //  upkernel_loops_v2<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // pack_ukernel_arr<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }
        }
    }

    // DONE
}
