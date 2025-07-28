#include "mm/core/ikernels.hpp"
#include "mm/core/kernels.hpp"
#include "mm/core/reorderMatrix.hpp"
#include "mm/matmul/matMul.hpp"

#include <cstring> // memcpy
#include <cmath>
#include <array>

namespace mm
{

///////////////////     MATMUL IMPLEMENTATION   ///////////////////////
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

void matMul_Naive_Order(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            for (int j = 0; j < N; ++j)
            {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

void matMul_Naive_Order_KIJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* a = A.data();
    const double* b = B.data();
    double*       c = C.data();

    for (int k = 0; k < K; ++k, b += N)
    {
        a = A.data();
        c = C.data();
        for (int i = 0; i < M; ++i, c += N, a += K)
        {
            for (int j = 0; j < N; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

void matMul_Naive_Tile(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto N = B.col();
    auto K = A.col();

    constexpr auto BLOCK = 64;
    if ((M % BLOCK != 0) || (K % BLOCK != 0) || (N % BLOCK != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += BLOCK)
    {
        for (int kb = 0; kb < K; kb += BLOCK)
        {
            for (int jb = 0; jb < N; jb += BLOCK)
            {
                const double* a  = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       c  = &C(ib, jb);

                //                ikernels::naive_block<BLOCK>(a, mb, c, N, K);
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
        }
    }
}

void matMul_Simd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    constexpr auto BLOCK = 64;
    if ((M % BLOCK != 0) || (K % BLOCK != 0) || (N % BLOCK != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += BLOCK)
    {
        for (int kb = 0; kb < K; kb += BLOCK)
        {
            for (int jb = 0; jb < N; jb += BLOCK)
            {
                const double* a  = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       c  = &C(ib, jb);

                ikernels::simd_block<BLOCK>(a, mb, c, N, K);
            }
        }
    }
}

void matMul_Avx(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    constexpr auto BLOCK = 64;
    if ((M % BLOCK != 0) || (K % BLOCK != 0) || (N % BLOCK != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += BLOCK)
    {
        for (int kb = 0; kb < K; kb += BLOCK)
        {
            for (int jb = 0; jb < N; jb += BLOCK)
            {
                const double* a = &A(ib, kb);
                const double* b = &B(kb, jb);
                double*       c = &C(ib, jb);

                ikernels::avx_block<BLOCK>(a, b, c, N, K);
            }
        }
    }
}

void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_naive<Nr, Mr, Kc, Nc>(c, a, b, N, K);
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

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_regs<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

void matMul_Avx_Cache_Regs_Unroll(const Matrix<double>& A,
                                  const Matrix<double>& B,
                                  Matrix<double>&       C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_regs_unroll<Nr, Mr, Kc, Nc>(c, a, b, N, K);
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

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_regs_unroll_rw<Nr, Mr, Kc, Nc>(c, a, b, N, K);
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

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                double*       mc = &C(ib, jb);

                auto          arr = packMatrix<Kc, Nc>(&B(kb, jb), N);
                const double* mb  = arr.data();

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_regs_unroll_bpack<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                        // avx_regs_v2_bpack<Nr, Mr, Kc, Nc>(c, a, b, N, K);
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

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

#pragma omp parallel for
    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_regs_unroll_kr<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                        // avx_regs_v2_bpack<Nr, Mr, Kc, Nc>(c, a, b, N, K);
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

    constexpr int Mc = 180;
    constexpr int Nc = 96;
    constexpr int Kc = 48;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    if ((M % Mc != 0) || (K % Kc != 0) || (N % Nc != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

#pragma omp parallel for
    for (int ib = 0; ib < M; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            for (int jb = 0; jb < N; jb += Nc)
            {
                const double* ma = &A(ib, kb);
                double*       mc = &C(ib, jb);

                auto          arr = packMatrix<Kc, Nc>(&B(kb, jb), N);
                const double* mb  = arr.data();

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        const double* a = &ma[i2 * K];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * N + j2];
                        ikernels::avx_regs_unroll_bpack<Nr, Mr, Kc, Nc>(c, a, b, N, K);
                    }
                }
            }
        }
    }
}

////  One more optimizatio order
///
///
void matMul_Avx_AddRegs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    if ((M % 64 != 0) || (K % 64 != 0) || (N % 64 != 0))
    {
        throw std::runtime_error("DIM % BLOCK != 0");
    }

    constexpr auto BLOCK = 64;
    constexpr int  REG_CNT{BLOCK / 4};

    for (int ib = 0; ib < M; ib += BLOCK)
    {
        for (int kb = 0; kb < K; kb += BLOCK)
        {
            for (int jb = 0; jb < N; jb += BLOCK)
            {
                const double* ma = &A(ib, kb);
                const double* mb = &B(kb, jb);
                double*       mc = &C(ib, jb);

                for (int i = 0; i < BLOCK; ++i, mc += N, ma += K)
                {
                    const double* b = mb;

                    std::array<__m256d, REG_CNT> cache;
                    std::array<__m256d, REG_CNT> res;
                    for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                    {
                        res[idx] = _mm256_setzero_pd(); //_mm256_loadu_pd(&c[j2]);
                    }

                    //_mm_prefetch(&a[8], _MM_HINT_NTA); // prefetch next cache line

                    for (int k = 0; k < BLOCK; ++k, b += N)
                    {
                        __m256d areg = _mm256_broadcast_sd(&ma[k]);
                        for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                        {
                            res[idx] = _mm256_fmadd_pd(areg, _mm256_loadu_pd(&b[j]), res[idx]);
                        }
                    }

                    for (int j = 0, idx = 0; j < BLOCK; j += 4, ++idx)
                    {
                        __m256d creg   = _mm256_loadu_pd(&mc[j]);
                        __m256d result = _mm256_add_pd(creg, res[idx]);
                        _mm256_store_pd(&mc[j], result);
                    }
                }
            }
        }
    }
}

void matMul_Avx_AddRegs_Unroll(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // + remove k loop
    auto M = A.row();
    auto K = A.col();
    auto N = B.col();

    const double* ma = A.data();
    const double* mb = B.data();
    double*       mc = C.data();

    constexpr auto Mr = 4;
    constexpr auto Nr = 12;

    if ((M % Mr != 0) || (N % Nr != 0))
    {
        throw std::runtime_error("size % BLOCK != 0");
    }

    auto a_cols = K;
    auto b_cols = N;

    for (int ib = 0; ib < M; ib += Mr)
    {
        for (int jb = 0; jb < N; jb += Nr)
        {
            double*       c  = &mc[ib * N + jb];
            const double* aa = &ma[ib * K];
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
            for (int k2 = 0; k2 < K; ++k2, b += b_cols)
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
    }
}

////////////////////////    Not packed + tail

template<int Mr, int Mc, int Kc, int... TailSize>
static inline void
handleJtail(const double* a2, const double* mb, double* mc, int K, int N, int jl, int j_tail_size)
{
    // TODO: Add multithreading
    (...,
     (
       [&]
       {
           double*       c2 = &mc[jl];
           const double* b2 = &mb[jl];
           while (j_tail_size >= TailSize)
           {
               for (int i2 = 0; i2 < Mc; i2 += Mr)
               {
                   kernels::cpp_generic_ukern<TailSize, Mr, Kc>(&a2[i2 * K], b2, &c2[i2 * N], N, K);
               }
               jl += TailSize;
               j_tail_size -= TailSize;
           }
       }()));
}

template<int Nr, int Mr, int Nc, int Mc, int Kc, int... TailSize>
static inline void handleItail(const double* ma,
                               const double* mb,
                               double*       mc,
                               int           K,
                               int           N,
                               int           ilast,
                               int           i_tail_size)
{
    // TODO: Add multithreading
    (...,
     (
       [&]
       {
           // #pragma omp parallel for
           while (i_tail_size >= TailSize)
           {
               constexpr int Mrr = TailSize;
               for (int kb = 0; kb < K; kb += Kc)
               {
                   const double* a = &ma[ilast * K + kb];

                   int j_tail_size = N % Nc;
                   int jl          = N - j_tail_size;

                   for (int jb = 0; jb < jl; jb += Nc)
                   {
                       double*       c2 = &mc[ilast * N + jb];
                       const double* b2 = &mb[kb * N + jb];

                       for (int j2 = 0; j2 < Nc; j2 += Nr)
                       {
                           kernels::cpp_generic_ukern<Nr, Mrr, Kc>(a, &b2[j2], &c2[j2], N, K);
                       }
                   }

                   handleJtail<Mrr, Mrr, Kc, 12, 8, 4, 2, 1>(
                     a, &mb[kb * N], &mc[ilast * N], K, N, jl, j_tail_size);
               }
               ilast += Mrr;
               i_tail_size -= Mrr;
           }
       }()));
}

void matMul_Tails(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
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

    int i_tail_size = M % Mc;
    int ilast       = M - i_tail_size;

#pragma omp parallel for
    for (int ib = 0; ib < ilast; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            const double* a2 = &ma[ib * K + kb];

            int j_tail_size = N % Nc;
            int jl          = N - j_tail_size;

            for (int jb = 0; jb < jl; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* b2 = &mb[kb * N + jb];

                // What if Mc %Mr != 0?
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    const double* a = &a2[i2 * K];
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        kernels::cpp_generic_ukern<Nr, Mr, Kc>(a, &b2[j2], &c2[i2 * N + j2], N, K);
                    }
                }
            }

            // TODO: Simplifiy the call
            handleJtail<Mr, Mc, Kc, 12, 8, 4, 2, 1>(
              a2, &mb[kb * N], &mc[ib * N], K, N, jl, j_tail_size);
        }
    }

    handleItail<Nr, Mr, Nc, Mc, Kc, 4, 3, 2, 1>(ma, mb, mc, K, N, ilast, i_tail_size);
}

//

void matMul_ManualTail(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
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

    auto i_tail_size = M % Mc;
    auto ilast       = M - i_tail_size;

#pragma omp parallel for
    for (int ib = 0; ib < ilast; ib += Mc)
    {
        for (int kb = 0; kb < K; kb += Kc)
        {
            const double* a2 = &ma[ib * K + kb];

            // tail is only in last block
            int j_tail_size = N % Nc;
            int jl          = N - j_tail_size;

            for (int jb = 0; jb < jl; jb += Nc)
            {
                double*       c2 = &mc[ib * N + jb];
                const double* b2 = &mb[kb * N + jb];
                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    const double* a = &a2[i2 * K];
                    for (int j2 = 0; j2 < Nc; j2 += Nr)
                    {
                        kernels::cpp_generic_ukern<Nr, Mr, Kc>(a, &b2[j2], &c2[i2 * N + j2], N, K);
                    }
                }
            }

            // Handle J tails
            while (j_tail_size >= 12)
            {
                double*       c2 = &mc[ib * N + jl];
                const double* b2 = &mb[kb * N + jl];

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    kernels::cpp_generic_ukern<12, Mr, Kc>(&a2[i2 * K], b2, &c2[i2 * N], N, K);
                }
                jl += 12;
                j_tail_size -= 12;
            }

            while (j_tail_size >= 8)
            {
                double*       c2 = &mc[ib * N + jl];
                const double* b2 = &mb[kb * N + jl];

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    kernels::cpp_generic_ukern<8, Mr, Kc>(&a2[i2 * K], b2, &c2[i2 * N], N, K);
                }
                jl += 8;
                j_tail_size -= 8;
            }

            while (j_tail_size >= 4)
            {
                double*       c2 = &mc[ib * N + jl];
                const double* b2 = &mb[kb * N + jl];

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    kernels::cpp_generic_ukern<4, Mr, Kc>(&a2[i2 * K], b2, &c2[i2 * N], N, K);
                }
                jl += 4;
                j_tail_size -= 4;
            }

            while (j_tail_size >= 2)
            {
                double*       c2 = &mc[ib * N + jl];
                const double* b2 = &mb[kb * N + jl];

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    kernels::cpp_generic_ukern<2, Mr, Kc>(&a2[i2 * K], b2, &c2[i2 * N], N, K);
                }
                jl += 2;
                j_tail_size -= 2;
            }

            while (j_tail_size > 0)
            {
                double*       c2 = &mc[ib * N + jl];
                const double* b2 = &mb[kb * N + jl];

                for (int i2 = 0; i2 < Mc; i2 += Mr)
                {
                    kernels::cpp_generic_ukern<1, Mr, Kc>(&a2[i2 * K], b2, &c2[i2 * N], N, K);
                }
                jl += 1;
                j_tail_size -= 1;
            }
        }
    }

    while (i_tail_size >= 4)
    {
        constexpr int Mrr = 4;
        // #pragma omp parallel for
        for (int kb = 0; kb < K; kb += Kc)
        {
            const double* a = &ma[ilast * K + kb];

            // tail is only in last block
            int j_tail_size = N % Nc;
            int jl          = N - j_tail_size;

            for (int jb = 0; jb < jl; jb += Nc)
            {
                double*       c2 = &mc[ilast * N + jb];
                const double* b2 = &mb[kb * N + jb];

                for (int j2 = 0; j2 < Nc; j2 += Nr)
                {
                    kernels::cpp_generic_ukern<Nr, Mrr, Kc>(a, &b2[j2], &c2[j2], N, K);
                }
            }

            // Handle J tails
            while (j_tail_size >= 12)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<12, Mrr, Kc>(a, b2, c2, N, K);
                jl += 12;
                j_tail_size -= 12;
            }

            while (j_tail_size >= 8)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<8, Mrr, Kc>(a, b2, c2, N, K);
                jl += 8;
                j_tail_size -= 8;
            }

            while (j_tail_size >= 4)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<4, Mrr, Kc>(a, b2, c2, N, K);
                jl += 4;
                j_tail_size -= 4;
            }

            while (j_tail_size >= 2)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<2, Mrr, Kc>(a, b2, c2, N, K);
                jl += 2;
                j_tail_size -= 2;
            }

            while (j_tail_size > 0)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<1, Mrr, Kc>(a, b2, c2, N, K);
                jl += 1;
                j_tail_size -= 1;
            }
        }
        ilast += Mrr;
        i_tail_size -= Mrr;
    }

    while (i_tail_size >= 2)
    {
        constexpr int Mrr = 2;
        // #pragma omp parallel for
        for (int kb = 0; kb < K; kb += Kc)
        {
            const double* a = &ma[ilast * K + kb];

            // tail is only in last block
            auto j_tail_size = N % Nc;
            auto jl          = N - j_tail_size;

            for (int jb = 0; jb < jl; jb += Nc)
            {
                double*       c2 = &mc[ilast * N + jb];
                const double* b2 = &mb[kb * N + jb];

                for (int j2 = 0; j2 < Nc; j2 += Nr)
                {
                    kernels::cpp_generic_ukern<Nr, Mrr, Kc>(a, &b2[j2], &c2[j2], N, K);
                }
            }

            // Handle J tails
            while (j_tail_size >= 12)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<12, Mrr, Kc>(a, b2, c2, N, K);
                jl += 12;
                j_tail_size -= 12;
            }

            while (j_tail_size >= 8)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<8, Mrr, Kc>(a, b2, c2, N, K);
                jl += 8;
                j_tail_size -= 8;
            }

            while (j_tail_size >= 4)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<4, Mrr, Kc>(a, b2, c2, N, K);
                jl += 4;
                j_tail_size -= 4;
            }

            while (j_tail_size >= 2)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<2, Mrr, Kc>(a, b2, c2, N, K);
                jl += 2;
                j_tail_size -= 2;
            }

            while (j_tail_size > 0)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<1, Mrr, Kc>(a, b2, c2, N, K);
                jl += 1;
                j_tail_size -= 1;
            }
        }
        ilast += Mrr;
        i_tail_size -= Mrr;
    }

    while (i_tail_size == 1)
    {
        constexpr int Mrr = 1;
        // #pragma omp parallel for
        for (int kb = 0; kb < K; kb += Kc)
        {
            const double* a = &ma[ilast * K + kb];

            // tail is only in last block
            auto j_tail_size = N % Nc;
            auto jl          = N - j_tail_size;

            for (int jb = 0; jb < jl; jb += Nc)
            {
                double*       c2 = &mc[ilast * N + jb];
                const double* b2 = &mb[kb * N + jb];

                for (int j2 = 0; j2 < Nc; j2 += Nr)
                {
                    kernels::cpp_generic_ukern<Nr, Mrr, Kc>(a, &b2[j2], &c2[j2], N, K);
                }
            }

            // Handle J tails
            while (j_tail_size >= 12)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<12, Mrr, Kc>(a, b2, c2, N, K);
                jl += 12;
                j_tail_size -= 12;
            }

            while (j_tail_size >= 8)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<8, Mrr, Kc>(a, b2, c2, N, K);
                jl += 8;
                j_tail_size -= 8;
            }

            while (j_tail_size >= 4)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<4, Mrr, Kc>(a, b2, c2, N, K);
                jl += 4;
                j_tail_size -= 4;
            }

            while (j_tail_size >= 2)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<2, Mrr, Kc>(a, b2, c2, N, K);
                jl += 2;
                j_tail_size -= 2;
            }

            while (j_tail_size > 0)
            {
                double*       c2 = &mc[ilast * N + jl];
                const double* b2 = &mb[kb * N + jl];

                kernels::cpp_generic_ukern<1, Mrr, Kc>(a, b2, c2, N, K);
                jl += 1;
                j_tail_size -= 1;
            }
        }
        ilast += Mrr;
        i_tail_size -= Mrr;
    }
}

} // namespace mm
