#include "mm/genai/matMulClaude.hpp"

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
#include <cstring>

// Haswell-specific optimizations for double precision:
// - Each AVX register can hold 4 doubles (vs 8 floats)
// - Need to adjust block sizes since doubles take twice the space

// Optimize for L1 cache: 32KB = 32768 bytes
// Each double is 8 bytes, so we want (BLOCK_SIZE * BLOCK_SIZE * 3 * 8) < 32768
// This gives us approximately 36x36, but we'll use 32 for better alignment
#define BLOCK_SIZE 32

// Kernel size for the inner multiplication
// Chosen to maximize register usage without spilling
#define KERNEL_SIZE 8

// Align matrices for AVX instructions (32-byte boundary)
#define ALIGNMENT 32

// Micro-kernel for 8x4 multiplication using AVX2 and FMA for doubles
inline void kernel_8x4(const double* A,
                       const double* B,
                       double*       C,
                       size_t        lda,
                       size_t        ldb,
                       size_t        ldc,
                       size_t        k)
{
    __m256d c[8];

    // Load C values
    for (int i = 0; i < 8; i++)
    {
        c[i] = _mm256_load_pd(&C[i * ldc]);
    }

    // Main computation loop
    for (size_t p = 0; p < k; p++)
    {
        __m256d b = _mm256_load_pd(&B[p * ldb]);

        for (int i = 0; i < 8; i++)
        {
            __m256d a = _mm256_broadcast_sd(&A[i * lda + p]);
            c[i]      = _mm256_fmadd_pd(a, b, c[i]);
        }
    }

    // Store results back
    for (int i = 0; i < 8; i++)
    {
        _mm256_store_pd(&C[i * ldc], c[i]);
    }
}

void multiply_matrices_optimized(const Matrix<double>& A,
                                 const Matrix<double>& B,
                                 Matrix<double>&       C)
{
    const size_t M = A.row();
    const size_t N = B.col();
    const size_t K = A.col();

// Three-level blocking for L1, L2, and L3 cache
#pragma omp parallel
    {
        // Block sizes adjusted for double precision
        const size_t L3_BLOCK = 256;        // For shared L3
        const size_t L2_BLOCK = 64;         // For per-core L2
        const size_t L1_BLOCK = BLOCK_SIZE; // For L1D cache

#pragma omp for schedule(dynamic, 1)
        for (size_t i3 = 0; i3 < M; i3 += L3_BLOCK)
        {
            for (size_t j3 = 0; j3 < N; j3 += L3_BLOCK)
            {
                for (size_t k3 = 0; k3 < K; k3 += L3_BLOCK)
                {
                    const size_t i3_end = std::min(i3 + L3_BLOCK, M);
                    const size_t j3_end = std::min(j3 + L3_BLOCK, N);
                    const size_t k3_end = std::min(k3 + L3_BLOCK, K);

                    for (size_t i2 = i3; i2 < i3_end; i2 += L2_BLOCK)
                    {
                        for (size_t j2 = j3; j2 < j3_end; j2 += L2_BLOCK)
                        {
                            for (size_t k2 = k3; k2 < k3_end; k2 += L2_BLOCK)
                            {
                                const size_t i2_end = std::min(i2 + L2_BLOCK, i3_end);
                                const size_t j2_end = std::min(j2 + L2_BLOCK, j3_end);
                                const size_t k2_end = std::min(k2 + L2_BLOCK, k3_end);

                                for (size_t i = i2; i < i2_end; i += KERNEL_SIZE)
                                {
                                    for (size_t j = j2; j < j2_end; j += 4)
                                    {
                                        // Prefetch next tiles
                                        _mm_prefetch((const char*)&A(i + KERNEL_SIZE, k2),
                                                     _MM_HINT_T0);
                                        _mm_prefetch((const char*)&B(k2, j + 4), _MM_HINT_T0);

                                        const size_t i_end  = std::min(i + KERNEL_SIZE, i2_end);
                                        const size_t k_size = k2_end - k2;

                                        if (i + KERNEL_SIZE <= i2_end)
                                        {
                                            // Full kernel
                                            kernel_8x4(
                                              &A(i, k2), &B(k2, j), &C(i, j), K, N, N, k_size);
                                        }
                                        else
                                        {
                                            // Cleanup code for remaining rows
                                            for (size_t ii = i; ii < i_end; ii++)
                                            {
                                                __m256d c = _mm256_load_pd(&C(ii, j));
                                                for (size_t k = k2; k < k2_end; k++)
                                                {
                                                    __m256d a = _mm256_broadcast_sd(&A(ii, k));
                                                    __m256d b = _mm256_load_pd(&B(k, j));
                                                    c         = _mm256_fmadd_pd(a, b, c);
                                                }
                                                _mm256_store_pd(&C(ii, j), c);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
