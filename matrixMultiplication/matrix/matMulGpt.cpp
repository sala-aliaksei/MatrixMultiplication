
#include "matMulGpt.hpp"

/*
 * Highly optimized DGEMM for 2880x2880 matrices on Intel Haswell.
 *
 * Hardware assumptions:
 *   - Haswell microarchitecture (4 cores)
 *   - L1: 128 KB, L2: 1 MB, L3: 6 MB
 *   - FMA latency ≈5 cycles, throughput ≈0.5 per cycle.
 *
 * Blocking parameters were chosen by experiment (and analytical modeling)
 * so that working sets of the packed blocks fit in the respective caches.
 *
 * Note: This “bare–metal” implementation uses OpenMP for parallelization and
 * AVX2 intrinsics (with FMA) for the inner micro–kernel.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
// Blocking parameters. (2880 is exactly divisible by these.)
#define MC 96
#define NC 96
#define KC 96

// Micro–kernel dimensions: 4×12 sub–block.
#define MR 4
#define NR 12

/*
 * microkernel_4x12_unrolled:
 *
 * Computes a 4×12 update:
 *
 *   C[0:MR-1,0:NR-1] += A_pack[0:MR-1,0:kc-1] * B_pack[0:kc-1,0:NR-1]
 *
 * where:
 *  - A_pack is assumed to be stored row–major with row–length = kc.
 *  - B_pack is stored row–major with row–length = ldb (the width of the packed B block).
 *  - C is stored in row–major order with leading dimension ldc.
 *
 * This micro–kernel unrolls the k–loop by two. (Since our chosen KC=96 is a multiple of 2,
 * this works out nicely.) The ordering of loads, broadcasts, and FMAs has been tuned to
 * resemble the following schedule:
 *
 *   vmovupd   (B),   vbroadcastsd (A row0) → FMAs into row0 accumulators
 *   vmovupd   (B+32), vbroadcastsd (A row1) → FMAs into row1 accumulators
 *   … then advance B and A pointers, and repeat.
 *
 * This schedule is similar to the one in the provided assembly sample.
 */
static void microkernel_4x12_unrolled(int           kc,
                                      const double* A_pack,
                                      const double* B_pack,
                                      double*       C,
                                      int           ldc,
                                      int           ldb)
{
    // Load the 4x12 block of C into registers.
    __m256d c0_0 = _mm256_loadu_pd(C + 0 * ldc);
    __m256d c0_1 = _mm256_loadu_pd(C + 0 * ldc + 4);
    __m256d c0_2 = _mm256_loadu_pd(C + 0 * ldc + 8);

    __m256d c1_0 = _mm256_loadu_pd(C + 1 * ldc);
    __m256d c1_1 = _mm256_loadu_pd(C + 1 * ldc + 4);
    __m256d c1_2 = _mm256_loadu_pd(C + 1 * ldc + 8);

    __m256d c2_0 = _mm256_loadu_pd(C + 2 * ldc);
    __m256d c2_1 = _mm256_loadu_pd(C + 2 * ldc + 4);
    __m256d c2_2 = _mm256_loadu_pd(C + 2 * ldc + 8);

    __m256d c3_0 = _mm256_loadu_pd(C + 3 * ldc);
    __m256d c3_1 = _mm256_loadu_pd(C + 3 * ldc + 4);
    __m256d c3_2 = _mm256_loadu_pd(C + 3 * ldc + 8);

    int p;
    // Unroll by 2.
    for (p = 0; p <= kc - 2; p += 2)
    {
        // --- Unrolled iteration for p
        // Load B vectors for iteration p.
        __m256d b0_p = _mm256_loadu_pd(B_pack + (p + 0) * ldb);
        __m256d b1_p = _mm256_loadu_pd(B_pack + (p + 0) * ldb + 4);
        __m256d b2_p = _mm256_loadu_pd(B_pack + (p + 0) * ldb + 8);
        // Row 0: broadcast A[ p+0 + 0*kc ] and update.
        __m256d a0_p = _mm256_broadcast_sd(A_pack + (p + 0) + 0 * kc);
        c0_0         = _mm256_fmadd_pd(a0_p, b0_p, c0_0);
        c0_1         = _mm256_fmadd_pd(a0_p, b1_p, c0_1);
        c0_2         = _mm256_fmadd_pd(a0_p, b2_p, c0_2);
        // Row 1.
        __m256d a1_p = _mm256_broadcast_sd(A_pack + (p + 0) + 1 * kc);
        c1_0         = _mm256_fmadd_pd(a1_p, b0_p, c1_0);
        c1_1         = _mm256_fmadd_pd(a1_p, b1_p, c1_1);
        c1_2         = _mm256_fmadd_pd(a1_p, b2_p, c1_2);
        // Row 2.
        __m256d a2_p = _mm256_broadcast_sd(A_pack + (p + 0) + 2 * kc);
        c2_0         = _mm256_fmadd_pd(a2_p, b0_p, c2_0);
        c2_1         = _mm256_fmadd_pd(a2_p, b1_p, c2_1);
        c2_2         = _mm256_fmadd_pd(a2_p, b2_p, c2_2);
        // Row 3.
        __m256d a3_p = _mm256_broadcast_sd(A_pack + (p + 0) + 3 * kc);
        c3_0         = _mm256_fmadd_pd(a3_p, b0_p, c3_0);
        c3_1         = _mm256_fmadd_pd(a3_p, b1_p, c3_1);
        c3_2         = _mm256_fmadd_pd(a3_p, b2_p, c3_2);

        // --- Unrolled iteration for p+1
        __m256d b0_p1 = _mm256_loadu_pd(B_pack + (p + 1) * ldb);
        __m256d b1_p1 = _mm256_loadu_pd(B_pack + (p + 1) * ldb + 4);
        __m256d b2_p1 = _mm256_loadu_pd(B_pack + (p + 1) * ldb + 8);
        __m256d a0_p1 = _mm256_broadcast_sd(A_pack + (p + 1) + 0 * kc);
        c0_0          = _mm256_fmadd_pd(a0_p1, b0_p1, c0_0);
        c0_1          = _mm256_fmadd_pd(a0_p1, b1_p1, c0_1);
        c0_2          = _mm256_fmadd_pd(a0_p1, b2_p1, c0_2);
        __m256d a1_p1 = _mm256_broadcast_sd(A_pack + (p + 1) + 1 * kc);
        c1_0          = _mm256_fmadd_pd(a1_p1, b0_p1, c1_0);
        c1_1          = _mm256_fmadd_pd(a1_p1, b1_p1, c1_1);
        c1_2          = _mm256_fmadd_pd(a1_p1, b2_p1, c1_2);
        __m256d a2_p1 = _mm256_broadcast_sd(A_pack + (p + 1) + 2 * kc);
        c2_0          = _mm256_fmadd_pd(a2_p1, b0_p1, c2_0);
        c2_1          = _mm256_fmadd_pd(a2_p1, b1_p1, c2_1);
        c2_2          = _mm256_fmadd_pd(a2_p1, b2_p1, c2_2);
        __m256d a3_p1 = _mm256_broadcast_sd(A_pack + (p + 1) + 3 * kc);
        c3_0          = _mm256_fmadd_pd(a3_p1, b0_p1, c3_0);
        c3_1          = _mm256_fmadd_pd(a3_p1, b1_p1, c3_1);
        c3_2          = _mm256_fmadd_pd(a3_p1, b2_p1, c3_2);
    }
    // Process any leftover iteration (if kc is odd).
    for (; p < kc; p++)
    {
        __m256d b0 = _mm256_loadu_pd(B_pack + p * ldb);
        __m256d b1 = _mm256_loadu_pd(B_pack + p * ldb + 4);
        __m256d b2 = _mm256_loadu_pd(B_pack + p * ldb + 8);
        __m256d a0 = _mm256_broadcast_sd(A_pack + p + 0 * kc);
        c0_0       = _mm256_fmadd_pd(a0, b0, c0_0);
        c0_1       = _mm256_fmadd_pd(a0, b1, c0_1);
        c0_2       = _mm256_fmadd_pd(a0, b2, c0_2);
        __m256d a1 = _mm256_broadcast_sd(A_pack + p + 1 * kc);
        c1_0       = _mm256_fmadd_pd(a1, b0, c1_0);
        c1_1       = _mm256_fmadd_pd(a1, b1, c1_1);
        c1_2       = _mm256_fmadd_pd(a1, b2, c1_2);
        __m256d a2 = _mm256_broadcast_sd(A_pack + p + 2 * kc);
        c2_0       = _mm256_fmadd_pd(a2, b0, c2_0);
        c2_1       = _mm256_fmadd_pd(a2, b1, c2_1);
        c2_2       = _mm256_fmadd_pd(a2, b2, c2_2);
        __m256d a3 = _mm256_broadcast_sd(A_pack + p + 3 * kc);
        c3_0       = _mm256_fmadd_pd(a3, b0, c3_0);
        c3_1       = _mm256_fmadd_pd(a3, b1, c3_1);
        c3_2       = _mm256_fmadd_pd(a3, b2, c3_2);
    }

    // Write the results back to C.
    _mm256_storeu_pd(C + 0 * ldc, c0_0);
    _mm256_storeu_pd(C + 0 * ldc + 4, c0_1);
    _mm256_storeu_pd(C + 0 * ldc + 8, c0_2);

    _mm256_storeu_pd(C + 1 * ldc, c1_0);
    _mm256_storeu_pd(C + 1 * ldc + 4, c1_1);
    _mm256_storeu_pd(C + 1 * ldc + 8, c1_2);

    _mm256_storeu_pd(C + 2 * ldc, c2_0);
    _mm256_storeu_pd(C + 2 * ldc + 4, c2_1);
    _mm256_storeu_pd(C + 2 * ldc + 8, c2_2);

    _mm256_storeu_pd(C + 3 * ldc, c3_0);
    _mm256_storeu_pd(C + 3 * ldc + 4, c3_1);
    _mm256_storeu_pd(C + 3 * ldc + 8, c3_2);
}

/*
 * dgemm_optimized:
 *
 * Computes C = A*B + C for m×k, k×n, and m×n matrices using three–level blocking,
 * packing, and the above AVX2/FMA micro–kernel.
 *
 * The blocking strategy is:
 *   - Outer loops over blocks of rows (i0) and columns (j0) break C into MC×NC blocks.
 *   - An inner loop over the k–dimension uses block size KC.
 *   - Blocks A (of size MC×KC) and B (of size KC×NC) are packed into contiguous buffers.
 *   - The micro–kernel computes MR×NR sub–blocks.
 *
 * OpenMP parallelizes the outer (row) loop.
 */
void dgemm_optimized(const int     m,
                     const int     n,
                     const int     k,
                     const double* A,
                     const double* B,
                     double*       C)
{
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < m; i0 += MC)
    {
        int     mc     = min(MC, m - i0);
        double* A_pack = (double*)aligned_alloc(64, MC * KC * sizeof(double));
        double* B_pack = (double*)aligned_alloc(64, KC * NC * sizeof(double));
        for (int p0 = 0; p0 < k; p0 += KC)
        {
            int kc = min(KC, k - p0);
            // Pack A: copy the mc×kc block from A.
            for (int i = 0; i < mc; i++)
            {
                for (int p = 0; p < kc; p++)
                {
                    A_pack[i * kc + p] = A[(i0 + i) * k + (p0 + p)];
                }
            }
            for (int j0 = 0; j0 < n; j0 += NC)
            {
                int nc = min(NC, n - j0);
                // Pack B: copy the kc×nc block from B.
                for (int p = 0; p < kc; p++)
                {
                    for (int j = 0; j < nc; j++)
                    {
                        B_pack[p * nc + j] = B[(p0 + p) * n + (j0 + j)];
                    }
                }
                // Compute the block C[i0:i0+mc, j0:j0+nc] in MR×NR sub–blocks.
                for (int i = 0; i < mc; i += MR)
                {
                    for (int j = 0; j < nc; j += NR)
                    {
                        // Call our optimized micro–kernel.
                        microkernel_4x12_unrolled(kc,
                                                  &A_pack[i * kc],
                                                  &B_pack[j],
                                                  &C[(i0 + i) * n + (j0 + j)],
                                                  n, // C's leading dimension.
                                                  nc // Packed B block's row stride.
                        );
                    }
                }
            }
        }
        free(A_pack);
        free(B_pack);
    }
}

void gpt_matrix_multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    dgemm_optimized(A.row(), B.col(), A.col(), A.data(), B.data(), C.data());
}

/*----------------------------------------------------------------------------
   (Optional) Test main() function.
   Multiply two 2880×2880 matrices and (optionally) verify the result.
----------------------------------------------------------------------------*/
#ifdef TEST_DGEMM
#include <time.h>
int main(void)
{
    const int N    = 2880;
    size_t    size = N * N * sizeof(double);
    double*   A    = (double*)aligned_alloc(64, size);
    double*   B    = (double*)aligned_alloc(64, size);
    double*   C    = (double*)aligned_alloc(64, size);

    // Initialize matrices A and B; zero initialize C.
    for (int i = 0; i < N * N; i++)
    {
        A[i] = drand48();
        B[i] = drand48();
        C[i] = 0.0;
    }

    double t0 = omp_get_wtime();
    optimized_dgemm(N, N, N, A, B, C);
    double t1 = omp_get_wtime();
    printf("Optimized DGEMM: %.3f seconds\n", t1 - t0);

    // (Optional) Verification code would go here.

    free(A);
    free(B);
    free(C);
    return 0;
}
#endif
