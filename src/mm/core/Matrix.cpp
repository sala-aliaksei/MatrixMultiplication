#include "mm/core/Matrix.hpp"
#include <random>
#include <immintrin.h>
#include <algorithm>

bool operator==(const MatrixSet& s1, const MatrixSet& s2)
{
    return s1.c == s2.c;
}

MatrixSet initPredictedMatrix(int isize, int jsize, int ksize)
{
    std::random_device               rd;
    std::mt19937                     gen(rd());
    double                           lower_bound = 0.0;
    double                           upper_bound = 10.0;
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    MatrixSet set{.a = Matrix<double>(isize, ksize),
                  .b = Matrix<double>(ksize, jsize),
                  .c = Matrix<double>(isize, jsize)};

    int val = 0;
    for (auto i = 0; i < set.a.row(); ++i)
    {
        for (auto j = 0; j < set.a.col(); ++j)
        {
            set.a[i * set.a.col() + j] = ++val;
        }
    }
    val = set.b.row() * set.b.col();
    for (auto i = 0; i < set.b.row(); ++i)
    {
        for (auto j = 0; j < set.b.col(); ++j)
        {
            set.b[i * set.b.col() + j] = --val;
        }
    }

    return set;
}

MatrixSet initDoubleMatrix(int M, int N, int K)
{
    std::random_device rd;
    std::mt19937       gen(rd());
    double             lower_bound = 0.0;
    double             upper_bound = 10.0;

    MatrixSet set{.a = Matrix<double>(M, K), .b = Matrix<double>(K, N), .c = Matrix<double>(M, N)};

    std::uniform_real_distribution<> dis(lower_bound, upper_bound);
    std::generate(set.a.data(), set.a.data() + set.a.size(), [&] { return (int)dis(gen); });
    std::generate(set.b.data(), set.b.data() + set.b.size(), [&] { return (int)dis(gen); });

    return set;
}

namespace _details
{

static inline void transpose_4x4_block_avx2_prefetch(const double* __restrict src,
                                                     double* __restrict dst,
                                                     std::size_t srcStride,
                                                     std::size_t dstStride)
{
    // Load 4 rows of 4 columns
    __m256d row0 = _mm256_loadu_pd(&src[0 * srcStride]);
    __m256d row1 = _mm256_loadu_pd(&src[1 * srcStride]);
    __m256d row2 = _mm256_loadu_pd(&src[2 * srcStride]);
    __m256d row3 = _mm256_loadu_pd(&src[3 * srcStride]);

    // Prefetch the next lines you might read soon (example usage)
    // (Adjust the offset or usage based on your loop structure)
    _mm_prefetch((const char*)(&src[0 * srcStride + 8]), _MM_HINT_T0);
    _mm_prefetch((const char*)(&src[1 * srcStride + 8]), _MM_HINT_T0);
    _mm_prefetch((const char*)(&src[2 * srcStride + 8]), _MM_HINT_T0);
    _mm_prefetch((const char*)(&src[3 * srcStride + 8]), _MM_HINT_T0);

    // Interleave (unpack) row0/row1 and row2/row3
    __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [ A00, A10, A01, A11 ]
    __m256d t1 = _mm256_unpackhi_pd(row0, row1); // [ A02, A12, A03, A13 ]
    __m256d t2 = _mm256_unpacklo_pd(row2, row3); // [ A20, A30, A21, A31 ]
    __m256d t3 = _mm256_unpackhi_pd(row2, row3); // [ A22, A32, A23, A33 ]

    // Shuffle to group transposed rows
    __m256d tt0 = _mm256_shuffle_pd(t0, t2, 0x0); // [ A00, A10, A20, A30 ]
    __m256d tt1 = _mm256_shuffle_pd(t0, t2, 0xF); // [ A01, A11, A21, A31 ]
    __m256d tt2 = _mm256_shuffle_pd(t1, t3, 0x0); // [ A02, A12, A22, A32 ]
    __m256d tt3 = _mm256_shuffle_pd(t1, t3, 0xF); // [ A03, A13, A23, A33 ]

    // Store transposed block
    _mm256_storeu_pd(&dst[0 * dstStride], tt0);
    _mm256_storeu_pd(&dst[1 * dstStride], tt1);
    _mm256_storeu_pd(&dst[2 * dstStride], tt2);
    _mm256_storeu_pd(&dst[3 * dstStride], tt3);
}

/**
 * Transpose an MxN matrix (M rows, N cols) into an NxM matrix using:
 *   - 64x64 cache-blocking
 *   - 4x4 AVX2 kernel
 *   - Software prefetching
 *
 * A is in row-major: A[row][col] = A[row*N + col]
 * B is also in row-major but sized NxM: B[row][col] = A[col][row]
 *
 * Compile with -O3 -march=haswell (or /arch:AVX2 on MSVC).
 */
void transpose_avx2_prefetch(const double* __restrict A,
                             double* __restrict B,
                             std::size_t M,
                             std::size_t N)
{
    // Block sizes. 64x64 is a good starting point for many L1/L2 cache sizes.
    constexpr std::size_t BLOCK_SIZE = 64;

    // Process the matrix in 64x64 tiles
    for (std::size_t iBlock = 0; iBlock < M; iBlock += BLOCK_SIZE)
    {
        // Height of this tile
        const std::size_t iMax = std::min(iBlock + BLOCK_SIZE, M);

        for (std::size_t jBlock = 0; jBlock < N; jBlock += BLOCK_SIZE)
        {
            // Width of this tile
            const std::size_t jMax = std::min(jBlock + BLOCK_SIZE, N);

            // Within each 64x64 tile, do the 4x4 AVX2 transposes
            std::size_t i = iBlock;
            for (; i + 3 < iMax; i += 4)
            {
                std::size_t j = jBlock;
                for (; j + 3 < jMax; j += 4)
                {
                    // The source block starts at A[i][j] in row-major
                    const double* srcPtr = &A[i * N + j];
                    // The destination for the transposed block is B[j][i]
                    // But B is NxM in row-major => B[row = j + something][col = i + something]
                    double* dstPtr = &B[j * M + i];

                    // stride of each row in 'src' is N, in 'dst' is M
                    transpose_4x4_block_avx2_prefetch(srcPtr, dstPtr, N, M);
                }

                // Handle leftover columns (if width not multiple of 4)
                for (; j < jMax; j++)
                {
                    B[j * M + i + 0] = A[(i + 0) * N + j];
                    if (i + 1 < iMax)
                        B[j * M + i + 1] = A[(i + 1) * N + j];
                    if (i + 2 < iMax)
                        B[j * M + i + 2] = A[(i + 2) * N + j];
                    if (i + 3 < iMax)
                        B[j * M + i + 3] = A[(i + 3) * N + j];
                }
            }

            // Handle leftover rows (if height not multiple of 4)
            for (; i < iMax; i++)
            {
                for (std::size_t j = jBlock; j < jMax; j++)
                {
                    B[j * M + i] = A[i * N + j];
                }
            }
        }
    }
}

} // namespace _details

template<typename T>
Matrix<T> generateIotaMatrix(int M, int N)
{
    std::random_device rd;
    std::mt19937       gen(rd());
    T                  iota = 1;

    Matrix<T> matrix(M, N);
    std::generate(matrix.data(), matrix.data() + matrix.size(), [&] { return iota++; });
    return matrix;
}

template Matrix<float>  generateIotaMatrix<float>(int M, int N);
template Matrix<double> generateIotaMatrix<double>(int M, int N);

template<typename T>
Matrix<T> generateRandomMatrix(int M, int N)
{
    std::random_device rd;
    std::mt19937       gen(rd());
    T                  lower_bound = 1.0;
    T                  upper_bound = 10.0;

    std::uniform_real_distribution<T> dis(lower_bound, upper_bound);

    Matrix<T> matrix(M, N);
    std::generate(matrix.data(), matrix.data() + matrix.size(), [&] { return dis(gen); });
    return matrix;
}

template Matrix<float>  generateRandomMatrix<float>(int M, int N);
template Matrix<double> generateRandomMatrix<double>(int M, int N);

#if __STDCPP_FLOAT64_T__ == 1
#include <stdfloat>
template Matrix<std::bfloat16_t> generateRandomMatrix<std::bfloat16_t>(int M, int N);
#endif