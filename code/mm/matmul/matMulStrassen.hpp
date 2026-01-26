#pragma once

#include <mm/core/Matrix.hpp>
#include <mm/matmul/matMulZen5.hpp>
#include <mm/core/utils/utils.hpp>
#include <xsimd/xsimd.hpp>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace mm::strassen
{

constexpr int STRASSEN_THRESHOLD = 256;

// Helper to add two matrices: C = A + B
template<typename T>
void add_matrix(const T* A, int lda, const T* B, int ldb, T* C, int ldc, int M, int N)
{
    using batch_type        = xsimd::batch<T>;
    constexpr int simd_size = batch_type::size;

    for (int i = 0; i < M; ++i)
    {
        int j = 0;
        for (; j <= N - simd_size; j += simd_size)
        {
            batch_type a_batch = batch_type::load_unaligned(A + i * lda + j);
            batch_type b_batch = batch_type::load_unaligned(B + i * ldb + j);
            batch_type c_batch = a_batch + b_batch;
            c_batch.store_unaligned(C + i * ldc + j);
        }
        for (; j < N; ++j)
        {
            C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
        }
    }
}

// Helper to subtract two matrices: C = A - B
template<typename T>
void sub_matrix(const T* A, int lda, const T* B, int ldb, T* C, int ldc, int M, int N)
{
    using batch_type        = xsimd::batch<T>;
    constexpr int simd_size = batch_type::size;

    for (int i = 0; i < M; ++i)
    {
        int j = 0;
        for (; j <= N - simd_size; j += simd_size)
        {
            batch_type a_batch = batch_type::load_unaligned(A + i * lda + j);
            batch_type b_batch = batch_type::load_unaligned(B + i * ldb + j);
            batch_type c_batch = a_batch - b_batch;
            c_batch.store_unaligned(C + i * ldc + j);
        }
        for (; j < N; ++j)
        {
            C[i * ldc + j] = A[i * lda + j] - B[i * ldb + j];
        }
    }
}

// Simple SIMD Matrix Multiplication for base case
// C = A * B (Overwrites C)
template<typename T>
void matmul_base(const T* A, int lda, const T* B, int ldb, T* C, int ldc, int M, int N, int K)
{
    // Initialize C to 0
    for (int i = 0; i < M; ++i)
        std::fill_n(C + i * ldc, N, T(0));

    using batch_type        = xsimd::batch<T>;
    constexpr int simd_size = batch_type::size;

    // Simple cache-friendly loop order (i, k, j)
    for (int i = 0; i < M; ++i)
    {
        for (int k = 0; k < K; ++k)
        {
            T          a_val = A[i * lda + k];
            batch_type a_vec(a_val);

            int j = 0;
            for (; j <= N - simd_size; j += simd_size)
            {
                batch_type b_vec = batch_type::load_unaligned(B + k * ldb + j);
                batch_type c_vec = batch_type::load_unaligned(C + i * ldc + j);
                c_vec += a_vec * b_vec;
                c_vec.store_unaligned(C + i * ldc + j);
            }
            for (; j < N; ++j)
            {
                C[i * ldc + j] += a_val * B[k * ldb + j];
            }
        }
    }
}

template<typename T>
void strassen_recursive(const T* A,
                        int      lda,
                        const T* B,
                        int      ldb,
                        T*       C,
                        int      ldc,
                        int      M,
                        int      N,
                        int      K,
                        int      depth)
{
    // Base case
    if (M <= STRASSEN_THRESHOLD || N <= STRASSEN_THRESHOLD || K <= STRASSEN_THRESHOLD || depth == 0)
    {
        matmul_base(A, lda, B, ldb, C, ldc, M, N, K);
        return;
    }

    int halfM = M / 2;
    int halfN = N / 2;
    int halfK = K / 2;

    size_t size_A = halfM * halfK;
    size_t size_B = halfK * halfN;
    size_t size_C = halfM * halfN;

    std::vector<T> buffer(5 * size_A + 5 * size_B + 7 * size_C);
    T*             ptr = buffer.data();

    T* s2 = ptr;
    ptr += size_A;
    T* s3 = ptr;
    ptr += size_A;
    T* s5 = ptr;
    ptr += size_A;
    T* s7 = ptr;
    ptr += size_A;
    T* s9 = ptr;
    ptr += size_A;

    T* s1 = ptr;
    ptr += size_B;
    T* s4 = ptr;
    ptr += size_B;
    T* s6 = ptr;
    ptr += size_B;
    T* s8 = ptr;
    ptr += size_B;
    T* s10 = ptr;
    ptr += size_B;

    T* p1 = ptr;
    ptr += size_C;
    T* p2 = ptr;
    ptr += size_C;
    T* p3 = ptr;
    ptr += size_C;
    T* p4 = ptr;
    ptr += size_C;
    T* p5 = ptr;
    ptr += size_C;
    T* p6 = ptr;
    ptr += size_C;
    T* p7 = ptr;
    ptr += size_C;

    // Pointers to submatrices of A and B
    // A is MxK
    const T* A11 = A;
    const T* A12 = A + halfK;
    const T* A21 = A + halfM * lda;
    const T* A22 = A + halfM * lda + halfK;

    // B is KxN
    const T* B11 = B;
    const T* B12 = B + halfN;
    const T* B21 = B + halfK * ldb;
    const T* B22 = B + halfK * ldb + halfN;

// Compute S matrices (Parallelizable)
#pragma omp task
    sub_matrix(B12, ldb, B22, ldb, s1, halfN, halfK, halfN); // S1 = B12 - B22

#pragma omp task
    add_matrix(A11, lda, A12, lda, s2, halfK, halfM, halfK); // S2 = A11 + A12

#pragma omp task
    add_matrix(A21, lda, A22, lda, s3, halfK, halfM, halfK); // S3 = A21 + A22

#pragma omp task
    sub_matrix(B21, ldb, B11, ldb, s4, halfN, halfK, halfN); // S4 = B21 - B11

#pragma omp task
    add_matrix(A11, lda, A22, lda, s5, halfK, halfM, halfK); // S5 = A11 + A22

#pragma omp task
    add_matrix(B11, ldb, B22, ldb, s6, halfN, halfK, halfN); // S6 = B11 + B22

#pragma omp task
    sub_matrix(A12, lda, A22, lda, s7, halfK, halfM, halfK); // S7 = A12 - A22

#pragma omp task
    add_matrix(B21, ldb, B22, ldb, s8, halfN, halfK, halfN); // S8 = B21 + B22

#pragma omp task
    sub_matrix(A11, lda, A21, lda, s9, halfK, halfM, halfK); // S9 = A11 - A21

#pragma omp task
    add_matrix(B11, ldb, B12, ldb, s10, halfN, halfK, halfN); // S10 = B11 + B12

#pragma omp taskwait

// Recursive calls
#pragma omp task
    strassen_recursive(A11, lda, s1, halfN, p1, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp task
    strassen_recursive(s2, halfK, B22, ldb, p2, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp task
    strassen_recursive(s3, halfK, B11, ldb, p3, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp task
    strassen_recursive(A22, lda, s4, halfN, p4, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp task
    strassen_recursive(s5, halfK, s6, halfN, p5, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp task
    strassen_recursive(s7, halfK, s8, halfN, p6, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp task
    strassen_recursive(s9, halfK, s10, halfN, p7, halfN, halfM, halfN, halfK, depth - 1);

#pragma omp taskwait

    // Compute Result Submatrices
    T* C11 = C;
    T* C12 = C + halfN;
    T* C21 = C + halfM * ldc;
    T* C22 = C + halfM * ldc + halfN;

    using batch_type        = xsimd::batch<T>;
    constexpr int simd_size = batch_type::size;

#pragma omp task
    {
        for (int i = 0; i < halfM; ++i)
        {
            int j = 0;
            for (; j <= halfN - simd_size; j += simd_size)
            {
                auto vP1 = batch_type::load_unaligned(p1 + i * halfN + j);
                auto vP2 = batch_type::load_unaligned(p2 + i * halfN + j);
                auto vP3 = batch_type::load_unaligned(p3 + i * halfN + j);
                auto vP4 = batch_type::load_unaligned(p4 + i * halfN + j);
                auto vP5 = batch_type::load_unaligned(p5 + i * halfN + j);
                auto vP6 = batch_type::load_unaligned(p6 + i * halfN + j);
                auto vP7 = batch_type::load_unaligned(p7 + i * halfN + j);

                auto vC11 = vP5 + vP4 - vP2 + vP6;
                auto vC12 = vP1 + vP2;
                auto vC21 = vP3 + vP4;
                auto vC22 = vP5 + vP1 - vP3 - vP7;

                vC11.store_unaligned(C11 + i * ldc + j);
                vC12.store_unaligned(C12 + i * ldc + j);
                vC21.store_unaligned(C21 + i * ldc + j);
                vC22.store_unaligned(C22 + i * ldc + j);
            }
            for (; j < halfN; ++j)
            {
                T valP1 = p1[i * halfN + j];
                T valP2 = p2[i * halfN + j];
                T valP3 = p3[i * halfN + j];
                T valP4 = p4[i * halfN + j];
                T valP5 = p5[i * halfN + j];
                T valP6 = p6[i * halfN + j];
                T valP7 = p7[i * halfN + j];

                C11[i * ldc + j] = valP5 + valP4 - valP2 + valP6;
                C12[i * ldc + j] = valP1 + valP2;
                C21[i * ldc + j] = valP3 + valP4;
                C22[i * ldc + j] = valP5 + valP1 - valP3 - valP7;
            }
        }
    }
#pragma omp taskwait
}

// Helper to pad matrix
template<typename T>
Matrix<T> pad_matrix(const Matrix<T>& A, int paddedM, int paddedN)
{
    Matrix<T> padded(paddedM, paddedN);
    // Copy and zero out rest
    // Since Matrix constructor initializes to 0, just copy
    for (int i = 0; i < A.row(); ++i)
    {
        std::copy(A.data() + i * A.col(),
                  A.data() + i * A.col() + A.col(),
                  padded.data() + i * padded.col());
    }
    return padded;
}

// Helper to unpad matrix
template<typename T>
void unpad_matrix(Matrix<T>& C, const Matrix<T>& paddedC, int originalM, int originalN)
{
    for (int i = 0; i < originalM; ++i)
    {
        std::copy(paddedC.data() + i * paddedC.col(),
                  paddedC.data() + i * paddedC.col() + originalN,
                  C.data() + i * C.col());
    }
}

template<typename T>
void matMulStrassen(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    int M = A.row();
    int K = A.col();
    int N = B.col();

    // Check if padding is needed
    // Simple approach: Pad to next power of 2.
    auto next_pow2 = [](int x)
    {
        int power = 1;
        while (power < x)
            power *= 2;
        return power;
    };

    int max_dim    = std::max({M, K, N});
    int padded_dim = next_pow2(max_dim);

    bool padding_needed = (padded_dim != M || padded_dim != K || padded_dim != N);

    if (padding_needed)
    {
        Matrix<T> Ap = pad_matrix(A, padded_dim, padded_dim);
        Matrix<T> Bp = pad_matrix(B, padded_dim, padded_dim);
        Matrix<T> Cp(padded_dim, padded_dim);

#pragma omp parallel
        {
#pragma omp single
            strassen_recursive(Ap.data(),
                               padded_dim,
                               Bp.data(),
                               padded_dim,
                               Cp.data(),
                               padded_dim,
                               padded_dim,
                               padded_dim,
                               padded_dim,
                               10);
        }

        unpad_matrix(C, Cp, M, N);
    }
    else
    {
#pragma omp parallel
        {
#pragma omp single
            strassen_recursive(A.data(), K, B.data(), N, C.data(), N, M, N, K, 10);
        }
    }
}

} // namespace mm::strassen
