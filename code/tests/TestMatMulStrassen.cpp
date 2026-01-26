#include <gtest/gtest.h>
#include <mm/matmul/matMulStrassen.hpp>
#include <mm/core/Matrix.hpp>
#include <mm/core/utils/utils.hpp>

// Simple naive matmul for verification
template<typename T>
void matMul_Verify(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
// Use OpenMP to speed up verification
#pragma omp parallel for collapse(2)
    for (int i = 0; i < A.row(); ++i)
    {
        for (int j = 0; j < B.col(); ++j)
        {
            T sum = 0;
            for (int k = 0; k < A.col(); ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

TEST(Strassen, SmallMatrix)
{
    int M = 64;
    int K = 64;
    int N = 64;

    auto           A = generateRandomMatrix<double>(M, K);
    auto           B = generateRandomMatrix<double>(K, N);
    Matrix<double> C(M, N);
    Matrix<double> C_ref(M, N);

    mm::strassen::matMulStrassen(A, B, C);
    matMul_Verify(A, B, C_ref);

    ASSERT_TRUE(C == C_ref);
}

TEST(Strassen, MediumMatrix)
{
    int M = 512;
    int K = 512;
    int N = 512;

    auto           A = generateRandomMatrix<double>(M, K);
    auto           B = generateRandomMatrix<double>(K, N);
    Matrix<double> C(M, N);
    Matrix<double> C_ref(M, N);

    mm::strassen::matMulStrassen(A, B, C);
    matMul_Verify(A, B, C_ref);

    // Use loose tolerance as Strassen can have different numerical properties
    ASSERT_TRUE(C == C_ref);
}

TEST(Strassen, OddDimensions)
{
    int M = 300;
    int K = 300;
    int N = 300;

    auto           A = generateRandomMatrix<double>(M, K);
    auto           B = generateRandomMatrix<double>(K, N);
    Matrix<double> C(M, N);
    Matrix<double> C_ref(M, N);

    mm::strassen::matMulStrassen(A, B, C);
    matMul_Verify(A, B, C_ref);

    ASSERT_TRUE(C == C_ref);
}

TEST(Strassen, Rectangular)
{
    int M = 200;
    int K = 300;
    int N = 150;

    auto           A = generateRandomMatrix<double>(M, K);
    auto           B = generateRandomMatrix<double>(K, N);
    Matrix<double> C(M, N);
    Matrix<double> C_ref(M, N);

    mm::strassen::matMulStrassen(A, B, C);
    matMul_Verify(A, B, C_ref);

    ASSERT_TRUE(C == C_ref);
}
