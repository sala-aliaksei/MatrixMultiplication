#include "mm/tpi/matMulEigen.hpp"

MatrixEigenSet initEigenMatrix(int M, int N, int K)
{
    MatrixEigenSet ms;

    ms.a = Eigen::MatrixXd(M, K);
    ms.b = Eigen::MatrixXd(K, N);
    ms.c = Eigen::MatrixXd(M, N);

    // Populate the matrix using two for loops
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            ms.a(i, j) = i + j;
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            ms.b(i, j) = j + i;
        }
    }
    return ms;
}

MatrixEigenSet initEigenMatrix(const Matrix<double>& a, const Matrix<double>& b)
{
    MatrixEigenSet ms;

    auto M = a.row();
    auto N = a.col();
    auto K = b.col();

    ms.a = Eigen::MatrixXd(M, K);
    ms.b = Eigen::MatrixXd(K, N);
    ms.c = Eigen::MatrixXd(M, N);

    // Populate the matrix using two for loops
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            ms.a(i, j) = a(i, j);
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            ms.b(i, j) = b(i, j);
        }
    }
    return ms;
}

void matrixMulEigen(MatrixEigenSet& ms)
{
    ms.c = ms.a * ms.b;
}
