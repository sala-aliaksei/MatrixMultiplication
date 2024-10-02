#include "MatrixMulEigen.hpp"
#include "Matrix.hpp"

MatrixEigenSet initEigenMatrix()
{
    MatrixEigenSet ms;

    auto rows = N;
    auto cols = N;

    ms.a = Eigen::MatrixXd(rows, cols);
    ms.b = Eigen::MatrixXd(rows, cols);
    ms.c = Eigen::MatrixXd(rows, cols);

    // Populate the matrix using two for loops
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            ms.a(i, j) = i;
            ms.b(i, j) = j;
            ms.c(i, j) = 0;
        }
    }
    return ms;
}

void matrixMulEigen(MatrixEigenSet& ms)
{
    ms.c = ms.a * ms.b;
}
