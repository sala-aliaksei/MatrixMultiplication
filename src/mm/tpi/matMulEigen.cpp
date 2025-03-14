#include "mm/tpi/matMulEigen.hpp"
#include <mm/core/Matrix.hpp>

MatrixEigenSet initEigenMatrix(std::size_t isize, std::size_t jsize, std::size_t ksize)
{
    MatrixEigenSet ms;

    ms.a = Eigen::MatrixXd(isize, ksize);
    ms.b = Eigen::MatrixXd(ksize, jsize);
    ms.c = Eigen::MatrixXd(isize, jsize);

    // Populate the matrix using two for loops
    for (int i = 0; i < isize; ++i)
    {
        for (int j = 0; j < ksize; ++j)
        {
            ms.a(i, j) = i + j;
        }
    }

    for (int i = 0; i < ksize; ++i)
    {
        for (int j = 0; j < jsize; ++j)
        {
            ms.b(i, j) = j - i;
        }
    }
    return ms;
}

void matrixMulEigen(MatrixEigenSet& ms)
{
    ms.c = ms.a * ms.b;
}
