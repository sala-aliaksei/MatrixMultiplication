#include "mm/tpi/matMulOpenBlas.hpp"

#include "openblas/cblas.h"

/*****************     OPEN BLAS     *******************/

void matrixMulOpenBlas(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c)
{
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                a.row(),
                b.col(),
                a.col(),
                1.0,
                a.data(),
                a.col(),
                b.data(),
                b.col(),
                1.0,
                c.data(),
                c.col());
}

void matrixMulOpenBlas_TP(MatrixSet& ms)
{
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                ms.a.row(),
                ms.b.col(),
                ms.a.col(),
                1.0,
                ms.a.data(),
                ms.a.col(),
                ms.b.data(),
                ms.b.col(),
                1.0,
                ms.c.data(),
                ms.c.col());
}

void matrixMulOpenBlas(MatrixSet& ms)
{
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                ms.a.row(),
                ms.b.col(),
                ms.a.col(),
                1.0,
                ms.a.data(),
                ms.a.col(),
                ms.b.data(),
                ms.b.col(),
                1.0,
                ms.c.data(),
                ms.c.col());
}
