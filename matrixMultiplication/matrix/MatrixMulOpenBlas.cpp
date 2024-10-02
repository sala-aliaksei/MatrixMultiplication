
#include "MatrixMulOpenBlas.hpp"

#include "openblas/cblas.h"

/*****************     OPEN BLAS     *******************/

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
                ms.a.row(),
                ms.b.data(),
                ms.b.row(),
                1.0,
                ms.res.data(),
                ms.res.row());
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
                ms.a.row(),
                ms.b.data(),
                ms.b.row(),
                1.0,
                ms.res.data(),
                ms.res.row());
}
