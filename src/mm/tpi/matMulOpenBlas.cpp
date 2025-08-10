#include "mm/tpi/matMulOpenBlas.hpp"

// #include "openblas/cblas.h"
#include <cblas.h>

/*****************     OPEN BLAS     *******************/
namespace mm::tpi
{
template<typename T>
void matrixMulOpenBlas(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c)
{
    if constexpr (std::is_same_v<T, float>)
    {
        cblas_sgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    a.row(),
                    b.col(),
                    a.col(),
                    1.0f,
                    a.data(),
                    a.col(),
                    b.data(),
                    b.col(),
                    1.0f,
                    c.data(),
                    c.col());
    }
    else if constexpr (std::is_same_v<T, double>)
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
    else
    {
        static_assert(false, "Unsupported type");
    }
}

template void matrixMulOpenBlas<float>(const Matrix<float>& a,
                                       const Matrix<float>& b,
                                       Matrix<float>&       c);
template void matrixMulOpenBlas<double>(const Matrix<double>& a,
                                        const Matrix<double>& b,
                                        Matrix<double>&       c);
} // namespace mm::tpi

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
