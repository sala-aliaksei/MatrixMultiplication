#pragma once
#include <mm/core/Matrix.hpp>

namespace mm::tpi
{
template<typename T>
void matrixMulOpenBlas(const Matrix<T>& a, const Matrix<T>& b, Matrix<T>& c);
} // namespace mm::tpi

void matrixMulOpenBlas(MatrixSet& ms);
void matrixMulOpenBlas_TP(MatrixSet& ms);
