#pragma once

#include <mm/core/Matrix.hpp>

namespace mm::zen5
{
void matMulZen5(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
} // namespace mm::zen5