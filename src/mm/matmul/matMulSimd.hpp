#pragma once
#include <mm/core/Matrix.hpp>

void matMulSimd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
