#pragma once
#include <mm/core/Matrix.hpp>

void matrixMulOpenBlas(const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c);
void matrixMulOpenBlas(MatrixSet& ms);
void matrixMulOpenBlas_TP(MatrixSet& ms);
