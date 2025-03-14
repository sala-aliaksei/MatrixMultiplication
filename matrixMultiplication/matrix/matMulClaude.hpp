#pragma once
#include "Matrix.hpp"

void multiply_matrices_optimized(const Matrix<double>& A,
                                 const Matrix<double>& B,
                                 Matrix<double>&       C);
