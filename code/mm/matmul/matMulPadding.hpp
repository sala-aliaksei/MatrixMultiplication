#pragma once
#include <mm/core/Matrix.hpp>

void matMulPadding(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

/*
1. What if Nc%Nr != 0 within matrix where padding is not applicable?
*/
