#pragma once
#include "Matrix.hpp"
#include <vector>

void matrixMul_MT_VT_BL_TP(MatrixSet& ms);

void matrixMulOpenBlas(MatrixSet& ms);
void matrixMulOpenBlas_TP(MatrixSet& ms);

void matrixMul_Naive(MatrixSet& set);
void matrixMul_Naive_TP(MatrixSet& set);
void matrixMul_MT_VT_BL(MatrixSet& set);
