#pragma once
#include <mm/core/Matrix.hpp>

void matMulRegOpt(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMulRegOptBuff(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

int packMatrixWithOrder(int order,
                        int m,
                        int n,
                        double* __restrict a,
                        int lda,
                        double* __restrict b);

void testReorderMatrix();
