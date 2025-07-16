#pragma once
#include <mm/core/Matrix.hpp>

void matMulLoops(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMulLoopsRepack(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMulLoopsIKJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMulLoopsBPacked(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMulLoopsRepackIKJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
