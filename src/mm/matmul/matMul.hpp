#pragma once
#include <mm/core/Matrix.hpp>

namespace mm
{
void matMul_Naive(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Naive_Order(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Naive_Order_KIJ(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Naive_Tile(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Simd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx_Cache_Regs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx_Cache_Regs_Unroll(const Matrix<double>& A,
                                  const Matrix<double>& B,
                                  Matrix<double>&       C);
void matMul_Avx_Cache_Regs_UnrollRW(const Matrix<double>& A,
                                    const Matrix<double>& B,
                                    Matrix<double>&       C);

void matMul_Avx_Cache_Regs_Unroll_BPack(const Matrix<double>& A,
                                        const Matrix<double>& B,
                                        Matrix<double>&       C);

void matMul_Avx_Cache_Regs_Unroll_MT(const Matrix<double>& A,
                                     const Matrix<double>& B,
                                     Matrix<double>&       C);

void matMul_Avx_Cache_Regs_Unroll_BPack_MT(const Matrix<double>& A,
                                           const Matrix<double>& B,
                                           Matrix<double>&       C);

void matMul_Avx_AddRegs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Avx_AddRegs_Unroll(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Tails(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_ManualTail(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
} // namespace mm
