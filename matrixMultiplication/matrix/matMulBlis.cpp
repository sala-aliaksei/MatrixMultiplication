#include "matMulBlis.hpp"

#define BLIS_FAMILY_HASWELL

#include <blis/blis.h>

void matmulBlis(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    int m = A.row();
    int n = B.col();
    int k = A.col();

    double alpha = 1.0, beta = 0.0; // Scaling factors
    int    N = A.row();
    int    M = B.col();
    int    K = A.col();

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    bli_dgemm(BLIS_NO_TRANSPOSE,
              BLIS_NO_TRANSPOSE,
              N,
              M,
              K,
              &alpha,
              A.data(),
              1,
              M,
              B.data(),
              1,
              K,
              &beta,
              C.data(),
              1,
              M);
}
