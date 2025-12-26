#pragma once

#include <mm/core/Matrix.hpp>

namespace mm::gpu{
void matmulGpuBlas(const Matrix<float>& A, const Matrix<float>& B, Matrix<float>& C);
}