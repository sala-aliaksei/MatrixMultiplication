#include "matrixMultiplication/matrix/MatrixMul.hpp"

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        constexpr std::size_t I = 768;
        constexpr std::size_t J = 768;
        constexpr std::size_t K = 768;

        auto matrices = initMatrix(I, J, K);

        // TODO: Select algorithm by analyzing cmdline args
        auto enable_block_opt     = true;
        bool transpose_matrix     = false;
        bool manual_vectorization = true;

        DynamicMatrixMul mul(
          MatrixMulConfig{true, enable_block_opt, transpose_matrix, manual_vectorization});

        mul(matrices.a, matrices.b, matrices.c);

        // std::cout << matrices.b;
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
