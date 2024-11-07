#include "matrixMultiplication/matrix/MatrixMul.hpp"

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        constexpr std::size_t I = 768 * 4;
        constexpr std::size_t J = 768 * 4;
        constexpr std::size_t K = 768 * 4;

        auto matrices = initMatrix(I, J, K);

        DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});

        mul(matrices.a, matrices.b, matrices.c);

        // std::cout << matrices.b;
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
