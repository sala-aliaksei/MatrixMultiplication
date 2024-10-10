#include "matrixMultiplication/matrix/MatrixMul.hpp"

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        constexpr std::size_t M        = 8 * 4 * 256;
        constexpr std::size_t N        = 8 * 4 * 256;
        auto                  matrices = initMatrix(N, M);

        // TODO: Select algorithm by analyzing cmdline args
        auto block_size           = 8u;
        bool transpose_matrix     = true;
        bool manual_vectorization = true;

        DynamicMatrixMul mul(MatrixMulConfig{
          std::thread::hardware_concurrency(), block_size, transpose_matrix, manual_vectorization});

        mul(matrices.a, matrices.b, matrices.c);

        // std::cout << matrices.b;
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
