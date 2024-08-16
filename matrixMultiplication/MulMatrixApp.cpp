#include "matrixMultiplication/matrix/MatrixMul.hpp"

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        // TODO: Select algorithm by analyzing cmdline args
        auto block_size           = 8u;
        bool transpose_matrix     = true;
        bool manual_vectorization = true;

        MatrixMul mul(
          std::thread::hardware_concurrency(), block_size, transpose_matrix, manual_vectorization);

        auto matrices = initMatrix();
        mul(matrices.a, matrices.b, matrices.res);

        // std::cout << matrices.b;
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
