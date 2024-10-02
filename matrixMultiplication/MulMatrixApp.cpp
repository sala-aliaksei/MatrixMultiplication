#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulFunctions.hpp"

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        bool dynamic  = true;
        auto matrices = initMatrix();
        if (dynamic == true)
        {
            // TODO: Select algorithm by analyzing cmdline args
            auto block_size           = 8u;
            bool transpose_matrix     = true;
            bool manual_vectorization = true;

            DynamicMatrixMul mul(std::thread::hardware_concurrency(),
                                 block_size,
                                 transpose_matrix,
                                 manual_vectorization);

            mul(matrices.a, matrices.b, matrices.c);
        }
        else
        {
            matrixMul_MT_VT_BL_TP(matrices);
        }

        // std::cout << matrices.b;
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
