#include <mm/matmul/matMul.hpp>
#include <mm/matmul/matMulRegOpt.hpp>

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        constexpr std::size_t NN = 480;
        constexpr std::size_t I  = NN;
        constexpr std::size_t J  = NN;
        constexpr std::size_t K  = NN;

        auto matrices = initMatrix(I, J, K);

        DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});

        // mul(matrices.a, matrices.b, matrices.c);

        // matMulRegOpt(matrices.a, matrices.b, matrices.c);

        testReorderMatrix();

        // std::cout << matrices.a;
        //         std::cout << "---------------------\n\n\n";
        //         std::cout << matrices.c;

        //        for (int i = 0; i < I; ++i)
        //        {
        //            for (int j = 0; j < J; ++j)
        //            {
        //                if (matrices.a(i, j) != matrices.c(j, i))
        //                {
        //                    std::cout << "transpose doesn't work!\n";
        //                    return -1;
        //                }
        //            }
        //        }
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
