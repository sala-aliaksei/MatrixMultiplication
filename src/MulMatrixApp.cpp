#include <mm/matmul/matMul.hpp>
#include <mm/matmul/matMulSimd.hpp>

#include <iostream>

// This application will be used to analyze perfomance of the fastest implementation
// Example: using perf.
int main()
{
    try
    {
        constexpr std::size_t NN = 720 * 4;
        constexpr std::size_t I  = NN;
        constexpr std::size_t J  = NN;
        constexpr std::size_t K  = NN;

        auto matrices = initMatrix(I, J, K);

        matMulSimd(matrices.a, matrices.b, matrices.c);
        std::cout << "Done!\n";
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
