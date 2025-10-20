#include "mm/core/Shape.hpp"

#include <chrono>
#include <memory>
#include <mm/matmul/matMulAutotune.hpp>
#include <mm/matmul/matMulZen5.hpp>
#include <mm/core/utils/utils.hpp>

#include <iostream>

// This application will be used to analyze perfomance of the fastest
// implementation Example: using perf.
#include <mdspan>
#include <stdfloat>
#include <print>
#include <vector>

#include <experimental/simd>

namespace stdx = std::experimental;

int main()
{

    try
    {
        constexpr std::size_t NN = 3072;
        constexpr std::size_t M  = NN;
        constexpr std::size_t N  = NN;
        constexpr std::size_t K  = NN;

        // auto matrices = initDoubleMatrix(I, J, K);
        auto a = generateRandomMatrix<double>(M, K);
        auto b = generateRandomMatrix<double>(K, N);
        auto c = generateRandomMatrix<double>(M, N);

        {
            PROFILE("matMulZen5MTBlockingL1");
            // mm::zen5::matMulZen5MTBlockingL1(a, b, c);
        }

        std::cout << "Done!\n";
        return 0;
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
