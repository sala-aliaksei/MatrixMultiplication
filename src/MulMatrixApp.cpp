#include "mm/matmul/matMul.hpp"
// #include <mm/matmul/matMulSimd.hpp>
#include <mm/matmul/matMulAutotune.hpp>

#include <iostream>

// This application will be used to analyze perfomance of the fastest
// implementation Example: using perf.

// #include <mdspan/mdspan.hpp>

template<typename T, int Mc, int Nc, int N>
struct MDArray
{
    MDArray(T* arr)
      : _arr(arr)
    {
    }

    // TODO: Compile time out of bound check
    T& operator()(int i, int j)
    {
        // i < Mc
        // j < Nc
        return *(_arr + i * N + j);
    }

  private:
    T* _arr;
};

// namespace stdx = std::experimental;

constexpr int N  = 6; // Original matrix dimensions
constexpr int Mr = 2; // Submatrix row size
constexpr int Nr = 3; // Submatrix column size

// infinite loop
// int testMDSpan()
//{
//    std::cout << "Matrix creating..." << std::endl;
//    // Define a storage for the N x N matrix
//    std::vector<int> elems(N * N);
//    // Initialize the matrix with sample values
//    for (int i = 0; i < N; ++i)
//    {
//        for (int j = 0; j < N; ++j)
//        {
//            elems[i * N + j] = i * N + j;
//        }
//    }
//    std::cout << "Matrix is initialized" << std::endl;

//    //
//    {
//        // Define stride array manually
//        std::array<std::size_t, 2> strides = {N, 1}; // Row-major layout

//        // Create a mapping object for layout_stride
//        Kokkos::layout_stride::mapping<Kokkos::extents<std::size_t, Mr, Nr>>
//        stride_map{
//          Kokkos::extents<std::size_t, Mr, Nr>{}, strides};

//        // Create a submatrix mdspan view using layout_stride
//        Kokkos::mdspan<int, Kokkos::extents<std::size_t, Mr, Nr>,
//        Kokkos::layout_stride> submatrix{
//          elems.data() + 1 + 2 * N, stride_map};

//        std::cout << "Matrix subspan is created" << std::endl;

//        for (int i = 0; i < Mr; ++i)
//        {
//            for (int j = 0; j < Nr; ++j)
//            {
//                std::cout << submatrix[i, j] << " ";
//            }

//            std::cout << std::endl;
//        }
//        std::cout << "------------------" << std::endl;
//    }
//    // Finalize Kokkos
//}

int main()
{
    try
    {
        constexpr std::size_t NN = 720 * 4;
        constexpr std::size_t I  = NN;
        constexpr std::size_t J  = NN;
        constexpr std::size_t K  = NN;

        auto matrices = initDoubleMatrix(I, J, K);

        // matMulSimd(matrices.a, matrices.b, matrices.c);

        // testMDSpan();

        std::cout << "Done!\n";
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
