#include "mm/core/Shape.hpp"
// #include "mm/matmul/matMul.hpp"
// #include <mm/matmul/matMulSimd.hpp>
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

namespace
{
template<typename T, int WIDTH>
using fix_simd = stdx::fixed_size_simd<T, WIDTH>;

void test_bfloat16()
{
    std::vector<std::bfloat16_t> v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    auto ms2 = std::mdspan(v.data(), 2, 6);
    auto ms3 = std::mdspan(v.data(), 2, 3, 2);

    for (std::size_t i = 0; i != ms2.extent(0); i++)
    {
        for (std::size_t j = 0; j != ms2.extent(1); j++)
        {
            std::print("{} ", ms2[i, j]);
        }
        std::println("");
    }

    for (std::size_t i = 0; i != ms3.extent(0); i++)
    {
        for (std::size_t j = 0; j != ms3.extent(1); j++)
        {
            for (std::size_t k = 0; k != ms3.extent(2); k++)
            {
                std::print("{} ", ms3[i, j, k]);
            }
            std::println("");
        }
        std::println("");
    }

    constexpr auto num_of_elems_in_reg =
      stdx::simd_size_v<std::bfloat16_t, stdx::simd_abi::native<std::bfloat16_t>>;
    std::println("num_of_elems_in_reg: {}", num_of_elems_in_reg);
    stdx::fixed_size_simd<std::bfloat16_t, num_of_elems_in_reg> r1 = {1};
    stdx::fixed_size_simd<std::bfloat16_t, num_of_elems_in_reg> r2 = {2};
    stdx::fixed_size_simd<std::bfloat16_t, num_of_elems_in_reg> r3 = {0};

    r3 = r1 + r2;
    std::bfloat16_t res[num_of_elems_in_reg];
    r3.copy_to(res, stdx::element_aligned);
    for (int i = 0; i < num_of_elems_in_reg; i++)
    {
        std::println("res[{}]: {}", i, res[i]);
    }
}

// void test_mdspan()
// {
//     std::vector v{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

//     // View data as contiguous memory representing 2 rows of 6 ints each
//     auto ms2 = std::mdspan(v.data(), 2, 6);
//     // View the same data as a 3D array 2 x 3 x 2
//     auto ms3 = std::mdspan(v.data(), 2, 3, 2);

//     // Write data using 2D view
//     for (std::size_t i = 0; i != ms2.extent(0); i++)
//         for (std::size_t j = 0; j != ms2.extent(1); j++)
//             ms2[i, j] = i * 1000 + j;

//     // Read back using 3D view
//     for (std::size_t i = 0; i != ms3.extent(0); i++)
//     {
//         std::println("slice @ i = {}", i);
//         for (std::size_t j = 0; j != ms3.extent(1); j++)
//         {
//             for (std::size_t k = 0; k != ms3.extent(2); k++)
//                 std::print("{} ", ms3[i, j, k]);
//             std::println("");
//         }
//     }
// }

} // namespace

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

template<std::size_t Mc, std::size_t Kc>
void pack_tile(std::mdspan<double, std::extents<std::size_t, Mc, Kc>> tile,
               std::mdspan<double, std::dextents<std::size_t, 2>>     matrix)
{
    for (std::size_t i = 0; i < Mc; i++)
    {
        for (std::size_t k = 0; k < Kc; k++)
        {
            tile[i, k] = matrix[i, k];
        }
    }
}

enum class CPUArch
{
    ZEN5
};

template<CPUArch arch>
struct CPUConfig;

template<>
struct CPUConfig<CPUArch::ZEN5>
{
    // static constexpr std::size_t l1_size = 48 * 1024;

    static constexpr int Mr{8};
    static constexpr int Nr{16};
    // static constexpr int Mc{96};  // 96 is optimal for ZEN5
    // static constexpr int Nc{160}; // 160 is optimal for ZEN5
    // static constexpr int Kc{192}; // 192 is optimal for ZEN5

    static constexpr int Mc{4 * 192};
    static constexpr int Nc{192};
    static constexpr int Kc{192};
    static_assert(Mc % Mr == 0, "Mc%  Mr == 0");
    static_assert(Nc % Nr == 0, "Mc%  Mr == 0");
    static constexpr std::size_t l1_size = Kc * (Mc + Nc);
};

template<typename config>
struct L1Cache
{
    L1Cache()
      : buf(config::l1_size, 0.0)
    {
    }

    // buf: a_tile + b_tile = l1_size

    // TODO: Temporarily disabled, pretend I have more cache mem to chech cache misseses
    // static_assert(config::Kc * (config::Mc + config::Nc) <= config::l1_size,
    //               "Kc*(Mc+Nc) <= l1_size");

    std::vector<double> buf;

    std::mdspan<double, std::extents<std::size_t, config::Mc, config::Kc>> a_tile{buf.data(),
                                                                                  config::Mc,
                                                                                  config::Kc};

    std::mdspan<double, std::extents<std::size_t, config::Kc, config::Nc>> b_tile{
      buf.data() + config::Kc * config::Mc,
      config::Kc,
      config::Nc};
};

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
            mm::zen5::matMulZen5MTBlockingL1(a, b, c);
        }
        return 0;

        // every i itertaion, elems from a matrix must be evecited from cahce
        L1Cache<CPUConfig<CPUArch::ZEN5>> l1_cache;

        // could be static extents
        auto& a_tile = l1_cache.a_tile;
        auto& b_tile = l1_cache.b_tile;

        std::println("a_tile.size(): {}", a_tile.size());
        std::println("b_tile.size(): {}", b_tile.size());
        std::println("b.size(): {}", b.size());
        std::println("l1_cache.buf size: {}", l1_cache.buf.size());

        // std::copy(b.data(), b.data() + b_tile.size(), l1_cache.buf.data() + Kc * Mc);
        pack_tile(b_tile, std::mdspan(&b(0, 0), K, N));

        using CPU = CPUConfig<CPUArch::ZEN5>;
        int k     = 0;
        for (int i = 0; i < M; i += CPU::Mc)
        {
            // std::println("i: {}, k: {}", i, k);
            //  here next 'a' tile must evict previous one
            pack_tile(a_tile, std::mdspan(&a(i, k), M - i, K - k));

            // tile multiplication
            for (int i = 0; i < CPU::Mc; i++)
            {
                for (int j = 0; j < CPU::Nc; j += 1)
                {
                    // c[i,j] is buggy, since i will be ignored
                    c[0, 0] = a_tile[i, k] * b_tile[k, j];
                }
            }
        }

        std::cout << "Done!\n";
    }
    catch (std::exception& ex)
    {
        std::cout << "What: " << ex.what();
    }
}
