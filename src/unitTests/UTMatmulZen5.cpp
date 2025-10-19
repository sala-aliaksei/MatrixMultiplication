#include "mm/matmul/matMulZen5.hpp"
#include "mm/tpi/matMulOpenBlas.hpp"
#include "mm/core/layout.hpp"

#include "mm/core/utils/utils.hpp"
#include <omp.h>

#include <gtest/gtest.h> //--gtest_filter=MatrixMulTest.MatMulLoopsRepack
#include <vector>

int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : 3072;
}

class MatrixMulZen5Test : public testing::Test
{
  protected:
    MatrixMulZen5Test()
      : a(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(generateIotaMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , valid_res(generateIotaMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        mm::tpi::matrixMulOpenBlas(a, b, valid_res);
    }

    ~MatrixMulZen5Test() override
    {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    Matrix<double> a;
    Matrix<double> b;
    Matrix<double> c;
    Matrix<double> valid_res;
};

TEST_F(MatrixMulZen5Test, mm)
{
    mm::zen5::matMulZen5(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}

TEST_F(MatrixMulZen5Test, mmblocking)
{
    mm::zen5::matMulZen5MTBlocking(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}

TEST_F(MatrixMulZen5Test, submatrix)
{
    // gtest --gtest_filter=MatrixMulZen5Test.matMulZen5Submatrix
    constexpr std::size_t N  = 512;
    constexpr std::size_t Mc = 128;
    constexpr std::size_t Kc = 128;

    Matrix<double> at = generateRandomMatrix<double>(N, N);

    constexpr std::size_t i0 = 40;
    constexpr std::size_t j0 = 41;

    auto tile = mm::core::submatrix<Mc, Kc>(at, i0, j0);
    for (std::size_t i = 0; i < tile.extent(0); i++)
    {
        for (std::size_t j = 0; j < tile.extent(1); j++)
        {
            EXPECT_EQ((tile[i, j] == at(i0 + i, j0 + j)), true);
        }
    }
}

TEST_F(MatrixMulZen5Test, mdspan)
{
    mm::zen5::matMulZen5MTBlockingSpan(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}

TEST_F(MatrixMulZen5Test, mdspan_l1)
{

    mm::zen5::matMulZen5MTBlockingL1(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}

/***********   FLOAT 32   ***********/

class MatrixMulZen5Float32Test : public testing::Test
{
  protected:
    MatrixMulZen5Float32Test()
      //: matrices(initDoubleMatrix(I, J, K))
      : a(generateRandomMatrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(generateRandomMatrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        // std::cout << "I : " << I << " J: " << J << " K: " << K << "\n";

        mm::tpi::matrixMulOpenBlas(a, b, c);
        expected = std::move(c);
        c        = Matrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv());
    }

    ~MatrixMulZen5Float32Test() override
    {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    void SetUp() override
    {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    void TearDown() override
    {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    Matrix<float> a;
    Matrix<float> b;
    Matrix<float> c;
    Matrix<float> expected;
};

TEST_F(MatrixMulZen5Float32Test, MatMulZen5)
{
    mm::zen5::matMulZen5(a, b, c);

    EXPECT_EQ((expected == c), true);
}

TEST_F(MatrixMulZen5Float32Test, MatMulZen5MTBlocking)
{
    mm::zen5::matMulZen5MTBlocking(a, b, c);
    EXPECT_EQ((expected == c), true);
}

TEST_F(MatrixMulZen5Float32Test, MatMulZen5MTBlockingTails)
{
    mm::zen5::matMulZen5MTBlockingTails(a, b, c);
    EXPECT_EQ((expected == c), true);
}

TEST_F(MatrixMulZen5Test, B_IndexMappingAndContiguity)
{
    using mm::core::layout_blocked_colmajor;

    constexpr std::size_t M  = 8;
    constexpr std::size_t N  = 8;
    constexpr std::size_t Mc = 6;
    constexpr std::size_t Nc = 4;
    constexpr std::size_t Mr = 2;
    constexpr std::size_t Nr = 2;

    static_assert(Mc % Mr == 0, "Mc%  Mr == 0");
    static_assert(Nc % Nr == 0, "Nc%  Nr == 0");

    Matrix<double> at = generateIotaMatrix<double>(M, N);

    constexpr std::size_t iofs = 1;
    constexpr std::size_t jofs = 1;

    using tile_ext_t = std::extents<std::size_t, Nc / Nr, Nr * Mc>;
    layout_blocked_colmajor<Nr>::mapping<tile_ext_t> mapping(tile_ext_t{}, M, N, iofs, jofs);

    constexpr std::size_t required = Mc * Nc;
    ASSERT_EQ(mapping.required_span_size(), required);

    std::mdspan ms(at.data(), mapping);

    auto map2 = ms.mapping();

    std::cout << "-------- ms matrix----------" << std::endl;
    for (std::size_t i = 0; i < ms.extent(0); ++i)
    {
        for (std::size_t j = 0; j < ms.extent(1); ++j)
        {
            std::cout << ms[i, j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------- at matrix----------" << std::endl;
    for (std::size_t i = 0; i < Mc; ++i)
    {
        for (std::size_t j = 0; j < Nc; ++j)
        {
            std::cout << at(i + iofs, j + jofs) << " ";
        }
        std::cout << std::endl;
    }

    int idx = 0;
    for (std::size_t j = 0; j < Nc; j += Nr)
    {
        for (std::size_t i = 0; i < Mc; i += Mr)
        {
            for (std::size_t i2 = 0; i2 < Mr; i2 += 1)
            {
                for (std::size_t j2 = 0; j2 < Nr; ++j2)
                {
                    auto val  = ms[idx / ms.extent(1), idx % ms.extent(1)];
                    auto val2 = at(i + i2, j + j2);

                    std::cout << "idx: " << idx << ",  val[ " << i + i2 << ",  " << j + j2
                              << "] = " << val2 << ", val1 = " << val << std::endl;

                    ASSERT_EQ(val, val2);
                    idx++;
                }
            }
        }
    }
}

TEST_F(MatrixMulZen5Test, A_IndexMappingAndContiguity)
{

    using namespace mm::core;

    constexpr std::size_t M  = 8; // deliberately not multiples of tile
    constexpr std::size_t N  = 8;
    constexpr std::size_t Mc = 6;
    constexpr std::size_t Nc = 4;
    constexpr std::size_t Mr = 2;
    constexpr std::size_t Nr = 2;

    static_assert(Mc % Mr == 0, "Mc%  Mr == 0");
    static_assert(Nc % Nr == 0, "Nc%  Nr == 0");

    Matrix<double> at = generateIotaMatrix<double>(M, N);

    constexpr std::size_t iofs = 0;
    constexpr std::size_t jofs = 0;

    using tile_ext_t = std::extents<std::size_t, Mc / Mr, Mr * Nc>;
    layout_microtile_colorder<Mr, Nr>::mapping mapping(tile_ext_t{}, M, N, iofs, jofs);

    constexpr std::size_t required = Mc * Nc;
    ASSERT_EQ(mapping.required_span_size(), required);

    std::mdspan ms(at.data(), mapping);

    auto map2 = ms.mapping();

    std::cout << "-------- ms matrix----------" << std::endl;
    for (std::size_t i = 0; i < ms.extent(0); ++i)
    {
        for (std::size_t j = 0; j < ms.extent(1); ++j)
        {
            std::cout << ms[i, j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------- at matrix----------" << std::endl;
    for (std::size_t i = 0; i < Mc; ++i)
    {
        for (std::size_t j = 0; j < Nc; ++j)
        {
            std::cout << at(i + iofs, j + jofs) << " ";
        }
        std::cout << std::endl;
    }

    int idx = 0;
    for (std::size_t i = 0; i < Mc; i += Mr)
    {
        for (std::size_t j = 0; j < Nc; j += Nr)
        {
            for (std::size_t j2 = 0; j2 < Nr; ++j2)
            {
                for (std::size_t i2 = 0; i2 < Mr; i2 += 1)
                {

                    auto val  = ms[idx / ms.extent(1), idx % ms.extent(1)];
                    auto val2 = at(i + i2, j + j2);

                    std::cout << "idx: " << idx << ",  val[ " << i + i2 << ",  " << j + j2
                              << "] = " << val2 << ", val1 = " << val << std::endl;

                    ASSERT_EQ(val, val2);
                    idx++;
                }
            }
        }
    }
}

TEST_F(MatrixMulZen5Test, C_IndexMappingAndContiguity)
{

    using namespace mm::core;

    constexpr std::size_t M  = 8; // deliberately not multiples of tile
    constexpr std::size_t N  = 8;
    constexpr std::size_t Mc = 6;
    constexpr std::size_t Nc = 4;
    constexpr std::size_t Mr = 2;
    constexpr std::size_t Nr = 2;

    static_assert(Mc % Mr == 0, "Mc%  Mr == 0");
    static_assert(Nc % Nr == 0, "Nc%  Nr == 0");

    Matrix<double> at = generateIotaMatrix<double>(M, N);

    constexpr std::size_t iofs = 0;
    constexpr std::size_t jofs = 0;

    using tile_ext_t = std::extents<std::size_t, Mc / Mr, Mr * Nc>;
    layout_microtile_colorder<Mr, Nr>::mapping mapping(tile_ext_t{}, M, N, iofs, jofs);

    constexpr std::size_t required = Mc * Nc;
    ASSERT_EQ(mapping.required_span_size(), required);

    std::mdspan ms(at.data(), mapping);

    auto map2 = ms.mapping();

    std::cout << "-------- ms matrix----------" << std::endl;
    for (std::size_t i = 0; i < ms.extent(0); ++i)
    {
        for (std::size_t j = 0; j < ms.extent(1); ++j)
        {
            std::cout << ms[i, j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "-------- at matrix----------" << std::endl;
    for (std::size_t i = 0; i < Mc; ++i)
    {
        for (std::size_t j = 0; j < Nc; ++j)
        {
            std::cout << at(i + iofs, j + jofs) << " ";
        }
        std::cout << std::endl;
    }

    int idx = 0;
    for (std::size_t i = 0; i < Mc; i += Mr)
    {
        for (std::size_t j = 0; j < Nc; j += Nr)
        {
            for (std::size_t j2 = 0; j2 < Nr; ++j2)
            {
                for (std::size_t i2 = 0; i2 < Mr; i2 += 1)
                {

                    auto val  = ms[idx / ms.extent(1), idx % ms.extent(1)];
                    auto val2 = at(i + i2, j + j2);

                    std::cout << "idx: " << idx << ",  val[ " << i + i2 << ",  " << j + j2
                              << "] = " << val2 << ", val1 = " << val << std::endl;

                    ASSERT_EQ(val, val2);
                    idx++;
                }
            }
        }
    }
}