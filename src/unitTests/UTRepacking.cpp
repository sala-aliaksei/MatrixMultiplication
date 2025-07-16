#include "mm/core/Matrix.hpp"
#include "mm/core/reorderMatrix.hpp"
#include "mm/core/utils/utils.hpp"
#include <gtest/gtest.h>

// reorderColOrderPaddingMatrix()

constexpr int M = 2880;
constexpr int N = 2880;

// Helper to generate a test matrix (row-major)
std::vector<double> generateMatrix(int rows, int cols)
{
    std::vector<double> m(rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m[i * cols + j] = i * 10 + j; // unique value per element
    return m;
}

// Helper to compare floating point vectors
void assertEqualMatrix(const std::vector<double>& actual, const std::vector<double>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i)
    {
        ASSERT_DOUBLE_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
    }
}

/*****************************************************************************************************/

// **Google Test Fixture**
class ReorderTest : public ::testing::Test
{
  protected:
    template<int Mc, int Nc, int Mr, int Nr>
    void runTest(int Mb, int Nb, const Matrix<double>& input, const Matrix<double>& expected)
    {
        Matrix<double> output(Mc, Nc);
        reorderColOrderPaddingMatrix<Mc, Nc, Mr, Nr>(
          input.data(), input.row(), input.col(), output.data());

        bool is_succeed = output == expected;
        if (!is_succeed)
            print(input, output, expected);

        EXPECT_TRUE(is_succeed);
    }

    void print(const Matrix<double>& input,
               const Matrix<double>& actual,
               const Matrix<double>& expected)
    {
        std::cout << "----------- input   -------------\n";
        std::cout << input << std::endl;
        std::cout << "----------- actual  result -------------\n";
        std::cout << actual << std::endl;
        std::cout << "------------ expected result ------------\n";
        std::cout << expected << std::endl;
    }
};

// **Test Case 1: No Padding Required (4x4)**
TEST_F(ReorderTest, NoPadding_4x4)
{
    constexpr int Mc = 4, Nc = 4, Mr = 4, Nr = 4;
    const int     Mb = 4, Nb = 4;

    Matrix<double> input(Mb, Nb);
    Matrix<double> expected(Mc, Nc);

    int val = 1;
    for (int i = 0; i < Mb; ++i)
        for (int j = 0; j < Nb; ++j)
        {
            expected(j, i) = val;
            input(i, j)    = val;
            ++val;
        }

    runTest<Mc, Nc, Mr, Nr>(Mb, Nb, input, expected);
}

// **Test Case 2: Padding Required (5x6)**
TEST_F(ReorderTest, Padding_5x6)
{
    constexpr int Mc = 8, Nc = 8, Mr = 4, Nr = 4;
    const int     Mb = 5, Nb = 6;

    Matrix<double> input(Mb, Nb);
    Matrix<double> expected(Mc, Nc);

    int val = 1;
    for (int i = 0; i < Mb; ++i)
        for (int j = 0; j < Nb; ++j)
        {
            input(i, j) = val++;
        }

    std::array<int, 8 * 8> ex_res = {
      1,  7,  13, 19, 2,  8,  14, 20,

      3,  9,  15, 21, 4,  10, 16, 22,

      5,  11, 17, 23, 6,  12, 18, 24,

      0,  0,  0,  0,  0,  0,  0,  0,

      25, 0,  0,  0,  26, 0,  0,  0,

      27, 0,  0,  0,  28, 0,  0,  0,

      29, 0,  0,  0,  30, 0,  0,  0,

      0,  0,  0,  0,  0,  0,  0,  0,
    };
    int idx = 0;
    for (int i = 0; i < Mc; i++)
        for (int j = 0; j < Nc; j++)
        {

            expected(i, j) = ex_res[idx++];
        }

    runTest<Mc, Nc, Mr, Nr>(Mb, Nb, input, expected);
}

// **Test Case 3: Single Element (1x1)**
TEST_F(ReorderTest, SingleElement_1x1)
{
    constexpr int Mc = 4, Nc = 4, Mr = 4, Nr = 4;
    const int     Mb = 1, Nb = 1;

    Matrix<double> input(Mb, Nb);
    Matrix<double> expected(Mc, Nc);

    input(0, 0) = expected(0, 0) = 42;

    runTest<Mc, Nc, Mr, Nr>(Mb, Nb, input, expected);
}

// **Test Case 4: Larger Matrix with Padding (7x7)**
TEST_F(ReorderTest, LargerMatrix_7x7)
{
    constexpr int Mc = 8, Nc = 8, Mr = 4, Nr = 4;
    const int     Mb = 7, Nb = 7;

    Matrix<double> input(Mb, Nb);
    Matrix<double> expected(Mc, Nc);

    int val = 1;
    for (int i = 0; i < Mb; ++i)
        for (int j = 0; j < Nb; ++j)
            input(i, j) = val++;

    std::array<int, 8 * 8> ex_res = {

      1,  8,  15, 22, 2,  9,  16, 23, 3,  10, 17, 24, 4,  11, 18, 25, 5,  12, 19, 26, 6,  13,
      20, 27, 7,  14, 21, 28, 0,  0,  0,  0,  29, 36, 43, 0,  30, 37, 44, 0,  31, 38, 45, 0,
      32, 39, 46, 0,  33, 40, 47, 0,  34, 41, 48, 0,  35, 42, 49, 0,  0,  0,  0,  0,
    };
    int idx = 0;
    for (int i = 0; i < Mc; i++)
        for (int j = 0; j < Nc; j++)
        {

            expected(i, j) = ex_res[idx++];
        }

    runTest<Mc, Nc, Mr, Nr>(Mb, Nb, input, expected);
}

// **Test Case 5: Edge Case (3x3)**
TEST_F(ReorderTest, EdgeCase_3x3)
{
    constexpr int Mc = 4, Nc = 4, Mr = 4, Nr = 4;
    const int     Mb = 3, Nb = 3;

    Matrix<double> input(Mb, Nb);
    Matrix<double> expected(Mc, Nc);

    int val = 1;
    for (int i = 0; i < Mb; ++i)
        for (int j = 0; j < Nb; ++j)
            input(i, j) = expected(j, i) = val++;

    // Ensure zero-padding
    for (int i = Mb; i < Mc; ++i)
        for (int j = 0; j < Nc; ++j)
            expected(i, j) = 0.0;

    for (int j = Nb; j < Nc; ++j)
        for (int i = 0; i < Mc; ++i)
            expected(i, j) = 0.0;

    runTest<Mc, Nc, Mr, Nr>(Mb, Nb, input, expected);
}

/*****************************************************************************************************/

// ------------------ TEST CASES ----------------------

TEST(ReorderColOrderMatrixTailTest, NoTails)
{
    constexpr int Ir = 2, Jr = 2;
    const int     Mc = 4, Nc = 4, N = 4;

    auto input = generateMatrix(Mc, N);
    // std::cout << "input matrix" << input << std::endl;

    std::vector<double> output(Mc * Nc, 0);

    reorderColOrderMatrixTail<Ir, Jr>(input.data(), N, output.data(), Mc, Nc);

    // clang-format off
    std::vector<double> expected = {
        0, 10, 1, 11,
        2, 12, 3, 13,
        20, 30, 21, 31,
        22, 32, 23, 33
    };
    // clang-format on

    assertEqualMatrix(output, expected);
}

TEST(ReorderColOrderMatrixTailTest, TailInRows)
{
    constexpr int Ir = 2, Jr = 2;
    const int     Mc = 5, Nc = 4, N = 4;

    auto                input = generateMatrix(Mc, N);
    std::vector<double> output(Mc * Nc, 0);

    reorderColOrderMatrixTail<Ir, Jr>(input.data(), N, output.data(), Mc, Nc);

    // Expected: First 4 rows are processed in blocks,
    // Last row (row 4) is handled as a tail in rows
    // clang-format off
    std::vector<double> expected = {
        0,  10, 1,  11,
        2,  12, 3,  13,
        20, 30, 21, 31,
        22, 32, 23, 33,
        40, 41, 42, 43 // tail row
    };
    // clang-format on

    assertEqualMatrix(output, expected);
}

TEST(ReorderColOrderMatrixTailTest, TailInCols)
{
    constexpr int Ir = 2, Jr = 2;
    const int     Mc = 4, Nc = 5, N = 5;

    auto                input = generateMatrix(Mc, N);
    std::vector<double> output(Mc * Nc, 0);

    reorderColOrderMatrixTail<Ir, Jr>(input.data(), N, output.data(), Mc, Nc);

    // clang-format off
    std::vector<double> expected = {
        0,  10, 1,  11,
        2,  12, 3,  13,
        4,  14,
        20, 30, 21, 31,
        22, 32, 23, 33,
        24, 34,         // tail col
    };
    // clang-format on

    // Flatten expected into row-major shape to match output
    assertEqualMatrix(output, expected);
}

TEST(ReorderColOrderMatrixTailTest, TailInRowsAndCols)
{
    constexpr int Ir = 2, Jr = 2;
    const int     Mc = 5, Nc = 5, N = 5;

    auto input = generateMatrix(Mc, N);

    std::vector<double> output(Mc * Nc, 0);

    reorderColOrderMatrixTail<Ir, Jr>(input.data(), N, output.data(), Mc, Nc);

    // clang-format off
    std::vector<double> expected = {
        0,  10, 1,  11,
        2,  12, 3,  13,
        4,  14,
        20, 30, 21, 31,
        22, 32, 23, 33,
        24, 34,         // tail col
        40, 41, 42, 43, // tail row
        44              // tail col
    };
    // clang-format on

    assertEqualMatrix(output, expected);
}

TEST(ReorderColOrderMatrixTailTest, OnlyITails)
{
    constexpr int Ir = 1, Jr = 2;
    const int     Mc = 2, Nc = 4, N = 4, M = 4;

    auto input = generateMatrix(M, N);
    // std::cout << "input matrix" << input << std::endl;

    std::vector<double> output(Mc * Nc, 0);

    reorderColOrderMatrixTail<Ir, Jr>(input.data(), N, output.data(), Mc, Nc);

    // clang-format off
    std::vector<double> expected = {
        0, 1 ,2, 3, 10, 11, 12, 13
    };
    // clang-format on

    assertEqualMatrix(output, expected);
}

TEST(ReorderColOrderMatrixTailTest, OnlyJTails)
{
    constexpr int Ir = 2, Jr = 1;
    const int     Mc = 4, Nc = 2, N = 2, M = 4;

    auto input = generateMatrix(M, N);
    // std::cout << "input matrix" << input << std::endl;

    std::vector<double> output(Mc * Nc, 0);

    reorderColOrderMatrixTail<Ir, Jr>(input.data(), N, output.data(), Mc, Nc);

    // clang-format off
    std::vector<double> expected = {
        0, 10, 1, 11, 20, 30, 21, 31
    };
    // clang-format on

    assertEqualMatrix(output, expected);
}

TEST(ReorderColOrderMatrixTailTest, ItailHigherMc)
{
    constexpr int Ir = 4, Jr = 4;
    const int     Mc = 4, Nc = 4, N = 32, M = 32;

    constexpr int ITAIL = 4;

    auto input = generateMatrix(M, N);
    // std::cout << "input matrix" << input << std::endl;

    std::vector<double> output(M * N, 0);

    blasReorderRowOrder4x4(M, N, input.data(), N, output.data());
    // reorderColOrderMatrixTail<Ir, Jr>(input.data() + M * N, N, output.data(), ITAIL, Nc);

    // clang-format off
    std::vector<double> expected = {
        0, 10, 1, 11, 20, 30, 21, 31
    };
    // clang-format on
    std::cout << input << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << output << std::endl;
    assertEqualMatrix(output, expected);
}

/*****************************************************************************************************/

// **Main Function**
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
