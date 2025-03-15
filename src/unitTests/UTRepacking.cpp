#include "mm/core/Matrix.hpp"
#include "mm/core/reorderMatrix.hpp"

#include <gtest/gtest.h>

// reorderColOrderPaddingMatrix()

constexpr int M = 2880;
constexpr int N = 2880;

// class ReorderTest : public testing::Test
//{
//   protected:
//     ReorderTest()
//       //: matrices(initMatrix(I, J, K))
//       : valid_res(M, N)
//     {
//         // You can do set-up work for each test here.
//     }

//    ~ReorderTest() override
//    {
//        // You can do clean-up work that doesn't throw exceptions here.
//    }

//    void SetUp() override
//    {
//        // Code here will be called immediately after the constructor (right
//        // before each test).
//    }

//    void TearDown() override
//    {
//        // Code here will be called immediately after each test (right
//        // before the destructor).
//    }

//    Matrix<double> valid_res;
//};

// **Google Test Fixture**
class ReorderTest : public ::testing::Test
{
  protected:
    template<int Mc, int Nc, int Mr, int Nr>
    void runTest(int Mb, int Nb, const Matrix<double>& input, const Matrix<double>& expected)
    {
        Matrix<double> output(Mc, Nc);
        reorderColOrderPaddingMatrix<Mc, Nc, Mr, Nr>(
          input.data(), input.col(), output.data(), Mb, Nb);

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

// **Main Function**
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
