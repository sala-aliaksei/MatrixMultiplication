
#include "mm/matmul/matMulAutotune.hpp"
#include "mm/core/utils/utils.hpp"
#include <gtest/gtest.h>
#include "mm/tpi/matMulOpenBlas.hpp"

constexpr std::size_t N = 3072;

constexpr std::size_t I = N;
constexpr std::size_t J = N;
constexpr std::size_t K = N;

class MatrixMulAutotuneTest : public testing::Test
{
  public:
    MatrixMulAutotuneTest()
      //: matrices(initDoubleMatrix(I, J, K))
      : a(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(GetMatrixDimFromEnv(), GetMatrixDimFromEnv())
      , expected(GetMatrixDimFromEnv(), GetMatrixDimFromEnv())
    {
        // std::cout << "I : " << I << " J: " << J << " K: " << K << "\n";

        mm::tpi::matrixMulOpenBlas(a, b, expected);
    }

    ~MatrixMulAutotuneTest() override = default;

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
    Matrix<double> expected;
};

/***********   FLOAT 32   ***********/
TEST_F(MatrixMulAutotuneTest, MatMulAutotune)
{
    matMulAutotune(a, b, c);

    EXPECT_EQ((expected == c), true);
}

/********************       MAIN        ********************/

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}