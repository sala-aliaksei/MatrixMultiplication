
#include "mm/matmul/matMulAutotune.hpp"

#include <gtest/gtest.h>

constexpr std::size_t N = 3072;

constexpr std::size_t I = N;
constexpr std::size_t J = N;
constexpr std::size_t K = N;

static int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : N;
}

class MatrixMulAutotuneTest : public testing::Test
{
  protected:
    MatrixMulAutotuneTest()
      //: matrices(initDoubleMatrix(I, J, K))
      : a(generateRandomMatrix<std::bfloat16_t>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<std::bfloat16_t>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(generateRandomMatrix<std::bfloat16_t>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        // std::cout << "I : " << I << " J: " << J << " K: " << K << "\n";

        mm::matMul_Naive_Order(a, b, c);
        expected = std::move(c);

        c = Matrix<std::bfloat16_t>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv());
    }

    ~MatrixMulAutotuneTest() override
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