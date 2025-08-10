#include "mm/matmul/matMul.hpp"
#include "mm/matmul/matMulZen5.hpp"

#include <gtest/gtest.h>

#if __STDCPP_FLOAT64_T__ == 1
#include <stdfloat>

constexpr std::size_t N = 3072;

constexpr std::size_t I = N;
constexpr std::size_t J = N;
constexpr std::size_t K = N;

static int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : N;
}

class MatrixMulBFP16Test : public testing::Test
{
  protected:
    MatrixMulBFP16Test()
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

    ~MatrixMulBFP16Test() override
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

    Matrix<std::bfloat16_t> a;
    Matrix<std::bfloat16_t> b;
    Matrix<std::bfloat16_t> c;
    Matrix<std::bfloat16_t> expected;
};

/***********   FLOAT 32   ***********/
TEST_F(MatrixMulBFP16Test, MatMulZen5)
{
    mm::zen5::matMulZen5(a, b, c);

    EXPECT_EQ((expected == c), true);
}

TEST_F(MatrixMulBFP16Test, MatMulZen5MTBlocking)
{
    mm::zen5::matMulZen5MTBlocking(a, b, c);
    EXPECT_EQ((expected == c), true);
}
#endif
/********************       MAIN        ********************/

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}