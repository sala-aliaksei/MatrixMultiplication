#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulFunctions.hpp"
#include "matrixMultiplication/matrix/MatrixMulGpt.hpp"
#include "matrixMultiplication/matrix/MatrixMulOpenBlas.hpp"
#include "matrixMultiplication/matrix/MatrixMulEigen.hpp"

#include <gtest/gtest.h>

// TODO: create matrices with fixed size, decouple from global N

class MatrixMulTest : public testing::Test
{
  protected:
    MatrixMulTest()
    {
        // You can do set-up work for each test here.
        auto matrices = initMatrix();
        matrixMulOpenBlas(matrices);
        valid_res = std::move(matrices.c);
    }

    ~MatrixMulTest() override
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

    Matrix<double> valid_res;
};

TEST_F(MatrixMulTest, MT_VT_BL_TP_STATIC)
{
    auto matrices = initMatrix();

    matrixMul_MT_VT_BL_TP(matrices);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MT_VT_BL)
{
    auto matrices = initMatrix();

    DynamicMatrixMul mul(std::thread::hardware_concurrency(), 8, false, true);
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MT_VT_BL_TP)
{
    auto matrices = initMatrix();

    DynamicMatrixMul mul(std::thread::hardware_concurrency(), 8, true, true);
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Naive_TP)
{
    auto matrices = initMatrix();

    DynamicMatrixMul mul(1, 1, true, false);
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Naive)
{
    auto matrices = initMatrix();

    DynamicMatrixMul mul(1, 1, false, false);
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, GPT)
{
    auto matrices = initMatrix();

    gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Eigen)
{
    auto ms = initEigenMatrix();

    matrixMulEigen(ms);
    for (auto row = 0; row < N; ++row)
    {
        for (auto col = 0; col < N; ++col)
        {
            EXPECT_EQ((valid_res(row, col) == ms.c(row, col)), true);
        }
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
