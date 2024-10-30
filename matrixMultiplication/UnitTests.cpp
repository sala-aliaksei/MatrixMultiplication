#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulGpt.hpp"
#include "matrixMultiplication/matrix/MatrixMulOpenBlas.hpp"
#include "matrixMultiplication/matrix/MatrixMulEigen.hpp"

#include <gtest/gtest.h>

// TODO: add ability to init matrices from cmdline

constexpr std::size_t I = 768;
constexpr std::size_t J = 768;
constexpr std::size_t K = 768;

class MatrixMulTest : public testing::Test
{
  protected:
    MatrixMulTest()
      : matrices(initMatrix(I, J, K))
    {
        // You can do set-up work for each test here.
        matrixMulOpenBlas(matrices);
        valid_res  = std::move(matrices.c);
        matrices.c = Matrix<double>(I, J);
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

    MatrixSet      matrices;
    Matrix<double> valid_res;
};

TEST_F(MatrixMulTest, MT_VT_BL)
{
    DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MT_VT_BL_TP)
{
    DynamicMatrixMul mul(MatrixMulConfig{true, true, true, true});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Naive_TP)
{
    DynamicMatrixMul mul(MatrixMulConfig{false, false, true, false});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Naive)
{
    DynamicMatrixMul mul(MatrixMulConfig{false, false, false, false});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, GPT)
{
    gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Eigen)
{
    auto ms = initEigenMatrix(I, J, K);

    matrixMulEigen(ms);

    auto rows = ms.c.rows();
    auto cols = ms.c.cols();
    for (auto row = 0; row < rows; ++row)
    {
        for (auto col = 0; col < cols; ++col)
        {
            bool res = valid_res(row, col) == ms.c(row, col);
            if (!res)
            {
                std::cout << "row: " << row << ", col: " << col << std::endl;
            }
            ASSERT_EQ(valid_res(row, col), ms.c(row, col));
        }
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
