#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulGpt.hpp"
#include "matrixMultiplication/matrix/MatrixMulOpenBlas.hpp"
#include "matrixMultiplication/matrix/MatrixMulEigen.hpp"

#include <gtest/gtest.h>

// TODO: create matrices with fixed size, decouple from global N
constexpr std::size_t M = 32;
constexpr std::size_t N = 32;

class MatrixMulTest : public testing::Test
{
  protected:
    MatrixMulTest()
      : matrices(initMatrix(N, M))
    {
        // You can do set-up work for each test here.
        matrixMulOpenBlas(matrices);
        valid_res  = std::move(matrices.c);
        matrices.c = Matrix<double>(N, M);
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
    DynamicMatrixMul mul(MatrixMulConfig{std::thread::hardware_concurrency(), 8, false, true});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MT_VT_BL_TP)
{
    DynamicMatrixMul mul(MatrixMulConfig{std::thread::hardware_concurrency(), 8, true, true});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Naive_TP)
{
    DynamicMatrixMul mul(MatrixMulConfig{1, 1, true, false});
    mul(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Naive)
{
    DynamicMatrixMul mul(MatrixMulConfig{1, 1, false, false});
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
    auto ms = initEigenMatrix(N, M);

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
