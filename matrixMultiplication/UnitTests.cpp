#include "matrixMultiplication/matrix/MatrixMul.hpp"
#include "matrixMultiplication/matrix/MatrixMulFunctions.hpp"

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
        valid_res = std::move(matrices.res);
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

TEST_F(MatrixMulTest, MT_VT_BL_TP_INLINE)
{
    auto matrices = initMatrix();

    matrixMul_MT_VT_BL_TP(matrices);
    EXPECT_EQ((valid_res == matrices.res), true);
}

TEST_F(MatrixMulTest, MT_VT_BL_TP)
{
    auto matrices = initMatrix();

    MatrixMul mul(std::thread::hardware_concurrency(), 8, true, true);
    mul(matrices.a, matrices.b, matrices.res);

    EXPECT_EQ((valid_res == matrices.res), true);
}

TEST_F(MatrixMulTest, Naive_TP)
{
    auto matrices = initMatrix();

    MatrixMul mul(1, 1, true, false);
    mul(matrices.a, matrices.b, matrices.res);

    EXPECT_EQ((valid_res == matrices.res), true);
}

TEST_F(MatrixMulTest, Naive)
{
    auto matrices = initMatrix();

    MatrixMul mul(1, 1, false, false);
    mul(matrices.a, matrices.b, matrices.res);

    EXPECT_EQ((valid_res == matrices.res), true);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
