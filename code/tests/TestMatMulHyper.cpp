#include "mm/matmul/matMulHyper.hpp"
#include "mm/tpi/matMulOpenBlas.hpp"

#include "mm/core/utils/utils.hpp"

#include <gtest/gtest.h> //--gtest_filter=MatrixMulTest.MatMulLoopsRepack

class MatrixMulHyperTest : public testing::Test
{
  protected:
    MatrixMulHyperTest()
      : a(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(generateIotaMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , valid_res(generateIotaMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        mm::tpi::matrixMulOpenBlas(a, b, valid_res);
    }

    ~MatrixMulHyperTest() override
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

TEST_F(MatrixMulHyperTest, matmul_hyper)
{
    mm::hyper::matMulHyper(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}
