#include "mm/matmul/matMulZen5.hpp"
#include "mm/tpi/matMulOpenBlas.hpp"

#include <gtest/gtest.h> //--gtest_filter=MatrixMulTest.MatMulLoopsRepack

int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : 3072;
}

class MatrixMulZen5Test : public testing::Test
{
  protected:
    MatrixMulZen5Test()
      : a(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(generateRandomMatrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        // std::cout << "I : " << I << " J: " << J << " K: " << K << "\n";

        mm::tpi::matrixMulOpenBlas(a, b, c);
        valid_res = std::move(c);
        c         = Matrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv());
    }

    ~MatrixMulZen5Test() override
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

TEST_F(MatrixMulZen5Test, matMulZen5)
{
    mm::zen5::matMulZen5(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}

TEST_F(MatrixMulZen5Test, matMulZen5MTBlocking)
{
    mm::zen5::matMulZen5MTBlocking(a, b, c);
    EXPECT_EQ((valid_res == c), true);
}

/***********   FLOAT 32   ***********/

class MatrixMulZen5Float32Test : public testing::Test
{
  protected:
    MatrixMulZen5Float32Test()
      //: matrices(initDoubleMatrix(I, J, K))
      : a(generateRandomMatrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , b(generateRandomMatrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
      , c(generateRandomMatrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        // std::cout << "I : " << I << " J: " << J << " K: " << K << "\n";

        mm::tpi::matrixMulOpenBlas(a, b, c);
        expected = std::move(c);
        c        = Matrix<float>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv());
    }

    ~MatrixMulZen5Float32Test() override
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

    Matrix<float> a;
    Matrix<float> b;
    Matrix<float> c;
    Matrix<float> expected;
};

TEST_F(MatrixMulZen5Float32Test, MatMulZen5)
{
    mm::zen5::matMulZen5(a, b, c);

    EXPECT_EQ((expected == c), true);
}

TEST_F(MatrixMulZen5Float32Test, MatMulZen5MTBlocking)
{
    mm::zen5::matMulZen5MTBlocking(a, b, c);
    EXPECT_EQ((expected == c), true);
}

TEST_F(MatrixMulZen5Float32Test, MatMulZen5MTBlockingTails)
{
    mm::zen5::matMulZen5MTBlockingTails(a, b, c);
    EXPECT_EQ((expected == c), true);
}