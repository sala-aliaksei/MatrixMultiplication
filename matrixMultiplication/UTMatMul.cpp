#include "mm/genai/matMulGpt.hpp"
#include "mm/genai/matMulClaude.hpp"

#include "mm/tpi/matMulOpenBlas.hpp"
#include "mm/tpi/matMulEigen.hpp"

#include "mm/matmul/matMul.hpp"
#include "mm/matmul/matMulRegOpt.hpp"
#include "mm/matmul/matMulColOpt.hpp"
#include "mm/matmul/matMulLoops.hpp"
#include "mm/matmul/matMulPadding.hpp"
#include "mm/matmul/matMulAutotune.hpp"
#include "mm/matmul/matMulSimd.hpp"

#include <gtest/gtest.h>

#define CN_MATMUL
#define MATMUL_LOOPS

// TODO: add ability to init matrices from cmdline
// constexpr std::size_t N = 12 * 4 * 2;
constexpr std::size_t N = 4 * 720;
//  constexpr std::size_t N = 4 * 768;
constexpr std::size_t I = N;
constexpr std::size_t J = N;
constexpr std::size_t K = N;

void analyzeResults(const Matrix<double>& actual, const Matrix<double>& expected)
{
    std::cout << "----------- expected  result -------------\n";
    std::cout << expected << std::endl;
    std::cout << "------------ actual result ------------\n";
    std::cout << actual << std::endl;
}

class MatrixMulTest : public testing::Test
{
  protected:
    MatrixMulTest()
      //: matrices(initMatrix(I, J, K))
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

#ifdef MATMUL_LOOPS

TEST_F(MatrixMulTest, MatMulLoops)
{
    matMulLoops(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MatMulLoopsRepack)
{
    matMulLoopsRepack(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MatMulLoopsRepackV2)
{
    matMulLoopsRepackV2(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MatMulLoopsIKJ)
{
    matMulLoopsIKJ(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MatMulLoopsBPacked)
{
    matMulLoopsBPacked(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, MatMulRegOpt)
{
    matMulRegOpt(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}
#endif

TEST_F(MatrixMulTest, GPT)
{
    gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Claude)
{
    multiply_matrices_optimized(matrices.a, matrices.b, matrices.c);

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

#ifdef CN_MATMUL

TEST_F(MatrixMulTest, CN_matMul_Naive_Order)
{
    cppnow::matMul_Naive_Order(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Naive)
{
    cppnow::matMul_Naive(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Naive_Order_KIJ)
{
    cppnow::matMul_Naive_Order_KIJ(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Naive_Block)
{
    cppnow::matMul_Naive_Block(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Simd)
{
    cppnow::matMul_Simd(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx)
{
    cppnow::matMul_Avx(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_AddRegs)
{
    cppnow::matMul_Avx_AddRegs(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_AddRegsV2)
{
    cppnow::matMul_Avx_AddRegsV2(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_AddRegs_Unroll)
{
    cppnow::matMul_Avx_AddRegs_Unroll(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache)
{
    cppnow::matMul_Avx_Cache(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs)
{
    cppnow::matMul_Avx_Cache_Regs(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_UnrollRW)
{
    cppnow::matMul_Avx_Cache_Regs_UnrollRW(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll)
{
    cppnow::matMul_Avx_Cache_Regs_Unroll(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll_BPack)
{
    cppnow::matMul_Avx_Cache_Regs_Unroll_BPack(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll_MT)
{
    cppnow::matMul_Avx_Cache_Regs_Unroll_MT(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll_BPack_MT)
{
    cppnow::matMul_Avx_Cache_Regs_Unroll_BPack_MT(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}
#endif

TEST_F(MatrixMulTest, matMulPadding)
{
    matMulPadding(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, matMulAutotune)
{
    matMulAutotune(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, matMulSimd)
{
    matMulSimd(matrices.a, matrices.b, matrices.c);
    // analyzeResults(matrices.c, valid_res);
    EXPECT_EQ((valid_res == matrices.c), true);
}

/********************       MAIN        ********************/

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// TEST_F(MatrixMulTest, matMulColOpt)
//{
//     matMulColOpt(matrices.a, matrices.b, matrices.c);

//    EXPECT_EQ((valid_res == matrices.c), true);
//}

// TODO: Uncomment and update
// TEST_F(MatrixMulTest, MT_VT_BL_TP)
//{
//    DynamicMatrixMul mul(MatrixMulConfig{true, true, true, true});
//    mul(matrices.a, matrices.b, matrices.c);

//    EXPECT_EQ((valid_res == matrices.c), true);
//}

// TEST_F(MatrixMulTest, Naive_TP)
//{
//     DynamicMatrixMul mul(MatrixMulConfig{false, false, true, false});
//     mul(matrices.a, matrices.b, matrices.c);

//    EXPECT_EQ((valid_res == matrices.c), true);
//}

// TEST_F(MatrixMulTest, Naive)
//{
//     DynamicMatrixMul mul(MatrixMulConfig{false, false, false, false});
//     mul(matrices.a, matrices.b, matrices.c);

//    EXPECT_EQ((valid_res == matrices.c), true);
//}

// TEST_F(MatrixMulTest, MT_VT_BL)
//{
//     DynamicMatrixMul mul(MatrixMulConfig{true, true, false, true});
//     mul(matrices.a, matrices.b, matrices.c);

//    EXPECT_EQ((valid_res == matrices.c), true);
//}
