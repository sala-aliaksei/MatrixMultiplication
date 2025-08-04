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
#include "mm/matmul/matMulZen5.hpp"

#include <gtest/gtest.h> //--gtest_filter=MatrixMulTest.MatMulLoopsRepack

// #define ENABLE_NAIVE_TESTS

// TODO: add ability to init matrices from cmdline
// constexpr std::size_t N = 12 * 4 * 2;
constexpr std::size_t N = 4 * 720;

constexpr std::size_t I = N;
constexpr std::size_t J = N;
constexpr std::size_t K = N;

int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : N;
}

template<typename T>
void print_diff(const Matrix<T>& a, const Matrix<T>& b)
{
    constexpr int     width = 6;
    const std::string red   = "\033[31m";
    const std::string reset = "\033[0m";

    for (int i = 0; i < a.row(); ++i)
    {
        for (int j = 0; j < a.col(); ++j)
        {

            if (std::abs(a(i, j) - b(i, j)) > __DBL_EPSILON__)
                std::cout << red << std::setw(width) << a(i, j) << reset << " ";
            else
                std::cout << std::setw(width) << a(i, j) << " ";
        }
        std::cout << "\n";
    }
}

void analyzeResults(const Matrix<double>& actual, const Matrix<double>& expected)
{
    std::cout << "------    Expected     ------ \n";
    print_diff(expected, actual);

    std::cout << "------    Actual     ------ \n";
    print_diff(actual, expected);
}

class MatrixMulTest : public testing::Test
{
  protected:
    MatrixMulTest()
      //: matrices(initMatrix(I, J, K))
      : matrices(initMatrix(GetMatrixDimFromEnv(), GetMatrixDimFromEnv(), GetMatrixDimFromEnv()))
    {
        // std::cout << "I : " << I << " J: " << J << " K: " << K << "\n";

        matrixMulOpenBlas(matrices);
        valid_res  = std::move(matrices.c);
        matrices.c = Matrix<double>(GetMatrixDimFromEnv(), GetMatrixDimFromEnv());
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
#ifdef ENABLE_NAIVE_TESTS

TEST_F(MatrixMulTest, CN_matMul_Naive_Order)
{
    mm::matMul_Naive_Order(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Naive)
{
    mm::matMul_Naive(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Naive_Order_KIJ)
{
    mm::matMul_Naive_Order_KIJ(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Naive_Block)
{
    mm::matMul_Naive_Block(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}
#endif

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

TEST_F(MatrixMulTest, MatMulLoopsRepackIKJ)
{
    matMulLoopsRepackIKJ(matrices.a, matrices.b, matrices.c);

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

TEST_F(MatrixMulTest, GPT)
{
    gpt_matrix_multiply(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Claude)
{
    matMulClaude(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, Eigen)
{
    auto ms = initEigenMatrix(matrices.a, matrices.b);
    auto M  = ms.c.rows();
    auto N  = ms.c.cols();
    auto K  = ms.a.cols();

    // std::cout << "------      a    -----" << std::endl;
    // std::cout << ms.a << std::endl;
    // std::cout << "\n------      b    -----" << std::endl;
    // std::cout << ms.b << std::endl;

    matrixMulEigen(ms);

    // std::cout << "\n------      c    -----" << std::endl;
    // std::cout << ms.c << std::endl; // print all zeroes??

    for (auto i = 0; i < M; ++i)
    {
        for (auto j = 0; j < N; ++j)
        {
            bool is_eq = std::abs(valid_res(i, j) - ms.c(i, j)) < __DBL_EPSILON__;
            if (!is_eq)
            {
                std::cout << "compute elem[" << i << "][" << j << "] " << ms.c(i, j)
                          << " != " << valid_res(i, j) << std::endl;
            }
            ASSERT_EQ(is_eq, true);
        }
    }
}

TEST_F(MatrixMulTest, CN_matMul_Simd)
{
    mm::matMul_Simd(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx)
{
    mm::matMul_Avx(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_AddRegs)
{
    mm::matMul_Avx_AddRegs(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_AddRegs_Unroll)
{
    mm::matMul_Avx_AddRegs_Unroll(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache)
{
    mm::matMul_Avx_Cache(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs)
{
    mm::matMul_Avx_Cache_Regs(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_UnrollRW)
{
    mm::matMul_Avx_Cache_Regs_UnrollRW(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll)
{
    mm::matMul_Avx_Cache_Regs_Unroll(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll_BPack)
{
    mm::matMul_Avx_Cache_Regs_Unroll_BPack(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll_MT)
{
    mm::matMul_Avx_Cache_Regs_Unroll_MT(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Avx_Cache_Regs_Unroll_BPack_MT)
{
    mm::matMul_Avx_Cache_Regs_Unroll_BPack_MT(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Tails)
{
    mm::matMul_Tails(matrices.a, matrices.b, matrices.c);

    EXPECT_EQ((valid_res == matrices.c), true);
}

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
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, matMulZen5)
{
    mm::zen5::matMulZen5(matrices.a, matrices.b, matrices.c);
    EXPECT_EQ((valid_res == matrices.c), true);
}

TEST_F(MatrixMulTest, CN_matMul_Tails_Range)
{
    // TODO: Range???
    for (int i = I; i < I + 1; ++i)
    {
        for (int j = J; j < J + 1; ++j)
        {
            auto matrices = initMatrix(i, j, K);

            std::cout << "Test - I : " << i << " J: " << j << " K: " << K << "\n";

            matrixMulOpenBlas(matrices);
            valid_res  = std::move(matrices.c);
            matrices.c = Matrix<double>(i, j);

            mm::matMul_Tails(matrices.a, matrices.b, matrices.c);

            ASSERT_EQ((valid_res == matrices.c), true);
        }
    }
}

TEST_F(MatrixMulTest, matMulSimd_Tails_Range)
{
    //    int iend = 21;
    //    int jend = 21;

    // TODO: Range???
    constexpr int Mc = 1;
    constexpr int Nc = 1;
    constexpr int Kc = 1;
    //    constexpr int Mc = 20;
    //    constexpr int Nc = 180;
    //    constexpr int Kc = 80;

    constexpr std::size_t I = 720; // Mc;
    constexpr std::size_t J = 720; // Nc;
    constexpr std::size_t K = 720; // Kc;

    int iend = 2;
    int jend = 2;

    for (int j = J; j < J + Nc; ++j) //+ 70
    {
        for (int i = I; i < I + Mc; ++i) //+ 1
        {
            // int k = K;
            for (int k = K; k < K + Kc; ++k)
            {
                auto matrices = initMatrix(i, j, k);

                std::cout << "Test - I : " << i << " J: " << j << " K: " << k << "\n";

                matrixMulOpenBlas(matrices);
                valid_res  = std::move(matrices.c);
                matrices.c = Matrix<double>(i, j);

                matMulSimdTails(matrices.a, matrices.b, matrices.c);
                if (valid_res != matrices.c)
                    analyzeResults(matrices.c, valid_res);
                ASSERT_EQ((valid_res == matrices.c), true);
            }
        }
    }
}

#define TEST_JHANDLE
#ifdef TEST_JHANDLE

TEST_F(MatrixMulTest, HandleJTail)
{
    constexpr int DIM_PRINT = 25;

    constexpr int iofs = 1;  // 7
    constexpr int jofs = 24; // 27
    constexpr int kofs = 2;  // 27

    constexpr int Mc = 20;
    constexpr int Nc = 36;
    constexpr int Kc = 20;

    constexpr int M = Mc + iofs;
    constexpr int N = Nc + jofs;
    constexpr int K = Kc + kofs;

    // constexpr int DIM  = 60;

    // for (int kofs = 0; kofs < 80; ++kofs)
    // for (int jofs = 0; jofs < 168; ++jofs)
    {
        // std::cout << "------   kofs     ------ \n" << kofs << std::endl;

        // can repro, need to min size
        //        constexpr int iofs = 1;  // 7;   // 7 to cover all mrr
        //        constexpr int jofs = 24; // works - 168; // 27;
        //        constexpr int kofs = 2;  // 41;
        //        constexpr int Mc   = 20;
        //        constexpr int Nc   = 180; // Nc % Nr != 0 produce valid result?
        //        constexpr int Kc   = 80;

        //        int M = Mc + iofs;
        //        int N = Nc + jofs;
        //        int K = Kc + kofs;

        constexpr int Nr = 12;
        constexpr int Mr = 4;
        constexpr int Kr = 1;

        static_assert(Mc % Mr == 0, "Invalid M constants");
        static_assert(Nc % Nr == 0, "Invalid N constants");

        //        auto matrices = initPredictedMatrix(M, N, K);
        auto matrices = initMatrix(M, N, K);

        std::cout << "Test - I : " << M << " J: " << N << " K: " << K << "\n";

        matrixMulOpenBlas(matrices);
        valid_res  = std::move(matrices.c);
        matrices.c = Matrix<double>(M, N);

        auto& A = matrices.a;
        auto& B = matrices.b;
        auto& C = matrices.c;

        int j_tail_size = N % Nc;
        int jl          = N - j_tail_size;

        int i_tail_size = M % Mr;
        int il          = M - i_tail_size;

        std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(Kc
                                                                                      * (Mc + Nc));

        double* a_buf = buffer.data();
        double* b_buf = a_buf + Mc * Kc;

        handleJtail<Mr, Kr, Mc, Kc, 12, 8, 4, 2, 1>(
          buffer.data(), A.data(), &B(0, jl), &C(0, jl), M, K, N, j_tail_size);

        // EXPECT_EQ
        // ASSERT_EQ
        for (int j = jl; j < N; j++)
        {
            for (int i = 0; i < M; i++)
            {
                if (matrices.c(i, j) != valid_res(i, j))
                {
                    std::cout << "------    A     ------ \n" << A << std::endl;
                    std::cout << "------    B     ------ \n" << B << std::endl;

                    analyzeResults(matrices.c, valid_res);
                }
                ASSERT_EQ(matrices.c(i, j), valid_res(i, j))
                  << "Elem[" << i << "][" << j << "]"
                  << ". Expected: " << valid_res(i, j) << "\nActual: " << matrices.c(i, j) << "\n";
            }
        }
    }
}
#endif

#define TEST_IHANDLE
#ifdef TEST_IHANDLE

TEST_F(MatrixMulTest, HandleITail)
{
    constexpr int DIM_PRINT = 25;
    constexpr int DIM       = 24;
    constexpr int iofs      = 8;

    constexpr int M = DIM + iofs;
    constexpr int N = DIM;
    constexpr int K = DIM;

    auto matrices = initPredictedMatrix(M, N, K); //
    //    auto matrices = initMatrix(DIM + i_tail_size, DIM, DIM); // initMatrix

    // std::cout << "Test - I : " << i << " J: " << j << " K: " << K << "\n";

    matrixMulOpenBlas(matrices);
    valid_res  = std::move(matrices.c);
    matrices.c = Matrix<double>(M, N);

    auto& A = matrices.a;
    auto& B = matrices.b;
    auto& C = matrices.c;

    constexpr int Mc = 12;
    constexpr int Nc = 4;
    constexpr int Kc = 4;

    constexpr int Mr = 4;
    constexpr int Nr = 4;
    constexpr int Kr = 1;

    std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(Kc * (Mc + Nc));

    double* a_buf = buffer.data();
    double* b_buf = a_buf + Mc * Kc;

    if (DIM < DIM_PRINT)
    {
        std::cout << "------    A     ------ \n" << A << std::endl;
        std::cout << "------    B     ------ \n" << B << std::endl;
    }

    constexpr int ilast       = M - M % Mc;
    constexpr int i_tail_size = M % Mc;
    for (int j_block = 0; j_block < N; j_block += Nc)
    {

        for (int k_block = 0; k_block < K; k_block += Kc)
        {
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(&B(k_block, j_block), N, b_buf);

            if (DIM < DIM_PRINT)
            {
                std::cout << "b\n";
                // printArr(&B(k_block, j_block), Kc, Nc, N);
                std::cout << "b_buf\n";
                // printArr(b_buf, Kc, Nc);
            }

            const double* Ac1 = &A(ilast, k_block);
            double*       Cc1 = &C(ilast, j_block);

            handleItail<Nr, Kr, Nc, Kc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, i_tail_size);
            // TODO: We have one more tail version
            // handleItail<Nr, Kr, Kc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, Nc,
            // i_tail_size);
        }
    }

    if (DIM < DIM_PRINT)
    {
        analyzeResults(matrices.c, valid_res);
    }
    for (int j = 0; j < N; j++)
    {
        for (int i = ilast; i < M; i++)
            ASSERT_EQ(matrices.c(i, j), valid_res(i, j)) << "Expected:\n"
                                                         << valid_res(i, j) << "\nActual:\n"
                                                         << matrices.c(i, j);
    }
}
#endif

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
