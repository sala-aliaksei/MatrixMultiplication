#include "mm/core/Matrix.hpp"
#include "mm/core/reorderMatrix.hpp"

// TODO: Delme
#include "mm/tpi/matMulOpenBlas.hpp"
#include "mm/core/utils/utils.hpp"

#include <gtest/gtest.h>

constexpr int M = 2880;
constexpr int N = 2880;

static void printArr(const double* arr, int row, int col)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            std::cout << arr[i * col + j] << ", ";
        }
        std::cout << std::endl;
    }
}

std::string vecToStr(const std::vector<double>& vec, int cols)
{
    std::ostringstream oss;
    //    oss << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (i > 0 && i % cols == 0)
            oss << '\n';
        oss << vec[i] << " ";
    }
    return oss.str();
}

// **Google Test Fixture**
class ReorderTest : public ::testing::Test
{
  protected:
    template<int Mc, int Nc, int Mr, int Nr>
    void runTest(int Mb, int Nb, const Matrix<double>& input, const Matrix<double>& expected)
    {
        Matrix<double> output(Mc, Nc);
        reorderColOrderPaddingMatrix<Mc, Nc, Mr, Nr>(
          input.data(), input.col(), output.data(), Mb, Nb);

        bool is_succeed = output == expected;
        if (!is_succeed)
            print(input, output, expected);

        EXPECT_TRUE(is_succeed);
    }

    void print(const Matrix<double>& input,
               const Matrix<double>& actual,
               const Matrix<double>& expected)
    {
        std::cout << "----------- input   -------------\n";
        std::cout << input << std::endl;
        std::cout << "----------- actual  result -------------\n";
        std::cout << actual << std::endl;
        std::cout << "------------ expected result ------------\n";
        std::cout << expected << std::endl;
    }
};

// TEST_F(ReorderTest, NoPadding_4x4)
//{
//     constexpr int Mc = 4, Nc = 4, Mr = 4, Nr = 4;
//     const int     Mb = 4, Nb = 4;

//    Matrix<double> input(Mb, Nb);
//    Matrix<double> expected(Mc, Nc);

//    int val = 1;
//    for (int i = 0; i < Mb; ++i)
//        for (int j = 0; j < Nb; ++j)
//        {
//            expected(j, i) = val;
//            input(i, j)    = val;
//            ++val;
//        }

//    runTest<Mc, Nc, Mr, Nr>(Mb, Nb, input, expected);
//}

TEST(ReorderMatrixTest, BasicBlockFit)
{
    constexpr int M  = 4;
    constexpr int N  = 4;
    constexpr int ib = 2;
    constexpr int jb = 2;

    // clang-format off
    double input[M * N] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    // clang-format on

    std::vector<double> output(M * N);

    reorderRowMajorMatrix<M, N, ib, jb>(input, N, output.data());

    // Manually packed expected output:
    // Block (0,0)
    // 1 2
    // 5 6
    // Block (2,0)
    // 9 10
    // 13 14
    // Block (0,2)
    // 3 4
    // 7 8
    // Block (2,2)
    // 11 12
    // 15 16

    std::vector<double> expected = {1, 2, 5, 6, 9, 10, 13, 14, 3, 4, 7, 8, 11, 12, 15, 16};

    ASSERT_EQ(output, expected) << "Expected:\n"
                                << vecToStr(expected, jb) << "\nActual:\n"
                                << vecToStr(output, jb);
}

TEST(ReorderMatrixTest, TailPadding)
{
    constexpr int M  = 5;
    constexpr int N  = 3;
    constexpr int ib = 2;
    constexpr int jb = 2;

    // clang-format off
    double input[M * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    };
    // clang-format on

    // Total number of output elements = ceil(M/ib) * ib * ceil(N/jb) * jb
    constexpr int blocksM = (M + ib - 1) / ib;
    constexpr int blocksN = (N + jb - 1) / jb;
    constexpr int total   = blocksM * ib * blocksN * jb;

    std::vector<double> output(total, -1.0); // fill with junk to detect uninit
    reorderRowMajorMatrixWithPadding<M, N, ib, jb>(input, N, output.data());

    // clang-format off
    std::vector<double> expected = {// Block (0,0)
                                    1,2,
                                    4,5,
                                    // Block (2,0)
                                    7,8,
                                    10,11,
                                    // Block (4,0) (tail row with padding)
                                    13,14,
                                    0,0,
                                    // Block (0,2)
                                    3,0,
                                    6,0,
                                    // Block (2,2)
                                    9,0,
                                    12,0,
                                    // Block (4,2)
                                    15,0,
                                    0,0
    };
    // clang-format on

    ASSERT_EQ(output, expected) << "Expected:\n"
                                << vecToStr(expected, jb) << "\nActual:\n"
                                << vecToStr(output, jb);
}

// 12 ( 4 doubles save to 3 regs)
//

consteval std::array<int, 3> genSeq(int N)
{
    // instead of static assert
    if (N > 12)
        throw std::runtime_error("Tails with N > 12 is not supported");

    int n4 = std::min(N / 4, 3); // Max 3 SIMD registers of width 4
    N %= 4;

    int n2 = N / 2;
    N %= 2;

    int n1 = N; // Remaining elements

    return {n4, n2, n1};
}

static_assert(genSeq(0) == std::array<int, 3>{0, 0, 0}, "genSeq(N=0)  failed");
static_assert(genSeq(1) == std::array<int, 3>{0, 0, 1}, "genSeq(N=1)  failed");
static_assert(genSeq(2) == std::array<int, 3>{0, 1, 0}, "genSeq(N=2)  failed");
static_assert(genSeq(3) == std::array<int, 3>{0, 1, 1}, "genSeq(N=3)  failed");
static_assert(genSeq(4) == std::array<int, 3>{1, 0, 0}, "genSeq(N=4)  failed");
static_assert(genSeq(5) == std::array<int, 3>{1, 0, 1}, "genSeq(N=5)  failed");
static_assert(genSeq(6) == std::array<int, 3>{1, 1, 0}, "genSeq(N=6)  failed");
static_assert(genSeq(7) == std::array<int, 3>{1, 1, 1}, "genSeq(N=7)  failed");
static_assert(genSeq(8) == std::array<int, 3>{2, 0, 0}, "genSeq(N=8)  failed");
static_assert(genSeq(9) == std::array<int, 3>{2, 0, 1}, "genSeq(N=9)  failed");
static_assert(genSeq(10) == std::array<int, 3>{2, 1, 0}, "genSeq(N=10) failed");
static_assert(genSeq(11) == std::array<int, 3>{2, 1, 1}, "genSeq(N=11) failed");
static_assert(genSeq(12) == std::array<int, 3>{3, 0, 0}, "genSeq(N=12) failed");

// template<std::tuple<int, int, int> Tuple>
// struct TupleToIntegerSequence;

// template<int A, int B, int C>
// struct TupleToIntegerSequence<std::tuple<A, B, C>> {
//     using type = std::integer_sequence<int, A, B, C>;
// };

template<std::array<int, 3> Arr>
struct getSeq
{
    using type = std::index_sequence<Arr[0], Arr[1], Arr[2]>;
};

template<std::array<int, 3> Arr>
using getSeq_t = typename getSeq<Arr>::type;

static_assert(std::is_same_v<getSeq_t<genSeq(12)>, std::index_sequence<3, 0, 0>>,
              "genSeq(N=12) failed");

static_assert(std::is_same_v<getSeq_t<genSeq(11)>, std::index_sequence<2, 1, 1>>,
              "genSeq(N=12) failed");

template<int N>
using GetSequence_t = getSeq_t<genSeq(N)>;

static_assert(std::is_same_v<GetSequence_t<11>, std::index_sequence<2, 1, 1>>,
              "GetSequence_t(N=12) failed");

// **Main Function**
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
