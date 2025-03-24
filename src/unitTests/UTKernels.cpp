
#include "mm/core/Matrix.hpp"
#include "mm/core/kernels.hpp"

#include <numeric>

#include <gtest/gtest.h>

// #define ENABLE_NAIVE_TESTS

// TODO: add ability to init matrices from cmdline
// constexpr std::size_t N = 12 * 4 * 2;
constexpr std::size_t NN = 4 * 720;

constexpr std::size_t M = NN;
constexpr std::size_t N = NN; // + 8;
constexpr std::size_t K = NN;

constexpr int Kc = 20;

// naive kernels
template<int Mr, int Nr, int Kc>
void matmul_NV(const double* __restrict a,
               const double* __restrict mb,
               double* __restrict c,
               const std::size_t M,
               const std::size_t N,
               const std::size_t K)
{
    //
    const double* b = mb;
    for (int i2 = 0; i2 < Mr; ++i2, c += N, a += K)
    {
        b = mb;
        for (int k2 = 0; k2 < Kc; ++k2, b += N)
        {
            for (int j2 = 0; j2 < Nr; ++j2)
            {
                c[j2] += a[k2] * b[j2];
            }
        }
    }
}

template<int Mr, int Nr, int Kc>
void matmul_NV_Packed(const double* __restrict a,
                      const double* __restrict b,
                      double* __restrict c,
                      const std::size_t M,
                      const std::size_t N)
{

    for (int i2 = 0; i2 < Mr; ++i2, c += N)
    {
        for (int k2 = 0; k2 < Kc; ++k2)
        {
            for (int j2 = 0; j2 < Nr; ++j2)
            {
                c[j2] += a[k2 * Mr + i2] * b[k2 * Nr + j2];
            }
        }
    }
}

class KernelTest : public testing::Test
{
  protected:
    KernelTest()
      : matrices(initMatrix(M, N, K))

    {
        // std::cout << "M : " << M << " N: " << N << " K: " << K << "\n";
        //  You can do set-up work for each test here.
        //  matmul_NV<Mr, Nr, Kc>(matrices.c.data(), matrices.a.data(), matrices.b.data(), M, N, K);
    }

    ~KernelTest() override
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
    MatrixSet matrices;
};

TEST_F(KernelTest, GenericKern12x4)
{
    constexpr int Nr = 12;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern8x4)
{
    constexpr int Nr = 8;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);

    EXPECT_EQ((c == res), true);
}
TEST_F(KernelTest, GenericKern2x4)
{
    constexpr int Nr = 2;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern1x4)
{
    constexpr int Nr = 1;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern12x2)
{
    constexpr int Nr = 12;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern8x2)
{
    constexpr int Nr = 8;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern2x2)
{
    constexpr int Nr = 2;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern1x2)
{
    constexpr int Nr = 1;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern12x1)
{
    constexpr int Nr = 12;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern8x1)
{
    constexpr int Nr = 8;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern2x1)
{
    constexpr int Nr = 2;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, GenericKern1x1)
{
    constexpr int Nr = 1;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr, K);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_generic_ukern<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr, K);
    EXPECT_EQ((c == res), true);
}

///////////////////// PACKED

TEST_F(KernelTest, PackedGenericKern12x4)
{
    constexpr int Nr = 12;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);

    Matrix<double> c(Mr, Nr);
    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern8x4)
{
    constexpr int Nr = 8;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern6x4)
{
    constexpr int Nr = 6;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}
TEST_F(KernelTest, PackedKern6x4)
{
    constexpr int Nr = 6;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::packed_ukernel6x4<Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern4x4)
{
    constexpr int Nr = 4;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern2x4)
{
    constexpr int Nr = 2;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedKern2x4)
{
    constexpr int Nr = 2;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::packed_ukernel2x4<Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedKern1x4)
{
    constexpr int Nr = 1;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::packed_ukernel1x4<Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedKern1x4_simd)
{
    constexpr int Nr = 1;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::packed_ukernel1x4_simd<Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern1x4)
{
    constexpr int Nr = 1;
    constexpr int Mr = 4;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

// MR=2
TEST_F(KernelTest, PackedGenericKern12x2)
{
    constexpr int Nr = 12;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);

    Matrix<double> c(Mr, Nr);
    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern8x2)
{
    constexpr int Nr = 8;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

// TEST_F(KernelTest, PackedGenericKern6x2)
//{
//     constexpr int Nr = 6;
//     constexpr int Mr = 2;

//    Matrix<double> res(Mr, Nr);
//    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
//    Matrix<double> c(Mr, Nr);

//    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

//    EXPECT_EQ((c == res), true);
//}

TEST_F(KernelTest, PackedGenericKern4x2)
{
    constexpr int Nr = 4;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern2x2)
{
    constexpr int Nr = 2;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern1x2)
{
    constexpr int Nr = 1;
    constexpr int Mr = 2;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

// MR=1
TEST_F(KernelTest, PackedGenericKern12x1)
{
    constexpr int Nr = 12;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);

    Matrix<double> c(Mr, Nr);
    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern8x1)
{
    constexpr int Nr = 8;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

// TEST_F(KernelTest, PackedGenericKern6x1)
//{
//     constexpr int Nr = 6;
//     constexpr int Mr = 1;

//    Matrix<double> res(Mr, Nr);
//    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
//    Matrix<double> c(Mr, Nr);

//    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

//    EXPECT_EQ((c == res), true);
//}

TEST_F(KernelTest, PackedGenericKern4x1)
{
    constexpr int Nr = 4;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);

    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern2x1)
{
    constexpr int Nr = 2;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

TEST_F(KernelTest, PackedGenericKern1x1)
{
    constexpr int Nr = 1;
    constexpr int Mr = 1;

    Matrix<double> res(Mr, Nr);
    matmul_NV_Packed<Mr, Nr, Kc>(matrices.a.data(), matrices.b.data(), res.data(), Mr, Nr);
    Matrix<double> c(Mr, Nr);

    kernels::cpp_packed_kernel<Nr, Mr, Kc>(matrices.a.data(), matrices.b.data(), c.data(), Nr);
    EXPECT_EQ((c == res), true);
}

/********************       MAIN        ********************/

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

//    std::cout << "---------   res   ---------\n" << res << std::endl;
//    std::cout << "---------   c   ---------\n" << c << std::endl;
