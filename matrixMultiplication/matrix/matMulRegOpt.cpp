#include "matMulRegOpt.hpp"
#include "matrixMultiplication/utils/utils.hpp"
#include "reorderMatrix.hpp"

#include <immintrin.h>
#include <array>
#include <future>
#include <cstring>
#include <execution>

#include <iostream>
#include <syncstream>
#include <thread>

// #include <boost/asio.hpp>
// #include <boost/thread/thread.hpp>

constexpr int L1_SIZE_PER_CORE = 32 * 1024;
constexpr int L2_SIZE_PER_CORE = 256 * 1024;

constexpr int AVG_N_FOR_L2_CACHE = 104; // 3 matrices of 104x104 of doublecan fit

// For naive
constexpr int ARYTH_INTENSITY_THRESHOLD = 3;  // flops per byte
constexpr int ARYTH_INTENSITY_N         = 36; // N > 36 should be compute bound

/******************  BLAS    ***************************/

// BUFFSIZE - ?
// range i
// range j
// range k

// sb offset = P*Q*SIZE = 512*256*8 = 1 MB

// GEMM_OFFSET_A = 0
// GEMM_P = 512
// GEMM_Q = 256
// COMPSIZE = 1
// SIZE = 8
// GEMM_ALIGN = 16383
// GEMM_OFFSET_B = 0

/*********************************************/

// For testing purpose
// constexpr auto GEMM_I = 12; // P
// constexpr auto GEMM_J = 24; // Q
// constexpr auto GEMM_K = 16; // R

// L2 cache size = 256KB per core. can fit
// constexpr auto GEMM_I = 120; // P
// constexpr auto GEMM_J = 120; // Q
// constexpr auto GEMM_K = 120;

constexpr auto GEMM_I = 180; // Q
constexpr auto GEMM_J = 48;  // P
constexpr auto GEMM_K = 96;

// constexpr auto GEMM_I = 48; // Q
// constexpr auto GEMM_J = 96; // P
// constexpr auto GEMM_K = 216;

template<int M, int N, int ib, int jb, bool is_col_order>
std::array<double, M * N> reorderMatrix(const double* b, int cols)
{
    /*
    |   ^|
    |  | |
    | |  |
    ||   |
    v    v
    */

    std::array<double, M * N> result;
    if constexpr (is_col_order)
    {
        int idx = 0;

        // DON'T REORDER LOOPS
        // Process columns in groups of 4
        for (size_t i = 0; i < M; i += ib)
        {
            for (size_t j = 0; j < N; j += jb)
            {
                for (size_t jc = 0; jc < jb; ++jc)
                {
                    for (size_t ic = 0; ic < ib; ++ic)
                    {
                        result[idx++] = b[(i + ic) * cols + j + jc];
                    }
                }
            }
        }
    }
    return result;
}

template<int M, int N, int ib, int jb, bool is_col_order>
void reorderMatrix(const double* b, int cols, double* dest)
{
    /*
    |   ^|
    |  | |
    | |  |
    ||   |
    v    v
    */
    // PROFILE("reorderMatrix");

    if constexpr (is_col_order)
    {
        int idx = 0;

        // DON'T REORDER LOOPS
        // Process columns in groups of 4
        for (size_t i = 0; i < M; i += ib)
        {
            for (size_t j = 0; j < N; j += jb)
            {
                for (size_t jc = 0; jc < jb; ++jc)
                {
                    for (size_t ic = 0; ic < ib; ++ic)
                    {
                        dest[idx++] = b[(i + ic) * cols + j + jc];
                    }
                }
            }
        }
    }
}

template<int M, int N, int ib, int jb>
std::array<double, M * N> reorderRowMajorMatrix(const double* b, int cols)
{
    /*
     */
    // PROFILE("reorderMatrix");
    std::array<double, M * N> result;
    // if constexpr (is_col_order)
    {
        int idx = 0;

        // DON'T REORDER LOOPS
        // Process columns in groups of 4
        for (size_t j = 0; j < N; j += jb)
        {
            for (size_t i = 0; i < M; i += ib)
            {
                for (size_t ic = 0; ic < ib; ++ic)
                {
                    const auto col = (i + ic) * cols;
                    _mm_prefetch(&b[(i + ic + 1) * cols], _MM_HINT_NTA);
                    for (size_t jc = 0; jc < jb; ++jc)
                    {
                        result[idx++] = b[col + j + jc];
                    }
                }
            }
        }
    }
    return result;
}

// template<int M, int N, int ib, int jb>
// void reorderRowMajorMatrix(const double* b, int cols, double* dest)
//{
//     /*
//      */

//    // if constexpr (is_col_order)
//    {
//        int idx = 0;

//        // DON'T REORDER LOOPS
//        // Process columns in groups of 4
//        for (size_t j = 0; j < N; j += jb)
//        {
//            for (size_t i = 0; i < M; i += ib)
//            {
//                for (size_t ic = 0; ic < ib; ++ic)
//                {
//                    const auto col = (i + ic) * cols;
//                    _mm_prefetch(&b[(i + ic + 1) * cols], _MM_HINT_NTA);
//                    for (size_t jc = 0; jc < jb; ++jc)
//                    {
//                        dest[idx++] = b[col + j + jc];
//                    }
//                }
//            }
//        }
//    }
//}

// template<int M, int N>
// void printPointer(const double* m)
//{
//     std::unique_lock _glock(_mtx);
//     std::osyncstream(std::cout) << "threadID: " << std::this_thread::get_id() << std::endl;
//     for (auto i = 0; i < M * N; ++i)
//     {
//         std::cout << m[i] << ", ";

//        if ((i + 1) % N == 0)
//            std::cout << "\n";
//    }
//}

// template<int M, int N>
// void printArr(const std::array<double, M * N>& m)
//{
//     std::unique_lock _glock(_mtx);
//     std::osyncstream(std::cout) << "threadID: " << std::this_thread::get_id() << std::endl;
//     for (auto i = 0; i < M * N; ++i)
//     {
//         std::cout << m[i] << ", ";

//        if ((i + 1) % N == 0)
//            std::cout << "\n";
//    }
//}

void testReorderMatrix()
{
    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    constexpr int M = 48;
    constexpr int N = 48;

    constexpr int KernM = 4;
    constexpr int KernN = 8;

    constexpr bool is_col_order = true;
    auto           set          = initPredictedMatrix(M, N, M);
    auto&          a            = set.a;

    auto na = reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(a.data() + 36 * 48, N);

    int idx = 0;

    std::cout << a << std::endl;
    std::cout << "--------- na ---------\n" << std::endl;
    // printArr<GEMM_I, GEMM_K>(na);

    // Process columns in groups of 4
    //    for (size_t i = 0; i < M; i += KernM)
    //    {
    //        for (size_t j = 0; j < N; j += KernN)
    //        {
    //            for (size_t jc = 0; jc < KernN; ++jc)
    //            {
    //                for (size_t ic = 0; ic < KernM; ++ic)
    //                {
    //                    bool r = na[idx++] == a[(i + ic) * a.col() + j + jc];
    //                }
    //            }
    //        }
    //    }
}

inline void transpose4x4_pd(const double* in0,
                            const double* in1,
                            const double* in2,
                            const double* in3,
                            double*       out)
{
    __m256d row0 = _mm256_loadu_pd(in0); // [ in0[0], in0[1], in0[2], in0[3] ]
    __m256d row1 = _mm256_loadu_pd(in1); // [ in1[0], in1[1], in1[2], in1[3] ]
    __m256d row2 = _mm256_loadu_pd(in2);
    __m256d row3 = _mm256_loadu_pd(in3);

    // Interleave low/high pairs
    __m256d t0 = _mm256_unpacklo_pd(row0, row1); // [ in0[0], in1[0], in0[1], in1[1] ]
    __m256d t1 = _mm256_unpackhi_pd(row0, row1); // [ in0[2], in1[2], in0[3], in1[3] ]
    __m256d t2 = _mm256_unpacklo_pd(row2, row3);
    __m256d t3 = _mm256_unpackhi_pd(row2, row3);

    // Merge t0/t2 and t1/t3 to complete the transpose
    // 0x20 = 0010 0000 => take the low halves
    // 0x31 = 0011 0001 => take the high halves
    __m256d o0 = _mm256_permute2f128_pd(t0, t2, 0x20); // col0
    __m256d o1 = _mm256_permute2f128_pd(t1, t3, 0x20); // col1
    __m256d o2 = _mm256_permute2f128_pd(t0, t2, 0x31); // col2
    __m256d o3 = _mm256_permute2f128_pd(t1, t3, 0x31); // col3

    // Now store them in col-major contiguous fashion:
    _mm256_storeu_pd(out + 0, o0);  // Column 0
    _mm256_storeu_pd(out + 4, o1);  // Column 1
    _mm256_storeu_pd(out + 8, o2);  // Column 2
    _mm256_storeu_pd(out + 12, o3); // Column 3
}

template<int M, int N>
std::array<double, M * N> reorderMatrix_4x8_AVX(const double* b, int cols)
{
    std::array<double, M * N> result{};
    double*                   out = result.data();
    int                       idx = 0;

    for (size_t i = 0; i < M; i += 4)
    {
        for (size_t j = 0; j < N; j += 8)
        {

            // Each row pointer (start of block)
            const double* row0 = b + (i + 0) * cols + j;
            const double* row1 = b + (i + 1) * cols + j;
            const double* row2 = b + (i + 2) * cols + j;
            const double* row3 = b + (i + 3) * cols + j;
            _mm_prefetch(row0 + 8, _MM_HINT_NTA);
            _mm_prefetch(row1 + 8, _MM_HINT_NTA);
            _mm_prefetch(row2 + 8, _MM_HINT_NTA);
            _mm_prefetch(row3 + 8, _MM_HINT_NTA);

            // We'll do two 4×4 transposes in succession.
            // Out block: 4×8 => 32 doubles
            double* outBlock = out + idx;
            idx += 32; // we will write 4×8 = 32 doubles

            // 1) Transpose columns [0..3]
            transpose4x4_pd(row0 + 0, row1 + 0, row2 + 0, row3 + 0, outBlock + 0);

            // 2) Transpose columns [4..7]
            transpose4x4_pd(row0 + 4, row1 + 4, row2 + 4, row3 + 4, outBlock + 16);
        }
    }
    return result;
}

template<int M, int N, bool is_col_order>
std::array<double, M * N> reorderMatrix_4x8(const double* b, int cols)
{
    // PROFILE("reorderMatrix_4x8");
    static_assert(is_col_order, "This specialized version only handles is_col_order = true.");

    std::array<double, M * N> result{};
    int                       idx = 0;

    // Outer loops: process 4-row × 8-col blocks
    for (size_t i = 0; i < M; i += 4)
    {

        for (size_t j = 0; j < N; j += 8)
        {
            // Pointers to the start of each row in the sub-block
            const double* row0 = b + (i + 0) * cols + j;
            const double* row1 = b + (i + 1) * cols + j;
            const double* row2 = b + (i + 2) * cols + j;
            const double* row3 = b + (i + 3) * cols + j;
            _mm_prefetch(row0 + 8, _MM_HINT_NTA);
            _mm_prefetch(row1 + 8, _MM_HINT_NTA);
            _mm_prefetch(row2 + 8, _MM_HINT_NTA);
            _mm_prefetch(row3 + 8, _MM_HINT_NTA);

            // Manually unrolled inner loop for jb=8 columns
            // We place columns as the outer dimension in result
            // so we write them in "col-major" sub-block format:
            result[idx + 0] = row0[0];
            result[idx + 1] = row1[0];
            result[idx + 2] = row2[0];
            result[idx + 3] = row3[0];

            result[idx + 4] = row0[1];
            result[idx + 5] = row1[1];
            result[idx + 6] = row2[1];
            result[idx + 7] = row3[1];

            result[idx + 8]  = row0[2];
            result[idx + 9]  = row1[2];
            result[idx + 10] = row2[2];
            result[idx + 11] = row3[2];

            result[idx + 12] = row0[3];
            result[idx + 13] = row1[3];
            result[idx + 14] = row2[3];
            result[idx + 15] = row3[3];

            result[idx + 16] = row0[4];
            result[idx + 17] = row1[4];
            result[idx + 18] = row2[4];
            result[idx + 19] = row3[4];

            result[idx + 20] = row0[5];
            result[idx + 21] = row1[5];
            result[idx + 22] = row2[5];
            result[idx + 23] = row3[5];

            result[idx + 24] = row0[6];
            result[idx + 25] = row1[6];
            result[idx + 26] = row2[6];
            result[idx + 27] = row3[6];

            result[idx + 28] = row0[7];
            result[idx + 29] = row1[7];
            result[idx + 30] = row2[7];
            result[idx + 31] = row3[7];

            // We wrote 4×8 = 32 values in the block
            idx += 32;
        }
    }
    return result;
}

template<int M, int N>
std::array<double, M * N> repackMatrix(const double* b, int cols)
{

    constexpr int             I_BLOCK = 4;
    std::array<double, M * N> result;

    int idx = 0;
    // Process columns in groups of 4
    for (size_t colStart = 0; colStart < N; colStart += I_BLOCK)
    {
        for (size_t row = 0; row < M; ++row)
        {
            for (size_t col = colStart; col < colStart + I_BLOCK; ++col)
            {
                result[idx++] = (b[row * cols + col]);
            }
        }
    }

    return result;
}

template<int M, int N>
std::array<double, M * N> packTMatrix(const double* b, int col)
{
    // TODO: TRY transposed matrix
    int idx = 0;

    std::array<double, M * N> b_packed;

    for (int k = 0; k < M; k++)
    {
        for (int j = 0; j < N; j++)
        {
            b_packed[j * M + k] = b[k * col + j];
        }
    }
    return b_packed;
}

__attribute__((always_inline)) static inline void load_inc_store_double(double* __restrict ptr,
                                                                        __m256d increment)
{
    // Load 4 double-precision values (256 bits) from memory into an AVX register
    __m256d vector = _mm256_load_pd(ptr);

    // Add the increment to the loaded vector
    __m256d result = _mm256_add_pd(vector, increment);

    // Store the result back to memory
    _mm256_store_pd(ptr, result);
    //_mm256_stream_pd(ptr, result);
}

static void massert(bool flag, std::string msg)
{
    using namespace std::literals;
    if (!flag)
    {
        throw std::runtime_error("Invalid expression: "s + msg);
    }
}

void mulMatrix_y(double*           pc,
                 const double*     na,
                 const double*     nb,
                 const std::size_t j_size,
                 const std::size_t k_size)
{
    // TODO: Try to utilize more regs relying on physical regs
    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
    static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
    static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

    const auto pb = packMatrix<GEMM_K, GEMM_J>(nb, j_size);

    const auto a_cols = k_size;
    const auto b_cols = GEMM_J;

    //        constexpr auto prefetch_type = _MM_HINT_T0;
    //        _mm_prefetch(&na[(i + 1) * k_size], prefetch_type);
    //        _mm_prefetch(&na[(i + 2) * k_size], prefetch_type);
    //        _mm_prefetch(&na[(i + 3) * k_size], prefetch_type);
    //        _mm_prefetch(&na[(i + 4) * k_size], prefetch_type);

    for (int i = 0; i < GEMM_I; i += I_BLOCK)
    {
        for (int j = 0; j < GEMM_J; j += J_BLOCK)
        {
            __m256d r00 = _mm256_setzero_pd();
            __m256d r01 = _mm256_setzero_pd();
            __m256d r02 = _mm256_setzero_pd();

            __m256d r10 = _mm256_setzero_pd();
            __m256d r11 = _mm256_setzero_pd();
            __m256d r12 = _mm256_setzero_pd();

            __m256d r20 = _mm256_setzero_pd();
            __m256d r21 = _mm256_setzero_pd();
            __m256d r22 = _mm256_setzero_pd();

            __m256d r30 = _mm256_setzero_pd();
            __m256d r31 = _mm256_setzero_pd();
            __m256d r32 = _mm256_setzero_pd();

            for (int k = 0; k < GEMM_K; k += K_BLOCK)
            {

                const double* b = &pb[k * b_cols + j];

                const double* ma = &na[i * a_cols + k];
                const double* a  = ma;

                for (int k2 = 0; k2 < K_BLOCK; ++k2, b += b_cols)
                {
                    a = ma;

                    __m256d b0 = _mm256_loadu_pd(&b[0]);
                    __m256d b1 = _mm256_loadu_pd(&b[4]);
                    __m256d b2 = _mm256_loadu_pd(&b[8]);

                    __m256d a0 = _mm256_broadcast_sd(&a[k2]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a += a_cols;
                    a0 = _mm256_broadcast_sd(&a[k2]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a += a_cols;
                    a0 = _mm256_broadcast_sd(&a[k2]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a += a_cols;
                    a0 = _mm256_broadcast_sd(&a[k2]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);
                }
            }

            double* c = &pc[i * j_size + j];

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r00);
            load_inc_store_double(&c[4], r01);
            load_inc_store_double(&c[8], r02);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r10);
            load_inc_store_double(&c[4], r11);
            load_inc_store_double(&c[8], r12);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r20);
            load_inc_store_double(&c[4], r21);
            load_inc_store_double(&c[8], r22);
            c += j_size;

            load_inc_store_double(&c[0], r30);
            load_inc_store_double(&c[4], r31);
            load_inc_store_double(&c[8], r32);
        }
    }
}

void mulMatrix_yy(double*           pc,
                  const double*     na,
                  const double*     nb,
                  const std::size_t j_size,
                  const std::size_t k_size)
{
    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
    static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
    static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

    // no impact? try to tune cache size
    // TODO: Must be moved to upper layer since no deps from J
    const auto pa = packMatrix<GEMM_I, GEMM_K>(na, k_size);

    const auto pb = packMatrix<GEMM_K, GEMM_J>(nb, j_size);

    const auto    a_cols = GEMM_K;
    const double* ma     = &pa[0];

    for (size_t i = 0; i < GEMM_I; i += I_BLOCK, ma += I_BLOCK * a_cols)
    {
        for (size_t j = 0; j < GEMM_J; j += J_BLOCK)
        {
            __m256d r00 = _mm256_setzero_pd();
            __m256d r01 = _mm256_setzero_pd();
            __m256d r02 = _mm256_setzero_pd();

            __m256d r10 = _mm256_setzero_pd();
            __m256d r11 = _mm256_setzero_pd();
            __m256d r12 = _mm256_setzero_pd();

            __m256d r20 = _mm256_setzero_pd();
            __m256d r21 = _mm256_setzero_pd();
            __m256d r22 = _mm256_setzero_pd();

            __m256d r30 = _mm256_setzero_pd();
            __m256d r31 = _mm256_setzero_pd();
            __m256d r32 = _mm256_setzero_pd();

            for (size_t k = 0; k < GEMM_K; k += K_BLOCK)
            {
                //                const auto    b_cols = j_size;
                //                const double* b      = &nb[k * b_cols + j];

                const auto    b_cols = GEMM_J;
                const double* b      = &pb[k * b_cols + j];

                //-------------------------------------

                const double* a = &ma[k];
                //_mm_prefetch(&na[(i + 1) * a_cols + k], _MM_HINT_NTA);

                //_mm_prefetch(a + 8, _MM_HINT_NTA);
                for (int k2 = 0; k2 < K_BLOCK; ++k2, b += b_cols)
                {
                    //_mm_prefetch(b + 8, _MM_HINT_NTA);
                    a = &ma[k];

                    __m256d b0 = _mm256_loadu_pd(&b[0]);
                    __m256d b1 = _mm256_loadu_pd(&b[4]);
                    __m256d b2 = _mm256_loadu_pd(&b[8]);

                    __m256d a0 = _mm256_broadcast_sd(&a[k2]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a += a_cols;
                    a0 = _mm256_broadcast_sd(&a[k2]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a += a_cols;
                    a0 = _mm256_broadcast_sd(&a[k2]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a += a_cols;
                    a0 = _mm256_broadcast_sd(&a[k2]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);
                }
            }

            double* c = &pc[i * j_size + j];

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r00);
            load_inc_store_double(&c[4], r01);
            load_inc_store_double(&c[8], r02);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r10);
            load_inc_store_double(&c[4], r11);
            load_inc_store_double(&c[8], r12);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r20);
            load_inc_store_double(&c[4], r21);
            load_inc_store_double(&c[8], r22);
            c += j_size;

            load_inc_store_double(&c[0], r30);
            load_inc_store_double(&c[4], r31);
            load_inc_store_double(&c[8], r32);
        }
    }
}

void mulMatrix_z(double*           pc,
                 const double*     na,
                 const double*     nb,
                 const std::size_t j_size,
                 const std::size_t k_size)
{
    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
    static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
    static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

    //    for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
    //    {

    // const auto pb = packMatrix<GEMM_K, GEMM_J>(&nb[k3 * j_size], j_size);
    // Must be moved  to upper layer
    const auto pb = packMatrix<GEMM_K, GEMM_J>(nb, j_size);

    // TODO: Repack A in col order,
    // const auto pa = reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(&na[k3], k_size);
    const auto pa = reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(na, k_size);
    // const auto pa = reorderMatrix_4x8<GEMM_I, GEMM_K, true>(&na[k3], k_size);
    //  const auto pa = reorderMatrix_4x8_AVX<GEMM_I, GEMM_K>(&na[k3], k_size);

    // printArr<GEMM_I, GEMM_K>(pa);

    constexpr auto a_cols = GEMM_K;
    const double*  ma     = pa.data();

    // MUL KERNEL

    for (size_t i = 0; i < GEMM_I; i += I_BLOCK, ma += I_BLOCK * GEMM_K)
    {
        //            std::cout << "a kernel \n";
        //            printPointer<I_BLOCK, GEMM_K>(ma);
        //                    _mm_prefetch(ma, _MM_HINT_T0);
        //                    _mm_prefetch(ma + 8, _MM_HINT_T0);
        //                    _mm_prefetch(ma + 16, _MM_HINT_T0);
        //                    _mm_prefetch(ma + 24, _MM_HINT_T0);
        for (size_t j = 0; j < GEMM_J; j += J_BLOCK)
        {
            __m256d r00 = _mm256_setzero_pd();
            __m256d r01 = _mm256_setzero_pd();
            __m256d r02 = _mm256_setzero_pd();

            __m256d r10 = _mm256_setzero_pd();
            __m256d r11 = _mm256_setzero_pd();
            __m256d r12 = _mm256_setzero_pd();

            __m256d r20 = _mm256_setzero_pd();
            __m256d r21 = _mm256_setzero_pd();
            __m256d r22 = _mm256_setzero_pd();

            __m256d r30 = _mm256_setzero_pd();
            __m256d r31 = _mm256_setzero_pd();
            __m256d r32 = _mm256_setzero_pd();

            const double* a = &ma[0];

            for (size_t k = 0; k < GEMM_K; k += K_BLOCK)
            {
                //                    std::cout << "--------------- i=" << i << "j=" << j << "
                //                    k=" << k
                //                              << "----------------\n";
                const auto    b_cols = GEMM_J;
                const double* b      = &pb[k * b_cols + j];

                for (int k2 = 0; k2 < K_BLOCK; k2 += 2, b += b_cols, a += 2 * I_BLOCK)
                {
                    //                        _mm_prefetch(a + 8, _MM_HINT_NTA);
                    // no impact

                    // TODO: If a read with stride 1, we can load  whole a to reg and use perm
                    // to retrieve proper elem
                    //                        std::cout << "a{ " << a[0] << "," << a[1] << ","
                    //                        << a[2] << "," << a[3]
                    //                                  << "}\n";

                    __m256d b0 = _mm256_loadu_pd(&b[0]);
                    __m256d b1 = _mm256_loadu_pd(&b[4]);
                    __m256d b2 = _mm256_loadu_pd(&b[8]);

                    __m256d a0 = _mm256_broadcast_sd(&a[0]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a0 = _mm256_broadcast_sd(&a[1]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a0 = _mm256_broadcast_sd(&a[2]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a0 = _mm256_broadcast_sd(&a[3]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);

                    b += b_cols;

                    // iter with prefetech
                    b0 = _mm256_loadu_pd(&b[0]);
                    b1 = _mm256_loadu_pd(&b[4]);
                    b2 = _mm256_loadu_pd(&b[8]);

                    a0 = _mm256_broadcast_sd(&a[4]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a0 = _mm256_broadcast_sd(&a[5]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a0 = _mm256_broadcast_sd(&a[6]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a0 = _mm256_broadcast_sd(&a[7]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);
                }
            }

            double* c = &pc[i * j_size + j];

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r00);
            load_inc_store_double(&c[4], r01);
            load_inc_store_double(&c[8], r02);

            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r10);
            load_inc_store_double(&c[4], r11);
            load_inc_store_double(&c[8], r12);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r20);
            load_inc_store_double(&c[4], r21);
            load_inc_store_double(&c[8], r22);
            c += j_size;

            load_inc_store_double(&c[0], r30);
            load_inc_store_double(&c[4], r31);
            load_inc_store_double(&c[8], r32);
        }
    }
    //}
}

void mulMatrix_zIJK(double*           pc,
                    const double*     na,
                    const double*     nb,
                    const std::size_t j_size,
                    const std::size_t k_size)
{
    // new order of loop, new repack algorithm should be used!!!
    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
    static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
    static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

    //    for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
    //    {

    // const auto pb = packMatrix<GEMM_K, GEMM_J>(&nb[k3 * j_size], j_size);
    // Must be moved to upper layer
    const auto pb = packMatrix<GEMM_K, GEMM_J>(nb, j_size);

    // TODO: Repack A in col order,
    // const auto pa = reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(&na[k3], k_size);
    const auto pa = reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(na, k_size);
    // const auto pa = reorderMatrix_4x8<GEMM_I, GEMM_K, true>(&na[k3], k_size);
    //  const auto pa = reorderMatrix_4x8_AVX<GEMM_I, GEMM_K>(&na[k3], k_size);

    // printArr<GEMM_I, GEMM_K>(pa);

    constexpr auto a_cols = GEMM_K;
    const double*  ma     = pa.data();

    // MUL KERNEL

    for (size_t j = 0; j < GEMM_J; j += J_BLOCK)
    {
        __m256d r00 = _mm256_setzero_pd();
        __m256d r01 = _mm256_setzero_pd();
        __m256d r02 = _mm256_setzero_pd();

        __m256d r10 = _mm256_setzero_pd();
        __m256d r11 = _mm256_setzero_pd();
        __m256d r12 = _mm256_setzero_pd();

        __m256d r20 = _mm256_setzero_pd();
        __m256d r21 = _mm256_setzero_pd();
        __m256d r22 = _mm256_setzero_pd();

        __m256d r30 = _mm256_setzero_pd();
        __m256d r31 = _mm256_setzero_pd();
        __m256d r32 = _mm256_setzero_pd();

        const double* a = &ma[0];

        for (size_t k = 0; k < GEMM_K; k += K_BLOCK)
        {
            //                    std::cout << "--------------- i=" << i << "j=" << j << "
            //                    k=" << k
            //                              << "----------------\n";
            const auto    b_cols = GEMM_J;
            const double* b      = &pb[k * b_cols + j];
            for (size_t i = 0; i < GEMM_I; i += I_BLOCK, ma += I_BLOCK * GEMM_K)
            {
                for (int k2 = 0; k2 < K_BLOCK; k2 += 2, b += b_cols, a += 2 * I_BLOCK)
                {
                    //                        _mm_prefetch(a + 8, _MM_HINT_NTA);
                    // no impact

                    // TODO: If a read with stride 1, we can load  whole a to reg and use perm
                    // to retrieve proper elem
                    //                        std::cout << "a{ " << a[0] << "," << a[1] << ","
                    //                        << a[2] << "," << a[3]
                    //                                  << "}\n";

                    __m256d b0 = _mm256_loadu_pd(&b[0]);
                    __m256d b1 = _mm256_loadu_pd(&b[4]);
                    __m256d b2 = _mm256_loadu_pd(&b[8]);

                    __m256d a0 = _mm256_broadcast_sd(&a[0]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a0 = _mm256_broadcast_sd(&a[1]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a0 = _mm256_broadcast_sd(&a[2]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a0 = _mm256_broadcast_sd(&a[3]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);

                    b += b_cols;

                    // iter with prefetech
                    b0 = _mm256_loadu_pd(&b[0]);
                    b1 = _mm256_loadu_pd(&b[4]);
                    b2 = _mm256_loadu_pd(&b[8]);

                    a0 = _mm256_broadcast_sd(&a[4]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a0 = _mm256_broadcast_sd(&a[5]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a0 = _mm256_broadcast_sd(&a[6]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a0 = _mm256_broadcast_sd(&a[7]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);
                }

                double* c = &pc[i * j_size + j];

                //_mm_prefetch(c + j_size, _MM_HINT_NTA);

                load_inc_store_double(&c[0], r00);
                load_inc_store_double(&c[4], r01);
                load_inc_store_double(&c[8], r02);

                c += j_size;

                // _mm_prefetch(c + j_size, _MM_HINT_NTA);

                load_inc_store_double(&c[0], r10);
                load_inc_store_double(&c[4], r11);
                load_inc_store_double(&c[8], r12);
                c += j_size;

                //_mm_prefetch(c + j_size, _MM_HINT_NTA);

                load_inc_store_double(&c[0], r20);
                load_inc_store_double(&c[4], r21);
                load_inc_store_double(&c[8], r22);
                c += j_size;

                load_inc_store_double(&c[0], r30);
                load_inc_store_double(&c[4], r31);
                load_inc_store_double(&c[8], r32);
            }
        }
    }
    //}
}

void mulMatrix_zz(double*           pc,
                  const double*     na,
                  const double*     nb,
                  const std::size_t j_size,
                  const std::size_t k_size,
                  double*           buffer)
{
    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
    static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
    static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

    // must be moved
    reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(na, k_size, buffer);
    const double* ma = buffer;

    reorderRowMajorMatrix<GEMM_K, GEMM_J, K_BLOCK, J_BLOCK>(nb, j_size, buffer + GEMM_I * GEMM_K);
    const double* mb = (buffer + GEMM_I * GEMM_K);

    // const auto pa = reorderMatrix_4x8<GEMM_I, GEMM_K, true>(&na[k3], k_size);
    //  const auto pa = reorderMatrix_4x8_AVX<GEMM_I, GEMM_K>(&na[k3], k_size);

    // MUL KERNEL

    //        _mm_prefetch(ma, _MM_HINT_NTA);
    //        _mm_prefetch(ma + 8, _MM_HINT_NTA);
    //        _mm_prefetch(ma + 16, _MM_HINT_NTA);
    //        _mm_prefetch(ma + 24, _MM_HINT_NTA);

    for (size_t i = 0; i < GEMM_I; i += I_BLOCK, ma += I_BLOCK * GEMM_K)
    {
        //            std::cout << "a kernel \n";
        //            printPointer<I_BLOCK, GEMM_K>(ma);

        const double* b = mb;
        for (size_t j = 0; j < GEMM_J; j += J_BLOCK) // mb += J_BLOCK * GEMM_K
        {
            __m256d r00 = _mm256_setzero_pd();
            __m256d r01 = _mm256_setzero_pd();
            __m256d r02 = _mm256_setzero_pd();

            __m256d r10 = _mm256_setzero_pd();
            __m256d r11 = _mm256_setzero_pd();
            __m256d r12 = _mm256_setzero_pd();

            __m256d r20 = _mm256_setzero_pd();
            __m256d r21 = _mm256_setzero_pd();
            __m256d r22 = _mm256_setzero_pd();

            __m256d r30 = _mm256_setzero_pd();
            __m256d r31 = _mm256_setzero_pd();
            __m256d r32 = _mm256_setzero_pd();

            const double* a = ma;
            for (size_t k = 0; k < GEMM_K; k += K_BLOCK)
            {

                //                _mm_prefetch(b + 8, _MM_HINT_NTA);
                //                _mm_prefetch(b + 16, _MM_HINT_NTA);
                //                _mm_prefetch(b + 24, _MM_HINT_NTA);
                for (int k2 = 0; k2 < K_BLOCK; k2 += 2, b += 2 * J_BLOCK, a += 2 * I_BLOCK)
                {
                    __m256d b0 = _mm256_loadu_pd(&b[0]);
                    __m256d b1 = _mm256_loadu_pd(&b[4]);
                    __m256d b2 = _mm256_loadu_pd(&b[8]);

                    __m256d a0 = _mm256_broadcast_sd(&a[0]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a0 = _mm256_broadcast_sd(&a[1]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a0 = _mm256_broadcast_sd(&a[2]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a0 = _mm256_broadcast_sd(&a[3]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);

                    // iter with prefetech
                    b0 = _mm256_loadu_pd(&b[12]);
                    b1 = _mm256_loadu_pd(&b[16]);
                    b2 = _mm256_loadu_pd(&b[20]);

                    a0 = _mm256_broadcast_sd(&a[4]);

                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                    a0 = _mm256_broadcast_sd(&a[5]);

                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                    a0 = _mm256_broadcast_sd(&a[6]);

                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                    a0 = _mm256_broadcast_sd(&a[7]);

                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                    r32 = _mm256_fmadd_pd(a0, b2, r32);
                }
            }

            double* c = &pc[i * j_size + j];

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r00);
            load_inc_store_double(&c[4], r01);
            load_inc_store_double(&c[8], r02);

            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r10);
            load_inc_store_double(&c[4], r11);
            load_inc_store_double(&c[8], r12);
            c += j_size;

            _mm_prefetch(c + j_size, _MM_HINT_NTA);

            load_inc_store_double(&c[0], r20);
            load_inc_store_double(&c[4], r21);
            load_inc_store_double(&c[8], r22);
            c += j_size;

            load_inc_store_double(&c[0], r30);
            load_inc_store_double(&c[4], r31);
            load_inc_store_double(&c[8], r32);
        }
    }
}

//_mm_prefetch(&na[(i + 1) * a_cols + k], _MM_HINT_NTA);
//_mm_prefetch(a + 8, _MM_HINT_NTA);
//_mm_prefetch(b + 8, _MM_HINT_NTA);

void matMulRegOpt(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    //    matMulRegOptBuff(A, B, C);
    //    return;
    const std::size_t num_threads = std::thread::hardware_concurrency();

    const auto i_size = A.row();
    const auto j_size = B.col();
    const auto k_size = A.col();

    const size_t block_inc = i_size / num_threads;

    // TODO: Will be replaced with tail computation
    massert(i_size % num_threads == 0, "i_size % num_threads == 0");
    massert(block_inc % GEMM_I == 0, "block_inc % GEMM_I == 0");
    massert(j_size % GEMM_J == 0, "j_size % GEMM_J == 0");
    massert(k_size % GEMM_K == 0, "k_size % GEMM_K == 0");

    double*       mc = C.data();
    const double* mb = B.data();
    const double* ma = A.data();

    //    auto task = [&](const std::size_t tid) -> void
    //    {
    //        std::size_t start = tid * block_inc;
    //        std::size_t last  = tid == (num_threads - 1) ? i_size : (tid + 1) * block_inc;

    //        for (size_t i3 = start; i3 < last; i3 += GEMM_I)
    //        {
    //            for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
    //            {
    //                for (size_t j3 = 0; j3 < j_size; j3 += GEMM_J)
    //                {

    //                    mulMatrix_y(&mc[i3 * j_size + j3],
    //                                &ma[i3 * k_size + k3],
    //                                &mb[k3 * j_size + j3],
    //                                j_size,
    //                                k_size);
    //                }
    //            }
    //        }
    //    };

    //    std::vector<std::thread> thread_pool;
    //    thread_pool.reserve(num_threads);
    //    thread_pool.emplace_back(task, 0);
    //    thread_pool.emplace_back(task, 1);
    //    thread_pool.emplace_back(task, 2);
    //    task(3);

    //    for (auto& t : thread_pool)
    //    {
    //        t.join();
    //    }

    //    task(0);

#pragma omp parallel for
    for (size_t i3 = 0; i3 < i_size; i3 += GEMM_I)
    {
        for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
        {
            for (size_t j3 = 0; j3 < j_size; j3 += GEMM_J)
            {

                mulMatrix_y(&mc[i3 * j_size + j3],
                            &ma[i3 * k_size + k3],
                            &mb[k3 * j_size + j3],
                            j_size,
                            k_size);
            }
        }
    }
}

void matMulRegOptBuff(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    const std::size_t num_threads = std::thread::hardware_concurrency();

    const auto i_size = A.row();
    const auto j_size = B.col();
    const auto k_size = A.col();

    const size_t block_inc = i_size / num_threads;

    // TODO: Will be replaced with tail computation
    massert(i_size % num_threads == 0, "i_size % num_threads == 0");
    massert(block_inc % GEMM_I == 0, "block_inc % GEMM_I == 0");
    massert(j_size % GEMM_J == 0, "j_size % GEMM_J == 0");
    massert(k_size % GEMM_K == 0, "k_size % GEMM_K == 0");

    double*       mc = C.data();
    const double* mb = B.data();
    const double* ma = A.data();

    std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(
      num_threads * GEMM_K * (GEMM_I + GEMM_J));

    auto task = [&](const std::size_t tid) -> void
    {
        std::size_t start   = tid * block_inc;
        std::size_t last    = tid == (num_threads - 1) ? i_size : (tid + 1) * block_inc;
        const auto  buf_ofs = tid * GEMM_K * (GEMM_I + GEMM_J);

        for (size_t i3 = start; i3 < last; i3 += GEMM_I)
        {

            for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
            {
                for (size_t j3 = 0; j3 < j_size; j3 += GEMM_J)
                {

                    //                    mulMatrix_y(
                    //                      &mc[i3 * j_size + j3], &ma[i3 * k_size + 0], &mb[0 +
                    //                      j3], j_size, k_size);

                    //                mulMatrix_yy(
                    //                  &mc[i3 * j_size + j3], &ma[i3 * k_size + 0], &mb[0 + j3],
                    //                  j_size, k_size);

                    mulMatrix_zz(&mc[i3 * j_size + j3],
                                 &ma[i3 * k_size + k3],
                                 &mb[k3 * j_size + j3],
                                 j_size,
                                 k_size,
                                 &buffer[buf_ofs]);
                }

                // mulMatrix_yAdvARepack(mc, ma, mb, j_size, k_size);
            }
        }
    };

    std::vector<std::thread> thread_pool;
    thread_pool.reserve(num_threads);
    thread_pool.emplace_back(task, 0);
    thread_pool.emplace_back(task, 1);
    thread_pool.emplace_back(task, 2);
    task(3);
    //    task(0);

    for (auto& t : thread_pool)
    {
        t.join();
    }
}

void matMulRegOptNoL2(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // need vectorize writing to c
    // impl diff kernels
    // add tail computation for diff kernels
    // a and b must be reordered
    // std::array should be reused and have same addess al the time to be in hot path/cache
    const std::size_t num_threads = std::thread::hardware_concurrency();

    const auto i_size = A.row();
    const auto j_size = B.col();
    const auto k_size = A.col();

    const size_t block_inc = i_size / num_threads;

    // TODO: Will be replaced with tail computation
    massert(i_size % num_threads == 0, "i_size % num_threads == 0");
    massert(block_inc % GEMM_I == 0, "block_inc % GEMM_I == 0");
    massert(j_size % GEMM_J == 0, "j_size % GEMM_J == 0");
    massert(k_size % GEMM_K == 0, "k_size % GEMM_K == 0");

    double*       mc = C.data();
    const double* mb = B.data();
    const double* ma = A.data();

    auto task = [&](const std::size_t tid) -> void
    {
        std::size_t start = tid * block_inc;
        std::size_t last  = tid == (num_threads - 1) ? i_size : (tid + 1) * block_inc;

        for (size_t i3 = start; i3 < last; i3 += GEMM_I)
        {
            const size_t i3_end = std::min(i3 + GEMM_I, i_size);
            for (size_t j3 = 0; j3 < j_size; j3 += GEMM_J)
            {
                const size_t j3_end = std::min(j3 + GEMM_J, j_size);
                //                mulMatrix_y(
                //                  &mc[i3 * j_size + j3], &ma[i3 * k_size + 0], &mb[0 + j3],
                //                  j_size, k_size);

                //                mulMatrix_yy(
                //                  &mc[i3 * j_size + j3], &ma[i3 * k_size + 0], &mb[0 + j3],
                //                  j_size, k_size);

                // FIXED. DON'T CHANGE
                constexpr size_t I_BLOCK = 4;
                constexpr size_t J_BLOCK = 12;
                // AUTOTUNE;
                constexpr size_t K_BLOCK = 8; // 48; // 32

                static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
                static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
                static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

                for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
                {

                    const auto pb = packMatrix<GEMM_K, GEMM_J>(&mb[k3 * j_size + j3], j_size);
                    // std::cout << "--------------- k=" << k3 << "----------------\n";

                    // TODO: Repack A in col order,
                    const auto pa = reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(
                      &ma[i3 * k_size + k3], k_size);
                    // const auto pa = reorderMatrix_4x8<GEMM_I, GEMM_K, true>(&na[k3], k_size);
                    //  const auto pa = reorderMatrix_4x8_AVX<GEMM_I, GEMM_K>(&na[k3], k_size);

                    // printArr<GEMM_I, GEMM_K>(pa);

                    constexpr auto a_cols = GEMM_K;
                    const double*  ma     = pa.data();

                    // MUL KERNEL

                    for (size_t i = 0; i < GEMM_I; i += I_BLOCK, ma += I_BLOCK * GEMM_K)
                    {
                        for (size_t j = 0; j < GEMM_J; j += J_BLOCK)
                        {
                            __m256d r00 = _mm256_setzero_pd();
                            __m256d r01 = _mm256_setzero_pd();
                            __m256d r02 = _mm256_setzero_pd();

                            __m256d r10 = _mm256_setzero_pd();
                            __m256d r11 = _mm256_setzero_pd();
                            __m256d r12 = _mm256_setzero_pd();

                            __m256d r20 = _mm256_setzero_pd();
                            __m256d r21 = _mm256_setzero_pd();
                            __m256d r22 = _mm256_setzero_pd();

                            __m256d r30 = _mm256_setzero_pd();
                            __m256d r31 = _mm256_setzero_pd();
                            __m256d r32 = _mm256_setzero_pd();

                            const double* a = &ma[0];

                            for (size_t k = 0; k < GEMM_K; k += K_BLOCK)
                            {
                                const auto    b_cols = GEMM_J;
                                const double* b      = &pb[k * b_cols + j];

                                for (int k2 = 0; k2 < K_BLOCK;
                                     k2 += 2, b += b_cols, a += 2 * I_BLOCK)
                                {
                                    __m256d b0 = _mm256_loadu_pd(&b[0]);
                                    __m256d b1 = _mm256_loadu_pd(&b[4]);
                                    __m256d b2 = _mm256_loadu_pd(&b[8]);

                                    __m256d a0 = _mm256_broadcast_sd(&a[0]);

                                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                                    a0 = _mm256_broadcast_sd(&a[1]);

                                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                                    a0 = _mm256_broadcast_sd(&a[2]);

                                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                                    a0 = _mm256_broadcast_sd(&a[3]);

                                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                                    r32 = _mm256_fmadd_pd(a0, b2, r32);

                                    b += b_cols;

                                    // iter with prefetech
                                    b0 = _mm256_loadu_pd(&b[0]);
                                    b1 = _mm256_loadu_pd(&b[4]);
                                    b2 = _mm256_loadu_pd(&b[8]);

                                    a0 = _mm256_broadcast_sd(&a[4]);

                                    r00 = _mm256_fmadd_pd(a0, b0, r00);
                                    r01 = _mm256_fmadd_pd(a0, b1, r01);
                                    r02 = _mm256_fmadd_pd(a0, b2, r02);

                                    a0 = _mm256_broadcast_sd(&a[5]);

                                    r10 = _mm256_fmadd_pd(a0, b0, r10);
                                    r11 = _mm256_fmadd_pd(a0, b1, r11);
                                    r12 = _mm256_fmadd_pd(a0, b2, r12);

                                    a0 = _mm256_broadcast_sd(&a[6]);

                                    r20 = _mm256_fmadd_pd(a0, b0, r20);
                                    r21 = _mm256_fmadd_pd(a0, b1, r21);
                                    r22 = _mm256_fmadd_pd(a0, b2, r22);

                                    a0 = _mm256_broadcast_sd(&a[7]);

                                    r30 = _mm256_fmadd_pd(a0, b0, r30);
                                    r31 = _mm256_fmadd_pd(a0, b1, r31);
                                    r32 = _mm256_fmadd_pd(a0, b2, r32);
                                }
                            }

                            double* c = &mc[(i3 + i) * j_size + j3 + j];

                            _mm_prefetch(c + j_size, _MM_HINT_NTA);

                            load_inc_store_double(&c[0], r00);
                            load_inc_store_double(&c[4], r01);
                            load_inc_store_double(&c[8], r02);

                            c += j_size;

                            _mm_prefetch(c + j_size, _MM_HINT_NTA);

                            load_inc_store_double(&c[0], r10);
                            load_inc_store_double(&c[4], r11);
                            load_inc_store_double(&c[8], r12);
                            c += j_size;

                            _mm_prefetch(c + j_size, _MM_HINT_NTA);

                            load_inc_store_double(&c[0], r20);
                            load_inc_store_double(&c[4], r21);
                            load_inc_store_double(&c[8], r22);
                            c += j_size;

                            load_inc_store_double(&c[0], r30);
                            load_inc_store_double(&c[4], r31);
                            load_inc_store_double(&c[8], r32);
                        }
                    }
                }
            }
        }
    };

    std::vector<std::thread> thread_pool;
    thread_pool.reserve(num_threads);
    thread_pool.emplace_back(task, 0);
    thread_pool.emplace_back(task, 1);
    thread_pool.emplace_back(task, 2);
    task(3);

    for (auto& t : thread_pool)
    {
        t.join();
    }
}

// BOOST THREAD POOL

//    boost::asio::io_service       io_service;
//    boost::asio::io_service::work work(io_service);

//    boost::thread_group thread_pool;
//    for (int i = 0; i < num_threads - 1; ++i)
//    {
//        thread_pool.create_thread(boost::bind(&boost::asio::io_service::run,
//        &io_service));
//    }

//    for (int i = 0; i < num_threads - 1; ++i)
//    { // Posting more tasks to utilize all cores effectively
//        io_service.post([&]() { task(i); });
//    }

//    task(3);
//    io_service.stop();
//    thread_pool.join_all();

//                    for (int k2 = 0; k2 < K_BLOCK; ++k2, b += b_cols, a +=
//                    I_BLOCK)
//                    {
//                        // no impact
//                        //_mm_prefetch(b + 8, _MM_HINT_NTA);

//                        // TODO: If a read with stride 1, we can load  whole a
//                        to reg and use perm
//                        // to retrieve proper elem
//                        //                        std::cout << "a{ " << a[0]
//                        << "," << a[1] << ","
//                        //                        << a[2] << "," << a[3]
//                        //                                  << "}\n";

//                        __m256d b0 = _mm256_loadu_pd(&b[0]);
//                        __m256d b1 = _mm256_loadu_pd(&b[4]);
//                        __m256d b2 = _mm256_loadu_pd(&b[8]);

//                        __m256d a0 = _mm256_broadcast_sd(&a[0]);

//                        r00 = _mm256_fmadd_pd(a0, b0, r00);
//                        r01 = _mm256_fmadd_pd(a0, b1, r01);
//                        r02 = _mm256_fmadd_pd(a0, b2, r02);

//                        a0 = _mm256_broadcast_sd(&a[1]);

//                        r10 = _mm256_fmadd_pd(a0, b0, r10);
//                        r11 = _mm256_fmadd_pd(a0, b1, r11);
//                        r12 = _mm256_fmadd_pd(a0, b2, r12);

//                        a0 = _mm256_broadcast_sd(&a[2]);

//                        r20 = _mm256_fmadd_pd(a0, b0, r20);
//                        r21 = _mm256_fmadd_pd(a0, b1, r21);
//                        r22 = _mm256_fmadd_pd(a0, b2, r22);

//                        a0 = _mm256_broadcast_sd(&a[3]);

//                        r30 = _mm256_fmadd_pd(a0, b0, r30);
//                        r31 = _mm256_fmadd_pd(a0, b1, r31);
//                        r32 = _mm256_fmadd_pd(a0, b2, r32);
//                    }
