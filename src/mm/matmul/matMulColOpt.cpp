#include "mm/matmul/matMulColOpt.hpp"
#include "mm/core/utils/utils.hpp"

#include <immintrin.h>

#include <array>
#include <thread>
#include <cstring> // std::memcpy
#include <cmath>   // std::fma

constexpr auto GEMM_I = 120; // Q
constexpr auto GEMM_J = 120; // P
constexpr auto GEMM_K = 120;

template<int M, int N, int ib, int jb, bool is_col_order>
std::array<double, M * N> reorderMatrix(const double* b, int cols)
{
    // PROFILE("reorderMatrix");
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

static void massert(bool flag, std::string msg)
{
    using namespace std::literals;
    if (!flag)
    {
        throw std::runtime_error("Invalid expression: "s + msg);
    }
}

template<int I, int J>
std::array<double, I * J> packMatrix(const double* b, int j_size)
{
    // BAD Implementation
    //    std::array<double, I * J> b_packed;
    //    for (int i = 0; i < I; i++)
    //    {
    //        for (int j = 0; j < J; j++)
    //        {
    //            b_packed[i * J + j] = b[i * j_size + j];
    //        }
    //    }

    constexpr int istep{I / 2};
    constexpr int jstep{J / 2};
    static_assert(I % 2 == 0, "");
    static_assert(J % 2 == 0, "");

    std::array<double, I * J> b_packed;
    for (int i = 0; i < I; i += istep)
    {

        for (int j = 0; j < J; j += jstep)
        {
            for (int i2 = 0; i2 < istep; i2++)
            {
                //_mm_prefetch(&b[(i + i2) * j_size + j], _MM_HINT_NTA);
                std::memcpy(&b_packed[(i + i2) * J] + j, &b[(i + i2) * j_size + j], jstep * 8);
            }
        }
    }
    // if (k2 + 1 < BLOCK_SIZE)

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

void matMulColOptNaive(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    constexpr auto GEMM_Q = 32; // Q
    constexpr auto GEMM_P = 48; // P
    constexpr auto GEMM_R = 64;
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
    massert(block_inc % GEMM_Q == 0, "block_inc % GEMM_I == 0");
    massert(j_size % GEMM_P == 0, "j_size % GEMM_J == 0");
    massert(k_size % GEMM_R == 0, "k_size % GEMM_K == 0");

    double*       nc = C.data();
    const double* nb = A.data();
    const double* na = B.data();

    auto task = [&](const std::size_t tid) -> void
    {
        std::size_t start = tid * block_inc;
        std::size_t last  = tid == (num_threads - 1) ? i_size : (tid + 1) * block_inc;

        for (size_t i3 = start; i3 < last; i3 += GEMM_Q)
        {
            for (size_t k3 = 0; k3 < k_size; k3 += GEMM_R)
            {
                for (size_t j3 = 0; j3 < i_size; j3 += GEMM_P)
                {
                    for (int i2 = 0; i2 < GEMM_Q; ++i2)
                    {
                        for (int k2 = 0; k2 < GEMM_R; ++k2)
                        {
                            for (int j2 = 0; j2 < GEMM_P; ++j2)
                            {
                                nc[(j3 + j2) + (i3 + i2) * j_size] =
                                  std::fma(na[(j3 + j2) + (k3 + k2) * j_size],
                                           nb[(k3 + k2) + (i3 + i2) * k_size],
                                           nc[(j3 + j2) + (i3 + i2) * j_size]);
                            }
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

void matMulColOpt(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    //    matMulColOptNaive(A, B, C);
    //    return;

    // need vectorize writing to c
    // impl diff kernels
    // add tail computation for diff kernels
    // a and b must be reordered
    // std::array should be reused and have same addess al the time to be in hot path/cache
    const std::size_t num_threads = std::thread::hardware_concurrency();

    const auto i_size = A.row();
    const auto j_size = B.col();
    const auto k_size = A.col();

    const size_t block_inc = j_size / num_threads;

    // TODO: Will be replaced with tail computation
    massert(j_size % num_threads == 0, "i_size % num_threads == 0");
    massert(block_inc % GEMM_I == 0, "block_inc % GEMM_I == 0");
    massert(j_size % GEMM_J == 0, "j_size % GEMM_J == 0");
    massert(k_size % GEMM_K == 0, "k_size % GEMM_K == 0");

    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(GEMM_I % I_BLOCK == 0, "invalid GEMM_I or I_BLOCK");
    static_assert(GEMM_J % J_BLOCK == 0, "invalid GEMM_J or J_BLOCK");
    static_assert(GEMM_K % K_BLOCK == 0, "invalid GEMM_K or K_BLOCK");

    double*       nc = C.data();
    const double* nb = A.data();
    const double* na = B.data();

    auto task = [&](const std::size_t tid) -> void
    {
        std::size_t start = tid * block_inc;
        std::size_t last  = tid == (num_threads - 1) ? i_size : (tid + 1) * block_inc;

        for (size_t i3 = start; i3 < last; i3 += GEMM_I)
        {
            for (size_t k3 = 0; k3 < k_size; k3 += GEMM_K)
            {
                for (size_t j3 = 0; j3 < i_size; j3 += GEMM_J)
                {
                    for (int i2 = 0; i2 < GEMM_I; ++i2)
                    {
                        for (int k2 = 0; k2 < GEMM_K; ++k2)
                        {
                            for (int j2 = 0; j2 < GEMM_J; ++j2)
                            {
                                const auto pa =
                                  reorderMatrix<GEMM_I, GEMM_K, I_BLOCK, K_BLOCK, true>(
                                    &na[j3 + k3 * j_size], i_size);

                                // TODO: reorder as well
                                const auto pb =
                                  packMatrix<GEMM_K, GEMM_J>(&nb[k3 + i3 * k_size], k_size);

                                const double* ma = pa.data();

                                // MUL KERNEL

                                for (size_t i = 0; i < GEMM_I;
                                     i += I_BLOCK, ma++) //= I_BLOCK * GEMM_K
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
                                            const double* b = &pb[k + i * GEMM_K];

                                            // 4 * 12 * 8
                                            // a[j][k] * b[k][i] = c[j][i]
                                            for (int k2 = 0; k2 < K_BLOCK;
                                                 k2 += 2, ++b, a += 2 * I_BLOCK)
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

                                                ++b;

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

                                        double* c = &nc[(i3 + i) + (j3 + j) * i_size];

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
