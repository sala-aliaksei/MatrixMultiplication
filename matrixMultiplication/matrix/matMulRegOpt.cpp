#include "matMulRegOpt.hpp"

#include <immintrin.h>
#include <array>
#include <future>
#include <cstring>
// #include <boost/asio.hpp>
// #include <boost/thread/thread.hpp>

// L3 cache for b block; 192 is higher than 1 MB cache
constexpr auto CACHE_BLOCK = 120; // 96; // 96(best when packed only b); 192; // 128;

template<int M, int N>
std::array<double, M * N> packTMatrix(const double* b, int col)
{
    // TODO: TRY
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

template<int M, int N>
std::array<double, M * N> packMatrix(const double* b, int col)
{
    constexpr int BLOCK_SIZE = CACHE_BLOCK / 2; // 48;
    static_assert(M % BLOCK_SIZE == 0, "Invalid M%BLOCK_SIZE==0");
    static_assert(N % BLOCK_SIZE == 0, "Invalid N%BLOCK_SIZE==0");

    std::array<double, M * N> b_packed;
    for (int k = 0; k < M; k += BLOCK_SIZE)
    {
        for (int j = 0; j < N; j += BLOCK_SIZE)
        {
            // _mm_prefetch(&b[(k + BLOCK_SIZE) * col + j], _MM_HINT_NTA);
            for (int k2 = 0; k2 < BLOCK_SIZE; k2++)
            {
                // if (k2 + 1 < BLOCK_SIZE)
                _mm_prefetch(&b[(k + k2 + 1) * col + j], _MM_HINT_NTA);
                std::memcpy(&b_packed[(k + k2) * N + j], &b[(k + k2) * col + j], BLOCK_SIZE * 8);
            }
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

void massert(bool flag, std::string msg)
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
                 const std::size_t k_size,
                 const std::size_t i3,
                 const std::size_t i3_end,
                 const std::size_t j3,
                 const std::size_t j3_end,
                 const std::size_t k3,
                 const std::size_t k3_end)
{

    // FIXED. DON'T CHANGE
    constexpr size_t I_BLOCK = 4;
    constexpr size_t J_BLOCK = 12;
    // AUTOTUNE;
    constexpr size_t K_BLOCK = 8; // 48; // 32

    static_assert(CACHE_BLOCK % I_BLOCK == 0, "invalid CACHE_BLOCK or I_BLOCK");
    static_assert(CACHE_BLOCK % J_BLOCK == 0, "invalid CACHE_BLOCK or J_BLOCK");
    static_assert(CACHE_BLOCK % K_BLOCK == 0, "invalid CACHE_BLOCK or K_BLOCK");

    size_t i_max = i3_end - i3_end % I_BLOCK;
    size_t j_max = j3_end - j3_end % J_BLOCK;
    size_t k_max = k3_end - k3_end % K_BLOCK;

    // no impact? try to tune cache size
    auto pa = packMatrix<CACHE_BLOCK, CACHE_BLOCK>(&na[i3 * k_size + k3], k_size);

    // must have [CACHE_BLOCK x CACHE_BLOCK] dim
    const auto pb = packMatrix<CACHE_BLOCK, CACHE_BLOCK>(&nb[k3 * j_size + j3], j_size);

    for (size_t i = i3; i < i_max; i += I_BLOCK)
    {
        for (size_t j = j3; j < j_max; j += J_BLOCK)
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

            for (size_t k = k3; k < k_max; k += K_BLOCK)
            {
                //                const auto    b_cols = j_size;
                //                const double* b      = &nb[k * b_cols + j];

                const auto    b_cols = CACHE_BLOCK;
                const double* b      = &pb[(k - k3) * b_cols + (j - j3)];

                //-------------------------------------

                const auto    a_cols = CACHE_BLOCK;
                const double* ma     = &pa[(i - i3) * a_cols + (k - k3)];

                //                const auto    a_cols = k_size;
                //                const double* ma     = &na[i * a_cols + k];

                const double* a = ma;
                //_mm_prefetch(&na[(i + 1) * a_cols + k], _MM_HINT_NTA);
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

    // tail
    //    std::cout << "i(" << i_max << ", " << i3_end << ")" << std::endl;
    //    std::cout << "j(" << j_max << ", " << j3_end << ")" << std::endl;
    //    std::cout << "k(" << k_max << ", " << k3_end << ")" << std::endl;
    //    for (int i = i_max; i < i3_end; ++i)
    //    {
    //        for (int k = k3; k < k3_end; ++k)
    //        {
    //            for (int j = j3; j < j3_end; ++j)
    //            {
    //                pc[i * j_size + j] += na[i * k_size + k] * nb[k * j_size + j];
    //            }
    //        }
    //    }

    //    for (int i = i_max; i < i3_end; ++i)
    //    {
    //        for (int k = k_max; k < k3_end; ++k)
    //        {
    //            for (int j = j_max; j < j3_end; ++j)
    //            {
    //                pc[i * j_size + j] += na[i * k_size + k] * nb[k * j_size + j];
    //            }
    //        }
    //    }
}

void matMulRegOpt(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    const std::size_t num_threads = std::thread::hardware_concurrency();

    const auto i_size = A.row();
    const auto j_size = B.col();
    const auto k_size = A.col();

    // const size_t block_inc = CACHE_BLOCK * num_threads;
    const size_t block_inc = i_size / num_threads;

    massert(i_size % num_threads == 0, "i_size % num_threads == 0");
    massert(block_inc % CACHE_BLOCK == 0, "block_inc % CACHE_BLOCK == 0");
    massert(j_size % CACHE_BLOCK == 0, "j_size % CACHE_BLOCK == 0");
    massert(k_size % CACHE_BLOCK == 0, "k_size % CACHE_BLOCK == 0");

    double*       mc = C.data();
    const double* mb = B.data();
    const double* ma = A.data();

    auto task = [&](const std::size_t tid) -> void
    {
        std::size_t start = tid * block_inc;
        std::size_t last  = tid == (num_threads - 1) ? i_size : (tid + 1) * block_inc;

        for (size_t i3 = start; i3 < last; i3 += CACHE_BLOCK)
        {
            for (size_t k3 = 0; k3 < k_size; k3 += CACHE_BLOCK)
            {
                for (size_t j3 = 0; j3 < j_size; j3 += CACHE_BLOCK)
                {
                    const size_t i3_end = std::min(i3 + CACHE_BLOCK, i_size);
                    const size_t j3_end = std::min(j3 + CACHE_BLOCK, j_size);
                    const size_t k3_end = std::min(k3 + CACHE_BLOCK, k_size);

                    mulMatrix_y(mc, ma, mb, j_size, k_size, i3, i3_end, j3, j3_end, k3, k3_end);
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
}

//#define INNER_KERNEL_k1m8n24 \
//    INNER_KERNEL_k1m4n24\
//    "vbroadcastsd 32(%0),%%zmm4;"\
//    "vfmadd231pd %%zmm5,%%zmm4,%%zmm20;"\
//    "vfmadd231pd %%zmm6,%%zmm4,%%zmm21;"\
//    "vfmadd231pd %%zmm7,%%zmm4,%%zmm22;"\
//    "vbroadcastsd 40(%0),%%zmm4;"\
//    "vfmadd231pd %%zmm5,%%zmm4,%%zmm23;"\
//    "vfmadd231pd %%zmm6,%%zmm4,%%zmm24;"\
//    "vfmadd231pd %%zmm7,%%zmm4,%%zmm25;"\
//    "vbroadcastsd 48(%0),%%zmm4;"\
//    "vfmadd231pd %%zmm5,%%zmm4,%%zmm26;"\
//    "vfmadd231pd %%zmm6,%%zmm4,%%zmm27;"\
//    "vfmadd231pd %%zmm7,%%zmm4,%%zmm28;"\
//    "vbroadcastsd 56(%0),%%zmm4;"\
//    "vfmadd231pd %%zmm5,%%zmm4,%%zmm29;"\
//    "vfmadd231pd %%zmm6,%%zmm4,%%zmm30;"\
//    "vfmadd231pd %%zmm7,%%zmm4,%%zmm31;"

// void mulMatrix_y2(double*           c,
//                   const double*     ma,
//                   const double*     mb,
//                   const std::size_t j_size,
//                   const std::size_t k_size)
//{

//    const auto packed_a = packMatrix<I_BLOCK, K_BLOCK>(ma, k_size);
//    const auto packed_b = packMatrix<K_BLOCK, J_BLOCK>(mb, j_size);

//    const double* a = packed_a.data();
//    const double* b = packed_b.data();
//    // tile dimension (in elems): 3:12:8

//    __m256d r00 = _mm256_setzero_pd();
//    __m256d r01 = _mm256_setzero_pd();
//    __m256d r02 = _mm256_setzero_pd();
//    __m256d r10 = _mm256_setzero_pd();
//    __m256d r11 = _mm256_setzero_pd();
//    __m256d r12 = _mm256_setzero_pd();
//    __m256d r20 = _mm256_setzero_pd();
//    __m256d r21 = _mm256_setzero_pd();
//    __m256d r22 = _mm256_setzero_pd();

//    __m256d r30 = _mm256_setzero_pd();
//    __m256d r31 = _mm256_setzero_pd();
//    __m256d r32 = _mm256_setzero_pd();

//    for (int k2 = 0; k2 < K_BLOCK; ++k2, b += J_BLOCK)
//    {
//        a = packed_a.data();

//        __m256d b0 = _mm256_loadu_pd(&b[0]);
//        __m256d b1 = _mm256_loadu_pd(&b[4]);
//        __m256d b2 = _mm256_loadu_pd(&b[8]);

//        __m256d a0 = _mm256_broadcast_sd(&a[k2]);

//        r00 = _mm256_fmadd_pd(a0, b0, r00);
//        r01 = _mm256_fmadd_pd(a0, b1, r01);
//        r02 = _mm256_fmadd_pd(a0, b2, r02);

//        a += K_BLOCK;
//        a0 = _mm256_broadcast_sd(&a[k2]);

//        r10 = _mm256_fmadd_pd(a0, b0, r10);
//        r11 = _mm256_fmadd_pd(a0, b1, r11);
//        r12 = _mm256_fmadd_pd(a0, b2, r12);

//        a += K_BLOCK;
//        a0 = _mm256_broadcast_sd(&a[k2]);

//        r20 = _mm256_fmadd_pd(a0, b0, r20);
//        r21 = _mm256_fmadd_pd(a0, b1, r21);
//        r22 = _mm256_fmadd_pd(a0, b2, r22);

//        a += K_BLOCK;
//        a0 = _mm256_broadcast_sd(&a[k2]);

//        r30 = _mm256_fmadd_pd(a0, b0, r30);
//        r31 = _mm256_fmadd_pd(a0, b1, r31);
//        r32 = _mm256_fmadd_pd(a0, b2, r32);
//    }

//    load_inc_store_double(&c[0], r00);
//    load_inc_store_double(&c[4], r01);
//    load_inc_store_double(&c[8], r02);
//    c += j_size;
//    load_inc_store_double(&c[0], r10);
//    load_inc_store_double(&c[4], r11);
//    load_inc_store_double(&c[8], r12);
//    c += j_size;
//    load_inc_store_double(&c[0], r20);
//    load_inc_store_double(&c[4], r21);
//    load_inc_store_double(&c[8], r22);
//    c += j_size;
//    load_inc_store_double(&c[0], r30);
//    load_inc_store_double(&c[4], r31);
//    load_inc_store_double(&c[8], r32);
//}

// void matMulRegOptOld(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
//{
//     const auto i_size = A.row();
//     const auto j_size = B.col();
//     const auto k_size = A.col();

//    double*       c = C.data();
//    const double* b = B.data();
//    const double* a = A.data();

//    // add multithreading
//    for (size_t i = 0; i < i_size; i += I_BLOCK)
//    {
//        for (size_t j = 0; j < j_size; j += J_BLOCK)
//        {
//            for (size_t k = 0; k < k_size; k += K_BLOCK)
//            {
//                mulMatrix_y(
//                  &c[i * j_size + j], &a[i * k_size + k], &b[k * j_size + j], j_size, k_size);
//            }
//        }
//    }
//}

//    _mm_prefetch(&c[j_size], _MM_HINT_NTA);
//    _mm_prefetch(&c[2 * j_size], _MM_HINT_NTA);
//    _mm_prefetch(&c[3 * j_size], _MM_HINT_NTA);
//    _mm_prefetch(&a[a_cols], _MM_HINT_NTA);
//    _mm_prefetch(&a[2 * a_cols], _MM_HINT_NTA);
//    _mm_prefetch(&a[3 * a_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[b_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[2 * b_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[3 * b_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[4 * b_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[5 * b_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[7 * b_cols], _MM_HINT_NTA);
//    _mm_prefetch(&b[6 * b_cols], _MM_HINT_NTA);

//                {
//                    __m256d b0 = _mm256_loadu_pd(&b[0]);
//                    __m256d b1 = _mm256_loadu_pd(&b[4]);
//                    __m256d b2 = _mm256_loadu_pd(&b[8]);

//                    __m256d a0 = _mm256_broadcast_sd(&a[0]);

//                    r00 = _mm256_mul_pd(a0, b0);
//                    r01 = _mm256_mul_pd(a0, b1);
//                    r02 = _mm256_mul_pd(a0, b2);

//                    a += a_cols;
//                    a0 = _mm256_broadcast_sd(&a[0]);

//                    r10 = _mm256_mul_pd(a0, b0);
//                    r11 = _mm256_mul_pd(a0, b1);
//                    r12 = _mm256_mul_pd(a0, b2);

//                    a += a_cols;
//                    a0 = _mm256_broadcast_sd(&a[0]);

//                    r20 = _mm256_mul_pd(a0, b0);
//                    r21 = _mm256_mul_pd(a0, b1);
//                    r22 = _mm256_mul_pd(a0, b2);

//                    a += a_cols;
//                    a0 = _mm256_broadcast_sd(&a[0]);

//                    r30 = _mm256_mul_pd(a0, b0);
//                    r31 = _mm256_mul_pd(a0, b1);
//                    r32 = _mm256_mul_pd(a0, b2);

//                    b += b_cols;
//                }

// void mulMatrix_y_old(double*           c,
//                      const double*     ma,
//                      const double*     b,
//                      const std::size_t j_size,
//                      const std::size_t k_size)
//{

//    const auto b_cols = CACHE_BLOCK; // j_size;
//    const auto a_cols = CACHE_BLOCK; // k_size;

//    __m256d r00 = _mm256_setzero_pd();
//    __m256d r01 = _mm256_setzero_pd();
//    __m256d r02 = _mm256_setzero_pd();
//    __m256d r10 = _mm256_setzero_pd();
//    __m256d r11 = _mm256_setzero_pd();
//    __m256d r12 = _mm256_setzero_pd();
//    __m256d r20 = _mm256_setzero_pd();
//    __m256d r21 = _mm256_setzero_pd();
//    __m256d r22 = _mm256_setzero_pd();

//    __m256d r30 = _mm256_setzero_pd();
//    __m256d r31 = _mm256_setzero_pd();
//    __m256d r32 = _mm256_setzero_pd();

//    const double* a = ma;
//    {
//        __m256d b0 = _mm256_loadu_pd(&b[0]);
//        __m256d b1 = _mm256_loadu_pd(&b[4]);
//        __m256d b2 = _mm256_loadu_pd(&b[8]);

//        __m256d a0 = _mm256_broadcast_sd(&a[0]);

//        r00 = _mm256_mul_pd(a0, b0);
//        r01 = _mm256_mul_pd(a0, b1);
//        r02 = _mm256_mul_pd(a0, b2);

//        a += a_cols;
//        a0 = _mm256_broadcast_sd(&a[0]);

//        r10 = _mm256_mul_pd(a0, b0);
//        r11 = _mm256_mul_pd(a0, b1);
//        r12 = _mm256_mul_pd(a0, b2);

//        a += a_cols;
//        a0 = _mm256_broadcast_sd(&a[0]);

//        r20 = _mm256_mul_pd(a0, b0);
//        r21 = _mm256_mul_pd(a0, b1);
//        r22 = _mm256_mul_pd(a0, b2);

//        a += a_cols;
//        a0 = _mm256_broadcast_sd(&a[0]);

//        r30 = _mm256_mul_pd(a0, b0);
//        r31 = _mm256_mul_pd(a0, b1);
//        r32 = _mm256_mul_pd(a0, b2);

//        b += b_cols;
//    }

// #pragma GCC unroll(K_BLOCK - 1)
//     for (int k2 = 1; k2 < K_BLOCK; ++k2, b += b_cols)
//     {
//         a = ma;

//        __m256d b0 = _mm256_loadu_pd(&b[0]);
//        __m256d b1 = _mm256_loadu_pd(&b[4]);
//        __m256d b2 = _mm256_loadu_pd(&b[8]);

//        __m256d a0 = _mm256_broadcast_sd(&a[k2]);

//        r00 = _mm256_fmadd_pd(a0, b0, r00);
//        r01 = _mm256_fmadd_pd(a0, b1, r01);
//        r02 = _mm256_fmadd_pd(a0, b2, r02);

//        a += a_cols;
//        a0 = _mm256_broadcast_sd(&a[k2]);

//        r10 = _mm256_fmadd_pd(a0, b0, r10);
//        r11 = _mm256_fmadd_pd(a0, b1, r11);
//        r12 = _mm256_fmadd_pd(a0, b2, r12);

//        a += a_cols;
//        a0 = _mm256_broadcast_sd(&a[k2]);

//        r20 = _mm256_fmadd_pd(a0, b0, r20);
//        r21 = _mm256_fmadd_pd(a0, b1, r21);
//        r22 = _mm256_fmadd_pd(a0, b2, r22);

//        a += a_cols;
//        a0 = _mm256_broadcast_sd(&a[k2]);

//        r30 = _mm256_fmadd_pd(a0, b0, r30);
//        r31 = _mm256_fmadd_pd(a0, b1, r31);
//        r32 = _mm256_fmadd_pd(a0, b2, r32);
//    }

//    load_inc_store_double(&c[0], r00);
//    load_inc_store_double(&c[4], r01);
//    load_inc_store_double(&c[8], r02);
//    c += j_size;

//    load_inc_store_double(&c[0], r10);
//    load_inc_store_double(&c[4], r11);
//    load_inc_store_double(&c[8], r12);
//    c += j_size;

//    load_inc_store_double(&c[0], r20);
//    load_inc_store_double(&c[4], r21);
//    load_inc_store_double(&c[8], r22);
//    c += j_size;

//    load_inc_store_double(&c[0], r30);
//    load_inc_store_double(&c[4], r31);
//    load_inc_store_double(&c[8], r32);
//}
