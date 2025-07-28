#include "mm/core/reorderMatrix.hpp"
#include "mm/matmul/matMulSimd.hpp"

#include "mm/core/kernels.hpp"

#include <experimental/simd>
#include <thread>

#include "omp.h"

constexpr unsigned long PAGE_SIZE = 4096;

#define massert(x, msg) \
    (bool((x)) == true ? void(0) : throw std::runtime_error("Assertion failed: " #x " " msg))

namespace stdx = std::experimental;

using simd_d = stdx::native_simd<double>;

template<typename T>
using simd = stdx::fixed_size_simd<double, 4>; // stdx::native_simd<T>;

static_assert(simd<double>::size() == 4, "Expect 4 doubles per simd register");

static inline void load_inc_store_double(double* __restrict ptr, simd_d increment)
{
    simd_d vector(ptr, stdx::vector_aligned);
    vector += increment;
    vector.copy_to(ptr, stdx::vector_aligned);
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

template<int Nrs, typename T>
void store_kernel(T* c, simd<T>* r, int N)
{
    simd_d v1, v2, v3;
    int    idx = 0;
    for (int i = 0; i < 4; ++i, c += N)
    {
        _mm_prefetch(&c[0], _MM_HINT_NTA);
        _mm_prefetch(&c[4], _MM_HINT_NTA);
        _mm_prefetch(&c[8], _MM_HINT_NTA);

        v1 = simd_d(&c[0], stdx::element_aligned);
        v2 = simd_d(&c[4], stdx::element_aligned);
        v3 = simd_d(&c[8], stdx::element_aligned);

        v1 += r[idx + 0];
        v2 += r[idx + 1];
        v3 += r[idx + 2];

        v1.copy_to(&c[0], stdx::element_aligned);
        v2.copy_to(&c[4], stdx::element_aligned);
        v3.copy_to(&c[8], stdx::element_aligned);
        idx += 3;
    }
}

template<int Nr, int Mr, int Kc>
inline void pack_ukernel_arr_simd(const double* __restrict a,
                                  const double* __restrict b,
                                  double* __restrict c,
                                  int N)
{
    constexpr int CREG_CNT{Mr * Nr / 4};

    std::array<simd_d, CREG_CNT> c_reg{};

    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        int idx = 0;
        for (int i = 0; i < Mr; ++i)
        {
            simd_d a_reg(a[i]);
            for (int j = 0; j < Nr; j += 4, ++idx)
            {
                c_reg[idx] += a_reg * simd_d(&b[j], stdx::element_aligned);
            }
        }
    }

    int idx = 0;
    for (int i = 0; i < Mr; ++i, c += N)
    {
        for (int j = 0; j < Nr; j += 4, ++idx)
        {
            load_inc_store_double(&c[j], c_reg[idx]);
        }
    }
}

template<int Nr, int Mr, int Kc>
static void upkernel(const double* __restrict ma,
                     const double* __restrict b,
                     double* __restrict mc,
                     int N)
{
    double* c = mc;

    simd_d r00(0.0);
    simd_d r01(0.0);
    simd_d r02(0.0);
    simd_d r10(0.0);
    simd_d r11(0.0);
    simd_d r12(0.0);
    simd_d r20(0.0);
    simd_d r21(0.0);
    simd_d r22(0.0);
    simd_d r30(0.0);
    simd_d r31(0.0);
    simd_d r32(0.0);

    constexpr auto prefetch_type = _MM_HINT_NTA;

    const double* a = ma;

    static_assert(Kc % 2 == 0, "Kc must be even for manual unrolling");
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        // k = 4, 9
        // if (k % 2 == 0)
        //        _mm_prefetch(a + 8, _MM_HINT_T0);
        _mm_prefetch(b + 4 + Nr, _MM_HINT_T0);
        simd_d b0(&b[0], stdx::element_aligned);
        simd_d b1(&b[4], stdx::element_aligned);
        simd_d b2(&b[8], stdx::element_aligned);

        simd_d a0(a[0]);
        r00 += a0 * b0;
        r01 += a0 * b1;
        r02 += a0 * b2;

        a0 = simd_d(a[1]);
        r10 += a0 * b0;
        r11 += a0 * b1;
        r12 += a0 * b2;

        a0 = simd_d(a[2]);
        r20 += a0 * b0;
        r21 += a0 * b1;
        r22 += a0 * b2;

        a0 = simd_d(a[3]);
        r30 += a0 * b0;
        r31 += a0 * b1;
        r32 += a0 * b2;
    }

    _mm_prefetch(c + N, prefetch_type);
    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);
    c += N;

    _mm_prefetch(c + N, prefetch_type);
    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    _mm_prefetch(c + N, prefetch_type);
    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
}

template<int mask>
void swap_elem_index(__m256d& r00, __m256d& r10)
{

    __m256d temp = r00; // Save original r00.
    // For r00: replace its index 1 element with that from r10.
    r00 = _mm256_blend_pd(r00, r10, mask);
    // For r10: replace its index 1 element with the old value from r00.
    r10 = _mm256_blend_pd(r10, temp, mask);
}

template<int Nr, int Mr, int Kc>
static inline void upkernelArIntrinsics(const double* __restrict ma,
                                        const double* __restrict b,
                                        double* __restrict mc,
                                        int N)
{
    double* c = mc;
    __m256d r[Nr];
    for (int idx = 0; idx < Nr; ++idx)
    {
        r[idx] = _mm256_setzero_pd();
    }

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        __m256d b0 = _mm256_loadu_pd(&b[0]);
        __m256d b1 = _mm256_loadu_pd(&b[4]);
        __m256d b2 = _mm256_loadu_pd(&b[8]);

        __m256d a0 = _mm256_broadcast_sd(&a[0]);
        r[0] += a0 * b0;
        r[1] += a0 * b1;
        r[2] += a0 * b2;

        a0 = _mm256_broadcast_sd(&a[1]);
        r[3] += a0 * b0;
        r[4] += a0 * b1;
        r[5] += a0 * b2;

        a0 = _mm256_broadcast_sd(&a[2]);
        r[6] += a0 * b0;
        r[7] += a0 * b1;
        r[8] += a0 * b2;

        a0 = _mm256_broadcast_sd(&a[3]);
        ;
        r[9] += a0 * b0;
        r[10] += a0 * b1;
        r[11] += a0 * b2;
    }

    load_inc_store_double(&c[0], r[0]);
    load_inc_store_double(&c[4], r[1]);
    load_inc_store_double(&c[8], r[2]);
    c += N;

    load_inc_store_double(&c[0], r[3]);
    load_inc_store_double(&c[4], r[4]);
    load_inc_store_double(&c[8], r[5]);
    c += N;

    load_inc_store_double(&c[0], r[6]);
    load_inc_store_double(&c[4], r[7]);
    load_inc_store_double(&c[8], r[8]);
    c += N;

    load_inc_store_double(&c[0], r[9]);
    load_inc_store_double(&c[4], r[10]);
    load_inc_store_double(&c[8], r[11]);
}

template<int Nr, int Mr, int Kc>
static void upkernelAr(const double* __restrict ma,
                       const double* __restrict b,
                       double* __restrict mc,
                       int N)
{
    double* c     = mc;
    simd_d  r[Nr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d b0(&b[0], stdx::element_aligned);
        simd_d b1(&b[4], stdx::element_aligned);
        simd_d b2(&b[8], stdx::element_aligned);

        simd_d a0(a[0]);
        r[0] += a0 * b0;
        r[1] += a0 * b1;
        r[2] += a0 * b2;

        a0 = simd_d(a[1]);
        r[3] += a0 * b0;
        r[4] += a0 * b1;
        r[5] += a0 * b2;

        a0 = simd_d(a[2]);
        r[6] += a0 * b0;
        r[7] += a0 * b1;
        r[8] += a0 * b2;

        a0 = simd_d(a[3]);
        r[9] += a0 * b0;
        r[10] += a0 * b1;
        r[11] += a0 * b2;
    }

    load_inc_store_double(&c[0], r[0]);
    load_inc_store_double(&c[4], r[1]);
    load_inc_store_double(&c[8], r[2]);
    c += N;

    load_inc_store_double(&c[0], r[3]);
    load_inc_store_double(&c[4], r[4]);
    load_inc_store_double(&c[8], r[5]);
    c += N;

    load_inc_store_double(&c[0], r[6]);
    load_inc_store_double(&c[4], r[7]);
    load_inc_store_double(&c[8], r[8]);
    c += N;

    load_inc_store_double(&c[0], r[9]);
    load_inc_store_double(&c[4], r[10]);
    load_inc_store_double(&c[8], r[11]);
}

template<typename T>
void compute_block(const T* a, const T* b, simd<T>* r)
{
    simd_d b0(&b[0], stdx::element_aligned);
    simd_d b1(&b[4], stdx::element_aligned);
    simd_d b2(&b[8], stdx::element_aligned);

    simd_d a0(a[0]);
    r[0] += a0 * b0;
    r[1] += a0 * b1;
    r[2] += a0 * b2;

    a0 = simd_d(a[1]);
    r[3] += a0 * b0;
    r[4] += a0 * b1;
    r[5] += a0 * b2;

    a0 = simd_d(a[2]);
    r[6] += a0 * b0;
    r[7] += a0 * b1;
    r[8] += a0 * b2;

    a0 = simd_d(a[3]);
    r[9] += a0 * b0;
    r[10] += a0 * b1;
    r[11] += a0 * b2;
}

template<std::size_t RowIdx, typename T, std::size_t... I>
void store_row(T* c, simd<T>* r, std::index_sequence<I...>)
{

    (..., (load_inc_store_double(&c[I * simd<T>::size()], r[RowIdx * sizeof...(I) + I])));
}

template<int Nrs, typename T, std::size_t... RowIndices>
void store_kernel(T* c, simd<T>* r, int N, std::index_sequence<RowIndices...>)
{
    (...,
     (_mm_prefetch(c + 8, _MM_HINT_NTA),
      store_row<RowIndices>(c, r, std::make_index_sequence<Nrs>{}),
      c += N,
      _mm_prefetch(c, _MM_HINT_NTA)));
}

template<typename T, std::size_t... J>
void compute_row(const simd<T>& a, simd<T>* b, simd<T>* r, std::index_sequence<J...>)
{
    (..., (r[J] += a * b[J]));
}

template<typename T, size_t... I, size_t... J>
void compute_kernel(const T* a,
                    const T* b,
                    simd<T>* r,
                    std::index_sequence<I...>,
                    std::index_sequence<J...>)
{
    constexpr int Nrs = sizeof...(J);

    simd<T> bs[Nrs] = {simd<T>(&b[J * simd<T>::size()], stdx::element_aligned)...};
    (..., (compute_row(simd<T>(a[I]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
}

// __attribute__((hot))
template<int Nr, int Mr, int Kc, typename T>
static inline void cpp_ukernel(const T* __restrict a, const T* __restrict b, T* __restrict c, int N)
{
    constexpr int  Nrs         = Nr / simd_d::size();
    simd_d         r[Mr * Nrs] = {};
    constexpr auto Nseq        = std::make_index_sequence<Nrs>{};
    constexpr auto Mseq        = std::make_index_sequence<Mr>{};

    for (int k = 0; k < Kc; k += 8, b += 8 * Nr, a += 8 * Mr)
    {
        _mm_prefetch(b + 1 * 8, _MM_HINT_T0);
        _mm_prefetch(a + 1 * 8, _MM_HINT_T0);

        compute_kernel(a, b, r, Mseq, Nseq);

        _mm_prefetch(b + 2 * 8, _MM_HINT_T0);
        _mm_prefetch(a + 2 * 8, _MM_HINT_T0);
        compute_kernel(a + Mr, b + Nr, r, Mseq, Nseq);

        _mm_prefetch(b + 3 * 8, _MM_HINT_T0);
        compute_kernel(a + 2 * Mr, b + 2 * Nr, r, Mseq, Nseq);

        _mm_prefetch(b + 4 * 8, _MM_HINT_T0);
        _mm_prefetch(a + 3 * 8, _MM_HINT_T0);
        compute_kernel(a + 3 * Mr, b + 3 * Nr, r, Mseq, Nseq);

        _mm_prefetch(b + 5 * 8, _MM_HINT_T0);
        _mm_prefetch(a + 4 * 8, _MM_HINT_T0);

        compute_kernel(a + 4 * Mr, b + 4 * Nr, r, Mseq, Nseq);

        _mm_prefetch(b + 6 * 8, _MM_HINT_T0);
        _mm_prefetch(a + 5 * 3, _MM_HINT_T0);
        compute_kernel(a + 5 * Mr, b + 5 * Nr, r, Mseq, Nseq);

        _mm_prefetch(b + 7 * 8, _MM_HINT_T0);
        //_mm_prefetch(b + 9 * 8, _MM_HINT_T0);
        //_mm_prefetch(b + 10 * 8, _MM_HINT_T0);
        _mm_prefetch(a + 6 * 8, _MM_HINT_T0);
        compute_kernel(a + 6 * Mr, b + 6 * Nr, r, Mseq, Nseq);

        _mm_prefetch(b + 8 * 8, _MM_HINT_T0);
        //_mm_prefetch(b + 11 * 8, _MM_HINT_T0);
        compute_kernel(a + 7 * Mr, b + 7 * Nr, r, Mseq, Nseq);
    }
    // handle k tail, no tail if inc with 1

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
    // store_kernel<Nrs>(c, r, N);
}

template<int Nr, int Mr, int Kc, typename T>
static void cpp_ukernelLambda(const T* __restrict ma,
                              const T* __restrict b,
                              T* __restrict mc,
                              int N)
{
    static_assert(Kc % 2 == 0, "Kc must be even for manual unrolling");

    T*       c = mc;
    const T* a = ma;

    simd_d r[Nr] = {};

    constexpr int Nrs = Nr / Mr;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {

        // Compute Block (Replaces compute_block)

        [&]<size_t... I, size_t... J>(std::index_sequence<I...>, std::index_sequence<J...>)
        {
            simd<T> bs[Nrs] = {simd<T>(&b[J * simd<T>::size()], stdx::element_aligned)...};

            (..., (compute_row(simd<T>(a[I]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
        }(std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nr / Mr>(c, r, N, std::make_index_sequence<Mr>{});
}

/*
 * Cause error
    constexpr int Nc = 660;
    constexpr int Mc = 24;
    constexpr int Kc = 96;
    constexpr int Nr = 12;
    constexpr int Mr = 4;
    constexpr int Kr = 1;
*/

void matMulSimd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // TODO: assert MNK % TileSize != 0
    // constexpr int Nc = 720 / 2;
    // constexpr int Mc = 30;
    // constexpr int Kc = 96; // 6 + 12 * 8; // Kc = 96+12*8 =best, Nc=720/2

    // constexpr int Mc = 24;
    // constexpr int Kc = 96;
    // constexpr int Nc = 720;

    // These values casue error in reorderRowMajorMatrix for zen5!
    // Reason: buffer hardcoded num of thread was 4
    constexpr int Nc = 180; // 180(best for hawswell)
    constexpr int Mc = 20;
    constexpr int Kc = 80;

    //    constexpr int Nc = 768;
    //    constexpr int Mc = 96; // 128;(452ms) // 96;(453ms)
    //    constexpr int Kc = 256;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    // consider to increase to improve repack perf
    // Kr = 1, no need for padding over k dim
    constexpr int Kr = 1;

    auto num_threads = 16; // std::thread::hardware_concurrency();
    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    massert(N % Nc == 0, "N % Nc == 0");
    massert(K % Kc == 0, "K % Kc == 0");
    massert(M % Mc == 0, "M % Mc == 0");
    massert(N % num_threads == 0, "N % num_threads == 0");
    massert((N / num_threads) % Nc == 0, "(N/num_threads) % Nc == 0");

    std::vector<double, boost::alignment::aligned_allocator<double, PAGE_SIZE>> buffer(
      num_threads * Kc * (Mc + Nc));

#pragma omp parallel for num_threads(num_threads)
    for (int j_block = 0; j_block < N; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        for (int k_block = 0; k_block < K; k_block += Kc)
        {
            // reorderRowMajorMatrixAVX<Kc, Nc, Kr, Nr>(
            //   B.data() + N * k_block + j_block, N, buf + Mc * Kc);
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(
              B.data() + N * k_block + j_block, N, buf + Mc * Kc);

            for (int i_block = 0; i_block < M; i_block += Mc)
            {
                // all threads should access same memory
                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K, buf);

                for (int j = 0; j < Nc; j += Nr)
                {
                    const double* Bc1 = buf + Mc * Kc + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = C.data() + N * i_block + j + N * i + j_block;
                        const double* Ac0 = buf + Kc * i;

                        kernels::cpp_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // cpp_ukernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        //  pack_ukernel_arr_simd<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N); // sligtly
                        //  worse

                        // upkernelArIntrinsics<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // upkernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // upkernelAr<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        //  cpp_ukernelLambda<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }
        }
    }
}

/////       TAILS
///
///

void matMulSimdTails(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    // NEW BEST
    constexpr int Nc = 180;
    constexpr int Mc = 20;
    constexpr int Kc = 80;

    constexpr int Nr = 12;
    constexpr int Mr = 4;

    // consider to increase to improve repack perf
    // Kr = 1, no need for padding over k dim
    constexpr int Kr = 1;

    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const auto N = B.col();
    const auto K = A.col();
    const auto M = A.row();

    auto num_threads = std::thread::hardware_concurrency();

    std::vector<double, boost::alignment::aligned_allocator<double, PAGE_SIZE>> buffer(
      num_threads * Kc * (Mc + Nc));

    // tail is only in last block
    int dNc = N % Nc;
    int jl  = N - dNc;

#pragma omp parallel for num_threads(num_threads)
    for (int j_block = 0; j_block < jl; j_block += Nc)
    {
        auto       tid   = omp_get_thread_num();
        const auto ofs   = tid * Kc * (Mc + Nc);
        double*    a_buf = buffer.data() + ofs;
        double*    b_buf = a_buf + Mc * Kc;

        // For dDEBUG:
        //        constexpr int dKc = 3;

        int dKc   = K % Kc;
        int klast = K - dKc;

        for (int k_block = 0; k_block < klast; k_block += Kc)
        {
            // I can guarantee the we always within the block and no padding needed
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(B.data() + N * k_block + j_block, N, b_buf);

            int dMc   = M % Mc;
            int ilast = M - dMc;
            for (int i_block = 0; i_block < ilast; i_block += Mc)
            {
                // Can be access out of bound if i+Mc > M. No
                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K, a_buf);

                for (int j = 0; j < Nc; j += Nr)
                {
                    const double* Bc1 = b_buf + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = C.data() + N * (i_block + i) + j_block + j;
                        const double* Ac0 = a_buf + Kc * i;

                        // TODO: deduce args from span?
                        kernels::cpp_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }

            const double* Ac1 = A.data() + k_block + ilast * K;
            double*       Cc1 = C.data() + j_block + ilast * N;

            handleItail<Nr, Kr, Nc, Kc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, dMc);
        }

        // TODO: Choose Ktails properly
        handleKtail<Mr, Nr, Kr, Mc, Nc, 20, 10, 4, 2, 1>(a_buf,
                                                         b_buf,
                                                         A.data() + klast,
                                                         B.data() + N * klast + j_block,
                                                         C.data() + j_block,
                                                         M,
                                                         N,
                                                         K,
                                                         dKc);

        // TODO: Can recalc b_buf address to be cllsoer to a_buf
    }

    // TODO: Add multithreading

    handleJtail<Mr, Kr, Mc, Kc, 12, 8, 4, 2, 1>(
      buffer.data(), A.data(), &B(0, jl), &C(0, jl), M, K, N, dNc);
}
