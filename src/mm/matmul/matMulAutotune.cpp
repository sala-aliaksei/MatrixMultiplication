#include "matMulAutotune.hpp"
#include "mm/core/reorderMatrix.hpp"
// #include "mm/core/kernels.hpp"

// #include "omp.h"

// #include <immintrin.h>

#ifdef N_CACHE_SIZE
constexpr int Nc = N_CACHE_SIZE;
#else
constexpr int Nc = 96;
#endif

#ifdef M_CACHE_SIZE
constexpr int Mc = M_CACHE_SIZE;
#else
constexpr int Mc = 96;
#endif

#ifdef K_CACHE_SIZE
constexpr int Kc = K_CACHE_SIZE;
#else
constexpr int Kc = 96;
#endif

// TODO: I copy impl to speed up compilation, go back to reuse existing impl

// BEST
// constexpr int Nc = 720;
// constexpr int Mc = 20;
// constexpr int Kc = 90;

// constexpr int Mc = 128;
// constexpr int Kc = 256;

// constexpr int N_LOG_DIM = 1;

// #include <experimental/simd>

// namespace stdx = std::experimental;

// template<typename T, int WIDTH>
// using fix_simd = stdx::fixed_size_simd<T, WIDTH>;

// namespace kernels
// {

// template<typename T, int WIDTH>
// static inline void load_inc_store_double(T* __restrict ptr, fix_simd<T, WIDTH> increment)
// {
//     fix_simd<T, WIDTH> vector(ptr, stdx::element_aligned);
//     vector += increment;
//     vector.copy_to(ptr, stdx::element_aligned);
// }

// template<std::size_t RowIdx, typename T, int WIDTH, std::size_t... I>
// static inline void store_row(T* c, fix_simd<T, WIDTH>* r, std::index_sequence<I...>)
// {
//     (..., (load_inc_store_double(&c[I * WIDTH], r[RowIdx * sizeof...(I) + I])));
// }

// template<int Nrs, typename T, int WIDTH, std::size_t... RowIndices>
// static inline void store_kernel(T*                  c,
//                                 fix_simd<T, WIDTH>* r,
//                                 int                 N,
//                                 std::index_sequence<RowIndices...>)
// {
//     (..., (store_row<RowIndices>(c, r, std::make_index_sequence<Nrs>{}), c += N));
// }

// template<typename T, int WIDTH, std::size_t... J>
// static inline void packed_compute_row(const fix_simd<T, WIDTH>& a,
//                                       fix_simd<T, WIDTH>*       b,
//                                       fix_simd<T, WIDTH>*       r,
//                                       std::index_sequence<J...>)
// {
//     (..., (r[J] += a * b[J]));
// }

// template<typename T, int WIDTH, size_t... I, size_t... J>
// static inline void packed_compute_kernel(const T*            a,
//                                          const T*            b,
//                                          fix_simd<T, WIDTH>* r,
//                                          std::index_sequence<I...>,
//                                          std::index_sequence<J...>)
// {
//     constexpr int Nrs = sizeof...(J);
//     // constexpr int Mrs = sizeof...(I);
//     //  Nrs*Mrs - size of r array

//     fix_simd<T, WIDTH> bs[Nrs] = {fix_simd<T, WIDTH>(&b[J * WIDTH], stdx::element_aligned)...};
//     (...,
//      (packed_compute_row(
//        fix_simd<T, WIDTH>(a[I]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
// }

// // Same perf as manual impl for Nr = 12, 8, 4;
// template<int Nr, int Mr, int Kc, typename T>
// static inline void cpp_packed_kernel(const T* __restrict a,
//                                      const T* __restrict b,
//                                      T* __restrict c,
//                                      int N)
//     requires(Nr % 4 == 0)
// {
//     constexpr int Nrs{Nr / 4};

//     fix_simd<T, 4> r[Nrs * Mr] = {};
//     for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
//     {
//         packed_compute_kernel(
//           a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
//     }
//     store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
// }

// template<int Nr, int Mr, int Kc, typename T>
// inline void cpp_packed_kernel(const T* __restrict ma, const T* __restrict b, T* __restrict c, int
// N)
//     requires(Nr == 6 && Mr == 4)
// {
//     // TODO: IT is not packed!!!
//     static_assert(std::is_same_v<T, double>);
//     packed_ukernel6x4<Kc>(ma, b, c, N);
// }

// template<int Nr, int Mr, int Kc, typename T>
// static inline void cpp_packed_kernel(const T* __restrict a,
//                                      const T* __restrict b,
//                                      T* __restrict c,
//                                      int N)
//     requires(Nr == 2 or Nr == 1)
// {
//     constexpr int Nrs = 1;

//     fix_simd<T, Nr> r[Mr] = {};
//     for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
//     {
//         packed_compute_kernel(
//           a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
//     }

//     store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
// }
// } // namespace kernels

// template<int Nr, int Kr, int Kc, int... TailSize>
// static inline void handleItail(double*       a_buf,
//                                const double* a,
//                                const double* packed_b,
//                                double*       c,
//                                int           M,
//                                int           N,
//                                int           K,
//                                int           dNc,
//                                int           i_tail_size)
// {
//     // TODO: Add multithreading

//     int i_ofs = 0;

//     (...,
//      (
//        [&]
//        {
//            constexpr int Mrr = TailSize;
//            if (i_tail_size >= Mrr)
//            {

//                int dMc = i_tail_size - i_tail_size % Mrr;
//                reorderColOrderMatrixTail<Mrr, Kr>(a + K * i_ofs, K, a_buf, dMc, Kc);

//                // TODO:[Critical] What if dNc%Nr!=0 ??? We need to handle tail here as well
//                for (int j = 0; j < dNc; j += Nr)
//                {

//                    int  idx    = 0;
//                    auto i_tail = i_tail_size;
//                    while (i_tail >= Mrr)
//                    {
//                        kernels::cpp_packed_kernel<Nr, Mrr, Kc>(
//                          &a_buf[idx * Kc], &packed_b[Kc * j], &c[(idx + i_ofs) * N + j], N);

//                        idx += Mrr;
//                        i_tail -= Mrr;
//                    }
//                }

//                i_ofs += i_tail_size - i_tail_size % Mrr;
//                i_tail_size %= Mrr;
//            }
//        }()));
// }

// template<int Nr, int Kr, int Nc, int Kc, int... TailSize>
// static inline void handleItail(double*       a_buf,
//                                const double* a,
//                                const double* packed_b,
//                                double*       c,
//                                int           M,
//                                int           N,
//                                int           K,
//                                int           i_tail_size)
// {
//     // TODO: Add multithreading

//     int i_ofs = 0;

//     (...,
//      (
//        [&]
//        {
//            constexpr int Mrr = TailSize;
//            if (i_tail_size >= Mrr)
//            {
//                int dMc = i_tail_size - i_tail_size % Mrr;
//                reorderColOrderMatrixTail<Mrr, Kr>(a + K * i_ofs, K, a_buf, dMc, Kc);

//                // TODO: What if Nc%Nr!=0 ???
//                for (int j = 0; j < Nc; j += Nr)
//                {
//                    int  idx    = 0;
//                    auto i_tail = i_tail_size;
//                    while (i_tail >= Mrr)
//                    {
//                        kernels::cpp_packed_kernel<Nr, Mrr, Kc>(
//                          &a_buf[idx * Kc], &packed_b[Kc * j], &c[(i_ofs + idx) * N + j], N);

//                        idx += Mrr;
//                        i_tail -= Mrr;
//                    }
//                }

//                i_ofs += i_tail_size - i_tail_size % Mrr;
//                i_tail_size %= Mrr;
//            }
//        }()));
// }

// template<int Mr, int Nr, int Kr, int Mc, int... TailSize>
// static inline void handleKtail(double*       a_buf,
//                                double*       b_buf,
//                                const double* a,
//                                const double* b,
//                                double*       c,
//                                int           M,
//                                int           N,
//                                int           K,
//                                int           dNc,
//                                int           k_tail_size)
// {
//     // TODO: Add multithreading
//     int kofs = 0;
//     (...,
//      (
//        [&]
//        {
//            constexpr int Kcc = TailSize;
//            if (k_tail_size >= Kcc)
//            {

//                int dKc = k_tail_size - k_tail_size % Kcc;
//                int kdx = kofs;

//                for (int k_block = 0; k_block < dKc; k_block += Kcc)
//                {
//                    reorderRowMajorMatrix<Kr, Nr>(b + N * (k_block + kdx), N, b_buf, Kcc, dNc);

//                    int dMc   = M % Mc;
//                    int ilast = M - dMc;
//                    for (int i_block = 0; i_block < ilast; i_block += Mc)
//                    {
//                        reorderColOrderMatrix<Mc, Kcc, Mr, Kr>(
//                          a + K * i_block + (k_block + kdx), K, a_buf);

//                        for (int j = 0; j < dNc; j += Nr)
//                        {
//                            const double* Bc1 = b_buf + Kcc * j;
//                            for (int i = 0; i < Mc; i += Mr)
//                            {
//                                double*       Cc0 = c + N * (i_block + i) + j;
//                                const double* Ac0 = a_buf + Kcc * i;

//                                // TODO: deduce args from span?
//                                kernels::cpp_packed_kernel<Nr, Mr, Kcc>(Ac0, Bc1, Cc0, N);
//                            }
//                        }
//                    }

//                    const double* Ac1 = a + (k_block + kdx) + K * ilast;
//                    double*       Cc1 = c + N * ilast;

//                    handleItail<Nr, Kr, Kcc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, dNc,
//                    dMc);
//                }

//                kofs += dKc;
//                k_tail_size %= Kcc;
//            }
//        }()));
// }

// template<int Mr, int Nr, int Kr, int Mc, int Nc, int... TailSize>
// static inline void handleKtail(double*       a_buf,
//                                double*       b_buf,
//                                const double* a,
//                                const double* b,
//                                double*       c,
//                                int           M,
//                                int           N,
//                                int           K,
//                                int           k_tail_size)
// {
//     // TODO: Add multithreading
//     int kofs = 0;
//     (...,
//      (
//        [&]
//        {
//            constexpr int Kcc = TailSize;
//            if (k_tail_size >= Kcc)
//            {

//                int dKc = k_tail_size - k_tail_size % Kcc;
//                int kdx = kofs;

//                for (int k_block = 0; k_block < dKc; k_block += Kcc)
//                {
//                    reorderRowMajorMatrix<Kcc, Nc, Kr, Nr>(b + N * (k_block + kdx), N, b_buf);

//                    int dMc   = M % Mc;
//                    int ilast = M - dMc;
//                    for (int i_block = 0; i_block < ilast; i_block += Mc)
//                    {
//                        reorderColOrderMatrix<Mc, Kcc, Mr, Kr>(
//                          a + K * i_block + k_block + kdx, K, a_buf);

//                        for (int j = 0; j < Nc; j += Nr)
//                        {
//                            const double* Bc1 = b_buf + Kcc * j;
//                            for (int i = 0; i < Mc; i += Mr)
//                            {
//                                double*       Cc0 = c + N * (i_block + i) + j;
//                                const double* Ac0 = a_buf + Kcc * i;

//                                // TODO: deduce args from span?
//                                kernels::cpp_packed_kernel<Nr, Mr, Kcc>(Ac0, Bc1, Cc0, N);
//                            }
//                        }
//                    }

//                    // What if we don't have Mr rows anymore and tail is 1, (new Mr == 1)?

//                    const double* Ac1 = a + k_block + kdx + K * ilast;
//                    double*       Cc1 = c + N * ilast;

//                    handleItail<Nr, Kr, Nc, Kcc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K,
//                    dMc);
//                }

//                kofs += dKc;
//                k_tail_size %= Kcc;
//            }
//        }()));
// }

// template<int Mr, int Kr, int Mc, int Kc, int... TailSize>
// static inline void handleJtail(double*       buf,
//                                const double* ma,
//                                const double* mb,
//                                double*       mc,
//                                int           M,
//                                int           K,
//                                int           N,
//                                int           j_tail_size)
// {

//     // TODO: Add multithreading
//     int j_ofs = 0;
//     (...,
//      (
//        [&]
//        {
//            constexpr int Nrr = TailSize;
//            if (j_tail_size >= Nrr)
//            {
//                // dNc % Nrr == 0 always
//                int dNc = j_tail_size - j_tail_size % Nrr;

//                double* a_buf = buf;
//                double* b_buf = a_buf + Mc * Kc;

//                int dKc   = K % Kc;
//                int klast = K - dKc;
//                for (int k_block = 0; k_block < klast; k_block += Kc)
//                {
//                    int i_tail_size = M % Mc;
//                    int ilast       = M - i_tail_size;

//                    for (int i_block = 0; i_block < ilast; i_block += Mc)
//                    {

//                        reorderColOrderMatrix<Mc, Kc, Mr, Kr>(ma + K * i_block + k_block, K,
//                        a_buf);

//                        int j_tail = j_tail_size;
//                        int jjdx   = j_ofs;

//                        while (j_tail >= Nrr)
//                        {
//                            reorderRowMajorMatrix<Kr, Nrr>(
//                              mb + N * k_block + jjdx, N, b_buf, Kc, dNc);

//                            // TODO: What if Mc%Mr != 0 ?
//                            for (int i = 0; i < Mc; i += Mr)
//                            {
//                                double*       Cc0 = mc + N * (i_block + i) + jjdx;
//                                const double* Ac0 = a_buf + Kc * i;

//                                kernels::cpp_packed_kernel<Nrr, Mr, Kc>(Ac0, b_buf, Cc0, N);
//                            }

//                            j_tail -= Nrr;
//                            jjdx += Nrr;
//                        }
//                    }

//                    const double* Ac1 = ma + k_block + K * ilast;
//                    const double* Bc1 = mb + N * k_block + j_ofs;
//                    double*       Cc1 = mc + N * ilast + j_ofs;

//                    reorderRowMajorMatrix<Kr, Nrr>(Bc1, N, b_buf, Kc, dNc);

//                    handleItail<Nrr, Kr, Kc, 4, 3, 2, 1>(
//                      a_buf, Ac1, b_buf, Cc1, M, N, K, dNc, i_tail_size);
//                }

//                // TODO: Choose probel block sizes for Kcc
//                handleKtail<Mr, Nrr, Kr, Mc, 20, 10, 4, 2, 1>(
//                  a_buf, b_buf, ma + klast, mb + N * klast + j_ofs, mc + j_ofs, M, N, K, dNc,
//                  dKc);

//                j_ofs += dNc;
//                j_tail_size %= Nrr;
//            }
//        }()));
// }

// void matMulAutotune(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
// {

//     // NEW BEST
//     //    constexpr int Nc = 180;
//     //    constexpr int Mc = 20;
//     //    constexpr int Kc = 80;

//     constexpr int Nr = 12;
//     constexpr int Mr = 4;

//     // consider to increase to improve repack perf
//     // Kr = 1, no need for padding over k dim
//     constexpr int Kr = 1;

//     static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
//     static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
//     static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

//     const auto N = B.col();
//     const auto K = A.col();
//     const auto M = A.row();

//     std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(4 * Kc
//                                                                                   * (Mc + Nc));

//     // tail is only in last block
//     int dNc = N % Nc;
//     int jl  = N - dNc;

// #pragma omp parallel for
//     for (int j_block = 0; j_block < jl; j_block += Nc)
//     {
//         auto       tid   = omp_get_thread_num();
//         const auto ofs   = tid * Kc * (Mc + Nc);
//         double*    b_buf = buffer.data() + ofs + Mc * Kc;
//         double*    a_buf = buffer.data() + ofs;

//         int dKc   = K % Kc;
//         int klast = K - dKc;

//         for (int k_block = 0; k_block < klast; k_block += Kc)
//         {
//             // Can be access out of bound if j+Nc > N
//             // TODO : I can guarantee the we always within the block and no padding needed
//             reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(B.data() + N * k_block + j_block, N, b_buf);

//             int dMc   = M % Mc;
//             int ilast = M - dMc;
//             for (int i_block = 0; i_block < ilast; i_block += Mc)
//             {
//                 // Can be access out of bound if i+Mc > M
//                 // how to we reorder if there is a tail?
//                 reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K,
//                 a_buf);

//                 for (int j = 0; j < Nc; j += Nr)
//                 {
//                     const double* Bc1 = b_buf + Kc * j;
//                     for (int i = 0; i < Mc; i += Mr)
//                     {
//                         double*       Cc0 = C.data() + N * (i_block + i) + j_block + j;
//                         const double* Ac0 = a_buf + Kc * i;

//                         // TODO: deduce args from span?
//                         kernels::cpp_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
//                     }
//                 }
//             }

//             // TODO: reorder I tail
//             // reorderColOrderPaddingMatrix

//             // What if we don't have Mr rows anymore and tail is 1, (new Mr == 1)?

//             const double* Ac1 = A.data() + k_block + ilast * K;
//             double*       Cc1 = C.data() + j_block + ilast * N;

//             handleItail<Nr, Kr, Nc, Kc, 4, 3, 2, 1>(a_buf, Ac1, b_buf, Cc1, M, N, K, dMc);
//         }

//         // TODO: Choose Ktails properly
//         handleKtail<Mr, Nr, Kr, Mc, Nc, 20, 10, 4, 2, 1>(a_buf,
//                                                          b_buf,
//                                                          A.data() + klast,
//                                                          B.data() + N * klast + j_block,
//                                                          C.data() + j_block,
//                                                          M,
//                                                          N,
//                                                          K,
//                                                          dKc);

//         // TODO: Can recalc b_buf address to be cllsoer to a_buf
//     }

//     // TODO: Add multithreading

//     handleJtail<Mr, Kr, Mc, Kc, 12, 8, 4, 2, 1>(
//       buffer.data(), A.data(), &B(0, jl), &C(0, jl), M, K, N, dNc);
// }

#include <mm/core/Matrix.hpp>
// #include <mm/core/kernels.hpp>
// #include <mm/core/reorderMatrix.hpp>
// #include <mm/core/bf16kernel.hpp>

#include <thread>
// #include <algorithm>
// #include <omp.h>
#include <experimental/simd>
namespace
{

constexpr int PAGE_SIZE = 4096;
namespace stdx          = std::experimental;

template<typename T, int WIDTH>
using fix_simd = stdx::fixed_size_simd<T, WIDTH>;

template<typename T, int WIDTH>
static inline void load_inc_store_double(T* __restrict ptr, fix_simd<T, WIDTH> increment)
{
    fix_simd<T, WIDTH> vector(ptr, stdx::element_aligned);
    vector += increment;
    vector.copy_to(ptr, stdx::element_aligned);
}

template<std::size_t RowIdx, typename T, int WIDTH, std::size_t... I>
static inline void store_row(T* c, fix_simd<T, WIDTH>* r, std::index_sequence<I...>)
{
    (..., (load_inc_store_double(&c[I * WIDTH], r[RowIdx * sizeof...(I) + I])));
}

template<int Nrs, typename T, int WIDTH, std::size_t... RowIndices>
static inline void store_kernel(T*                  c,
                                fix_simd<T, WIDTH>* r,
                                int                 N,
                                std::index_sequence<RowIndices...>)
{
    (..., (store_row<RowIndices>(c, r, std::make_index_sequence<Nrs>{}), c += N));
}

template<int Nrs, int Mr, typename T, int WIDTH>
static inline void store_kernel(T* c, fix_simd<T, WIDTH>* r, int N)
{
    _mm_prefetch(c + N, _MM_HINT_NTA);
    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

template<typename T, int WIDTH, std::size_t... J>
static inline void packed_compute_row(const fix_simd<T, WIDTH>& a,
                                      fix_simd<T, WIDTH>*       b,
                                      fix_simd<T, WIDTH>*       r,
                                      std::index_sequence<J...>)
{
    (..., (r[J] += a * b[J]));
}

template<typename T, int WIDTH, size_t... I, size_t... J>
static inline void packed_compute_kernel(const T*            a,
                                         const T*            b,
                                         fix_simd<T, WIDTH>* r,
                                         std::index_sequence<I...>,
                                         std::index_sequence<J...>)
{
    constexpr int Nrs = sizeof...(J);
    // constexpr int Mrs = sizeof...(I);
    //  Nrs*Mrs - size of r array
    //

    fix_simd<T, WIDTH> bs[Nrs] = {fix_simd<T, WIDTH>(&b[J * WIDTH], stdx::element_aligned)...};
    (...,
     (packed_compute_row(
       fix_simd<T, WIDTH>(a[I]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
}

template<int Mr, int Nrs, typename T, int WIDTH>
static inline void packed_compute_kernel(const T* a, const T* b, fix_simd<T, WIDTH>* r)
{
    packed_compute_kernel(a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
}

template<int Nr, int Mr, int Kc, typename T>
static inline void zen5_packed_kernel(const T* __restrict a,
                                      const T* __restrict b,
                                      T* __restrict c,
                                      int N)
{
    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;
    constexpr int  Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        packed_compute_kernel<Mr, Nrs>(a, b, r);
    }
    store_kernel<Nrs, Mr>(c, r, N);
}

// Padded variant: packs a Kc x Nc tile in Row-major micro-tiles (Kr x Nr),
// reading only rowsToCopy x colsToCopy elements from source and zero-filling the rest
template<int Kc, int Nc, int Kr, int Nr, typename T>
inline void reorderRowMajorMatrixPadded(const T* __restrict matrix,
                                        int cols,
                                        T* __restrict dest,
                                        int rowsToCopy,
                                        int colsToCopy)
{
    static_assert(Nc % Nr == 0, "Invalid n pattern");
    static_assert(Kc % Kr == 0, "Invalid k pattern");

    int idx = 0;
    for (int j = 0; j < Nc; j += Nr)
    {
        for (int i = 0; i < Kc; i += Kr)
        {
            for (int ic = 0; ic < Kr; ++ic)
            {
                const int srcRow = i + ic;
                const int base   = srcRow * cols;
                for (int jc = 0; jc < Nr; ++jc)
                {
                    const int  srcCol = j + jc;
                    const bool inside = (srcRow < rowsToCopy) && (srcCol < colsToCopy);
                    dest[idx++]       = inside ? matrix[base + srcCol] : T(0);
                }
            }
        }
    }
}

// Padded variant: packs an Mc x Kc tile in Col-major micro-tiles (Mr x Kr),
// reading only rowsToCopy x colsToCopy elements from source and zero-filling the rest
template<int Mc, int Kc, int Mr, int Kr, typename T>
inline void reorderColOrderMatrixPadded(const T* __restrict matrix,
                                        int cols,
                                        T* __restrict dest,
                                        int rowsToCopy,
                                        int colsToCopy)
{
    static_assert(Mc % Mr == 0, "Invalid m pattern");
    static_assert(Kc % Kr == 0, "Invalid k pattern");

    int idx = 0;
    for (int i = 0; i < Mc; i += Mr)
    {
        for (int j = 0; j < Kc; j += Kr)
        {
            for (int jc = 0; jc < Kr; ++jc)
            {
                for (int ic = 0; ic < Mr; ++ic)
                {
                    const int  row    = i + ic;
                    const int  col    = j + jc;
                    const bool inside = (row < rowsToCopy) && (col < colsToCopy);
                    dest[idx++]       = inside ? matrix[row * cols + col] : T(0);
                }
            }
        }
    }
}

template<typename T>
void matMulZen5MTBlockingTails(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C)
{
    // std::cout << "Nc: " << Nc << ", Mc: " << Mc << ", Kc: " << Kc << std::endl;

    constexpr auto num_of_regs = 32;
    constexpr auto bregs_cnt   = 3;
    constexpr auto aregs_cnt   = 1;

    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;

    constexpr int Kr = 1;
    constexpr int Nr = bregs_cnt * num_of_elems_in_reg; // 24
    constexpr int Mr{8}; //{(num_of_regs - aregs_cnt - bregs_cnt) / bregs_cnt};

    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    auto num_threads = std::thread::hardware_concurrency();
    static_assert(Mc % Mr == 0, "invalid cache/reg size of the block");
    static_assert(Nc % Nr == 0, "invalid cache/reg size of the block");
    static_assert(Kc % Kr == 0, "invalid cache/reg size of the block");

    const int N = static_cast<int>(B.col());
    const int K = static_cast<int>(A.col());
    const int M = static_cast<int>(A.row());

    //
    // Minimal per-tail padding during repacking enables arbitrary M/N/K
    const std::size_t per_thread_buf_elems =
      static_cast<std::size_t>(Kc) * (Mc + Nc) + static_cast<std::size_t>(Mc) * Nc;
    std::vector<T, boost::alignment::aligned_allocator<T, PAGE_SIZE>> buffer(
      num_threads * per_thread_buf_elems);

    // Grid threading like matMulZen5MTBlocking
    constexpr int      GRID_I       = 4;
    constexpr int      GRID_J       = 8;
    constexpr unsigned grid_threads = GRID_I * GRID_J;

    // Square-chunking in block units; use ceil for tail tiles
    constexpr int ChunkBlocks = 2;
    const int     blocksI     = (M + Mc - 1) / Mc;
    const int     blocksJ     = (N + Nc - 1) / Nc;
    const int     chunkRows   = (blocksI + ChunkBlocks - 1) / ChunkBlocks;
    const int     chunkCols   = (blocksJ + ChunkBlocks - 1) / ChunkBlocks;

    auto worker_fn = [&](unsigned t)
    {
        const std::size_t ofs  = static_cast<std::size_t>(t) * per_thread_buf_elems;
        T* const          buf  = buffer.data() + ofs;
        T* const          bufA = buf;
        T* const          bufB = buf + Mc * Kc;
        T* const          bufC = buf + static_cast<std::size_t>(Kc) * (Mc + Nc);

        const int ti = static_cast<int>(t) / GRID_J;
        const int tj = static_cast<int>(t) % GRID_J;

        for (int chi = ti; chi < chunkRows; chi += GRID_I)
        {
            for (int chj = tj; chj < chunkCols; chj += GRID_J)
            {
                const int ibegin = chi * ChunkBlocks;
                const int iend   = std::min(ibegin + ChunkBlocks, blocksI);
                const int jbegin = chj * ChunkBlocks;
                const int jend   = std::min(jbegin + ChunkBlocks, blocksJ);

                for (int jb = jbegin; jb < jend; ++jb)
                {
                    const int j_block   = jb * Nc;
                    const int N_blk     = std::min(Nc, N - j_block);
                    const int N_blk_pad = blockWithPadding(N_blk, Nr);
                    for (int k_block = 0; k_block < K; k_block += Kc)
                    {
                        const int K_blk = std::min(Kc, K - k_block);

                        reorderRowMajorMatrixPadded<Kc, Nc, Kr, Nr>(
                          B.data() + static_cast<std::size_t>(N) * k_block + j_block,
                          N,
                          bufB,
                          K_blk,
                          N_blk);

                        for (int ib = ibegin; ib < iend; ++ib)
                        {
                            const int i_block   = ib * Mc;
                            const int M_blk     = std::min(Mc, M - i_block);
                            const int M_blk_pad = blockWithPadding(M_blk, Mr);

                            reorderColOrderMatrixPadded<Mc, Kc, Mr, Kr>(
                              A.data() + static_cast<std::size_t>(K) * i_block + k_block,
                              K,
                              bufA,
                              M_blk,
                              K_blk);

                            const std::size_t c_tile_elems =
                              static_cast<std::size_t>(M_blk_pad) * N_blk_pad;
                            std::fill(bufC, bufC + c_tile_elems, T(0));

                            for (int j = 0; j < N_blk_pad; j += Nr)
                            {
                                const T* Bc1 = bufB + Kc * j;
                                for (int i = 0; i < M_blk_pad; i += Mr)
                                {
                                    T* Cc0 = bufC + static_cast<std::size_t>(N_blk_pad) * i + j;
                                    const T* Ac0 = bufA + Kc * i;

                                    zen5_packed_kernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N_blk_pad);
                                }
                            }

                            for (int ii = 0; ii < M_blk; ++ii)
                            {
                                T* c_row =
                                  C.data() + static_cast<std::size_t>(N) * (i_block + ii) + j_block;
                                const T* tile_row = bufC + static_cast<std::size_t>(N_blk_pad) * ii;
                                for (int jj = 0; jj < N_blk; ++jj)
                                {
                                    c_row[jj] += tile_row[jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    std::vector<std::jthread> workers;
    workers.reserve(grid_threads - 1);
    for (unsigned t = 0; t + 1 < grid_threads; ++t)
    {
        workers.emplace_back([&, t]() { worker_fn(t); });
    }
    worker_fn(grid_threads - 1);
}

} // namespace
void matMulAutotune(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    matMulZen5MTBlockingTails(A, B, C);
}