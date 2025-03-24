#include "mm/matmul/matMulSimd.hpp"
#include "mm/core/reorderMatrix.hpp"

// #include "mm/core/kernels.hpp"

#include <experimental/simd>
// TODO: Check with mdspan

#include <mdspan/mdspan.hpp>

#include "omp.h"

namespace stdx = std::experimental;

using simd_d = stdx::native_simd<double>;

template<typename T>
using simd = stdx::native_simd<T>;

template<typename T, int Mc, int Kc>
using tile = Kokkos::mdspan<double, Kokkos::extents<int, Mc, Kc>>;

template<typename T, int Mc, int Kc>
using ctile = Kokkos::mdspan<double, Kokkos::extents<int, Mc, Kc>>;

static_assert(simd<double>::size() == 4, "Expect 4 doubles per simd register");

static inline void load_inc_store_double(double* __restrict ptr, simd_d increment)
{
    simd_d vector(ptr, stdx::vector_aligned);
    vector += increment;
    vector.copy_to(ptr, stdx::vector_aligned);
}

template<int Nr, int Mr, int Kc>
static void cpp_upkernelSpan(tile<double, Kc, Mr> a,
                             tile<double, Kc, Nr> b,
                             double* __restrict mc,
                             int N)
{

    double* c     = mc;
    simd_d  r[Nr] = {};

    for (int k = 0; k < Kc; ++k)
    {
        simd_d b0(&b[k, 0], stdx::element_aligned);
        simd_d b1(&b[k, 4], stdx::element_aligned);
        simd_d b2(&b[k, 8], stdx::element_aligned);

        simd_d a0(a[k, 0]);
        r[0] += a0 * b0;
        r[1] += a0 * b1;
        r[2] += a0 * b2;

        a0 = simd_d(a[k, 1]);
        r[3] += a0 * b0;
        r[4] += a0 * b1;
        r[5] += a0 * b2;

        a0 = simd_d(a[k, 2]);
        r[6] += a0 * b0;
        r[7] += a0 * b1;
        r[8] += a0 * b2;

        a0 = simd_d(a[k, 3]);
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

    const double* a = ma;

    static_assert(Kc % 2 == 0, "Kc must be even for manual unrolling");
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
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

    load_inc_store_double(&c[0], r00);
    load_inc_store_double(&c[4], r01);
    load_inc_store_double(&c[8], r02);
    c += N;

    load_inc_store_double(&c[0], r10);
    load_inc_store_double(&c[4], r11);
    load_inc_store_double(&c[8], r12);
    c += N;

    load_inc_store_double(&c[0], r20);
    load_inc_store_double(&c[4], r21);
    load_inc_store_double(&c[8], r22);
    c += N;

    load_inc_store_double(&c[0], r30);
    load_inc_store_double(&c[4], r31);
    load_inc_store_double(&c[8], r32);
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

// template<typename T, std::size_t... RowIndices>
// void store_kernel(T* c, simd<T>* r, int N, std::index_sequence<RowIndices...>)
//{
//     (...,
//      (load_inc_store_double(&c[0], r[RowIndices * 3 + 0]),
//       load_inc_store_double(&c[4], r[RowIndices * 3 + 1]),
//       load_inc_store_double(&c[8], r[RowIndices * 3 + 2]),
//       c += N // Move to the next row after each triplet
//       ));
// }

// template<size_t Nr, typename T, size_t... I>
// void compute_block(const T* a, const T* b, simd<T>* r, std::index_sequence<I...>)
//{
//     // b range (0,2)
//     // a range (0,3)
//     constexpr int Mr = sizeof...(I);
//     constexpr int Br = Nr / Mr;

//    simd<T> bs[Mr] = {simd<T>(&b[I * simd<T>::size()], std::experimental::element_aligned)...};

//    (..., (compute_row(simd<T>(a[I]), bs, &r[I * Br], std::make_index_sequence<Br>{})));
//}

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
    (..., (store_row<RowIndices>(c, r, std::make_index_sequence<Nrs>{}), c += N));
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

template<int Nr, int Mr, int Kc, typename T>
static inline void cpp_ukernel(const T* __restrict a, const T* __restrict b, T* __restrict c, int N)
{
    constexpr int Nrs   = Nr / simd_d::size();
    simd_d        r[Nr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        compute_kernel(a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }
    // handle k tail, no tail if inc with 1

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
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

void matMulSimd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    // BEST
    //    constexpr int Nc = 720;
    //    constexpr int Mc = 180;
    //    constexpr int Kc = 240;

    // NEW BEST
    constexpr int Nc = 720;
    constexpr int Mc = 20;
    constexpr int Kc = 80;

    //    constexpr int Nc = 768;
    //    constexpr int Mc = 20;
    //    constexpr int Kc = 80;

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

    std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(4 * Kc
                                                                                  * (Mc + Nc));

#pragma omp parallel for
    for (int j_block = 0; j_block < N; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        for (int k_block = 0; k_block < K; k_block += Kc)
        {
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(
              B.data() + N * k_block + j_block, N, buf + Mc * Kc);

            for (int i_block = 0; i_block < M; i_block += Mc)
            {
                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K, buf);

                for (int j = 0; j < Nc; j += Nr)
                {
                    const double* Bc1 = buf + Mc * Kc + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = C.data() + N * i_block + j + N * i + j_block;
                        const double* Ac0 = buf + Kc * i;

                        cpp_ukernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // upkernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // upkernelAr<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                        // cpp_ukernelLambda<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                }
            }
        }
    }
}

void matMulSimdTails(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{

    // NEW BEST
    constexpr int Nc = 720;
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

    std::vector<double, boost::alignment::aligned_allocator<double, 4096>> buffer(4 * Kc
                                                                                  * (Mc + Nc));

#pragma omp parallel for
    for (int j_block = 0; j_block < N; j_block += Nc)
    {
        auto       tid = omp_get_thread_num();
        const auto ofs = tid * Kc * (Mc + Nc);
        double*    buf = buffer.data() + ofs;

        for (int k_block = 0; k_block < K; k_block += Kc)
        {
            // Can be access out of bound if j+Nc > N
            reorderRowMajorMatrix<Kc, Nc, Kr, Nr>(
              B.data() + N * k_block + j_block, N, buf + Mc * Kc);

            for (int i_block = 0; i_block < M; i_block += Mc)
            {
                // Can be access out of bound if i+Mc > M
                // how to we reorder if there is a tail?
                reorderColOrderMatrix<Mc, Kc, Mr, Kr>(A.data() + K * i_block + k_block, K, buf);

                for (int j = 0; j < Nc; j += Nr)
                {
                    const double* Bc1 = buf + Mc * Kc + Kc * j;
                    for (int i = 0; i < Mc; i += Mr)
                    {
                        double*       Cc0 = C.data() + N * i_block + j + N * i + j_block;
                        const double* Ac0 = buf + Kc * i;

                        // TODO: deduce args from span?
                        cpp_ukernel<Nr, Mr, Kc>(Ac0, Bc1, Cc0, N);
                    }
                    // compute i tail; 4,2,1
                    // auto c = C(i+i_last_block,j+j_block);
                    //  if( i>=2)  cpp_ukernel<Nr, 2, Kc>(Ac0, Bc1, c, N);
                    //  else  cpp_ukernel<Nr, 1, Kc>(Ac0, Bc1, c, N);
                }
                // compute j tail
                {
                    // if( j>=8)
                    // if( j>=6)
                    // if( j>=4)
                    // if( j>=2)
                    // else //1
                }
            }
        }
    }
}
