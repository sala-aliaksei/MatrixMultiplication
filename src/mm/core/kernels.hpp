#pragma once

#include <cstddef> // for size_t

#include <experimental/simd>

#include <immintrin.h>

namespace stdx = std::experimental;

namespace kernels
{

////////////////////////////     SIMD KERNELS

using simd_d     = stdx::fixed_size_simd<double, 4>;
using halfsimd_d = stdx::fixed_size_simd<double, 2>;

using simd_to_double = stdx::fixed_size_simd<double, 1>;

// template<typename T>
// using simd = stdx::native_simd<T>;

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
    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

//////////////////////////////    PACKED , MANUL (NOT GENERIC)
///
template<int Kc>
static void packed_ukernel8x4(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict mc,
                              int N)
{
    constexpr int Nr = 8;
    constexpr int Mr = 4;
    static_assert(Nr == 8);
    static_assert(Mr == 4);

    double* c     = mc;
    simd_d  r[Nr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d b0(&b[0], stdx::element_aligned);
        simd_d b1(&b[4], stdx::element_aligned);

        simd_d a0(a[0]);
        r[0] += a0 * b0;
        r[1] += a0 * b1;

        a0 = simd_d(a[1]);
        r[2] += a0 * b0;
        r[3] += a0 * b1;

        a0 = simd_d(a[2]);
        r[4] += a0 * b0;
        r[5] += a0 * b1;

        a0 = simd_d(a[3]);
        r[6] += a0 * b0;
        r[7] += a0 * b1;
    }

    load_inc_store_double(&c[0], r[0]);
    load_inc_store_double(&c[4], r[1]);

    c += N;

    load_inc_store_double(&c[0], r[2]);
    load_inc_store_double(&c[4], r[3]);

    c += N;

    load_inc_store_double(&c[0], r[4]);
    load_inc_store_double(&c[4], r[5]);

    c += N;

    load_inc_store_double(&c[0], r[6]);
    load_inc_store_double(&c[4], r[7]);
}

template<int Kc>
static void packed_ukernel8x4_more_a_regs(const double* __restrict ma,
                                          const double* __restrict b,
                                          double* __restrict mc,
                                          int N)
{
    constexpr int Nr = 8;
    constexpr int Mr = 4;
    static_assert(Nr == 8);
    static_assert(Mr == 4);

    double* c     = mc;
    simd_d  r[Nr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d b0(&b[0], stdx::element_aligned);
        simd_d b1(&b[4], stdx::element_aligned);

        simd_d a0(a[0]);
        r[0] += a0 * b0;
        r[1] += a0 * b1;

        simd_d a1(a[1]);
        r[2] += a1 * b0;
        r[3] += a1 * b1;

        simd_d a2(a[2]);
        r[4] += a2 * b0;
        r[5] += a2 * b1;

        simd_d a3(a[3]);
        r[6] += a3 * b0;
        r[7] += a3 * b1;
    }

    load_inc_store_double(&c[0], r[0]);
    load_inc_store_double(&c[4], r[1]);

    c += N;

    load_inc_store_double(&c[0], r[2]);
    load_inc_store_double(&c[4], r[3]);

    c += N;

    load_inc_store_double(&c[0], r[4]);
    load_inc_store_double(&c[4], r[5]);

    c += N;

    load_inc_store_double(&c[0], r[6]);
    load_inc_store_double(&c[4], r[7]);
}

template<int Kc>
static void packed_ukernel6x4(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict mc,
                              int N)
{

    constexpr int Nr = 6;
    constexpr int Mr = 4;

    double*    c      = mc;
    simd_d     r[Mr]  = {};
    halfsimd_d rh[Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d     b0(&b[0], stdx::element_aligned);
        halfsimd_d b1(&b[4], stdx::element_aligned);

        simd_d     a0(a[0]);
        halfsimd_d a1(a[0]);
        r[0] += a0 * b0;
        rh[0] += a1 * b1;

        a0 = simd_d(a[1]);
        a1 = halfsimd_d(a[1]);
        r[1] += a0 * b0;
        rh[1] += a1 * b1;

        a0 = simd_d(a[2]);
        a1 = halfsimd_d(a[2]);
        r[2] += a0 * b0;
        rh[2] += a1 * b1;

        a0 = simd_d(a[3]);
        a1 = halfsimd_d(a[3]);
        r[3] += a0 * b0;
        rh[3] += a1 * b1;
    }

    load_inc_store_double(&c[0], r[0]);
    load_inc_store_double(&c[4], rh[0]);

    c += N;

    load_inc_store_double(&c[0], r[1]);
    load_inc_store_double(&c[4], rh[1]);

    c += N;

    load_inc_store_double(&c[0], r[2]);
    load_inc_store_double(&c[4], rh[2]);

    c += N;

    load_inc_store_double(&c[0], r[3]);
    load_inc_store_double(&c[4], rh[3]);
}

template<int Kc>
static void packed_ukernel4x4(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict mc,
                              int N)
{
    constexpr int Nr = 4;
    constexpr int Mr = 4;

    double* c              = mc;
    simd_d  r[Nr / 4 * Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d b0(&b[0], stdx::element_aligned);

        simd_d a0(a[0]);
        r[0] += a0 * b0;

        a0 = simd_d(a[1]);
        r[1] += a0 * b0;

        a0 = simd_d(a[2]);
        r[2] += a0 * b0;

        a0 = simd_d(a[3]);
        r[3] += a0 * b0;
    }

    load_inc_store_double(&c[0], r[0]);

    c += N;

    load_inc_store_double(&c[0], r[1]);

    c += N;

    load_inc_store_double(&c[0], r[2]);

    c += N;

    load_inc_store_double(&c[0], r[3]);
}

template<int Kc>
static void packed_ukernel4x2(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict mc,
                              int N)
{
    constexpr int Nr = 4;
    constexpr int Mr = 2;

    double* c              = mc;
    simd_d  r[Nr / 4 * Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d b0(&b[0], stdx::element_aligned);

        simd_d a0(a[0]);
        r[0] += a0 * b0;

        a0 = simd_d(a[1]);
        r[1] += a0 * b0;
    }

    load_inc_store_double(&c[0], r[0]);

    c += N;

    load_inc_store_double(&c[0], r[1]);
}

template<int Kc>
static void packed_ukernel4x1(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict c,
                              int N)
{
    constexpr int Nr = 4;
    constexpr int Mr = 1;

    simd_d r[Nr / 4 * Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_d b0(&b[0], stdx::element_aligned);
        simd_d a0(a[0]);
        r[0] += a0 * b0;
    }

    load_inc_store_double(&c[0], r[0]);
}

template<int Kc>
static void packed_ukernel2x4(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict mc,
                              int N)
{
    constexpr int Nr = 2;
    constexpr int Mr = 4;

    double*    c     = mc;
    halfsimd_d r[Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        halfsimd_d b0(&b[0], stdx::element_aligned);

        halfsimd_d a0(a[0]);
        r[0] += a0 * b0;

        a0 = halfsimd_d(a[1]);
        r[1] += a0 * b0;

        a0 = halfsimd_d(a[2]);
        r[2] += a0 * b0;

        a0 = halfsimd_d(a[3]);
        r[3] += a0 * b0;
    }

    load_inc_store_double(&c[0], r[0]);

    c += N;

    load_inc_store_double(&c[0], r[1]);

    c += N;

    load_inc_store_double(&c[0], r[2]);

    c += N;

    load_inc_store_double(&c[0], r[3]);
}

template<int Kc>
static void packed_ukernel1x4_simd(const double* __restrict ma,
                                   const double* __restrict b,
                                   double* __restrict mc,
                                   int N)
{
    constexpr int Nr = 1;
    constexpr int Mr = 4;

    double*        c     = mc;
    simd_to_double r[Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        simd_to_double b0(&b[0], stdx::element_aligned);

        simd_to_double a0(a[0]);
        r[0] += a0 * b0;

        a0 = simd_to_double(a[1]);
        r[1] += a0 * b0;

        a0 = simd_to_double(a[2]);
        r[2] += a0 * b0;

        a0 = simd_to_double(a[3]);
        r[3] += a0 * b0;
    }

    load_inc_store_double(&c[0], r[0]);

    c += N;

    load_inc_store_double(&c[0], r[1]);

    c += N;

    load_inc_store_double(&c[0], r[2]);

    c += N;

    load_inc_store_double(&c[0], r[3]);
}

template<int Kc>
static void packed_ukernel1x4(const double* __restrict ma,
                              const double* __restrict b,
                              double* __restrict mc,
                              int N)
{
    constexpr int Nr = 1;
    constexpr int Mr = 4;

    double* c     = mc;
    double  r[Mr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        r[0] += a[0] * b[0];
        r[1] += a[1] * b[0];
        r[2] += a[2] * b[0];
        r[3] += a[3] * b[0];
    }

    c[0] += r[0];

    c += N;

    c[0] += r[1];

    c += N;

    c[0] += r[2];

    c += N;

    c[0] += r[3];
}

//////////      GENERIC PACKED   ///////////////////////////////////

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

// Same perf as manual impl for Nr = 12, 8, 4;
template<int Nr, int Mr, int Kc, typename T>
static inline void cpp_packed_kernel(const T* __restrict a,
                                     const T* __restrict b,
                                     T* __restrict c,
                                     int N)
    requires(Nr % 4 == 0)
{
    constexpr int Nrs{Nr / 4};

    fix_simd<T, 4> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        packed_compute_kernel(
          a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }
    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

template<int Nr, int Mr, int Kc, typename T>
inline void cpp_packed_kernel(const T* __restrict ma, const T* __restrict b, T* __restrict c, int N)
    requires(Nr == 6 && Mr == 4)
{
    // TODO: IT is not packed!!!
    static_assert(std::is_same_v<T, double>);
    packed_ukernel6x4<Kc>(ma, b, c, N);
}

template<int Nr, int Mr, int Kc, typename T>
static inline void cpp_packed_kernel(const T* __restrict a,
                                     const T* __restrict b,
                                     T* __restrict c,
                                     int N)
    requires(Nr == 2 or Nr == 1)
{
    constexpr int Nrs = 1;

    fix_simd<T, Nr> r[Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        packed_compute_kernel(
          a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

//////////////////////////////// NOT PACKED, GENERIC

template<typename T, int WIDTH, std::size_t... J>
static inline void compute_row(const fix_simd<T, WIDTH>& a,
                               fix_simd<T, WIDTH>*       b,
                               fix_simd<T, WIDTH>*       r,
                               std::index_sequence<J...>)
{
    (..., (r[J] += a * b[J]));
}

template<typename T, int WIDTH, size_t... I, size_t... J>
static inline void compute_kernel(const T*            a,
                                  const T*            b,
                                  fix_simd<T, WIDTH>* r,
                                  int                 K,
                                  int                 k,
                                  std::index_sequence<I...>,
                                  std::index_sequence<J...>)
{
    constexpr int Nrs = sizeof...(J);

    fix_simd<T, WIDTH> bs[Nrs] = {fix_simd<T, WIDTH>(&b[J * WIDTH], stdx::element_aligned)...};
    (...,
     (compute_row(
       fix_simd<T, WIDTH>(a[I * K + k]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
}

template<int Nr, int Mr, int Kc, typename T>
inline void cpp_generic_ukern(const T* __restrict a,
                              const T* __restrict b,
                              T* __restrict c,
                              int N,
                              int K)
    requires(Nr % 4 == 0)
{
    constexpr int  Nrs{Nr / 4};
    fix_simd<T, 4> r[Nrs * Mr] = {};

    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        compute_kernel(
          a, b, r, K, k2, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

template<int Nr, int Mr, int Kc, typename T>
inline void cpp_generic_ukern(const T* __restrict a,
                              const T* __restrict b,
                              T* __restrict c,
                              int N,
                              int K)
    requires(Nr == 2 or Nr == 1)
{
    constexpr int   Nrs{1};
    fix_simd<T, Nr> r[Nrs * Mr] = {};

    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        compute_kernel(
          a, b, r, K, k2, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}

//////////////     EXAMPLE of array
template<int Nr, int Mr, int Kc>
static void ukernelAr(const double* __restrict ma,
                      const double* __restrict b,
                      double* __restrict mc,
                      int N)
{
    static_assert(Nr == 12);
    static_assert(Mr == 4);

    double* c     = mc;
    simd_d  r[Nr] = {};

    const double* a = ma;
    for (int k = 0; k < Kc; ++k, b += N)
    {
        a = ma;
        simd_d b0(&b[0], stdx::element_aligned);
        simd_d b1(&b[4], stdx::element_aligned);
        simd_d b2(&b[8], stdx::element_aligned);

        simd_d a0(a[k]);
        r[0] += a0 * b0;
        r[1] += a0 * b1;
        r[2] += a0 * b2;
        a += N;

        a0 = simd_d(a[k]);
        r[3] += a0 * b0;
        r[4] += a0 * b1;
        r[5] += a0 * b2;
        a += N;

        a0 = simd_d(a[k]);
        r[6] += a0 * b0;
        r[7] += a0 * b1;
        r[8] += a0 * b2;
        a += N;

        a0 = simd_d(a[k]);
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

template<int Nr, int Mr, int Kc, typename T>
static inline void zen5_packed_kernel(const T* __restrict a,
                                      const T* __restrict b,
                                      T* __restrict c,
                                      int N)
{
    constexpr auto num_of_regs    = 32;
    constexpr int  avx_width_bits = 512;

    constexpr int num_of_elems_in_reg = avx_width_bits / 8 / sizeof(T);

    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    constexpr int Nrs{Nr / num_of_elems_in_reg};

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        packed_compute_kernel<Mr, Nrs>(a, b, r);
    }
    store_kernel<Nrs, Mr>(c, r, N);
}

} // namespace kernels
