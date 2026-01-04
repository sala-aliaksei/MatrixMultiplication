#pragma once

#include <cstddef> // for size_t

#include <xsimd/xsimd.hpp>
#include <immintrin.h>


namespace xkernels
{

////////////////////////////     SIMD KERNELS

template<typename T, int WIDTH>
using fix_simd = xsimd::make_sized_batch_t<T, WIDTH>;

template<typename T, int WIDTH>
__attribute__((always_inline))static inline void load_inc_store_double(T* __restrict ptr, fix_simd<T, WIDTH> increment)
{
    auto vector = fix_simd<T, WIDTH>::load_unaligned(ptr);
    vector += increment;
    vector.store_unaligned(ptr);
}

template<std::size_t RowIdx, typename T, int WIDTH, std::size_t... I>
__attribute__((always_inline))static inline void store_row(T* c, fix_simd<T, WIDTH>* r, std::index_sequence<I...>)
{
    (..., (load_inc_store_double<T, WIDTH>(&c[I * WIDTH], r[RowIdx * sizeof...(I) + I])));
}

template<int Nrs, typename T, int WIDTH, std::size_t... RowIndices>
__attribute__((always_inline)) static inline void store_kernel(T*                  c,
                                fix_simd<T, WIDTH>* r,
                                int                 N,
                                std::index_sequence<RowIndices...>)
{
    (..., (store_row<RowIndices, T, WIDTH>(c, r, std::make_index_sequence<Nrs>{}), c += N));
}

template<int Nrs, int Mr, typename T, int WIDTH>
__attribute__((always_inline)) static inline void store_kernel(T* c, fix_simd<T, WIDTH>* r, int N)
{
    _mm_prefetch(c + N, _MM_HINT_NTA);
    store_kernel<Nrs, T, WIDTH>(c, r, N, std::make_index_sequence<Mr>{});
}


//////////      GENERIC PACKED   ///////////////////////////////////

template<typename T, int WIDTH, std::size_t... J>
__attribute__((always_inline)) static inline void packed_compute_row(const fix_simd<T, WIDTH>& a,
                                      fix_simd<T, WIDTH>*       b,
                                      fix_simd<T, WIDTH>*       r,
                                      std::index_sequence<J...>)
{
    (..., (r[J] += a * b[J]));
}

template<typename T, int WIDTH, size_t... I, size_t... J>
__attribute__((always_inline)) static inline void packed_compute_kernel(const T*            a,
                                         const T*            b,
                                         fix_simd<T, WIDTH>* r,
                                         std::index_sequence<I...>,
                                         std::index_sequence<J...>)
{
    constexpr int Nrs = sizeof...(J);
    // constexpr int Mrs = sizeof...(I);
    //  Nrs*Mrs - size of r array

    fix_simd<T, WIDTH> bs[Nrs] = {fix_simd<T, WIDTH>::load_unaligned(&b[J * WIDTH])...};
    (...,
     (packed_compute_row<T, WIDTH>(
       fix_simd<T, WIDTH>(a[I]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
}

template<int Mr, int Nrs, typename T, int WIDTH>
__attribute__((always_inline)) static inline void packed_compute_kernel(const T* a, const T* b, fix_simd<T, WIDTH>* r)
{
    packed_compute_kernel<T, WIDTH>(a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
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

    fix_simd<T, WIDTH> bs[Nrs] = {fix_simd<T, WIDTH>::load_unaligned(&b[J * WIDTH])...};
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
    fix_simd<T, 4> r[Nrs * Mr]{};


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
    fix_simd<T, Nr> r[Nrs * Mr]{};


    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        compute_kernel(
          a, b, r, K, k2, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}



template<int Nr, int Mr, int Kc, typename T>
static inline void ukern(const T* __restrict a,
                         const T* __restrict b,
                         T* __restrict c,
                         int N,
                         int K)

{
    constexpr auto num_of_elems_in_reg = xsimd::batch<T>::size;
    constexpr int  Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr]{};


    for (int k = 0; k < Kc; ++k, b += N)
    {
        compute_kernel(
          a, b, r, K, k, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nrs>(c, r, N, std::make_index_sequence<Mr>{});
}


/// ZEN 5 
template<int Nr, int Mr, typename T>
__attribute__((always_inline)) static inline void zen5_packed_kernel(const T* __restrict a,
                                      const T* __restrict b,
                                      T* __restrict c,
                                      int N,
                                      int Kc)
{
    constexpr auto num_of_elems_in_reg = xsimd::batch<T>::size;
    static_assert( num_of_elems_in_reg == 8, "Nr must be divisible by num_of_elems_in_reg");

    constexpr int  Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        // TODO: why can't deduce width?
        packed_compute_kernel<Mr, Nrs,T, num_of_elems_in_reg>(a, b, r);
    }
    store_kernel<Nrs, Mr,T, num_of_elems_in_reg>(c, r, N);
}

} // namespace kernels
