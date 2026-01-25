#pragma once

#include <cstddef> // for size_t

#include <xsimd/xsimd.hpp>
#include <immintrin.h>
#include <mdspan>

namespace xkernels
{

// 1. Define a scalar wrapper that mimics xsimd::batch interface
template<typename T>
struct ScalarBatch
{
    T value;

    ScalarBatch() = default;
    ScalarBatch(T v)
      : value(v)
    {
    }

    static ScalarBatch load_unaligned(const T* ptr)
    {
        return ScalarBatch(*ptr);
    }

    void store_unaligned(T* ptr) const
    {
        *ptr = value;
    }

    ScalarBatch& operator+=(const ScalarBatch& other)
    {
        value += other.value;
        return *this;
    }

    ScalarBatch operator*(const ScalarBatch& other) const
    {
        return ScalarBatch(value * other.value);
    }
};

// 2. Create a helper trait to select between xsimd batch and ScalarBatch
template<typename T, int WIDTH>
struct fix_simd_helper
{
    using type = xsimd::make_sized_batch_t<T, WIDTH>;
};

// Specialization for WIDTH=1 uses ScalarBatch
template<typename T>
struct fix_simd_helper<T, 1>
{
    using type = ScalarBatch<T>;
};

// 3. Define the alias using the helper
template<typename T, int WIDTH>
using fix_simd = typename fix_simd_helper<T, WIDTH>::type;

// template<typename T, int WIDTH>
// using fix_simd = xsimd::make_sized_batch_t<T, WIDTH>;

////////////////////////////     SIMD KERNELS

template<typename T, int WIDTH>
__attribute__((always_inline)) static inline void load_inc_store_double(
  T* __restrict ptr,
  fix_simd<T, WIDTH> increment)
{
    auto vector = fix_simd<T, WIDTH>::load_unaligned(ptr);
    vector += increment;
    vector.store_unaligned(ptr);
}

template<std::size_t RowIdx, typename T, int WIDTH, std::size_t... I>
__attribute__((always_inline)) static inline void store_row(T*                  c,
                                                            fix_simd<T, WIDTH>* r,
                                                            std::index_sequence<I...>)
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
__attribute__((always_inline)) static inline void packed_compute_kernel(const T*            a,
                                                                        const T*            b,
                                                                        fix_simd<T, WIDTH>* r)
{
    packed_compute_kernel<T, WIDTH>(
      a, b, r, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
}

//////////////////////////////// NOT PACKED, GENERIC UTILS

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
     (compute_row<T, WIDTH>(
       fix_simd<T, WIDTH>(a[I * K + k]), bs, &r[I * Nrs], std::make_index_sequence<Nrs>{})));
}

template<int Mr, int Nrs, typename T, int WIDTH>
static inline void compute_kernel(const T* a, const T* b, fix_simd<T, WIDTH>* r, int K, int k)
{
    compute_kernel<T, WIDTH>(
      a, b, r, K, k, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
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
        compute_kernel<T, 4>(
          a, b, r, K, k2, std::make_index_sequence<Mr>{}, std::make_index_sequence<Nrs>{});
    }

    store_kernel<Nrs, T, 4>(c, r, N, std::make_index_sequence<Mr>{});
}

template<int Nr, int Mr, int Kc, typename T>
inline void cpp_generic_ukern(const T* __restrict a,
                              const T* __restrict b,
                              T* __restrict c,
                              int N,
                              int K)
    requires(Nr == 2 or Nr == 1)
{
    // 2 double: 16 bytes,  128 bit, need to choose simd128
    // 1 double: 8 bytes, no simd needed

    // 2 floats: 8 bytes, 64 bit, no simd needed
    // 1 float: 4 bytes, 32 bit, no simd needed

    // TODO: Fix me
    constexpr int   Nrs{1};
    fix_simd<T, Nr> r[Nrs * Mr]{};

    for (int k2 = 0; k2 < Kc; ++k2, b += N)
    {
        compute_kernel<Mr, Nrs, T, Nr>(a, b, r, K, k2);
    }

    store_kernel<Nrs, Mr, T, Nr>(c, r, N);
}

template<int Nr, int Mr, typename T>
__attribute__((always_inline)) static inline void cpp_packed_kernel(const T* __restrict a,
                                                                    const T* __restrict b,
                                                                    T* __restrict c,
                                                                    int N,
                                                                    int Kc)
    requires(Nr % 4 == 0)
{
    // TODO:: Fixme
    // avx256 for backward compart with haswell implementation
    constexpr auto num_of_elems_in_reg = 4; // xsimd::batch<T>::size;
    static_assert(num_of_elems_in_reg == 4, "Nr must be divisible by num_of_elems_in_reg");

    constexpr int Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        // TODO: why can't deduce width?
        packed_compute_kernel<Mr, Nrs, T, num_of_elems_in_reg>(a, b, r);
    }
    store_kernel<Nrs, Mr, T, num_of_elems_in_reg>(c, r, N);
}

template<int Nr, int Mr, typename T>
__attribute__((always_inline)) static inline void cpp_packed_kernel(const T* __restrict a,
                                                                    const T* __restrict b,
                                                                    T* __restrict c,
                                                                    int N,
                                                                    int Kc)
    requires(Nr == 2 or Nr == 1)
{

    constexpr int num_b_registers{1};

    fix_simd<T, Nr> r[num_b_registers * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        // TODO: why can't deduce width?
        packed_compute_kernel<Mr, num_b_registers, T, Nr>(a, b, r);
    }
    store_kernel<num_b_registers, Mr, T, Nr>(c, r, N);
}

/// ZEN 5
template<int Nr, int Mr, typename T>
__attribute__((always_inline)) static inline void zen5_packed_kernel(const T* __restrict a,
                                                                     const T* __restrict b,
                                                                     T* __restrict c,
                                                                     int N,
                                                                     int Kc)
{
    // TODO: Check : std::min(Nr, xsimd::batch<T>::size)
    constexpr auto num_of_elems_in_reg = xsimd::batch<T>::size;

    constexpr int Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k, b += Nr, a += Mr)
    {
        // TODO: why can't deduce width?
        packed_compute_kernel<Mr, Nrs, T, num_of_elems_in_reg>(a, b, r);
    }
    store_kernel<Nrs, Mr, T, num_of_elems_in_reg>(c, r, N);
}

template<std::size_t Nr, std::size_t Mr, std::size_t Kc, typename T>
static inline void zen5_mdspan_kernel(const std::mdspan<T, std::extents<std::size_t, Kc, Mr>> a,
                                      const std::mdspan<T, std::extents<std::size_t, Kc, Nr>> b,
                                      T* __restrict c,
                                      int N) noexcept
{
    constexpr auto num_of_elems_in_reg = xsimd::batch<T>::size;
    constexpr int  Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k)
    {
        packed_compute_kernel<Mr, Nrs, T, num_of_elems_in_reg>(&a[k, 0], &b[k, 0], r);
    }
    store_kernel<Nrs, Mr, T, num_of_elems_in_reg>(c, r, N);
}

} // namespace xkernels
