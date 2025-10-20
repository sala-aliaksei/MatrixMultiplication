#pragma once

#include <mm/core/kernels.hpp>
#include <mdspan>

namespace kernels
{

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

template<std::size_t Nr, std::size_t Mr, std::size_t Kc, typename T>
static inline void zen5_mdspan_kernel(const std::mdspan<T, std::extents<std::size_t, Kc, Mr>> a,
                                      const std::mdspan<T, std::extents<std::size_t, Kc, Nr>> b,
                                      T* __restrict c,
                                      int N) noexcept
{
    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;
    constexpr int  Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k)
    {
        packed_compute_kernel<Mr, Nrs>(&a[k, 0], &b[k, 0], r);
    }
    store_kernel<Nrs, Mr>(c, r, N);
}

template<std::size_t Nr, std::size_t Mr, std::size_t Kc, typename T>
static inline void zen5_mdspan_kernel(
  const std::mdspan<T, std::extents<std::size_t, Kc, Mr>>           a,
  const std::mdspan<T, std::extents<std::size_t, Kc, Nr>>           b,
  std::mdspan<T, std::dextents<std::size_t, 2>, std::layout_stride> c) noexcept
{
    constexpr auto num_of_elems_in_reg = stdx::simd_size_v<T, stdx::simd_abi::native<T>>;
    constexpr int  Nrs{Nr / num_of_elems_in_reg};
    static_assert(Nr % num_of_elems_in_reg == 0, "Nr must be divisible by num_of_elems_in_reg");

    fix_simd<T, num_of_elems_in_reg> r[Nrs * Mr] = {};
    for (int k = 0; k < Kc; ++k)
    {
        packed_compute_kernel<Mr, Nrs>(&a[k, 0], &b[k, 0], r);
    }
    store_kernel<Nrs, Mr>(c.data_handle(), r, 3072); //(int)c.extent(1)
}
} // namespace kernels