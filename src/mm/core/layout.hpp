#pragma once

#include <mm/core/Matrix.hpp>
#include <cstddef>
#include <emmintrin.h>
#include <mdspan>
#include <array>

namespace mm::core
{
enum class TileLayout
{
    ColMajor,
    RowMajor,
};

// Layout that matches reorderColOrderMatrix access pattern
// Iteration order within a Mc x Nc tile:
//   for (i in 0..Mc step Mr)
//     for (j in 0..Nc step Nr)
//       for (jc in 0..Nr-1)
//         for (ic in 0..Mr-1)
//           idx++  // reading matrix[(i+ic) * N + (j+jc)]
template<std::size_t Mr, std::size_t Nr>
struct layout_microtile_colorder
{
    template<class Extents>
    class mapping
    {
      public:
        using extents_type = Extents;
        using layout_type  = layout_microtile_colorder<Mr, Nr>;
        using index_type   = std::size_t;
        using size_type    = std::size_t;

        constexpr mapping() noexcept = default;
        constexpr explicit mapping(const extents_type& e,
                                   std::size_t         M,
                                   std::size_t         N,
                                   std::size_t         iofs,
                                   std::size_t         jofs) noexcept
          : _extents(e)
          , _M(M)
          , _N(N)
          , _iofs(iofs)
          , _jofs(jofs)
        {
        }

        template<class OtherExtents>
        constexpr explicit mapping(const mapping<OtherExtents>& other) noexcept
          : _extents(other.extents())
        {
        }

        [[nodiscard]] constexpr const extents_type& extents() const noexcept
        {
            return _extents;
        }

        [[nodiscard]] constexpr index_type required_span_size() const noexcept
        {
            return _extents.static_extent(0) * _extents.static_extent(1);
        }

        template<class I0, class I1>
        [[nodiscard]] constexpr index_type operator()(I0 i0, I1 j0) const noexcept
        {
            const index_type ic = static_cast<index_type>(j0) % Mr;
            const index_type q  = static_cast<index_type>(j0) / Mr;
            const index_type jc = q % Nr;
            const index_type jb = q / Nr;

            const index_type row = _iofs + static_cast<index_type>(i0) * Mr + ic;
            const index_type col = _jofs + jb * Nr + jc;
            (void)_M; // bounds assumed valid by mdspan usage

            return row * _N + col;
        }

        // Traits
        static constexpr bool is_always_unique() noexcept
        {
            return true;
        }
        static constexpr bool is_always_strided() noexcept
        {
            return false;
        }
        static constexpr bool is_always_contiguous() noexcept
        {
            return false;
        }
        static constexpr bool is_always_exhaustive() noexcept
        {
            return false;
        }

        [[nodiscard]] constexpr bool is_unique() const noexcept
        {
            return true;
        }
        [[nodiscard]] constexpr bool is_strided() const noexcept
        {
            return false;
        }
        [[nodiscard]] constexpr bool is_contiguous() const noexcept
        {
            return false;
        }
        [[nodiscard]] constexpr bool is_exhaustive() const noexcept
        {
            return false;
        }

        [[nodiscard]] constexpr index_type stride(std::size_t) const noexcept
        {
            return 0;
        }

        friend constexpr bool operator==(const mapping&, const mapping&) = default;

      private:
        extents_type _extents{};
        std::size_t  _iofs{};
        std::size_t  _jofs{};
        std::size_t  _M{};
        std::size_t  _N{};
    };
};

template<std::size_t Mc, std::size_t Kc>
using shape_t = std::extents<std::size_t, Mc, Kc>;

// Layout that matches layout_blocked_colmajor access pattern

template<std::size_t Nr>
struct layout_blocked_colmajor
{
    template<class Extents>
    class mapping
    {
      public:
        using extents_type = Extents;
        using layout_type  = layout_blocked_colmajor<Nr>;
        using index_type   = std::size_t;
        using size_type    = std::size_t;

        constexpr mapping() noexcept = default;
        constexpr explicit mapping(const extents_type& e,
                                   std::size_t         M,
                                   std::size_t         N,
                                   std::size_t         iofs,
                                   std::size_t         jofs) noexcept
          : _extents(e)
          , _M(M)
          , _N(N)
          , _iofs(iofs)
          , _jofs(jofs)
        {
        }

        template<class OtherExtents>
        constexpr explicit mapping(const mapping<OtherExtents>& other) noexcept
          : _extents(other.extents())
        {
        }

        [[nodiscard]] constexpr const extents_type& extents() const noexcept
        {
            return _extents;
        }

        // Required span size reserves full tile capacity including tails (with padding)
        [[nodiscard]] constexpr index_type required_span_size() const noexcept
        {
            return _extents.static_extent(0) * _extents.static_extent(1);
        }

        template<class I0, class I1>
        [[nodiscard]] constexpr index_type operator()(I0 i0, I1 j0) const noexcept
        {
            const index_type i = j0 / Nr;
            const index_type j = i0 * Nr + j0 % Nr;

            return i * _N + j;
        }

        // Traits
        static constexpr bool is_always_unique() noexcept
        {
            return true;
        }
        static constexpr bool is_always_strided() noexcept
        {
            return false;
        }
        static constexpr bool is_always_contiguous() noexcept
        {
            return false;
        }
        static constexpr bool is_always_exhaustive() noexcept
        {
            return false; // uses padded capacity Mc*Nc per tile
        }

        [[nodiscard]] constexpr bool is_unique() const noexcept
        {
            return true;
        }
        [[nodiscard]] constexpr bool is_strided() const noexcept
        {
            return false;
        }
        [[nodiscard]] constexpr bool is_contiguous() const noexcept
        {
            return false;
        }
        [[nodiscard]] constexpr bool is_exhaustive() const noexcept
        {
            return false;
        }

        // Optional: expose stride when it would be queried conditionally
        [[nodiscard]] constexpr index_type stride(std::size_t) const noexcept
        {
            return 0;
        }

        friend constexpr bool operator==(const mapping&, const mapping&) = default;

      private:
        extents_type _extents{};
        std::size_t  _iofs{};
        std::size_t  _jofs{};
        std::size_t  _M{};
        std::size_t  _N{};
    };
};

namespace
{
    __attribute__((no_sanitize("coverage"))) static inline void*
    inline_memcpy(void* __restrict dst_, const void* __restrict src_, size_t size)
    {
        /// We will use pointer arithmetic, so char pointer will be used.
        /// Note that __restrict makes sense (otherwise compiler will reload data from memory
        /// instead of using the value of registers due to possible aliasing).
        char* __restrict dst       = reinterpret_cast<char* __restrict>(dst_);
        const char* __restrict src = reinterpret_cast<const char* __restrict>(src_);

        /// Standard memcpy returns the original value of dst. It is rarely used but we have to do
        /// it. If you use memcpy with small but non-constant sizes, you can call inline_memcpy
        /// directly for inlining and removing this single instruction.
        void* ret = dst;

    tail:
        /// Small sizes and tails after the loop for large sizes.
        /// The order of branches is important but in fact the optimal order depends on the
        /// distribution of sizes in your application. This order of branches is from the
        /// disassembly of glibc's code. We copy chunks of possibly uneven size with two overlapping
        /// movs. Example: to copy 5 bytes [0, 1, 2, 3, 4] we will copy tail [1, 2, 3, 4] first and
        /// then head [0, 1, 2, 3].
        if (size <= 16)
        {
            if (size >= 8)
            {
                /// Chunks of 8..16 bytes.
                __builtin_memcpy(dst + size - 8, src + size - 8, 8);
                __builtin_memcpy(dst, src, 8);
            }
            else if (size >= 4)
            {
                /// Chunks of 4..7 bytes.
                __builtin_memcpy(dst + size - 4, src + size - 4, 4);
                __builtin_memcpy(dst, src, 4);
            }
            else if (size >= 2)
            {
                /// Chunks of 2..3 bytes.
                __builtin_memcpy(dst + size - 2, src + size - 2, 2);
                __builtin_memcpy(dst, src, 2);
            }
            else if (size >= 1)
            {
                /// A single byte.
                *dst = *src;
            }
            /// No bytes remaining.
        }
        else
        {
            /// Medium and large sizes.
            if (size <= 128)
            {
                /// Medium size, not enough for full loop unrolling.

                /// We will copy the last 16 bytes.
                _mm_storeu_si128(
                  reinterpret_cast<__m128i*>(dst + size - 16),
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + size - 16)));

                /// Then we will copy every 16 bytes from the beginning in a loop.
                /// The last loop iteration will possibly overwrite some part of already copied last
                /// 16 bytes. This is Ok, similar to the code for small sizes above.
                while (size > 16)
                {
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst),
                                     _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
                    dst += 16;
                    src += 16;
                    size -= 16;
                }
            }
            else
            {
                /// Large size with fully unrolled loop.

                /// Align destination to 16 bytes boundary.
                size_t padding = (16 - (reinterpret_cast<size_t>(dst) & 15)) & 15;

                /// If not aligned - we will copy first 16 bytes with unaligned stores.
                if (padding > 0)
                {
                    __m128i head = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), head);
                    dst += padding;
                    src += padding;
                    size -= padding;
                }

                /// Aligned unrolled copy. We will use half of available SSE registers.
                /// It's not possible to have both src and dst aligned.
                /// So, we will use aligned stores and unaligned loads.
                __m128i c0, c1, c2, c3, c4, c5, c6, c7;

                while (size >= 128)
                {
                    c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 0);
                    c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 1);
                    c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 2);
                    c3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 3);
                    c4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 4);
                    c5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 5);
                    c6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 6);
                    c7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src) + 7);
                    src += 128;
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 0), c0);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 1), c1);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 2), c2);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 3), c3);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 4), c4);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 5), c5);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 6), c6);
                    _mm_store_si128((reinterpret_cast<__m128i*>(dst) + 7), c7);
                    dst += 128;

                    size -= 128;
                }

                /// The latest remaining 0..127 bytes will be processed as usual.
                goto tail;
            }
        }

        return ret;
    }
} // namespace

template<std::size_t Mc, std::size_t Kc, typename T>
std::mdspan<const T, shape_t<Mc, Kc>, std::layout_stride> submatrix(const Matrix<T>& m,
                                                                    std::size_t      i0,
                                                                    std::size_t      j0)
{
    auto mapping = std::layout_stride::mapping<shape_t<Mc, Kc>>(
      shape_t<Mc, Kc>{},
      std::array<std::size_t, 2>{static_cast<std::size_t>(m.col()), std::size_t{1}});
    return std::mdspan<const T, shape_t<Mc, Kc>, std::layout_stride>(&m(i0, j0), mapping);
}

template<typename T, std::size_t Kc, std::size_t Nc, std::size_t Nr>
void initBTile(
  std::mdspan<const T, std::extents<std::size_t, Kc, Nc>, std::layout_stride> submatrix,
  std::mdspan<T, std::extents<std::size_t, Kc, Nr>>                           utile)
{
    /*
     * ------->
     *      -
     *    -
     *  -
     * ------->
     *
     */

    int            idx           = 0;
    constexpr auto prefetch_type = _MM_HINT_NTA;

    //_mm_prefetch(b + jb + j, prefetch_type);
    for (int i = 0; i < utile.static_extent(0); i++)
    {
        //_mm_prefetch(b + (i + 1) * cols + j, prefetch_type);
        inline_memcpy(
          utile.data_handle() + i, &submatrix[i, 0], utile.static_extent(1) * sizeof(T));
    }
}

template<std::size_t Mc, std::size_t Kc, std::size_t Mr, typename T>
void initATile(
  std::mdspan<const T, std::extents<std::size_t, Mc, Kc>, std::layout_stride> submatrix,
  std::mdspan<T, std::extents<std::size_t, Kc, Mr>>                           utile)
{
    /*
    |   ^|
    |  | |
    | |  |
    ||   |
    v    v
    */

    static_assert(Mc % Mr == 0, "Invalid m pattern");

    int  idx = 0;
    auto dst = utile.data_handle();
    {
        for (int j = 0; j < submatrix.static_extent(1); j++)
        {
            for (int ic = 0; ic < Mr; ++ic)
            {
                dst[idx++] = submatrix[(ic), j];
            }
        }
    }
}

} // namespace mm::core