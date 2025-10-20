#pragma once

#include <cstdint>
#include <cstring>
#include <immintrin.h>

// Detect and alias a bfloat16 type as bf16_std
#if defined(__has_include)
#if __has_include(<stdfloat>)
#include <stdfloat>
#if defined(__STDCPP_BFLOAT16_T__)
using bf16_std = std::bfloat16_t;
#endif
#endif
#endif

#if !defined(bf16_std)
#if defined(__clang__) || defined(__GNUC__)
using bf16_std = __bf16;
#else
#include <type_traits>
struct alignas(2) bf16_fallback
{
    uint16_t v;
};
static_assert(sizeof(bf16_fallback) == 2, "bf16_fallback must be 2 bytes");
using bf16_std = bf16_fallback;
#endif
#endif

static_assert(sizeof(bf16_std) == 2, "bf16 type must be 2 bytes");

namespace kernels
{

// Build a 512-bit BF16 vector with alternating pair [a0, a1] repeated across 32 lanes
// Requires AVX-512BW for 16-bit lane operations.
static inline __m512bh bf16_broadcast_pair(bf16_std a0, bf16_std a1)
{
#if defined(__AVX512BF16__) && defined(__AVX512BW__)
    uint16_t a0_bits{};
    uint16_t a1_bits{};
    std::memcpy(&a0_bits, &a0, sizeof(a0_bits));
    std::memcpy(&a1_bits, &a1, sizeof(a1_bits));

    const __m512i va0 = _mm512_set1_epi16(static_cast<int16_t>(a0_bits));
    const __m512i va1 = _mm512_set1_epi16(static_cast<int16_t>(a1_bits));
    // Mask with 1s in odd 16-bit positions: 0b1010... (32 lanes)
    const __mmask32 odd_mask = 0xAAAAAAAAu;
    const __m512i   v        = _mm512_mask_blend_epi16(odd_mask, va0, va1);
    return (__m512bh)v;
#else
    (void)a0;
    (void)a1;
    return (__m512bh)_mm512_setzero_si512();
#endif
}

// Interleave two 16-element BF16 rows into one 32-element BF16 vector:
// result = [row0[0], row1[0], row0[1], row1[1], ..., row0[15], row1[15]]
static inline __m512bh bf16_interleave2x16_to_32(const bf16_std* row0, const bf16_std* row1)
{
#if defined(__AVX512BF16__)
    const __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row0));
    const __m256i r1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row1));
    const __m256i lo = _mm256_unpacklo_epi16(r0, r1);
    const __m256i hi = _mm256_unpackhi_epi16(r0, r1);

    __m512i v = _mm512_castsi256_si512(lo);
    v         = _mm512_inserti64x4(v, hi, 1);
    return (__m512bh)v;
#else
    (void)row0;
    (void)row1;
    return (__m512bh)_mm512_setzero_si512();
#endif
}

// Convert 16 std::bfloat16_t values to 16 float32 values in an __m512 vector
// using integer widening and bit-shift (bf16 occupies the upper 16 bits of f32).
static inline __m512 bf16_load16_to_ps(const bf16_std* src)
{
    // Load 16x u16 (32 bytes)
    const __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    // Zero-extend to 16x u32
    __m512i v32 = _mm512_cvtepu16_epi32(v16);
    // Shift to bf16 position in f32 layout
    v32 = _mm512_slli_epi32(v32, 16);
    // Reinterpret as 16x f32
    return _mm512_castsi512_ps(v32);
}

// Store 16 float32 values as bf16
static inline void bf16_store16_from_ps(bf16_std* dst, __m512 v)
{
#if defined(__AVX512BF16__)
    __m256bh packed_bh = _mm512_cvtneps_pbh(v);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), (__m256i)packed_bh);
#else
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, v);
    uint16_t out16[16];
    for (int i = 0; i < 16; ++i)
    {
        uint32_t u;
        std::memcpy(&u, &tmp[i], sizeof(u));
        // Round to nearest even
        uint32_t lsb  = (u >> 16) & 1u;
        uint32_t bias = 0x7FFFu + lsb;
        u += bias;
        out16[i] = static_cast<uint16_t>(u >> 16);
    }
    std::memcpy(dst, out16, sizeof(out16));
#endif
}

// Convert one std::bfloat16_t to float32 (scalar)
static inline float bf16_to_float(bf16_std v)
{
    uint16_t h{};
    std::memcpy(&h, &v, sizeof(h));
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float          f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Store r (accumulators) into C with increment (C += r), for Mr rows and Nrs vector blocks per row
static inline void store_rows_inc_add(bf16_std* c, const __m512* r, int N, int Mr, int Nrs)
{
    constexpr int VEC_WIDTH = 16; // number of float32 elements per __m512

    _mm_prefetch(reinterpret_cast<const char*>(c + N), _MM_HINT_NTA);

    for (int row = 0; row < Mr; ++row)
    {
        for (int j = 0; j < Nrs; ++j)
        {
            // Load 16 bf16 -> f32, add, convert back to bf16 and store
            bf16_std* cj = c + j * VEC_WIDTH;
            __m512    cv = bf16_load16_to_ps(cj);
            cv           = _mm512_add_ps(cv, r[row * Nrs + j]);
            bf16_store16_from_ps(cj, cv);
        }
        c += N;
    }
}

// Packed bf16 kernel for Zen5 using AVX-512 intrinsics.
// Layout/assumptions mirror the double-precision packed kernel:
// - a: points to packed A micro-panel of size (Mr x Kc), advancing by Mr each k
// - b: points to packed B micro-panel of size (Kc x Nr), advancing by Nr each k
// - c: points to C tile top-left (row-major, leading dimension N), accumulate into float32
// - Nr must be divisible by 16 (vector width of __m512 floats)
template<int Nr, int Mr, int Kc>
static inline void zen5_packed_kernel_bf16(const bf16_std* __restrict a,
                                           const bf16_std* __restrict b,
                                           bf16_std* __restrict c,
                                           int N)
{
    static_assert(Nr % 16 == 0, "Nr must be divisible by 16 for AVX-512 float accumulators");

    constexpr int VEC_WIDTH = 16; // float32 lanes in __m512
    constexpr int Nrs       = Nr / VEC_WIDTH;

    __m512 r[Nrs * Mr];
    for (int i = 0; i < Nrs * Mr; ++i)
    {
        r[i] = _mm512_setzero_ps();
    }

    const bf16_std* a_iter = a;
    const bf16_std* b_iter = b;

#if defined(__AVX512BF16__)
    // Process K in pairs using BF16 dot-product accumulate
    int k = 0;
    for (; k + 1 < Kc; k += 2, b_iter += 2 * Nr, a_iter += 2 * Mr)
    {
        const bf16_std* b0 = b_iter;
        const bf16_std* b1 = b_iter + Nr;

        __m512bh b_pairs[Nrs];
        for (int j = 0; j < Nrs; ++j)
        {
            // Interleave columns for k and k+1
            b_pairs[j] = bf16_interleave2x16_to_32(b0 + j * VEC_WIDTH, b1 + j * VEC_WIDTH);
        }

        const bf16_std* a0 = a_iter;
        const bf16_std* a1 = a_iter + Mr;

        for (int i = 0; i < Mr; ++i)
        {
            const __m512bh a_pair = bf16_broadcast_pair(a0[i], a1[i]);
            __m512*        rrow   = &r[i * Nrs];
            for (int j = 0; j < Nrs; ++j)
            {
                rrow[j] = _mm512_dpbf16_ps(rrow[j], a_pair, b_pairs[j]);
            }
        }
    }

    // Handle odd K tail: pair with zeros
    if (k < Kc)
    {
        // Build B interleave with zero for the missing partner
        __m512bh        b_pairs[Nrs];
        const bf16_std* b0 = b_iter;
        for (int j = 0; j < Nrs; ++j)
        {
            const bf16_std* row0 = b0 + j * VEC_WIDTH;
            // Interleave row0 with zeros: [row0[i], 0]
            const __m256i r0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row0));
            const __m256i z  = _mm256_setzero_si256();
            const __m256i lo = _mm256_unpacklo_epi16(r0, z);
            const __m256i hi = _mm256_unpackhi_epi16(r0, z);
            __m512i       v  = _mm512_castsi256_si512(lo);
            v                = _mm512_inserti64x4(v, hi, 1);
            b_pairs[j]       = (__m512bh)v;
        }

        const bf16_std* a0        = a_iter;
        const bf16_std  zero_bf16 = static_cast<bf16_std>(0);
        for (int i = 0; i < Mr; ++i)
        {
            const __m512bh a_pair = bf16_broadcast_pair(a0[i], zero_bf16);
            __m512*        rrow   = &r[i * Nrs];
            for (int j = 0; j < Nrs; ++j)
            {
                rrow[j] = _mm512_dpbf16_ps(rrow[j], a_pair, b_pairs[j]);
            }
        }
        // a_iter += Mr; b_iter += Nr; // not needed since we exit
    }
#else
    // Fallback: process using float conversion + fmadd if BF16 DP not available
    for (int kf = 0; kf < Kc; ++kf, b_iter += Nr, a_iter += Mr)
    {
        __m512 bvecs[Nrs];
        for (int j = 0; j < Nrs; ++j)
        {
            bvecs[j] = bf16_load16_to_ps(b_iter + j * VEC_WIDTH);
        }
        for (int i = 0; i < Mr; ++i)
        {
            const float  a_scalar = bf16_to_float(a_iter[i]);
            const __m512 a_bcast  = _mm512_set1_ps(a_scalar);
            __m512*      rrow     = &r[i * Nrs];
            for (int j = 0; j < Nrs; ++j)
            {
                rrow[j] = _mm512_fmadd_ps(a_bcast, bvecs[j], rrow[j]);
            }
        }
    }
#endif

    // Store accumulators into C (C += R)
    store_rows_inc_add(c, r, N, Mr, Nrs);
}

} // namespace kernels
