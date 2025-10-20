#pragma once

#include <mm/core/Shape.hpp>

#include <cstring> // memcpy
#include <array>
#include <immintrin.h>

// TODO: Debug logs
// #include <iostream>

static constexpr int mceil(int value, int elem)
{
    auto v = value / elem;
    return value % elem == 0 ? v : v + 1;
}

static_assert(mceil(100, 12) == 9, "mceil failed");

constexpr int blockWithPadding(int Nc, int Nr)
{
    return mceil(Nc, Nr) * Nr;
}

static_assert(blockWithPadding(256, 12) == 264, "blockWithPadding failed");

template<int I, int J>
std::array<double, I * J> packMatrix(const double* b, int j_size)
{

    constexpr int istep{I / 2};
    constexpr int jstep{J / 2};
    static_assert(I % 2 == 0, "");
    static_assert(J % 2 == 0, "");

    std::array<double, I * J> b_packed;
    for (int i = 0; i < I; i += istep)
    {

        for (int j = 0; j < J; j += jstep)
        {
            for (int i2 = 0; i2 < istep; i2++)
            {
                //_mm_prefetch(&b[(i + i2) * j_size + j], _MM_HINT_NTA);
                std::memcpy(
                  &b_packed[(i + i2) * J] + j, &b[(i + i2) * j_size + j], jstep * sizeof(double));
            }
        }
    }

    return b_packed;
}

template<int Mc, int Nc, int Mr, int Nr>
void reorderColOrderPaddingMatrix(const double* b, int rows, int cols, double* dest)
{
    /*
    |   ^|
    |  | |
    | |  |
    ||   |
    v    v
    */

    // Do we need consider special case Kr==1?

    // static_assert(Mc % Mr == 0, "Invalid m pattern");
    // static_assert(Nc % Nr == 0, "Invalid n pattern");

    // Nr is Kr; skip for now;

    // Case1: Mc < Mr;
    // Case2: Mc > Mr but Mc%Mr!=0;
    // Case3: Mc > Mr but Mc%Mr==0;
    // Same for Nc,Nr pair? Skip for now

    int idx = 0;

    // DON'T REORDER LOOPS
    // Process columns in groups of 4
    for (int i = 0; i < Mc; i += Mr)
    {
        for (int j = 0; j < Nc; j += Nr)
        {
            for (int jc = 0; jc < Nr; ++jc)
            {
                for (int ic = 0; ic < Mr; ++ic)
                {
                    // Check if we're out of original bounds
                    if ((i + ic) < rows && (j + jc) < cols)
                    {
                        dest[idx] = b[(i + ic) * cols + (j + jc)];
                    }
                    else
                    {
                        dest[idx] = 0.0; // Padding value
                    }
                    idx++;
                }
            }
        }
    }
}

template<int Mc, int Nc, int Mr, int Nr>
void reorderRowMajorPaddingMatrix(const double* b, int cols, double* dest)
{
    // TODO: Implement
    static_assert(Mc % Mr == 0, "Invalid m pattern");
    static_assert(Nc % Nr == 0, "Invalid n pattern");

    // reorder B;

    /*
     * ------->
     *      -
     *    -
     *  -
     * ------->
     *
     */

    int            idx           = 0;
    int            jtail         = Nc % Nr;
    int            itail         = Mc % Mr;
    constexpr auto prefetch_type = _MM_HINT_T0;

    for (int j = 0; j < Nc; j += Nr)
    {
        for (int i = 0; i < Mc; i += Mr)
        {
            for (int ic = 0; ic < Mr; ++ic)
            {
                const auto col = (i + ic) * cols;
                // _mm_prefetch(b + (i + ic + 1) * N + j, prefetch_type);
                // _mm_prefetch(b + (i + ic + 2) * N + j, prefetch_type);
                // _mm_prefetch(b + (i + ic + 3) * N + j, prefetch_type);

                for (int jc = 0; jc < Nr; ++jc)
                {
                    dest[idx++] = b[col + j + jc];
                }
            }
        }
    }
}

template<int Mr, int Nr>
void reorderColOrderMatrixTail(const double* matrix, int N, double* dest, int Mc, int Nc)
{
    /*
     *  IJJI
    |   ^|
    |  | |
    | |  |
    ||   |
    v    v
    */

    // TODO:
    // order loop shoud match order loop from matmul
    // tail order should match order loop from matmul tail computation
    // Don't handle Nr case, since we use Kr heere which is 1
    int idx = 0;

    // if (Mc > Mr)
    {
        // Process full tiles
        int i_limit = Mc - Mc % Mr; //(Mc / Ir) * Ir;
        int j_limit = Nc - Nc % Nr; //(Nc / Jr) * Jr;

        for (int i = 0; i < i_limit; i += Mr)
        {
            for (int j = 0; j < j_limit; j += Nr)
            {
                for (int jc = 0; jc < Nr; ++jc)
                {
                    for (int ic = 0; ic < Mr; ++ic)
                    {
                        dest[idx++] = matrix[(i + ic) * N + j + jc];
                    }
                }
            }

            for (int j = j_limit; j < Nc; ++j)
            {
                for (int ic = 0; ic < Mr; ++ic)
                {
                    dest[idx++] = matrix[(i + ic) * N + j];
                }
            }
        }

        // Tail in M (rows)
        for (int i = i_limit; i < Mc; ++i)
        {
            for (int j = 0; j < j_limit; j += Nr)
            {
                for (int jc = 0; jc < Nr; ++jc)
                {
                    dest[idx++] = matrix[i * N + j + jc];
                }
            }

            for (int j = j_limit; j < Nc; ++j)
            {
                dest[idx++] = matrix[i * N + j];
            }
        }
    }
    //    else if (Mc == Mr)
    //    {
    //        // Process full tiles
    //        int i_limit = Mc - Mc % Mr; //(Mc / Ir) * Ir;
    //        int j_limit = Nc - Nc % Nr; //(Nc / Jr) * Jr;

    //        for (int j = 0; j < j_limit; j += Nr)
    //        {
    //            for (int jc = 0; jc < Nr; ++jc)
    //            {
    //                for (int ic = 0; ic < Mr; ++ic)
    //                {
    //                    dest[idx++] = matrix[(ic)*N + j + jc];
    //                }
    //            }
    //        }

    //        for (int j = j_limit; j < Nc; ++j)
    //        {
    //            for (int ic = 0; ic < Mr; ++ic)
    //            {
    //                dest[idx++] = matrix[(ic)*N + j];
    //            }
    //        }
    //    }
    //    else
    //    {
    //        throw std::runtime_error("Undefined Mr,Nr sizes");
    //    }
}

template<int Mc, int Nc, int Mr, int Nr, typename T>
void reorderColOrderMatrix(const T* matrix, int cols, T* dest)
{
    /*
    |   ^|
    |  | |
    | |  |
    ||   |
    v    v
    */
    static_assert(Mc % Mr == 0, "Invalid m pattern");
    static_assert(Nc % Nr == 0, "Invalid n pattern");
    int idx = 0;

    constexpr auto prefetch_type = _MM_HINT_NTA;

    // DON'T REORDER LOOPS
    // Process columns in groups of 4
    for (int i = 0; i < Mc; i += Mr)
    {
        for (int j = 0; j < Nc; j += Nr)
        {
            _mm_prefetch(matrix + (i + 0) * cols + j + Nr, prefetch_type);
            //_mm_prefetch(matrix + (i + 1) * cols + j + Nr, prefetch_type);
            //_mm_prefetch(matrix + (i + 2) * cols + j + Nr, prefetch_type);
            //_mm_prefetch(matrix + (i + 3) * cols + j + Nr, prefetch_type);

            for (int jc = 0; jc < Nr; ++jc)
            {
                //                _mm_prefetch(matrix + (i + 1) * cols + j, prefetch_type);
                //                _mm_prefetch(matrix + (i + 2) * cols + j, prefetch_type);
                //                _mm_prefetch(matrix + (i + 3) * cols + j, prefetch_type);

                for (int ic = 0; ic < Mr; ++ic)
                {

                    dest[idx++] = matrix[(i + ic) * cols + j + jc];
                }
            }
        }
    }

    // TODO: Add tailes
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

template<int ib, int jb>
void reorderRowMajorMatrix(const double* b, int cols, double* dest, int M, int N)
{
    // reorder B; J I I J order

    /*
     * ------->
     *      -
     *    -
     *  -
     * ------->
     *
     */

    int            idx           = 0;
    constexpr auto prefetch_type = _MM_HINT_T0;

    for (int j = 0; j < N; j += jb)
    {
        for (int i = 0; i < M; i += ib)
        {
            for (int ic = 0; ic < ib; ++ic)
            {
                const auto col = (i + ic) * cols;
                for (int jc = 0; jc < jb; ++jc)
                {
                    dest[idx++] = b[col + j + jc];
                }
            }
        }
    }
}

static inline void* inline_memcpy(void* __restrict dst_, const void* __restrict src_, size_t size)
{
    /// We will use pointer arithmetic, so char pointer will be used.
    /// Note that __restrict makes sense (otherwise compiler will reload data from memory
    /// instead of using the value of registers due to possible aliasing).
    char* __restrict dst       = reinterpret_cast<char* __restrict>(dst_);
    const char* __restrict src = reinterpret_cast<const char* __restrict>(src_);

    /// Standard memcpy returns the original value of dst. It is rarely used but we have to do it.
    /// If you use memcpy with small but non-constant sizes, you can call inline_memcpy directly
    /// for inlining and removing this single instruction.
    void* ret = dst;

tail:
    /// Small sizes and tails after the loop for large sizes.
    /// The order of branches is important but in fact the optimal order depends on the distribution
    /// of sizes in your application. This order of branches is from the disassembly of glibc's
    /// code. We copy chunks of possibly uneven size with two overlapping movs. Example: to copy 5
    /// bytes [0, 1, 2, 3, 4] we will copy tail [1, 2, 3, 4] first and then head [0, 1, 2, 3].
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
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + size - 16),
                             _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + size - 16)));

            /// Then we will copy every 16 bytes from the beginning in a loop.
            /// The last loop iteration will possibly overwrite some part of already copied last 16
            /// bytes. This is Ok, similar to the code for small sizes above.
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
template<int M, int N, int ib, int jb, typename T>
void reorderRowMajorMatrixAVX(const T* b, int cols, T* dest)
{
    static_assert(ib == 1, "ib != 1, but we optimize for ib =1");
    // reorder B; J I I J order

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

    for (int j = 0; j < N; j += jb)
    {
        //_mm_prefetch(b + jb + j, prefetch_type);
        for (int i = 0; i < M; i += ib)
        {
            // what is dependency of the offset from type?
            _mm_prefetch(b + (i + 1) * cols + j, prefetch_type);
            inline_memcpy(dest + idx, &b[i * cols + j], sizeof(T) * jb);
            idx += jb;
        }
    }
}

template<int M, int N, int ib, int jb>
void reorderRowMajorMatrix(const double* b, int cols, double* dest)
{
    // reorder B; J I I J order

    /*
     * ------->
     *      -
     *    -
     *  -
     * ------->
     *
     */

    static_assert(M % ib == 0, "M % ib == 0");
    static_assert(N % jb == 0, "N % jb == 0");

    int            idx           = 0;
    constexpr auto prefetch_type = _MM_HINT_T0;

    for (int j = 0; j < N; j += jb)
    {
        //_mm_prefetch(b + jb + j, prefetch_type);
        for (int i = 0; i < M; i += ib)
        {
            for (int ic = 0; ic < ib; ++ic)
            {
                const auto col = (i + ic) * cols;
                _mm_prefetch(b + (i + ic + 1) * cols + j, prefetch_type);
                for (int jc = 0; jc < jb; ++jc)
                {
                    dest[idx++] = b[col + j + jc];
                }
            }
        }
    }
}

template<int Mc, int Nc, int Mr, int Nr>
void reorderRowMajorMatrix123(const double* b, int cols, double* dest)
{
    // reorder B; JIIJ order

    int            idx           = 0;
    constexpr auto prefetch_type = _MM_HINT_T0;

    auto store_block = [&](int i_start, int j_start, int ib_size, int jb_size, bool pad = false)
    {
        for (int ic = 0; ic < ib_size; ++ic)
        {
            const int row  = i_start + ic;
            const int base = row * cols;
            for (int jc = 0; jc < jb_size; ++jc)
            {
                const int col = j_start + jc;
                dest[idx++]   = b[base + col];
            }
        }
    };

    const int itail = Mc % Mr;
    const int jtail = Nc % Nr;

    const int i_full = Mc - itail;
    const int j_full = Nc - jtail;

    // Main blocks
    for (int j = 0; j < j_full; j += Nr)
    {
        for (int i = 0; i < i_full; i += Mr)
        {
            store_block(i, j, Mr, Nr);
        }

        // i tail with j full
        if (itail != 0)
        {
            store_block(i_full, j, itail, Nr);
        }
    }

    // j tail blocks (with padding)
    if (jtail != 0)
    {
        for (int i = 0; i < i_full; i += Mr)
        {
            store_block(i, j_full, Mr, Nr);
        }

        if (itail != 0)
        {
            store_block(i_full, j_full, itail, Nr);
        }
    }
}

/// TODO: Should be slow due to if cond in inner loop
template<int Mc, int Nc, int Mr, int Nr>
void reorderRowMajorMatrixWithPadding(const double* b, int cols, double* dest)
{
    // reorder B; JIIJ order

    int            idx           = 0;
    constexpr auto prefetch_type = _MM_HINT_T0;

    auto store_block = [&](int i_start, int j_start, int ib_size, int jb_size, bool pad = false)
    {
        for (int ic = 0; ic < ib_size; ++ic)
        {
            const int row  = i_start + ic;
            const int base = row * cols;
            for (int jc = 0; jc < jb_size; ++jc)
            {
                const int col    = j_start + jc;
                bool      inside = col < cols; // TODO: Add check for row; row < Mc &&
                dest[idx++]      = (inside && !pad) ? b[base + col] : 0.0;
            }
        }
    };

    const int itail = Mc % Mr;
    const int jtail = Nc % Nr;

    const int i_full = Mc - itail;
    const int j_full = Nc - jtail;

    // Main blocks
    for (int j = 0; j < j_full; j += Nr)
    {
        for (int i = 0; i < i_full; i += Mr)
        {
            store_block(i, j, Mr, Nr);
        }

        // i tail with j full
        if (itail != 0)
        {
            store_block(i_full, j, itail, Nr);
            store_block(Mc, j, Mr - itail, Nr, true);
        }
    }

    // j tail blocks (with padding)
    if (jtail != 0)
    {
        for (int i = 0; i < i_full; i += Mr)
        {
            store_block(i, j_full, Mr, Nr); // will zero-fill beyond N
        }

        // bottom-right corner (i tail + j tail)
        if (itail != 0)
        {
            store_block(i_full, j_full, itail, Nr);
            store_block(Mc, j_full, Mr - itail, Nr, true);
        }
    }
}

// TODO: Rename???
template<int M, int N, int ib, int jb>
void reorderRowMajorMatrixOldWithPadding(const double* b, int cols, double* dest)
{
    //    static_assert(M % ib == 0, "Invalid m pattern");
    //    static_assert(N % jb == 0, "Invalid n pattern");

    // reorder B; J I I J order

    /*
     * ------->
     *      -
     *    -
     *  -
     * ------->
     *
     */

    int            idx           = 0;
    int            jtail         = N % jb;
    int            itail         = M % ib;
    constexpr auto prefetch_type = _MM_HINT_T0;

    for (int j = 0; j < N - jtail; j += jb)
    {
        for (int i = 0; i < M - itail; i += ib)
        {
            for (int ic = 0; ic < ib; ++ic)
            {
                const auto col = (i + ic) * cols;
                for (int jc = 0; jc < jb; ++jc)
                {
                    dest[idx++] = b[col + j + jc];
                }
            }
        }
        // handle itail/ last block
        if (itail != 0)
        {
            for (int i = M - itail; i < M; i++)
            {
                for (int jc = 0; jc < jb; ++jc)
                {
                    dest[idx++] = b[i * cols + j + jc];
                }
            }
            for (int i = M; i < M - itail + ib; i++)
            {
                for (int jc = 0; jc < jb; ++jc)
                {
                    dest[idx++] = 0;
                }
            }
        }
    }
    // handle jtail
    if (jtail != 0)
    {
        for (int i = 0; i < M - itail; i += ib)
        {
            for (int ic = 0; ic < ib; ++ic)
            {
                const auto col = (i + ic) * cols;

                for (int j = N - jtail; j < N - jtail + jb; j++)
                {
                    if (j < N)
                        dest[idx++] = b[col + j];
                    else
                        dest[idx++] = 0;
                }
            }
        }

        // handle itail/ last block
        if (itail != 0)
        {
            for (int i = M - itail; i < M; i++)
            {
                for (int j = N - jtail; j < N - jtail + jb; j++)
                {
                    if (j < N)
                        dest[idx++] = b[i * cols + j];
                    else
                        dest[idx++] = 0;
                }
            }

            for (int i = M; i < M - itail + ib; i++)
            {
                for (int j = N - jtail; j < N - jtail + jb; j++)
                {
                    dest[idx++] = 0;
                }
            }
        }
    }
}

static int blasReorderRowOrder4x4(long m, long n, double* a, long lda, double* b)
{

    long i, j;

    double *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    double *b_offset, *b_offset1, *b_offset2, *b_offset3;
    double  ctemp1, ctemp2, ctemp3, ctemp4;
    double  ctemp5, ctemp6, ctemp7, ctemp8;
    double  ctemp9, ctemp10, ctemp11, ctemp12;
    double  ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = a;
    b_offset = b;

    b_offset2 = b + m * (n & ~3);
    b_offset3 = b + m * (n & ~1);

    j = (m >> 2);
    if (j > 0)
    {
        do
        {
            a_offset1 = a_offset;
            a_offset2 = a_offset1 + lda;
            a_offset3 = a_offset2 + lda;
            a_offset4 = a_offset3 + lda;
            a_offset += 4 * lda;

            b_offset1 = b_offset;
            b_offset += 16;

            i = (n >> 2);
            if (i > 0)
            {
                do
                {
                    ctemp1 = *(a_offset1 + 0);
                    ctemp2 = *(a_offset1 + 1);
                    ctemp3 = *(a_offset1 + 2);
                    ctemp4 = *(a_offset1 + 3);

                    ctemp5 = *(a_offset2 + 0);
                    ctemp6 = *(a_offset2 + 1);
                    ctemp7 = *(a_offset2 + 2);
                    ctemp8 = *(a_offset2 + 3);

                    ctemp9  = *(a_offset3 + 0);
                    ctemp10 = *(a_offset3 + 1);
                    ctemp11 = *(a_offset3 + 2);
                    ctemp12 = *(a_offset3 + 3);

                    ctemp13 = *(a_offset4 + 0);
                    ctemp14 = *(a_offset4 + 1);
                    ctemp15 = *(a_offset4 + 2);
                    ctemp16 = *(a_offset4 + 3);

                    a_offset1 += 4;
                    a_offset2 += 4;
                    a_offset3 += 4;
                    a_offset4 += 4;

                    *(b_offset1 + 0) = ctemp1;
                    *(b_offset1 + 1) = ctemp2;
                    *(b_offset1 + 2) = ctemp3;
                    *(b_offset1 + 3) = ctemp4;

                    *(b_offset1 + 4) = ctemp5;
                    *(b_offset1 + 5) = ctemp6;
                    *(b_offset1 + 6) = ctemp7;
                    *(b_offset1 + 7) = ctemp8;

                    *(b_offset1 + 8)  = ctemp9;
                    *(b_offset1 + 9)  = ctemp10;
                    *(b_offset1 + 10) = ctemp11;
                    *(b_offset1 + 11) = ctemp12;

                    *(b_offset1 + 12) = ctemp13;
                    *(b_offset1 + 13) = ctemp14;
                    *(b_offset1 + 14) = ctemp15;
                    *(b_offset1 + 15) = ctemp16;

                    b_offset1 += m * 4;
                    i--;
                } while (i > 0);
            }

            if (n & 2)
            {
                ctemp1 = *(a_offset1 + 0);
                ctemp2 = *(a_offset1 + 1);

                ctemp3 = *(a_offset2 + 0);
                ctemp4 = *(a_offset2 + 1);

                ctemp5 = *(a_offset3 + 0);
                ctemp6 = *(a_offset3 + 1);

                ctemp7 = *(a_offset4 + 0);
                ctemp8 = *(a_offset4 + 1);

                a_offset1 += 2;
                a_offset2 += 2;
                a_offset3 += 2;
                a_offset4 += 2;

                *(b_offset2 + 0) = ctemp1;
                *(b_offset2 + 1) = ctemp2;
                *(b_offset2 + 2) = ctemp3;
                *(b_offset2 + 3) = ctemp4;

                *(b_offset2 + 4) = ctemp5;
                *(b_offset2 + 5) = ctemp6;
                *(b_offset2 + 6) = ctemp7;
                *(b_offset2 + 7) = ctemp8;

                b_offset2 += 8;
            }

            if (n & 1)
            {
                ctemp1 = *(a_offset1 + 0);
                ctemp2 = *(a_offset2 + 0);
                ctemp3 = *(a_offset3 + 0);
                ctemp4 = *(a_offset4 + 0);

                *(b_offset3 + 0) = ctemp1;
                *(b_offset3 + 1) = ctemp2;
                *(b_offset3 + 2) = ctemp3;
                *(b_offset3 + 3) = ctemp4;

                b_offset3 += 4;
            }

            j--;
        } while (j > 0);
    }

    if (m & 2)
    {
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + lda;
        a_offset += 2 * lda;

        b_offset1 = b_offset;
        b_offset += 8;

        i = (n >> 2);
        if (i > 0)
        {
            do
            {
                ctemp1 = *(a_offset1 + 0);
                ctemp2 = *(a_offset1 + 1);
                ctemp3 = *(a_offset1 + 2);
                ctemp4 = *(a_offset1 + 3);

                ctemp5 = *(a_offset2 + 0);
                ctemp6 = *(a_offset2 + 1);
                ctemp7 = *(a_offset2 + 2);
                ctemp8 = *(a_offset2 + 3);

                a_offset1 += 4;
                a_offset2 += 4;

                *(b_offset1 + 0) = ctemp1;
                *(b_offset1 + 1) = ctemp2;
                *(b_offset1 + 2) = ctemp3;
                *(b_offset1 + 3) = ctemp4;

                *(b_offset1 + 4) = ctemp5;
                *(b_offset1 + 5) = ctemp6;
                *(b_offset1 + 6) = ctemp7;
                *(b_offset1 + 7) = ctemp8;

                b_offset1 += m * 4;
                i--;
            } while (i > 0);
        }

        if (n & 2)
        {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset1 + 1);

            ctemp3 = *(a_offset2 + 0);
            ctemp4 = *(a_offset2 + 1);

            a_offset1 += 2;
            a_offset2 += 2;

            *(b_offset2 + 0) = ctemp1;
            *(b_offset2 + 1) = ctemp2;
            *(b_offset2 + 2) = ctemp3;
            *(b_offset2 + 3) = ctemp4;

            b_offset2 += 4;
        }

        if (n & 1)
        {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset2 + 0);

            *(b_offset3 + 0) = ctemp1;
            *(b_offset3 + 1) = ctemp2;
            b_offset3 += 2;
        }
    }

    if (m & 1)
    {
        a_offset1 = a_offset;
        b_offset1 = b_offset;

        i = (n >> 2);
        if (i > 0)
        {
            do
            {
                ctemp1 = *(a_offset1 + 0);
                ctemp2 = *(a_offset1 + 1);
                ctemp3 = *(a_offset1 + 2);
                ctemp4 = *(a_offset1 + 3);

                a_offset1 += 4;

                *(b_offset1 + 0) = ctemp1;
                *(b_offset1 + 1) = ctemp2;
                *(b_offset1 + 2) = ctemp3;
                *(b_offset1 + 3) = ctemp4;

                b_offset1 += 4 * m;

                i--;
            } while (i > 0);
        }

        if (n & 2)
        {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset1 + 1);
            a_offset1 += 2;

            *(b_offset2 + 0) = ctemp1;
            *(b_offset2 + 1) = ctemp2;
        }

        if (n & 1)
        {
            ctemp1           = *(a_offset1 + 0);
            *(b_offset3 + 0) = ctemp1;
        }
    }

    return 0;
}

static int blasReorderColOrder8x8(long m,
                                  long n,
                                  double* __restrict a,
                                  long lda,
                                  double* __restrict b)
{
    long i, j;

    double* aoffset;
    double *aoffset1, *aoffset2, *aoffset3, *aoffset4;
    double *aoffset5, *aoffset6, *aoffset7, *aoffset8;

    double*                      boffset;
    double                       ctemp01, ctemp02, ctemp03, ctemp04;
    double                       ctemp05, ctemp06, ctemp07, ctemp08;
    double                       ctemp09, ctemp10, ctemp11, ctemp12;
    double                       ctemp13, ctemp14, ctemp15, ctemp16;
    double                       ctemp17 /*, ctemp18, ctemp19, ctemp20*/;
    double /*ctemp21, ctemp22,*/ ctemp23, ctemp24;
    double                       ctemp25 /*, ctemp26, ctemp27, ctemp28*/;
    double /*ctemp29, ctemp30,*/ ctemp31, ctemp32;
    double                       ctemp33 /*, ctemp34, ctemp35, ctemp36*/;
    double /*ctemp37, ctemp38,*/ ctemp39, ctemp40;
    double                       ctemp41 /*, ctemp42, ctemp43, ctemp44*/;
    double /*ctemp45, ctemp46,*/ ctemp47, ctemp48;
    double                       ctemp49 /*, ctemp50, ctemp51, ctemp52*/;
    double /*ctemp53, ctemp54,*/ ctemp55, ctemp56;
    double                       ctemp57 /*, ctemp58, ctemp59, ctemp60*/;
    double /*ctemp61, ctemp62,*/ ctemp63, ctemp64;

    aoffset = a;
    boffset = b;

    j = (n >> 3);
    if (j > 0)
    {
        do
        {
            aoffset1 = aoffset;
            aoffset2 = aoffset1 + lda;
            aoffset3 = aoffset2 + lda;
            aoffset4 = aoffset3 + lda;
            aoffset5 = aoffset4 + lda;
            aoffset6 = aoffset5 + lda;
            aoffset7 = aoffset6 + lda;
            aoffset8 = aoffset7 + lda;
            aoffset += 8 * lda;

            i = (m >> 3);
            if (i > 0)
            {
                do
                {
                    __m128d xmm0, xmm1;
                    xmm0 = _mm_load_pd1(aoffset2 + 0);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 0);
                    _mm_storeu_pd(boffset + 0, xmm0);

                    ctemp07 = *(aoffset1 + 6);
                    ctemp08 = *(aoffset1 + 7);

                    xmm1 = _mm_load_pd1(aoffset4 + 0);
                    xmm1 = _mm_loadl_pd(xmm1, aoffset3 + 0);
                    _mm_storeu_pd(boffset + 2, xmm1);

                    xmm0 = _mm_load_pd1(aoffset6 + 0);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 0);
                    _mm_storeu_pd(boffset + 4, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 0);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 0);
                    _mm_storeu_pd(boffset + 6, xmm0);

                    ctemp15 = *(aoffset2 + 6);
                    ctemp16 = *(aoffset2 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 1);
                    _mm_storeu_pd(boffset + 8, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 1);
                    _mm_storeu_pd(boffset + 10, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 1);
                    _mm_storeu_pd(boffset + 12, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 1);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 1);
                    _mm_storeu_pd(boffset + 14, xmm0);

                    xmm0 = _mm_load_pd1(aoffset2 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 2);
                    _mm_storeu_pd(boffset + 16, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 2);
                    _mm_storeu_pd(boffset + 18, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 2);
                    _mm_storeu_pd(boffset + 20, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 2);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 2);
                    _mm_storeu_pd(boffset + 22, xmm0);

                    ctemp23 = *(aoffset3 + 6);
                    ctemp24 = *(aoffset3 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 3);
                    _mm_storeu_pd(boffset + 24, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 3);
                    _mm_storeu_pd(boffset + 26, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 3);
                    _mm_storeu_pd(boffset + 28, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 3);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 3);
                    _mm_storeu_pd(boffset + 30, xmm0);

                    ctemp31 = *(aoffset4 + 6);
                    ctemp32 = *(aoffset4 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 4);
                    _mm_storeu_pd(boffset + 32, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 4);
                    _mm_storeu_pd(boffset + 34, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 4);
                    _mm_storeu_pd(boffset + 36, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 4);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 4);
                    _mm_storeu_pd(boffset + 38, xmm0);

                    ctemp39 = *(aoffset5 + 6);
                    ctemp40 = *(aoffset5 + 7);

                    xmm0 = _mm_load_pd1(aoffset2 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset1 + 5);
                    _mm_storeu_pd(boffset + 40, xmm0);

                    xmm0 = _mm_load_pd1(aoffset4 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset3 + 5);
                    _mm_storeu_pd(boffset + 42, xmm0);

                    xmm0 = _mm_load_pd1(aoffset6 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset5 + 5);
                    _mm_storeu_pd(boffset + 44, xmm0);

                    xmm0 = _mm_load_pd1(aoffset8 + 5);
                    xmm0 = _mm_loadl_pd(xmm0, aoffset7 + 5);
                    _mm_storeu_pd(boffset + 46, xmm0);

                    ctemp47 = *(aoffset6 + 6);
                    ctemp48 = *(aoffset6 + 7);

                    ctemp55 = *(aoffset7 + 6);
                    ctemp56 = *(aoffset7 + 7);

                    ctemp63 = *(aoffset8 + 6);
                    ctemp64 = *(aoffset8 + 7);

                    *(boffset + 48) = ctemp07;
                    *(boffset + 49) = ctemp15;
                    *(boffset + 50) = ctemp23;
                    *(boffset + 51) = ctemp31;
                    *(boffset + 52) = ctemp39;
                    *(boffset + 53) = ctemp47;
                    *(boffset + 54) = ctemp55;
                    *(boffset + 55) = ctemp63;

                    *(boffset + 56) = ctemp08;
                    *(boffset + 57) = ctemp16;
                    *(boffset + 58) = ctemp24;
                    *(boffset + 59) = ctemp32;
                    *(boffset + 60) = ctemp40;
                    *(boffset + 61) = ctemp48;
                    *(boffset + 62) = ctemp56;
                    *(boffset + 63) = ctemp64;

                    aoffset1 += 8;
                    aoffset2 += 8;
                    aoffset3 += 8;
                    aoffset4 += 8;
                    aoffset5 += 8;
                    aoffset6 += 8;
                    aoffset7 += 8;
                    aoffset8 += 8;
                    boffset += 64;
                    i--;
                } while (i > 0);
            }

            i = (m & 7);
            if (i > 0)
            {
                do
                {
                    ctemp01 = *(aoffset1 + 0);
                    ctemp09 = *(aoffset2 + 0);
                    ctemp17 = *(aoffset3 + 0);
                    ctemp25 = *(aoffset4 + 0);
                    ctemp33 = *(aoffset5 + 0);
                    ctemp41 = *(aoffset6 + 0);
                    ctemp49 = *(aoffset7 + 0);
                    ctemp57 = *(aoffset8 + 0);

                    *(boffset + 0) = ctemp01;
                    *(boffset + 1) = ctemp09;
                    *(boffset + 2) = ctemp17;
                    *(boffset + 3) = ctemp25;
                    *(boffset + 4) = ctemp33;
                    *(boffset + 5) = ctemp41;
                    *(boffset + 6) = ctemp49;
                    *(boffset + 7) = ctemp57;

                    aoffset1++;
                    aoffset2++;
                    aoffset3++;
                    aoffset4++;
                    aoffset5++;
                    aoffset6++;
                    aoffset7++;
                    aoffset8++;

                    boffset += 8;
                    i--;
                } while (i > 0);
            }
            j--;
        } while (j > 0);
    } /* end of if(j > 0) */

    if (n & 4)
    {
        aoffset1 = aoffset;
        aoffset2 = aoffset1 + lda;
        aoffset3 = aoffset2 + lda;
        aoffset4 = aoffset3 + lda;
        aoffset += 4 * lda;

        i = (m >> 2);
        if (i > 0)
        {
            do
            {
                ctemp01 = *(aoffset1 + 0);
                ctemp02 = *(aoffset1 + 1);
                ctemp03 = *(aoffset1 + 2);
                ctemp04 = *(aoffset1 + 3);

                ctemp05 = *(aoffset2 + 0);
                ctemp06 = *(aoffset2 + 1);
                ctemp07 = *(aoffset2 + 2);
                ctemp08 = *(aoffset2 + 3);

                ctemp09 = *(aoffset3 + 0);
                ctemp10 = *(aoffset3 + 1);
                ctemp11 = *(aoffset3 + 2);
                ctemp12 = *(aoffset3 + 3);

                ctemp13 = *(aoffset4 + 0);
                ctemp14 = *(aoffset4 + 1);
                ctemp15 = *(aoffset4 + 2);
                ctemp16 = *(aoffset4 + 3);

                *(boffset + 0) = ctemp01;
                *(boffset + 1) = ctemp05;
                *(boffset + 2) = ctemp09;
                *(boffset + 3) = ctemp13;

                *(boffset + 4) = ctemp02;
                *(boffset + 5) = ctemp06;
                *(boffset + 6) = ctemp10;
                *(boffset + 7) = ctemp14;

                *(boffset + 8)  = ctemp03;
                *(boffset + 9)  = ctemp07;
                *(boffset + 10) = ctemp11;
                *(boffset + 11) = ctemp15;

                *(boffset + 12) = ctemp04;
                *(boffset + 13) = ctemp08;
                *(boffset + 14) = ctemp12;
                *(boffset + 15) = ctemp16;

                aoffset1 += 4;
                aoffset2 += 4;
                aoffset3 += 4;
                aoffset4 += 4;
                boffset += 16;
                i--;
            } while (i > 0);
        }

        i = (m & 3);
        if (i > 0)
        {
            do
            {
                ctemp01 = *(aoffset1 + 0);
                ctemp02 = *(aoffset2 + 0);
                ctemp03 = *(aoffset3 + 0);
                ctemp04 = *(aoffset4 + 0);

                *(boffset + 0) = ctemp01;
                *(boffset + 1) = ctemp02;
                *(boffset + 2) = ctemp03;
                *(boffset + 3) = ctemp04;

                aoffset1++;
                aoffset2++;
                aoffset3++;
                aoffset4++;

                boffset += 4;
                i--;
            } while (i > 0);
        }
    } /* end of if(j > 0) */

    if (n & 2)
    {
        aoffset1 = aoffset;
        aoffset2 = aoffset1 + lda;
        aoffset += 2 * lda;

        i = (m >> 1);
        if (i > 0)
        {
            do
            {
                ctemp01 = *(aoffset1 + 0);
                ctemp02 = *(aoffset1 + 1);
                ctemp03 = *(aoffset2 + 0);
                ctemp04 = *(aoffset2 + 1);

                *(boffset + 0) = ctemp01;
                *(boffset + 1) = ctemp03;
                *(boffset + 2) = ctemp02;
                *(boffset + 3) = ctemp04;

                aoffset1 += 2;
                aoffset2 += 2;
                boffset += 4;
                i--;
            } while (i > 0);
        }

        if (m & 1)
        {
            ctemp01 = *(aoffset1 + 0);
            ctemp02 = *(aoffset2 + 0);

            *(boffset + 0) = ctemp01;
            *(boffset + 1) = ctemp02;

            aoffset1++;
            aoffset2++;
            boffset += 2;
        }
    } /* end of if(j > 0) */

    if (n & 1)
    {
        aoffset1 = aoffset;

        i = m;
        if (i > 0)
        {
            do
            {
                ctemp01 = *(aoffset1 + 0);

                *(boffset + 0) = ctemp01;

                aoffset1++;
                boffset++;
                i--;
            } while (i > 0);
        }

    } /* end of if(j > 0) */

    return 0;
}
