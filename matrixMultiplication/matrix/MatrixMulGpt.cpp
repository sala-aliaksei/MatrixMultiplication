
#include "MatrixMulGpt.hpp"
#include <immintrin.h>
#include <thread>

constexpr auto        nthreads   = 4u;
constexpr auto        SM         = 32; //(64 / sizeof(double));
constexpr std::size_t BLOCK_SIZE = SM;

static double sumElemFromReg(__m256d c_vec)
{
    alignas(32) double c_array[4];
    _mm256_store_pd(c_array, c_vec);
    return c_array[0] + c_array[1] + c_array[2] + c_array[3];
}

/*****************     Chat GTP kernel   *****************************/

static void gpt_kernel(const Matrix<double>& A,
                       const Matrix<double>& B,
                       Matrix<double>&       C,
                       size_t                ii,
                       size_t                jj,
                       size_t                kk,
                       size_t                end_row)
{
    size_t i_max = std::min(ii + BLOCK_SIZE, end_row);
    size_t j_max = std::min(jj + BLOCK_SIZE, C.col());
    size_t k_max = std::min(kk + BLOCK_SIZE, A.col());
    for (size_t i = ii; i < i_max; ++i)
    {
        for (size_t j = jj; j < j_max; ++j)
        {
            __m256d c_vec = _mm256_setzero_pd();
            size_t  k;
            for (k = kk; k + 3 < k_max; k += 4)
            {
                __m256d a_vec = _mm256_loadu_pd(&A(i, k));
                __m256d b_vec = _mm256_loadu_pd(&B(j, k));
                c_vec         = _mm256_fmadd_pd(a_vec, b_vec, c_vec);
            }
            // Handle remaining elements
            //            double c_sum = 0.0;
            //            for (; k < k_max; ++k)
            //            {
            //                c_sum += A(i, k) * B(j, k);
            //            }

            // Horizontal addition of c_vec
            double c_sum = sumElemFromReg(c_vec);
            C(i, j) += c_sum;
        }
    }
}

static void multiply_transposed(const Matrix<double>& A,
                                const Matrix<double>& B,
                                Matrix<double>&       C,
                                size_t                start_row,
                                size_t                end_row)
{
    size_t n = A.row();
    size_t m = A.col(); // Same as B.rows
    size_t p = B.row();

    for (size_t ii = start_row; ii < end_row; ii += BLOCK_SIZE)
    {
        for (size_t jj = 0; jj < p; jj += BLOCK_SIZE)
        {
            for (size_t kk = 0; kk < m; kk += BLOCK_SIZE)
            {
                gpt_kernel(A, B, C, ii, jj, kk, end_row);
            }
        }
    }
}

// Optimized matrix multiplication function using std::thread, AVX2 intrinsics, and prefetching
void gpt_matrix_multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // Check for compatible dimensions
    if (A.col() != B.row())
    {
        throw std::invalid_argument("Incompatible matrix dimensions.");
    }

    size_t n = A.row();

    // Initialize the result matrix C
    // C = Matrix<double>(n, B.col());

    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1; // Fallback to 1 thread if detection fails

    // Split the work among threads
    std::vector<std::thread> threads;

    size_t rows_per_thread = n / num_threads;
    size_t extra_rows      = n % num_threads;
    size_t start_row       = 0;

    Matrix B_T = transpose(B);

    for (unsigned int t = 0; t < num_threads; ++t)
    {
        size_t end_row = start_row + rows_per_thread + (t < extra_rows ? 1 : 0);
        threads.emplace_back(
          multiply_transposed, std::cref(A), std::cref(B_T), std::ref(C), start_row, end_row);
        start_row = end_row;
    }

    // Join threads
    for (auto& thread : threads)
    {
        thread.join();
    }
}

/********************   V2  ***********************/

// Define block sizes for cache optimization
constexpr size_t MC = 256; // L2 cache blocking
constexpr size_t KC = 128; // L1 cache blocking
constexpr size_t NC = 256; // L3 cache blocking

// Micro-kernel block size
constexpr size_t MR = 4; // Rows in micro-kernel
constexpr size_t NR = 4; // Columns in micro-kernel

// Align data to 32 bytes
// constexpr size_t ALIGN_SIZE = 32;

// Function to allocate aligned memory
template<typename T>
T* aligned_alloc(size_t size)
{
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGN_SIZE, size * sizeof(T)) != 0)
    {
        throw std::bad_alloc();
    }
    return reinterpret_cast<T*>(ptr);
}

// Micro-kernel for matrix multiplication (MR x NR)
void micro_kernel(size_t kc,
                  const double* __restrict__ A_block,
                  const double* __restrict__ B_block,
                  double* __restrict__ C_block,
                  size_t inc_row_C,
                  size_t inc_col_C)
{
    __m256d C[MR][NR];

    // Initialize C registers to zero
    for (size_t i = 0; i < MR; ++i)
    {
        for (size_t j = 0; j < NR; ++j)
        {
            C[i][j] = _mm256_setzero_pd();
        }
    }

    // Main computation loop
    for (size_t p = 0; p < kc; ++p)
    {
        __m256d B_col[NR];
        for (size_t j = 0; j < NR; ++j)
        {
            B_col[j] = _mm256_load_pd(&B_block[p * NR * 4 + j * 4]);
        }

        for (size_t i = 0; i < MR; ++i)
        {
            __m256d A_elem = _mm256_broadcast_sd(&A_block[i * kc + p]);
            for (size_t j = 0; j < NR; ++j)
            {
                C[i][j] = _mm256_fmadd_pd(A_elem, B_col[j], C[i][j]);
            }
        }
    }

    // Store results back to C
    for (size_t i = 0; i < MR; ++i)
    {
        for (size_t j = 0; j < NR; ++j)
        {
            _mm256_store_pd(
              &C_block[i * inc_row_C + j * inc_col_C],
              _mm256_add_pd(_mm256_load_pd(&C_block[i * inc_row_C + j * inc_col_C]), C[i][j]));
        }
    }
}

// Pack block of A into continuous memory
void pack_A(size_t mc, size_t kc, const Matrix<double>& A, double* A_block, size_t i, size_t k)
{
    for (size_t ii = 0; ii < mc; ++ii)
    {
        for (size_t p = 0; p < kc; ++p)
        {
            A_block[ii * kc + p] = A(i + ii, k + p);
        }
    }
}

// Pack block of B into continuous memory
void pack_B(size_t kc, size_t nc, const Matrix<double>& B, double* B_block, size_t k, size_t j)
{
    for (size_t p = 0; p < kc; ++p)
    {
        for (size_t jj = 0; jj < nc; jj += 4)
        {
            _mm256_store_pd(&B_block[p * nc + jj], _mm256_loadu_pd(&B(k + p, j + jj)));
        }
    }
}

// Multiply macro blocks
void gemm_macro_kernel(size_t          mc,
                       size_t          nc,
                       size_t          kc,
                       const double*   A_block,
                       const double*   B_block,
                       Matrix<double>& C,
                       size_t          i,
                       size_t          j)
{
    for (size_t ii = 0; ii < mc; ii += MR)
    {
        for (size_t jj = 0; jj < nc; jj += NR * 4)
        {
            micro_kernel(kc, &A_block[ii * kc], &B_block[jj], &C(i + ii, j + jj), C.cols, 4);
        }
    }
}

// Function to multiply matrices using cache blocking and micro-kernels
static void multiply_block(const Matrix<double>& A,
                           const Matrix<double>& B,
                           Matrix<double>&       C,
                           size_t                start_i,
                           size_t                end_i)
{
    size_t m = A.rows;
    size_t n = B.cols;
    size_t k = A.cols;

    double* A_block = aligned_alloc<double>(MC * KC);
    double* B_block = aligned_alloc<double>(KC * NC);

    for (size_t j = 0; j < n; j += NC)
    {
        size_t nc = std::min(NC, n - j);
        for (size_t l = 0; l += KC, l < k;)
        {
            size_t kc = std::min(KC, k - l);
            pack_B(kc, nc, B, B_block, l, j);
            for (size_t i = start_i; i < end_i; i += MC)
            {
                size_t mc = std::min(MC, end_i - i);
                pack_A(mc, kc, A, A_block, i, l);
                gemm_macro_kernel(mc, nc, kc, A_block, B_block, C, i, j);
            }
        }
    }

    free(A_block);
    free(B_block);
}

// Optimized matrix multiplication function
void matrix_multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C)
{
    // Check for compatible dimensions
    if (A.cols != B.rows)
    {
        throw std::invalid_argument("Incompatible matrix dimensions.");
    }

    size_t m = A.rows;

    // Determine the number of hardware threads available
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1; // Fallback to 1 thread if detection fails

    // Split the work among threads
    std::vector<std::thread> threads;

    size_t rows_per_thread = (m + num_threads - 1) / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t)
    {
        size_t start_i = t * rows_per_thread;
        size_t end_i   = std::min(start_i + rows_per_thread, m);

        threads.emplace_back(
          multiply_block, std::cref(A), std::cref(B), std::ref(C), start_i, end_i);
    }

    // Join threads
    for (auto& thread : threads)
    {
        thread.join();
    }
}
