#include <cstddef> // for size_t

// TODO: Block size should be private info
// looks like b[32x32] fit into l1 cache
constexpr std::size_t block_size_i = 48; // 4;
constexpr std::size_t block_size_j = 48; // 12;
constexpr std::size_t block_size_k = 48; // 8;

namespace kernels
{

void matmul_NV(double* __restrict c,
               const double* __restrict a,
               const double* __restrict mb,
               const std::size_t i_size,
               const std::size_t j_size,
               const std::size_t k_size);

void matmul_TP_NV(double* __restrict c,
                  const double* __restrict a,
                  const double* __restrict mb,
                  const std::size_t i_size,
                  const std::size_t j_size,
                  const std::size_t k_size);

// Kernels, wrap to namespace.
void kernelMulMatrix_BL_NV(double* __restrict r,
                           const double* __restrict a,
                           const double* __restrict b,
                           const std::size_t j_size,
                           const std::size_t k_size);

void kernelMulMatrix_TP_BL_NV(double*           r,
                              const double*     a,
                              const double*     b,
                              const std::size_t j_size,
                              const std::size_t k_size);

void kernelMulMatrix_VT_BL_TP(double*           r,
                              const double*     a,
                              const double*     b,
                              const std::size_t j_size,
                              const std::size_t k_size);

void kernelMulMatrix_VT_BL(double*           r,
                           const double*     a,
                           const double*     b,
                           const std::size_t j_size,
                           const std::size_t k_size);
} // namespace kernels
