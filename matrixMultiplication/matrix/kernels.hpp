#include <cstddef> // for size_t

namespace kernels
{

// Kernels, wrap to namespace.
void kernelMulMatrix_BL_NV(double*           r,
                           const double*     a,
                           const double*     b,
                           const std::size_t block_size,
                           const std::size_t j_size,
                           const std::size_t k_size);
void kernelMulMatrix_TP_BL_NV(double*           r,
                              const double*     a,
                              const double*     b,
                              const std::size_t block_size,
                              const std::size_t j_size,
                              const std::size_t k_size);
void kernelMulMatrix_VT_BL_TP(double*           r,
                              const double*     a,
                              const double*     b,
                              const std::size_t block_size,
                              const std::size_t j_size,
                              const std::size_t k_size);
void kernelMulMatrix_VT_BL(double*           r,
                           const double*     a,
                           const double*     b,
                           const std::size_t block_size,
                           const std::size_t j_size,
                           const std::size_t k_size);
} // namespace kernels
