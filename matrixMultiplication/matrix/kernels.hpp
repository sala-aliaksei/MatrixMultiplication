#include <cstddef> // for size_t

// TODO: Should be private info
constexpr std::size_t block_size   = 32;
constexpr std::size_t block_size_j = 32; // bigger than 32 lead to bad asm, fma reading mem

namespace kernels
{

// Kernels, wrap to namespace.
void kernelMulMatrix_BL_NV(double*           r,
                           const double*     a,
                           const double*     b,
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
