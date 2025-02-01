#pragma once
#include "Matrix.hpp"
#include "kernels.hpp"
#include <vector>
#include <future>
#include <optional>

namespace cppnow
{
void matMul_Naive(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Naive_Order(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Naive_Block(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Simd_Global();

void matMul_Simd(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Avx_Cache(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx_Cache_Regs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx_Cache_Regs_Unroll(const Matrix<double>& A,
                                  const Matrix<double>& B,
                                  Matrix<double>&       C);
void matMul_Avx_Cache_Regs_UnrollRW(const Matrix<double>& A,
                                    const Matrix<double>& B,
                                    Matrix<double>&       C);

void matMul_Avx_Cache_Regs_Unroll_BPack(const Matrix<double>& A,
                                        const Matrix<double>& B,
                                        Matrix<double>&       C);

void matMul_Avx_Cache_Regs_Unroll_MT(const Matrix<double>& A,
                                     const Matrix<double>& B,
                                     Matrix<double>&       C);

void matMul_Avx_Cache_Regs_Unroll_BPack_MT(const Matrix<double>& A,
                                           const Matrix<double>& B,
                                           Matrix<double>&       C);

void matMul_Avx_AddRegs(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
void matMul_Avx_AddRegsV2(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Avx_AddRegs_Unroll(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

void matMul_Avx_Unroll_Cache_Regs(const Matrix<double>& A,
                                  const Matrix<double>& B,
                                  Matrix<double>&       C);

void matMul_Avx_Unroll_Cache_Regs_Rename(const Matrix<double>& A,
                                         const Matrix<double>& B,
                                         Matrix<double>&       C);

void matMul_Avx_Unroll_Cache_Regs_Rename_PackB(const Matrix<double>& A,
                                               const Matrix<double>& B,
                                               Matrix<double>&       C);

void matMul_Avx_Unroll_Cache_Regs_Rename_ReorderAB(const Matrix<double>& A,
                                                   const Matrix<double>& B,
                                                   Matrix<double>&       C);

void matMul_Avx_Unroll_Cache_Regs_Rename_ReorderAB_Multithreads(const Matrix<double>& A,
                                                                const Matrix<double>& B,
                                                                Matrix<double>&       C);
} // namespace cppnow

template<bool is_transposed>
struct MulMatrixOnThread
{
    template<typename T, typename Kernel>
    void run(Matrix<T>&       cc,
             const Matrix<T>& aa,
             const Matrix<T>& bb,
             std::size_t      thread_num,
             Kernel&&         kernel_mul) const
    {
        // must use the same var as DynamicMatrixMul when create threads
        std::size_t num_threads = std::thread::hardware_concurrency();

        double*       c = cc.data();
        const double* b = bb.data();
        const double* a = aa.data();

        auto i_size = aa.row();
        auto k_size = aa.col();
        auto j_size = bb.col();

        // TODO : add tail computation
        // TODO: block=1 is broken

        auto block_inc = block_size_i * num_threads;

        int i = block_size_i * thread_num;
        for (; i < i_size; i += block_inc)
        {
            if constexpr (is_transposed)
            {
                for (int j = 0; j < j_size; j += block_size_j)
                {
                    for (int k = 0; k < k_size; k += block_size_k)
                    {
                        kernel_mul(&c[i * j_size + j],
                                   &a[i * k_size + k],
                                   &b[j * k_size + k],
                                   j_size,
                                   k_size);
                    }
                }
            }
            else
            {
                for (int k = 0; k < k_size; k += block_size_k)
                {
                    for (int j = 0; j < j_size; j += block_size_j)
                    {
                        kernel_mul(&c[i * j_size + j],
                                   &a[i * k_size + k],
                                   &b[k * j_size + j],
                                   i_size,
                                   j_size,
                                   k_size,
                                   i,
                                   j,
                                   k);
                    }
                    // add tail computation for jtail, j loop
                }
                // add tail computation for ktail, k,j loop.  still block opt can be applied
            }
        }
    }
    // add tail computation for itail, i,k,j loop. still block opt can be applied
};

struct MatrixMulConfig
{
    bool enable_par_opt;
    bool enable_block_opt;
    bool transpose_matrix;
    bool enable_vectorization;
};

struct DynamicMatrixMul
{
    MatrixMulConfig _cfg;
    DynamicMatrixMul(MatrixMulConfig cfg)
      : _cfg(std::move(cfg))
    {
    }

    template<typename T = double>
    void operator()(Matrix<T>& a, Matrix<T>& b, Matrix<T>& c) const
    {
        std::optional<Matrix<T>> transposed;
        if (_cfg.transpose_matrix)
        {
            transposed = transpose(b);
        }

        auto& bb = _cfg.transpose_matrix ? *transposed : b;

        // TODO: Use enable_par_opt
        std::size_t num_threads = std::thread::hardware_concurrency();

        std::vector<std::future<void>> fret(num_threads);

        auto exec = [&](const std::size_t tid, const std::launch policy)
        {
            if (_cfg.enable_block_opt && _cfg.enable_vectorization)
            {
                if (_cfg.transpose_matrix)
                {
                    MulMatrixOnThread<true> mt;

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_VT_BL_TP); });
                }
                else
                {
                    MulMatrixOnThread<false> mt;

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_VT_BL); });
                }
            }
            else // vectorization isn't possible without block optimization
            {
                if (_cfg.transpose_matrix)
                {
                    MulMatrixOnThread<true> mt;

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_TP_BL_NV); });
                }
                else
                {
                    MulMatrixOnThread<false> mt;

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_BL_NV); });
                }
            }
        };

        for (std::size_t tid = 0; tid < fret.size() - 1; ++tid)
        {
            exec(tid, std::launch::async);
        }

        exec(fret.size() - 1, std::launch::deferred);
        fret[fret.size() - 1].wait();
    }
};
