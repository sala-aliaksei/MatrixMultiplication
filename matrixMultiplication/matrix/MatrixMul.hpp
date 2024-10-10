#pragma once
#include "Matrix.hpp"
#include "kernels.hpp"
#include <vector>
#include <future>
#include <optional>

template<bool is_transposed>
struct MulMatrixOnThread
{

    std::size_t _num_threads;

    MulMatrixOnThread(std::size_t num_threads)
      : _num_threads(num_threads)

    {
    }

    template<typename T, typename Kernel>
    void run(Matrix<T>&       cc,
             const Matrix<T>& aa,
             const Matrix<T>& bb,
             std::size_t      thread_num,
             Kernel&&         kernel_mul) const
    {
        double*       c = cc.data();
        const double* b = bb.data();
        const double* a = aa.data();

        auto i_size = aa.row();
        auto k_size = aa.col();
        auto j_size = bb.col();

        std::size_t step  = i_size / _num_threads;
        std::size_t start = thread_num * step;
        std::size_t last  = thread_num == (_num_threads - 1) ? i_size : (thread_num + 1) * step;

        // block_size must be constant, significantly impact on performance
        constexpr std::size_t block_size = 8;

        // TODO: Check if compiler inline kernels
        for (int i = start; i < last; i += block_size)
        {
            for (int j = 0; j < j_size; j += block_size)
            {
                for (int k = 0; k < k_size; k += block_size)
                {
                    if constexpr (is_transposed)
                    {
                        kernel_mul(&c[i * j_size + j],
                                   &a[i * k_size + k],
                                   &b[j * k_size + k],
                                   block_size,
                                   j_size,
                                   k_size);
                    }
                    else
                    {
                        kernel_mul(&c[i * j_size + j],
                                   &a[i * k_size + k],
                                   &b[k * j_size + j],
                                   block_size,
                                   j_size,
                                   k_size);
                    }
                }
            }
        }
    }
};

struct MatrixMulConfig
{
    std::size_t num_threads; // get from runtime
    std::size_t block_size;  // TODO: change to bool
    bool        transpose_matrix;
    bool        manual_vectorization;
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
        std::size_t              step = a.row() / _cfg.num_threads;
        std::optional<Matrix<T>> transposed;
        if (_cfg.transpose_matrix)
        {
            transposed = transpose(b);
        }

        auto& bb = _cfg.transpose_matrix ? *transposed : b;

        std::vector<std::future<void>> fret(_cfg.num_threads);

        auto exec = [&](const std::size_t tid, const std::launch policy)
        {
            if (_cfg.block_size != 1 && _cfg.manual_vectorization)
            {
                if (_cfg.transpose_matrix)
                {
                    MulMatrixOnThread<true> mt(_cfg.num_threads);

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_VT_BL_TP); });
                }
                else
                {
                    MulMatrixOnThread<false> mt(_cfg.num_threads);

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_VT_BL); });
                }
            }
            else // vectorization isn't possible for block_size == 1
            {
                if (_cfg.transpose_matrix)
                {
                    MulMatrixOnThread<true> mt(_cfg.num_threads);

                    fret[tid] =
                      std::async(policy,
                                 [mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_TP_BL_NV); });
                }
                else
                {
                    MulMatrixOnThread<false> mt(_cfg.num_threads);

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
