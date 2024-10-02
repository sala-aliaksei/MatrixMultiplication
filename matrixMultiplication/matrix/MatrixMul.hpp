#pragma once
#include "Matrix.hpp"
#include "kernels.hpp"
#include <vector>
#include <future>
#include <optional>

// should take compile-time param only, optimized, parametrized matrix mul
template<bool is_transposed>
struct MulMatrixOnThread
{
    std::size_t _block_size;
    std::size_t _num_threads;

    MulMatrixOnThread(std::size_t num_threads, std::size_t block_size)
      : _num_threads(num_threads)
      , _block_size(block_size)
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

        // const 8 doesn't help compiler to optimize
        // but block_size as constexpr does
        constexpr std::size_t block_size = 8; // * _block_cnt;

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
    std::size_t num_threads;
    std::size_t block_size;
    bool        transpose_matrix;
    bool        manual_vectorization;
};

struct DynamicMatrixMul
{
    MatrixMulConfig _cfg;
    DynamicMatrixMul(MatrixMulConfig cfg)
      : _cfg(std::move(cfg))
    {
        // TODO: add runtime asserts

        if ((_cfg.block_size != 1) && _cfg.manual_vectorization == false
            && (_cfg.block_size % 8 != 0))
        {
            throw std::runtime_error("Invalid block size for matrix = "
                                     + std::to_string(_cfg.block_size));
        }
    }

    template<typename T = double>
    void operator()(Matrix<T>& a, Matrix<T>& b, Matrix<T>& c) const
    {
        std::size_t step = a.row() / _cfg.num_threads;
        if ((step % _cfg.block_size) != 0)
        {
            // TODO: All param from equation
            throw std::runtime_error(
              "Invalid block size per thread, (N/thread_cnt)%block_size must be zero, block_size= "
              + std::to_string(_cfg.block_size));
        }

        std::optional<Matrix<T>> transposed;
        if (_cfg.transpose_matrix)
        {
            transposed = transpose(b);
        }

        auto& bb = _cfg.transpose_matrix ? *transposed : b;

        std::vector<std::future<void>> fret(_cfg.num_threads);
        for (std::size_t tid = 0; tid < fret.size(); ++tid)
        {
            if (_cfg.block_size != 1 && _cfg.manual_vectorization)
            {
                if (_cfg.transpose_matrix)
                {
                    MulMatrixOnThread<true> mt(_cfg.num_threads, _cfg.block_size);

                    fret[tid] =
                      std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_VT_BL_TP); });
                }
                else
                {
                    MulMatrixOnThread<false> mt(_cfg.num_threads, _cfg.block_size);

                    fret[tid] =
                      std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_VT_BL); });
                }
            }
            else // vectorization isn't possible for block_size == 1
            {
                if (_cfg.transpose_matrix)
                {
                    MulMatrixOnThread<true> mt(_cfg.num_threads, _cfg.block_size);

                    fret[tid] =
                      std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_TP_BL_NV); });
                }
                else
                {
                    MulMatrixOnThread<false> mt(_cfg.num_threads, _cfg.block_size);

                    fret[tid] =
                      std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                 { mt.run(c, a, bb, tid, kernels::kernelMulMatrix_BL_NV); });
                }
            }
        }

        fret.resize(0); // wait all threads
    }
};
