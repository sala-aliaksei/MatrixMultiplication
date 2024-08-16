#pragma once
#include "Matrix.hpp"
#include <vector>
#include <future>
#include <optional>

// Kernels, wrap to namespace.
void kernelMulMatrix_BL_NV(double* r, const double* a, const double* b, std::size_t block_size);

void kernelMulMatrix_TP_BL_NV(double* r, const double* a, const double* b, std::size_t block_size);

void kernelMulMatrix_VT_BL_TP(double* r, const double* a, const double* b, std::size_t block_size);

void kernelMulMatrix_VT_BL(double* r, const double* a, const double* b, std::size_t block_size);

// should take compiletime param only, optimized, parametrized matrix mul
template<bool is_transposed, std::size_t block_size, bool enable_manual_vectorization = true>
struct MulMatrixOnThread
{
    std::size_t _num_threads;

    MulMatrixOnThread(std::size_t num_threads)
      : _num_threads(num_threads)
    {
        // TODO: add more static_assert
        if constexpr (block_size == 1)
        {
            static_assert(enable_manual_vectorization == false,
                          "vectorization is not available for block_size == 1");
        }
    }

    template<typename T>
    void run(Matrix<T>& cc, const Matrix<T>& aa, const Matrix<T>& bb, std::size_t thread_num) const
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

        // std::size_t block_size = 8 * _block_cnt;
        // const 8 doesn't help compiler to optimize
        // but block_size as constexpr does

        for (int i = start; i < last; i += block_size)
        {
            for (int j = 0; j < j_size; j += block_size)
            {
                for (int k = 0; k < k_size; k += block_size)
                {
                    if constexpr (is_transposed)
                    {
                        if constexpr (enable_manual_vectorization && block_size % 8 == 0)
                        {
                            kernelMulMatrix_VT_BL_TP(&c[i * j_size + j],
                                                     &a[i * k_size + k],
                                                     &b[j * k_size + k],
                                                     block_size);
                        }
                        else
                        {
                            kernelMulMatrix_TP_BL_NV(&c[i * j_size + j],
                                                     &a[i * k_size + k],
                                                     &b[j * k_size + k],
                                                     block_size);
                        }
                    }
                    else
                    {
                        if constexpr (enable_manual_vectorization)
                        {
                            kernelMulMatrix_VT_BL(&c[i * j_size + j],
                                                  &a[i * k_size + k],
                                                  &b[k * j_size + j],
                                                  block_size);
                        }
                        else
                        {
                            kernelMulMatrix_BL_NV(&c[i * j_size + j],
                                                  &a[i * k_size + k],
                                                  &b[k * j_size + j],
                                                  block_size);
                        }
                    }
                }
            }
        }
    }
};

// cast runtime param to compiletime param
struct MatrixMul
{
    std::size_t _num_threads;
    std::size_t _block_size;
    bool        _transpose_matrix;
    bool        _manual_vectorization;
    MatrixMul(std::size_t num_threads,
              std::size_t block_size,
              bool        transpose_matrix,
              bool        manual_vectorization)
      : _num_threads(num_threads)
      , _block_size(block_size)
      , _transpose_matrix(transpose_matrix)
      , _manual_vectorization(manual_vectorization)

    {
        // TODO: add runtime asserts
        // TODO: use hardware_destructive_interference_size ?

        if ((_block_size != 1) && _manual_vectorization == false && (_block_size % 8 != 0))
        {
            throw std::runtime_error("Invalid block size for matrix = "
                                     + std::to_string(_block_size));
        }
    }

    template<typename T = double>
    void operator()(Matrix<T>& a, Matrix<T>& b, Matrix<T>& c) const
    {
        std::size_t step = a.row() / _num_threads;
        if ((step % _block_size) != 0)
        {
            throw std::runtime_error(
              "Invalid block size per thread (N/thread_cnt)%block size must be zero, block_size= "
              + std::to_string(_block_size));
        }

        std::optional<Matrix<T>> transposed;
        if (_transpose_matrix)
        {
            transposed = transpose(b);
        }

        auto& bb = _transpose_matrix ? *transposed : b;

        std::vector<std::future<void>> fret(_num_threads);
        for (std::size_t tid = 0; tid < fret.size(); ++tid)
        {
            if (_block_size != 1 && _manual_vectorization)
            {
                if (_transpose_matrix)
                {
                    MulMatrixOnThread<true, 8, true> mt(_num_threads);

                    fret[tid] = std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                           { mt.run(c, a, bb, tid); });
                }
                else
                {
                    MulMatrixOnThread<false, 8, true> mt(_num_threads);

                    fret[tid] = std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                           { mt.run(c, a, bb, tid); });
                }
            }
            else // vectorization isn't possible for block_size == 1
            {
                if (_transpose_matrix)
                {
                    MulMatrixOnThread<true, 8, false> mt(_num_threads);

                    fret[tid] = std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                           { mt.run(c, a, bb, tid); });
                }
                else
                {
                    MulMatrixOnThread<false, 8, false> mt(_num_threads);

                    fret[tid] = std::async([mt = std::move(mt), &a, &bb, &c, tid]()
                                           { mt.run(c, a, bb, tid); });
                }
            }
        }

        // TODO: Move to dctor?
        fret.resize(0); // wait all threads
    }
};

// TODO: Check how amount of args affect performance(don't pass j_size, k_size )
struct Kernels final
{
  private:
    void mulMatrix_128VL_BL(double* c, const double* a, const double* b);
    void mulMatrix_256VL_BL(double* c, const double* a, const double* b);

    std::size_t _block_size;
    std::size_t _j_size;
    std::size_t _k_size;

  public:
    Kernels(std::size_t block_size, std::size_t j_size, std::size_t k_size)
      : _block_size(block_size)
      , _j_size(j_size)
      , _k_size(k_size)
    {
    }

    void kernelMulMatrix_BL_NV(double* r, double* a, double* b);

    void kernelMulMatrix_TP_BL_NV(double* r, double* a, double* b);

    void kernelMulMatrix_VT_BL_TP(double* r, double* a, double* b);

    void kernelMulMatrix_VT_BL(double* r, double* a, double* b);
};
