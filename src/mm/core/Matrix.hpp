#pragma once
#include <iostream>
#include <iomanip> // for std::setw

#include <vector>
#include <boost/align/aligned_allocator.hpp>

#ifdef __cpp_lib_hardware_interference_size
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
// 64 bytes on x86-64 │ L1_CACHE_BYTES │ L1_CACHE_SHIFT │ __cacheline_aligned │ ...
constexpr std::size_t hardware_constructive_interference_size = 64;
constexpr std::size_t hardware_destructive_interference_size  = 64;
#endif

// Align data to 64 bytes
constexpr size_t ALIGN_SIZE = hardware_destructive_interference_size;

template<typename T>
using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, ALIGN_SIZE>>;

template<typename T>
class Matrix
{
  public:
    using value_type = T;

    Matrix() = default;

    Matrix(std::size_t row_cnt, std::size_t col_cnt)
      : rows(row_cnt)
      , cols(col_cnt)
      , _matrix(rows * cols, 0)
    {
    }

    [[__nodiscard__]] T operator[](std::size_t idx) const noexcept
    {
        return _matrix[idx];
    }

    [[__nodiscard__]] T& operator[](std::size_t idx) noexcept
    {
        return _matrix[idx];
    }

    [[__nodiscard__]] constexpr T* data() noexcept
    {
        return _matrix.data();
    }

    [[__nodiscard__]] constexpr const T* data() const noexcept
    {
        return _matrix.data();
    }

    [[__nodiscard__]] std::size_t size() const noexcept
    {
        return cols * rows;
    }

    [[__nodiscard__]] std::size_t col() const noexcept
    {
        return cols;
    }

    [[__nodiscard__]] std::size_t row() const noexcept
    {
        return rows;
    }

    const T& operator()(std::size_t i, std::size_t j) const noexcept
    {
        return _matrix[i * cols + j];
    }

    T& operator()(std::size_t i, std::size_t j)
    {
        return _matrix[i * cols + j];
    }

    std::size_t rows;
    std::size_t cols;

  private:
    aligned_vector<T> _matrix;
};

template<typename T>
Matrix<T> transpose(const Matrix<T> m)
{
    auto col_cnt = m.col();
    auto row_cnt = m.row();

    Matrix<T> transposed(col_cnt, row_cnt);

    for (auto i = 0; i < row_cnt; ++i)
    {
        for (auto j = 0; j < col_cnt; ++j)
        {
            // TODO: vectorize transpose
            //_mm_prefetch(&ms.b[(j + 1) * N], _MM_HINT_NTA);
            // we don't need b in cache, transposed should be in cache,
            // reading from b can be streamed
            transposed[i * col_cnt + j] = m[j * row_cnt + i];
        }
    }
    return transposed;
}

template<typename Stream, typename T>
Stream& operator<<(Stream& os, const Matrix<T>& m)
{
    constexpr int width = 6; // Adjust width as needed
    for (auto i = 0; i < m.row(); ++i)
    {
        for (auto j = 0; j < m.col(); ++j)
        {
            os << std::setw(width) << m[i * m.col() + j] << ", ";
        }
        os << "\n";
    }
    return os;
}

template<typename T>
bool operator==(const Matrix<T>& s1, const Matrix<T>& s2)
{
    auto row_cnt = s1.row();
    auto col_cnt = s1.col();

    if (col_cnt != s2.col())
        return false;

    if (row_cnt != s2.row())
        return false;

    for (int i = 0; i < row_cnt; i++)
    {
        for (int j = 0; j < col_cnt; j++)
        {
            if constexpr (std::is_floating_point_v<T>)
            {
                if (std::abs(s1[i * col_cnt + j] - s2[i * col_cnt + j]) > __DBL_EPSILON__)
                {
                    std::cout << "elem[" << i << "][" << j << "] doesn't match. "
                              << s1[i * col_cnt + j] << " != " << s2[i * col_cnt + j] << std::endl;
                    return false;
                }
            }
            else
            {
                return s1[i * col_cnt + j] == s2[i * col_cnt + j];
            }
        }
    }

    return true;
}

template<typename T>
bool operator!=(const Matrix<T>& s1, const Matrix<T>& s2)
{
    return !(s1 == s2);
}

// TODO: What if we have matrix<float>?
struct MatrixSet
{
    using value_type = double;
    Matrix<value_type> a;
    Matrix<value_type> b;
    Matrix<value_type> c;
};

MatrixSet initDoubleMatrix(int i, int j, int k);
MatrixSet initPredictedMatrix(int i, int j, int k);

template<typename Stream>
Stream& operator<<(Stream& os, MatrixSet& s1)
{
    os << s1.c;
    return os;
}

bool operator==(const MatrixSet&, const MatrixSet&);

template<typename T>
Matrix<T> generateRandomMatrix(int M, int N);

template<typename T>
Matrix<T> generateIotaMatrix(int M, int N);