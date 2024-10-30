#include "Matrix.hpp"

bool operator==(const MatrixSet& s1, const MatrixSet& s2)
{
    return s1.c == s2.c;
}

// TODO: What if we have matrix<float>?
// we need i,j,k for matrix init
MatrixSet initMatrix(int isize, int jsize, int ksize)
{
    MatrixSet set{.a = Matrix<double>(isize, ksize),
                  .b = Matrix<double>(ksize, jsize),
                  .c = Matrix<double>(isize, jsize)};

    for (auto i = 0; i < set.a.row(); ++i)
    {
        for (auto j = 0; j < set.a.col(); ++j)
        {
            set.a[i * set.a.col() + j] = i + j;
        }
    }

    for (auto i = 0; i < set.b.row(); ++i)
    {
        for (auto j = 0; j < set.b.col(); ++j)
        {
            set.b[i * set.b.col() + j] = j - i;
        }
    }

    return set;
}
