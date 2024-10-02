#include "Matrix.hpp"

bool operator==(const MatrixSet& s1, const MatrixSet& s2)
{
    return s1.c == s2.c;
}

MatrixSet initMatrix(int n, int m)
{
    MatrixSet set{.a = Matrix<double>(n, m), .b = Matrix<double>(n, m), .c = Matrix<double>(n, m)};

    auto row_cnt = set.a.row();
    auto col_cnt = set.a.col();

    for (auto i = 0; i < row_cnt; ++i)
    {
        for (auto j = 0; j < col_cnt; ++j)
        {
            set.a[i * col_cnt + j] = i;
            set.b[i * col_cnt + j] = j;
        }
    }
    return set;
}

MatrixSet initMatrix()
{
    // TODO: Set matrix size
    MatrixSet set;

    auto row_cnt = set.a.row();
    auto col_cnt = set.a.col();

    for (auto i = 0; i < row_cnt; ++i)
    {
        for (auto j = 0; j < col_cnt; ++j)
        {
            set.a[i * col_cnt + j] = i;
            set.b[i * col_cnt + j] = j;
        }
    }
    return set;
}
