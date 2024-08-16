#include "Matrix.hpp"

bool operator==(const MatrixSet& s1, const MatrixSet& s2)
{
    return s1.res == s2.res;
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
