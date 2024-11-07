#include "Matrix.hpp"
#include <random>

bool operator==(const MatrixSet& s1, const MatrixSet& s2)
{
    return s1.c == s2.c;
}

// TODO: What if we have matrix<float>?
// we need i,j,k for matrix init
MatrixSet initMatrix(int isize, int jsize, int ksize)
{
    std::random_device               rd;
    std::mt19937                     gen(rd());
    double                           lower_bound = 0.0;
    double                           upper_bound = 1000.0;
    std::uniform_real_distribution<> dis(lower_bound, upper_bound);

    MatrixSet set{.a = Matrix<double>(isize, ksize),
                  .b = Matrix<double>(ksize, jsize),
                  .c = Matrix<double>(isize, jsize)};

    for (auto i = 0; i < set.a.row(); ++i)
    {
        for (auto j = 0; j < set.a.col(); ++j)
        {
            set.a[i * set.a.col() + j] = ((int)dis(gen));
        }
    }

    for (auto i = 0; i < set.b.row(); ++i)
    {
        for (auto j = 0; j < set.b.col(); ++j)
        {
            set.b[i * set.b.col() + j] = ((int)dis(gen));
        }
    }

    return set;
}
