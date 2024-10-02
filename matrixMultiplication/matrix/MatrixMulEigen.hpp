#pragma once
#include <Eigen/Dense>

struct MatrixEigenSet
{
    Eigen::MatrixXd a;
    Eigen::MatrixXd b;
    Eigen::MatrixXd c;
};

void           matrixMulEigen(MatrixEigenSet& ms);
MatrixEigenSet initEigenMatrix(std::size_t rows, std::size_t cols);
