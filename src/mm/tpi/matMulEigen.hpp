#pragma once
#include <Eigen/Dense>

struct MatrixEigenSet
{
    Eigen::MatrixXd a;
    Eigen::MatrixXd b;
    Eigen::MatrixXd c;
};

void           matrixMulEigen(MatrixEigenSet& ms);
MatrixEigenSet initEigenMatrix(std::size_t isize, std::size_t jsize, std::size_t ksize);
