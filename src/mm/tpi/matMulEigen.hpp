#pragma once
#include <mm/core/Matrix.hpp>
#include <Eigen/Dense>

struct MatrixEigenSet
{
    Eigen::MatrixXd a;
    Eigen::MatrixXd b;
    Eigen::MatrixXd c;
};

void           matrixMulEigen(MatrixEigenSet& ms);
MatrixEigenSet initEigenMatrix(const Matrix<double>& a, const Matrix<double>& b);
MatrixEigenSet initEigenMatrix(int M, int N, int K);
