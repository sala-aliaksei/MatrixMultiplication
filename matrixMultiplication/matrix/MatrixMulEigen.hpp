#pragma once
#include <Eigen/Dense>

struct MatrixEigenSet
{
    Eigen::MatrixXd a;
    Eigen::MatrixXd b;
    Eigen::MatrixXd c;
};

void           matrixMulEigen(MatrixEigenSet& ms);
MatrixEigenSet initEigenMatrix();
