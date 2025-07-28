#!/bin/bash

if [ -z "$1" ]; then
    # TEST_NAME="MatrixMulTest.matMulAutotune"
    TEST_NAME="MatrixMulTest.matMulPadding"
else
    TEST_NAME=$1
fi


if [ -z "$2" ]; then
    Matrix_Size="2880"
else
    Matrix_Size=$2
fi


WORKSPACE=$(realpath $(dirname $0)/..)

MATRIX_DIM=$Matrix_Size ${WORKSPACE}/build/UT_Matmul --gtest_filter=$TEST_NAME --gtest_output=xml:${WORKSPACE}/build/results/UT_Matmul.xml
