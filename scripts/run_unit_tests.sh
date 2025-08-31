#!/bin/bash

if [ -z "$1" ]; then
    TEST_NAME="MatrixMulTest.matMulAutotune"
    #TEST_NAME="MatrixMulTest.matMulPadding"
else
    TEST_NAME=$1
fi


if [ -z "$2" ]; then
    Matrix_Size="3072"
else
    Matrix_Size=$2
fi


if [ -z "$3" ]; then
    #UT_Name="UT_Matmul"
    UT_Name="UT_Matmul_Zen5"
else
    UT_Name=$3
fi

WORKSPACE=$(realpath $(dirname $0)/..)

MATRIX_DIM=$Matrix_Size ${WORKSPACE}/build/${UT_Name} --gtest_filter=$TEST_NAME --gtest_output=xml:${WORKSPACE}/build/results/${UT_Name}.xml
