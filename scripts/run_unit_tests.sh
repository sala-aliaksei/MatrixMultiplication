#!/bin/bash

if [ -z "$1" ]; then
    # TEST_NAME="MatrixMulTest.matMulAutotune"
    TEST_NAME="MatrixMulTest.matMulPadding"
else
    TEST_NAME=$1
fi

WORKSPACE=$(realpath $(dirname $0)/..)

${WORKSPACE}/build/UT_Matmul --gtest_filter=$TEST_NAME
