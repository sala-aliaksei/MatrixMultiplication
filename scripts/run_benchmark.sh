#/bin/bash

# Args:
# add name of benchmark
# add size of the matrix


if [ -z "$1" ]; then
    BM_Name="BM_MatMulSimd"
else
    BM_Name=$1
fi

if [ -z "$2" ]; then
    Matrix_Size="2880"
else
    Matrix_Size=$2
fi

if [ -z "$3" ]; then
    #BM_Name="BM_Matmul"
    BM_BIN_Name="BM_Matmul_Zen5"
else
    BM_BIN_Name=$3
fi

# mkdir -p ./build/results/$BM_Name
# --benchmark_out=./build/results/$BM_Name/$Matrix_Size.json
WORKSPACE=$(realpath $(dirname $0)/..)

MATRIX_DIM=$Matrix_Size perf stat -d -d ${WORKSPACE}/build/${BM_BIN_Name} --benchmark_filter=$BM_Name/$Matrix_Size$ --benchmark_time_unit=ms  --benchmark_repetitions=10 --benchmark_report_aggregates_only=false
