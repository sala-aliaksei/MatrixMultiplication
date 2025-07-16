#!/bin/bash

BENCHMARK_LIST=(
    "BM_CN_MatMulNaive"
    "BM_CN_MatMulNaive_Order"
    "BM_CN_MatMulNaive_Block"
    "BM_CN_MatMul_Simd"
    "BM_CN_MatMul_Avx"
    "BM_CN_MatMul_Avx_AddRegs"
    "BM_CN_MatMul_Avx_AddRegs_Unroll"
    "BM_CN_MatMul_Avx_Cache"
    "BM_CN_MatMul_Avx_Cache_Regs"
    "BM_CN_MatMul_Avx_Cache_Regs_Unroll"
    "BM_CN_MatMul_Avx_Cache_Regs_UnrollRW"
    "BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack"
    "BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT"
    "BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT"
    "BM_MatMulRegOpt"
    "BM_MatMulLoopRepack"
    "BM_MatMulLoopBPacked"
    "BM_MatrixMulOpenBLAS"
    "BM_MatMulSimd"
)

set -e

WORKSPACE=$(realpath $(dirname $0)/..)
PERF_RES_DIR="${WORKSPACE}/output/perf-result"
mkdir -p ${PERF_RES_DIR}
export MATRIX_DIM=2880

for bm in "${BENCHMARK_LIST[@]}"; do
    echo "Benchmark = ${bm}"
    perf record -o ${PERF_RES_DIR}/perf_${bm}.data ${WORKSPACE}/build/BM_Matmul --benchmark_filter=${bm}/$MATRIX_DIM 
done
