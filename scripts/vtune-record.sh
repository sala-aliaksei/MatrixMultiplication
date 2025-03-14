#!/bin/bash
BENCHMARK_LIST=(
    "BM_CN_MatMulNaive"
    "BM_CN_MatMulNaive_Order"
    "BM_CN_MatMulNaive_Block"
    "BM_CN_MatMul_Simd"
    "BM_CN_MatMul_Avx"
    "BM_CN_MatMul_Avx_AddRegs"
    "BM_CN_MatMul_Avx_AddRegsV2"
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
)

set -e

if [[ $(sysctl kernel.perf_event_paranoid) != "kernel.perf_event_paranoid = 0" ]]; then
    echo "kernel.perf_event_paranoid is not 0"
    echo "please run 'sudo sysctl -w kernel.perf_event_paranoid=0'"
    exit 1
fi


WORKSPACE=$(realpath $(dirname $0)/..)


VTUNE_RES_DIR="${WORKSPACE}/output/vtune-result"
mkdir -p ${VTUNE_RES_DIR}
VTUNE_BIN="/home/alex/intel/oneapi/vtune/2025.0/bin64/"

export MATRIX_DIM=2880

for bm in "${BENCHMARK_LIST[@]}"; do
    rm -rf  $VTUNE_RES_DIR/${bm}
    mkdir -p $VTUNE_RES_DIR/${bm}
    
    echo "vtune benchmark = ${bm}"
    ${VTUNE_BIN}/vtune \
        -collect uarch-exploration \
        -knob sampling-interval=0.1 \
        -knob collect-bad-speculation=false \
        -knob collect-memory-bandwidth=true \
        --app-working-dir=${WORKSPACE}/build \
        -result-dir ${VTUNE_RES_DIR}/${bm} \
        -- ${WORKSPACE}/build/BM_Matmul --benchmark_filter=${bm}/$MATRIX_DIM
done


