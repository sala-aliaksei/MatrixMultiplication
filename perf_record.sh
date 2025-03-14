
BENCHMARK_LIST=(
    
#    "BM_CN_MatMulNaive"
#    "BM_CN_MatMulNaive_Order"
#    "BM_CN_MatMulNaive_Block"
#    "BM_CN_MatMul_Simd"
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

pushd build > /dev/null
mkdir -p perf-result
export MATRIX_DIM=2880

for bm in "${BENCHMARK_LIST[@]}"; do
    echo "bm = ${bm}"
    perf record ./BM_Matmul --benchmark_filter=${bm}/$MATRIX_DIM
    mv perf.data perf-result/perf_${bm}.data
done


popd > /dev/null
