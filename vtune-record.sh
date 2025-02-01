
BENCHMARK_LIST=(
    
#    "BM_CN_MatMulNaive"
#    "BM_CN_MatMulNaive_Order"
#    "BM_CN_MatMulNaive_Block"
#    "BM_CN_MatMul_Simd"
#    "BM_CN_MatMul_Avx"
#    "BM_CN_MatMul_Avx_AddRegs"
#    "BM_CN_MatMul_Avx_AddRegsV2"
#    "BM_CN_MatMul_Avx_AddRegs_Unroll"
#    "BM_CN_MatMul_Avx_Cache"
#    "BM_CN_MatMul_Avx_Cache_Regs"
#    "BM_CN_MatMul_Avx_Cache_Regs_Unroll"
#    "BM_CN_MatMul_Avx_Cache_Regs_UnrollRW"
#    "BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack"
#    "BM_CN_MatMul_Avx_Cache_Regs_Unroll_MT"
   "BM_CN_MatMul_Avx_Cache_Regs_Unroll_BPack_MT"
)

set -e
set -x

pushd build > /dev/null

WORKSPACE="/home/alex/workspace/cpp-projects/MatrixMultiplication/"
VTUNE_RES_DIR="${WORKSPACE}/perf-result/vtune-result"


export MATRIX_DIM=2880

for bm in "${BENCHMARK_LIST[@]}"; do
    mkdir -p $VTUNE_RES_DIR/${bm}
    rm -rf  $VTUNE_RES_DIR/${bm}/*
    echo "vtune bm = ${bm}"
    /home/alex/intel/oneapi/vtune/2025.0/bin64/vtune \
        -collect uarch-exploration \
        -knob sampling-interval=0.1 \
        -knob collect-bad-speculation=false \
        -knob collect-memory-bandwidth=true \
        --app-working-dir=/home/alex/workspace/cpp-projects/MatrixMultiplication/build \
        -result-dir ${VTUNE_RES_DIR}/${bm} \
        -- /home/alex/workspace/cpp-projects/MatrixMultiplication/build/Benchmarks --benchmark_filter=${bm}/$MATRIX_DIM
    
    #mv perf.data perf-result/perf_${bm}.data
done


popd > /dev/null
