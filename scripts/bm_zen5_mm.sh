#!/bin/bash

CURR_DIR=$(dirname $0)
pushd $CURR_DIR > /dev/null

./run_unit_tests.sh MatrixMulZen5Test.mm 3072
./run_unit_tests.sh MatrixMulZen5Test.mmblocking 3072 
./run_unit_tests.sh MatrixMulZen5Test.submatrix 3072
./run_unit_tests.sh MatrixMulZen5Test.mdspan 3072 
#./run_unit_tests.sh MatrixMulZen5Test.mdspan_l1 3072 (doesn't exist)

./run_unit_tests.sh MatrixMulZen5Float32Test.MatMulZen5 3072
./run_unit_tests.sh MatrixMulZen5Float32Test.MatMulZen5MTBlocking 3072
./run_unit_tests.sh MatrixMulZen5Float32Test.MatMulZen5MTBlockingTails 3072

# ./run_unit_tests.sh MatrixMulBFP16Test.MatMulZen5 3072  TestMatmul_BFP16 #slow since using naive implementation
# ./run_unit_tests.sh MatrixMulBFP16Test.MatMulZen5MTBlocking 3072 TestMatmul_BFP16  #slow since using naive implementation


## BENCHMARKS ##

./run_benchmark.sh "mm::zen5::matMulZen5" 3072
./run_benchmark.sh "mm::zen5::matMulZen5MTBlocking" 3072 
./run_benchmark.sh "mm::zen5::matMulZen5MTBlockingSpan" 3072 
./run_benchmark.sh "mm::zen5::matMulZen5MTBlockingTails" 3072

./run_benchmark.sh "mm::zen5::matMulZen5" 3072  BM_Matmul_Float
./run_benchmark.sh "mm::zen5::matMulZen5MTBlocking" 3072 BM_Matmul_Float
./run_benchmark.sh "mm::zen5::matMulZen5MTBlockingTails" 3072 BM_Matmul_Float
./run_benchmark.sh mm::tpi::matrixMulOpenBlas 3072  BM_Matmul_Float


./run_benchmark.sh "mm::zen5::matMulZen5" 3072 BM_Matmul_BFP16
./run_benchmark.sh "mm::zen5::matMulZen5MTBlocking" 3072 BM_Matmul_BFP16



./run_benchmark.sh mm::tpi::matrixMulOpenBlas 3072 BM_Matmul
# ./run_benchmark.sh matmulBlis 3072 BM_Matmul (not enabled by default)

popd > /dev/null