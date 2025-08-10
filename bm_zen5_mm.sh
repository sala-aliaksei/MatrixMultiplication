#!/bin/bash

# ./scripts/run_unit_tests.sh MatrixMulTest.matMulZen5 3072
./scripts/run_unit_tests.sh MatrixMulTest.matMulZen5MTBlocking 3072

./scripts/run_unit_tests.sh MatrixMulFloat32Test.MatMulZen5 3072
./scripts/run_unit_tests.sh MatrixMulFloat32Test.MatMulZen5MTBlocking 3072


./scripts/run_benchmark.sh "mm::zen5::matMulZen5" 3072
./scripts/run_benchmark.sh "mm::zen5::matMulZen5MTBlocking" 3072

./scripts/run_float_benchmark.sh "mm::zen5::matMulZen5" 3072
./scripts/run_float_benchmark.sh "mm::zen5::matMulZen5MTBlocking" 3072

export OPENBLAS_NUM_THREADS=$(nproc)
export OMP_NUM_THREADS=$(nproc)
./scripts/run_benchmark.sh mm::tpi::matrixMulOpenBlas 3072
./scripts/run_float_benchmark.sh mm::tpi::matrixMulOpenBlas 3072

export BLIS_NUM_THREADS=$(nproc)
./scripts/run_benchmark.sh matmulBlis 3072