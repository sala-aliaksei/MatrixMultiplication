#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <Nc> <Mc> <Kc>"
    exit 1
fi

Nc=$1
Mc=$2
Kc=$3
DIM=$4

rm -f build/CMakeCache.txt && cmake -DN_CACHE_SIZE=$Nc -DM_CACHE_SIZE=$Mc -DK_CACHE_SIZE=$Kc \
    -DCMAKE_BUILD_TYPE=Release -DENABLE_UNIT_TESTS=OFF \
    -DCMAKE_TOOLCHAIN_FILE=./build/Release/generators/conan_toolchain.cmake \
    -B ./build && time cmake --build ./build -j$(nproc) --config Release --target BM_MatmulAutotune

MATRIX_DIM=$DIM perf stat -d -d -d ./build/BM_MatmulAutotune  --benchmark_time_unit=ms