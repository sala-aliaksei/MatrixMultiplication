
import os
import math


# constexpr int Kc = 240;


# Given constants
Nr, Mr, Kr = 12, 4, 1

Nstep = 2*Nr
Mstep = 10*Mr
Kstep = 20*Mr

# constexpr int Nc = 720;
Nc = [ i for i in range(720-4*Nstep,720+4*Nstep,Nstep)]
# constexpr int Mc = 180;
Mc = [ i for i in range(180-4*Mstep,180+4*Mstep,Mstep)]
# constexpr int Kc = 240;
Kc = [ i for i in range(240-2*Kstep,240+2*Kstep,Kstep)]

# Nc = [ i for i in range(48,65,Nr)]
# Mc = [ i for i in range(32,34,Mr)]
# Kc = [ i for i in range(32,37,Mr)]


BM_NAME = "BM_MatMulAutotune"

for Ncc in Nc:
    for Mcc in Mc:
        for Kcc in Kc:
            size = 2880
            # calc new size which should be divisible by Ncc and Mcc and value should be around 2880
            if size % Ncc != 0 or size % Mcc != 0 or size % Kcc != 0:
                size = math.lcm(Mcc,Ncc,Kcc)
                if size > 7000:
                    print("size is too large")
                    continue
            # set env variable    

            print(f"Run experiment with Ncc: {Ncc}, Mcc: {Mcc}, Kcc: {Kcc}, size: {size}")
            # run cmake ffrom current directory
            os.system(f"pwd && cmake -DN_CACHE_SIZE={Ncc} -DM_CACHE_SIZE={Mcc} -DK_CACHE_SIZE={Kcc} \
                      -DCMAKE_BUILD_TYPE=Release -DENABLE_UNIT_TESTS=OFF \
                      -DCMAKE_TOOLCHAIN_FILE=./build/Release/generators/conan_toolchain.cmake \
                      -B ./build && cmake --build ./build -j4") 
            
            os.environ["MATRIX_DIM"] = str(size)
            #  | grep -E 'Nc:|BM_MatMulAutotune|BM_MatrixMulOpenBLAS'
            os.system(f"./build/Benchmarks --benchmark_filter=\"{BM_NAME}/{size}|BM_MatrixMulOpenBLAS/{size}\" --benchmark_time_unit=ms | grep -E 'Nc:|BM_MatMulAutotune|BM_MatrixMulOpenBLAS' >> output.log 2>&1")
