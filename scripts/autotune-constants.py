
import os
import math
import sys
# Given constants
Nr, Mr, Kr = 12, 4, 1

Nstep = 2*Nr  # 48
Mstep = 4*Mr # 16
Kstep = 20


# constexpr int Mc = 20;
Mc = [ i for i in range(4,204,Mstep)]

# constexpr int Nc = 180;
Nc = [ i for i in range(144,720,Nstep)]

# constexpr int Kc = 80;
Kc = [ i for i in range(20,260,Kstep)]


Mb = 128 # 144 # 128
Nb = 720
Kb = 256

Mb = 20 # 144 # 128
Nb = 180
Kb = 80

Mc = [ i for i in range(Mb,Mb+Mstep,Mstep)]
Nc = [ i for i in range(Nb,Nb+Nstep,Nstep)]
Kc = [ i for i in range(Kb,Kb+Kstep,Kstep)]


BM_NAME = "BM_MatMulAutotune"


script_dir = os.path.dirname(os.path.abspath(__file__))
workspace = script_dir + "/../"
print("workspace: ", workspace)

output_dir = workspace + "output/autotune-constants"
os.makedirs(output_dir, exist_ok=True)

for Ncc in Nc:
    for Mcc in Mc:
        for Kcc in Kc:
            size = 2880
            # calc new size which should be divisible by Ncc and Mcc and value should be around 2880
            # if size % Ncc != 0 or size % Mcc != 0 or size % Kcc != 0:
            #     size = math.lcm(Mcc,Ncc,Kcc)
            #     if size > 7000:
            #         print("size is too large")
            #         continue
            # set env variable    
            with open(f"{output_dir}/output.log", 'a') as f:
                f.write(f"{Mcc}, {Ncc}, {Kcc}, ")
            print(f"Run experiment with Ncc: {Ncc}, Mcc: {Mcc}, Kcc: {Kcc}, size: {size}")
            # run cmake ffrom current directory
            ret = os.system(f"cd {workspace} && cmake -DN_CACHE_SIZE={Ncc} -DM_CACHE_SIZE={Mcc} -DK_CACHE_SIZE={Kcc} \
                      -DCMAKE_BUILD_TYPE=Release -DENABLE_UNIT_TESTS=OFF \
                      -DCMAKE_TOOLCHAIN_FILE=./build/Release/generators/conan_toolchain.cmake \
                      -B ./build && cmake --build ./build -j4 --target BM_MatmulAutotune")
            
            if ret != 0:
                print("\n[Error]Failed to build matmul !!!")
                sys.exit(1)

            os.environ["MATRIX_DIM"] = str(size)
            #os.system(f"cd {workspace} && ./build/BM_Matmul --benchmark_filter=\"{BM_NAME}/{size}\" --benchmark_time_unit=ms | grep -oP '(?<=BM_MatMulAutotune\\s+)\\d+' >> {output_dir}/output.log 2>&1")
            os.system(f"cd {workspace} && ./build/BM_MatmulAutotune --benchmark_filter=\"{BM_NAME}/{size}\" --benchmark_time_unit=ms | grep -E 'Nc:|BM_MatMulAutotune' |  " + "awk '{print $2}'" + f" >> {output_dir}/output.log 2>&1")
            os.system(f"echo '' >> {output_dir}/output.log 2>&1")
            #os.system(f"cd {workspace} && ./build/BM_Matmul --benchmark_filter=\"{BM_NAME}/{size}\" --benchmark_time_unit=ms | grep -E 'Nc:|BM_MatMulAutotune' >> {output_dir}/output.log 2>&1")
            #os.system(f"cd {workspace} && ./build/BM_Matmul --benchmark_filter=\"{BM_NAME}/{size}|BM_MatrixMulOpenBLAS/{size}\" --benchmark_time_unit=ms | grep -E 'Nc:|BM_MatMulAutotune|BM_MatrixMulOpenBLAS' >> {output_dir}/output.log 2>&1")

print("Done")
