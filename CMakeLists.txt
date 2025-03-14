cmake_minimum_required(VERSION 3.5)

project("MatrixMultiplication" VERSION 1.0)

option(ENABLE_GPU_BUILD "Enable GPU build" OFF)
option(ENABLE_UNIT_TESTS "Enable Unit Tests" ON)
option(ENABLE_BENCHMARKS "Enable Benchmarks" ON)


set("CMAKE_C_COMPILER" "/usr/bin/clang" CACHE "STRING" "" FORCE)
set("CMAKE_CXX_COMPILER" "/usr/bin/clang++" CACHE "STRING" "" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD 11)

#set(LINALG_ENABLE_KOKKOS ON)
# LINALG_ENABLE_BLAS=ON

# Other compiler flags
#-fprofile-generate=${CMAKE_CURRENT_SOURCE_DIR}
#-fprofile-use=${CMAKE_CURRENT_SOURCE_DIR}
#-fsanitize=address
#-fsanitize=undefined
# -g
# -fsave-optimization-record
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -g -ffast-math -march=haswell -masm=intel -fopenmp") #-O3
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O3   -march=haswell -masm=intel")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} --no-relax")


#find_package(tensorflow-lite REQUIRED)
#find_package(tensorflowlite REQUIRED)
#find_package(mdspan  REQUIRED)
#find_package(LinAlg  REQUIRED) # LinAlg
find_package(OpenBLAS REQUIRED)
find_package(Boost REQUIRED) # for aligment allocator # COMPONENTS system
find_package(Eigen3)

find_library(BLIS_LIB NAMES blis PATHS /usr/local/lib)
find_path(BLIS_INCLUDE_DIR NAMES blis.h PATHS /usr/local/include)
if (NOT BLIS_LIB OR NOT BLIS_INCLUDE_DIR)
    message(FATAL_ERROR "BLIS not found!")
else()
    message(STATUS "Found BLIS: ${BLIS_LIB}")
endif()


if(ENABLE_GPU_BUILD)
    find_package(CUDA REQUIRED)
endif()



#######    CORE     ##########
add_library(Core SHARED
    matrixMultiplication/matrix/kernels.cpp
    matrixMultiplication/matrix/kernels.hpp
    matrixMultiplication/matrix/Matrix.cpp
    matrixMultiplication/matrix/Matrix.hpp
    matrixMultiplication/utils/utils.cpp
    matrixMultiplication/utils/utils.hpp
    matrixMultiplication/matrix/reorderMatrix.hpp
)
target_include_directories(Core PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
)


######      MATMUL     #################
add_library(${PROJECT_NAME}
    SHARED
    matrixMultiplication/matrix/matMul.hpp
    matrixMultiplication/matrix/matMul.cpp
    matrixMultiplication/matrix/matMulSimd.hpp
    matrixMultiplication/matrix/matMulSimd.cpp
    matrixMultiplication/matrix/matMulPadding.hpp
    matrixMultiplication/matrix/matMulPadding.cpp
    matrixMultiplication/matrix/matMulLoops.cpp
    matrixMultiplication/matrix/matMulLoops.hpp
    matrixMultiplication/matrix/matMulColOpt.hpp
    matrixMultiplication/matrix/matMulColOpt.cpp
    matrixMultiplication/matrix/matMulRegOpt.hpp
    matrixMultiplication/matrix/matMulRegOpt.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC Core)
target_link_libraries(${PROJECT_NAME} Core)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROJECT_VERSION="${PROJECT_VERSION}")


#########     3rd Party implementation      #######

# TPI - 3rd Party Impl
add_library(TPI
    SHARED
    matrixMultiplication/matrix/matMulEigen.cpp
    matrixMultiplication/matrix/matMulEigen.hpp
    matrixMultiplication/matrix/matMulOpenBlas.hpp
    matrixMultiplication/matrix/matMulOpenBlas.cpp
    matrixMultiplication/matrix/matMulBlis.hpp
    matrixMultiplication/matrix/matMulBlis.cpp
)

target_include_directories(TPI PUBLIC
    Core
    ${CMAKE_SOURCE_DIR}
    #std::linalg
    #-lblis # installed to system
    ${BLIS_INCLUDE_DIR}
)

target_link_libraries(TPI
    Core
    #-lblis
    OpenBLAS::OpenBLAS
    #-lpthread
    #-lstdc++fs

    Eigen3::Eigen
    #tensorflow::tensorflowlite
    #std::linalg
    ${BLIS_LIB}
)



######     GEN AI     #################
add_library(GenAI
    SHARED
    matrixMultiplication/matrix/matMulClaude.cpp
    matrixMultiplication/matrix/matMulClaude.hpp
    matrixMultiplication/matrix/matMulGpt.cpp
    matrixMultiplication/matrix/matMulGpt.hpp
)

target_include_directories(GenAI PUBLIC Core)
target_link_libraries(GenAI Core)


######     MatMulAutotune      #####
# Move to subdir?

add_library(MatMulAutotune
    #STATIC
    SHARED
    matrixMultiplication/matrix/matMulAutotune.hpp
    matrixMultiplication/matrix/matMulAutotune.cpp
)

target_link_libraries(MatMulAutotune Core)

if(DEFINED M_CACHE_SIZE)
    target_compile_definitions(MatMulAutotune PUBLIC M_CACHE_SIZE=${M_CACHE_SIZE})
endif()
if(DEFINED N_CACHE_SIZE)
    target_compile_definitions(MatMulAutotune PUBLIC N_CACHE_SIZE=${N_CACHE_SIZE})
endif()
if(DEFINED K_CACHE_SIZE)
    target_compile_definitions(MatMulAutotune PUBLIC K_CACHE_SIZE=${K_CACHE_SIZE})
endif()

######     APPLICATIONS      #####

add_executable(MulMatrixApp matrixMultiplication/MulMatrixApp.cpp)
target_link_libraries(MulMatrixApp  ${PROJECT_NAME})
target_include_directories(MulMatrixApp PUBLIC ${CMAKE_SOURCE_DIR})

if(ENABLE_BENCHMARKS)
    find_package(benchmark REQUIRED)

    add_executable(BM_Matmul matrixMultiplication/Benchmarks.cpp)
    target_link_libraries(BM_Matmul benchmark::benchmark ${PROJECT_NAME} MatMulAutotune GenAI TPI)
    target_include_directories(BM_Matmul PUBLIC ${CMAKE_SOURCE_DIR})

    add_executable(BM_Prefetch src/stride/benchEffectiveColStride.cpp)
    target_link_libraries(BM_Prefetch benchmark::benchmark)
    target_include_directories(BM_Prefetch PUBLIC ${CMAKE_SOURCE_DIR}/src)

    add_executable(BM_Repacking src/reorder/reorderBenchmark.cpp)
    target_link_libraries(BM_Repacking benchmark::benchmark)
    target_include_directories(BM_Repacking PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/matrixMultiplication)
endif()

if(ENABLE_UNIT_TESTS)
    find_package(GTest REQUIRED)

    add_executable(UT_Matmul matrixMultiplication/UTMatMul.cpp)
    target_link_libraries(UT_Matmul ${PROJECT_NAME} ${PROJECT_NAME} MatMulAutotune GenAI TPI GTest::gtest GTest::gtest_main )
    target_include_directories(UT_Matmul PUBLIC ${CMAKE_SOURCE_DIR})


    add_executable(UT_Repacking matrixMultiplication/UTRepacking.cpp)
    target_link_libraries(UT_Repacking ${PROJECT_NAME} GTest::gtest GTest::gtest_main )
    target_include_directories(UT_Repacking PUBLIC ${CMAKE_SOURCE_DIR})
endif()

