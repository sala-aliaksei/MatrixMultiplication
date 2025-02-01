cmake_minimum_required(VERSION 3.5)

project("MatrixMultiplication" VERSION 1.0)

option(ENABLE_GPU_BUILD "Enable GPU build" ON)
option(ENABLE_UNIT_TESTS "Enable Unit Tests" ON)
option(ENABLE_BENCHMARKS "Enable Benchmarks" ON)


#set("CMAKE_C_COMPILER" "/usr/bin/clang" CACHE "STRING" "" FORCE)
#set("CMAKE_CXX_COMPILER" "/usr/bin/clang++" CACHE "STRING" "" FORCE)

#set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 17)
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

message(INFO "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")
find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(Boost REQUIRED COMPONENTS system ) # for aligment allocator
find_package(Eigen3)

#if(ENABLE_GPU_BUILD)
#    find_package(CUDA REQUIRED)
#endif()

#find_package(tensorflow-lite REQUIRED)
#find_package(tensorflowlite REQUIRED)
#find_package(mdspan  REQUIRED)
#find_package(LinAlg  REQUIRED) # LinAlg

#add_subdirectory(stdBLAS)


#add_library(matrix_c
#    STATIC
#    matrixMultiplication/matrix/cmatrix.c
#    matrixMultiplication/matrix/cmatrix.h
#)

add_library(${PROJECT_NAME}
    #STATIC # speedup mine, slow down openBlas?
    SHARED
    matrixMultiplication/matrix/matMulBlis.hpp
    matrixMultiplication/matrix/matMulBlis.cpp
    matrixMultiplication/matrix/matMulLoops.cpp
    matrixMultiplication/matrix/matMulLoops.hpp
    matrixMultiplication/matrix/matMulColOpt.hpp
    matrixMultiplication/matrix/matMulColOpt.cpp
    matrixMultiplication/matrix/matMulRegOpt.hpp
    matrixMultiplication/matrix/matMulRegOpt.cpp
    matrixMultiplication/matrix/claudeMatMul.cpp
    matrixMultiplication/matrix/claudeMatMul.hpp
#    matrixMultiplication/matrix/disasm.cpp
#    matrixMultiplication/matrix/disasm.hpp
    matrixMultiplication/matrix/MatrixMulEigen.cpp
    matrixMultiplication/matrix/MatrixMulEigen.hpp
    matrixMultiplication/matrix/MatrixMulOpenBlas.hpp
    matrixMultiplication/matrix/MatrixMulOpenBlas.cpp
    matrixMultiplication/matrix/MatrixMulGpt.cpp
    matrixMultiplication/matrix/MatrixMulGpt.hpp
    matrixMultiplication/matrix/kernels.cpp
    matrixMultiplication/matrix/kernels.hpp
    matrixMultiplication/matrix/MatrixMul.hpp
    matrixMultiplication/matrix/MatrixMul.cpp
    matrixMultiplication/matrix/Matrix.cpp
    matrixMultiplication/matrix/Matrix.hpp
    matrixMultiplication/utils/utils.cpp
    matrixMultiplication/utils/utils.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}
    #std::linalg
    "/home/alex/workspace/cpp-projects/blis"
)

target_link_libraries(${PROJECT_NAME} # PRIVATE
    #-lblis
    OpenBLAS::OpenBLAS
    #-lpthread
    #-lstdc++fs

    Eigen3::Eigen
    #tensorflow::tensorflowlite
    #std::linalg
    #Boost::thread Boost::system
    #matrix_c
)

target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROJECT_VERSION="${PROJECT_VERSION}")

######     APPLICATIONS      #####

add_executable(MulMatrixApp matrixMultiplication/MulMatrixApp.cpp)
target_link_libraries(MulMatrixApp  ${PROJECT_NAME})
target_include_directories(MulMatrixApp PUBLIC ${CMAKE_SOURCE_DIR})

if(ENABLE_BENCHMARKS)
    add_executable(Benchmarks matrixMultiplication/Benchmarks.cpp)
    target_link_libraries(Benchmarks benchmark::benchmark ${PROJECT_NAME})
    target_include_directories(Benchmarks PUBLIC ${CMAKE_SOURCE_DIR})

    add_executable(ColumnStrideBenchmarks src/stride/benchEffectiveColStride.cpp)
    target_link_libraries(ColumnStrideBenchmarks benchmark::benchmark)
    target_include_directories(ColumnStrideBenchmarks PUBLIC ${CMAKE_SOURCE_DIR}/src)

    add_executable(ReorderBenchmarks src/reorder/reorderBenchmark.cpp)
    target_link_libraries(ReorderBenchmarks benchmark::benchmark)
    target_include_directories(ReorderBenchmarks PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/matrixMultiplication)
endif()

if(ENABLE_UNIT_TESTS)
    add_executable(UnitTests matrixMultiplication/UnitTests.cpp)
    target_link_libraries(UnitTests ${PROJECT_NAME} GTest::gtest GTest::gtest_main )
    target_include_directories(UnitTests PUBLIC ${CMAKE_SOURCE_DIR})
endif()

