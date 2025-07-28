cmake_minimum_required(VERSION 3.5)

project("MatrixMultiplication" VERSION 1.0)

option(ENABLE_GPU_BUILD "Enable GPU build" OFF)
option(ENABLE_UNIT_TESTS "Enable Unit Tests" ON)
option(ENABLE_BENCHMARKS "Enable Benchmarks" ON)

#set("CMAKE_C_COMPILER" "/usr/bin/clang" CACHE "STRING" "" FORCE)
#set("CMAKE_CXX_COMPILER" "/usr/bin/clang++" CACHE "STRING" "" FORCE)

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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -g -ffast-math -march=native -masm=intel -fopenmp") #-O3
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O3   -march=native -masm=intel")
#set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} --no-relax")

#find_package(tensorflow-lite REQUIRED)
#find_package(tensorflowlite REQUIRED)
#find_package(mdspan  REQUIRED)
#find_package(LinAlg  REQUIRED) # LinAlg
find_package(OpenBLAS REQUIRED)
find_package(Boost REQUIRED) # for aligment allocator # COMPONENTS system
find_package(Eigen3 REQUIRED)
#find_package(Kokkos REQUIRED)

# Use blis from system path (no conan receipt)
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
    src/mm/core/kernels.cpp
    src/mm/core/kernels.hpp
    src/mm/core/Matrix.cpp
    src/mm/core/Matrix.hpp
    src/mm/core/reorderMatrix.hpp
    src/mm/core/utils/utils.cpp
    src/mm/core/utils/utils.hpp
)
target_include_directories(Core PUBLIC
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)


######      MATMUL     #################
add_library(${PROJECT_NAME} SHARED
    src/mm/matmul/matMul.hpp
    src/mm/matmul/matMul.cpp
    src/mm/matmul/matMulSimd.hpp
    src/mm/matmul/matMulSimd.cpp
    src/mm/matmul/matMulPadding.hpp
    src/mm/matmul/matMulPadding.cpp
    src/mm/matmul/matMulLoops.cpp
    src/mm/matmul/matMulLoops.hpp
    src/mm/matmul/matMulColOpt.hpp
    src/mm/matmul/matMulColOpt.cpp
    src/mm/matmul/matMulRegOpt.hpp
    src/mm/matmul/matMulRegOpt.cpp
    src/mm/matmul/matMulZen5.cpp
    src/mm/matmul/matMulZen5.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC Core)
target_link_libraries(${PROJECT_NAME} Core)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROJECT_VERSION="${PROJECT_VERSION}" MDSPAN_USE_BRACKET_OPERATOR)


#########     3rd Party implementation      #######

# TPI - 3rd Party Impl
add_library(TPI SHARED
    src/mm/tpi/matMulEigen.cpp
    src/mm/tpi/matMulEigen.hpp
    src/mm/tpi/matMulOpenBlas.hpp
    src/mm/tpi/matMulOpenBlas.cpp
    src/mm/tpi/matMulBlis.hpp
    src/mm/tpi/matMulBlis.cpp
    src/mm/tpi/matMulKokkos.hpp
    src/mm/tpi/matMulKokkos.cpp
)

target_include_directories(TPI PUBLIC
    Core
    ${CMAKE_SOURCE_DIR}
    ${BLIS_INCLUDE_DIR}
    Kokkos::kokkos
    #std::linalg
)

target_link_libraries(TPI
    Core
    OpenBLAS::OpenBLAS
    Eigen3::Eigen
    ${BLIS_LIB}
    #tensorflow::tensorflowlite
    #std::linalg
)

target_compile_definitions(TPI PRIVATE KOKKOS_ENABLE_OPENMP)
# set(Kokkos_ENABLE_CUDA ON)




######     GEN AI     #################
add_library(GenAI
    SHARED
    src/mm/genai/matMulClaude.cpp
    src/mm/genai/matMulClaude.hpp
    src/mm/genai/matMulGpt.cpp
    src/mm/genai/matMulGpt.hpp
)

target_include_directories(GenAI PUBLIC Core)
target_link_libraries(GenAI Core)


######     MatMulAutotune      #####

add_library(MatMulAutotune SHARED
    src/mm/matmul/matMulAutotune.hpp
    src/mm/matmul/matMulAutotune.cpp
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

add_executable(MulMatrixApp src/MulMatrixApp.cpp)
target_link_libraries(MulMatrixApp  ${PROJECT_NAME})
target_include_directories(MulMatrixApp PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_definitions(MulMatrixApp PUBLIC MDSPAN_USE_BRACKET_OPERATOR)


if(ENABLE_BENCHMARKS)
    find_package(benchmark REQUIRED)

    add_executable(BM_Matmul src/benchmarks/matMulBenchmarks.cpp)
    target_link_libraries(BM_Matmul benchmark::benchmark ${PROJECT_NAME} MatMulAutotune GenAI TPI)
    target_include_directories(BM_Matmul PUBLIC ${CMAKE_SOURCE_DIR})

    add_executable(BM_Prefetch src/benchmarks/stride/benchEffectiveColStride.cpp)
    target_link_libraries(BM_Prefetch benchmark::benchmark)
    target_include_directories(BM_Prefetch PUBLIC ${CMAKE_SOURCE_DIR}/src)

    add_executable(BM_Repacking src/benchmarks/reorder/reorderBenchmark.cpp)
    target_link_libraries(BM_Repacking benchmark::benchmark)
    target_include_directories(BM_Repacking PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/matrixMultiplication)

    add_executable(BM_Kernels src/benchmarks/kernelsBenchmarks.cpp)
    target_link_libraries(BM_Kernels benchmark::benchmark Core)
    target_include_directories(BM_Kernels PUBLIC ${CMAKE_SOURCE_DIR})
endif()


if(ENABLE_UNIT_TESTS)
    find_package(GTest REQUIRED)

    add_executable(UT_Matmul src/unitTests/UTMatMul.cpp)
    target_link_libraries(UT_Matmul ${PROJECT_NAME} MatMulAutotune GenAI TPI GTest::gtest GTest::gtest_main )
    target_include_directories(UT_Matmul PUBLIC ${CMAKE_SOURCE_DIR})

    add_executable(UT_Kernels src/unitTests/UTKernels.cpp)
    target_link_libraries(UT_Kernels Core GTest::gtest GTest::gtest_main )
    target_include_directories(UT_Kernels PUBLIC ${CMAKE_SOURCE_DIR})


    add_executable(UT_Repacking src/unitTests/UTRepacking.cpp)
    target_link_libraries(UT_Repacking ${PROJECT_NAME} GTest::gtest GTest::gtest_main )
    target_include_directories(UT_Repacking PUBLIC ${CMAKE_SOURCE_DIR})

    add_executable(UT_Zorder src/unitTests/UTZorder.cpp)
    target_link_libraries(UT_Zorder ${PROJECT_NAME} GTest::gtest GTest::gtest_main )
    target_include_directories(UT_Zorder PUBLIC ${CMAKE_SOURCE_DIR})


endif()

