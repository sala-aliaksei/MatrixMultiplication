cmake_minimum_required(VERSION 3.10)

project("MatrixMultiplication" VERSION 1.0)

option(ENABLE_LLVM_MCA "Run llvm-mca performance analysis" OFF)
option(ENABLE_GPU_BUILD "Enable GPU build" OFF)
option(ENABLE_UNIT_TESTS "Enable Unit Tests" ON)
option(ENABLE_BENCHMARKS "Enable Benchmarks" ON)
option(ENABLE_TPI_BLIS "Use BLIS library" OFF)
# set(Kokkos_ENABLE_CUDA ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 11)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -g -ffast-math -march=native -Wno-interference-size -fdiagnostics-color")
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O3 -march=native ") # -masm=intel
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp=libomp")
endif()

# Add custom OpenBLAS
# set(CUSTOM_OPENBLAS_ROOT "/home/aliaksei/wp/cpp/lib/openblas")
# add_library(CustomOpenBLAS SHARED IMPORTED)
# set_target_properties(CustomOpenBLAS PROPERTIES
#     IMPORTED_LOCATION "${CUSTOM_OPENBLAS_ROOT}/lib/libopenblas.so"
#     INTERFACE_INCLUDE_DIRECTORIES "${CUSTOM_OPENBLAS_ROOT}/include"
#     IMPORTED_NO_SONAME ON
# )
# add_library(OpenBLAS::OpenBLAS ALIAS CustomOpenBLAS)

# Use blis from system path (no conan receipt)
# set(CUSTOM_BLIS_ROOT "/home/aliaksei/wp/cpp/lib/amdblis")
# set(BLIS_LIB "${CUSTOM_BLIS_ROOT}/lib/libblis.so")
# set(BLIS_INCLUDE_DIR "${CUSTOM_BLIS_ROOT}/include")

# # Verify files exist
# if(NOT EXISTS "${BLIS_LIB}")
#     message(FATAL_ERROR "AMD BLIS library not found at: ${BLIS_LIB}")
# endif()

# if(NOT EXISTS "${BLIS_INCLUDE_DIR}/blis/blis.h")
#     message(FATAL_ERROR "AMD BLIS headers not found at: ${BLIS_INCLUDE_DIR}")
# endif()

# add_library(CustomBLIS SHARED IMPORTED)
# set_target_properties(CustomBLIS PROPERTIES
#     IMPORTED_LOCATION "${CUSTOM_BLIS_ROOT}/lib/libblis.so"
#     INTERFACE_INCLUDE_DIRECTORIES "${CUSTOM_BLIS_ROOT}/include"
#     IMPORTED_NO_SONAME ON
# )
# add_library(BLIS::BLIS ALIAS CustomBLIS)


#find_package(tensorflow-lite REQUIRED)
#find_package(tensorflowlite REQUIRED)
#find_package(mdspan  REQUIRED)
#find_package(LinAlg  REQUIRED) # LinAlg
find_package(OpenBLAS REQUIRED)
find_package(Eigen3 REQUIRED)

#find_package(Kokkos REQUIRED)
#find_package(Boost REQUIRED) # for aligment allocator # COMPONENTS system
#find_package(OpenMP REQUIRED)



if(ENABLE_GPU_BUILD)
    find_package(CUDA REQUIRED)
endif()



#######    CORE     ##########
# TODO: Make public and private headers
add_library(Core SHARED
    code/mm/core/kernels.cpp
    code/mm/core/kernels.hpp
    code/mm/core/ikernels.hpp
    code/mm/core/Matrix.cpp
    code/mm/core/Matrix.hpp
    code/mm/core/reorderMatrix.hpp
    code/mm/core/utils/utils.cpp
    code/mm/core/utils/utils.hpp
)
target_include_directories(Core PUBLIC
    ${CMAKE_SOURCE_DIR}/code
)


######      MATMUL     #################
add_library(${PROJECT_NAME} SHARED
    code/mm/matmul/matMul.hpp
    code/mm/matmul/matMul.cpp
    code/mm/matmul/matMulSimd.hpp
    code/mm/matmul/matMulSimd.cpp
    code/mm/matmul/matMulPadding.hpp
    code/mm/matmul/matMulPadding.cpp
    code/mm/matmul/matMulLoops.cpp
    code/mm/matmul/matMulLoops.hpp
    code/mm/matmul/matMulColOpt.hpp
    code/mm/matmul/matMulColOpt.cpp
    code/mm/matmul/matMulRegOpt.hpp
    code/mm/matmul/matMulRegOpt.cpp
    code/mm/matmul/matMulZen5.cpp
    code/mm/matmul/matMulZen5.hpp
    code/mm/matmul/matMulHyper.cpp
    code/mm/matmul/matMulHyper.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC Core)
target_link_libraries(${PROJECT_NAME} Core)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROJECT_VERSION="${PROJECT_VERSION}" MDSPAN_USE_BRACKET_OPERATOR MATRIX_DIMS=3072)


#########     3rd Party implementation      #######

# Third Party Implementations (TPI)
add_library(TPI SHARED
    code/mm/tpi/matMulEigen.cpp
    code/mm/tpi/matMulEigen.hpp
    code/mm/tpi/matMulOpenBlas.hpp
    code/mm/tpi/matMulOpenBlas.cpp
    code/mm/tpi/matMulBlis.hpp
    code/mm/tpi/matMulBlis.cpp
    code/mm/tpi/matMulKokkos.hpp
    code/mm/tpi/matMulKokkos.cpp
)

target_include_directories(TPI PUBLIC
    Core
    #Kokkos::kokkos
    #std::linalg
)

target_link_libraries(TPI
    Core
    OpenBLAS::OpenBLAS
    Eigen3::Eigen
    #tensorflow::tensorflowlite
    #std::linalg
)

target_compile_definitions(TPI PRIVATE KOKKOS_ENABLE_OPENMP)


if(ENABLE_TPI_BLIS)
    target_compile_definitions(TPI PRIVATE ENABLED_TPI_BLIS)
    target_include_directories(TPI PRIVATE BLIS::BLIS)
    target_link_libraries(TPI ${BLIS_LIB})
endif()




######     GEN AI     #################
add_library(GenAI
    SHARED
    code/mm/genai/matMulClaude.cpp
    code/mm/genai/matMulClaude.hpp
    code/mm/genai/matMulGpt.cpp
    code/mm/genai/matMulGpt.hpp
)

target_include_directories(GenAI PUBLIC Core)
target_link_libraries(GenAI Core)


######     MatMulAutotune      #####

add_library(MatMulAutotune SHARED
    code/mm/matmul/matMulAutotune.hpp
    code/mm/matmul/matMulAutotune.cpp
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

add_executable(MulMatrixApp code/MulMatrixApp.cpp)
target_link_libraries(MulMatrixApp  ${PROJECT_NAME})
target_compile_definitions(MulMatrixApp PUBLIC MDSPAN_USE_BRACKET_OPERATOR)

#-fprofile-generate=${CMAKE_CURRENT_SOURCE_DIR}
#-fprofile-use=${CMAKE_CURRENT_SOURCE_DIR}

if(ENABLE_BENCHMARKS)
    find_package(benchmark REQUIRED)

    add_library(benchmark_utils STATIC code/benchmarks/benchmark_utils.cpp)
    target_include_directories(benchmark_utils PUBLIC ${CMAKE_SOURCE_DIR}/code)
    target_link_libraries(benchmark_utils PUBLIC benchmark::benchmark ${PROJECT_NAME} MatMulAutotune GenAI TPI)

    add_executable(BM_Matmul code/benchmarks/matMulBenchmarks.cpp)
    target_link_libraries(BM_Matmul benchmark_utils)

    # add_executable(BM_Matmul_Float code/benchmarks/matMulFloatBenchmarks.cpp)
    # target_link_libraries(BM_Matmul_Float benchmark_utils)

    # add_executable(BM_Matmul_BFP16 code/benchmarks/matMulBf16Benchmarks.cpp)
    # target_link_libraries(BM_Matmul_BFP16 benchmark_utils)

    add_executable(BM_Matmul_Zen5 code/benchmarks/matMulZen5Benchmark.cpp)
    target_link_libraries(BM_Matmul_Zen5 benchmark_utils)

    add_executable(BM_Prefetch code/benchmarks/stride/benchEffectiveColStride.cpp)
    target_link_libraries(BM_Prefetch benchmark::benchmark)

    add_executable(BM_Repacking code/benchmarks/reorder/reorderBenchmark.cpp)
    target_link_libraries(BM_Repacking benchmark::benchmark)
    target_include_directories(BM_Repacking PUBLIC ${CMAKE_SOURCE_DIR}/code ${CMAKE_SOURCE_DIR}/matrixMultiplication)

    add_executable(BM_Kernels code/benchmarks/kernelsBenchmarks.cpp)
    target_link_libraries(BM_Kernels benchmark::benchmark Core)

    add_executable(BM_MatmulAutotune code/benchmarks/matMulAutotuneBenchmarks.cpp)
    target_link_libraries(BM_MatmulAutotune benchmark::benchmark  MatMulAutotune )

    add_executable(BM_Cache code/benchmarks/cache/cache.cpp)
    target_link_libraries(BM_Cache benchmark_utils)

    add_executable(BM_MM_Hyper code/benchmarks/matmulHyperBenchmark.cpp)
    target_link_libraries(BM_MM_Hyper benchmark_utils)
    
endif()


if(ENABLE_UNIT_TESTS)
    find_package(GTest REQUIRED)
    set(SANITIZER_FLAGS -fsanitize=address,undefined) # address,undefined -fno-omit-frame-pointer
    #set(SANITIZER_FLAGS "")

    add_library(UnitTestUtils INTERFACE)
    target_compile_options(UnitTestUtils INTERFACE ${SANITIZER_FLAGS})
    target_link_options(UnitTestUtils INTERFACE ${SANITIZER_FLAGS})
    target_link_libraries(UnitTestUtils INTERFACE GTest::gtest GTest::gtest_main)
    
    add_executable(TestMatmul code/tests/TestMatMul.cpp)
    target_link_libraries(TestMatmul ${PROJECT_NAME} MatMulAutotune GenAI TPI UnitTestUtils)

    add_executable(TestMatmul_Zen5 code/tests/TestMatmulZen5.cpp)
    target_link_libraries(TestMatmul_Zen5 ${PROJECT_NAME} MatMulAutotune TPI UnitTestUtils)

    add_executable(TestMatmul_BFP16 code/tests/TestMatMulBFP16.cpp)
    target_link_libraries(TestMatmul_BFP16 ${PROJECT_NAME} UnitTestUtils)

    add_executable(TestKernels code/tests/TestKernels.cpp)
    target_link_libraries(TestKernels Core UnitTestUtils)

    add_executable(TestRepacking code/tests/TestRepacking.cpp)
    target_link_libraries(TestRepacking ${PROJECT_NAME} UnitTestUtils)

    add_executable(TestZorder code/tests/TestZorder.cpp)
    target_link_libraries(TestZorder ${PROJECT_NAME} UnitTestUtils)

    add_executable(TestMMHyper code/tests/TestMatMulHyper.cpp)
    target_link_libraries(TestMMHyper ${PROJECT_NAME} TPI UnitTestUtils)

    set(ALL_MY_TESTS
        TestMatmul
        TestMatmul_Zen5
        TestMatmul_BFP16
        TestKernels
        TestRepacking
        TestZorder
        TestMMHyper
    )
    foreach(test_target ${ALL_MY_TESTS})
        set_target_properties(${test_target} PROPERTIES
            # This tells the executable to look for .so files in its own directory
            INSTALL_RPATH "$ORIGIN;/home/aliaksei/wp/gcc-16/lib64"
        )
    endforeach()

endif()



# CMAKE_CXX_COMPILER_ID doesn't work
if(ENABLE_LLVM_MCA AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")

    # 1. Locate llvm-mca
    find_program(LLVM_MCA_EXECUTABLE
    NAMES llvm-mca
    HINTS $ENV{LLVM_HOME}/bin
    )
    if(NOT LLVM_MCA_EXECUTABLE)
    message(FATAL_ERROR "llvm-mca not found; install LLVM tools or set LLVM_HOME")
    endif()

    # 2) Create the output directory for .s/.mca
    set(MCA_DIR "${CMAKE_BINARY_DIR}/llvm_mca")
    file(MAKE_DIRECTORY "${MCA_DIR}")

    get_target_property(_SRCS  MatMulAutotune SOURCES)
    get_target_property(_INC_DIRS Core INCLUDE_DIRECTORIES)
    get_target_property(_DEFS      MatMulAutotune COMPILE_DEFINITIONS)
    get_target_property(_OPTS      MatMulAutotune COMPILE_OPTIONS)

    set(MYAPP_SRC "")
    foreach(f IN LISTS _SRCS)
      # Make f absolute, assuming it’s relative to the tree root:
      get_filename_component(abs_f "${CMAKE_SOURCE_DIR}/${f}" ABSOLUTE)
      list(APPEND MYAPP_SRC "${abs_f}")
    endforeach()

    # 2. Build a list of -I flags:
    set(MYAPP_INCLUDE_FLAGS)
    foreach(dir IN LISTS _INC_DIRS)
      list(APPEND MYAPP_INCLUDE_FLAGS -I${dir})
    endforeach()

    set(MYAPP_DEFINE_FLAGS "")
    foreach(d IN LISTS _DEFS)
      list(APPEND MYAPP_DEFINE_FLAGS "-D${d}")
    endforeach()

    # 5. Put it all together
#    set(MYAPP_COMPILER_FLAGS
#        ${CMAKE_CXX_FLAGS}
#        ${MYAPP_DEFINE_FLAGS}
#    )
    string(REPLACE " " ";" MYAPP_COMPILER_FLAGS
        "${CMAKE_CXX_FLAGS} ${MYAPP_DEFINE_FLAGS} -std=c++20"
    )

    # 3) Custom target using generator expressions
    add_custom_target(llvm_mca ALL
      COMMENT "→ Generating assembly and running llvm-mca"

      COMMAND ${CMAKE_CXX_COMPILER}
        # Emit assembly
        -S
        -o "${MCA_DIR}/MatMulAutotune.s"

        ${MYAPP_COMPILER_FLAGS}
        ${MYAPP_INCLUDE_FLAGS}
        # Pull in any compile_options you passed via target_compile_options()
        $<JOIN:$<TARGET_PROPERTY:MulMatrixApp,COMPILE_OPTIONS>," ">

        # Inject all -I paths: private/public + interface includes
        #-I$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,INCLUDE_DIRECTORIES>," -I">

        # Test with it
        #"-I$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,INCLUDE_DIRECTORIES>, -I>"
        #-I$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,INCLUDE_DIRECTORIES>," -I">

        #-I$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,INCLUDE_DIRECTORIES>," -I">
        #-I$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,INTERFACE_INCLUDE_DIRECTORIES>," -I">

        # And all -D definitions
        #-D
        #$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,COMPILE_DEFINITIONS>," -D">

        # Finally, list out the source files (absolute paths under the hood)
        #$<JOIN:$<TARGET_PROPERTY:MulMatrixApp,SOURCES>," ">
        ${CMAKE_SOURCE_DIR}/code/mm/matmul/matMulAutotune.cpp

      # Then run llvm-mca over that asm
      COMMAND ${LLVM_MCA_EXECUTABLE}
        -mcpu=haswell  #TODO: Can we deduce current cpu architectire?
        "${MCA_DIR}/MatMulAutotune.s"
        > "${MCA_DIR}/MatMulAutotune.mca"

      DEPENDS MulMatrixApp
      VERBATIM
    )

    message(STATUS "llvm-mca analysis enabled; results in ${MCA_DIR}/MulMatrixApp.mca")
endif()

