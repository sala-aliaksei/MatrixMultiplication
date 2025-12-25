# from here:
#
# https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Avai
# lable.md



function(set_compiler_flags  ARCH)
    option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" TRUE)

    if(NOT ARCH)
        set(ARCH native)
    endif()

    set(COMMON_FLAGS -O3 -ffast-math -march=${ARCH} -Wno-interference-size -fdiagnostics-color) # -masm=intel

    set(CLANG_WARNINGS
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a
        # parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a
        # non-virtual destructor. This helps catch hard to
        # track down memory errors
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual
        # function
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wnull-dereference # warn if a null dereference is detected
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output
        # (ie printf)
    )
    if(WARNINGS_AS_ERRORS)
        set(CLANG_WARNINGS ${CLANG_WARNINGS} -Werror)
    endif()

    set(GCC_WARNINGS
        ${CLANG_WARNINGS}
        -Wmisleading-indentation # warn if indentation implies blocks where blocks
        # do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were
        # probably wanted
        -Wuseless-cast # warn if you perform a cast to the same type
    )

    # Addd optional flag -g for release with debug symbols build

    if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(PROJECT_FLAGS ${CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PROJECT_FLAGS ${GCC_WARNINGS})
    else()
        message(AUTHOR_WARNING "No compiler flags set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()

#-fprofile-generate=${CMAKE_CURRENT_SOURCE_DIR}
#-fprofile-use=${CMAKE_CURRENT_SOURCE_DIR}
    #target_compile_options(${project_name} INTERFACE ${COMMON_FLAGS} ${PROJECT_FLAGS})
    set(CMAKE_CXX_FLAGS  ${COMMON_FLAGS} ${PROJECT_FLAGS})

endfunction()


