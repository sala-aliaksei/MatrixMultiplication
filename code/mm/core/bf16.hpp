#pragma once

#include <compare>
#include <complex>
// #include <format> // GCC 16 format might also use it, but clang 20 might not support format fully yet? Include if needed.

#if defined(__clang__) && !defined(__STDCPP_BFLOAT16_T__) && !defined(__BFLT16_DIG__)
namespace __gnu_cxx {
    using __bfloat16_t = __bf16;
}
#define __STDCPP_BFLOAT16_T__ 1
#endif

#include <stdfloat>