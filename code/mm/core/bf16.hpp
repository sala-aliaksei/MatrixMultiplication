#pragma once

#include <compare>
#include <complex>
// #include <format> // GCC 16 format might also use it, but clang 20 might not support format fully
// yet? Include if needed.

#if defined(__clang__) && !defined(__STDCPP_BFLOAT16_T__) && !defined(__BFLT16_DIG__)

namespace mm
{
using bfloat16_t = __bf16;
}
// #define __STDCPP_BFLOAT16_T__ 1
#else
#include <stdfloat>
#endif

// #if __has_include(<stdfloat>)
// #include <stdfloat>
// #endif