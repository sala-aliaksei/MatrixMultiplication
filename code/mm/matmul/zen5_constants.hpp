#pragma once

#include <stdfloat>

namespace mm::constants
{

constexpr int PAGE_SIZE = 4096;

template<typename T>
struct MatMulZen5DebugConfig;

template<>
struct MatMulZen5DebugConfig<double>
{
    static constexpr int Nc = 16;
    static constexpr int Mc = 16;
    static constexpr int Kc = 16;

    static constexpr int Nr = 8;
    static constexpr int Mr = 8;
    static constexpr int Kr = 1;
};

template<typename T>
struct MatMulZen5Config;

template<>
struct MatMulZen5Config<double>
{
    static constexpr int Nc = 96;
    static constexpr int Mc = 96;
    static constexpr int Kc = 96 * 2;

    static constexpr int Nr = 24;
    static constexpr int Mr = 8;
    static constexpr int Kr = 1;
};

template<>
struct MatMulZen5Config<float>
{
    static constexpr int Nc = 96 * 2;
    static constexpr int Mc = 96;
    static constexpr int Kc = 96 * 2 * 2;
};

template<>
struct MatMulZen5Config<std::bfloat16_t>
{
    static constexpr int Nc = 96 * 2;
    static constexpr int Mc = 96;
    static constexpr int Kc = 96 * 2 * 2;
};
} // namespace mm::constants