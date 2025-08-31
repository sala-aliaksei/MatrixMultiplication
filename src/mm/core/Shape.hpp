#pragma once

#include <array>

template<int rank>
struct Shape
{
    std::array<int, rank> dims;

    [[nodiscard]] constexpr int operator[](int i) const
    {
        return dims[i];
    }

    [[nodiscard]] constexpr int& operator[](int i)
    {
        return dims[i];
    }
};

using Shape2D = Shape<2>;
