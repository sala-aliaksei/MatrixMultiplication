#pragma once

#include <utility>

// force inlining

template<long unsigned int N, class F>
__attribute__((always_inline)) static constexpr void static_for(F&& f)
{
    [&]<std::size_t... I>(std::index_sequence<I...>)
    { (..., f.template operator()<I>()); }(std::make_index_sequence<N>{});
}