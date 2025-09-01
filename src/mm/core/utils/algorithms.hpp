#pragma once

template<int I, int N, class F>
constexpr void static_for(F f)
{
    if constexpr (I < N)
    {
        f.template operator()<I>();
        static_for<I + 1, N>(f);
    }
}