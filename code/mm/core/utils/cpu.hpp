#pragma once

constexpr int map_thread_id_to_core_id(int n)
{
    // // Check if the input is within the specified range [0, 31].
    // if (n < 0 || n > 31) {
    //     // Return an error code or handle as appropriate for out-of-range input.
    //     return -1;
    // }

    if ((n & 1) == 0)
    {                  // If n is even (least significant bit is 0)
        return n >> 1; // Equivalent to n / 2
    }
    else
    {                         // If n is odd
        return 16 + (n >> 1); // Equivalent to 16 + (n - 1) / 2
    }
}