#pragma once

constexpr int map_thread_id_to_core_id(int thread_id)
{
    if ((thread_id & 1) == 0)
    {
        return thread_id >> 1;
    }
    else
    {
        return 16 + (thread_id >> 1);
    }
}