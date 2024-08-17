#include "utils.hpp"
#include <atomic>
#include <iostream>

Profiler::Profiler()
{
    _start = std::chrono::high_resolution_clock::now();
    _name  = "deafult";
}

Profiler::Profiler(std::string name)
  : _start(std::chrono::high_resolution_clock::now())
  , _name(std::move(name))
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

Profiler::~Profiler()
{
    std::cout << "[Profiling] " << _name << ". Took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::high_resolution_clock::now() - _start)
                   .count()
              << " ms" << std::endl;
}

std::ostream& operator<<(std::ostream& os, std::vector<std::string> array)
{
    for (const auto& v : array)
    {
        os << v << ", ";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, std::vector<int> array)
{
    for (const int v : array)
    {
        os << v << ", ";
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, std::vector<std::vector<int>> array)
{
    for (const auto& varr : array)
    {
        os << "[ " << varr << ']' << std::endl;
    }
    return os;
}
