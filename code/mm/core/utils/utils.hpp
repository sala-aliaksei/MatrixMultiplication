#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <stdexcept>
#include <fmt/core.h>

int GetMatrixDimFromEnv();

#define massert(x, ...)                                        \
    (bool((x)) == true ? void(0)                               \
                       : throw std::runtime_error(fmt::format( \
                           "\nAssertion failed: {}\n {}", #x, fmt::format(__VA_ARGS__))))

class Profiler
{
  public:
    Profiler();
    Profiler(std::string name);
    ~Profiler();

  private:
    std::chrono::high_resolution_clock::time_point _start;

    std::string _name;
};

#define PROFILE(NAME)     \
    Profiler p_##__LINE__ \
    {                     \
        NAME              \
    }

std::ostream& operator<<(std::ostream& os, std::vector<std::vector<int>> array);
std::ostream& operator<<(std::ostream& os, std::vector<std::string> array);
std::ostream& operator<<(std::ostream& os, std::vector<int> array);
std::ostream& operator<<(std::ostream& os, std::vector<double> array);
