#pragma once

#include <chrono>
#include <string>
#include <vector>

class Profiler
{
public:
    Profiler();
    Profiler(std::string name);
    ~Profiler();

private:
    std::chrono::steady_clock::time_point _start;
    std::string                           _name;
};

#define PROFILE(NAME) Profiler p_##__LINE__{NAME}

std::ostream& operator<<(std::ostream& os, std::vector<std::vector<int>> array);
std::ostream& operator<<(std::ostream& os, std::vector<std::string> array);
std::ostream& operator<<(std::ostream& os, std::vector<int> array);
