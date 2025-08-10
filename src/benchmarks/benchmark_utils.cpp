#include "benchmark_utils.hpp"

int GetMatrixDimFromEnv()
{
    constexpr std::size_t NN = 4 * 720; // for haswell

    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : NN;
}