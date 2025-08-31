
#include "mm/matmul/matMulAutotune.hpp"

#include <benchmark/benchmark.h>

// Most of the implementations don't handle matrix tailes, so we need to hardcode the size
constexpr std::size_t NN        = 3072;
constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

int GetMatrixDimFromEnv()
{
    const char* env = std::getenv("MATRIX_DIM");
    return env ? std::atoi(env) : NN;
}

static void BM_MatMulAutotune(benchmark::State& state)
{
    std::size_t DIM = state.range(0);
    std::size_t M   = DIM;
    std::size_t N   = DIM;
    std::size_t K   = DIM;

    auto matrices = initDoubleMatrix(M, N, K);

    for (auto _ : state)
    {
        matMulAutotune(matrices.a, matrices.b, matrices.c);
    }
}

#define REGISTER(NAME, DIM) benchmark::RegisterBenchmark(#NAME, NAME)->Arg(DIM);

int main(int argc, char** argv)
{
    int matrix_dim = GetMatrixDimFromEnv();

    REGISTER(BM_MatMulAutotune, matrix_dim);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
