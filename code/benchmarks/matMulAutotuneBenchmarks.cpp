
#include "mm/matmul/matMulAutotune.hpp"
#include "mm/core/utils/utils.hpp"
#include <benchmark/benchmark.h>

constexpr std::size_t ITER_NUM  = 1;
benchmark::TimeUnit   TIME_UNIT = benchmark::kMillisecond;

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
