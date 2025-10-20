#pragma once

#include <benchmark/benchmark.h>
#include <mm/core/Matrix.hpp>

template<typename Float_t,
         void (*matMul)(const Matrix<Float_t>&, const Matrix<Float_t>&, Matrix<Float_t>&)>
static void BM_MatMul(benchmark::State& state)
{
    std::size_t N = state.range(0);

    auto a = generateRandomMatrix<Float_t>(N, N);
    auto b = generateRandomMatrix<Float_t>(N, N);
    auto c = Matrix<Float_t>(N, N);

    for (auto _ : state)
    {
        matMul(a, b, c);
    }
    double flops_per_iter = 2.0 * N * N * N;
    state.counters["FLOPS"] =
      benchmark::Counter(flops_per_iter * state.iterations(), benchmark::Counter::kIsRate);
}

#define REGISTER(NAME, DIM) benchmark::RegisterBenchmark(#NAME, (NAME))->Arg(DIM);

#define REGISTER_DOUBLE(NAME, DIM) \
    benchmark::RegisterBenchmark(#NAME, (BM_MatMul<double, &NAME>))->Arg(DIM);

#define REGISTER_DOUBLE_RANGE(NAME, DIM) \
    benchmark::RegisterBenchmark(#NAME, (BM_MatMul<double, &NAME>))->DenseRange(1024, 16384, 1024);

#define REGISTER_FLOAT(NAME, DIM) \
    benchmark::RegisterBenchmark(#NAME, (BM_MatMul<float, &NAME>))->Arg(DIM);

#define REGISTER_BF16(NAME, DIM) \
    benchmark::RegisterBenchmark(#NAME, (BM_MatMul<std::bfloat16_t, &NAME>))->Arg(DIM);

#define REGISTER_FLOAT_RANGE(NAME, DIM) \
    benchmark::RegisterBenchmark(#NAME, (BM_MatMul<float, &NAME>))->DenseRange(1024, 16384, 1024);