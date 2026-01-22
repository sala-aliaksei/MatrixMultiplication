# Mojo BF16 Matmul (RTX 5090)

This directory contains a Mojo implementation of BF16 matrix multiplication optimized for NVIDIA RTX 5090 Tensor Cores, plus a CLI benchmark and a simple unit test for correctness.

## Files
- `matmul_bf16.mojo`: GPU kernel using BF16 MMA fragments with FP32 accumulation, plus CPU reference fallback.
- `benchmark.mojo`: CLI benchmark that reports TFLOPS.
- `test_matmul.mojo`: Unit test comparing optimized path with CPU reference on small sizes.

## Run benchmark
```bash
cd /home/aliaksei/wp/cpp/MatrixMultiplication/mojo
mojo run benchmark.mojo --m 8192 --n 8192 --k 8192 --iters 20 --warmup 5
```

Use `--cpu` to force the CPU reference path:
```bash
mojo run benchmark.mojo --cpu
```

## Run unit test
```bash
cd /home/aliaksei/wp/cpp/MatrixMultiplication/mojo
mojo run test_matmul.mojo
```

## Performance notes
- Targeting 150+ TFLOPS requires large M/N/K (8k+), BF16 inputs, and Tensor Core use.
- Sustained TFLOPS will depend on matrix sizes, alignment, and GPU clocks.

