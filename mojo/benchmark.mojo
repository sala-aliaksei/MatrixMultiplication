from sys import argv
from python import Python

from matmul_bf16 import matmul_bf16

alias BF16 = BFloat16
alias F32 = Float32

fn parse_int(flag: String, default_value: Int) raises -> Int:
    var args = argv()
    var i = 0
    while i + 1 < len(args):
        if args[i] == flag:
            return Int(args[i + 1])
        i += 1
    return default_value

fn parse_bool(flag: String, default_value: Bool) -> Bool:
    var args = argv()
    var i = 0
    while i < len(args):
        if args[i] == flag:
            return True
        i += 1
    return default_value

fn main() raises:
    # Large default sizes to target high TFLOPS.
    var M = parse_int("--m", 8192)
    var N = parse_int("--n", 8192)
    var K = parse_int("--k", 8192)
    var iters = parse_int("--iters", 20)
    var warmup = parse_int("--warmup", 5)
    var use_gpu = not parse_bool("--cpu", False)

    # Allocate raw buffers. In practice, these should be on device memory when use_gpu is True.
    var a_elems = M * K
    var b_elems = K * N

    var A = List[BF16]()
    var B = List[BF16]()

    # Initialize A and B with small deterministic values.
    var i = 0
    while i < a_elems:
        A.append(BF16(i % 7))
        i += 1
    i = 0
    while i < b_elems:
        B.append(BF16(i % 5))
        i += 1

    # Warmup.
    var w = 0
    while w < warmup:
        var _ = matmul_bf16(A, B, M, N, K, K, N, N, use_gpu)
        w += 1

    # Timed iterations (use Python perf_counter for a monotonic clock).
    var pytime = Python.import_module("time")
    var t0 = pytime.perf_counter()
    var it = 0
    while it < iters:
        var _ = matmul_bf16(A, B, M, N, K, K, N, N, use_gpu)
        it += 1
    var t1 = pytime.perf_counter()

    var elapsed = F32(t1 - t0) / F32(iters)
    var ops = F32(2.0) * F32(M) * F32(N) * F32(K)
    var tflops = ops / (elapsed * F32(1.0e12))

    print("M,N,K:", M, N, K)
    print("Avg time (s):", elapsed)
    print("TFLOPS (BF16):", tflops)

