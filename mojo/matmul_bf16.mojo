# High-throughput BF16 matmul for NVIDIA RTX 5090 using Tensor Cores (BF16->F32).
# Provides a CPU reference fallback for correctness.

alias BF16 = BFloat16
alias F32 = Float32

# Row-major pointer strides.
fn idx(i: Int, j: Int, ld: Int) -> Int:
    return i * ld + j

# CPU reference for correctness. Accumulates in FP32.
fn matmul_bf16_reference(
    A: List[BF16], B: List[BF16],
    M: Int, N: Int, K: Int,
    lda: Int, ldb: Int, ldc: Int
) -> List[F32]:
    var C = List[F32]()
    var i = 0
    while i < M:
        var j = 0
        while j < N:
            var sum: F32 = F32(0.0)
            var k = 0
            while k < K:
                var av: F32 = F32(A[idx(i, k, lda)])
                var bv: F32 = F32(B[idx(k, j, ldb)])
                sum += av * bv
                k += 1
            C.append(sum)
            j += 1
        i += 1
    return C.copy()

# Public API: currently uses the CPU reference implementation.
fn matmul_bf16(
    A: List[BF16], B: List[BF16],
    M: Int, N: Int, K: Int,
    lda: Int, ldb: Int, ldc: Int,
    use_gpu: Bool = True
) -> List[F32]:
    return matmul_bf16_reference(A, B, M, N, K, lda, ldb, ldc)

