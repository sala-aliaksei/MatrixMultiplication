from matmul_bf16 import matmul_bf16, matmul_bf16_reference

alias BF16 = BFloat16
alias F32 = Float32

fn assert_close(a: F32, b: F32, tol: F32, msg: String) raises:
    var diff = abs(a - b)
    if diff > tol:
        raise Error(msg)

fn test_small() raises:
    var M = 32
    var N = 32
    var K = 32

    var a_elems = M * K
    var b_elems = K * N

    var A = List[BF16]()
    var B = List[BF16]()

    var i = 0
    while i < a_elems:
        A.append(BF16(i % 11))
        i += 1
    i = 0
    while i < b_elems:
        B.append(BF16(i % 7))
        i += 1

    var C_ref = matmul_bf16_reference(A, B, M, N, K, K, N, N)
    var C_opt = matmul_bf16(A, B, M, N, K, K, N, N, use_gpu=False)

    i = 0
    while i < len(C_ref):
        assert_close(C_ref[i], C_opt[i], F32(1.0e-2), "Mismatch at index " + String(i))
        i += 1

fn main() raises:
    test_small()
    print("test_matmul: OK")

