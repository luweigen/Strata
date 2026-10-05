// hipBLAS GemmEx on gfx1151 across dtypes, output types, layouts and shapes: which variants the library runs fast
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct Case { const char* name; int M, N, K; hipblasOperation_t opA, opB; hipDataType in, out; hipblasComputeType_t comp; };
int main() {
    hipblasHandle_t h; hipblasCreate(&h);
    const size_t big = 64ull << 20;   // elements
    void *A, *B, *C; hipMalloc(&A, big * 2); hipMalloc(&B, big * 2); hipMalloc(&C, big * 4);
    hipMemset(A, 0x3C, big * 2); hipMemset(B, 0x38, big * 2); hipMemset(C, 0, big * 4);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    const Case cases[] = {
        {"engine: N=3072 T=8192 K=2048, T/N, BF16->F32, 32F", 3072, 8192, 2048, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_32F, HIPBLAS_COMPUTE_32F},
        {"same, FP16->F32", 3072, 8192, 2048, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
        {"same, FP16->FP16 out", 3072, 8192, 2048, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F},
        {"same, FP16->FP16, compute 16F", 3072, 8192, 2048, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_16F},
        {"same, BF16->BF16 out", 3072, 8192, 2048, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F},
        {"N/N layout, FP16->F32", 3072, 8192, 2048, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
        {"N/N layout, FP16->FP16", 3072, 8192, 2048, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F},
        {"square 4096^3, T/N, FP16->F32", 4096, 4096, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
        {"square 4096^3, N/N, FP16->FP16", 4096, 4096, 4096, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F},
        {"square 4096^3, T/N, BF16->BF16", 4096, 4096, 4096, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F},
        {"expert gate/up: N=1280 T=160 K=2560, T/N, FP16->F32", 1280, 160, 2560, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
        {"expert down: N=2560 T=160 K=640, T/N, FP16->F32", 2560, 160, 640, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
        {"expert gate/up, FP16->FP16", 1280, 160, 2560, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F},
        {"decode-like: N=3072 T=8 K=2048, T/N, FP16->F32", 3072, 8, 2048, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
    };
    const float alpha = 1.f, beta = 0.f;
    for (const Case& c : cases) {
        const int lda = c.opA == HIPBLAS_OP_T ? c.K : c.M, ldb = c.opB == HIPBLAS_OP_N ? c.K : c.N;
        auto run = [&]() {
            return hipblasGemmEx(h, c.opA, c.opB, c.M, c.N, c.K, &alpha, A, c.in, lda, B, c.in, ldb, &beta, C, c.out, c.M,
                                    c.comp, HIPBLAS_GEMM_DEFAULT);
        };
        hipblasStatus_t st = run(); hipDeviceSynchronize();
        if (st != HIPBLAS_STATUS_SUCCESS) { std::printf("%-55s status %d\n", c.name, (int) st); continue; }
        const int reps = c.N <= 160 ? 200 : 10;
        for (int i = 0; i < 3; ++i) run();
        hipEventRecord(e0); for (int i = 0; i < reps; ++i) run(); hipEventRecord(e1); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= reps;
        std::printf("%-55s %8.3f ms  %6.1f TFLOP/s\n", c.name, ms, 2.0 * c.M * c.N * c.K / (ms * 1e-3) / 1e12);
    }
    return 0;
}
