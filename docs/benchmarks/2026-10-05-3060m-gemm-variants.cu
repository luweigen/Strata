// hipBLAS GemmEx on gfx1151 across dtypes, output types, layouts and shapes: which variants the library runs fast
// This copy: 2026-10-05-halo-gemm-variants.cpp (branch AIMAX395-ROCm) ported to cuBLAS for the RTX 3060 Laptop GPU:
// hip*/HIPBLAS_* renamed to cuda*/cublas*/CUBLAS_*, and for COMPUTE_16F alpha/beta are __half (cuBLAS reads
// them in the compute type: a float 1.0 read as half is 0, and cuBLAS then skipped the product, 0.16 ms = 661 TFLOP/s).
// Build: nvcc -O3 -arch=sm_86 docs/benchmarks/2026-10-05-3060m-gemm-variants.cu -lcublas -o build/gemm_variants
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct Case { const char* name; int M, N, K; cublasOperation_t opA, opB; cudaDataType in, out; cublasComputeType_t comp; };
int main() {
    cublasHandle_t h; cublasCreate(&h);
    const size_t big = 64ull << 20;   // elements
    void *A, *B, *C; cudaMalloc(&A, big * 2); cudaMalloc(&B, big * 2); cudaMalloc(&C, big * 4);
    cudaMemset(A, 0x3C, big * 2); cudaMemset(B, 0x38, big * 2); cudaMemset(C, 0, big * 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const Case cases[] = {
        {"engine: N=3072 T=8192 K=2048, T/N, BF16->F32, 32F", 3072, 8192, 2048, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F},
        {"same, FP16->F32", 3072, 8192, 2048, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
        {"same, FP16->FP16 out", 3072, 8192, 2048, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F},
        {"same, FP16->FP16, compute 16F", 3072, 8192, 2048, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F},
        {"same, BF16->BF16 out", 3072, 8192, 2048, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F},
        {"N/N layout, FP16->F32", 3072, 8192, 2048, CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
        {"N/N layout, FP16->FP16", 3072, 8192, 2048, CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F},
        {"square 4096^3, T/N, FP16->F32", 4096, 4096, 4096, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
        {"square 4096^3, N/N, FP16->FP16", 4096, 4096, 4096, CUBLAS_OP_N, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F},
        {"square 4096^3, T/N, BF16->BF16", 4096, 4096, 4096, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F},
        {"expert gate/up: N=1280 T=160 K=2560, T/N, FP16->F32", 1280, 160, 2560, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
        {"expert down: N=2560 T=160 K=640, T/N, FP16->F32", 2560, 160, 640, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
        {"expert gate/up, FP16->FP16", 1280, 160, 2560, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F},
        {"decode-like: N=3072 T=8 K=2048, T/N, FP16->F32", 3072, 8, 2048, CUBLAS_OP_T, CUBLAS_OP_N, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
    };
    const float alpha = 1.f, beta = 0.f;
    const __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);
    for (const Case& c : cases) {
        const int lda = c.opA == CUBLAS_OP_T ? c.K : c.M, ldb = c.opB == CUBLAS_OP_N ? c.K : c.N;
        auto run = [&]() {
            return cublasGemmEx(h, c.opA, c.opB, c.M, c.N, c.K, c.comp == CUBLAS_COMPUTE_16F ? (const void*) &alpha_h : (const void*) &alpha, A, c.in, lda, B, c.in, ldb, c.comp == CUBLAS_COMPUTE_16F ? (const void*) &beta_h : (const void*) &beta, C, c.out, c.M,
                                    c.comp, CUBLAS_GEMM_DEFAULT);
        };
        cublasStatus_t st = run(); cudaDeviceSynchronize();
        if (st != CUBLAS_STATUS_SUCCESS) { std::printf("%-55s status %d\n", c.name, (int) st); continue; }
        const int reps = c.N <= 160 ? 200 : 10;
        for (int i = 0; i < 3; ++i) run();
        cudaEventRecord(e0); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1); ms /= reps;
        std::printf("%-55s %8.3f ms  %6.1f TFLOP/s\n", c.name, ms, 2.0 * c.M * c.N * c.K / (ms * 1e-3) / 1e12);
    }
    return 0;
}
