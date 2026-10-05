// the engine's dense-projection GEMM call (src/prefill/gemm.cu:388) through Strata's own HIP shim, checked against the CPU
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <cstdint>
struct __nv_bfloat16 { uint16_t b; };
static inline __nv_bfloat16 __float2bfloat16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); __nv_bfloat16 r; r.b = (uint16_t) (u >> 16); return r; }
static inline float __bfloat162float(__nv_bfloat16 h) { uint32_t u = (uint32_t) h.b << 16; float f; std::memcpy(&f, &u, 4); return f; }

#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const int N = 3072, T = 512, K = 2048;
    std::vector<__nv_bfloat16> W((size_t) N * K), X((size_t) T * K);
    std::vector<__half> Wh((size_t) N * K), Xh((size_t) T * K);
    std::vector<double> Wd(W.size()), Xd(X.size());
    srand(7);
    for (size_t i = 0; i < W.size(); ++i) { float v = (rand() / (float) RAND_MAX - 0.5f); W[i] = __float2bfloat16(v); Wh[i] = __float2half(v); Wd[i] = __bfloat162float(W[i]); }
    for (size_t i = 0; i < X.size(); ++i) { float v = (rand() / (float) RAND_MAX - 0.5f); X[i] = __float2bfloat16(v); Xh[i] = __float2half(v); Xd[i] = __bfloat162float(X[i]); }
    void *dW, *dX; float* dY;
    cudaMalloc(&dW, W.size() * 2); cudaMalloc(&dX, X.size() * 2); cudaMalloc(&dY, (size_t) N * T * 4);
    cublasHandle_t h; cublasCreate(&h);
    const float alpha = 1.f, beta = 0.f;
    for (int pass = 0; pass < 2; ++pass) {
        const bool bf = pass == 0;
        cudaMemcpy(dW, bf ? (void*) W.data() : (void*) Wh.data(), W.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dX, bf ? (void*) X.data() : (void*) Xh.data(), X.size() * 2, cudaMemcpyHostToDevice);
        cudaGetLastError();
        cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, dW, bf ? CUDA_R_16BF : CUDA_R_16F, K,
                                         dX, bf ? CUDA_R_16BF : CUDA_R_16F, K, &beta, dY, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
        cudaError_t sync = cudaDeviceSynchronize();
        cudaError_t sticky = cudaGetLastError();
        std::vector<float> Y((size_t) N * T);
        cudaMemcpy(Y.data(), dY, Y.size() * 4, cudaMemcpyDeviceToHost);
        double maxabs = 0, maxdiff = 0;
        for (int t = 0; t < T; t += 7) for (int n = 0; n < N; n += 5) {
            double ref = 0;
            for (int k = 0; k < K; ++k) ref += (bf ? Wd[(size_t) n * K + k] : (double) __half2float(Wh[(size_t) n * K + k])) *
                                             (bf ? Xd[(size_t) t * K + k] : (double) __half2float(Xh[(size_t) t * K + k]));
            maxabs = std::max(maxabs, std::fabs(ref));
            maxdiff = std::max(maxdiff, std::fabs(ref - Y[(size_t) t * N + n]));
        }
        // timing: 20 calls
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < 20; ++i)
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &alpha, dW, bf ? CUDA_R_16BF : CUDA_R_16F, K, dX,
                         bf ? CUDA_R_16BF : CUDA_R_16F, K, &beta, dY, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(b); cudaEventSynchronize(b); float ms = 0; cudaEventElapsedTime(&ms, a, b);
        std::printf("%s GEMM N=%d T=%d K=%d: status %d, sync %s, stale error after it: %s, max|ref| %.2f, max|diff| %.5f (%.2e rel), %.2f ms/call = %.1f TFLOP/s\n",
                    bf ? "BF16" : "FP16", N, T, K, (int) st, cudaGetErrorString(sync), cudaGetErrorString(sticky), maxabs, maxdiff,
                    maxdiff / maxabs, ms / 20, 2.0 * N * T * K / (ms / 20 * 1e-3) / 1e12);
    }
    return 0;
}
