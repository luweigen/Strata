// is the fast FP16-output GEMM a real computation?  FP16 in, FP16 out (compute 32F, and compute 16F) against the CPU
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const int N = 3072, T = 8192, K = 2048;
    std::vector<__half> W((size_t) N * K), X((size_t) T * K);
    std::vector<double> Wd(W.size()), Xd(X.size());
    srand(7);
    for (size_t i = 0; i < W.size(); ++i) { float v = (rand() / (float) RAND_MAX - 0.5f) * 0.1f; W[i] = __float2half(v); Wd[i] = __half2float(W[i]); }
    for (size_t i = 0; i < X.size(); ++i) { float v = (rand() / (float) RAND_MAX - 0.5f); X[i] = __float2half(v); Xd[i] = __half2float(X[i]); }
    void *dW, *dX, *dY; hipMalloc(&dW, W.size() * 2); hipMalloc(&dX, X.size() * 2); hipMalloc(&dY, (size_t) N * T * 2);
    hipMemcpy(dW, W.data(), W.size() * 2, hipMemcpyHostToDevice); hipMemcpy(dX, X.data(), X.size() * 2, hipMemcpyHostToDevice);
    hipblasHandle_t h; hipblasCreate(&h);
    const float alpha = 1.f, beta = 0.f; const __half alpha_h = __float2half(1.f), beta_h = __float2half(0.f);
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    for (int pass = 0; pass < 2; ++pass) {
        const bool c16 = pass == 1;
        hipMemset(dY, 0, (size_t) N * T * 2);
        auto run = [&]() {
            return c16 ? hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &alpha_h, dW, HIP_R_16F, K, dX, HIP_R_16F, K, &beta_h, dY, HIP_R_16F, N, HIPBLAS_COMPUTE_16F, HIPBLAS_GEMM_DEFAULT)
                       : hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &alpha, dW, HIP_R_16F, K, dX, HIP_R_16F, K, &beta, dY, HIP_R_16F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        };
        hipblasStatus_t st = run(); hipDeviceSynchronize();
        hipEventRecord(a); for (int i = 0; i < 10; ++i) run(); hipEventRecord(b); hipEventSynchronize(b);
        float ms = 0; hipEventElapsedTime(&ms, a, b); ms /= 10;
        std::vector<__half> Y((size_t) N * T); hipMemcpy(Y.data(), dY, Y.size() * 2, hipMemcpyDeviceToHost);
        double maxabs = 0, maxdiff = 0; size_t nonzero = 0;
        for (int t = 0; t < T; t += 97) for (int n = 0; n < N; n += 5) {
            double ref = 0; for (int k = 0; k < K; ++k) ref += Wd[(size_t) n * K + k] * Xd[(size_t) t * K + k];
            const double y = __half2float(Y[(size_t) t * N + n]); if (y != 0) ++nonzero;
            maxabs = std::fmax(maxabs, std::fabs(ref)); maxdiff = std::fmax(maxdiff, std::fabs(ref - y));
        }
        std::printf("FP16->FP16 compute %s: status %d, %.3f ms = %.1f TFLOP/s, max|ref| %.3f, max|diff| %.5f (%.1e rel), nonzero samples %zu\n",
                    c16 ? "16F" : "32F", (int) st, ms, 2.0 * N * T * K / (ms * 1e-3) / 1e12, maxabs, maxdiff, maxdiff / maxabs, nonzero);
    }
    return 0;
}
