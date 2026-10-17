// The verify window's batched router for a non-512 geometry against the per-token path, bit for bit.
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/router_top10.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace strata::kernels;
int main() {
    const int N = 2560, K = 10;
    int bad = 0, cases = 0;
    for (int NE : {256, 512, 128}) for (int trial = 0; trial < 20; ++trial) for (int n = 1; n <= 8; ++n) {
        std::mt19937 rng(trial * 131 + n + NE);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<uint16_t> w((size_t) NE * N); std::vector<float> x((size_t) n * N);
        for (auto& v : w) { float f = nd(rng) * 0.02f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t) (u >> 16); }
        for (auto& v : x) v = nd(rng);
        if (trial % 4 == 3)   // near ties: duplicate rows of the weight
            for (int e = 1; e < NE; e += 3) std::memcpy(&w[(size_t) e * N], &w[(size_t) (e - 1) * N], N * 2);
        uint16_t* dw; float *dx, *dl1, *dl2, *dw1, *dw2; int *di1, *di2;
        cudaMalloc(&dw, w.size() * 2); cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dl1, (size_t) n * NE * 4); cudaMalloc(&dl2, (size_t) n * NE * 4);
        cudaMalloc(&di1, n * K * 4); cudaMalloc(&di2, n * K * 4); cudaMalloc(&dw1, n * K * 4); cudaMalloc(&dw2, n * K * 4);
        cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaStream_t s; cudaStreamCreate(&s);
        for (int t = 0; t < n; ++t) {   // the per-token path (moe_route under --native)
            bf16_gemv_fp32_mmvf(dx + t * N, dw, dl1 + t * NE, N, NE, s);
            router_top10(dl1 + t * NE, 1, NE, K, di1 + t * K, dw1 + t * K, s);
        }
        bf16_gemv_fp32_mmvf_multi(dx, N, dw, dl2, NE, N, NE, n, s);   // the new window path
        router_top10(dl2, n, NE, K, di2, dw2, s);
        cudaStreamSynchronize(s);
        std::vector<float> l1(n * NE), l2(n * NE), w1(n * K), w2(n * K); std::vector<int> i1(n * K), i2(n * K);
        cudaMemcpy(l1.data(), dl1, l1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(l2.data(), dl2, l2.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(i1.data(), di1, i1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(i2.data(), di2, i2.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(w1.data(), dw1, w1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(w2.data(), dw2, w2.size() * 4, cudaMemcpyDeviceToHost);
        ++cases;
        if (std::memcmp(l1.data(), l2.data(), l1.size() * 4) || std::memcmp(i1.data(), i2.data(), i1.size() * 4) ||
            std::memcmp(w1.data(), w2.data(), w1.size() * 4)) { ++bad; std::printf("MISMATCH NE=%d trial=%d n=%d\n", NE, trial, n); }
        cudaFree(dw); cudaFree(dx); cudaFree(dl1); cudaFree(dl2); cudaFree(di1); cudaFree(di2); cudaFree(dw1); cudaFree(dw2);
        cudaStreamDestroy(s);
    }
    std::printf("%d cases, %d mismatches (logits, ids, weights bitwise)\n", cases, bad);
    return bad != 0;
}
