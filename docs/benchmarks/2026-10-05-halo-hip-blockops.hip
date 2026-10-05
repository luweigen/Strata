// docs/benchmarks/2026-10-05-halo-hip-blockops.hip - what the building blocks of the chunked GDN scan kernel cost on
// the Radeon 8060S, in the kernel's own shape (192 blocks of 128 threads, 42 KB of LDS per block, so 3 blocks per
// WGP): a barrier, an LDS store with the transposed tiles' 4-way bank conflict, a float4 LDS load pair feeding 16
// FMAs, a batch of coalesced global loads, a dependent FMA chain, and a sum-of-float4-dot (the 4x4 tile inner loop).
// Each kernel repeats its operation N times; the table prints the time per repetition per block.
// Build: as 2026-10-05-halo-hip-latency.hip.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int LDS_FLOATS = 42240 / 4;

__global__ void __launch_bounds__(128) k_barrier(float* out, int n) {
    __shared__ float lds[LDS_FLOATS];
    lds[threadIdx.x] = 0.0f;
    for (int i = 0; i < n; ++i) __syncthreads();
    if (lds[threadIdx.x ^ 1] != 0.0f) out[blockIdx.x] = 1.0f;
}
__global__ void __launch_bounds__(128) k_lds_store_t(float* out, int n) {   // lanes vary the row: stride 68
    __shared__ float lds[LDS_FLOATS];
    const int kk = threadIdx.x & 31, t = threadIdx.x >> 5;
    for (int i = 0; i < n; ++i) lds[kk * 68 + t * 4 + (i & 3)] = (float) i;
    __syncthreads();
    out[blockIdx.x] = lds[threadIdx.x];
}
__global__ void __launch_bounds__(128) k_lds_store_c(float* out, int n) {   // lanes contiguous
    __shared__ float lds[LDS_FLOATS];
    for (int i = 0; i < n; ++i) lds[(i & 15) * 128 + threadIdx.x] = (float) i;
    __syncthreads();
    out[blockIdx.x] = lds[threadIdx.x];
}
__global__ void __launch_bounds__(128) k_tile(float* out, int n) {   // the 4x4 tile loop: 2 float4 loads, 16 FMAs
    __shared__ __align__(16) float sX[64 * 68];
    __shared__ __align__(16) float sS[128 * 32];
    for (int e = threadIdx.x; e < 64 * 68; e += 128) sX[e] = 1e-3f * e;
    for (int e = threadIdx.x; e < 128 * 32; e += 128) sS[e] = 1e-3f * e;
    __syncthreads();
    const int tc = threadIdx.x % 8, tt = threadIdx.x / 8;
    float acc[4][4] = {};
    for (int i = 0; i < n; ++i) {
        const int kk = i & 31;
        const float4 a = *(const float4*) (sX + kk * 68 + tt * 4);
        const float4 b = *(const float4*) (sS + kk * 32 + tc * 4);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int y = 0; y < 4; ++y) acc[x][y] = fmaf(av[x], bv[y], acc[x][y]);
    }
    float s = 0.0f;
#pragma unroll
    for (int x = 0; x < 4; ++x)
#pragma unroll
        for (int y = 0; y < 4; ++y) s += acc[x][y];
    out[blockIdx.x * 128 + threadIdx.x] = s;
}
__global__ void __launch_bounds__(128) k_fma_chain(float* out, int n) {
    __shared__ float lds[LDS_FLOATS];
    float x = threadIdx.x * 1e-3f;
    for (int i = 0; i < n; ++i) {
#pragma unroll
        for (int j = 0; j < 32; ++j) x = fmaf(x, 0.999f, 1e-3f);
    }
    lds[threadIdx.x] = x;
    __syncthreads();
    out[blockIdx.x * 128 + threadIdx.x] = lds[threadIdx.x ^ 1];
}
__global__ void __launch_bounds__(128) k_gload(const float* __restrict__ h, float* out, int n) {   // 16 coalesced loads a batch
    __shared__ float lds[LDS_FLOATS];
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) v[j] = h[((size_t) (i * 16 + j) % 4096) * 10240 + blockIdx.x * 32 + threadIdx.x];
#pragma unroll
        for (int j = 0; j < 16; ++j) s += v[j];
    }
    lds[threadIdx.x] = s;
    __syncthreads();
    out[blockIdx.x * 128 + threadIdx.x] = lds[threadIdx.x ^ 1];
}

int main(int argc, char** argv) {
    const int grid = argc > 1 ? std::atoi(argv[1]) : 192;
    float* out = nullptr;
    float* h = nullptr;
    CHECK(hipMalloc(&out, 192 * 128 * 4));
    CHECK(hipMalloc(&h, (size_t) 4096 * 10240 * 4));
    CHECK(hipMemset(h, 0, (size_t) 4096 * 10240 * 4));
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    auto time = [&](const char* label, auto launch, int n, int per_iter) {
        float best = 1e30f;
        for (int r = 0; r < 3; ++r) {
            CHECK(hipEventRecord(a, nullptr));
            launch(n);
            CHECK(hipEventRecord(b, nullptr));
            CHECK(hipEventSynchronize(b));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, a, b));
            if (ms < best) best = ms;
        }
        std::printf("%-52s grid %3d, %6d reps: %8.3f ms = %7.1f ns per rep per block, %6.2f ns per %s\n", label, grid, n,
                    best, best * 1e6 / n, best * 1e6 / n / per_iter, "unit");
    };
    const int n = 20000;
    time("barrier (128 threads, 42 KB LDS)", [&](int k) { k_barrier<<<grid, 128>>>(out, k); }, n, 1);
    time("LDS store, lanes down a column (stride 68)", [&](int k) { k_lds_store_t<<<grid, 128>>>(out, k); }, n, 1);
    time("LDS store, lanes contiguous", [&](int k) { k_lds_store_c<<<grid, 128>>>(out, k); }, n, 1);
    time("4x4 tile step: 2 float4 LDS loads + 16 FMAs", [&](int k) { k_tile<<<grid, 128>>>(out, k); }, n, 16);
    time("32 dependent FMAs", [&](int k) { k_fma_chain<<<grid, 128>>>(out, k); }, n, 32);
    time("16 coalesced global loads (40 KB row stride)", [&](int k) { k_gload<<<grid, 128>>>(h, out, k); }, 2000, 16);
    return 0;
}
