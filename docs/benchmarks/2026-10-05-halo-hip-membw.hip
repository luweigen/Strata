#include <hip/hip_runtime.h>
#include <chrono>
#include <thread>
#include <cstdio>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s -> %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)
__global__ void rd(const float4* p, size_t n, float* out) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x, st = (size_t) gridDim.x * blockDim.x;
    float4 acc = {0, 0, 0, 0};
    for (; i < n; i += st) { float4 v = p[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    if (acc.x + acc.y + acc.z + acc.w == 12345.f) out[0] = 1;
}
__global__ void wr(float4* p, size_t n) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x, st = (size_t) gridDim.x * blockDim.x;
    for (; i < n; i += st) p[i] = make_float4(1, 2, 3, 4);
}
__global__ void clk(long long* c) { if (threadIdx.x == 0) c[0] = (long long) wall_clock64(); }
static float ms(hipEvent_t a, hipEvent_t b) { float t = 0; hipEventElapsedTime(&t, a, b); return t; }
int main() {
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    float* out; CK(hipMalloc(&out, 4));
    // 1. a 48 GiB device allocation (the expert cache of a big model would be ~30-45 GiB)
    const size_t big = 48ull << 30; float4* d = nullptr;
    hipError_t e = hipMalloc(&d, big);
    std::printf("hipMalloc 48 GiB: %s\n", hipGetErrorString(e));
    if (e != hipSuccess) return 1;
    size_t n = big / 16;
    for (int r = 0; r < 2; ++r) {
        CK(hipEventRecord(a)); wr<<<2048, 256>>>(d, n); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        std::printf("  write 48 GiB pass %d: %.0f ms = %.1f GB/s\n", r, ms(a, b), big / ms(a, b) / 1e6);
    }
    for (int r = 0; r < 3; ++r) {
        CK(hipEventRecord(a)); rd<<<2048, 256>>>(d, n, out); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        std::printf("  read 48 GiB pass %d: %.0f ms = %.1f GB/s\n", r, ms(a, b), big / ms(a, b) / 1e6);
    }
    // 2. pinned host memory, 4 GiB: copies both ways, and the GPU reading it in place through the mapped pointer
    const size_t sz = 4ull << 30; void* h = nullptr;
    CK(hipHostMalloc(&h, sz, hipHostMallocMapped));
    std::memset(h, 1, sz);
    for (int r = 0; r < 2; ++r) {
        CK(hipEventRecord(a)); CK(hipMemcpyAsync(d, h, sz, hipMemcpyHostToDevice, 0)); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        std::printf("  H2D 4 GiB pinned pass %d: %.0f ms = %.1f GB/s\n", r, ms(a, b), sz / ms(a, b) / 1e6);
    }
    CK(hipEventRecord(a)); CK(hipMemcpyAsync(h, d, sz, hipMemcpyDeviceToHost, 0)); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    std::printf("  D2H 4 GiB pinned: %.0f ms = %.1f GB/s\n", ms(a, b), sz / ms(a, b) / 1e6);
    void* hd = nullptr; CK(hipHostGetDevicePointer(&hd, h, 0));
    for (int r = 0; r < 2; ++r) {
        CK(hipEventRecord(a)); rd<<<2048, 256>>>((const float4*) hd, sz / 16, out); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
        std::printf("  GPU reads 4 GiB mapped host memory pass %d: %.0f ms = %.1f GB/s\n", r, ms(a, b), sz / ms(a, b) / 1e6);
    }
    // 3. wall_clock64 rate (the engine assumes 100 MHz on gfx10.3 / gfx11 / gfx12)
    long long* c; CK(hipMallocManaged(&c, 16));
    clk<<<1, 32>>>(c); CK(hipDeviceSynchronize()); long long c0 = c[0];
    auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    clk<<<1, 32>>>(c); CK(hipDeviceSynchronize());
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("wall_clock64: %.2f MHz\n", (c[0] - c0) / s / 1e6);
    return 0;
}
