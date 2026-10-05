// host-to-device copy rates on the APU, timed by the host clock around synchronous and stream-synchronised copies
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s -> %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)
__global__ void rd(const float4* p, size_t n, float* out) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x, st = (size_t) gridDim.x * blockDim.x;
    float4 acc = {0, 0, 0, 0};
    for (; i < n; i += st) { float4 v = p[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
    if (acc.x + acc.y + acc.z + acc.w == 12345.f) out[0] = 1;
}
__global__ void cp(const float4* s, float4* d, size_t n) {
    size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x, st = (size_t) gridDim.x * blockDim.x;
    for (; i < n; i += st) d[i] = s[i];
}
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
    const size_t sz = 2ull << 30;
    void *pinned, *pageable = std::malloc(sz); float4* d; float* out;
    CK(hipHostMalloc(&pinned, sz, hipHostMallocMapped)); CK(hipMalloc(&d, sz)); CK(hipMalloc(&out, 4));
    std::memset(pinned, 1, sz); std::memset(pageable, 2, sz);
    for (int r = 0; r < 3; ++r) {
        double t = now(); CK(hipMemcpy(d, pinned, sz, hipMemcpyHostToDevice)); CK(hipDeviceSynchronize());
        std::printf("hipMemcpy H2D 2 GiB from pinned (sync):      %.0f ms = %.1f GB/s\n", (now() - t) * 1e3, sz / (now() - t) / 1e9);
    }
    for (int r = 0; r < 2; ++r) {
        double t = now(); CK(hipMemcpy(d, pageable, sz, hipMemcpyHostToDevice)); CK(hipDeviceSynchronize());
        std::printf("hipMemcpy H2D 2 GiB from pageable (sync):    %.0f ms = %.1f GB/s\n", (now() - t) * 1e3, sz / (now() - t) / 1e9);
    }
    hipStream_t s; CK(hipStreamCreate(&s));
    for (int r = 0; r < 2; ++r) {
        double t = now(); CK(hipMemcpyAsync(d, pinned, sz, hipMemcpyHostToDevice, s)); CK(hipStreamSynchronize(s));
        std::printf("hipMemcpyAsync H2D 2 GiB pinned + stream sync: %.0f ms = %.1f GB/s\n", (now() - t) * 1e3, sz / (now() - t) / 1e9);
    }
    void* dp; CK(hipHostGetDevicePointer(&dp, pinned, 0));
    for (int r = 0; r < 2; ++r) {
        double t = now(); cp<<<2048, 256>>>((const float4*) dp, d, sz / 16); CK(hipDeviceSynchronize());
        std::printf("kernel copies 2 GiB mapped host -> device:   %.0f ms = %.1f GB/s\n", (now() - t) * 1e3, sz / (now() - t) / 1e9);
    }
    // 2.66 MB pieces, the size of one Coder expert blob: 512 copies in a row on one stream (the prompt path's shape)
    const size_t blob = 2660000; const int nb = 512;
    double t = now();
    for (int i = 0; i < nb; ++i) CK(hipMemcpyAsync((char*) d + (size_t) i * blob, (char*) pinned + (size_t) i * blob, blob, hipMemcpyHostToDevice, s));
    CK(hipStreamSynchronize(s));
    std::printf("512 x 2.66 MB hipMemcpyAsync pinned -> device: %.0f ms = %.1f GB/s\n", (now() - t) * 1e3, blob * (double) nb / (now() - t) / 1e9);
    return 0;
}
