#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if __has_builtin(__builtin_amdgcn_sudot4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    return -999999;
#endif
}
__global__ void k(int* out, unsigned* perm, int* shfl, long long* clk) {
    int a = 0x01FF0203;  // bytes: 3, 2, -1, 1
    int b = 0x04030201;  // bytes: 1, 2, 3, 4
    out[0] = dp4a(a, b, 10);   // 3*1 + 2*2 + (-1)*3 + 1*4 + 10 = 18
    perm[0] = __builtin_amdgcn_perm(0xAABBCCDDu, 0x11223344u, 0x05040100u);
    int v = threadIdx.x;
    int s = __shfl_xor(v, 1, 32);
    if (threadIdx.x == 0) { shfl[0] = s; shfl[1] = warpSize; }
    if (threadIdx.x == 0) clk[0] = (long long) wall_clock64();
}
int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    std::printf("device: %s gcnArchName=%s warp=%d LDS/block=%zu totalGlobalMem=%.1f GiB integrated=%d\n",
        p.name, p.gcnArchName, p.warpSize, p.sharedMemPerBlock, p.totalGlobalMem / 1073741824.0, p.integrated);
    int* out; unsigned* perm; int* shfl; long long* clk;
    hipMallocManaged(&out, 4); hipMallocManaged(&perm, 4); hipMallocManaged(&shfl, 8); hipMallocManaged(&clk, 16);
    k<<<1, 32>>>(out, perm, shfl, clk);
    hipError_t e = hipDeviceSynchronize();
    std::printf("launch: %s\n", hipGetErrorString(e));
    long long c0 = clk[0];
    k<<<1, 32>>>(out, perm, shfl, clk); hipDeviceSynchronize();
    std::printf("dp4a=%d (want 18)  perm=0x%08x  shfl_xor(0,1)=%d warpSize=%d  wall_clock64 delta=%lld\n",
        out[0], perm[0], shfl[0], shfl[1], clk[0] - c0);
    // pinned host memory + mapped pointer, as the engine uses
    void* h = nullptr; void* d = nullptr;
    e = hipHostMalloc(&h, 1u << 30, hipHostMallocMapped);
    std::printf("hipHostMalloc 1 GiB mapped: %s\n", hipGetErrorString(e));
    if (e == hipSuccess) { hipHostGetDevicePointer(&d, h, 0); std::printf("  host %p device %p %s\n", h, d, h == d ? "(same pointer)" : "(different)"); }
    size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot);
    std::printf("hipMemGetInfo: free %.1f GiB / total %.1f GiB\n", fr / 1073741824.0, tot / 1073741824.0);
    int rv = 0, dv = 0; hipRuntimeGetVersion(&rv); hipDriverGetVersion(&dv);
    std::printf("runtime %d driver %d\n", rv, dv);
    return 0;
}
