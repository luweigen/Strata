// does pinning (registering, mapped) host memory lower hipMemGetInfo's free figure on this APU, and by how much?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <windows.h>
static void info(const char* what) {
    size_t fr = 0, tot = 0; hipMemGetInfo(&fr, &tot);
    std::printf("%-46s free %6.1f GiB of %6.1f\n", what, fr / 1073741824.0, tot / 1073741824.0);
}
__global__ void touch(float* p, size_t n) { size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; if (i < n) p[i] = 1.f; }
int main(int argc, char** argv) {
    const size_t gib = argc > 1 ? std::atoi(argv[1]) : 16;
    info("start");
    void* h = VirtualAlloc(nullptr, gib << 30, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    std::memset(h, 1, gib << 30);
    info("after VirtualAlloc + memset (pageable)");
    hipError_t e = hipHostRegister(h, gib << 30, hipHostRegisterPortable | hipHostRegisterMapped);
    std::printf("hipHostRegister %zu GiB: %s\n", gib, hipGetErrorString(e));
    info("after hipHostRegister (pinned, mapped)");
    float* d = nullptr; const size_t dsz = 20ull << 30;
    e = hipMalloc(&d, dsz);
    std::printf("hipMalloc 20 GiB: %s\n", hipGetErrorString(e));
    if (e == hipSuccess) {
        touch<<<(unsigned) ((dsz / 4 + 255) / 256), 256>>>(d, dsz / 4);
        std::printf("kernel over it: %s\n", hipGetErrorString(hipDeviceSynchronize()));
        info("after hipMalloc 20 GiB");
    }
    hipHostUnregister(h);
    info("after hipHostUnregister");
    return 0;
}
