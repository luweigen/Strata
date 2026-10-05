// docs/benchmarks/2026-10-05-halo-hip-latency.hip - the GPU's load latency on the Radeon 8060S (TODO 5): one lane
// chases pointers through a device buffer, so every load waits for the previous one.  Strides: 128 B (one cache line
// after another, the TLB warm), 40 KB (one prompt-path h row to the next, as the GDN kernels walk h), and random over
// the buffer.  Also the same chase from a single wave with 32 lanes on 32 chains (the latency does not change; the
// throughput does), and a plain dependent chain inside a 64 KB window (the L0/L1 hit latency).
// Build (the SDK clang++, see windows-shell-gotchas):
//   clang++ --offload-arch=gfx1151 --rocm-path=<sdk> --rocm-device-lib-path=<sdk>/lib/llvm/amdgcn/bitcode -std=c++20
//     -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -O2 -x hip 2026-10-05-halo-hip-latency.hip -x none <sdk>/lib/amdhip64.lib
//     -fuse-ld=lld -Xlinker /subsystem:console -o hip-latency.exe
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

__global__ void chase(const unsigned* __restrict__ next, unsigned start, int steps, unsigned* out, long long* cycles) {
    unsigned p = start + threadIdx.x;
    const long long t0 = clock64();
    for (int i = 0; i < steps; ++i) p = next[p];
    const long long t1 = clock64();
    if (threadIdx.x == 0) { out[0] = p; cycles[0] = t1 - t0; }
}

int main() {
    const size_t bytes = 512ull << 20, n = bytes / 4;
    std::vector<unsigned> h(n, 0u);
    unsigned* d = nullptr;
    unsigned* out = nullptr;
    long long* cyc = nullptr;
    CHECK(hipMalloc(&d, bytes));
    CHECK(hipMalloc(&out, 4096));
    CHECK(hipMalloc(&cyc, 64));
    hipDeviceProp_t prop{};
    CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("%s, clock %d kHz\n", prop.gcnArchName, prop.clockRate);
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    auto run = [&](const char* label, size_t stride_elems, bool random, int lanes, int steps) {
        // chain c (c < lanes): the lane's start index is c, the chase steps by stride (or randomly) from there
        std::vector<unsigned> order;
        const size_t slots = n / stride_elems;
        if (random) {
            order.resize(slots);
            for (size_t i = 0; i < slots; ++i) order[i] = (unsigned) i;
            std::shuffle(order.begin(), order.end(), std::mt19937(7));
        }
        for (size_t i = 0; i < slots; ++i) {
            const size_t cur = random ? order[i] : i, nxt = random ? order[(i + 1) % slots] : (i + 1) % slots;
            for (int c = 0; c < 32; ++c) h[cur * stride_elems + c] = (unsigned) (nxt * stride_elems);
        }
        CHECK(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice));
        float best = 1e30f;
        long long best_cyc = 0;
        for (int r = 0; r < 3; ++r) {
            CHECK(hipEventRecord(a, nullptr));
            chase<<<1, lanes>>>(d, 0u, steps, out, cyc);
            CHECK(hipEventRecord(b, nullptr));
            CHECK(hipEventSynchronize(b));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, a, b));
            long long c = 0;
            CHECK(hipMemcpy(&c, cyc, 8, hipMemcpyDeviceToHost));
            if (ms < best) { best = ms; best_cyc = c; }
        }
        std::printf("%-44s %d lanes, %d steps: %.3f ms, %.0f ns per load, %lld clocks per load (clock64 / steps)\n", label, lanes,
                    steps, best, best * 1e6 / steps, best_cyc / steps);
    };
    run("sequential 128 B stride (TLB warm)", 32, false, 1, 20000);
    run("40 KB stride (an h row to the next)", 10240, false, 1, 20000);
    run("random over 512 MB, 40 KB slots", 10240, true, 1, 20000);
    run("random over 512 MB, 4 KB slots", 1024, true, 1, 20000);
    run("random over 512 MB, 64 KB slots", 16384, true, 1, 8000);
    run("40 KB stride, a wave of 32 chains", 10240, false, 32, 20000);
    // the cache hit latency: a 4 KB ring, 128 B apart
    {
        const size_t stride = 32, slots = 32;
        for (size_t i = 0; i < slots; ++i)
            for (int c = 0; c < 32; ++c) h[i * stride + c] = (unsigned) (((i + 1) % slots) * stride);
        CHECK(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice));
        for (int r = 0; r < 3; ++r) {
            CHECK(hipEventRecord(a, nullptr));
            chase<<<1, 1>>>(d, 0u, 200000, out, cyc);
            CHECK(hipEventRecord(b, nullptr));
            CHECK(hipEventSynchronize(b));
            float ms = 0.0f;
            CHECK(hipEventElapsedTime(&ms, a, b));
            long long c = 0;
            CHECK(hipMemcpy(&c, cyc, 8, hipMemcpyDeviceToHost));
            if (r == 2) std::printf("%-44s 1 lanes, 200000 steps: %.3f ms, %.0f ns per load, %lld clocks per load\n",
                                    "4 KB ring (cache hit)", ms, ms * 1e6 / 200000, c / 200000);
        }
    }
    return 0;
}
