// the shader clock the GPU actually runs at under load: HW_REG_SHADER_CYCLES (a 20-bit cycle counter on gfx10/11,
// s_getreg id 29) sampled over 100 us of the 100 MHz wall clock; gfx11 has no s_memtime, so clock64() is useless here
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define SHADER_CYCLES_REG (29 | (0 << 6) | ((20 - 1) << 11))
__global__ void spin(long long* out, long long ticks) {
    const long long w0 = wall_clock64();
    float x = threadIdx.x * 1.0001f;
    double mhz_sum = 0; int n = 0;
    while (wall_clock64() - w0 < ticks) {
        const long long a = wall_clock64();
        const unsigned c0 = __builtin_amdgcn_s_getreg(SHADER_CYCLES_REG);
        while (wall_clock64() - a < 10000) {   // 100 us
            #pragma unroll
            for (int i = 0; i < 64; ++i) x = x * 1.000001f + 0.5f;
        }
        const unsigned c1 = __builtin_amdgcn_s_getreg(SHADER_CYCLES_REG);
        const long long b = wall_clock64();
        const unsigned cyc = (c1 - c0) & 0xFFFFFu;
        mhz_sum += cyc / ((b - a) / 100.0);   // cycles per us = MHz
        ++n;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = wall_clock64() - w0; out[1] = (long long) (mhz_sum / n); }
    if (x == 12345.f) out[2] = 1;
}
int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    const int blocks = argc > 2 ? std::atoi(argv[2]) : 40;
    long long* d; hipMallocManaged(&d, 32);
    for (int r = 0; r < reps; ++r) {
        spin<<<blocks, 256>>>(d, 50000000);   // 0.5 s
        hipDeviceSynchronize();
        std::printf("sample %d: %.2f s, shader clock ~%lld MHz\n", r, d[0] / 1e8, d[1]);
    }
    return 0;
}
