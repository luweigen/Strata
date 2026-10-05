// docs/TODO.md item 3, second probe: does the device-side MMQ config (a constexpr table chosen by compiler macro:
// RDNA3_5 / RDNA3 / RDNA4 ...) agree with the host-side one (chosen by the compute-capability number the glue in
// src/prefill/ggml_cuda_host.cu derives from the arch name)?  The kernel `mul_mat_q` returns at once when its own
// table has no entry for (type, J, fallback), and NO_DEVICE_CODE may be a silent no-op on HIP - which would leave the
// output unwritten, the pattern 2026-10-05-halo-mmq-parity-probe.cpp shows.  Prints both tables' rows for Q2_0 and
// IQ3_XXS at every J, the macros the device compile saw, and whether NO_DEVICE_CODE traps.
// Build (repository root; strata_mmq's flags, see build-hip-win/build.ninja):
//   clang++ -x hip --offload-arch=gfx1151 --rocm-path=<sdk> --rocm-device-lib-path=<sdk>/lib/llvm/amdgcn/bitcode
//           -O2 -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -DGGML_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1
//           -D__HIP_ROCclr__=1 -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/ggml/src
//           -Ithird_party/llama.cpp/ggml/src/ggml-cuda -Iinclude docs/benchmarks/2026-10-05-halo-mmq-config-probe.hip
//           -x none build-hip-win/strata_mmq.lib build-hip-win/ggml/src/ggml-base.lib <sdk>/lib/amdhip64.lib
//           -fuse-ld=lld -o build-hip-win/mmq_config_probe.exe
#include "common.cuh"
#include "mmq.cuh"

#include <cstdio>

// per J (8..128 step 8): type (GGML_TYPE_COUNT = no entry), nthreads, I, J, stream_k, then the macro bits
struct Row { int type, nthreads, I, J, stream_k; };

template <ggml_type type, bool fallback>
__device__ void dev_rows(Row* rows) {
#pragma unroll
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, fallback);
        rows[J / 8 - 1] = {(int) c.type, c.nthreads, c.I, c.J, (int) c.stream_k};
    }
}

__global__ void dump(Row* q2, Row* iq3, int* misc) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    dev_rows<GGML_TYPE_Q2_0, false>(q2);
    dev_rows<GGML_TYPE_IQ3_XXS, false>(iq3);
    misc[0] = ggml_cuda_get_physical_warp_size();
    int bits = 0;
#ifdef RDNA3
    bits |= 1;
#endif
#ifdef RDNA3_5
    bits |= 2;
#endif
#ifdef RDNA4
    bits |= 4;
#endif
#ifdef AMD_WMMA_AVAILABLE
    bits |= 8;
#endif
#ifdef __gfx1151__
    bits |= 16;
#endif
#ifdef __GFX11__
    bits |= 32;
#endif
#ifdef GGML_USE_HIP
    bits |= 64;
#endif
#ifdef __HIP_DEVICE_COMPILE__
    bits |= 128;
#endif
#ifdef __CUDA_ARCH__
    bits |= 256;
#endif
    misc[1] = bits;
    misc[2] = ggml_cuda_mmq_get_config(GGML_TYPE_Q2_0, 16, false).use_mma_data_layout();
}

__global__ void trap_test(int* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    out[0] = 1;
    NO_DEVICE_CODE;
    out[1] = 2;   // reached only if NO_DEVICE_CODE is a no-op
}

static void host_rows(ggml_type type, bool fallback, int cc, Row* rows) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, fallback, cc);
        rows[J / 8 - 1] = {(int) c.type, c.nthreads, c.I, c.J, (int) c.stream_k};
    }
}

static void print_rows(const char* who, const Row* r) {
    std::printf("%-6s", who);
    for (int i = 0; i < 16; ++i) {
        if (r[i].type == (int) GGML_TYPE_COUNT) std::printf(" %3d:-", (i + 1) * 8);
        else std::printf(" %3d:%d/%d/%d%s", (i + 1) * 8, r[i].nthreads, r[i].I, r[i].J, r[i].stream_k ? "k" : "");
    }
    std::printf("\n");
}

int main() {
    const ggml_cuda_device_info& info = ggml_cuda_info();
    const int cc = info.devices[0].cc;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s %s: glue cc 0x%x (RDNA3 %d, RDNA3_5 %d, RDNA4 %d), host warp %d, smpbo %zu, nsm %d, host use_mma_data_layout %d\n",
                prop.name, prop.gcnArchName, cc - GGML_CUDA_CC_OFFSET_AMD, (int) GGML_CUDA_CC_IS_RDNA3(cc),
                (int) GGML_CUDA_CC_IS_RDNA3_5(cc), (int) GGML_CUDA_CC_IS_RDNA4(cc), info.devices[0].warp_size,
                info.devices[0].smpbo, info.devices[0].nsm,
                ggml_cuda_mmq_get_config(GGML_TYPE_Q2_0, 16, false, cc).use_mma_data_layout(cc));

    Row *dq2, *diq3; int *dmisc, *dtrap;
    CUDA_CHECK(cudaMalloc(&dq2, 16 * sizeof(Row))); CUDA_CHECK(cudaMalloc(&diq3, 16 * sizeof(Row)));
    CUDA_CHECK(cudaMalloc(&dmisc, 16 * sizeof(int))); CUDA_CHECK(cudaMalloc(&dtrap, 2 * sizeof(int)));
    CUDA_CHECK(cudaMemset(dq2, 0, 16 * sizeof(Row))); CUDA_CHECK(cudaMemset(diq3, 0, 16 * sizeof(Row)));
    CUDA_CHECK(cudaMemset(dmisc, 0, 16 * sizeof(int))); CUDA_CHECK(cudaMemset(dtrap, 0, 2 * sizeof(int)));
    dump<<<1, 32>>>(dq2, diq3, dmisc);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    Row q2[16], iq3[16], hq2[16], hiq3[16]; int misc[16];
    CUDA_CHECK(cudaMemcpy(q2, dq2, sizeof(q2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(iq3, diq3, sizeof(iq3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(misc, dmisc, sizeof(misc), cudaMemcpyDeviceToHost));
    host_rows(GGML_TYPE_Q2_0, false, cc, hq2);
    host_rows(GGML_TYPE_IQ3_XXS, false, cc, hiq3);
    std::printf("device compile: warp %d, macros: RDNA3 %d RDNA3_5 %d RDNA4 %d AMD_WMMA_AVAILABLE %d __gfx1151__ %d __GFX11__ %d GGML_USE_HIP %d __HIP_DEVICE_COMPILE__ %d __CUDA_ARCH__ %d; device use_mma_data_layout %d\n",
                misc[0], misc[1] & 1, !!(misc[1] & 2), !!(misc[1] & 4), !!(misc[1] & 8), !!(misc[1] & 16), !!(misc[1] & 32),
                !!(misc[1] & 64), !!(misc[1] & 128), !!(misc[1] & 256), misc[2]);
    std::printf("Q2_0 non-fallback rows per J (nthreads/I/J, k = stream-k, - = no entry):\n");
    print_rows("host", hq2); print_rows("device", q2);
    std::printf("IQ3_XXS non-fallback rows per J:\n");
    print_rows("host", hiq3); print_rows("device", iq3);

    std::fflush(stdout);
    trap_test<<<1, 32>>>(dtrap);
    const cudaError_t launch = cudaGetLastError();
    const cudaError_t sync = cudaDeviceSynchronize();
    int trap[2] = {-1, -1};
    if (sync == cudaSuccess) (void) cudaMemcpy(trap, dtrap, sizeof(trap), cudaMemcpyDeviceToHost);
    std::printf("NO_DEVICE_CODE: launch %s, sync %s, out[0]=%d out[1]=%d (%s)\n", cudaGetErrorString(launch),
                cudaGetErrorString(sync), trap[0], trap[1],
                sync != cudaSuccess ? "traps" : trap[1] == 2 ? "silent no-op: the kernel went on" : "stopped the thread");
    return 0;
}
