// docs/TODO.md item 3, third probe: the MMQ kernel itself, instantiated in this translation unit so its optimization
// level, mode and stream can be varied.  2026-10-05-halo-mmq-parity-probe.cpp showed every output element unwritten
// through the glue; 2026-10-05-halo-mmq-config-probe.hip showed the host and device tile tables agree.  Here one Q2_0
// expert [1280 x 2560] times 16 activation rows runs several ways and the written elements are counted and checked:
//   lib   : mul_mat_q_case<Q2_0> from strata_mmq.lib (what the glue calls), MoE mode (ids + expert bounds)
//   moe   : launch_mul_mat_q<Q2_0, J, false> instantiated here, MoE mode
//   plain : the same, plain GEMM mode (no ids, no bounds)
// all on the null stream; then `lib` on a blocking stream and on a non-blocking stream (the parity test's kind), the
// quantizer and the sentinel fill on the same stream, the result read after hipStreamSynchronize; then the
// non-blocking stream with hipDeviceSynchronize before the read; then the test's exact order (sentinel fill with
// hipMemset on the null stream, products on the non-blocking stream).
// Build twice, -O3 and -O1, and compare (repository root; strata_mmq's flags, see build-hip-win/build.ninja):
//   clang++ -x hip --offload-arch=gfx1151 --rocm-path=<sdk> --rocm-device-lib-path=<sdk>/lib/llvm/amdgcn/bitcode
//           -O3 -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -DGGML_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1
//           -D__HIP_ROCclr__=1 -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/ggml/src
//           -Ithird_party/llama.cpp/ggml/src/ggml-cuda -Iinclude docs/benchmarks/2026-10-05-halo-mmq-kernel-probe.hip
//           -x none build-hip-win/strata_mmq.lib build-hip-win/ggml/src/ggml-base.lib <sdk>/lib/amdhip64.lib
//           -fuse-ld=lld -o build-hip-win/mmq_kernel_probe_O3.exe
#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int64_t ROWS_W = 1280, COLS = 2560;
constexpr int J_TILE = 16;   // the tile mul_mat_q_switch_J picks for up to 16 rows on RDNA 3.5

std::vector<uint8_t> q2_weights(int64_t rows, int64_t cols) {
    std::vector<float> f((size_t) rows * cols);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            f[(size_t) r * cols + k] = 0.025f * std::sin(0.013f * (float) (k + 1) + 0.17f * (float) r) +
                                       0.0015f * (float) ((int) ((k * 37 + r * 19) % 101) - 50);
    const auto* tr = ggml_get_type_traits(GGML_TYPE_Q2_0);
    const size_t rb = ggml_row_size(GGML_TYPE_Q2_0, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    for (int64_t r = 0; r < rows; ++r) tr->from_float_ref(f.data() + (size_t) r * cols, out.data() + (size_t) r * rb, cols);
    return out;
}

std::vector<float> activations(int rows, int64_t cols) {
    std::vector<float> x((size_t) rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * cols + k] = 0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r) +
                                       0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r);
    return x;
}

struct Setup {
    int rows;
    int64_t pad;
    std::vector<uint8_t> w;
    std::vector<float> x, ref;
    char* dw; int* dxq; float *dx, *dy; int32_t *dids, *dbounds;
    size_t ny;
};

Setup make(int rows) {
    Setup s; s.rows = rows;
    s.w = q2_weights(ROWS_W, COLS);
    s.x = activations(rows, COLS);
    // reference
    const auto* tr = ggml_get_type_traits(GGML_TYPE_Q2_0);
    const size_t rb = ggml_row_size(GGML_TYPE_Q2_0, COLS);
    std::vector<float> wd((size_t) ROWS_W * COLS);
    for (int64_t o = 0; o < ROWS_W; ++o) tr->to_float(s.w.data() + (size_t) o * rb, wd.data() + (size_t) o * COLS, COLS);
    s.ref.assign((size_t) rows * ROWS_W, 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int64_t o = 0; o < ROWS_W; ++o) {
            double acc = 0;
            for (int64_t k = 0; k < COLS; ++k) acc += (double) wd[(size_t) o * COLS + k] * s.x[(size_t) r * COLS + k];
            s.ref[(size_t) r * ROWS_W + o] = (float) acc;
        }
    s.ny = (size_t) rows * ROWS_W;
    std::vector<int32_t> ids(rows), bounds{0, rows};
    for (int i = 0; i < rows; ++i) ids[i] = i;
    s.pad = (COLS + 511) / 512 * 512;
    const size_t xq_bytes = (size_t) rows * s.pad * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
    CUDA_CHECK(cudaMalloc(&s.dw, s.w.size() + 4096));
    CUDA_CHECK(cudaMemset(s.dw, 0, s.w.size() + 4096));
    CUDA_CHECK(cudaMemcpy(s.dw, s.w.data(), s.w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&s.dx, s.x.size() * 4));
    CUDA_CHECK(cudaMemcpy(s.dx, s.x.data(), s.x.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&s.dxq, xq_bytes));
    CUDA_CHECK(cudaMalloc(&s.dy, s.ny * 4));
    CUDA_CHECK(cudaMalloc(&s.dids, rows * 4));
    CUDA_CHECK(cudaMalloc(&s.dbounds, 2 * 4));
    CUDA_CHECK(cudaMemcpy(s.dids, ids.data(), rows * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s.dbounds, bounds.data(), 2 * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    return s;
}

void quantize_on(const Setup& s, cudaStream_t st) {
    quantize_mmq_q8_1_cuda(s.dx, nullptr, s.dxq, GGML_TYPE_Q2_0, COLS, COLS, (int64_t) s.rows * COLS, (int64_t) s.rows * COLS,
                           s.pad, s.rows, 1, 1, st);
    CUDA_CHECK(cudaGetLastError());
}

mmq_args args_for(const Setup& s, bool moe) {
    const int64_t bpr = COLS / QK2_0;
    if (moe) {
        return {s.dw, GGML_TYPE_Q2_0, s.dxq, s.dids, s.dbounds, s.dy, nullptr,
                COLS, ROWS_W, s.rows, bpr, s.rows, ROWS_W,
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                s.rows, s.rows};
    }
    return {s.dw, GGML_TYPE_Q2_0, s.dxq, nullptr, nullptr, s.dy, nullptr,
            COLS, ROWS_W, s.rows, bpr, s.rows, ROWS_W,
            1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            s.rows, s.rows};
}

// read the result after the given synchronization and classify it
void report(const char* who, const Setup& s, cudaStream_t st, bool device_sync) {
    CUDA_CHECK(cudaGetLastError());
    const cudaError_t sync = device_sync ? cudaDeviceSynchronize() : cudaStreamSynchronize(st);
    std::vector<float> got(s.ny);
    const cudaError_t cp = cudaMemcpy(got.data(), s.dy, s.ny * 4, cudaMemcpyDeviceToHost);
    if (sync != cudaSuccess || cp != cudaSuccess) {
        std::printf("%-34s sync %s, copy %s\n", who, cudaGetErrorString(sync), cudaGetErrorString(cp));
        std::exit(1);
    }
    size_t sent = 0, nan = 0; double err2 = 0, ref2 = 0, max_abs = 0;
    for (size_t i = 0; i < s.ny; ++i) {
        uint32_t bits; std::memcpy(&bits, &got[i], 4);
        if (bits == 0xffffffffu) { ++sent; continue; }
        if (std::isnan(got[i])) { ++nan; continue; }
        const double d = (double) got[i] - s.ref[i];
        err2 += d * d; ref2 += (double) s.ref[i] * s.ref[i]; max_abs = std::max(max_abs, std::fabs(d));
    }
    const size_t ok = s.ny - sent - nan;
    const double rms = ok ? std::sqrt(ref2 / (double) ok) : 0;
    std::printf("%-34s written %zu of %zu (unwritten %zu, nan %zu), rel_l2 %.4f, max/rms %.3f\n", who, ok, s.ny, sent, nan,
                ok && rms > 0 ? std::sqrt(err2 / (double) ok) / rms : 0.0, rms > 0 ? max_abs / rms : 0.0);
}
}  // namespace

int main() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    ggml_backend_cuda_context ctx(dev);
    const int cc = ggml_cuda_info().devices[dev].cc;
    const ggml_cuda_mmq_config cfg = ggml_cuda_mmq_get_config(GGML_TYPE_Q2_0, J_TILE, false, cc);
    std::printf("Q2_0 [%lld x %lld] x 16 rows, J=%d: nthreads %d, I %d, shared %zu B, stream-k %d; compiled at %s\n",
                (long long) ROWS_W, (long long) COLS, J_TILE, cfg.nthreads, cfg.I, mmq_get_nbytes_shared(cfg, cc),
                (int) cfg.stream_k,
#ifdef __OPTIMIZE__
                "-O (optimized)"
#else
                "-O0"
#endif
    );
    Setup s = make(16);
    cudaStream_t sb, snb;
    CUDA_CHECK(hipStreamCreate(&sb));
    CUDA_CHECK(hipStreamCreateWithFlags(&snb, hipStreamNonBlocking));

    // null stream: the three kernels
    quantize_on(s, nullptr);
    CUDA_CHECK(cudaMemsetAsync(s.dy, 0xff, s.ny * 4, nullptr));
    mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args_for(s, true), nullptr);
    report("lib, null stream", s, nullptr, false);
    CUDA_CHECK(cudaMemsetAsync(s.dy, 0xff, s.ny * 4, nullptr));
    launch_mul_mat_q<GGML_TYPE_Q2_0, J_TILE, false>(ctx, args_for(s, true), nullptr);
    report("moe (this TU), null stream", s, nullptr, false);
    CUDA_CHECK(cudaMemsetAsync(s.dy, 0xff, s.ny * 4, nullptr));
    launch_mul_mat_q<GGML_TYPE_Q2_0, J_TILE, false>(ctx, args_for(s, false), nullptr);
    report("plain (this TU), null stream", s, nullptr, false);

    // the lib kernel on created streams, everything on that stream
    for (int k = 0; k < 2; ++k) {
        cudaStream_t st = k == 0 ? sb : snb;
        const char* name = k == 0 ? "lib, blocking stream" : "lib, non-blocking stream";
        CUDA_CHECK(cudaMemsetAsync(s.dy, 0xff, s.ny * 4, st));
        quantize_on(s, st);
        mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args_for(s, true), st);
        report(name, s, st, false);
    }
    // non-blocking stream, read after hipDeviceSynchronize
    CUDA_CHECK(cudaMemsetAsync(s.dy, 0xff, s.ny * 4, snb));
    quantize_on(s, snb);
    mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args_for(s, true), snb);
    report("lib, non-blocking, device sync", s, snb, true);
    // the parity test's exact order: sentinel fill on the null stream, products on the non-blocking stream
    CUDA_CHECK(cudaMemset(s.dy, 0xff, s.ny * 4));
    quantize_on(s, snb);
    mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args_for(s, true), snb);
    report("lib, test order (null memset)", s, snb, false);
    // and once more on the null stream, to see the state after the non-blocking runs
    CUDA_CHECK(cudaMemsetAsync(s.dy, 0xff, s.ny * 4, nullptr));
    quantize_on(s, nullptr);
    mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args_for(s, true), nullptr);
    report("lib, null stream again", s, nullptr, false);
    return 0;
}
