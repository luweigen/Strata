// docs/TODO.md item 3: `hip_prefill_mmq_parity` fails on gfx1151 with "synthetic-Q2_0-GU-pass0: non-finite or
// unwritten MMQ output".  The test stops at the first bad value; this probe runs the same products through the same
// glue (src/prefill/moe_mmq.cu -> llama.cpp's mmq.cuh, linked from build-hip-win/strata_mmq.lib) and classifies every
// output element: the 0xffffffff sentinel the output was filled with (never written), another NaN or an inf (computed
// garbage), or a finite value compared with a CPU double product over ggml's own dequantizer.  Varied: the weight type
// (Q2_0, Q8_0, IQ4_NL, IQ3_XXS, IQ2_S), the shape (gate/up [1280 x 2560], down [2560 x 640]), the rows per expert
// (the test's {1,3,3} and more), the dst row map (a permutation or the identity) and Product::opt_rows.  `race` as an
// argument fills the sentinel with hipMemset on the null stream as the test did, the products on a non-blocking
// stream (nothing orders the two); the default fills it with hipMemsetAsync on that stream.
// Build (repository root; the parity test's flags, see build-hip-win/build.ninja):
//   clang++ -x hip --offload-arch=gfx1151 --rocm-path=<sdk> --rocm-device-lib-path=<sdk>/lib/llvm/amdgcn/bitcode
//           -O2 -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1
//           -Iinclude -Ithird_party/llama.cpp/ggml/include docs/benchmarks/2026-10-05-halo-mmq-parity-probe.cpp
//           build-hip-win/strata_mmq.lib build-hip-win/ggml/src/ggml-base.lib <sdk>/lib/amdhip64.lib
//           -fuse-ld=lld -o build-hip-win/mmq_parity_probe.exe
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {
using namespace strata::prefill::mmq;

struct Case { ggml_type type; int64_t rows, cols; std::vector<int> counts; bool identity; int64_t opt; };

// one expert's weights: the test's sine + code pattern, quantized by ggml (Q2_0 through its row quantizer, the
// rest through ggml_quantize_chunk, which the i-quants need)
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, int expert, int trial) {
    std::vector<float> f((size_t) rows * cols);
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k) {
            const int code = (int) ((k * 37 + r * 19 + expert * 23 + trial * 11) % 101) - 50;
            f[(size_t) r * cols + k] = 0.025f * std::sin(0.013f * (float) (k + 1) + 0.17f * (float) r + 0.31f * (float) expert) +
                                       0.0015f * (float) code;
        }
    const size_t rb = ggml_row_size(t, cols);
    std::vector<uint8_t> out((size_t) rows * rb);
    const auto* tr = ggml_get_type_traits(t);
    if (t == GGML_TYPE_Q2_0) {
        for (int64_t r = 0; r < rows; ++r) tr->from_float_ref(f.data() + (size_t) r * cols, out.data() + (size_t) r * rb, cols);
    } else {
        ggml_quantize_init(t);
        ggml_quantize_chunk(t, f.data(), out.data(), 0, rows, cols, nullptr);
    }
    return out;
}

std::vector<float> activations(int rows, int64_t cols, int trial) {
    std::vector<float> x((size_t) rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k)
            x[(size_t) r * cols + k] = 0.45f * std::sin(0.009f * (float) (k + 1) + 0.23f * (float) r + 0.07f * (float) trial) +
                                       0.17f * std::cos(0.021f * (float) (k + 3) - 0.19f * (float) r) +
                                       0.002f * (float) (((k * 7 + r * 13 + trial * 5) % 19) - 9);
    return x;
}

std::string counts_str(const std::vector<int>& c) {
    std::string s = "{";
    for (size_t i = 0; i < c.size(); ++i) s += (i ? "," : "") + std::to_string(c[i]);
    return s + "}";
}

bool g_race = false;   // "race": the sentinel fill on the null stream, as the test did it

bool run(Context& ctx, hipStream_t s, const Case& c, int trial) {
    const int rows = std::accumulate(c.counts.begin(), c.counts.end(), 0), n = (int) c.counts.size();
    std::vector<int32_t> src(rows), dst(rows), bounds(n + 1, 0);
    for (int i = 0; i < rows; ++i) {
        src[i] = c.identity ? i : (i + trial + 1) % rows;
        dst[i] = c.identity ? i : (rows - 1 - i + trial + 2) % rows;
    }
    for (int e = 0; e < n; ++e) bounds[e + 1] = bounds[e] + c.counts[e];
    const size_t eb = matrix_bytes((int) c.type, c.rows, c.cols), tail = 4096;
    std::vector<std::vector<uint8_t>> w(n);
    std::vector<uint8_t> wall((size_t) n * eb + tail, 0);
    for (int e = 0; e < n; ++e) { w[e] = weights(c.type, c.rows, c.cols, e, trial); std::memcpy(wall.data() + (size_t) e * eb, w[e].data(), eb); }
    const std::vector<float> x = activations(rows, c.cols, trial);

    float *dx, *dy; int32_t *dsrc, *ddst, *dbounds; void *dw, *dxq;
    HC(hipMalloc(&dx, x.size() * 4)); HC(hipMalloc(&dsrc, rows * 4)); HC(hipMalloc(&ddst, rows * 4));
    HC(hipMalloc(&dbounds, (n + 1) * 4)); HC(hipMalloc(&dw, wall.size())); HC(hipMalloc(&dxq, q8_bytes(rows, c.cols)));
    const size_t ny = (size_t) rows * c.rows;
    HC(hipMalloc(&dy, ny * 4));
    HC(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dsrc, src.data(), rows * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(ddst, dst.data(), rows * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dbounds, bounds.data(), (n + 1) * 4, hipMemcpyHostToDevice));
    HC(hipMemcpy(dw, wall.data(), wall.size(), hipMemcpyHostToDevice));
    if (g_race) HC(hipMemset(dy, 0xff, ny * 4));        // null stream; `s` is non-blocking: nothing orders the two
    else HC(hipMemsetAsync(dy, 0xff, ny * 4, s));         // on the compute stream, before the quantizer and the products
    quantize(dx, dsrc, dxq, (int) c.type, c.cols, c.cols, rows, s);
    Product p;
    p.w = dw; p.type = (int) c.type; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = eb; p.n = n; p.xq = dxq;
    p.bounds = dbounds; p.ids = ddst; p.total_rows = rows; p.max_rows = *std::max_element(c.counts.begin(), c.counts.end());
    p.opt_rows = c.opt; p.dst = dy; p.ld_dst = c.rows;
    ctx.run(p, s);
    HC(hipGetLastError());
    HC(hipStreamSynchronize(s));
    std::vector<float> got(ny);
    HC(hipMemcpy(got.data(), dy, ny * 4, hipMemcpyDeviceToHost));

    // reference
    std::vector<float> ref(ny, 0.0f);
    const auto* tr = ggml_get_type_traits(c.type);
    const size_t rb = ggml_row_size(c.type, c.cols);
    std::vector<float> wd((size_t) c.rows * c.cols);
    for (int e = 0; e < n; ++e) {
        for (int64_t o = 0; o < c.rows; ++o) tr->to_float(w[e].data() + (size_t) o * rb, wd.data() + (size_t) o * c.cols, c.cols);
        for (int r = bounds[e]; r < bounds[e + 1]; ++r) {
            const float* xr = x.data() + (size_t) src[r] * c.cols;
            for (int64_t o = 0; o < c.rows; ++o) {
                double acc = 0;
                const float* wr = wd.data() + (size_t) o * c.cols;
                for (int64_t k = 0; k < c.cols; ++k) acc += (double) wr[k] * xr[k];
                ref[(size_t) dst[r] * c.rows + o] = (float) acc;
            }
        }
    }
    // classify
    size_t n_sent = 0, n_nan = 0, n_inf = 0;
    double err2 = 0, ref2 = 0, max_abs = 0;
    std::vector<int> sent_rows(rows, 0), nan_rows(rows, 0);
    int64_t sent_cmin = c.rows, sent_cmax = -1;
    for (size_t i = 0; i < ny; ++i) {
        uint32_t bits; std::memcpy(&bits, &got[i], 4);
        const int row = (int) (i / c.rows); const int64_t col = (int64_t) (i % c.rows);
        if (bits == 0xffffffffu) { ++n_sent; ++sent_rows[row]; sent_cmin = std::min(sent_cmin, col); sent_cmax = std::max(sent_cmax, col); continue; }
        if (std::isnan(got[i])) { ++n_nan; ++nan_rows[row]; continue; }
        if (std::isinf(got[i])) { ++n_inf; continue; }
        const double d = (double) got[i] - ref[i];
        err2 += d * d; ref2 += (double) ref[i] * ref[i]; max_abs = std::max(max_abs, std::fabs(d));
    }
    const size_t n_ok = ny - n_sent - n_nan - n_inf;
    const double rms = n_ok ? std::sqrt(ref2 / (double) n_ok) : 0, rel = rms > 0 ? std::sqrt(err2 / (double) n_ok) / rms : 0;
    const bool pass = n_sent == 0 && n_nan == 0 && n_inf == 0 && rel <= 0.04 && (rms == 0 || max_abs / rms <= 0.35);
    std::printf("%-8s %5lld x %4lld counts=%-12s %s opt=%-3lld | %s | sentinel %zu  nan %zu  inf %zu  rel_l2 %.4f  max/rms %.3f",
                ggml_type_name(c.type), (long long) c.rows, (long long) c.cols, counts_str(c.counts).c_str(),
                c.identity ? "ident" : "perm ", (long long) c.opt, pass ? "PASS" : "FAIL", n_sent, n_nan, n_inf, rel,
                rms > 0 ? max_abs / rms : 0.0);
    if (n_sent) {
        std::printf("\n    unwritten: cols %lld..%lld; per dst row (row: unwritten/of, expert, src):", (long long) sent_cmin, (long long) sent_cmax);
        for (int d = 0; d < rows; ++d) if (sent_rows[d]) {
            const int pos = (int) (std::find(dst.begin(), dst.end(), d) - dst.begin());
            int e = 0; while (bounds[e + 1] <= pos) ++e;
            std::printf(" %d:%d/%lld e%d s%d", d, sent_rows[d], (long long) c.rows, e, src[pos]);
        }
    }
    if (n_nan) {
        std::printf("\n    other NaN per dst row:");
        for (int d = 0; d < rows; ++d) if (nan_rows[d]) std::printf(" %d:%d", d, nan_rows[d]);
    }
    std::printf("\n");
    for (void* q : {(void*) dx, (void*) dy, (void*) dsrc, (void*) ddst, (void*) dbounds, dw, dxq}) (void) hipFree(q);
    return pass;
}
}  // namespace

int main(int argc, char** argv) {
    HC(hipSetDevice(0));
    hipDeviceProp_t prop; HC(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s %s\n", prop.name, prop.gcnArchName);
    hipStream_t s; HC(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "quick") quick = true;
        if (std::string(argv[i]) == "race") g_race = true;
    }
    std::printf("sentinel fill: %s\n", g_race ? "hipMemset on the null stream (the test's order)" : "hipMemsetAsync on the compute stream");
    std::vector<Case> cases = {
        // the test's first product and its variations
        {GGML_TYPE_Q2_0, 1280, 2560, {1, 3, 3}, false, 0}, {GGML_TYPE_Q2_0, 1280, 2560, {1, 3, 3}, true, 0},
        {GGML_TYPE_Q2_0, 1280, 2560, {3}, false, 0},       {GGML_TYPE_Q2_0, 1280, 2560, {1}, true, 0},
        {GGML_TYPE_Q2_0, 1280, 2560, {16}, true, 0},       {GGML_TYPE_Q2_0, 1280, 2560, {64}, true, 0},
        {GGML_TYPE_Q2_0, 1280, 2560, {5, 130}, false, 0},  {GGML_TYPE_Q2_0, 1280, 2560, {5, 130}, false, 5},
        // other types at the test's geometry
        {GGML_TYPE_Q8_0, 1280, 2560, {1, 3, 3}, false, 0},  {GGML_TYPE_IQ4_NL, 1280, 2560, {1, 3, 3}, false, 0},
        {GGML_TYPE_IQ3_XXS, 1280, 2560, {1, 3, 3}, false, 0}, {GGML_TYPE_IQ2_S, 1280, 2560, {1, 3, 3}, false, 0},
        {GGML_TYPE_IQ3_XXS, 1280, 2560, {64}, true, 0},
        // the down shape (640 columns: block-32 types only)
        {GGML_TYPE_Q2_0, 2560, 640, {1, 3, 3}, false, 0},   {GGML_TYPE_Q2_0, 2560, 640, {64}, true, 0},
        {GGML_TYPE_IQ4_NL, 2560, 640, {1, 3, 3}, false, 0}, {GGML_TYPE_Q8_0, 2560, 640, {64}, true, 0},
    };
    if (quick) cases.resize(2);
    Context ctx;
    int fails = 0, trial = 0;
    for (const Case& c : cases) { if (!run(ctx, s, c, trial)) ++fails; ++trial; }
    std::printf("%d of %zu cases failed\n", fails, cases.size());
    return fails ? 1 : 0;
}
