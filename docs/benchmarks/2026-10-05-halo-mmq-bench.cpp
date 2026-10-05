// The prompt path's expert products through Strata's MMQ glue (src/prefill/moe_mmq.cu -> llama.cpp's mmq.cuh) on
// gfx1151, timed at one layer's shape (docs/TODO.md item 2): 512 experts whose row counts follow a Zipf-like curve
// summing to 41,650 rows (a 4,165-token chunk x 10 experts), in the engine's groups of MMQ_GROUP experts taken in
// expert-id order (the row counts shuffled over the ids, as routing gives them), one `Context::run` per group with
// the group's largest count as max_rows - exactly prefill.cpp's loop minus the gathers.  Products: gate/up
// [1280 x 2560] in IQ3_XXS and IQ2_S (the Coder's layers), down [2560 x 640] in IQ4_NL and Q2_0; UD-IQ4_XS's IQ4_XS
// and IQ3_S too.  Weights are random bytes (the i-quant decoders index codebooks, every byte pattern is valid; the
// values are nonsense, the time is not).  Variants: the group size (16 / 32 / 64) and the experts sorted by row
// count before grouping (so a group's J tile fits all its members).
// Build (repository root; the parity test's flags, see build-hip-win/build.ninja):
//   hipcc --offload-arch=gfx1151 -O3 -std=c++20 -DSTRATA_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1
//         -include include/strata/hip_compat/cuda_runtime.h -Iinclude -Iinclude/strata/hip_compat
//         -Ithird_party/llama.cpp/ggml/include docs/benchmarks/2026-10-05-halo-mmq-bench.cpp
//         build-hip-win/strata_mmq.lib build-hip-win/ggml/src/ggml-base.lib <sdk>/lib/amdhip64.lib -o build-hip-win/mmq_bench.exe
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {
using namespace strata::prefill::mmq;
constexpr int NE = 512, TOPK = 10, TOKENS = 4165, N = 2560, FF = 640;
constexpr size_t TAIL = 4096;   // prefill.cpp's MMQ_TAIL

struct Shape { const char* name; ggml_type type; int64_t rows, cols; };
const Shape SHAPES[] = {
    {"gate/up IQ3_XXS", GGML_TYPE_IQ3_XXS, 2 * FF, N}, {"gate/up IQ2_S", GGML_TYPE_IQ2_S, 2 * FF, N},
    {"gate/up IQ4_XS", GGML_TYPE_IQ4_XS, 2 * FF, N},   {"gate/up IQ3_S", GGML_TYPE_IQ3_S, 2 * FF, N},
    {"down IQ4_NL", GGML_TYPE_IQ4_NL, N, FF},          {"down Q2_0", GGML_TYPE_Q2_0, N, FF},
    {"down IQ4_XS", GGML_TYPE_IQ4_XS, N, FF},          {"down Q8_0", GGML_TYPE_Q8_0, N, FF},
};

// the layer's row count per expert id: Zipf-like over ranks, the ranks shuffled over the ids
std::vector<int> layer_counts(unsigned seed) {
    std::vector<double> z(NE);
    double s = 0;
    for (int i = 0; i < NE; ++i) { z[i] = std::pow(i + 1.0, -0.8); s += z[i]; }
    std::vector<int> ne(NE);
    for (int i = 0; i < NE; ++i) ne[i] = (int) std::lround(z[i] / s * (double) TOKENS * TOPK);
    std::mt19937 rng(seed);
    std::shuffle(ne.begin(), ne.end(), rng);
    return ne;
}

struct Layer {   // the active experts in order and the row offset of each position (rows laid out in that order)
    std::vector<int> order;            // expert ids with rows, in the order the groups take them
    std::vector<int> cnt;              // rows per id
    std::vector<int32_t> pos_off;      // row offset of order[j] (prefix over positions: prefill.cpp's m.off is the
                                       // same thing in id order; a sorted order would lay the rows out sorted)
    int64_t rows = 0;
};

Layer make_layer(unsigned seed, bool sorted) {
    Layer L;
    L.cnt = layer_counts(seed);
    for (int e = 0; e < NE; ++e) if (L.cnt[e] > 0) L.order.push_back(e);
    if (sorted) std::stable_sort(L.order.begin(), L.order.end(), [&](int a, int b) { return L.cnt[a] > L.cnt[b]; });
    L.pos_off.assign(L.order.size() + 1, 0);
    for (size_t j = 0; j < L.order.size(); ++j) L.pos_off[j + 1] = L.pos_off[j] + L.cnt[L.order[j]];
    L.rows = L.pos_off.back();
    return L;
}

float time_ms(hipStream_t s, int reps, const std::function<void()>& f) {
    hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
    f(); HIP_CHECK(hipStreamSynchronize(s));
    HIP_CHECK(hipEventRecord(a, s));
    for (int i = 0; i < reps; ++i) f();
    HIP_CHECK(hipEventRecord(b, s));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a); hipEventDestroy(b);
    return ms / reps;
}
}  // namespace

int main(int argc, char** argv) {
    const int group_arg = argc > 1 ? std::atoi(argv[1]) : 0;   // 0: 16, 32, 64
    hipStream_t s; HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s %s; one layer = %d experts, %d x %d routed rows; ms per layer (3 layers' distributions averaged, 5 reps each)\n",
                prop.name, prop.gcnArchName, NE, TOKENS, TOPK);
    Context ctx;
    const int groups[] = {16, 32, 64};
    for (const Shape& sh : SHAPES) {
        if (!supported((int) sh.type) || !fits((int) sh.type, sh.rows)) { std::printf("%s: not supported / no tile\n", sh.name); continue; }
        const size_t eb = matrix_bytes((int) sh.type, sh.rows, sh.cols);
        const int64_t flop_rows = (int64_t) TOKENS * TOPK;
        const double tops = 2.0 * flop_rows * sh.rows * sh.cols;
        // one group buffer holds up to 64 experts: random bytes, then the zero tail
        std::vector<uint8_t> w((size_t) 64 * eb + TAIL);
        std::mt19937 rng(7); for (size_t i = 0; i < (size_t) 64 * eb; ++i) w[i] = (uint8_t) rng();
        void* dw; HIP_CHECK(hipMalloc(&dw, w.size())); HIP_CHECK(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice));
        // activations: random floats for the layer's rows, quantized once per layer like the engine
        std::vector<float> x((size_t) flop_rows * sh.cols);
        for (auto& v : x) v = (rng() / 4294967296.0f - 0.5f);
        float* dx; HIP_CHECK(hipMalloc(&dx, x.size() * 4)); HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
        void* dxq; HIP_CHECK(hipMalloc(&dxq, q8_bytes(flop_rows, sh.cols)));
        float* dy; HIP_CHECK(hipMalloc(&dy, (size_t) flop_rows * sh.rows * 4));
        int32_t* dids; HIP_CHECK(hipMalloc(&dids, (size_t) flop_rows * 4)); iota(dids, flop_rows, s);
        int32_t* dbounds; HIP_CHECK(hipMalloc(&dbounds, (size_t) (NE + 1 + 64 * 65) * 4));
        const float q_ms = time_ms(s, 5, [&] { quantize(dx, nullptr, dxq, (int) sh.type, sh.cols, sh.cols, flop_rows, s); });
        std::printf("== %s: %.1f MB per expert, quantize %.2f ms\n", sh.name, eb / 1e6, q_ms);
        // opt: the row count the J tile is chosen for (Product::opt_rows): the group's largest (the engine so far),
        // its mean, or its median; the launch grid always covers the largest
        const char* opt_names[] = {"tile for the largest", "tile for the mean   ", "tile for the median "};
        for (int sorted = 0; sorted < 2; ++sorted) {
            for (int G : groups) {
                if (group_arg && G != group_arg) continue;
              for (int opt = 0; opt < 3; ++opt) {
                if (sorted && opt) continue;   // sorted groups are uniform: the choice barely matters
                double sum = 0; int padded_total = 0, launches = 0;
                for (unsigned seed = 1; seed <= 3; ++seed) {
                    const Layer L = make_layer(seed, sorted != 0);
                    // absolute bounds per group (gate/up style: rows of the whole layer), one run per group
                    const size_t n = L.order.size(), ng = (n + G - 1) / G;
                    std::vector<int32_t> bounds;
                    std::vector<int> maxr(ng, 0), optr(ng, 0);
                    for (size_t g = 0; g < ng; ++g) {
                        const size_t j0 = g * G, j1 = std::min(n, j0 + G);
                        std::vector<int> cs;
                        for (size_t j = j0; j < j1; ++j) { bounds.push_back(L.pos_off[j]); maxr[g] = std::max(maxr[g], L.cnt[L.order[j]]); cs.push_back(L.cnt[L.order[j]]); }
                        bounds.push_back(L.pos_off[j1]);   // the group's end
                        std::sort(cs.begin(), cs.end());
                        const int mean = (int) ((std::accumulate(cs.begin(), cs.end(), 0LL) + (int64_t) cs.size() - 1) / (int64_t) cs.size());
                        optr[g] = opt == 0 ? maxr[g] : opt == 1 ? mean : cs[cs.size() / 2];
                        // the J tile llama.cpp picks is at most 128: padding = sum over experts of ceil(ne/J)*J - ne
                        const int J = std::min(128, (optr[g] + 7) / 8 * 8);
                        for (size_t j = j0; j < j1; ++j) padded_total += (L.cnt[L.order[j]] + J - 1) / J * J - L.cnt[L.order[j]];
                    }
                    HIP_CHECK(hipMemcpy(dbounds, bounds.data(), bounds.size() * 4, hipMemcpyHostToDevice));
                    // each group's bounds: n_g + 1 consecutive entries (its experts' row offsets, then the end)
                    const float ms = time_ms(s, 5, [&] {
                        size_t bi = 0;
                        for (size_t g = 0; g < ng; ++g) {
                            const size_t j0 = g * G, ngx = std::min(n, j0 + G) - j0;
                            Product p;
                            p.w = (const uint8_t*) dw + 0; p.type = (int) sh.type; p.w_rows = sh.rows; p.w_cols = sh.cols;
                            p.expert_bytes = eb; p.n = (int) ngx; p.xq = dxq; p.bounds = dbounds + bi; p.ids = dids;
                            p.total_rows = flop_rows; p.max_rows = maxr[g]; p.opt_rows = optr[g]; p.dst = dy; p.ld_dst = sh.rows;
                            ctx.run(p, s);
                            bi += ngx + 1;
                        }
                    });
                    launches = (int) ng;
                    sum += ms;
                }
                const double ms = sum / 3;
                std::printf("   groups of %2d%s, %s: %7.2f ms/layer = %5.1f TOPS, %d launches, J padding %.0f%% of the rows\n", G,
                            sorted ? ", experts sorted by rows" : "                        ", opt_names[opt], ms,
                            tops / (ms * 1e-3) / 1e12, launches, 100.0 * padded_total / 3 / (double) flop_rows);
              }
            }
        }
        hipFree(dw); hipFree(dx); hipFree(dxq); hipFree(dy); hipFree(dids); hipFree(dbounds);
    }
    return 0;
}
