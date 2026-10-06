// TODO 2b: the MMQ kernels' own speed (written for gfx1151), per J tile, for A/B-ing the RDNA 3.5 tile table
// (third_party/llama.cpp/ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh).  Two measurements per expert type:
//  1. "uniform": 512 experts of exactly J rows each (no padding), groups of 16, the tile chosen for J: the kernel's
//     own throughput at each tile the engine's classes use (16..128), and at 256 rows (two 128 tiles per expert).
//  2. "layer": one layer as the engine runs it since TODO 2a: 512 experts with Zipf-like row counts summing to
//     41,650 (a 4,165-token chunk x 10), laid out class-major by row count with the classes 16,32,...,128, groups of
//     16 within a class, the tile for the group's median; ms per layer and per class.
// The same products as 2026-10-05-halo-mmq-bench.cpp (random weight bytes: the time is real, the values are not).
// This copy: the CUDA port of 2026-10-06-halo-mmq-tiles.cpp (branch AIMAX395-ROCm) for the RTX 3060 Laptop GPU
// (sm_86): the hip* calls renamed to cuda*, the device line prints sm_XY instead of gcnArchName, and no
// Product::opt_rows (this branch picks the tile for the group's max_rows, so "layer" uses the max, not the median;
// "uniform" is unchanged: there max = median).
// Build (repository root, Linux), after `cmake --build build --target strata_mmq`:
//   nvcc -O3 -std=c++20 -arch=sm_86 -Iinclude -Ithird_party/llama.cpp/ggml/include
//       docs/benchmarks/2026-10-06-3060m-mmq-tiles.cu build/libstrata_mmq.a build/ggml/src/libggml-base.a
//       -lcuda -lcublas -o build/mmq_tiles
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); std::exit(2); } } while (0)

namespace {
using namespace strata::prefill::mmq;
constexpr int NE = 512, TOPK = 10, TOKENS = 4165, N = 2560, FF = 640, G = 16;
constexpr size_t TAIL = 4096;
const int CLASSES[] = {16, 32, 48, 64, 80, 96, 112, 128};   // the engine's default; the last class takes the rest
constexpr int NCLS = 9;

struct Shape { const char* name; ggml_type type; int64_t rows, cols; };
const Shape SHAPES[] = {
    {"gate/up IQ3_XXS", GGML_TYPE_IQ3_XXS, 2 * FF, N}, {"gate/up IQ2_S", GGML_TYPE_IQ2_S, 2 * FF, N},
    {"gate/up IQ4_XS", GGML_TYPE_IQ4_XS, 2 * FF, N},   {"gate/up IQ3_S", GGML_TYPE_IQ3_S, 2 * FF, N},
    {"down IQ4_NL", GGML_TYPE_IQ4_NL, N, FF},          {"down Q2_0", GGML_TYPE_Q2_0, N, FF},
    {"down IQ4_XS", GGML_TYPE_IQ4_XS, N, FF},          {"down Q8_0", GGML_TYPE_Q8_0, N, FF},
};

int class_of(int rows) {
    for (int c = 0; c < NCLS - 1; ++c) if (rows <= CLASSES[c]) return c;
    return NCLS - 1;
}

std::vector<int> zipf_counts(unsigned seed) {
    std::vector<double> z(NE);
    double s = 0;
    for (int i = 0; i < NE; ++i) { z[i] = std::pow(i + 1.0, -0.8); s += z[i]; }
    std::vector<int> ne(NE);
    for (int i = 0; i < NE; ++i) ne[i] = (int) std::lround(z[i] / s * (double) TOKENS * TOPK);
    std::mt19937 rng(seed);
    std::shuffle(ne.begin(), ne.end(), rng);
    return ne;
}

struct Group { int n; int maxr, optr; size_t bi; int cls; int64_t rows; };

float time_ms(cudaStream_t s, int reps, const std::function<void()>& f) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    f(); f(); CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < reps; ++i) f();
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}
}  // namespace

int main(int argc, char** argv) {
    const char* only = argc > 1 ? argv[1] : nullptr;   // a substring of a shape's name
    cudaStream_t s; CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s sm_%d%d; groups of %d; uniform: 512 experts x J rows; layer: Zipf rows (3 seeds), classes 16..128\n",
                prop.name, prop.major, prop.minor, G);
    Context ctx;
    const int UJ[] = {16, 32, 48, 64, 80, 96, 112, 128, 256};
    const int64_t max_rows_total = (int64_t) NE * 256;
    for (const Shape& sh : SHAPES) {
        if (only && !std::strstr(sh.name, only)) continue;
        if (!supported((int) sh.type) || !fits((int) sh.type, sh.rows)) { std::printf("%s: not supported / no tile\n", sh.name); continue; }
        const size_t eb = matrix_bytes((int) sh.type, sh.rows, sh.cols);
        std::vector<uint8_t> w((size_t) G * eb + TAIL, 0);
        std::mt19937 rng(7); for (size_t i = 0; i < (size_t) G * eb; ++i) w[i] = (uint8_t) rng();
        void* dw; CUDA_CHECK(cudaMalloc(&dw, w.size())); CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
        // 4,096 random rows, repeated over the whole buffer
        std::vector<float> x((size_t) 4096 * sh.cols);
        for (auto& v : x) v = (rng() / 4294967296.0f - 0.5f);
        float* dx; CUDA_CHECK(cudaMalloc(&dx, (size_t) max_rows_total * sh.cols * 4));
        for (int64_t r = 0; r < max_rows_total; r += 4096)
            CUDA_CHECK(cudaMemcpy(dx + r * sh.cols, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
        void* dxq; CUDA_CHECK(cudaMalloc(&dxq, q8_bytes(max_rows_total, sh.cols)));
        quantize(dx, nullptr, dxq, (int) sh.type, sh.cols, sh.cols, max_rows_total, s);
        float* dy; CUDA_CHECK(cudaMalloc(&dy, (size_t) max_rows_total * sh.rows * 4));
        int32_t* dids; CUDA_CHECK(cudaMalloc(&dids, (size_t) max_rows_total * 4)); iota(dids, max_rows_total, s);
        int32_t* dbounds; CUDA_CHECK(cudaMalloc(&dbounds, (size_t) (NE + 1 + 64 * 65) * 4));
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::printf("== %s (%.1f MB per expert)\n", sh.name, eb / 1e6);

        auto run_groups = [&](const std::vector<Group>& gs, int64_t total, int cls) {
            for (const Group& g : gs) {
                if (cls >= 0 && g.cls != cls) continue;
                Product p;
                p.w = dw; p.type = (int) sh.type; p.w_rows = sh.rows; p.w_cols = sh.cols; p.expert_bytes = eb;
                p.n = g.n; p.xq = dxq; p.bounds = dbounds + g.bi; p.ids = dids; p.total_rows = total;
                p.max_rows = g.maxr; p.dst = dy; p.ld_dst = sh.rows;
                ctx.run(p, s);
            }
        };
        // 1. uniform experts of J rows
        std::printf("   uniform, TOPS at J =");
        for (int J : UJ) std::printf(" %5d", J);
        std::printf("\n                       ");
        for (int J : UJ) {
            std::vector<int32_t> bounds; std::vector<Group> gs;
            for (int g0 = 0; g0 < NE; g0 += G) {
                gs.push_back({G, J, J, bounds.size(), 0, (int64_t) G * J});
                for (int j = 0; j <= G; ++j) bounds.push_back((g0 + j) * J);
            }
            CUDA_CHECK(cudaMemcpy(dbounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice));
            const int64_t total = (int64_t) NE * J;
            const float ms = time_ms(s, 5, [&] { run_groups(gs, total, -1); });
            std::printf(" %5.1f", 2.0 * total * sh.rows * sh.cols / (ms * 1e-3) / 1e12);
        }
        std::printf("\n");
        // 2. the layer, class-major
        double cls_ms[NCLS] = {}, layer_ms = 0, useful = 0;
        int64_t cls_rows[NCLS] = {};
        for (unsigned seed = 1; seed <= 3; ++seed) {
            const std::vector<int> cnt = zipf_counts(seed);
            std::vector<int> order;
            for (int c = 0; c < NCLS; ++c)
                for (int e = 0; e < NE; ++e) if (cnt[e] > 0 && class_of(cnt[e]) == c) order.push_back(e);
            std::vector<int32_t> off(order.size() + 1, 0);
            for (size_t j = 0; j < order.size(); ++j) off[j + 1] = off[j] + cnt[order[j]];
            const int64_t total = off.back();
            useful += 2.0 * total * sh.rows * sh.cols;
            std::vector<int32_t> bounds; std::vector<Group> gs;
            for (size_t j0 = 0; j0 < order.size();) {
                const int c = class_of(cnt[order[j0]]);
                size_t j1 = j0;
                while (j1 < order.size() && j1 - j0 < (size_t) G && class_of(cnt[order[j1]]) == c) ++j1;
                std::vector<int> cs;
                for (size_t j = j0; j < j1; ++j) cs.push_back(cnt[order[j]]);
                std::sort(cs.begin(), cs.end());
                gs.push_back({(int) (j1 - j0), cs.back(), cs[cs.size() / 2], bounds.size(), c, off[j1] - off[j0]});
                for (size_t j = j0; j <= j1; ++j) bounds.push_back(off[j]);
                cls_rows[c] += off[j1] - off[j0];
                j0 = j1;
            }
            CUDA_CHECK(cudaMemcpy(dbounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice));
            layer_ms += time_ms(s, 5, [&] { run_groups(gs, total, -1); });
            for (int c = 0; c < NCLS; ++c) cls_ms[c] += time_ms(s, 5, [&] { run_groups(gs, total, c); });
        }
        std::printf("   layer: %7.2f ms = %5.1f TOPS; per class (<=16 .. >128) ms [rows]:", layer_ms / 3, useful / (layer_ms * 1e-3) / 1e12);
        for (int c = 0; c < NCLS; ++c) std::printf(" %.2f[%lld]", cls_ms[c] / 3, (long long) (cls_rows[c] / 3));
        std::printf("\n");
        cudaFree(dw); cudaFree(dx); cudaFree(dxq); cudaFree(dy); cudaFree(dids); cudaFree(dbounds);
    }
    return 0;
}
