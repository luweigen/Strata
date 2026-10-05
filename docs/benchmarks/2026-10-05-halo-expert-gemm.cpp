// The expert GEMMs' candidates on gfx1151 (docs/TODO.md item 2), at the prompt path's two expert shapes
// (prefill.cpp: gate/up [ne x 2560] x [2560 x 1280] and down [ne x 640] x [640 x 2560], FP16 in, FP32 out) and the
// row counts ne an expert gets in a 4K chunk:
//   blas32   hipblasGemmEx FP16 -> FP32 (the engine's call; with ROCBLAS_USE_HIPBLASLT=1 it is hipBLASLt's heuristic)
//   lt<id>   hipblasLtMatmul with a solution id from tune_hipblaslt (FP32 out), the best of the candidate list
//   blas16   hipblasGemmEx FP16 -> FP16, compute 32F (route a of the TODO: a 16-bit output)
//   wmma     PR #313's strata_wmma_gemm_f16 (route b: 64x64 / 16x16 WMMA tiles, FP32 accumulate)
//   grouped  hipblaslt_ext::GroupedGemm, 16 experts in one launch (FP32 out)
// Timed as 20 back-to-back launches between two events (launch latency amortized), then one layer's worth of
// experts: 512 GEMMs whose ne follow a Zipf-like curve summing to 41,650 rows (4,165 tokens x 10 experts), over 16
// distinct weight sets (one expert's FP16 weights are 9.8 MB; cycling 16 keeps the MALL from holding them all).
// Build (the conda env's hipcc, from the repository root):
//   hipcc --offload-arch=gfx1151 -O2 -DSTRATA_WMMA_GFX11=1 -Isrc/prefill docs/benchmarks/2026-10-05-halo-expert-gemm.cpp
//         src/prefill/wmma_gemm.cu <sdk>/lib/hipblas.lib <sdk>/lib/libhipblaslt.dll.a -o build-hip-win/expert_gemm_probe.exe
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "wmma_gemm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(2); } } while (0)

namespace {
struct Shape { const char* name; int N, K; };
const Shape SHAPES[] = {{"gate/up", 1280, 2560}, {"down", 2560, 640}};
const int TS[] = {16, 32, 48, 64, 96, 128, 192, 256, 384, 512};
const int CANDIDATE_IDS[] = {2537, 2538, 2539, 2549, 2551};   // tune_hipblaslt's winners at these shapes, T=16..512
constexpr int REPS = 20, WSETS = 16, NE_MAX = 1536;
constexpr size_t WS_BYTES = 32u << 20;

hipblasHandle_t blas;
hipblasLtHandle_t lt;
hipStream_t stream;
void* ws;
const float f_one = 1.0f, f_zero = 0.0f;

struct LtDesc {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    LtDesc(int t, int n, int k) {
        const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, k, n, k);
        hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, k, t, k);
        hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, n);
    }
    ~LtDesc() {
        hipblasLtMatrixLayoutDestroy(c); hipblasLtMatrixLayoutDestroy(b); hipblasLtMatrixLayoutDestroy(a);
        hipblasLtMatmulDescDestroy(op);
    }
};

// the engine's lookup: getAlgosFromIndex, then matmulIsAlgoSupported at the actual T
bool lt_algo(int id, int t, int n, int k, hipblasLtMatmulAlgo_t& algo) {
    std::vector<int> ids{id};
    std::vector<hipblasLtMatmulHeuristicResult_t> res;
    if (hipblaslt_ext::getAlgosFromIndex(lt, ids, res) != HIPBLAS_STATUS_SUCCESS || res.empty() ||
        res.front().state != HIPBLAS_STATUS_SUCCESS)
        return false;
    LtDesc d(t, n, k);
    size_t need = 0;
    algo = res.front().algo;
    return hipblaslt_ext::matmulIsAlgoSupported(lt, d.op, &f_one, d.a, d.b, &f_zero, d.c, d.c, algo, need) ==
               HIPBLAS_STATUS_SUCCESS && need <= WS_BYTES;
}

float time_ms(const std::function<void()>& f, int reps) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
    f(); HIP_CHECK(hipStreamSynchronize(stream));   // warm
    HIP_CHECK(hipEventRecord(a, stream));
    for (int i = 0; i < reps; ++i) f();
    HIP_CHECK(hipEventRecord(b, stream));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a); hipEventDestroy(b);
    return ms / reps;
}

struct Bufs {   // X: NE_MAX x K, W[WSETS]: N x K, Y: NE_MAX x N fp32, Y16: fp16
    uint16_t* X; uint16_t* W[WSETS]; float* Y; uint16_t* Y16;
    std::vector<float> hx, hw;   // the fp32 values behind X and W[0]
};
uint16_t h16(float v) { __half h = __float2half_rn(v); uint16_t b; std::memcpy(&b, &h, 2); return b; }
float f16(uint16_t b) { __half h; std::memcpy(&h, &b, 2); return __half2float(h); }

Bufs make(int N, int K) {
    Bufs b;
    std::vector<uint16_t> x((size_t) NE_MAX * K), w((size_t) N * K);
    b.hx.resize(x.size()); b.hw.resize(w.size());
    srand(11);
    for (size_t i = 0; i < x.size(); ++i) { x[i] = h16((rand() / (float) RAND_MAX - 0.5f)); b.hx[i] = f16(x[i]); }
    for (size_t i = 0; i < w.size(); ++i) { w[i] = h16((rand() / (float) RAND_MAX - 0.5f) * 0.1f); b.hw[i] = f16(w[i]); }
    HIP_CHECK(hipMalloc(&b.X, x.size() * 2)); HIP_CHECK(hipMemcpy(b.X, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    for (int s = 0; s < WSETS; ++s) {
        HIP_CHECK(hipMalloc(&b.W[s], w.size() * 2));
        HIP_CHECK(hipMemcpy(b.W[s], w.data(), w.size() * 2, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMalloc(&b.Y, (size_t) NE_MAX * N * 4));
    HIP_CHECK(hipMalloc(&b.Y16, (size_t) NE_MAX * N * 2));
    return b;
}

// max relative error of Y (fp32 or fp16) against a double product of the first `t` rows, sampled
double check(const Bufs& b, int t, int N, int K, bool out16) {
    std::vector<float> y((size_t) t * N);
    if (out16) {
        std::vector<uint16_t> y16((size_t) t * N);
        HIP_CHECK(hipMemcpy(y16.data(), b.Y16, y16.size() * 2, hipMemcpyDeviceToHost));
        for (size_t i = 0; i < y.size(); ++i) y[i] = f16(y16[i]);
    } else {
        HIP_CHECK(hipMemcpy(y.data(), b.Y, y.size() * 4, hipMemcpyDeviceToHost));
    }
    double maxabs = 0, maxdiff = 0;
    for (int r = 0; r < t; r += 3)
        for (int n = 0; n < N; n += 7) {
            double ref = 0;
            for (int k = 0; k < K; ++k) ref += (double) b.hw[(size_t) n * K + k] * b.hx[(size_t) r * K + k];
            const double d = std::fabs(ref - y[(size_t) r * N + n]);
            if (!std::isfinite(y[(size_t) r * N + n])) return INFINITY;
            maxabs = std::max(maxabs, std::fabs(ref)); maxdiff = std::max(maxdiff, d);
        }
    return maxdiff / maxabs;
}

void blas32(const Bufs& b, int ws_i, int t, int N, int K) {
    hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, t, K, &f_one, b.W[ws_i], HIP_R_16F, K, b.X, HIP_R_16F, K,
                  &f_zero, b.Y, HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}
void blas16(const Bufs& b, int ws_i, int t, int N, int K) {
    hipblasGemmEx(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, N, t, K, &f_one, b.W[ws_i], HIP_R_16F, K, b.X, HIP_R_16F, K,
                  &f_zero, b.Y16, HIP_R_16F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
}
void lt_run(const Bufs& b, int ws_i, int t, int N, int K, const hipblasLtMatmulAlgo_t& algo) {
    LtDesc d(t, N, K);   // the engine makes the descriptors per call too
    hipblasLtMatmul(lt, d.op, &f_one, b.W[ws_i], d.a, b.X, d.b, &f_zero, b.Y, d.c, b.Y, d.c, &algo, ws, WS_BYTES, stream);
}

// ne per expert for one layer: Zipf-like (rank^-0.8), scaled to 41,650 rows, capped at NE_MAX, zeros dropped
std::vector<int> layer_counts() {
    std::vector<double> z(512);
    double s = 0;
    for (int i = 0; i < 512; ++i) { z[i] = std::pow(i + 1.0, -0.8); s += z[i]; }
    std::vector<int> ne;
    for (int i = 0; i < 512; ++i) {
        const int c = std::min(NE_MAX, (int) std::lround(z[i] / s * 41650.0));
        if (c > 0) ne.push_back(c);
    }
    return ne;
}
}  // namespace

int main() {
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipblasCreate(&blas); hipblasSetStream(blas, stream);
    hipblasLtCreate(&lt);
    HIP_CHECK(hipMalloc(&ws, WS_BYTES));
    hipblasSetWorkspace(blas, ws, WS_BYTES);
    int ver = 0; hipblasLtGetVersion(lt, &ver);
    hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device %s %s, hipBLASLt %d, ROCBLAS_USE_HIPBLASLT=%s\n", prop.name, prop.gcnArchName, ver,
                std::getenv("ROCBLAS_USE_HIPBLASLT") ? std::getenv("ROCBLAS_USE_HIPBLASLT") : "(unset)");

    for (const Shape& sh : SHAPES) {
        const int N = sh.N, K = sh.K;
        Bufs b = make(N, K);
        std::map<int, int> best_id;   // T -> the best Lt solution id (the layer simulation uses it)
        std::printf("\n== %s: N=%d K=%d, us per GEMM (%d back-to-back) and TFLOP/s; relerr = max relative error vs double\n",
                    sh.name, N, K, REPS);
        std::printf("%5s | %9s | %16s | %9s | %9s | %9s\n", "T", "blas32", "lt<best id>", "blas16", "wmma", "lt/blas32");
        for (int t : TS) {
            const double flop = 2.0 * t * N * K;
            auto tf = [&](float ms) { return flop / (ms * 1e-3) / 1e12; };
            const float m32 = time_ms([&] { blas32(b, 0, t, N, K); }, REPS);
            const double e32 = check(b, t, N, K, false);
            float mlt = INFINITY; int id_best = -1; double elt = 0;
            for (int id : CANDIDATE_IDS) {
                hipblasLtMatmulAlgo_t algo;
                if (!lt_algo(id, t, N, K, algo)) continue;
                const float ms = time_ms([&] { lt_run(b, 0, t, N, K, algo); }, REPS);
                const double e = check(b, t, N, K, false);
                if (e > 1e-4) { std::printf("  (lt %d at T=%d: relerr %.2e, rejected)\n", id, t, e); continue; }
                if (ms < mlt) { mlt = ms; id_best = id; elt = e; }
            }
            if (id_best >= 0) best_id[t] = id_best;
            const float m16 = time_ms([&] { blas16(b, 0, t, N, K); }, REPS);
            const double e16 = check(b, t, N, K, true);
            float mw = INFINITY; double ew = 0;
            if (strata_wmma_gemm_f16(b.X, b.W[0], b.Y, t, N, K, N, 0.0f, stream)) {
                mw = time_ms([&] { strata_wmma_gemm_f16(b.X, b.W[0], b.Y, t, N, K, N, 0.0f, stream); }, REPS);
                ew = check(b, t, N, K, false);
            }
            std::printf("%5d | %5.0f %4.1f | %5.0f %4.1f lt%d | %5.0f %4.1f | %5.0f %4.1f | %.2fx   relerr %.1e / %.1e / %.1e / %.1e\n",
                        t, m32 * 1e3, tf(m32), mlt * 1e3, tf(mlt), id_best, m16 * 1e3, tf(m16), mw * 1e3, tf(mw),
                        m32 / mlt, e32, elt, e16, ew);
        }

        // one layer's experts
        const std::vector<int> ne = layer_counts();
        int64_t rows = 0; for (int c : ne) rows += c;
        std::printf("-- one layer: %zu experts with rows, %lld rows in all (max %d), weights cycling over %d sets; ms per layer, GPU (events) and wall\n",
                    ne.size(), (long long) rows, ne.front(), WSETS);
        auto nearest_id = [&](int t) {
            int best = -1, dist = 1 << 30;
            for (auto& [tt, id] : best_id) if (std::abs(tt - t) < dist) { dist = std::abs(tt - t); best = id; }
            return best;
        };
        // the algos per distinct ne, resolved once (the engine caches them per T as well)
        std::map<int, hipblasLtMatmulAlgo_t> algo_of;
        for (int c : ne) if (!algo_of.count(c)) { hipblasLtMatmulAlgo_t a; if (lt_algo(nearest_id(c), c, N, K, a)) algo_of[c] = a; }
        auto layer = [&](const char* name, const std::function<void(int, int)>& one) {
            hipEvent_t ea, eb; HIP_CHECK(hipEventCreate(&ea)); HIP_CHECK(hipEventCreate(&eb));
            for (size_t j = 0; j < ne.size(); ++j) one((int) (j % WSETS), ne[j]);   // warm
            HIP_CHECK(hipStreamSynchronize(stream));
            const auto w0 = std::chrono::steady_clock::now();
            HIP_CHECK(hipEventRecord(ea, stream));
            for (int rep = 0; rep < 3; ++rep) for (size_t j = 0; j < ne.size(); ++j) one((int) (j % WSETS), ne[j]);
            HIP_CHECK(hipEventRecord(eb, stream));
            HIP_CHECK(hipEventSynchronize(eb));
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() / 3;
            float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, ea, eb)); ms /= 3;
            std::printf("   %-8s %7.2f ms GPU, %7.2f ms wall, %5.1f TFLOP/s\n", name, ms, wall, 2.0 * rows * N * K / (ms * 1e-3) / 1e12);
        };
        layer("blas32", [&](int s, int t) { blas32(b, s, t, N, K); });
        layer("lt", [&](int s, int t) { auto it = algo_of.find(t); if (it != algo_of.end()) lt_run(b, s, t, N, K, it->second); else blas32(b, s, t, N, K); });
        layer("blas16", [&](int s, int t) { blas16(b, s, t, N, K); });
        layer("wmma", [&](int s, int t) { if (!strata_wmma_gemm_f16(b.X, b.W[s], b.Y, t, N, K, N, 0.0f, stream)) blas32(b, s, t, N, K); });

        // grouped: 16 experts per launch, the problem set and heuristic per group (as a layer would need, ne changes)
        {
            hipblaslt_ext::GemmPreference pref; pref.setMaxWorkspaceBytes(WS_BYTES);
            double gpu_ms = 0, wall_ms = 0; bool ok = true;
            for (int rep = 0; rep < 3 && ok; ++rep) {
                hipEvent_t ea, eb; HIP_CHECK(hipEventCreate(&ea)); HIP_CHECK(hipEventCreate(&eb));
                const auto w0 = std::chrono::steady_clock::now();
                HIP_CHECK(hipEventRecord(ea, stream));
                for (size_t j0 = 0; j0 < ne.size() && ok; j0 += 16) {
                    const size_t n = std::min<size_t>(16, ne.size() - j0);
                    std::vector<int64_t> m(n, N), nn(n), k(n, K), bc(n, 1);
                    std::vector<hipblaslt_ext::GemmEpilogue> ep(n);
                    std::vector<hipblaslt_ext::GemmInputs> in(n);
                    for (size_t i = 0; i < n; ++i) {
                        nn[i] = ne[j0 + i];
                        in[i].setA(b.W[(j0 + i) % WSETS]); in[i].setB(b.X); in[i].setC(b.Y); in[i].setD(b.Y);
                        in[i].setAlpha(&f_one); in[i].setBeta(&f_zero);
                    }
                    hipblaslt_ext::GroupedGemm g(lt, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F);
                    std::vector<hipblasLtMatmulHeuristicResult_t> hr;
                    if (g.setProblem(m, nn, k, bc, ep, in) != HIPBLAS_STATUS_SUCCESS ||
                        g.algoGetHeuristic(1, pref, hr) != HIPBLAS_STATUS_SUCCESS || hr.empty() ||
                        g.initialize(hr[0].algo, ws) != HIPBLAS_STATUS_SUCCESS || g.run(stream) != HIPBLAS_STATUS_SUCCESS) {
                        std::printf("   grouped: hipBLASLt grouped GEMM unsupported at group %zu\n", j0 / 16); ok = false;
                    }
                }
                HIP_CHECK(hipEventRecord(eb, stream));
                HIP_CHECK(hipEventSynchronize(eb));
                wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
                float ms = 0; HIP_CHECK(hipEventElapsedTime(&ms, ea, eb)); gpu_ms = ms;
            }
            if (ok) std::printf("   %-8s %7.2f ms GPU, %7.2f ms wall, %5.1f TFLOP/s (groups of 16; setProblem+heuristic+initialize per group on the host)\n",
                                "grouped", gpu_ms, wall_ms, 2.0 * rows * N * K / (gpu_ms * 1e-3) / 1e12);
        }
    }
    return 0;
}
