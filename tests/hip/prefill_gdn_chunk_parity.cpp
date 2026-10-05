// tests/hip/prefill_gdn_chunk_parity.cpp - the chunked GDN prefill recurrence (gdn_recurrence_chunked) against the
// token-serial kernels (gdn_recurrence without scratch) and, at the shorter lengths, against a CPU double
// recurrence; then the time of both at the engine's chunk lengths (warm, back to back; STRATA_GDN_CHUNK_TIMING=1 prints
// each kernel's time).  The inputs are shaped like the model's: q and k
// L2-normalised per head, log gates in [-e, -1e-3], betas in (0, 1), a non-zero state carried in.
// Usage: hip_prefill_gdn_chunk_parity [--time-only]
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(call)                                                                                             \
    do {                                                                                                        \
        const cudaError_t e_ = (call);                                                                          \
        if (e_ != cudaSuccess) {                                                                                \
            std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_));        \
            std::exit(2);                                                                                       \
        }                                                                                                       \
    } while (0)

namespace {
constexpr int S = 128, HK = 16, HV = 48, C = 10240, ZV = HV * S;
constexpr float EPS = 1e-6f;

struct Inputs {
    int64_t T;
    std::vector<float> h, gate, beta, z, gamma, state0;
};

Inputs make(int64_t T, uint32_t seed) {
    Inputs in;
    in.T = T;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    in.h.resize((size_t) T * C);
    for (int64_t t = 0; t < T; ++t) {
        float* ht = in.h.data() + t * C;
        for (int i = 0; i < 2 * HK * S; ++i) ht[i] = nd(rng);
        for (int hh = 0; hh < 2 * HK; ++hh) {   // q then k heads: L2-normalised like gdn_conv's output
            float ss = 0.0f;
            for (int i = 0; i < S; ++i) ss += ht[hh * S + i] * ht[hh * S + i];
            const float r = 1.0f / std::sqrt(ss + EPS);   // unit length, as gdn_l2_kernel leaves them
            for (int i = 0; i < S; ++i) ht[hh * S + i] *= r;
        }
        for (int i = 2 * HK * S; i < C; ++i) ht[i] = nd(rng);
    }
    in.gate.resize((size_t) T * HV);
    in.beta.resize((size_t) T * HV);
    for (size_t i = 0; i < in.gate.size(); ++i) {
        in.gate[i] = -std::exp(-7.0f + 8.0f * ud(rng));   // log decay in [-e, -0.0009]
        in.beta[i] = 1.0f / (1.0f + std::exp(-1.5f * nd(rng)));
    }
    in.z.resize((size_t) T * ZV);
    for (auto& v : in.z) v = nd(rng);
    in.gamma.resize(S);
    for (auto& v : in.gamma) v = 1.0f + 0.1f * nd(rng);
    in.state0.resize((size_t) S * HV * S);
    for (auto& v : in.state0) v = 0.1f * nd(rng);
    return in;
}

// the recurrence in double: state[k][head][v], y the normalised output
void reference(const Inputs& in, std::vector<double>& y, std::vector<double>& state) {
    const int64_t T = in.T;
    state.assign(in.state0.begin(), in.state0.end());
    y.assign((size_t) T * ZV, 0.0);
    std::vector<double> kv(S), o(S);
    for (int head = 0; head < HV; ++head) {
        const int qh = head % HK;
        for (int64_t t = 0; t < T; ++t) {
            const float* ht = in.h.data() + t * C;
            const float* q = ht + qh * S;
            const float* k = ht + HK * S + qh * S;
            const float* v = ht + 2 * HK * S + head * S;
            const double g = std::exp((double) in.gate[t * HV + head]), b = in.beta[t * HV + head];
            for (int col = 0; col < S; ++col) {
                double a = 0.0;
                for (int r = 0; r < S; ++r) a += state[((size_t) r * HV + head) * S + col] * k[r];
                kv[col] = a;
            }
            for (int col = 0; col < S; ++col) {
                const double delta = (v[col] - g * kv[col]) * b;
                double oc = 0.0;
                for (int r = 0; r < S; ++r) {
                    double& s = state[((size_t) r * HV + head) * S + col];
                    s = g * s + k[r] * delta;
                    oc += s * q[r];
                }
                o[col] = oc / std::sqrt((double) S);
            }
            double ss = 0.0;
            for (int col = 0; col < S; ++col) ss += o[col] * o[col];
            const double rn = 1.0 / std::sqrt(ss / S + (double) EPS);
            for (int col = 0; col < S; ++col) {
                const double zz = in.z[(size_t) t * ZV + head * S + col];
                y[(size_t) t * ZV + head * S + col] = o[col] * rn * in.gamma[col] / (1.0 + std::exp(-zz));
            }
        }
    }
}

struct Dev {
    float *h = nullptr, *gate = nullptr, *beta = nullptr, *z = nullptr, *gamma = nullptr, *state = nullptr, *y = nullptr,
          *scratch = nullptr;
    uint16_t* y16 = nullptr;
    size_t scratch_floats = 0;
    explicit Dev(const Inputs& in) {
        CHECK(cudaMalloc(&h, in.h.size() * 4));
        CHECK(cudaMalloc(&gate, in.gate.size() * 4));
        CHECK(cudaMalloc(&beta, in.beta.size() * 4));
        CHECK(cudaMalloc(&z, in.z.size() * 4));
        CHECK(cudaMalloc(&gamma, in.gamma.size() * 4));
        CHECK(cudaMalloc(&state, in.state0.size() * 4));
        CHECK(cudaMalloc(&y, (size_t) in.T * ZV * 4));
        CHECK(cudaMalloc(&y16, (size_t) in.T * ZV * 2));
        scratch_floats = (size_t) in.T * C;   // the engine's: m.qkv
        CHECK(cudaMalloc(&scratch, scratch_floats * 4));
        CHECK(cudaMemcpy(gate, in.gate.data(), in.gate.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(beta, in.beta.data(), in.beta.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(z, in.z.data(), in.z.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(gamma, in.gamma.data(), in.gamma.size() * 4, cudaMemcpyHostToDevice));
        reset(in);
    }
    void reset(const Inputs& in) {   // fresh inputs and state before a parity run
        CHECK(cudaMemcpy(h, in.h.data(), in.h.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(state, in.state0.data(), in.state0.size() * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemset(y, 0, (size_t) in.T * ZV * 4));
        CHECK(cudaMemset(y16, 0, (size_t) in.T * ZV * 2));
    }
    ~Dev() {
        for (void* p : {(void*) h, (void*) gate, (void*) beta, (void*) z, (void*) gamma, (void*) state, (void*) y,
                        (void*) y16, (void*) scratch})
            (void) cudaFree(p);
    }
};

struct Out {
    std::vector<float> y, state;
    std::vector<uint16_t> y16;
};
Out fetch(const Dev& d, const Inputs& in) {
    Out o;
    o.y.resize((size_t) in.T * ZV);
    o.state.resize(in.state0.size());
    o.y16.resize((size_t) in.T * ZV);
    CHECK(cudaMemcpy(o.y.data(), d.y, o.y.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(o.state.data(), d.state, o.state.size() * 4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(o.y16.data(), d.y16, o.y16.size() * 2, cudaMemcpyDeviceToHost));
    return o;
}

struct Err { double rel_l2 = 0.0, max_abs = 0.0; bool finite = true; };
template <class A, class B>
Err compare(const A& a, const B& b) {
    Err e;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = (double) a[i], r = (double) b[i];
        if (!std::isfinite(x)) e.finite = false;
        num += (x - r) * (x - r);
        den += r * r;
        e.max_abs = std::max(e.max_abs, std::fabs(x - r));
    }
    e.rel_l2 = std::sqrt(num / std::max(den, 1e-300));
    return e;
}

void run_serial(Dev& d, const Inputs& in) {
    // without scratch gdn_recurrence takes the token-serial kernels whatever STRATA_GDN_CHUNK says
    strata::prefill::gdn_recurrence(d.state, d.h, d.gate, d.beta, d.z, d.gamma, EPS, d.y, d.y16, in.T, nullptr, 0,
                                    nullptr);
}
bool run_chunked(Dev& d, const Inputs& in) {
    return strata::prefill::gdn_recurrence_chunked(d.state, d.h, d.gate, d.beta, d.z, d.gamma, EPS, d.y, d.y16, in.T,
                                                   d.scratch, d.scratch_floats, nullptr);
}

// warm, back to back: the GPU's clock ramps after the host copies, so a single run after a reset reads high.  The
// inputs are only read (the chunked path writes y, state and its scratch), so repeats need no reset; the state drifts,
// which changes no timing.  The mean of `reps` runs after two warm-up runs.
float time_ms(Dev& d, const Inputs& in, bool chunked, int reps) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    d.reset(in);
    for (int r = 0; r < 2; ++r) { if (chunked) run_chunked(d, in); else run_serial(d, in); }
    CHECK(cudaEventRecord(a, nullptr));
    for (int r = 0; r < reps; ++r) { if (chunked) run_chunked(d, in); else run_serial(d, in); }
    CHECK(cudaEventRecord(b, nullptr));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return ms / reps;
}
}  // namespace

int main(int argc, char** argv) {
    const bool time_only = argc > 1 && std::string(argv[1]) == "--time-only";
    int fails = 0;
    {   // the chunked path exists in HIP builds with the MMQ prompt path only
        const Inputs in = make(64, 1);
        Dev d(in);
        if (!run_chunked(d, in)) { std::printf("the chunked GDN path is not built here: skipped\n"); return 77; }
    }
    if (!time_only) {
        const int64_t lengths[] = {1, 7, 63, 64, 65, 130, 333, 1024, 4165};
        for (int64_t T : lengths) {
            const Inputs in = make(T, (uint32_t) (1234 + T));
            Dev d(in);
            run_serial(d, in);
            CHECK(cudaDeviceSynchronize());
            const Out os = fetch(d, in);
            {   // the serial path's output bits, to compare across processes (STRATA_GDN_NORM_OLD=1 against the default)
                uint64_t hsh = 1469598103934665603ull;
                auto mix = [&](const void* p, size_t n) {
                    const auto* b = static_cast<const unsigned char*>(p);
                    for (size_t i = 0; i < n; ++i) { hsh ^= b[i]; hsh *= 1099511628211ull; }
                };
                mix(os.y.data(), os.y.size() * 4);
                mix(os.y16.data(), os.y16.size() * 2);
                mix(os.state.data(), os.state.size() * 4);
                std::printf("T=%lld serial output hash %016llx\n", (long long) T, (unsigned long long) hsh);
            }
            d.reset(in);
            if (!run_chunked(d, in)) {   // the engine's scratch (T x C floats) holds the chunk tables from T = 11 up
                std::printf("T=%lld: the chunked path declined (%s)\n", (long long) T, T < 11 ? "expected below 11" : "FAIL");
                if (T >= 11) ++fails;
                continue;
            }
            CHECK(cudaDeviceSynchronize());
            const Out oc = fetch(d, in);
            const Err ey = compare(oc.y, os.y), es = compare(oc.state, os.state);
            size_t h16 = 0;
            for (size_t i = 0; i < oc.y16.size(); ++i) h16 += oc.y16[i] != os.y16[i];
            std::printf("T=%lld chunked vs serial: y rel_l2 %.2e max_abs %.2e, state rel_l2 %.2e max_abs %.2e, "
                        "fp16 y differing %zu of %zu (%.2f%%)\n", (long long) T, ey.rel_l2, ey.max_abs, es.rel_l2,
                        es.max_abs, h16, oc.y16.size(), 100.0 * h16 / oc.y16.size());
            // the chunked products take FP16 operands: ~4.5e-4 of the output and ~3e-4 of the state, flat in T
            bool ok = ey.finite && es.finite && ey.rel_l2 < 2e-3 && es.rel_l2 < 2e-3;
            if (T <= 333) {
                std::vector<double> ry, rs;
                reference(in, ry, rs);
                const Err sy = compare(os.y, ry), ss = compare(os.state, rs), cy = compare(oc.y, ry), cs = compare(oc.state, rs);
                std::printf("        against double: serial y %.2e state %.2e | chunked y %.2e state %.2e\n", sy.rel_l2,
                            ss.rel_l2, cy.rel_l2, cs.rel_l2);
                ok = ok && sy.rel_l2 < 1e-5 && ss.rel_l2 < 1e-5 && cy.rel_l2 < 2e-3 && cs.rel_l2 < 2e-3;
            }
            if (!ok) { std::printf("        FAIL\n"); ++fails; }
        }
    }
    for (int64_t T : {1024LL, 4165LL, 8192LL}) {
        const Inputs in = make(T, 99);
        Dev d(in);
        const float ms_s = time_ms(d, in, false, 10), ms_c = time_ms(d, in, true, 10);
        std::printf("T=%lld: serial %.1f ms, chunked %.1f ms (%.1fx)\n", (long long) T, ms_s, ms_c, ms_s / ms_c);
    }
    std::printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
