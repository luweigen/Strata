#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>
double probe_host_read_gbps(int threads) {
    constexpr size_t kBytes = 1ull << 30;
    constexpr size_t kWords = kBytes / sizeof(uint64_t);
    threads = std::max(1, threads);
    std::unique_ptr<uint64_t[]> buf(new (std::nothrow) uint64_t[kWords]);
    if (!buf) return -1.0;
    std::atomic<uint64_t> sink{0};
    double best = std::numeric_limits<double>::max();
    for (int pass = 0; pass < 4; ++pass) {          // pass 0 faults the pages in from the threads that read them
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> ts;
        ts.reserve((size_t) threads);
        for (int k = 0; k < threads; ++k) {
            ts.emplace_back([&, k] {
                const size_t lo = kWords * (size_t) k / (size_t) threads, hi = kWords * (size_t) (k + 1) / (size_t) threads;
                uint64_t* p = buf.get();
                if (pass == 0) {
                    std::memset(p + lo, 1, (hi - lo) * sizeof(uint64_t));
                    return;
                }
                uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;   // four chains: the adds never limit the reads
                size_t i = lo;
                for (; i + 4 <= hi; i += 4) {
                    a0 += p[i];
                    a1 += p[i + 1];
                    a2 += p[i + 2];
                    a3 += p[i + 3];
                }
                for (; i < hi; ++i) a0 += p[i];
                sink.fetch_add(a0 + a1 + a2 + a3, std::memory_order_relaxed);
            });
        }
        for (auto& t : ts) t.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (pass > 0) best = std::min(best, s);
    }
    return sink.load() != 0 && best > 0.0 ? (double) kBytes / best / 1e9 : -1.0;
}

// The default PCIe share of the missed experts (plan v0.3 P6), for both CUDA0 and a layer split's other cards.
// The link-only rule (PR #44): 0.55 for native packs from 20 GB/s up, scaled down on slower links, none below 4.
// DRAFT, opt-in with STRATA_PCIE_HOST_RULE=1 (docs/3060M.md, "--pcie-frac"): the link is weighed against the
// host's RAM read bandwidth too.  0.55 was measured on a 6-core machine with an x16 link of ~27 GB/s whose pool
// kernel reads 44.14 GB/s (see `no_host_worker`); a PC whose CPU side is faster relative to its link wants a smaller
// share - measured on an RTX 3060 Laptop GPU + Ryzen 9 8945HX (26.8 GB/s link, ~55 GB/s RAM read): best at 0-0.25,
// 20-25% faster decode than 0.55.  The ratio rule alone gives 0.44 there: the GPU-side cost of the copy kernel is
// not in it yet, so it lowers the share only part of the way.  The rule never raises the link-only value.
double default_pcie_frac(bool native_pack, double link_gbps, double host_gbps) {
    constexpr double kBase = 0.55;
    if (!native_pack) return 0.2;   // the canonical pack's 0.2 was never measured against the link
    if (link_gbps <= 0.0) return kBase;
    double f = link_gbps >= 20.0 ? kBase
             : link_gbps < 4.0  ? 0.0
                                : std::min(kBase, std::max(0.05, kBase * (link_gbps / 26.0)));
    const char* on = std::getenv("STRATA_PCIE_HOST_RULE");
    if (on != nullptr && std::strcmp(on, "1") == 0 && host_gbps > 0.0 && f > 0.0) {
        constexpr double kRefLinkToHost = 27.0 / 44.14;   // the measuring machine's link : pool kernel read
        f = std::min(f, std::max(0.05, kBase * (link_gbps / host_gbps) / kRefLinkToHost));
    }
    return f;
}


int main() {
    double h = probe_host_read_gbps(15);
    std::printf("host read, 15 threads: %.1f GB/s\n", h);
    struct C { double link, host; } cs[] = {{26.8, h}, {27.0, 44.14}, {50.0, 55.0}, {13.0, 55.0}, {2.0, 55.0}};
    for (int on = 0; on < 2; ++on) {
        if (on) setenv("STRATA_PCIE_HOST_RULE", "1", 1); else unsetenv("STRATA_PCIE_HOST_RULE");
        for (auto c : cs) std::printf("  rule %s link %5.1f host %5.1f -> %.2f\n", on ? "on " : "off", c.link, c.host, default_pcie_frac(true, c.link, c.host));
    }
    std::printf("  canonical pack -> %.2f\n", default_pcie_frac(false, 26.8, h));
}
