"""Decode estimate for other cards in the 3060M PC (2026-10-17): an ESTIMATE, not a measurement.
Inputs are the 3060M's measured verify windows (docs/3060M.md, "Where the decode time goes" and "The Coder's router"):
per layer-window D distinct experts, the CPU's time per expert, the GPU's time per VRAM hit, the GPU's other work G0,
the draft, tokens per window.  Two assumptions carry it:
  - the hit share for more slots: a Zipf law over each layer's experts, its exponent fitted to the measured hit share
    (Coder 2666 slots -> 76.0%, UD-IQ4_XS 1864 -> 58.5%); the adaptive swaps are not modelled;
  - g, the other card's GPU time relative to the 3060M's for the same work (not measured; a range per card).
The GPU's wait for the CPU is the linear fit of the two measured windows: 0.88 x CPU - 0.48 x (GPU expert time), at
least 0.5 ms.  PCIe 3.0 cards: --pcie-frac 0 (the misses all on the CPU).  Usage: python <this file>"""
def H(k, s):
    n = int(k)
    return sum(i ** -s for i in range(1, n + 1)) + (k - n) * (n + 1) ** -s
def hit(k, N, s): return H(min(k, N), s) / H(N, s)
def fit(k, N, h):
    lo, hi = 0.0, 3.0
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if hit(k, N, m) < h else (lo, m)
    return (lo + hi) / 2
# measured on the 3060M; blob = the cache's bytes per slot (MB), cache = its GiB, tpw = tokens committed per window
M = {"Coder IQ1_M": dict(N=256, slots=2666, h=28.6 / 37.6, D=37.6, c=15.8 / (7.8 * 48), hg=11.46 / (28.6 * 48),
                         G0=31.0, draft=5.2, tpw=3.72, blob=2.06, cache=5.11, now=66.4),
     "UD-IQ4_XS": dict(N=512, slots=1864, h=22.3 / 38.1, D=38.1, c=33.0 / (12.2 * 48), hg=9.45 / (22.3 * 48),
                       G0=29.8, draft=5.2, tpw=3.87, blob=2.41, cache=4.19, now=55.4)}
# extra GiB for the expert cache against the 3060M's 12 GB; g range; per-window hand-off of a layer split (ms)
CARDS = [("3060M 12 GB (the model on the measured PC)", 0, (1.0,), 0.0),
         ("Tesla P40 24 GB", 12, (1.2, 1.4, 1.6), 0.0),
         ("RTX 2080 Ti 22 GB", 10, (0.6, 0.75, 0.9), 0.0),
         ("2 x RTX 2080 Ti 22 GB (layer split)", 10 + 15, (0.6, 0.75, 0.9), 0.3)]
def window(d, cache_gib, g, hand):
    slots = min(cache_gib * 1.0737e9 / (d["blob"] * 1e6), d["N"] * 48)
    h = hit(slots / 48, d["N"], d["s"])
    cpu = d["D"] * (1 - h) * 48 * d["c"]
    hits = d["D"] * h * 48 * d["hg"] * g
    wait = max(0.5, 0.88 * cpu - 0.48 * hits) if cpu > 0.05 else 0.5
    w = d["G0"] * g + hits + wait + d["draft"] * g + 0.1 + hand
    return slots, h, cpu, w, 1000 * d["tpw"] / w
for name, d in M.items():
    d["s"] = fit(d["slots"] / 48, d["N"], d["h"])
    print(f"== {name}: Zipf exponent {d['s']:.3f}; measured now {d['now']} tokens/s")
    for card, extra, gs, hand in CARDS:
        for g in gs:
            s, h, cpu, w, t = window(d, d["cache"] + extra, g, hand)
            print(f"  {card:40s} g={g:<4}: slots {s:6.0f} ({s / (d['N'] * 48):4.0%}) hit {h:5.1%} CPU {cpu:5.1f} ms"
                  f"  window {w:5.1f} ms -> {t:4.0f} tokens/s")
