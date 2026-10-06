"""TODO 2b: the one table's Coder rows, TODO 2a's engine ("old") and TODO 2b's ("new") alternating, from
2026-10-06-halo-onetable-todo2b-coder-pairs.log (each config line starts a run; runs alternate old, new).  Per row:
the mean and sample SD of each engine, the mean of the paired differences new - old (pair k = the k-th old run and
the k-th new run, back to back), its 95% interval (Student t, n - 1 df) and how many pairs went the same way.
Usage (repository root, the strata conda env's python): python docs/benchmarks/2026-10-06-halo-onetable-todo2b-pairs-stats.py
"""
import json, math, statistics as st
from pathlib import Path

LOG = Path(__file__).resolve().parent / "2026-10-06-halo-onetable-todo2b-coder-pairs.log"
# two-sided 95% Student t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

runs, cur = [], None
for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.startswith("{"):
        continue
    d = json.loads(line)
    if "config" in d:
        cur = {"engine": "old" if "todo2a" in d["config"] else "new", "rows": {}}
        runs.append(cur)
    elif cur is not None:
        cur["rows"][d["test"]] = d
old = [r for r in runs if r["engine"] == "old" and len(r["rows"]) == 9]
new = [r for r in runs if r["engine"] == "new" and len(r["rows"]) == 9]
n = min(len(old), len(new))
print(f"{n} pairs (old = TODO 2a's engine, new = TODO 2b's)\n")
ROWS = [("server: ~4.75K-token prompt, prefill + decode tail", "prompt_per_second", "prefill @ ~4.75K"),
        ("pp4096 @ d0", "prompt_per_second", "pp4096 @ d0"),
        ("depth 16384 prefix (cached for the next rows)", "prompt_per_second", "16,384 prefix"),
        ("pp4096 @ d16384", "prompt_per_second", "pp4096 @ d16384"),
        ("server: cold first request (short code prompt)", "predicted_per_second", "decode: cold first request"),
        ("server: fresh code prompt", "predicted_per_second", "decode: fresh code prompt"),
        ("server: ~4.75K-token prompt, prefill + decode tail", "predicted_per_second", "decode: tail after 4.75K"),
        ("tg128 @ d0", "predicted_per_second", "tg128 @ d0"),
        ("tg128 @ d16384", "predicted_per_second", "tg128 @ d16384")]
print(f"{'row (t/s)':28s} {'old mean+-SD':>16s} {'new mean+-SD':>16s} {'new-old, 95% CI':>26s} {'%':>6s} {'new>old':>8s}")
for test, key, name in ROWS:
    a = [r["rows"][test][key] for r in old[:n]]
    b = [r["rows"][test][key] for r in new[:n]]
    dif = [y - x for x, y in zip(a, b)]
    md, sd = st.mean(dif), st.stdev(dif)
    h = T95[n - 1] * sd / math.sqrt(n)
    print(f"{name:28s} {st.mean(a):8.1f} +-{st.stdev(a):5.1f} {st.mean(b):8.1f} +-{st.stdev(b):5.1f} "
          f"{md:+7.1f} [{md - h:+6.1f}, {md + h:+6.1f}] {100 * md / st.mean(a):+5.1f}% {sum(x > 0 for x in dif):>4d}/{n}")
