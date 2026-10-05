"""One pp4096-shaped request against a running Strata server (temperature 0, 1 token out): prints the timings line.
Usage: pp4k.py <label> [repeat]   (the prompt: the start of EngramHalo.cpp's llama-model-loader.cpp, ~4K tokens,
the folder from STRATA_ENGRAM_SRC as 2026-10-02-3060m-bench_halo.py takes it)."""
import json, os, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080"
E = Path(os.environ.get("STRATA_ENGRAM_SRC", "/mnt/evox2/large/work/AI/EngramHalo.cpp/src"))
text = (E / "llama-model-loader.cpp").read_text(encoding="utf-8")


def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.load(f)


def count(t):
    return post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": t}]})["input_tokens"]


n = count(text[:16000])
prompt = text[:int(16000 * 4096 / n)]
label = sys.argv[1] if len(sys.argv) > 1 else "pp4k"
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 1):
    # a different first line each time, so the prompt cache does not reuse the previous request's prefix
    body = {"model": "x", "temperature": 0, "max_tokens": 1,
            "messages": [{"role": "user", "content": f"// request {label} {i} {time.time()}\n" + prompt}]}
    t0 = time.time()
    r = post("/v1/chat/completions", body)
    t = r.get("timings") or {}
    print(json.dumps({"label": label, "i": i, "wall_s": round(time.time() - t0, 2), "cache_n": t.get("cache_n"),
                      "prompt_n": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second")}), flush=True)
