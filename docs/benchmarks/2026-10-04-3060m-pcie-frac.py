"""--pcie-frac A/B: per value a fresh server, a warm-up, decode x3 (same prompt), prefill 4K and 16K (1 token out).
Usage: pcie_ab2.py <base config> <label> <frac> [<frac> ...]   -> JSON lines on stdout"""
import json, subprocess, sys, time, urllib.request
from pathlib import Path
S = Path(__file__).parent; ROOT = Path("/home/luwei/work/AI/Strata"); URL = "http://127.0.0.1:8080"
base, label, fracs = sys.argv[1], sys.argv[2], sys.argv[3:]
E = Path("/mnt/evox2/large/work/AI/EngramHalo.cpp/src")
L = (E / "llama-model-loader.cpp").read_text()
i = L.index("static std::vector<std::string> llama_get_list_splits"); j = L.index("\n}\n", i) + 3
DECODE = "Rewrite this C++ function to be clearer and faster. Keep the behavior. Reply with the code.\n\n```cpp\n" + L[i:j] + "```"
SRC = "".join(p.read_text() for p in sorted(E.glob("llama-*.cpp")) if p.name != "llama-model-loader.cpp")

def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.load(f)
def chat(text, n):
    return post("/v1/chat/completions", {"model": "x", "temperature": 0, "max_tokens": n,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": text}]})
def count(text):
    return post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": text}]})["input_tokens"]
def cut(text, tokens):
    n = int(len(text) * tokens / count(text[:400000]) * len(text[:400000]) / len(text)) if False else int(tokens * 3.3)
    for _ in range(4):
        c = count(text[:n])
        if abs(c - tokens) <= tokens * 0.01: break
        n = int(n * tokens / c)
    return text[:n]

texts = None
for x in fracs:
    c = json.load(open(ROOT / base)); c["args"] += ["--pcie-frac", x]; c["log"] = str(S / f"ab-{label}-{x}-engine.log")
    cfg = S / f"ab-{label}-{x}.json"; cfg.write_text(json.dumps(c))
    log = open(S / f"ab-{label}-{x}-server.log", "w")
    p = subprocess.Popen([sys.executable, "serve/server.py", "--engine", "strata", "--config", str(cfg), "--port", "8080"],
                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while "ready: http" not in (S / f"ab-{label}-{x}-server.log").read_text():
        if p.poll() is not None: sys.exit(f"server died at {x}")
        time.sleep(1)
    ready = round(time.time() - t0)
    args = subprocess.run(["bash", "-c", "ps -o args= -p $(pgrep -x strata) | grep -oE -- '--pcie-frac [0-9.]+'"],
                          capture_output=True, text=True).stdout.strip()
    if texts is None:                       # the same prefill texts for every value (a fresh server each time)
        a = cut(SRC, 4096); b = cut(SRC[len(a):], 16384); texts = (a, b)
    chat("Write a thread-safe LRU cache in C++.", 128)                       # warm-up
    for k in range(3):
        t = chat(DECODE, 512)["timings"]
        print(json.dumps({"model": label, "pcie_frac": x, "engine_args": args, "test": f"decode#{k+1}",
                          "tok_s": t["predicted_per_second"], "tokens": t["predicted_n"],
                          "drafts": f'{t.get("draft_n_accepted")}/{t.get("draft_n")}', "cache_n": t["cache_n"]}), flush=True)
    for name, text in (("prefill 4K", texts[0]), ("prefill 16K", texts[1])):
        t = chat(text, 1)["timings"]
        print(json.dumps({"model": label, "pcie_frac": x, "test": name, "tok_s": t["prompt_per_second"],
                          "tokens": t["prompt_n"], "cache_n": t["cache_n"], "ready_s": ready}), flush=True)
    p.terminate(); p.wait()
    while subprocess.run(["pgrep", "-x", "strata"], capture_output=True).returncode == 0: time.sleep(1)
