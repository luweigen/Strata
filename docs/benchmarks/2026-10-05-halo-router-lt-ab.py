"""docs/TODO.md item 4, the engine-level check: pp4096 requests through the hipBLASLt table before and after the rows for
the router projection (ffn_gate_inp.weight, bf16 N=256 K=2560, 256 experts, one GEMM per layer), one server start per table, the engine's prompt-path
timing on (STRATA_PREFILL_TIMING=1) and its Lt lookups logged (STRATA_HIPBLASLT_VERBOSE=1).
Usage (repository root, the strata conda env's python):
  python docs/benchmarks/2026-10-05-halo-router-lt-ab.py <before-table> <after-table> [config.json] [requests]
The server: serve/server.py --engine strata --config <a copy of the config with the table swapped in> --port 8080.
The prompt: the start of EngramHalo.cpp's llama-model-loader.cpp (STRATA_ENGRAM_SRC), ~4,170 tokens, as pp4k.py sends it,
a different first line per request so the prompt cache does not reuse a prefix.  The engine log of each run lands in
docs/benchmarks/2026-10-05-halo-router-lt-ab-<label>.log; the script prints each request's timings and the engine's
"prefill timing", "N=256" and "hipBLASLt summary" lines.  The server tree is killed between runs (taskkill /T).
"""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BEFORE, AFTER = sys.argv[1], sys.argv[2]
CFG = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "strata-coder-iq1_m.json"
REQUESTS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
URL = "http://127.0.0.1:8080"
E = Path(os.environ.get("STRATA_ENGRAM_SRC", "E:/work/AI/EngramHalo.cpp/src"))
TEXT = (E / "llama-model-loader.cpp").read_text(encoding="utf-8")


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def count(t):
    return post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": t}]})["input_tokens"]


def run(label, table):
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg.setdefault("env", {})
    cfg["env"]["STRATA_HIPBLASLT_TUNING"] = table
    cfg["env"]["STRATA_HIPBLASLT_VERBOSE"] = "1"
    cfg["env"]["STRATA_PREFILL_TIMING"] = "1"
    engine_log = ROOT / "docs" / "benchmarks" / f"2026-10-05-halo-router-lt-ab-{label}.log"
    cfg["log"] = str(engine_log)
    tmp = Path(tempfile.gettempdir()) / f"strata-router-lt-ab-{label}.json"
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    server_log = open(ROOT / "docs" / "benchmarks" / f"2026-10-05-halo-router-lt-ab-{label}-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(tmp), "--port", "8080"], cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        while True:
            try:
                get("/v1/models")
                break
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit(f"{label}: the server exited early ({proc.returncode}); see its log")
                if time.time() - t0 > 300:
                    raise SystemExit(f"{label}: no answer from the server after 300 s")
                time.sleep(1)
        print(json.dumps({"label": label, "table": table, "server_up_s": round(time.time() - t0, 1)}), flush=True)
        n = count(TEXT[:16000])
        prompt = TEXT[:int(16000 * 4096 / n)]
        for i in range(REQUESTS):
            body = {"model": "x", "temperature": 0, "max_tokens": 1,
                    "messages": [{"role": "user", "content": f"// request {label} {i} {time.time()}\n" + prompt}]}
            t1 = time.time()
            r = post("/v1/chat/completions", body)
            t = r.get("timings") or {}
            print(json.dumps({"label": label, "i": i, "wall_s": round(time.time() - t1, 2), "cache_n": t.get("cache_n"),
                              "prompt_n": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second")}), flush=True)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=60)
        server_log.close()
        for _ in range(30):   # until the engine process is gone and its GPU memory with it
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True, text=True).stdout
            if "strata.exe" not in out:
                break
            time.sleep(1)
        time.sleep(2)
    for line in engine_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "prefill timing:" in line or "N=256 " in line or "hipBLASLt summary" in line or "fallback shape" in line \
                or "hipBLASLt tuning enabled" in line:
            print(f"[{label}] {line}", flush=True)


if __name__ == "__main__":
    run("before", BEFORE)
    run("after", AFTER)
