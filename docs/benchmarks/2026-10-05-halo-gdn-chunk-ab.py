"""docs/TODO.md item 5, the engine-level check: pp4096 requests with the old GDN output norm (STRATA_GDN_NORM_OLD=1),
with the new one (the default), and with the chunked WMMA recurrence (STRATA_GDN_CHUNK=1), one server start each, the engine's prompt-path timing on
(STRATA_PREFILL_TIMING=1); then, per mode, one 64-token continuation of a code prompt at temperature 0, so the texts can
be compared.
Usage (repository root, the strata conda env's python):
  python docs/benchmarks/2026-10-05-halo-gdn-chunk-ab.py [config.json] [requests] [max_tokens]
The server: serve/server.py --engine strata --config <a copy of the config with the env added> --port 8080.
The pp4096 prompt: the start of EngramHalo.cpp's llama-model-loader.cpp (STRATA_ENGRAM_SRC), ~4,170 tokens, as pp4k.py
sends it, a different first line per request so the prompt cache does not reuse a prefix.  The text prompt: the first
~4,500 characters of serve/server.py (about 1,000 tokens), as the MMQ A/B used.  The engine log of each run lands in
docs/benchmarks/2026-10-05-halo-gdn-chunk-ab-<label>.log; the script prints each request's timings, the engine's
"prefill timing" lines and the texts.  The server tree is killed between runs (taskkill /T).
"""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CFG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "strata-coder-iq1_m.json"
REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
URL = "http://127.0.0.1:8080"
E = Path(os.environ.get("STRATA_ENGRAM_SRC", "E:/work/AI/EngramHalo.cpp/src"))
TEXT = (E / "llama-model-loader.cpp").read_text(encoding="utf-8")
CODE = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")[:4500]


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def count(t):
    return post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": t}]})["input_tokens"]


def run(label, env_extra):
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg.setdefault("env", {})
    cfg["env"].update(env_extra)
    cfg["env"]["STRATA_PREFILL_TIMING"] = "1"
    engine_log = ROOT / "docs" / "benchmarks" / f"2026-10-05-halo-gdn-chunk-ab-{label}.log"
    cfg["log"] = str(engine_log)
    tmp = Path(tempfile.gettempdir()) / f"strata-gdn-chunk-ab-{label}.json"
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    server_log = open(ROOT / "docs" / "benchmarks" / f"2026-10-05-halo-gdn-chunk-ab-{label}-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(tmp), "--port", "8080"], cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    t0 = time.time()
    texts = []
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
        print(json.dumps({"label": label, "env": env_extra, "server_up_s": round(time.time() - t0, 1)}), flush=True)
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
        body = {"model": "x", "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
                "messages": [{"role": "user", "content": "Continue this Python source file where it stops. Output only code.\n\n" + CODE}]}
        t1 = time.time()
        r = post("/v1/chat/completions", body)
        msg = r["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or json.dumps(msg, ensure_ascii=False)
        t = r.get("timings") or {}
        print(json.dumps({"label": label, "text_request": True, "wall_s": round(time.time() - t1, 2), "prompt_n": t.get("prompt_n"),
                          "prompt_per_second": t.get("prompt_per_second"), "predicted_per_second": t.get("predicted_per_second")}), flush=True)
        print(f"[{label}] text:\n{text}\n", flush=True)
        texts.append(text)
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=60)
        server_log.close()
        for _ in range(30):   # until the engine process is gone and its GPU memory with it
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout.decode(
                "utf-8", errors="replace")   # the console code page is not UTF-8 here
            if "strata.exe" not in out:
                break
            time.sleep(1)
        time.sleep(2)
    for line in engine_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "prefill timing:" in line and "GPU timeline" in line:
            print(f"[{label}] {line}", flush=True)
    return texts


if __name__ == "__main__":
    a = run("norm-old", {"STRATA_GDN_NORM_OLD": "1"})
    b = run("default", {})
    c = run("chunked", {"STRATA_GDN_CHUNK": "1"})
    print("texts: old norm == new norm:", a == b, "| new norm == chunked:", b == c, flush=True)
