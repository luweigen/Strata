"""docs/TODO.md item 2a, the engine-level check: a streamed layer's MMQ groups formed within row-count classes
(STRATA_PREFILL_MMQ_CLASSES: "16,32,64,128", "16,32,48,64,96,128", one class per J tile = the shipped default) against one class (=0: groups of 16 in id order, the engine
before), one server start per mode, the engine's prompt-path timing on (STRATA_PREFILL_TIMING=1): pp4096 requests,
then 1,3xx-token requests (a chunk of 1,024 tokens or more streams its experts), then one 64-token continuation of a
code prompt at temperature 0, so the texts can be compared (the products are the same, only their grouping changes).
Usage (repository root, the strata conda env's python):
  python docs/benchmarks/2026-10-06-halo-mmq-classes-ab.py [config.json] [requests] [max_tokens] [exe]
The server: serve/server.py --engine strata --config <a copy of the config with the env added> --port 8080.  The
engine: `exe` (default build-hip-win/strata.exe, the build under test).  The prompts: the start of EngramHalo.cpp's
llama-model-loader.cpp (STRATA_ENGRAM_SRC), ~4,170 and ~1,300 tokens, a different first line per request so the
prompt cache does not reuse a prefix; the text prompt: the first ~4,500 characters of serve/server.py (~1,200
tokens).  Engine logs: docs/benchmarks/<AB_TAG>-<label>.log (AB_TAG defaults to this file's name); AB_ONLY=a,b runs
those modes only.  The first runs (one-class, classes-4 then named classes-default, classes-fine; then
classes-every-tile) used the build whose default was "16,32,64,128"; "shipped" is the build with the final default.
"""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CFG = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "strata-coder-iq1_m.json"
REQUESTS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 64
EXE = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "build-hip-win" / "strata.exe"
URL = "http://127.0.0.1:8080"
E = Path(os.environ.get("STRATA_ENGRAM_SRC", "E:/work/AI/EngramHalo.cpp/src"))
TEXT = (E / "llama-model-loader.cpp").read_text(encoding="utf-8")
CODE = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")[:4500]
TAG = os.environ.get("AB_TAG", "2026-10-06-halo-mmq-classes-ab")


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def count(t):
    return post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": t}]})["input_tokens"]


def prompt_requests(label, prompt, kind):
    for i in range(REQUESTS):
        body = {"model": "x", "temperature": 0, "max_tokens": 1,
                "messages": [{"role": "user", "content": f"// request {label} {kind} {i} {time.time()}\n" + prompt}]}
        t1 = time.time()
        r = post("/v1/chat/completions", body)
        t = r.get("timings") or {}
        print(json.dumps({"label": label, "kind": kind, "i": i, "wall_s": round(time.time() - t1, 2),
                          "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
                          "prompt_per_second": t.get("prompt_per_second")}), flush=True)


def run(label, env_extra):
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    cfg["exe"] = str(EXE)
    cfg.setdefault("env", {})
    cfg["env"].update(env_extra)
    cfg["env"]["STRATA_PREFILL_TIMING"] = "1"
    engine_log = ROOT / "docs" / "benchmarks" / f"{TAG}-{label}.log"
    cfg["log"] = str(engine_log)
    tmp = Path(tempfile.gettempdir()) / f"strata-mmq-classes-ab-{label}.json"
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    server_log = open(ROOT / "docs" / "benchmarks" / f"{TAG}-{label}-server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                             "--config", str(tmp), "--port", "8080"], cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    t0 = time.time()
    text = None
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
        prompt_requests(label, TEXT[:int(16000 * 4096 / n)], "pp4096")
        prompt_requests(label, TEXT[:int(16000 * 1300 / n)], "pp1300")
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
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=60)
        server_log.close()
        for _ in range(30):   # until the engine process is gone and its GPU memory with it
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout.decode(
                "utf-8", errors="replace")
            if "strata.exe" not in out:
                break
            time.sleep(1)
        time.sleep(2)
    for line in engine_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "prefill timing:" in line and ("GPU timeline" in line or "experts:" in line):
            print(f"[{label}] {line}", flush=True)
    return text


if __name__ == "__main__":
    modes = [("one-class", {"STRATA_PREFILL_MMQ_CLASSES": "0"}), ("classes-4", {"STRATA_PREFILL_MMQ_CLASSES": "16,32,64,128"}),
             ("classes-fine", {"STRATA_PREFILL_MMQ_CLASSES": "16,32,48,64,96,128"}),
             ("classes-every-tile", {"STRATA_PREFILL_MMQ_CLASSES": "16,32,48,64,80,96,112,128"}),
             ("shipped", {})]
    only = os.environ.get("AB_ONLY")
    texts = {label: run(label, env) for label, env in modes if not only or label in only.split(",")}
    labels = list(texts)
    for a in labels[1:]:
        print(f"texts: {labels[0]} == {a}:", texts[labels[0]] == texts[a], flush=True)
