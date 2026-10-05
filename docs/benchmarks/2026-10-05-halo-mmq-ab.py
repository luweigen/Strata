"""docs/TODO.md item 3, the engine-level check: the same prompt through the prompt path's MMQ expert products (the
default) and through the FP16 path (STRATA_PREFILL_MMQ=0), one server start each, temperature 0, the texts printed.
Usage (repository root, the strata conda env's python):
  python docs/benchmarks/2026-10-05-halo-mmq-ab.py [config.json] [max_tokens]
The server: serve/server.py --engine strata --config <config> --port 8080; it inherits this process's environment,
so STRATA_PREFILL_MMQ is set per run.  The prompt: the first ~4,500 characters of serve/server.py (about 1,000
tokens, a prefill chunk).  The server tree is killed between runs (taskkill /T), so the GPU memory is free again.
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "strata-coder-iq1_m.json")
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 64
URL = "http://127.0.0.1:8080"
PROMPT = open(os.path.join(ROOT, "serve", "server.py"), encoding="utf-8").read()[:4500]


def get(path, timeout=5):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def run(label, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    log = open(os.path.join(ROOT, "docs", "benchmarks", f"2026-10-05-halo-mmq-ab-{label}.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata",
                             "--config", CFG, "--port", "8080"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
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
        load_s = time.time() - t0
        body = {"model": "x", "temperature": 0, "max_tokens": MAX_TOKENS, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": "Continue this Python source file where it stops. Output only code.\n\n" + PROMPT}]}
        t1 = time.time()
        r = post("/v1/chat/completions", body)
        wall = time.time() - t1
        msg = r["choices"][0]["message"]
        text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or json.dumps(msg, ensure_ascii=False)
        usage = r.get("usage", {})
        out = {"label": label, "env": env_extra, "server_up_s": round(load_s, 1), "request_s": round(wall, 2),
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "text": text}
        print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
        return text
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=60)
        log.close()
        for _ in range(30):   # until the engine process is gone and its GPU memory with it
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True, text=True).stdout
            if "strata.exe" not in out:
                break
            time.sleep(1)
        time.sleep(2)


if __name__ == "__main__":
    a = run("mmq-default", {})
    b = run("mmq-off", {"STRATA_PREFILL_MMQ": "0"})
    same = 0
    for x, y in zip(a, b):
        if x != y:
            break
        same += 1
    print(json.dumps({"identical": a == b, "common_prefix_chars": same, "len_default": len(a), "len_off": len(b)}), flush=True)
