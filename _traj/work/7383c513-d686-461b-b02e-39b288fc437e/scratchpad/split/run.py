"""Per-token decode split: per config and env arm a fresh server, a warm-up, decode x3 (the pcie-frac A/B's prompt)."""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path
S = Path(__file__).parent; ROOT = Path("/home/luwei/work/AI/Strata"); URL = "http://127.0.0.1:8080"
E = Path("/mnt/evox2/large/work/AI/EngramHalo.cpp/src")
L = (E / "llama-model-loader.cpp").read_text()
i = L.index("static std::vector<std::string> llama_get_list_splits"); j = L.index("\n}\n", i) + 3
DECODE = "Rewrite this C++ function to be clearer and faster. Keep the behavior. Reply with the code.\n\n```cpp\n" + L[i:j] + "```"
def chat(text, n):
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps({"model": "x", "temperature": 0, "max_tokens": n,
        "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": text}]}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f: return json.load(f)
RUNS = [("coder", "strata-coder-iq1_m.json"), ("ud", "strata-unsloth-ud-iq4_xs.json")]
ARMS = [("timing", {"STRATA_DECODE_TIMING": "1"}), ("profile", {"STRATA_DECODE_TIMING": "1", "STRATA_VERIFY_PROFILE": "1"})]
for label, base in RUNS:
    for arm, env in ARMS:
        c = json.load(open(ROOT / base)); c["log"] = str(S / f"{label}-{arm}-engine.log")
        cfg = S / f"{label}-{arm}.json"; cfg.write_text(json.dumps(c))
        slog = S / f"{label}-{arm}-server.log"
        p = subprocess.Popen([sys.executable, "serve/server.py", "--engine", "strata", "--config", str(cfg), "--port", "8080"],
                             cwd=ROOT, stdout=open(slog, "w"), stderr=subprocess.STDOUT, env={**os.environ, **env})
        while "ready: http" not in slog.read_text():
            if p.poll() is not None: sys.exit(f"server died: {label} {arm}")
            time.sleep(1)
        args = subprocess.run(["bash", "-c", "ps -o args= -p $(pgrep -x strata) | grep -oE -- '--pcie-frac [0-9.]+'"],
                              capture_output=True, text=True).stdout.strip()
        chat("Write a thread-safe LRU cache in C++.", 128)
        for k in range(3):
            t = chat(DECODE, 512)["timings"]
            print(json.dumps({"model": label, "arm": arm, "args": args, "test": f"decode#{k+1}", "tok_s": t["predicted_per_second"],
                              "tokens": t["predicted_n"], "drafts": f'{t.get("draft_n_accepted")}/{t.get("draft_n")}'}), flush=True)
        p.terminate(); p.wait()
        while subprocess.run(["pgrep", "-x", "strata"], capture_output=True).returncode == 0: time.sleep(1)
