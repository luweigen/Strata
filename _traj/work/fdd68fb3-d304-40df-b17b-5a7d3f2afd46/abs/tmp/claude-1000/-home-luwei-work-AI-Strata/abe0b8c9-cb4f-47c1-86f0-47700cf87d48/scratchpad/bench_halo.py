"""Strata on the EngramHalo windows.md tasks: the Coder server sequence and llama-bench-shaped pp/tg at depth 0 and
16384, through /v1/chat/completions (temperature 0, numbers from `timings`).  Usage: bench_halo.py <label> <out.json>"""
import json, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080"
E = Path("/mnt/evox2/large/work/AI/EngramHalo.cpp/src")
LOADER = (E / "llama-model-loader.cpp").read_text()


def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as f:
        return json.load(f)


def count(text):
    return post("/v1/messages/count_tokens", {"model": "x", "messages": [{"role": "user", "content": text}]})["input_tokens"]


def chat(text, max_tokens, label):
    t0 = time.time()
    r = post("/v1/chat/completions", {"model": "x", "temperature": 0, "max_tokens": max_tokens,
                                      "messages": [{"role": "user", "content": text}]})
    t = r.get("timings") or {}
    m = r["choices"][0]["message"]
    row = {"test": label, "wall_s": round(time.time() - t0, 2), "finish": r["choices"][0].get("finish_reason"),
           **{k: t.get(k) for k in ("cache_n", "prompt_n", "prompt_per_second", "predicted_n",
                                    "predicted_per_second", "draft_n", "draft_n_accepted")},
           "answer_tail": ((m.get("content") or "") or (m.get("reasoning_content") or ""))[-300:]}
    if t.get("draft_n"):
        row["accept_pct"] = round(100 * t["draft_n_accepted"] / t["draft_n"], 1)
    print(json.dumps({k: v for k, v in row.items() if k != "answer_tail"}), flush=True)
    return row


def cut(text, tokens):
    """The first part of text with about `tokens` tokens (char ratio from a count, refined once)."""
    n = int(len(text) * tokens / count(text))
    for _ in range(3):
        c = count(text[:n])
        if abs(c - tokens) <= tokens * 0.01:
            break
        n = int(n * tokens / c)
    return text[:n]


def function(name):
    i = LOADER.index(name)
    j = LOADER.index("\n}\n", i) + 3
    return LOADER[LOADER.rfind("\n", 0, i) + 1:j]


REWRITE = "Rewrite this C++ function to be clearer and faster. Keep the behavior. Reply with the code.\n\n```cpp\n%s```"
rows = []
# --- the server sequence (windows.md, "Expert-pruned GSQ-RCO Coder"): 400 tokens out each
p1 = REWRITE % function("static std::vector<std::string> llama_get_list_splits")
p2 = REWRITE % function("static ggml_backend_buffer_type_t select_weight_buft")
rows.append(chat(p1, 400, "server: cold first request (short code prompt)"))
rows.append(chat(p2, 400, "server: fresh code prompt"))
chunk = cut(LOADER, 4700)
rows.append(chat(chunk + "\n\nSummarize what this code does, function by function.", 400,
                 "server: ~4.75K-token prompt, prefill + decode tail"))
rows.append(chat(p1, 400, "server: repeated first prompt (reference only)"))

# --- llama-bench shape: pp4096 / tg128 at depth 0 and 16384, on real code (other EngramHalo sources)
src = "".join(p.read_text() for p in sorted(E.glob("llama-*.cpp")) if p.name != "llama-model-loader.cpp")
pp = cut(src, 4096)
rows.append(chat(pp, 1, "pp4096 @ d0"))
tg_prompt = "Write a complete C++ implementation of a thread-safe LRU cache with comments."
rows.append(chat(tg_prompt, 128, "tg128 @ d0"))
deep = cut(src[len(pp):], 16384)
more = cut(src[len(pp) + len(deep):], 4096)
rows.append(chat(deep, 1, "depth 16384 prefix (cached for the next rows)"))
rows.append(chat(deep + more, 1, "pp4096 @ d16384"))
rows.append(chat(deep + "\n\n" + tg_prompt, 128, "tg128 @ d16384"))
Path(sys.argv[2]).write_text(json.dumps({"label": sys.argv[1], "rows": rows}, indent=1))
