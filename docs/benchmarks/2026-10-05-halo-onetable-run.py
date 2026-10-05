"""The one table's runs: for each config, start the server, time its start to the first /v1/models answer (the table's
"model load to listening"), run 2026-10-02-3060m-bench_halo.py against it, stop the server tree.
Usage (repository root, the strata conda env's python, STRATA_ENGRAM_SRC set):
  python docs/benchmarks/2026-10-05-halo-onetable-run.py <suffix> [config.json:label ...]
Default: strata-coder-iq1_m.json as halo-8060S-coder-iq1_m-arena-<suffix> and strata-unsloth-ud-iq4_xs.json as
halo-8060S-ud-iq4_xs-32-96-cache10000-<suffix>.  Writes docs/benchmarks/2026-10-05-halo-<short>-<suffix>.{json,log}
(the benchmark's rows plus "load_s"; the engine log) and ...-server.log.
"""
import json, os, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
B = ROOT / "docs" / "benchmarks"
SUFFIX = sys.argv[1]
RUNS = [a.split(":", 1) for a in sys.argv[2:]] or [
    ["strata-coder-iq1_m.json", "coder-iq1_m"], ["strata-unsloth-ud-iq4_xs.json", "ud-iq4_xs"]]
LABELS = {"coder-iq1_m": "halo-8060S-coder-iq1_m-arena", "ud-iq4_xs": "halo-8060S-ud-iq4_xs-32-96-cache10000"}


def up():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


for cfg_name, short in RUNS:
    cfg = json.loads((ROOT / cfg_name).read_text(encoding="utf-8"))
    engine_log = B / f"2026-10-05-halo-{short}-{SUFFIX}.log"
    cfg["log"] = str(engine_log)
    tmp = Path(tempfile.gettempdir()) / f"strata-onetable-{short}.json"
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    server_log = open(B / f"2026-10-05-halo-{short}-{SUFFIX}-server.log", "w", encoding="utf-8")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(tmp),
                             "--port", "8080"], cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT)
    try:
        while not up():
            if proc.poll() is not None:
                raise SystemExit(f"{short}: the server exited early ({proc.returncode})")
            if time.time() - t0 > 600:
                raise SystemExit(f"{short}: no answer after 600 s")
            time.sleep(0.2)
        load_s = round(time.time() - t0, 1)
        print(json.dumps({"config": cfg_name, "load_s": load_s}), flush=True)
        out = B / f"2026-10-05-halo-{short}-{SUFFIX}.json"
        subprocess.run([sys.executable, str(B / "2026-10-02-3060m-bench_halo.py"), f"{LABELS.get(short, short)}-{SUFFIX}",
                        str(out)], cwd=ROOT, check=True)
        d = json.loads(out.read_text(encoding="utf-8"))
        d["load_s"] = load_s
        out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait(timeout=60)
        server_log.close()
        for _ in range(30):
            o = subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe"], capture_output=True).stdout.decode("utf-8", "replace")
            if "strata.exe" not in o:
                break
            time.sleep(1)
        time.sleep(3)
