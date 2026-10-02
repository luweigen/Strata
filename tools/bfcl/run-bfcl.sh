#!/bin/bash
# One BFCL multi-turn run against Strata: generate + evaluate, archived under $BFCL_HOME/runs/<config>-<path>-<think>-
# <cat>-<stamp>[-<label>]/, the same layout and run.json keys as freetoken/bfcl/run-bfcl.ps1 (so its summarize-bfcl.py
# reads both).  Set up the environment first with tools/bfcl/setup-bfcl.sh.
#
#   tools/bfcl/run-bfcl.sh --config strata-gsq --path A --think off --start
#   tools/bfcl/run-bfcl.sh --config strata-gsq --path B --think off --start      # weights only: BFCL's own prompt
#   tools/bfcl/run-bfcl.sh --config strata-iq4xs --think on --start --label r1
#   tools/bfcl/run-bfcl.sh --config strata-gsq --think off --pilot          # the 40 entries of pilot_ids.json
#
#   --config strata-gsq | strata-iq4xs   the model (the same files as the Halo's gsq-hip / flashnext-hip runs)
#   --path A | B                         A (default): native function calling, /v1/chat/completions + tools - the
#                                        server's template and tool parser are part of what is measured.  B: BFCL
#                                        renders the Qwen3 prompt itself and parses the text, through
#                                        /v1/completions - only the weights and the engine are measured
#   --think off | on                     the handler.  A: local-fc-nothink (max_tokens 4096) / local-fc-think
#                                        (16384).  B: local-qwen-b-nothink / local-qwen-b-think (max_tokens 4096)
#   --category LIST                      default multi_turn_base,multi_turn_miss_param (what the Halo ran for these
#                                        two models); long_context needs a 37K+ prompt and is refused at 32K
#   --threads N                          default 1 (Strata serves one request at a time)
#   --pilot                              the 40 pilot entries instead of --category
#   --label TEXT                         appended to the run folder
#   --start                              start the Strata server for this config and stop it at the end
#                                        (without it, a server must already answer on $STRATA_URL)
#   --resume RUN_DIR                     continue a stopped run in its own folder: config, path, think, category
#                                        and pilot come from its run.json (other flags than --start/--threads are
#                                        ignored); entries that ended in an inference error are removed first (BFCL
#                                        keeps them and scores them 0), then BFCL generates only the missing ids
# Environment: BFCL_HOME (default ~/work/AI/bfcl), BFCL_ENV (default bfcl-py312), STRATA_URL (default
# http://127.0.0.1:8080/v1), STRATA_ENV (the conda environment the server runs in, default strata).
set -uo pipefail

STRATA=$(cd "$(dirname "$0")/../.." && pwd)
BFCL_HOME=${BFCL_HOME:-$HOME/work/AI/bfcl}
BFCL_ENV=${BFCL_ENV:-bfcl-py312}
STRATA_URL=${STRATA_URL:-http://127.0.0.1:8080/v1}
STRATA_ENV=${STRATA_ENV:-strata}

config="" path=A think=off category=multi_turn_base,multi_turn_miss_param threads=1 pilot=0 label="" start=0 resume=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) config=$2; shift 2 ;;
    --path) path=$2; shift 2 ;;
    --think) think=$2; shift 2 ;;
    --category) category=$2; shift 2 ;;
    --threads) threads=$2; shift 2 ;;
    --pilot) pilot=1; shift ;;
    --label) label=$2; shift 2 ;;
    --start) start=1; shift ;;
    --resume) resume=${2%/}; shift 2 ;;
    *) echo "unknown option $1 (see the header of $0)"; exit 2 ;;
  esac
done
if [ -n "$resume" ]; then         # the stopped run's own settings
  [ -f "$resume/run.json" ] || { echo "no run.json in $resume"; exit 2; }
  eval "$(python3 - "$resume/run.json" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1], encoding="utf-8-sig"))
for k in ("config", "path", "think", "category"):
    print(f"{k}={shlex.quote(str(m[k]))}")
print(f"pilot={1 if m.get('pilot') else 0}")
EOF
)" || { echo "cannot read $resume/run.json"; exit 2; }
  echo "resuming $resume: --config $config --path $path --think $think --category $category$([ $pilot = 1 ] && echo ' --pilot')"
fi
case "$config" in       # config -> Strata's model config and the model id /v1/models must list
  strata-gsq)   cfg="$STRATA/strata-coder-iq1_m.json";       want=qwen3.8-flash-next-coder-iq1_m ;;
  strata-iq4xs) cfg="$STRATA/strata-unsloth-ud-iq4_xs.json"; want=qwen3.8-flash-next-ud-iq4_xs ;;
  *) echo "--config strata-gsq | strata-iq4xs"; exit 2 ;;
esac
case "$path-$think" in   # the handler: path + thinking (the names run-bfcl.ps1 uses)
  A-off) model=local-fc-nothink ;;
  A-on)  model=local-fc-think ;;
  B-off) model=local-qwen-b-nothink ;;
  B-on)  model=local-qwen-b-think ;;
  *) echo "--path A | B, --think off | on"; exit 2 ;;
esac
TOKENIZER="$BFCL_HOME/qwen3.8-tokenizer"   # path B: BFCL counts the prompt's tokens with it (setup-bfcl.sh copies it)
if [ $pilot = 1 ]; then
  echo "note: the pilot set has 10 long_context entries (up to 37K-token prompts on the Halo); at 32K they can fail"
elif [[ ",$category," == *long_context* || "$category" == multi_turn ]]; then
  echo "long_context needs prompts of 37K+ tokens (Halo runs); this config serves 32K: leave it out of --category"
  exit 2
fi

eval "$(conda shell.bash hook)"
PY="$(conda run -n "$BFCL_ENV" python -c 'import sys; print(sys.executable)')" || { echo "no conda env $BFCL_ENV: run setup-bfcl.sh"; exit 1; }
SPY="$(conda run -n "$STRATA_ENV" python -c 'import sys; print(sys.executable)')" || { echo "no conda env $STRATA_ENV"; exit 1; }
PROJ="$BFCL_HOME/gorilla/berkeley-function-call-leaderboard"
[ -d "$PROJ" ] || { echo "no BFCL checkout in $BFCL_HOME: run setup-bfcl.sh"; exit 1; }
[ "$path" = A ] || [ -f "$TOKENIZER/tokenizer.json" ] || { echo "no tokenizer in $TOKENIZER: run setup-bfcl.sh"; exit 1; }

stamp=$(date +%Y%m%d-%H%M)
if [ -n "$resume" ]; then
  run=$(cd "$resume" && pwd)
else
  cat_tag=${category//,/+}; [ $pilot = 1 ] && cat_tag=pilot
  run="$BFCL_HOME/runs/$config-$path-$think-$cat_tag-$stamp${label:+-$label}"
  mkdir -p "$run"
fi

# the server: started here (--start) or already running
server_pid=""
stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2>/dev/null; wait "$server_pid" 2>/dev/null
  while pgrep -x strata >/dev/null; do sleep 1; done
  echo "server stopped"
}
trap stop_server EXIT
if [ $start = 1 ]; then
  if curl -s -m 3 "$STRATA_URL/models" >/dev/null; then echo "a server already answers on $STRATA_URL: stop it or drop --start"; exit 1; fi
  port=${STRATA_URL##*:}; port=${port%%/*}
  echo "starting Strata: $cfg"
  t0=$(date +%s)
  (cd "$STRATA" && exec "$SPY" serve/server.py --engine strata --config "$cfg" --port "$port") >> "$run/server.log" 2>&1 &
  server_pid=$!
  skip=$(wc -l < "$run/server.log")                 # a resumed run's log already has an earlier start
  until tail -n +$((skip + 1)) "$run/server.log" | grep -q "^ready: http"; do
    kill -0 "$server_pid" 2>/dev/null || { echo "the server exited:"; tail -20 "$run/server.log"; server_pid=""; exit 1; }
    sleep 1
  done
  echo "ready after $(( $(date +%s) - t0 )) s"
fi
served=$(curl -s -m 10 "$STRATA_URL/models") || { echo "$STRATA_URL not answering"; exit 1; }
ids=$(echo "$served" | "$PY" -c 'import sys, json; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))') \
  || { echo "$STRATA_URL/models returned: $served"; exit 1; }
echo "endpoint $STRATA_URL serves: $ids"
[[ ",$ids," == *",$want,"* ]] || { echo "expected $want for --config $config"; exit 1; }

# run.json: the keys run-bfcl.ps1 writes, plus what identifies the Strata side (a resumed run keeps its own and
# records this session under "resumed")
if [ -n "$resume" ]; then
  "$PY" - "$run/run.json" <<EOF
import json, subprocess, sys, datetime
git = lambda d, *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True).stdout.strip()
p = sys.argv[1]; m = json.load(open(p, encoding="utf-8-sig"))
m.setdefault("resumed", []).append({"started": datetime.datetime.now().isoformat(timespec="seconds"),
                                    "served": "$ids".split(","), "strata_commit": git("$STRATA", "rev-parse", "--short", "HEAD")})
json.dump(m, open(p, "w"), indent=1)
EOF
  # entries that ended in an inference error (a crash, a lost connection) are kept by BFCL and scored 0: remove them
  # so they are generated again; the removed lines are kept beside the file
  "$PY" - "$run/result" "$stamp" <<'EOF'
import glob, json, os, sys
for f in glob.glob(os.path.join(sys.argv[1], "**", "*_result.json"), recursive=True):
    lines = [l for l in open(f, encoding="utf-8") if l.strip()]
    bad = [l for l in lines if "Error during inference" in json.dumps(json.loads(l).get("result"))]
    if bad:
        open(f"{f}.removed-{sys.argv[2]}", "w", encoding="utf-8").writelines(bad)
        open(f, "w", encoding="utf-8").writelines(l for l in lines if l not in bad)
    print(f"{os.path.basename(f)}: {len(lines) - len(bad)} kept" + (f", {len(bad)} with an inference error removed" if bad else ""))
EOF
else
"$PY" - "$run/run.json" <<EOF
import json, subprocess, sys, datetime
git = lambda d, *a: subprocess.run(["git", "-C", d, *a], capture_output=True, text=True).stdout.strip()
build = json.load(open("$STRATA/engine/BUILD.json"))
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
json.dump({"config": "$config", "path": "$path", "think": "$think", "category": "$category", "pilot": bool($pilot),
           "threads": $threads, "model": "$model", "endpoint": "$STRATA_URL", "served": "$ids".split(","),
           "bfcl_commit": git("$PROJ", "rev-parse", "--short", "HEAD"), "started": datetime.datetime.now().isoformat(timespec="seconds"),
           "engine": "strata", "strata_commit": git("$STRATA", "rev-parse", "--short", "HEAD"),
           "engine_version": build.get("version"), "strata_config": json.load(open("$cfg")), "gpu": gpu},
          open(sys.argv[1], "w"), indent=1)
EOF
fi

export PYTHONUTF8=1
# `bfcl evaluate` builds the handler too, and the OpenAI client refuses to start without a key: always set one
export OPENAI_BASE_URL="$STRATA_URL" OPENAI_API_KEY=x
extra=()
if [ "$path" = B ]; then
  # the handler's own client (base_oss_handler.py): the endpoint, and a Qwen3.8 tokenizer for counting the prompt;
  # --skip-server-setup: the server is Strata's, BFCL does not start one
  export REMOTE_OPENAI_BASE_URL="$STRATA_URL" REMOTE_OPENAI_API_KEY=x REMOTE_OPENAI_TOKENIZER_PATH="$TOKENIZER"
  extra=(--skip-server-setup)
fi
cd "$PROJ"
if [ $pilot = 1 ]; then cp "$BFCL_HOME/pilot_ids.json" test_case_ids_to_generate.json; sel=(--run-ids)
else sel=(--test-category "$category"); fi

echo "generate -> $run"
overwrite=(--allow-overwrite); [ -n "$resume" ] && overwrite=()   # resumed: BFCL skips the ids it already has
g0=$(date +%s)
"$PY" -m bfcl_eval generate --model "$model" "${sel[@]}" --num-threads "$threads" --include-input-log \
  --result-dir "$run/result" "${overwrite[@]}" "${extra[@]}" 2>&1 | tee -a "$run/generate.log"
g1=$(date +%s)
echo "evaluate"
partial=(); [ $pilot = 1 ] || [ "$category" != multi_turn ] && partial=(--partial-eval)
"$PY" -m bfcl_eval evaluate --model "$model" --test-category multi_turn "${partial[@]}" \
  --result-dir "$run/result" --score-dir "$run/score" 2>&1 | tee -a "$run/evaluate.log"
"$PY" - "$run/run.json" $((g1 - g0)) <<'EOF'
import json, sys, datetime
p = sys.argv[1]; m = json.load(open(p))
m["generate_wall_s"] = m.get("generate_wall_s", 0) + int(sys.argv[2]); m["ended"] = datetime.datetime.now().isoformat(timespec="seconds")
json.dump(m, open(p, "w"), indent=1)
EOF
echo "done: $run  (generate $(( (g1 - g0) / 60 )) min)"
