#!/bin/bash
# The BFCL environment on Linux, the same as the Windows one in freetoken/bfcl (README.md there, "BFCL multi-turn
# 测试计划"): gorilla at the same commit with the same local handlers, in its own conda environment (not `strata`:
# BFCL's dependencies stay out of Strata's).  Run it once; every step is skipped when it is already done.
#
#   tools/bfcl/setup-bfcl.sh
#
# Settings (environment variables):
#   WIN_BFCL   the Windows bfcl folder, source of the handler changes, pilot_ids.json and summarize-bfcl.py
#              (default /mnt/evox2/large/work/AI/freetoken/bfcl)
#   BFCL_HOME  where the checkout and the runs go (default ~/work/AI/bfcl)
#   BFCL_ENV   the conda environment (default bfcl-py312, as on Windows)
set -euo pipefail

WIN_BFCL=${WIN_BFCL:-/mnt/evox2/large/work/AI/freetoken/bfcl}
BFCL_HOME=${BFCL_HOME:-$HOME/work/AI/bfcl}
BFCL_ENV=${BFCL_ENV:-bfcl-py312}
COMMIT=6ea5797                       # the Windows checkout's HEAD (run.json "bfcl_commit" of every Halo run)
LEADERBOARD=berkeley-function-call-leaderboard
# the handlers added on Windows (untracked there, so not in `git diff`)
NEW_FILES=("$LEADERBOARD/bfcl_eval/model_handler/api_inference/local_openai_compat.py"
           "$LEADERBOARD/bfcl_eval/model_handler/local_inference/qwen_fc_nothink.py")

say() { echo "== $*"; }
[ -d "$WIN_BFCL/gorilla/.git" ] || { echo "no gorilla checkout in $WIN_BFCL (set WIN_BFCL)"; exit 1; }
mkdir -p "$BFCL_HOME/runs"

# 1. the conda environment
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$BFCL_ENV"; then
  say "conda environment $BFCL_ENV exists"
else
  say "creating conda environment $BFCL_ENV (Python 3.12)"
  conda create -y -n "$BFCL_ENV" python=3.12
fi
PY="$(conda run -n "$BFCL_ENV" python -c 'import sys; print(sys.executable)')"

# 2. gorilla at the pinned commit
G="$BFCL_HOME/gorilla"
if [ ! -d "$G/.git" ]; then
  say "cloning gorilla"
  git clone https://github.com/ShishirPatil/gorilla.git "$G"
fi
head=$(git -C "$G" rev-parse --short HEAD)
if [ "$head" != "$COMMIT" ]; then
  [ -z "$(git -C "$G" status --porcelain)" ] || { echo "$G has local changes and is not at $COMMIT: fix it by hand"; exit 1; }
  say "checking out $COMMIT"
  git -C "$G" checkout -q "$COMMIT"
fi

# 3. the Windows handler changes: the diff of the tracked files, plus the two new files
PATCH="$BFCL_HOME/local-handlers.patch"
git -C "$WIN_BFCL/gorilla" diff > "$PATCH"
if git -C "$G" apply --reverse --check "$PATCH" 2>/dev/null; then
  say "handler changes already applied"
else
  say "applying the handler changes ($(grep -c '^diff --git' "$PATCH") files)"
  git -C "$G" apply "$PATCH"
fi
for f in "${NEW_FILES[@]}"; do
  cp "$WIN_BFCL/gorilla/$f" "$G/$f"
done
sed -i 's/\r$//' "${NEW_FILES[@]/#/$G/}"     # a Windows checkout may carry CRLF

# 4. bfcl-eval, editable (so the handler changes are what runs), and soundfile (qwen_agent needs it; the base
#    install misses it, the same as on Windows)
if "$PY" -c "import bfcl_eval, soundfile" 2>/dev/null; then
  say "bfcl-eval installed"
else
  say "installing bfcl-eval (editable) and soundfile"
  "$PY" -m pip install -e "$G/$LEADERBOARD" soundfile
fi

# 5. the Windows tools that are not part of gorilla
cp "$WIN_BFCL/pilot_ids.json" "$WIN_BFCL/summarize-bfcl.py" "$BFCL_HOME/"

# 6. check: the local handlers are registered
"$PY" - <<'EOF'
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING as M
need = ["local-fc-nothink", "local-fc-think", "local-qwen-b-nothink", "local-qwen-b-think"]
missing = [n for n in need if n not in M]
assert not missing, f"not registered: {missing}"
print("handlers registered:", ", ".join(need))
EOF
say "done: $G (python: $PY)"
