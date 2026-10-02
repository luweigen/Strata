#!/bin/bash
# run_bench.sh <label> <config>: start the server, time load-to-listening, run bench_halo.py, stop the server
S=$(dirname "$0"); cd /home/luwei/work/AI/Strata
t0=$(date +%s.%N)
python serve/server.py --engine strata --config "$2" --port 8080 > "$S/server-$1.log" 2>&1 &
pid=$!
until grep -q "^ready: http" "$S/server-$1.log"; do
  kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 "$S/server-$1.log"; exit 1; }; sleep 0.5; done
echo "load_to_listening_s $(echo "$(date +%s.%N) - $t0" | bc)"
python "$S/bench_halo.py" "$1" "$S/result-$1.json"; rc=$?
kill $pid; wait $pid 2>/dev/null
until ! pgrep -x strata >/dev/null; do sleep 1; done
exit $rc
