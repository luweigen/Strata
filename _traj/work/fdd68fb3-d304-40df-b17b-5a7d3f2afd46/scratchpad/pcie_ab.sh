#!/bin/bash
# decode with --pcie-frac X on the Coder: fresh server per value, one warm-up, then the measured request
S=$(dirname "$0"); cd /home/luwei/work/AI/Strata
python3 - > $S/pcie_prompt.json <<'PY'
import json
L = open("/mnt/evox2/large/work/AI/EngramHalo.cpp/src/llama-model-loader.cpp").read()
i = L.index("static std::vector<std::string> llama_get_list_splits"); j = L.index("\n}\n", i) + 3
p = "Rewrite this C++ function to be clearer and faster. Keep the behavior. Reply with the code.\n\n```cpp\n" + L[i:j] + "```"
json.dump({"model": "x", "temperature": 0, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False},
           "messages": [{"role": "user", "content": p}]}, open("/dev/stdout", "w"))
PY
for x in "$@"; do
  python3 -c "
import json; c = json.load(open('strata-coder-iq1_m.json')); c['args'] += ['--pcie-frac', '$x']
c['log'] = '$S/pcie-$x-engine.log'; json.dump(c, open('$S/pcie-$x.json', 'w'), indent=1)"
  python serve/server.py --engine strata --config $S/pcie-$x.json --port 8080 > $S/pcie-$x-server.log 2>&1 & pid=$!
  until grep -q "^ready: http" $S/pcie-$x-server.log; do kill -0 $pid 2>/dev/null || { echo "server died ($x)"; tail -5 $S/pcie-$x-engine.log; exit 1; }; sleep 1; done
  args=$(ps -o args= -p $(pgrep -x strata) | grep -oE "\-\-pcie-frac [0-9.]+"); probe=$(grep -c "PCIe probe" $S/pcie-$x-engine.log)
  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","temperature":0,"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Write a thread-safe LRU cache in C++."}]}' > /dev/null
  r=$(curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d @$S/pcie_prompt.json)
  hit=$(grep -oE "expert cache [0-9.]+% hit" $S/pcie-$x-server.log | tail -1)
  echo "$r" | python3 -c "
import sys, json; r = json.load(sys.stdin); t = r['timings']; m = r['choices'][0]
print(f'pcie_frac $x (engine args: $args; probe lines: $probe): decode {t[\"predicted_per_second\"]} t/s over {t[\"predicted_n\"]} tokens, '
      f'drafts {t.get(\"draft_n_accepted\")}/{t.get(\"draft_n\")}, prefill {t[\"prompt_per_second\"]} t/s ({t[\"prompt_n\"]} tok), $hit, finish {m[\"finish_reason\"]}')
open('$S/pcie-$x-answer.txt', 'w').write(m['message']['content'] or '')"
  kill $pid; wait $pid 2>/dev/null; while pgrep -x strata >/dev/null; do sleep 1; done
done
