import glob, json, os, sys
for f in glob.glob(os.path.join(sys.argv[1], "**", "*_result.json"), recursive=True):
    lines = [l for l in open(f, encoding="utf-8") if l.strip()]
    bad = [l for l in lines if "Error during inference" in json.dumps(json.loads(l).get("result"))]
    if bad:
        open(f"{f}.removed-{sys.argv[2]}", "w", encoding="utf-8").writelines(bad)
        open(f, "w", encoding="utf-8").writelines(l for l in lines if l not in bad)
    print(f"{os.path.basename(f)}: {len(lines) - len(bad)} kept" + (f", {len(bad)} with an inference error removed" if bad else ""))
