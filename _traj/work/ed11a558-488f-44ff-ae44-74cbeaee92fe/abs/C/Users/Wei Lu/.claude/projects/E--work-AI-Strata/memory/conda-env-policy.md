---
name: conda-env-policy
description: "Never modify the user's existing conda envs (C:\\conda_envs\\rocm100-py312 etc.); a Strata install goes into a new env `conda create -n strata python=3.14 -y`"
metadata:
  node_type: memory
  type: feedback
  originSessionId: ed11a558-488f-44ff-ae44-74cbeaee92fe
  modified: 2026-10-05T08:26:30.165Z
---

On the Strix Halo PC (GMKtec EVO-X2), the existing conda envs under `C:\conda_envs\` (e.g. `rocm100-py312`, which
holds the TheRock ROCm 10.0.0 wheels with gfx1151) are read-only for me. If one is not suitable for Strata, create a
fresh env with `conda create -n strata python=3.14 -y` instead of installing into or changing an existing one.

**Why:** the user said so on 2026-10-05 ("如果当前conda env不合用，要新建conda create -n strata python=3.14 -y，不要破现在的环境"); those envs back other projects (llama.cpp/EngramHalo runs, BFCL).

**How to apply:** use `rocm100-py312` only for its `hipcc`/`hipInfo`/libraries; never `pip install` into it. Note
`tools/hip/build_windows.bat` expects a venv with `Scripts\python.exe` and would try to pip-install into whatever
`ROCM_VENV` points at, so do not point it at a conda env. See [[strix-halo-port]] for the port analysis in
docs/AIMAX+395-ROCm.md.
