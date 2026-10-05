---
name: strix-halo-port
description: State of the Strata gfx1151 (Radeon 8060S) port as of 2026-10-05 evening: serving with the calibrated hipBLASLt table (tools/hip/gfx1151-hipblaslt-100401.txt); Coder prefill 434-451 t/s (1.4x llama.cpp), decode 36-40; next: WMMA attention (PR #313), expert GEMMs
metadata:
  type: project
---

Branch `AIMAX395-ROCm` (the user's trajectory hook auto-commits every turn). Engine zip in `dist\`, unpacked in
`engine\`; build tree `build-hip-win` (ROCm from the read-only `rocm100-py312` env, `STRATA_ROCM_ROOT`). BIOS split
now 64/64 (choices are 32/64/96 only). Recommended config `strata-coder-iq1_m.json`: arena mode, env
`ROCBLAS_USE_HIPBLASLT=1` + `STRATA_HIPBLASLT_TUNING=tools\hip\gfx1151-hipblaslt-100401.txt` (made with
`build-hip-win\tune_hipblaslt.exe` in 60 s, 5.2-5.4x on the dense shapes). Started by `docs/benchmarks/2026-10-05-run-coder-iq1_m.ps1` from the
`strata` conda env. UD-IQ4_XS configs need `STRATA_ARENA_PIN_GIB=24` at 32/96 (pinned memory counts double in HIP's
free figure). Full record: `docs/AIMAX+395-ROCm.md`.

**Why:** measured: Coder decode 36-40 t/s, prefill 434-451 t/s flat 4K-20K (llama.cpp 322-391 / 273-281); the
slow library path is any GEMM with FP32 output (4 TFLOP/s vs 24 with 16-bit out); prompt time now: FP32 attention
2.0 s, expert GEMMs 2.6 s + dequant 0.5 s of 9.2 s per 4K.

**How to apply:** next: port PR #313 (gfx11 WMMA GEMM + prompt attention, add `__gfx1151__` guards) and a WMMA or
16-bit-out path for the expert GEMMs; measure with `STRATA_PREFILL_TIMING=1` + `docs/benchmarks/2026-10-05-halo-pp4k.py`.
Don't redo the probes; cite the doc. See [[conda-env-policy]], [[where-things-go]].
