---
name: strix-halo-port
description: "State of the Strata gfx1151 (Radeon 8060S) port as of 2026-10-05 late night, branch wmma-gfx1151 (TODO 1 and 2 done): Coder prefill 537-568 t/s (4K-20K), decode 37-40; the Coder's experts run through MMQ by default (not FP16 GEMMs); next: TODO 2a (size-class groups), 2b (MMQ tiles), 3 (MMQ parity)"
metadata:
  node_type: memory
  type: project
  originSessionId: 0c48fab3-1b8b-4b85-888e-da64babaa7a2
  modified: 2026-10-05T17:42:50.698Z
---

Branch `wmma-gfx1151` (the user's trajectory hook auto-commits every turn). Engine zip in `dist\`, installed copy in
`engine\strata.exe` (backups `engine\strata-0.1.34-pre313.exe`, `engine\strata-0.1.34-todo1.exe`); build tree
`build-hip-win` (ROCm from the read-only `rocm100-py312` env; `cmake --build build-hip-win --target strata`, then copy
to `engine\`). BIOS split 32/96 (choices 32/64/96 only; the Coder's 12,288 experts fit the 32 GiB side, 100% hits).
Config `strata-coder-iq1_m.json`: arena mode, env `ROCBLAS_USE_HIPBLASLT=1` +
`STRATA_HIPBLASLT_TUNING=tools\hip\gfx1151-hipblaslt-100401.txt` (48 rows now: dense shapes + the two expert shapes
at T=16-512) + `STRATA_PA_WMMA=1`. Server: `docs/benchmarks/2026-10-05-run-coder-iq1_m.ps1` from the `strata` conda
env; quick check `docs/benchmarks/2026-10-05-halo-pp4k.py <label> [repeat] [tokens]` with
`STRATA_ENGRAM_SRC=E:\work\AI\EngramHalo.cpp\src`; phases with `STRATA_PREFILL_TIMING=1`. UD-IQ4_XS needs
`STRATA_ARENA_PIN_GIB=24` at 32/96. Full record: `docs/AIMAX+395-ROCm.md` (sections "TODO 1 done", "TODO 2 done").

**Why:** TODO 2's premise was wrong: the Coder's (and UD-IQ4_XS's) expert types are MMQ types, so the prompt path
runs llama.cpp's int8 MMQ by default (`STRATA_PREFILL_MMQ=0` gives the FP16 GEMMs). Shipped 2026-10-05 night:
`mmq::Product::opt_rows` = the group's median row count for the J tile (`STRATA_PREFILL_MMQ_OPT=max|mean|median`):
pp4096 509-525 -> 524-541 t/s, identical text; the full benchmark (`docs/benchmarks/2026-10-05-halo-coder-iq1_m-todo2.json`): 548 / 537 / ~560 t/s at 4.75K / pp4096 / 16-20K, UD-IQ4_XS unchanged (streams); the table rows for the FP16 path; experts sorted by rows only when
the chunk does not stream (< 1,024 tokens; arena mode stages every expert in id order). Harnesses:
`docs/benchmarks/2026-10-05-halo-expert-gemm.cpp` (FP16 routes), `2026-10-05-halo-mmq-bench.cpp` (MMQ, links
`build-hip-win/strata_mmq.lib`; build lines in the file headers; hipcc needs `-Wl,<lib>` and `-D_DLL -D_MT -Xclang
--dependent-lib=msvcrt`).

**How to apply:** next: TODO 2a (per-size-class group buffers so sorted groups work while streaming: harness says
-38% on gate/up), 2b (MMQ tile configs for gfx1151 via the harness), 3 (MMQ parity failure on Q2_0 - the engine runs
MMQ on Q2_0 down layers regardless). Don't redo the probes; cite the doc. See [[conda-env-policy]], [[where-things-go]].
