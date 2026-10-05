---
name: strix-halo-port
description: "State of the Strata gfx1151 (Radeon 8060S, Ryzen AI MAX+ 395) port on branch AIMAX395-ROCm as of 2026-10-05: built, serving on the BIOS 64/64 split with ROCBLAS_USE_HIPBLASLT=1; decode 1.3-1.6x llama.cpp, prefill 0.85x"
metadata:
  node_type: memory
  type: project
  originSessionId: ed11a558-488f-44ff-ae44-74cbeaee92fe
  modified: 2026-10-05T10:26:53.436Z
---

On 2026-10-05 the gfx1151 port was made on branch `AIMAX395-ROCm` (the user's trajectory hook auto-commits every
turn): arch lists in `cmake/hip_backend.cmake`, `intrinsics.hpp`, `setup.py` (PCI id 0x1586), `build_windows.bat`
(new `STRATA_ROCM_ROOT`), two fixes in `tools/hip/package_windows.py`. Engine zip in `dist\`, unpacked in `engine\`;
build tree `build-hip-win` (ROCm from the read-only `rocm100-py312` env). The BIOS memory split is now **64/64**
(was 96/32). `strata-coder-iq1_m.json` = the recommended config (arena mode, `ROCBLAS_USE_HIPBLASLT=1`), started by
`run-coder-iq1_m.ps1` from the `strata` conda env; `strata-coder-iq1_m-arena.json` is the measured variant without
the env switch. Pack and MTP live in `C:\Users\Wei Lu\Documents\Strata-data`. Full record: `docs/AIMAX+395-ROCm.md`.

**Why:** measured: Coder decodes 33-36 t/s (MTP), all 12,288 experts on the GPU; prefill 264-279 t/s at 4K with
hipBLASLt routing (llama.cpp on the same PC: 318-322). The prompt path's time is in dense GEMMs and the engine's
own kernels (FP32 prompt attention, GDN recurrence, hc read), not in expert transfers (0.6%). ctest:
`hip_prefill_mmq_parity` fails on gfx1151 (MMQ is opt-in, off by default).

**How to apply:** next gains are kernel work: a gfx11-layout WMMA prompt attention, the expert GEMMs, hc read.
Use `STRATA_PREFILL_TIMING=1` and `docs/benchmarks/2026-10-05-halo-pp4k.py` to measure. Don't redo the probes;
cite the doc. See [[conda-env-policy]], [[where-things-go]].
