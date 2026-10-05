# DONE: the Strix Halo (gfx1151) port

Finished items of [TODO.md](TODO.md), newest first, each with where the record is. The port itself (the day's work
before the TODO list existed) is in [AIMAX+395-ROCm.md](AIMAX+395-ROCm.md).

## 2026-10-05

**TODO 1 - the FP32 prompt attention, 2.0 s (22%) of a 4K prompt.** Port Strata's PR #313 (branch `wmma-optin`:
gfx11 WMMA prompt attention + dense WMMA GEMM, opt-in `STRATA_PA_WMMA=1` / `STRATA_WMMA_GEMM=1`) to gfx1151.
Done on branch `wmma-gfx1151`
([the section in AIMAX+395-ROCm.md](AIMAX+395-ROCm.md#todo-1-done-pr-313s-gfx11-matrix-core-prompt-attention-on-gfx1151-2026-10-05-night)):
the PR merged onto 0.1.34 with its device guards extended to gfx1150/gfx1151 (the gfx12 and gfx11 kernels coexist as
`prompt_attn_wmma_kernel` / `prompt_attn_wmma11_kernel`); the existing attention parity test
(`src/kernels/qsa_prompt_attn_parity.cpp`, ctest `hip_prompt_attn_wmma`) now drives the gfx11 kernel with both KV
formats instead of skipping: 4 of 4 pass, 3.9e-6 of FP64, 2.9-3.4x per chunk; the PR's GEMM parity test passes its
3,520 cases (max abs 2.4e-7); `STRATA_PA_WMMA=1` is in `strata-coder-iq1_m.json`: pp4096 452 -> 517-543 t/s, the
attention phase 2,034 -> 611 ms, decode unchanged (37-41 t/s), the same `is_prime`. The PR's WMMA GEMM is slower
than the calibrated hipBLASLt table on this card and declines the expert shapes: left off (its follow-ups stay as
TODO 1). Raw results: `benchmarks/2026-10-05-halo-coder-iq1_m-arena-lt-table-pa-wmma.json`, ctest log
`build-hip-win/ctest-wmma.log`.
