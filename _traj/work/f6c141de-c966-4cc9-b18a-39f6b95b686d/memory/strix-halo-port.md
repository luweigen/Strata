---
name: strix-halo-port
description: "State of the Strix Halo (gfx1151, Radeon 8060S) port on branch wmma-gfx1151 - what is done, the numbers, what is next"
metadata:
  node_type: memory
  type: project
  originSessionId: 53cb97e9-f88c-45ee-a459-ee7480855903
  modified: 2026-10-05T18:55:27.983Z
---

Branch `wmma-gfx1151`. docs/TODO.md items 1-5, 2a and 2b are done (2026-10-05/06); the write-ups are the "## TODO n done"
sections at the end of docs/AIMAX+395-ROCm.md (new data goes in a new subsection, never into an older table).

- Coder IQ1_M, every expert on the GPU: prefill 537-568 t/s (4K-20K), decode 37-40 t/s. Experts run through
  llama.cpp's MMQ by default (`Product::opt_rows` picks the J tile for the median row count).
- TODO 3 (2026-10-05 night): `hip_prefill_mmq_parity` failed only because the test filled its sentinel with
  `hipMemset` on the null stream while the products ran on a non-blocking stream; on Windows HIP the fill landed
  after the kernel. The MMQ kernels are correct on gfx1151 (rel_l2 0.0010 vs a double product). Fixed in the test
  with `hipMemsetAsync` on the stream. Known: a kernel trap here is reported by the next `hipMemcpy`, not by
  `hipDeviceSynchronize`.
- TODO 4 (2026-10-05 night): the table's last hole was the router GEMM (bf16 N=256 K=2560, 256 experts, 48 per chunk),
  not the alpha/beta projections; rows at T=4096/8192 (solution 1251) added, pp4096 543-546 -> 547-553 t/s. The
  tuner runs as `PATH=engine/rocm/bin:$PATH build-hip-win/tune_hipblaslt.exe --case dtype,T,N,K,ldy`; logs under docs/benchmarks are tracked since
  2026-10-05 (.gitignore exception; the user asked for it) - commit them with each item.
- TODO 5 (2026-10-05 night, commit 43ef6ff): the GDN phase's waste was gdn_out_norm launching 200K tiny blocks
  (11-25 ms/layer); a wave-per-head norm takes 1.5 ms (within 5 ulp, not bit-exact). pp4096 550 -> 562-566 t/s.
  llama.cpp's chunked GDN kernels are ported (src/prefill/gdn_chunk_wmma.cu, on the vendored ggml mma.cuh) but are
  no faster than Strata's column-split serial recurrence (14-15 ms/layer): opt-in STRATA_GDN_CHUNK=1.
  Time kernels warm and back to back; a single run after a big host copy reads high (clock ramp).
- TODO 2a (2026-10-06, commit c90320c): streamed MMQ layers group experts by row-count class (one group buffer per
  class, rows class-major, ring still walked in id order). Default one class per RDNA 3.5 J tile
  (STRATA_PREFILL_MMQ_CLASSES, =0 old walk). Coder pp4096 558-563 -> 571-574, 1.66K 460 -> 493 t/s, same text.
  UD-IQ4_XS decode text is NOT repeatable run to run at 32/96 (partial cache, CPU misses, MTP): compare its prompt
  path with STRATA_STATE_HASH_GDN=1, not with decoded text. A/B script: docs/benchmarks/2026-10-06-halo-mmq-classes-ab.py.
- TODO 2b (2026-10-06): MMQ on gfx1151 is WMMA iu8 (not sudot4). Its kernels spilled (IQ3_XXS/IQ2_S J=128, most types J=32);
  CMake STRATA_MMQ_GFX1151_SCHED=max-ilp (-Xarch_gfx1151 -mllvm=-amdgpu-sched-strategy=max-ilp on the 9 instance files)
  takes 5-10% off the products in the harness, +1-2% prompt in the engine, bit-identical. third_party/llama.cpp is
  gitignored and built unmodified (pinned FetchContent / STRATA_GGML_DIR): ship tweaks as Strata-side flags, not edits.
  Fast loop: compile one mmq-instance-<t>.cu with -Rpass-analysis=kernel-resource-usage (7 s) to see VGPRs/spills.
  Engine A/B: never compile on the box during a timed run (APU shares CPU/memory with the GPU; it cost 20%).
- Next: TODO 6 (gaps), then decode items 7-8.

**Why:** the user drives this port item by item ("do docs/TODO.md item N"); each item ends with measured numbers in
the port log and the TODO entry struck through with a summary and link.

**How to apply:** when asked to do the next item, read docs/TODO.md and the matching AIMAX+395-ROCm.md section
first; put probe sources and logs in docs/benchmarks/ as `2026-10-05-halo-*`; build probes with the SDK clang++
(see [[windows-shell-gotchas]]).
