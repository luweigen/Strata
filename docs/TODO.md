# TODO: the Strix Halo (gfx1151) port, after 2026-10-05

What [AIMAX+395-ROCm.md](AIMAX+395-ROCm.md) left open, in the order of the time or risk at stake. Every item names
the measurement that decides it. Test outputs (engine logs, server output, benchmark JSON) go to
`docs/benchmarks/`, not the project root; the probe sources of that day are there too (`2026-10-05-halo-*`).

## Prompt speed (the Coder's 4K prompt: 9.2 s of GPU time, 434-451 t/s)

1. ~~**The FP32 prompt attention, 2.0 s (22%).** Port Strata's PR #313~~ **Done 2026-10-05** (branch
   `wmma-gfx1151`, [the section in AIMAX+395-ROCm.md](AIMAX+395-ROCm.md#todo-1-done-pr-313s-gfx11-matrix-core-prompt-attention-on-gfx1151-2026-10-05-night)):
   the PR merged onto 0.1.34 with its guards extended to gfx1150/gfx1151, the existing attention parity test now
   drives the gfx11 kernel (4 of 4 pass, 3.3x per chunk), `STRATA_PA_WMMA=1` in the config: pp4096 452 -> 517-543
   t/s, the attention phase 2,034 -> 611 ms. The PR's WMMA GEMM (`STRATA_WMMA_GEMM=1`) is slower than the calibrated
   hipBLASLt table here and declines the expert shapes: left off. Still open from it: the PR is not upstream (the
   maintainer asked for a rebase on 0.1.39 and per-arch guards), and the gfx11 attention kernel has no path for
   K8V4 pools.
2. ~~**The expert GEMMs, 2.6 s + 0.5 s dequant (39% of the 7.7 s a 4K prompt now takes).** 512 FP16 GEMMs of `ne`
   rows per layer with FP32 output run at 3.5-3.7 TFLOP/s ... Decide between a 16-bit output, a WMMA kernel, or the
   MMQ path~~ **Done 2026-10-05 night, with a corrected premise**
   ([the section in AIMAX+395-ROCm.md](AIMAX+395-ROCm.md#todo-2-done-the-expert-products-are-mmq-not-fp16-gemms-and-their-j-tile-2026-10-05-night)):
   the Coder's expert products run through llama.cpp's MMQ (int8) by default, not the FP16 GEMMs - those only run
   with `STRATA_PREFILL_MMQ=0`. Measured on this card: the three FP16 routes (hipBLASLt solutions from the tuner,
   16-bit output, PR #313's WMMA) and the MMQ products at one layer's shape. Shipped: `Product::opt_rows` - the MMQ
   J tile chosen for a group's median row count instead of its largest (`STRATA_PREFILL_MMQ_OPT=max` restores the
   old choice): gate/up 1,800-1,832 -> 1,581-1,604 ms, down 802-816 -> 764-769 ms, pp4096 509-525 -> 524-541 t/s,
   a 1.3K prompt 398 -> 424 t/s, the same text out; the full benchmark: Coder prefill 528 -> 548 t/s at 4.75K,
   517 -> 537 at pp4096, ~530 -> ~560 at 16-20K, decode unchanged, UD-IQ4_XS unchanged (its chunks stream); the gfx1151 hipBLASLt table extended with the expert shapes at
   small T (the FP16 path's gate/up 4,231 -> 1,593-1,671 ms, now on par with MMQ); experts sorted by row count
   before grouping where the chunk does not stream (below `STRATA_PREFILL_STREAM_MIN` = 1,024 tokens: 1,001-token
   prompts 360-388 -> 388-428 t/s; in arena mode every expert is staged from the arena in id order, so 4K chunks
   keep id order). Left open, in the next two items.
2a. **Group the experts by row count while they stream.** The MMQ products pay for the J tile the group's experts
   pad to: a tile for the largest pads 46% of a 4K chunk's rows and 161% of a 1.3K chunk's (the engine's own count,
   `STRATA_PREFILL_TIMING=1`); the median tile recovers part, sorted groups all of it (the harness
   `docs/benchmarks/2026-10-05-halo-mmq-bench.cpp`: gate/up IQ3_XXS 47.8 -> 34.5 ms per layer with the median,
   29.3 sorted). The streamed walk consumes ring slots in id order, so sorting needs the gathers to land in
   per-size-class group buffers (one group buffer per class, rows laid out class-major at routing time), or a prompt
   path that computes resident experts from their cache slots without staging (items 10, 11).
2b. **The MMQ kernels themselves: 6-12 TOPS at the expert shapes** (the harness, sorted groups: 8.7-12.4), against
   the card's int8 dot peak. The vendored ggml already has the RDNA 3.5 tile tables (`mmq-config-rdna3-5.cuh`: 256
   threads, I = 128 from J = 48 up) and uses `__builtin_amdgcn_sudot4`; llama.cpp#21284's smaller tiles (128
   threads, I = 64, J = 48; closed unmerged, pp128 +61-74% claimed) are untested here. The harness links
   `strata_mmq.lib`, so a config edit and `cmake --build build-hip-win --target strata_mmq` measure it.
3. ~~**`hip_prefill_mmq_parity` fails on gfx1151** ("synthetic-Q2_0-GU-pass0: non-finite or unwritten MMQ output").
   Find out whether it is ggml's gfx1151 handling (llama.cpp#21284: MMQ tile `mmq_x=48, mmq_y=64, nwarps=4`
   against VGPR spills; `__builtin_amdgcn_sudot4`) or Strata's host glue (`src/prefill/ggml_cuda_host.cu`). This
   matters more than it looked: the Coder's and UD-IQ4_XS's experts run through MMQ by default on this card (item 2),
   including layers whose down matrix is Q2_0 (`native_experts.txt`: layer 1 and others), and the day's benchmark
   outputs came from that path.~~ **Done 2026-10-05 night: neither**
   ([the section in AIMAX+395-ROCm.md](AIMAX+395-ROCm.md#todo-3-done-the-mmq-parity-failure-was-the-tests-stream-order-not-the-kernels-2026-10-05-night)):
   the test filled its output sentinel with `hipMemset` on the null stream and ran the products on a non-blocking
   stream; on Windows HIP the fill landed after the products, so every element read back as the sentinel (a probe
   over 5 types, both shapes, 1-135 rows: 17 of 17 cases, 100% of the elements, no error reported). The host and
   device tile tables agree for gfx1151, and the kernel itself - the library's and a fresh instantiation, MoE and
   plain mode, -O3 and -O1, any stream - writes every element at rel_l2 0.0010 against a CPU double product. The
   test now fills with `hipMemsetAsync` on its stream and passes (6 products, rel_l2 0.0005-0.0012); the ctest tally
   on this card is 4 known failures. The engine's prompt path has no synchronous memset, and the engine with the
   default and with `STRATA_PREFILL_MMQ=0` continues a 1,173-token code prompt with the same code (one whitespace
   run differs: int8 against FP16 rounding), so the day's benchmark outputs were computed. Probes:
   `docs/benchmarks/2026-10-05-halo-mmq-{parity,config,kernel}-probe*`, the engine A/B `2026-10-05-halo-mmq-ab.py`. Also seen:
   a kernel trap on this HIP is reported by the next `hipMemcpy`, not by `hipDeviceSynchronize`.
4. ~~**The hipBLASLt table's holes.** `tools/hip/gfx1151-hipblaslt-100401.txt` covers the dense shapes at T = 4096 and
   8192 and, since item 2, the two expert shapes at T = 16-512; the engine still logs `Lt fallback; no calibration
   for dtype=bf16 T=4165 N=256 K=2560` for the 256-wide alpha/beta projections. Calibrate those (the tuner takes
   `--case`; the lookup already takes the nearest T). Also re-run `tune_hipblaslt` whenever the ROCm wheels change
   (ids are per version).~~ **Done 2026-10-05 night**
   ([the section in AIMAX+395-ROCm.md](AIMAX+395-ROCm.md#todo-4-done-the-last-hole-in-the-hipblaslt-table-was-the-router-not-the-alphabeta-projections-2026-10-05-night)):
   the shape is the router (`ffn_gate_inp.weight`, 256 experts, one BF16 GEMM per layer, 48 per chunk), not the
   alpha/beta projections (those are N = 48 and were in the table). Two rows (T = 4096 and 8192, solution 1251: the
   best at every T from 1024 to 8192, `docs/benchmarks/2026-10-05-halo-router-lt-tuning.log`) close the last hole:
   the GEMM 1.30 -> 0.27 ms at T = 4096 in the tuner (against the engine's `ROCBLAS_USE_HIPBLASLT=1` fallback), the
   engine's "router+shared" phase 222-223 -> 161-162 ms per 4K chunk, pp4096 543-546 -> 547-553 t/s, no fallback
   logged (`2026-10-05-halo-router-lt-ab.py`). On a wheel change the engine refuses the file (its header carries the
   arch and version) and says so; re-tune then.
5. **The GDN recurrence, 0.8 s.** llama.cpp's chunked GDN prefill kernel gained 8% of a whole prompt on this card
   (EngramHalo `docs/strix-halo/windows.md`, 2026-10-04); Strata's `gdn_recurrence` is the same serial form.
6. **Gaps, 1.5 s.** Launch overhead of many small kernels and GEMMs; worth a trace (`STRATA_TRACE`) before any
   fusing.

## Decode

7. **UD-IQ4_XS decodes at 22-32 t/s whether 97% or 60% of its experts are on the GPU** (the Coder's decode does
   depend on residency). Run with `STRATA_DECODE_TIMING=1` at 32/96 and find the shared cost; candidates: the GPU
   expert kernels for this file's IQ3_S / IQ4_NL / Q8_0 blobs on RDNA 3.5, the per-token PLE rows from the NVMe,
   or a copy path (pytorch/pytorch#171687 reports gfx1151 decode 90% in `hipMemcpyWithStream`).
8. **The Coder's decode at 36-40 t/s with every expert on the GPU** is 0.8-0.9x a 12 GB RTX 3060 that computes
   four fifths of the experts on its CPU. The decode phase timing (`STRATA_DECODE_TIMING=1`) has not been read on
   this card yet.

## Memory and setup on an APU

9. **Pinned host memory counts twice in HIP's free figure** (`docs/benchmarks/2026-10-05-halo-hip-pinfree.hip`:
   16 GiB registered -> 32 GiB less free). `--expert-cache auto` sizes from that figure and gave UD-IQ4_XS 0 slots
   at 32/96; the remedy was `STRATA_ARENA_PIN_GIB=24` by hand. Setup and the engine should do this themselves on an
   integrated GPU: cap the pin, and size the cache from the dedicated carve-out rather than the WDDM figure.
10. **The in-place mmap mode read 47.6 GB of GGUF per 16K prompt** at 96/32 with all 12,288 experts resident
    (`expert tiers: ... files ... MB read`), far more than the 1,945 lent slots' experts. Find what reads (the
    routing prefetch of whole layers?) and whether it is needed when the prompt path's experts are resident.
11. **UD-IQ4_XS has no good split on this BIOS** (32 / 64 / 96 only): its 55.4 GiB of experts need ~61 GiB of GPU
    memory and its pinned arena ~69 GiB of RAM. A prompt path that takes resident experts from the cache instead of
    the host source would make 96/32 the right split for every model; until then 32/96 with a partial cache.
12. **Setup for gfx1151 users.** The arch is in the lists; still missing: gfx1151 in the release zip
    (`build_windows.bat` default list has it, the zip must be built and published), the docs' "integrated GPUs are
    not supported" lines ([AI_SETUP.md](AI_SETUP.md), [INSTALL.md](INSTALL.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md),
    [AMD_HIP.md](AMD_HIP.md)), setup choosing `STRATA_HIPBLASLT_TUNING` from `tools/hip/gfx1151-hipblaslt-100401.txt`
    (it should, by arch and version: verify), and `ROCBLAS_USE_HIPBLASLT=1` in the config's env for AMD cards.
13. **Display timeout.** Windows Strix Halo drops the GPU to ~600 MHz for compute when the console display is off
    (ROCm/legacy-rocm-build#6675; not seen here, 2.8-2.9 GHz under load over RDP with a 300 s timeout). The engine
    or server could hold `SetThreadExecutionState(ES_DISPLAY_REQUIRED)` while a request runs.

## Housekeeping

14. The `--pcie-frac` rule reads a meaningless probe on an APU (8-16 TB/s with a fully mapped arena, 68.8 GB/s with
    the pin cap) and keeps 0.55; measured here 0.2 vs 0.55 made no difference. Nothing to fix for speed; the log line
    should say what it saw.
15. `tests/hip/handoff` times out on Windows (the mapped-pointer alias, known, #325): skip it on `_WIN32` or make it
    report rather than hang.
16. The 3060M-style benchmark script now reads the EngramHalo checkout from `STRATA_ENGRAM_SRC` and its sources as
    UTF-8; `pp4k.py` is the quick single-shape check. Keep new runs' JSON and logs in `docs/benchmarks/`.
