## TODO 2b done: the MMQ kernels spilled registers; LLVM's max-ilp scheduler takes 5-10% off them (2026-10-06)

[TODO.md](TODO.md) item 2b: the expert products' MMQ kernels run at 6-12 TOPS at the expert shapes. The lead was
llama.cpp#21284's smaller tiles. Two premises were wrong. First, MMQ on gfx1151 does not use `sudot4`: the
vendored ggml defines `AMD_WMMA_AVAILABLE` for every gfx11 chip, and each kernel instance holds 276
`v_wmma_i32_16x16x16_iu8` instructions and no `v_dot4` (`llvm-objdump` on the gfx1151 code object of
`mmq-instance-{iq3_xxs,q2_0,iq4_nl}.cu`). `sudot4` is the path for cards without matrix cores. Second, the pinned
llama.cpp (3cf0325) is newer than #21284. It has one tile table per architecture (`mmq-config-rdna3-5.cuh`, 256
threads and I = 128 for J >= 48), and #21284's tile (128 threads, I = 64) is what the RDNA 3.0 table already uses.

**The harness** (`docs/benchmarks/2026-10-06-halo-mmq-tiles.cpp`, linking `strata_mmq.lib`; random weight bytes as in
TODO 2's harness) times each expert type two ways. "Uniform" is 512 experts of exactly J rows, groups of 16, the
tile for J: the kernel's own throughput at each tile. "Layer" is one 4,165-token layer as the engine runs it since
TODO 2a: Zipf-like rows, class-major, groups of 16 within a class, the tile for the group's median. The baseline
(`...-tiles-base.log`) showed two things. IQ3_XXS and IQ2_S (the Coder's gate/up) run *slower* at J = 128 than at
J = 112 (IQ3_XXS 9.8-9.9 against 12.5-13.1 TOPS). Most types run J = 32 no faster per operation than J = 16 (IQ4_XS
gate/up 6.5 against 7.6 TOPS). The compiler's resource report (`-Rpass-analysis=kernel-resource-usage`,
`...-mmq-registers.log`) explains both:

| VGPRs / scratch bytes per lane, default scheduler | J = 16 | J = 32 | J = 48-112 | J = 128 |
|---|---|---|---|---|
| IQ3_XXS (Coder gate/up) | 195 / 0 | 256 / 16 | 191-237 / 0 | **256 / 208** |
| IQ2_S (Coder gate/up) | 256 / 28 | 186 / 0 | 213-256 / 0-56 | **256 / 132** |
| IQ4_NL, Q2_0 (Coder down) | 211 / 0 | **256 / 108** | 165-241 / 0 | 228-251 / 0 |
| IQ4_XS, IQ3_S, Q8_0 (UD-IQ4_XS) | 212-227 / 0 | **256 / 120-180** | 125-245 / 0 | 229-251 / 0 |

256 VGPRs is the most a wave32 kernel can have here; the rest spills to scratch memory. The spills do not come from
the tile's area: a thread holds J/2 accumulators whatever I is, and moving J = 32 to 256 threads / I = 128 still
spilled. They come from how LLVM's default scheduler orders the fully unrolled WMMA loop.

**What was tried** (the nine MMQ instances rebuilt with extra device flags and linked ahead of `strata_mmq.lib`;
"layer" ms per layer, 2-3 runs each, `...-tiles-variants.log`):

| ms per layer | IQ3_XXS g/u | IQ2_S g/u | IQ4_XS g/u | IQ3_S g/u | IQ4_NL down | Q2_0 down | IQ4_XS down | Q8_0 down |
|---|---|---|---|---|---|---|---|---|
| default scheduler | 28.1-28.2 | 30.9-33.7 | 25.1-25.7 | 28.4-28.5 | 15.1-15.5 | 15.3-15.6 | 10.8-11.2 | 15.6-15.8 |
| `-amdgpu-sched-strategy=iterative-minreg` (no spills, 104-239 VGPRs) | 33.0 | 39.6 | 33.3 | 33.9 | 19.5 | 17.9 | 14.6 | 20.2 |
| `-amdgpu-sched-strategy=max-memory-clause` | 26.3 | 30.8 | 24.7 | 28.6 | 15.0 | 14.8 | 11.0 | 15.9 |
| `-amdgpu-use-amdgpu-trackers=1` | 26.7 | 30.8 | 26.1 | 29.2 | 15.5 | 15.7 | 11.2 | 17.3 |
| `-unroll-threshold=50` (fixes most J = 32 spills) | 27.9 | 31.2 | 23.4 | 26.4 | 14.4 | 14.2 | 10.2 | 14.6 |
| **`-amdgpu-sched-strategy=max-ilp` (shipped)** | **25.8-25.9** | **30.1-30.7** | **22.6-23.1** | **26.4-26.8** | **14.1-14.4** | **14.1-14.3** | **10.2-10.5** | **14.8-15.0** |
| max-ilp + `-unroll-threshold=50` | 25.7-25.9 | 30.0-30.1 | 22.0-22.2 | 26.6 | 14.4 | 14.2-14.4 | 10.2 | 14.7-14.9 |
| max-ilp + the #21284 tiles (128 threads, I = 64, J >= 48) | 25.3-25.6 | 31.0-31.9 | 22.0-22.5 | 26.5-27.0 | 13.9-14.0 | 13.6-13.9 | 9.7-9.9 | 14.7-15.1 |

max-ilp is the best or within the spread of the best on every type: -8% on IQ3_XXS, -10% on IQ4_XS gate/up, -6 to -8%
on the down products. It still spills a little (IQ3_XXS 72 bytes at J = 128 instead of 208, most J = 32 cases none),
and IQ3_XXS's uniform J = 128 goes 9.8 -> 12.9 TOPS. The fewest registers (iterative-minreg) is the slowest: the
WMMA loop wants the registers it asks for, in a better order. The #21284 tiles on top of max-ilp gain 2-3% more on the
down products and lose on IQ2_S; they would also need a patched copy of upstream's table (Strata builds the pinned
commit unmodified, `third_party/llama.cpp` is not in this repository). Not taken.

**Shipped** (`CMakeLists.txt`, the HIP `strata_mmq` block): the cache variable `STRATA_MMQ_GFX1151_SCHED`, default
`max-ilp`, adds `-Xarch_gfx1151 -mllvm=-amdgpu-sched-strategy=max-ilp` to the nine MMQ instance files only.
`-Xarch_gfx1151` passes it to the gfx1151 device compile alone (checked with `-###` on a gfx1151 + gfx1100 build:
the gfx1100 job does not get it), since no other card was measured. `-DSTRATA_MMQ_GFX1151_SCHED=` turns it off. The
built library times like the variant (`...-tiles-maxilp.log`), and `hip_prefill_mmq_parity` passes at the same
rel_l2 (0.0005-0.0012).

**In the engine** (`docs/benchmarks/2026-10-06-halo-mmq-sched-ab.py`: TODO 2a's engine against this build, alternating,
one server start each, `strata-coder-iq1_m.json` at 32/96 with every expert resident, `STRATA_PREFILL_TIMING=1` and
`STRATA_STATE_HASH_GDN=1`; `...-sched-ab.out`, the engine logs beside it). The first pair ran while the register
survey was compiling on the same machine (pp4096 449 t/s for the old engine, against 561 in the second pair) and is
left out. The second pair:

| Coder IQ1_M, per chunk | TODO 2a's engine | **max-ilp** |
|---|---|---|
| 4K: gemm gate/up / gemm down, ms | 1,463-1,479 / 706-713 | **1,396-1,430 / 703-710** |
| 4K: GPU timeline, ms; pp4096 requests 2-3 | 7,130-7,291; 560-561 t/s | **7,052-7,252; 566-567 t/s** |
| 1.66K: gemm gate/up / gemm down, ms | 688-696 / 327-333 | **654-660 / 319-320** |
| 1.66K: GPU timeline, ms; prompt t/s | 3,243-3,291; 466-468 | **3,205-3,227; 476-477** |
| 1,208 tokens: gate/up / down, ms; t/s | 553 / 265; 414 | **518 / 249; 424** |

The products lose 4-6% (gate/up) and 1-6% (down), less than in the harness, and the down products of a 4K chunk
hardly move. The harness's Zipf layer is not the real routing (it has no expert of 16 rows or fewer, for one). The prompt gains +1% at 4K and +2% at
1.2-1.7K. **The same results:** a scheduler only reorders instructions. The GDN state hash after the 1,208-token
prompt is `f298 0c4e ...` in all four runs, the same as TODO 2a's runs logged. The 64-token continuation is the same
text in all four runs. `ctest` on this build: 56 tests (`hip_handoff` left out), the 3 known failures
(`ple_parity`, `expert_parity`, `pool_test`), `hip_prefill_hipblaslt_gemm` skipped, `hip_prefill_mmq_parity` passes
(`2026-10-06-halo-mmq-sched-ctest.log`). The engine was copied into `engine\`; the previous one is
`engine\strata-0.1.34-todo2a.exe`. The kernels still run at 9-13 TOPS per layer (the harness, max-ilp). Neither
the tile table nor the register allocation moves that much further. Where the rest goes inside the kernel (the
tile loads, the scale arithmetic around each WMMA) was not measured.

### The one table, redone with TODO 2b's engine

The same table as [TODO 2a's](#the-one-table-redone-with-todo-2as-engine), with this section's engine in both Strata
columns; the earlier tables are left as they were. The same benchmark (`2026-10-02-3060m-bench_halo.py`), the same
configs, run back to back by `2026-10-05-halo-onetable-run.py todo2b` on 2026-10-06
(`benchmarks/2026-10-05-halo-coder-iq1_m-todo2b.json`, `benchmarks/2026-10-05-halo-ud-iq4_xs-todo2b.json`, the engine
logs beside them, the rows in `2026-10-05-halo-onetable-todo2b.log`). llama.cpp and the 3060M PC columns are the
night's. MTP on in every decode row (llama.cpp with the EasiiX MTP sidecar; its tg128 rows are plain decode). Where
the other Halo engine wins a row, its number is in brackets; TODO 2a's Strata number follows in parentheses where it
moved.

| Coder IQ1_M / UD-IQ4_XS | **Coder: Halo best = Strata, every expert on the GPU, TODO 2b's engine** | Coder: 3060M PC | **UD-IQ4_XS: Halo, llama.cpp 96/32 with the chunked GDN kernel, against Strata 32/96 (TODO 2b's engine) in brackets; the better one in bold** | UD-IQ4_XS: 3060M PC |
|---|---|---|---|---|
| model load to listening | 11.2 s (was 15.4) | 11.5 s | 83-94 s (**Strata: 36.2 s**) | 42 s |
| cold first request, decode | **37.7 t/s** (was 39.1; llama.cpp: 26.1) | 43.3 t/s | 28.1-30.4 t/s (Strata: 30.6) - even | 37.6 t/s |
| fresh code prompt, decode | **37.2 t/s** (was 38.0; llama.cpp: 21.9) | 42.1 t/s | 25.1-30.7 t/s (Strata: 25.9) - even | 38.7 t/s |
| prefill @ ~4.75K | **564.7 t/s** (was 587.5; llama.cpp: 321.5) | 961.3 t/s | 322 t/s (**Strata: 351.7**, was 355.2) | 662.8 t/s |
| decode tail after that prefill | **32.5 t/s** (llama.cpp: 20.8) | 39.3 t/s | 21.3-25.1 t/s (Strata: 22.5) - even | 33.1 t/s |
| repeated prompt (reference) | 38.1 t/s (llama.cpp: 49.9, n-gram drafts) | 45.1 t/s | **58.6 t/s** (Strata: 30.9) | 42.6 t/s |
| pp4096 @ d0 | **577.4 t/s** (was 576.1; llama.cpp: 318.1) | 926.9 t/s | **372-391 t/s** (Strata: 349.4, was 339.1) | 819.4 t/s |
| tg128 @ d0 | **27.9 t/s** (llama.cpp: 20.9 plain) | 36.2 t/s | 21.8-22.8 plain (Strata: 21.0 MTP) - even | 32.2 t/s |
| pp4096 @ d16384 | **~594 t/s** (595.2 over 20,421, the 16,384 prefix at 593.4; was ~595; llama.cpp: 269.9) | ~1,126 t/s | 273-281 t/s (**Strata: 352-357** over 16-20K, was 362-366) | ~949 t/s |
| tg128 @ d16384 | **28.0 t/s** (llama.cpp: 18.0 plain) | 37.4 t/s | 16.3-19.4 plain (**Strata: 19.2** MTP, was 22.5) | 31.7 t/s |
| ratio to the 3060M PC, decode | 0.75-0.88 | 1 | 0.61-0.81 (Strata) | 1 |
| ratio to the 3060M PC, prefill | 0.53-0.62 | 1 | 0.37-0.53 (Strata; llama.cpp 0.4-0.5) | 1 |

This run does not show the A/B's gain. The Coder's single 4.75K request read 564.7 t/s against TODO 2a's 587.5, and
the other prompt rows moved by -1 to +0.7%. UD-IQ4_XS: pp4096 +3%, its other prompt rows -1 to -3%, and its tg128 @
d16384 19.2 t/s at 55.3% MTP acceptance (72.6% in TODO 2a's run; its decode is not repeatable from run to run, see
TODO 2a). These moves are the size of the spread between runs of one engine. So the Coder's rows were run again,
TODO 2a's engine and this one alternating, two pairs (`2026-10-06-halo-onetable-todo2b-coder-pairs.log`,
`benchmarks/2026-10-05-halo-coder-iq1_m-{old,new}-pair{1,2}.*`):

| Coder IQ1_M, the one table's rows, t/s | TODO 2a's engine (2 runs) | **TODO 2b's engine (2 runs)** |
|---|---|---|
| prefill @ ~4.75K | 577.5-577.6 | **582.6-583.1** |
| pp4096 @ d0 | 568.0-571.1 | 564.3-576.0 |
| depth 16384 prefix / pp4096 @ d16384 (20,421 tokens) | 594.2 / 586.1-588.0 | **602.0-602.9 / 594.3-596.4** |
| decode rows (cold, fresh, 4.75K tail, tg128 @ d0, tg128 @ d16384) | 36.6-38.8, 37.8, 33.6, 27.6-28.0, 27.9-28.4 | 38.6-38.8, 37.6-37.7, 33.3-33.8, 27.9-28.0, 27.7 |

Back to back, the Coder's prompt gains +1% at 4.75K and +1.4% at 16-20K. pp4096 is even within its spread, and
decode does not use this path. That agrees with the A/B above (+1% at 4K, +2% at 1.2-1.7K). The 564.7 of the one-table
run was low for that run, not for the engine. The gain is small next to TODO 2a's (+2-3%). The scheduler takes 5-10%
off the products, and the products are about 30% of a 4K chunk's GPU time.

