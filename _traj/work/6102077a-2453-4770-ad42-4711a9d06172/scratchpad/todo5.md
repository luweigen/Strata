## TODO 5 done: the GDN phase's time was the output norm's launch, not the recurrence (2026-10-05, night)

[TODO.md](TODO.md) item 5 took llama.cpp's chunked GDN prefill (+8% of a whole prompt on this card in EngramHalo's
A/B) as the model for Strata's 0.8 s "gdn recurrence" phase. The premise was half right. Strata's
`gdn_recurrence` is token-serial, but it is not llama.cpp's serial form (one wave per head, 48 waves). Since D-2 it
splits each head's value columns over 4 blocks (192 blocks) and pipelines the next token's loads. And a third of
the phase was not the recurrence at all.

**What one layer's phase is made of** (`tests/hip/prefill_gdn_chunk_parity.cpp` with `STRATA_GDN_CHUNK_TIMING=1`,
events around each kernel, T = 4,165, warm and back to back; `docs/benchmarks/2026-10-05-halo-gdn-chunk-kernels.log`):

| kernel, T = 4,165, one layer | ms |
|---|---|
| the serial recurrence (`gdn_rec_cols_pipe_kernel`, 192 blocks) | 14.0-15.4 |
| the output norm before (`gdn_out_norm_kernel`, one 128-thread block per token and head: 199,920 blocks) | 11-25 |
| the output norm now (`gdn_out_norm_heads_kernel`, a wave per head, 8 heads per block: 24,990 blocks) | 1.5 |
| the chunked recurrence: fwdsub + Q K^T + the walk | 1 + 1 + 12-16 |

**The output norm.** The old kernel's work is 350 MB of traffic, under 2 ms at this card's bandwidth; it took as
long as the recurrence because the GPU dispatched 200,000 tiny workgroups. The new kernel gives each head one
wave. Lane l holds columns l, l + 32, l + 64 and l + 96, so each warp sum covers the same 32 columns as before and
the four are added in the same order. It is not bit-exact: the compiler rounds a few values differently, at most 5
ulp, in 3.5% of the FP32 outputs, at most 4.8e-7 absolute (T = 333, compared across two runs of the test).
`STRATA_GDN_NORM_OLD=1` restores the old kernel. Both prompt paths use the new one.

**The chunked recurrence.** Two versions were built and measured:

- **FP32 on the plain ALUs** (`docs/benchmarks/2026-10-05-halo-gdn-chunk-fp32.cu`, not built). Its 64-token chunks
  and later 32-token chunks matched the serial kernels to rel_l2 3e-7. It took 45 ms against the serial path's 26 ms
  at T = 4,165. The chunkwise form does the same FMA count as the serial one; it only wins when the products run on
  matrix cores. The probes it needed are in `docs/benchmarks/2026-10-05-halo-hip-{latency,blockops}.*`: 138-248 ns
  per dependent load, 43 ns per barrier with 192 blocks resident, 2.6-2.8 TFMA/s of plain FP32 at 2.1 GHz.
- **llama.cpp's kernels on the WMMA** (`src/prefill/gdn_chunk_wmma.cu`). These are the three kernels of
  `chunk_gated_delta_net.cu` with EngramHalo's RDNA layout fixes, built in `strata_mmq` against the vendored ggml's
  `mma.cuh` (identical to EngramHalo's). They read q, k and v from the conv output, keep V_corr in the output rows
  and the state in Strata's [k][head][v] layout. Against the serial kernels: y rel_l2 4.4-4.5e-4, state 2.9-3.0e-4,
  flat from T = 7 to 4,165; against a double recurrence, serial 1.8e-7 and chunked 4.4e-4: the FP16 operands.
  The walk takes 12-16 ms, the serial recurrence 14-15 ms. A walk tile of 16 value columns (384 blocks) was slower,
  18-21 ms.

The WMMA path is `STRATA_GDN_CHUNK=1`, off by default: it is no faster here and it rounds to FP16.
`hip_prefill_gdn_chunk_parity` checks both paths against each other and against double, and skips (77) where the
chunked path is not built.

**The engine** (`docs/benchmarks/2026-10-05-halo-gdn-chunk-ab.py`: one server start per mode from
`strata-coder-iq1_m.json` at 32/96, every expert resident, three pp4096 requests and one 1,213-token continuation;
`...-ab.log`, the engine logs `...-ab-{norm-old,default,chunked}.log`):

| Coder IQ1_M, 4,162-token chunk, requests 2-3 | old norm | **new norm (the default)** | new norm + chunked WMMA |
|---|---|---|---|
| "gdn recurrence" phase | 816-836 ms | **592-600 ms** | 689-693 ms |
| GPU timeline | 7,344-7,456 ms | **7,183-7,220 ms** | 7,310-7,340 ms |
| pp4096 | 550-551 t/s (first 530) | **562-566 t/s** (first 547) | 553-556 t/s (first 535) |
| 1,208-token prompt: "gdn recurrence" | 219 ms | **178 ms** | 233 ms |

The phase lost 230 ms per 4K chunk, 6.4 ms per layer: less than the test's 9.5 ms, because the old kernel was
slower in the test than in the engine (the test's z and y are cold). The 64-token continuations are not identical.
All three start with the same sentence and part where near-tied tokens fall the other way: the old and new norms
differ by a few ulp, which flips some FP16 roundings of the out projection's input. Decode does not use this path.

`ctest` on this build: 57 tests, the new `hip_prefill_gdn_chunk_parity` passes, the 4 known failures
(`hip_handoff`, `ple_parity`, `expert_parity`, `pool_test`), `hip_prefill_hipblaslt_gemm` skipped without
`STRATA_HIPBLASLT_TUNING` (`docs/benchmarks/2026-10-05-halo-gdn-ctest.log`). The engine was copied into `engine\`;
the previous one is `engine\strata-0.1.34-todo4.exe`.

What is left of the phase is the serial recurrence, 14-15 ms per layer, at about 0.5 TFMA/s. Neither chunked form
beat it on this card. A faster one would need the walk itself to be faster: more of it on the matrix cores, or
fewer barriers per chunk.

