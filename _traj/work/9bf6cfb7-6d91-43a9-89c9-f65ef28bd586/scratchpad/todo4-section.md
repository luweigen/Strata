## TODO 4 done: the last hole in the hipBLASLt table was the router, not the alpha/beta projections (2026-10-05, night)

[TODO.md](TODO.md) item 4: the engine with the table still logged `Lt fallback; no calibration for dtype=bf16 T=4165
N=256 K=2560 ldy=256` once per prompt. The shape is the router projection, `ffn_gate_inp.weight` [2560 x 256]: 256
experts per layer, one BF16 GEMM per layer in the prompt path's "router+shared" phase, 48 per chunk - not the GDN
alpha/beta projections as the earlier note said (those are [2560 x 48] each, written side by side into a 96-wide
buffer: the `bf16 48 2560 96` rows, in the table since the first run). A 4K prompt is one chunk (`prompt chunk auto:
8192`), so T is the whole prompt, and the lookup takes the nearest T: a row per chunk size covers every prompt.

**The tuner** (`tune_hipblaslt --case bf16,T,256,2560,256` at T = 1024, 2048, 4096 and 8192;
`docs/benchmarks/2026-10-05-halo-router-lt-tuning.log`, 10 s): the same solution, 1251 (the `MT96x96x32 ... SAV`
kernel the dense bf16 shapes use), is the best at every T, FP32 output at rel_l2 2.4e-6 against `hipblasGemmEx`:

| bf16 [T x 2560] x [2560 x 256], ms | T = 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| `hipblasGemmEx` (rocBLAS) | 0.43 | 1.37 | 3.16 | 5.63 |
| the same with `ROCBLAS_USE_HIPBLASLT=1`: the engine's real fallback (`...-tuning-rocblas-lt.log`) | - | - | 1.30 | 2.85 |
| solution 1251 | 0.081 | 0.14 | 0.27 | 0.53-0.55 |

Two rows, T = 4096 and 8192, are now in `tools/hip/gfx1151-hipblaslt-100401.txt` (50 rows).

**The engine** (`docs/benchmarks/2026-10-05-halo-router-lt-ab.py`: one server start per table from
`strata-coder-iq1_m.json` at 32/96, every expert resident, three pp4096 requests each, `STRATA_PREFILL_TIMING=1` and
`STRATA_HIPBLASLT_VERBOSE=1`; the engine logs `...-ab-before.log` and `...-ab-after.log`, the script's `...-ab.log`):

| Coder IQ1_M, 4,162-token chunk, requests 2-3 (the first carries the warm-up) | the table before | with the router rows |
|---|---|---|
| hipBLASLt lookups | 48 rows, `Lt fallback; no calibration ... N=256` | 50 rows, `Lt solution=1251 dtype=bf16 T=4162 N=256`, no fallback |
| "router+shared" phase | 222-223 ms | 161-162 ms |
| GPU timeline | 7,447-7,486 ms | 7,355-7,419 ms |
| pp4096 | 543-546 t/s (first request 523) | 547-553 t/s (first 531) |

48 GEMMs x (1.30 - 0.27 ms) is 49 ms; the phase lost 61 ms. The other phases are unchanged within their spread (gdn
922-952 against 927-949 ms, gate/up 1,576-1,611 against 1,582-1,613). Under 1% of a chunk, as the shape's size said it
would be; the item is closed because this was the last shape the engine fell back on: with this table every GEMM of
the Coder's prompt path runs through a calibrated solution (the dense shapes are the model's, so UD-IQ4_XS's too).

**On re-tuning.** The file's header carries the arch and the hipBLASLt version (`gfx1151 100401`); the engine
compares both with the runtime's and refuses a mismatch (`hipBLASLt version mismatch: file=... runtime=...; using
hipBLASEx`), so a ROCm wheel change cannot apply stale ids - it loses the table until `tune_hipblaslt` runs again
(the 32 dense shapes 60 s, a `--case` row seconds; `--tuning-out` writes a whole file, the `--case` rows of items 2
and 4 were appended by hand with their comments). Setup choosing the file by arch and version is item 12.

