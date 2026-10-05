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
