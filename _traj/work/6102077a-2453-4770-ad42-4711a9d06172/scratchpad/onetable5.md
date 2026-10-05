### The one table, redone with TODO 5's engine

The same table as [TODO 2's](#the-one-table-redone-with-todo-2s-engine), with the engine after TODO 4 (the router's
hipBLASLt rows) and TODO 5 (the wave-per-head GDN output norm) in both Strata columns; the earlier tables are left as
they were. The same benchmark (`2026-10-02-3060m-bench_halo.py`), the same configs (`strata-coder-iq1_m.json` at
32/96, all 12,288 experts resident, 100% hits; `strata-unsloth-ud-iq4_xs.json` at 32/96, `STRATA_ARENA_PIN_GIB=24`,
10,000 slots asked and 14,358 made, 96.7-99.0% hits), run back to back by `2026-10-05-halo-onetable-run.py` late on
2026-10-05 (`benchmarks/2026-10-05-halo-coder-iq1_m-todo5.json`, `benchmarks/2026-10-05-halo-ud-iq4_xs-todo5.json`,
the engine logs beside them; the load time is the server's start to its first `/v1/models` answer). llama.cpp and the
3060M PC columns are the night's. MTP on in every decode row (llama.cpp with the EasiiX MTP sidecar; its tg128 rows
are plain decode). Where the other Halo engine wins a row, its number is in brackets; TODO 2's Strata number follows
in parentheses where it moved.

| Coder IQ1_M / UD-IQ4_XS | **Coder: Halo best = Strata, every expert on the GPU, TODO 5's engine** | Coder: 3060M PC | **UD-IQ4_XS: Halo, llama.cpp 96/32 with the chunked GDN kernel, against Strata 32/96 (TODO 5's engine) in brackets; the better one in bold** | UD-IQ4_XS: 3060M PC |
|---|---|---|---|---|
| model load to listening | **11.7 s** (was 12.6) | 11.5 s | 83-94 s (**Strata: 36.1 s**) | 42 s |
| cold first request, decode | **38.6 t/s** (was 40.4; llama.cpp: 26.1) | 43.3 t/s | 28.1-30.4 t/s (Strata: 29.5, was 30.9) - even | 37.6 t/s |
| fresh code prompt, decode | **36.9 t/s** (llama.cpp: 21.9) | 42.1 t/s | 25.1-30.7 t/s (Strata: 26.5) - even | 38.7 t/s |
| prefill @ ~4.75K | **573.0 t/s** (was 548.1; llama.cpp: 321.5) | 961.3 t/s | 322 t/s (**Strata: 341.3**, was 317.5) | 662.8 t/s |
| decode tail after that prefill | **33.6 t/s** (was 32.3; llama.cpp: 20.8) | 39.3 t/s | 21.3-25.1 t/s (Strata: 22.3) - even | 33.1 t/s |
| repeated prompt (reference) | 39.1 t/s (was 40.8; llama.cpp: 49.9, n-gram drafts) | 45.1 t/s | **58.6 t/s** (Strata: 30.4) | 42.6 t/s |
| pp4096 @ d0 | **557.5 t/s** (was 536.6; llama.cpp: 318.1) | 926.9 t/s | **372-391 t/s** (Strata: 335.0, was 322.0) | 819.4 t/s |
| tg128 @ d0 | **27.9 t/s** (llama.cpp: 20.9 plain) | 36.2 t/s | 21.8-22.8 plain (Strata: 21.1 MTP) - even | 32.2 t/s |
| pp4096 @ d16384 | **~585 t/s** (581.7 over 20,421, the 16,384 prefix at 592.7; was ~560; llama.cpp: 269.9) | ~1,126 t/s | 273-281 t/s (**Strata: 351-356** over 16-20K, was 347-354) | ~949 t/s |
| tg128 @ d16384 | **28.1 t/s** (llama.cpp: 18.0 plain) | 37.4 t/s | 16.3-19.4 plain (Strata: 19.1 MTP, was 21.8) - even | 31.7 t/s |
| ratio to the 3060M PC, decode | 0.75-0.89 | 1 | 0.60-0.78 (Strata) | 1 |
| ratio to the 3060M PC, prefill | 0.52-0.60 | 1 | 0.37-0.51 (Strata; llama.cpp 0.4-0.5) | 1 |

What moved: prefill, on both models. The Coder +3-5% at every length (548 -> 573 at 4.75K, 537 -> 558 at pp4096,
~560 -> ~585 at 16-20K), as the two items' A/Bs said (the router rows under 1%, the GDN norm 2-3%). UD-IQ4_XS +1-7%
(317.5 -> 341.3 at 4.75K, 322 -> 335 at pp4096, 347-354 -> 351-356 at 16-20K): the GDN layers are the same dense
weights in both models, so the norm's 230 ms per 4K chunk applies to it as well, while its long chunks stay bound by
the arena staging (TODO items 10 and 11). Its 4.75K prefill now beats llama.cpp's 322 t/s; llama.cpp still leads at
pp4096 @ d0. Decode does not use either change: the rows that moved down are the ones whose MTP acceptance moved
down (Coder cold 91.1 -> 86.4%, UD-IQ4_XS tg128 @ d16384 73.9 -> 58.3%), the prompt path's last bits being
slightly different and the text drafted along a different path; the decode rows stay within the spread of the
earlier tables. The day's prompt speed for the Coder at 4.75K: 146 -> 209 -> 274 -> 451 -> 528 -> 548 -> 573 t/s.

