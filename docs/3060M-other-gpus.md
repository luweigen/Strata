# Other cards in the 3060M PC: an estimate

What a Tesla P40 (24 GB), one RTX 2080 Ti modified to 22 GB, or two of them would do for the two models in this PC
(Ryzen 9 8945HX, 96 GB DDR5-5200, see [3060M.md](3060M.md)). **This is an estimate, not a measurement**: none of these
cards was in the PC. It starts from the 3060M's measured verify windows (2026-10-17, [3060M.md](3060M.md) "Where the
decode time goes" and "The Coder's router") and changes two things: how many experts the bigger card holds, and how
fast its GPU does the same work. The model is `benchmarks/2026-10-17-3060m-gpu-estimate.py`; every number in the
tables below is its output.

## The cards

| | RTX 3060 Laptop (now) | Tesla P40 | RTX 2080 Ti, 22 GB mod |
|---|---|---|---|
| Architecture, compute capability | Ampere, 8.6 | Pascal, 6.1 | Turing, 7.5 |
| SMs, FP32 lanes | 30, 3840 | 30, 3840 | 68, 4352 |
| VRAM, bandwidth (spec) | 12 GB, 336 GB/s | 24 GB, 346 GB/s | 22 GB, 616 GB/s |
| Tensor cores (int8 / fp16) | yes | **no** (int8 via `dp4a` only; fp16 at 1/64 rate) | yes (no BF16) |
| PCIe | 4.0 x16, 26.8 GB/s measured | 3.0 x16 (~12 GB/s) | 3.0 x16 (~12 GB/s) |
| Strata support | release engine | **only the experimental build** (`-DSTRATA_EXPERIMENTAL_SM60=ON`, CUDA 12.x toolkit; written for sm_60 and sm_70, sm_61 untested) | release engine (since 0.1.27; QSA scorer as FP32 FMAs, older prompt-attention kernel) |

Two cards: Strata splits the layers (pipeline, [MULTI_GPU.md](MULTI_GPU.md)); a token crosses once per verify window
through pinned host RAM, a few hundred KB. **NVLink is not used** (no peer-to-peer access), so a bridge between two
2080 Ti changes nothing. The doc's measurement (RTX 5080 + 3090): decode about the faster card alone, prompts +18-20%,
plus what the second cache adds.

## How the estimate is made

From the 3060M's windows (decodes 2-3 of the C++ rewrite request, `--spec 4`):

| per verify window | Coder IQ1_M | UD-IQ4_XS |
|---|---|---|
| distinct experts per layer | 37.6 | 38.1 |
| measured hit share (slots) | 76.0% (2666) | 58.5% (1864) |
| CPU time per expert | 42.2 us | 56.4 us |
| GPU time per VRAM expert | 8.35 us | 8.83 us |
| other GPU work (dense, attention, hyper-connections, router, head) | 31.0 ms | 29.8 ms |
| MTP draft | 5.2 ms | 5.2 ms |
| tokens committed per window | 3.72 | 3.87 |

- **Hit share with more slots**: a Zipf law over each layer's experts, its exponent fitted to the measured share
  (Coder 1.016, UD-IQ4_XS 0.953). The adaptive swaps are not modelled. Slots = the extra VRAM over the 12 GB card /
  the cache's bytes per slot (2.06 MB, 2.41 MB); for two 22 GB cards, the second adds about 15 GiB (it holds its own
  copy of the dense weights, its state and prompt buffers).
- **The GPU's wait for the CPU**: the linear fit of the two measured windows, 0.88 x CPU time - 0.48 x GPU expert
  time, at least 0.5 ms. PCIe 3.0 cards: `--pcie-frac 0` (all misses on the CPU; on the 3060M, 0 was as fast as the best
  share: Coder 65.1 against 62.3 tokens/s at 0.2, UD-IQ4_XS 55.4 at both).
- **g**, the card's GPU time relative to the 3060M's for the same work, is not measured and is the largest
  uncertainty. These decode kernels reach about 40% of the 3060M's VRAM bandwidth ([3060M.md](3060M.md)), so
  bandwidth alone does not scale them. The 2080 Ti has 2.27 times the SMs and 1.83 times the bandwidth, but half
  the FP32 lanes per SM: g = 0.6-0.9. The P40 has the same SM count, ~1.5 GHz, and an older core: g = 1.2-1.6.
- **The model on the measured PC** gives Coder 64 and UD-IQ4_XS 50 tokens/s, against 66.4 and 55.4 measured: it
  reads 4% and 10% low, so the tables below lean low.

## Decode, estimated (tokens/s)

| | experts in VRAM (Coder / UD) | Coder IQ1_M | UD-IQ4_XS |
|---|---|---|---|
| RTX 3060 Laptop 12 GB (measured) | 22% / 8% | **66** | **55** |
| Tesla P40 24 GB (g 1.2 / 1.4 / 1.6) | 73% / 29% | 61 / 52 / 46 | 56 / 50 / 45 |
| RTX 2080 Ti 22 GB (g 0.6 / 0.75 / 0.9) | 64% / 26% | 121 / 97 / 81 | 85 / 75 / 67 |
| 2 x RTX 2080 Ti 22 GB (g 0.6 / 0.75 / 0.9) | 100% / 53% | 117 / 95 / 79 | 109 / 92 / 80 |

- **P40: no gain.** Twice the VRAM cuts the Coder's CPU experts from ~16 ms to ~4 ms per window, but the Coder is
  GPU-bound already (the GPU computes 44.5 of its 56.0 ms), and a slower GPU costs more than that. UD-IQ4_XS waits
  less for the CPU, and the slower GPU takes it back.
- **One 2080 Ti 22 GB: the most for the money** - more VRAM and a faster GPU: Coder +20-80%, UD-IQ4_XS +20-55%.
- **Two 2080 Ti**: the Coder fits whole, but one card already hits 93%, and the split adds a hand-off per window
  while the cards take turns: no gain for the Coder. UD-IQ4_XS goes from 77% to 89% hits: +20-30% over one card.

## Prompts (not modelled)

The prompt path streams every expert the card does not hold, once per chunk (up to 8,192 tokens), and both cards
have PCIe 3.0 (about 12 GB/s against 26.8 measured here):

- **2080 Ti, Coder**: 8 GiB to stream instead of 18, at half the link speed, about the same 0.7 s per chunk; the int8
  tensor cores are faster than the 3060M's. Likely 1.3-1.8 times today's 896 / 1065 tokens/s (4K / 16K).
- **2080 Ti, UD-IQ4_XS**: about 41 GiB per chunk, ~3.6 s at 12 GB/s (51 GiB at 26.8 GB/s = ~2 s now): the link
  likely sets the pace, 1.0-1.4 times today's 795 / 888 tokens/s. Two cards: the measured +18-20% from the pipeline.
- **P40**: no tensor cores (MMQ through `dp4a`), FP16 at 1/64 rate, and the older prompt-attention kernel: likely
  slower than the 3060M, perhaps 0.3-0.7 times.

## In this PC

- **One GPU slot**: `lspci` shows one x16 slot for a graphics card; a second card needs an M.2-to-PCIe adapter
  (x4 works for a layer split, [MULTI_GPU.md](MULTI_GPU.md)), and two 2080 Ti draw about 2 x 250 W.
- **P40**: passively cooled (it needs a fan on its fins) and powered through an 8-pin EPS (CPU-style) connector; the
  experimental build needs a CUDA 12.x toolkit next to the CUDA 13 one used now.
- **More context**: with 22-24 GB the 64K context fits without KV streaming (12 GB: 32K recommended).
- What would settle it: the measured window split ([3060M.md](3060M.md)) on the real card. g is the number to
  measure first; the GPU stage profile (`STRATA_VERIFY_PROFILE=1`) gives it directly.
