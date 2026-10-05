# Strata on the Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151) with ROCm on Windows: can it run?

**Short answer: yes, after a small port, and nothing found so far stands in the way.** This PC (a GMKtec EVO-X2:
Ryzen AI MAX+ 395, the Radeon 8060S integrated GPU, 128 GB of LPDDR5X, Windows 11) has everything Strata's HIP
backend needs: wave32, 64 KiB of LDS per workgroup, the signed dot4 instruction, the 100 MHz GPU wall clock, mapped
pinned host memory, and hipBLAS kernels for gfx1151 in the ROCm wheels already installed here. On 2026-10-05, 11 of
the engine's kernel files were compiled for gfx1151 unchanged, and small programs checked each of those
assumptions on the card ([What was checked](#what-was-checked-on-this-pc-2026-10-05)). What stops it today is a
list of places where Strata names its supported architectures and does not know gfx1151
([What stops it today](#what-stops-it-today)). **Later the same day the port was made, built and run** (the
[Execution log](#execution-log-2026-10-05-the-same-afternoon)): the Coder IQ1_M serves on the 8060S with every
expert on the GPU, decodes 1.3-1.6x faster than llama.cpp on this same PC (35-37 vs 22-26 tokens/s with MTP), and
reads prompts 2.2x slower (146 vs 321 tokens/s) for a reason the log shows and a BIOS setting should fix.

The analysis is the record of one afternoon's checks, in the order they were made. The three probe programs are in
`docs/benchmarks/2026-10-05-halo-*`.

## The PC

Read on 2026-10-05 with WMI, the registry, `hipInfo` from the ROCm wheels and the probes below. "spec" marks a
vendor figure.

| | |
|---|---|
| System | GMKtec NucBox EVO-X2 mini PC; Windows 11 Pro 10.0.26200, 64-bit |
| CPU | AMD Ryzen AI MAX+ 395 (Strix Halo, Zen 5), 16 cores / 32 threads |
| GPU | AMD Radeon 8060S Graphics, `gfx1151` (RDNA 3.5), integrated; PCI `VEN_1002&DEV_1586`; HIP reports 20 multiprocessors (WGPs; 40 CUs spec), 2900 MHz, wave32, 64 KiB LDS per block, 2 MiB L2, `isIntegrated 1`, `isLargeBar 0`, `canMapHostMemory 1`, `hostNativeAtomicSupported 0` |
| Driver | AMD Software: Adrenalin 32.0.31041.1004 (driver date 2026-08-17); HIP runtime and driver version 70260201 |
| Memory | 8 x 16 GB Micron LPDDR5X at 8532 MT/s (WMI), 256-bit: 273 GB/s theoretical (spec) |
| Memory split | BIOS gives the GPU a 96 GiB dedicated carve-out (`HardwareInformation.qwMemorySize` = 103,079,215,104 bytes); Windows sees **31.6 GiB** of system RAM (21.3 GiB free at the time); HIP reports **107.9 GiB total, 105.6 GiB free** (`hipMemGetInfo`: the carve-out plus shared memory) |
| Commit limit | 131.6 GiB (a 100 GB pagefile on C:) |
| Disks | C: NVMe, 1.9 TB, **89 GB free**; E: 11 TB USB hard disk (exFAT), 2.4 TB free: this repository is on E: |
| ROCm | AMD's TheRock Python wheels `rocm==10.0.0` with `libraries`, `devel`, `device-gfx1151` in the conda env `C:\conda_envs\rocm100-py312` (Python 3.12.14): HIP 7.15.26333, AMD clang 23.0.0git targeting `x86_64-pc-windows-msvc`, `hipblas.dll`, `rocblas.dll`, `libhipblaslt.dll`, and `rocblas/library/gfx1151`, `hipblaslt/library/gfx1151`, `.kpack/blas_lib_gfx1151.kpack` |
| Build tools | Visual Studio Professional 2022 with the C++ tools (found by `vswhere`), CMake 4.2.3, git 2.51 |
| Other GPU | an NVIDIA GeForce RTX 5090 shows in Windows' device list with status "Unknown": an external card over Thunderbolt that was not attached; `nvidia-smi.exe` is installed. Not used here |
| Model files | see [The model files already here](#the-model-files-already-here) |

The same machine was measured with llama.cpp (EngramHalo.cpp's `docs/strix-halo/windows.md`): those numbers are
quoted in [3060M.md](3060M.md#against-llamacpp-on-strix-halo-the-same-tasks) as the comparison for Strata on an RTX
3060 Laptop GPU. They are the only token/s this PC has so far: with llama.cpp, the Coder IQ1_M decodes at 21-26 t/s
(MTP) and prefills 4.75K-token prompts at 321 t/s; UD-IQ4_XS 23-30 t/s and 277-284 t/s.

## What Strata's HIP backend needs, and what gfx1151 has

From [AMD_HIP.md](AMD_HIP.md), `cmake/hip_backend.cmake` and `include/strata/hip_compat/intrinsics.hpp`, the
backend is "the same engine as on NVIDIA compiled for AMD", for wave32 RDNA2/RDNA3/RDNA4. The supported list is
gfx1100, gfx1101, gfx1102, gfx1200, gfx1201 and gfx1030. gfx1151 is RDNA 3.5: the gfx11 instruction set with
RDNA3's features, so every assumption below was checked on the card rather than taken from the name.

| The backend assumes | gfx1151 on this PC | How it was checked |
|---|---|---|
| wave32 (`p.warpSize == 32`, checked at startup by `src/core/device.cu`) | warpSize 32 | `hipInfo`; the probe kernel |
| 64 KiB of LDS per workgroup (`fused_gr.cu`'s tile) | `sharedMemPerBlock` 65,536 | `hipInfo`; the probe |
| the signed dot4 instruction `__builtin_amdgcn_sudot4` for `__dp4a` (quantized kernels) | present and correct: a known dot gave 18 | the probe, compiled for gfx1151 |
| `v_perm_b32` (`__builtin_amdgcn_perm`) for `__byte_perm` | correct selection (0xccdd3344 from the test words) | the probe |
| wave shuffles (`__shfl_xor`) | correct | the probe |
| a 100 MHz GPU wall clock (`wall_clock64`, `verify_kernels.cu:549`, "gfx10.3 / gfx11 / gfx12") | **99.99 MHz** measured against a 1 s host sleep | `2026-10-05-halo-hip-membw.hip` |
| mapped pinned host memory (`cudaHostAlloc` + device pointer: the expert arena, the doorbell ring) | a 1 GiB and a 4 GiB mapped allocation succeed; the device pointer **is the host pointer**, as [AMD_HIP.md](AMD_HIP.md#windows) notes for Windows (#325): `tests/hip/handoff` will time out here as on the RX 9070 XT, the engine does not use that copy | the probe |
| hipBLAS `GemmEx` BF16/FP16 in, FP32 out, FP32 compute (`src/prefill/gemm.cu:388`), and the Windows stale-`hipErrorInvalidValue` workaround (#247) | both GEMMs correct (max relative error 8e-7 BF16, 3.6e-6 FP16 against a double-precision CPU product); **no stale error** after them on this card and ROCm 10.0.0 | `2026-10-05-halo-hipblas-gemm.cu`, through Strata's own `cublas_v2.h` shim |
| hipBLASLt (optional, needs a calibrated table per architecture in `tools/hip`) | the library and gfx1151 kernels exist; no table for gfx1151 -> the plain hipBLAS path, as on the R9700 where a table gained 0-4% | file listing |
| the RDNA4-only WMMA kernels (`STRATA_HIP_WMMA`, `STRATA_SELECT_WMMA`: `#if defined(__gfx1200__) \|\| defined(__gfx1201__)`) | not compiled for gfx1151 (gfx11.5 has WMMA with gfx11's fragment layout, which the engine has no kernel for): prompt attention and the QSA scorer run the portable FP32 kernels, as on gfx1100 | source |
| `__threadfence_system()` on mapped memory (the doorbell ring, `elementwise.cu:210`) | `hostNativeAtomicSupported 0`; the engine uses fences and volatile stores, not host-native atomics. Not exercised yet: a real run (or `ctest`) will tell | `hipInfo`, source |

## What was checked on this PC (2026-10-05)

All with the conda env's `hipcc.exe` (`--offload-arch=gfx1151 -O2`), MSVC's headers found by clang on its own,
each program compiled in 2-4 s. Nothing in the existing conda env was changed.

**1. The runtime sees the card.** `hipInfo` lists device 0 "AMD Radeon(TM) 8060S Graphics", gfx1151, the properties
in [The PC](#the-pc). This is what `engine\strata-device.exe --list-devices` will print once there is an engine with
gfx1151 code ("arch gfx1151, 107.9 GiB, wave32").

**2. Instruction probe** (`docs/benchmarks/2026-10-05-halo-hip-probe.hip`): one 32-thread kernel on the card:

```
device: AMD Radeon(TM) 8060S Graphics gcnArchName=gfx1151 warp=32 LDS/block=65536 totalGlobalMem=107.9 GiB integrated=1
dp4a=18 (want 18)  perm=0xccdd3344  shfl_xor(0,1)=1 warpSize=32
hipHostMalloc 1 GiB mapped: no error   host 0000001008000000 device 0000001008000000 (same pointer)
hipMemGetInfo: free 105.6 GiB / total 107.9 GiB
```

**3. Memory** (`docs/benchmarks/2026-10-05-halo-hip-membw.hip`; 2048 blocks x 256 threads, `float4` streams, timed
with HIP events, best of the passes shown):

| | |
|---|---|
| `hipMalloc` of 48 GiB (an expert cache the size of IQ3_S's 50 GB of experts needs about this) | succeeds |
| GPU writes the 48 GiB | 206-212 GB/s |
| GPU reads the 48 GiB | **235-237 GB/s** (86% of the 273 GB/s spec; llama.cpp's own test on this PC: 233-239) |
| GPU reads 4 GiB of mapped pinned **host** memory through the device pointer | **232-235 GB/s**: the same as device memory. On a discrete card this is the PCIe path (16-25 GB/s on x16 Gen4); here it is the same DRAM |
| `hipMemcpy` device to host, 4 GiB pinned | 46.7 GB/s |
| `hipMemcpy` host to device, 4 GiB pinned | the events measured ~0 ms both times (the copy engine's work is not ordered with the events here); not a number |
| `wall_clock64` | 99.99 MHz |

**4. The engine's GEMM** (`docs/benchmarks/2026-10-05-halo-hipblas-gemm.cu`; N=3072, T=512, K=2048, the shape of a
dense projection on a 512-token prefill chunk; 20 timed calls):

```
BF16 GEMM: status 0, sync no error, stale error after it: no error, max|diff| 0.00001 (8.00e-07 rel), 3.36 ms/call = 1.9 TFLOP/s
FP16 GEMM: status 0, sync no error, stale error after it: no error, max|diff| 0.00006 (3.56e-06 rel), 2.66 ms/call = 2.4 TFLOP/s
```

1.9-2.4 TFLOP/s is what hipBLAS 10.0.0 gives on gfx1151 at this shape; the card's matrix-core peak is far above it.
The prompt path's dense projections are a part of prefill, the streamed experts another: whether this GEMM rate
bounds prefill here is a question for a real run.

**5. The engine's own kernel files compile for gfx1151.** With the flags `cmake/hip_backend.cmake` uses (the
force-included `cuda_runtime.h` shim, `STRATA_USE_HIP=1`, `STRATA_HIP_ARCHS="gfx1151"`, C++20), these compiled to
objects without a source change: `src/core/device.cu`, `src/prefill/gemm.cu`, and in `src/kernels/cuda/`
`elementwise.cu`, `router_top10.cu`, `qsa_select.cu`, `qsa_prompt_attn.cu`, `fused_gr.cu`, `native_mmvq.cu` (37 s),
`iq_kernels.cu` (13 s, needs `-Ithird_party/ggml`), `verify_kernels.cu`, `s2_expert_grouped.cu`. The warnings are the
usual ones (unused results of `nodiscard` HIP calls). The remaining `.cu` files and the host code were not compiled
one by one: the full CMake build is the next step, not this page.

## What stops it today

In the order a user would hit them. Every item is a list that does not contain gfx1151, except the last two.

1. **`cmake/hip_backend.cmake:4-37`** - `CMAKE_HIP_ARCHITECTURES=gfx1151` is a `FATAL_ERROR` ("Strata HIP supports
   wave32 gfx1100, gfx1101, gfx1200 and gfx1201 ..."). gfx1151 belongs in `_strata_hip_unvalidated` until a model
   has run, then in the validated list.
2. **`include/strata/hip_compat/intrinsics.hpp:18-19`** - `dp4a` uses `__builtin_amdgcn_sudot4` only under
   `__gfx1100__ || __gfx1101__ || __gfx1102__ || __gfx1200__ || __gfx1201__`. Without `__gfx1151__` (and
   `__gfx1150__`) in that list a gfx1151 build **silently takes the portable four-multiply loop**: correct, but the
   quantized kernels lose the instruction the whole backend is built around. The probe shows the builtin exists and
   is right on gfx1151.
3. **`setup.py`** - `AMD_ARCHS` (line 967), `AMD_NAMES` (968), `AMD_CARDS` (973) and `_WIN_AMD_DID` (1071: the PCI
   id `0x1586` -> gfx1151; the name rule `8060S`) do not know the card, so setup says "not supported - Strata's AMD
   backend runs on ... only, this is gfx1151" (`amd_problem`, line 1019), and on Windows `win_amd_arch` returns `""`
   for it (an "integrated Radeon"). `ROCM_INDEXES` (960, Linux only) has no TheRock index for gfx1151.
   The RAM and VRAM arithmetic needs no change (next section). The #325 rule "an integrated Radeon is HIP device 0
   and pushes the discrete card to 1" holds here trivially: there is one device.
4. **The ready-made Windows engine** (`strata-windows-x64-hip.zip`, built by `tools\hip\build_windows.bat` for
   `gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030`) carries no gfx1151 code: the engine's startup check would stop
   with "GPU 0 (AMD Radeon(TM) 8060S Graphics, gfx1151) is not an architecture this Strata engine was compiled
   for". `STRATA_HIP_ARCHS=gfx1151` builds one that is; a release zip would need gfx1151 added to the list (one more
   `device-gfx1151` wheel, and `rocblas/library/gfx1151` + `hipblaslt/library/gfx1151` in the zip; TheRock's Windows
   wheels have them, checked here for 10.0.0).
5. **ROCm version.** `build_windows.bat` installs `rocm==10.2.0a20260930` from the `whl-next` nightly index into
   `.rocm-win`. Whether that nightly has a `device-gfx1151` wheel was not checked; the 10.0.0 release wheels here
   do, and passed every probe above. `STRATA_ROCM_VERSION=10.0.0` with `STRATA_ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/`
   (where this PC's wheels came from, per EngramHalo's notes) is the known-good choice.
6. **The docs** say integrated GPUs are not supported ([AI_SETUP.md](AI_SETUP.md), [INSTALL.md](INSTALL.md),
   [TROUBLESHOOTING.md](TROUBLESHOOTING.md), [AMD_HIP.md](AMD_HIP.md)): true for the small iGPUs those lines were
   written for (a gfx1036 with 2 GiB), to be qualified for Strix Halo once it runs.
7. **ggml's MMQ prefill path** (`STRATA_PREFILL_MMQ=ON` in the Windows build): `src/prefill/ggml_cuda_host.cu` maps
   gfx1151 to ggml's cc `0x01001151`, inside ggml's RDNA3 range, as upstream ggml does; the pinned ggml (llama.cpp
   `3cf0325`) is the same lineage EngramHalo.cpp built and ran on this card "with no source changes". Not built here
   yet.

No kernel, host-code or memory-model change was found to be necessary. The port is lists and a build.

## Memory: how Strata's design maps onto a unified-memory APU

Strata was designed for a discrete card and system RAM ([DETAILS.md](DETAILS.md#how-it-works)): the experts live
pinned in RAM, the GPU holds the dense layers, the KV cache and a cache of the most-used experts; the CPU computes
the experts the GPU does not hold, or they are copied over PCIe (`--pcie-frac`). On this PC the two pools are
the same LPDDR5X, split by the BIOS: 96 GiB for the GPU, 31.6 GiB for Windows. What that means, model by model
(setup's sizes from `setup.py`'s `MODELS`; its rules `low_ram_needed`, `low_ram_gpu_gb`, `low_ram_resident`):

| Model | Experts (`arena_gb`) | Default mode (experts pinned in RAM, needs arena + 10 GB) | Low-RAM mode on this PC (GPU holds `min(arena, VRAM - 5)`) |
|---|---|---|---|
| Coder IQ1_M | 23.4 | needs 33.4 GB > 31.6: **no** | GPU holds 100%, 0 GB left for RAM: **resident variant, nothing in RAM** |
| Q2_0 | 34.0 | no | the same |
| IQ2_XS | 35.5 | no | the same |
| IQ3_XXS | 42.9 | no | the same |
| IQ3_S | 50.3 | no | the same |
| UD-Q4_K_XL (experimental) | 77.0 | no | GPU holds 100% of 77 GB (107.9 - 5 = 102.9 > 77); setup's RAM-budget mode is not needed |

So with a gfx1151 entry, setup's existing arithmetic already chooses the right thing: the low-RAM mode's resident
variant with the whole expert set in the GPU's cache and the CPU expert pool idle. That is the configuration the
R9700 32 GB ran the Coder in ([AMD_HIP.md](AMD_HIP.md#rdna4-gfx1201): 12,288 slots, all of the Coder's experts
on the card, 45-60 tok/s decode on a 640 GB/s card). Four details:

- **Disk, not RAM, is the tight resource.** The low-RAM mode's `experts.bin` copy is +23-50 GB on the disk; C: has
  89 GB free, E: is a USB hard disk. The engine's GGUF-in-place mode (0.1.31, `--mmap-experts` with no
  `experts.bin` in the pack: it reads each expert's rows from the GGUF files themselves) avoids the copy; setup does
  not choose it yet, a hand-written config can (as [3060M.md](3060M.md) did for UD-IQ4_XS). With every expert in
  the GPU cache after the load, the mapped files are read once at start.
- **`--pcie-frac` and the mapped reads cost nothing extra here.** The GPU reads mapped host memory at 232-235 GB/s,
  the speed of its own memory. On a discrete card the "PCIe share" of missed experts is a bandwidth trade; here a
  miss copied to the GPU is a memory-to-memory copy. With 100% of the experts resident there are no misses to
  trade, so this matters only for a model that does not fit, or a carve-out smaller than the experts.
- **The CPU half competes for the same 236 GB/s.** Strata's "CPU computes the experts the GPU does not hold, at the
  same time" gains nothing on an APU: both sides read the same DRAM. The all-on-GPU placement is the right one here,
  and the 16 Zen 5 cores are left for the server, the tokenizer and the prompt path's host work.
- **Windows commit.** The engine commits what it pins; the resident low-RAM variant leaves 4 GB of free RAM
  (`STRATA_RESIDENT_HEADROOM_GIB`). 31.6 GiB with a 131.6 GiB commit limit is enough for the engine's non-expert
  memory (about 26 GB resident on the R9700 runs; less here with no RAM copy of the experts). The pagefile was
  enlarged to 100 GB on 2026-09-27 after llama.cpp's `bad allocation` on this PC; Strata will not need it for the
  experts.

The BIOS carve-out could be changed (64/64, or a smaller GPU share), and the engine would then split the experts
between the cache and RAM as it does on a discrete card. Written before the run: "there is no reason to". The run
showed one: the prompt path streams the experts from the host-side source for every chunk, and on 31.6 GiB that
source has no file cache (see [Step 5](#execution-log-2026-10-05-the-same-afternoon)). For decode the reasoning
holds.

**What bounds decode.** The GPU's 235 GB/s read bandwidth is the ceiling for a memory-bound decode: at that rate
every token can read at most ~10 GB of weights at 23 t/s, ~5 GB at 47 t/s. The Coder's per-token expert bytes
(IQ1_M-sized, half the experts), the MTP draft layer's acceptance and the kernels' efficiency on RDNA 3.5 decide
where in that range Strata lands, and none of that is measured here.

## The model files already here

In `C:\Users\Wei Lu\Documents` (the user confirmed these are the weights to use); sizes are apparent sizes:

| Folder | Files | For Strata |
|---|---|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF\IQ1_M\` | `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` (28 GB), `-00002-of-00002.gguf` (27 GB); `mmproj-...-BF16.gguf` (866 MB) in the parent | setup's **Coder** (`--family coder --model IQ1_M`): `--gguf-dir` takes these two, no download |
| `Qwen3.8-Flash-Next\` | `Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` (11 MB, metadata), `-00002` (47 GB), `-00003` (41 GB); `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (3.9 GB, llama.cpp's sidecar, not used by Strata); `mmproj-F16.gguf` | Unsloth **UD-IQ4_XS**, not a setup choice: packed and run by hand in [3060M.md](3060M.md#unsloth-ud-iq4_xs-not-a-setup-choice-it-runs) (`tools/iq_pack.py --compat-bf16`, 55.4 GiB of experts: all of them fit the GPU here) |
| `Qwen3.8-27B-GSQ-RCO-GGUF\` | a 27B dense model | not a Strata model |

The MTP draft layer Strata uses (~5-6 GB, from the original checkpoint) is not here: setup's step 6 downloads it.
**Where things go on this PC** (the user's rule): the conda environments and everything model-sized - the GGUFs,
the packs, the draft layer - stay on C: (the NVMe, 89 GB free; `--gguf-dir` uses the files where they are,
`--models-dir` on C: takes the rest). The Strata source, its build (`build-hip-win`, `dist\`, `engine\`) and the
results (logs, benchmark JSON, these pages) stay in this repository on E:.

## The plan from here

Not done on this page; each step is a check with a definite outcome.

1. **The port** (branch `AIMAX395-ROCm`): gfx1151 (and gfx1150) in `cmake/hip_backend.cmake`'s lists and
   `intrinsics.hpp`'s dot4 condition; the setup entries of item 3 above; `build_windows.bat`'s arch list. About 15
   lines.
2. **The build**, on this PC. `tools\hip\build_windows.bat` with `STRATA_HIP_ARCHS=gfx1151`, and either its own
   `.rocm-win` venv with the 10.0.0 wheels (`STRATA_ROCM_VERSION=10.0.0`, `STRATA_ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/`)
   or the conda root passed by hand (the script's `cmake` line with `-DCMAKE_HIP_COMPILER=C:/conda_envs/rocm100-py312/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe`,
   `-DCMAKE_PREFIX_PATH=<that root>` and `--rocm-device-lib-path=<root>/lib/llvm/amdgcn/bitcode`). The build
   directory is the script's default, `build-hip-win` in this repository (the USB disk is slower, but the build and
   its results belong with the source); the ROCm wheels are not copied there: they come from the conda env on C:.
   **Environments:** the existing `rocm100-py312` env is read, never written (the script's venv install must not
   target it: it expects `Scripts\python.exe`, which a conda env does not have, so `ROCM_VENV` is never pointed at
   a conda env). If a new environment is needed for Strata itself, it is a new conda env on C:, `conda create -n
   strata python=3.14 -y`; whether every pinned package in `requirements.txt` has a Python 3.14 wheel is checked at
   that step (numpy 2.5.3, pillow 12.3, psutil 7.2 do; the ROCm wheels are `py3-none`).
   Outcome: `engine\strata-device.exe --list-devices` prints the card with no "cannot run", `--selftest` passes.
3. **ctest** with `build_windows.bat tests`: the R9700's known results are 42 of 45 (`hip_handoff` times out on
   Windows, `ple_parity` needs the Q2_0 fixture, `expert_multi_test` needs AVX-512; this CPU has AVX-512), so
   expect `hip_handoff` to fail and little else.
4. **The Coder, end to end:** `START-HERE.bat --backend hip --prebuilt dist\ --family coder --model IQ1_M
   --gguf-dir "C:\Users\Wei Lu\Documents\Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF\IQ1_M" --models-dir C:\...` (the
   pack, the draft layer, the config), then the server smoke (`serve/server.py`: chat, streaming, the Anthropic
   endpoint). Outcome: the log's `expert tiers: GPU ... hits` line shows every expert on the GPU, and coherent
   answers.
5. **Measure**, with the same prompts as [3060M.md](3060M.md)'s Halo comparison (`benchmarks/2026-10-02-3060m-bench_halo.py`),
   so this PC gets Strata-vs-llama.cpp numbers on the same hardware for the first time: decode with MTP, 4.75K and
   16K prefill, load time. Then UD-IQ4_XS the same way.
6. **Then** `STRATA_HIP_WMMA` for gfx11's WMMA layout is the one kernel-level gain to look at (the RDNA4 kernel gave
   7x on prompt attention); not before the baseline exists.

## Execution log (2026-10-05, the same afternoon)

The plan above was then carried out on this PC; what each step gave.

**Step 1, the port** (branch `AIMAX395-ROCm`): gfx1151 and gfx1150 in `cmake/hip_backend.cmake`'s unvalidated list
(the configure prints "gfx1151 builds, but it is not validated on a real card yet"), the two arch macros in
`intrinsics.hpp`'s dot4 condition, setup's `AMD_ARCHS`, `AMD_NAMES`, `AMD_CARDS`, `ROCM_INDEXES` (TheRock has a
`gfx1151` family index on Linux, checked) and the Windows PCI id `0x1586` plus a `8060S / 8050S` name rule;
`build_windows.bat` takes `STRATA_ROCM_ROOT` (an installed ROCm instead of its own venv) and has gfx1151 in its
default arch list. `tools/test_setup_amd.py` (18 tests) and `tools/test_setup_choices.py` (15) pass.

**Step 2, the build.** A new conda env `strata` (Python 3.14.8, `C:\conda_envs\strata`) took every pinned package of
`requirements.txt` (numpy 2.5.3 included) and gave `cmake`/`ninja`; the ROCm was the untouched `rocm100-py312` env's
root through `STRATA_ROCM_ROOT`; `STRATA_GGML_DIR` pointed at `third_party/llama.cpp` (setup's `get_llama_cpp()`
zip) because CMake's own git clone into `build-hip-win` on the exFAT disk failed with git's "dubious ownership".
`tools\hip\build_windows.bat tests`: 255 ninja steps, `strata.exe` (15.6 MB), `strata-device.exe`, the test
programs, no source error. Two fixes to `tools/hip/package_windows.py` on the way: the ROCm 10.0.0 wheels keep
rocBLAS's kernels in `rocblas/library/gfx1151/` (a folder per arch; the 10.2 nightlies the script was written for
put the files side by side), and the script read `CMakeLists.txt` with the locale's codec (GBK here) and fell over
its non-ASCII comment: now `encoding="utf-8"`. The zip: 114 MiB (269 unpacked), gfx1151, ROCm 10.0.0, hipBLASLt
1.4.1, unpacked into `engine\`. With only `engine\rocm\bin` on the PATH:

```
device 0: AMD Radeon(TM) 8060S Graphics
  arch gfx1151, 107.9 GiB, wave32
```

and `strata-device --selftest` passes ("HIP arch gfx1151 wave32 (compiled for gfx1151)", VRAM 107.9 GiB total /
107.7 free, the 20480-context plan FITS).

**Step 3, ctest** (55 tests, `build-hip-win\ctest.log`): 48 pass, 2 skip, 5 fail. The skips are the gfx12-only
WMMA attention and the hipBLASLt table test (no table for gfx1151). Four failures are the known ones:
`hip_handoff` (the Windows mapped-pointer alias, see above), `ple_parity` (needs the Q2_0 model fixture),
`expert_parity` and `pool_test` (need `pack/full/experts.bin`). **One is new and real on this card:**
`hip_prefill_mmq_parity` fails with "synthetic-Q2_0-GU-pass0: non-finite or unwritten MMQ output": ggml's MMQ
prefill kernels, compiled for gfx1151, do not write their output here. The engine uses that path only when
`STRATA_PREFILL_MMQ=1` is set at run time (off by default: `src/prefill/prefill.cpp:476`), so the model runs below
are not affected; it is the first gfx1151-specific defect, to be looked at after the baseline. Everything else -
the HIP intrinsics, the router, the native QSA score, the expert cache staging, the hipBLAS prefill batch, the
sampler, the KV cache modes, the GDN and GR kernels - passes on the 8060S.

**Step 4, the Coder end to end** (by hand, the [3060M.md](3060M.md) route, with the `strata` conda env; not
`START-HERE.bat`, which would make a `.venv` here). The pack from the GGUF already on C:
(`tools/iq_pack.py`, 4 s: 1079 tensors, 302 served natively, arena 1.37 GiB) into
`C:\Users\Wei Lu\Documents\Strata-data\packs\coder-iq1_m`; the MTP draft layer (`tools/mtp_fetch.py`, 4.9 GB
from the original checkpoint, then `mtp_pack.py --experts q2_0` and `mtp_rt.py`) into `...\Strata-data\mtp\rt`;
the config `strata-coder-iq1_m.json` (32K context, 8-bit KV, `--expert-cache auto`, `--prefill auto`, MTP with
`--spec 4`, and `--mmap-experts` with no `experts.bin`: the GGUF read in place) and `run-coder-iq1_m.ps1`.
The first start answered on `/v1/models` after 45 s. The engine's log:

- `expert cache auto: 101.40 GiB free ... -> 12288 slots`; `expert cache 12288 slots, 23.42 GiB of VRAM`;
  `pre-filled 12288 of 12288 slots from the profile`: **every expert of the Coder on the GPU**, as the memory
  section predicted; `78112 MiB of VRAM free with everything loaded`.
- `experts via mmap (--mmap-experts; the GGUF shards in place, no experts.bin)`; the prompt path borrows 1945
  cache slots (3.71 GiB); prompt chunk auto: 8192 tokens.
- `PCIe probe: 8666.2 GB/s host->device, host RAM read 54.0 GB/s -> pcie_frac 0.55`: the host-to-device figure is
  the APU artifact seen in the bandwidth probe (a copy the events cannot time). It does not matter here: every
  decode request logged `decode expert cache hit rate: 100.0%`, so no expert is ever computed by the CPU or
  copied in.

The test request of 3060M.md ("Write a Python function that checks whether a number is prime. Code only.",
temperature 0, `max_tokens` 600) gave a correct `is_prime` (trial division by odd numbers up to the square root)
after a short reasoning: 67 prompt tokens in 1.38 s, **245 tokens out in 6.83 s = 35.8 tokens/s**, 162 of 210
drafts accepted (77%), `finish_reason` stop. The RTX 3060 Laptop PC did this request at 41.0 tokens/s.

**Step 5, the same prompts as the llama.cpp measurement on this very PC**
(`benchmarks/2026-10-02-3060m-bench_halo.py`, now taking the EngramHalo.cpp checkout from `STRATA_ENGRAM_SRC`
and reading its sources as UTF-8; raw results `benchmarks/2026-10-05-halo-coder-iq1_m.json`, one session, n=1
per row, the server started once, temperature 0, 400 tokens out in the server rows). The Halo llama.cpp column
is windows.md's (quoted in 3060M.md); the RTX 3060 column is 3060M.md's.

| | **this PC, Strata** (8060S, all experts on the GPU) | this PC, llama.cpp (windows.md) | RTX 3060 Laptop 12 GB, Strata |
|---|---|---|---|
| model load to listening | **45 s** | 30 s | 11.5 s |
| cold first request, decode (MTP) | **36.8 t/s**, 89.7% acc | 26.1 t/s, 86.5% acc | 43.3 t/s, 88.1% acc |
| fresh code prompt, decode (MTP) | **34.7 t/s**, 81.9% acc | 21.9 t/s, 74.5% acc | 42.1 t/s, 80.0% acc |
| prefill @ ~4.75K | **146.1 t/s** (4749 tokens) | 321.5 t/s | 961.3 t/s |
| decode tail after that prefill | **30.2 t/s**, 71.3% acc | 20.8 t/s, 81.9% acc | 39.3 t/s, 77.5% acc |
| repeated prompt (reference only) | 39.1 t/s, 89.7% acc (`cache_n` 0: not reused, as on the 3060) | 49.9 t/s | 45.1 t/s |

| test | **this PC, Strata server** | this PC, llama-bench | RTX 3060 Laptop, Strata server |
|---|---|---|---|
| pp4096 @ d0 | **146.4** (4085 tokens) | 318.1 +/- 70.0 | 926.9 |
| tg128 @ d0 | **27.3** (MTP, 57.6% acc) | 20.92 (plain decode) | 36.2 (MTP, 61.4% acc) |
| pp4096 @ d16384 | **~137** (derived: 4037 tokens in 144.2 - 114.3 s); 141.6 over all 20421 | 269.9 +/- 1.2 | ~1126 (derived) |
| tg128 @ d16384 | **26.9** (MTP, 58.4% acc) | 17.98 (plain decode) | 37.4 (MTP, 65.3% acc) |

The same caveats as in 3060M.md: MTP here against plain decode in llama-bench's tg rows, the chat template here
against `/completion` there, the depth prefill derived from two whole-prompt requests. The answers are coherent
code reasoning on every row (the JSON keeps the last 300 characters of each).

**Reading it.** On the same hardware Strata decodes the Coder **1.3-1.6x faster than llama.cpp** (35-37 vs 22-26
t/s with MTP on both; 27 vs 21 t/s at tg128 where llama.cpp has no MTP), the first Strata-vs-llama.cpp numbers on
one machine. **Prefill is the problem: 141-146 t/s, 2.2x slower than llama.cpp here and 6.5x slower than the
RTX 3060 PC.** The log says why. The prompt path does not use the GPU-resident experts; per 8192-token chunk it
streams every expert of every layer from the expert source, which here is the GGUF in place: `expert tiers: ...
files 0 blobs 47584.5 MB read (the GGUF in place)` after the 16K prompt, 75.5 GB after the session. On a discrete
card that source is the pinned RAM arena (the R9700 read 4K prompts at ~1,000-1,800 t/s with the same 12,288
resident experts). Here the source is the OS file cache, and there is none: with the server up, Windows had
**1.2 GiB free and 0.6 GiB of standby cache** of its 31.6 GiB (the engine 22 GiB working set, 26 GiB private),
so each chunk's 23.4 GB come from the NVMe again, in the in-place mode's three reads per expert: 47.6 GB in 114 s
is 416 MB/s against the drive's 2.6 GB/s sequential. Decode is untouched because it needs no expert from the
source (100% hits).

**Not tried: the default mode (the arena pinned in RAM, no `--mmap-experts`).** With the engine stopped this PC
has 23.6 GiB free; the Coder's arena is 23.4 GiB plus the engine's other ~2.5 GB. `strata-coder-iq1_m-arena.json`
is that config, kept for after the fix below. **The fix is the BIOS split**: 64 GiB for the GPU and 64 GiB for
Windows (or 80/48) gives the file cache, or the pinned arena, the room the prompt path needs, while 64 GiB of
GPU memory still holds every expert of every size up to IQ3_S (50.3 GB + KV). That is the opposite of the
"96 GiB side holds every model" reasoning above, which was right for decode and missed the prompt path's source.
Until the BIOS is changed, the prompt path on this PC is NVMe-bound. A second route, in the engine, would be a
prompt path that takes resident experts from the cache instead of the source; that is a code change with its own
measurement, not a setting.

## The RTX 5090 over Thunderbolt (not pursued)

Windows lists an RTX 5090 (32 GB) as an external card that was attached before. With it attached, Strata's
ready-made CUDA engine runs without any port, in the low-RAM resident mode (a 32 GB card holds all of the Coder's
experts, most of Q2_0's, [DETAILS.md](DETAILS.md)). Its limits are the 31.6 GiB of system RAM for everything the
card does not hold and the Thunderbolt link (about 3 GB/s, a tenth of PCIe x16 Gen4: `--pcie-frac` near 0). It is
the easier route to a running Strata on this PC and the less interesting one; the question of this page is the
8060S.
