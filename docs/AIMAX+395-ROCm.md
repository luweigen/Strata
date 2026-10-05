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
expert on the GPU, decodes 1.3-1.6x faster than llama.cpp on this same PC (35-40 vs 22-26 tokens/s with MTP), and
read prompts 2.2x slower at first (146 vs 321 tokens/s). The BIOS split, a hipBLASLt switch and the engine's own
calibrated hipBLASLt table for gfx1151 took that to 434-451 tokens/s, 1.4x llama.cpp
([the evening's section](#a-better-gemm-what-was-searched-what-was-measured-what-it-gave-2026-10-05-evening)).

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
showed one: the prompt path streams the non-resident experts from the host-side source for every chunk, and in
the in-place mode on 31.6 GiB that source had no file cache and read 47.6 GB per 16K prompt (see
[Step 5](#execution-log-2026-10-05-the-same-afternoon)). For decode the reasoning holds.

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
are not affected; it is the first gfx1151-specific defect, to be looked at after the baseline (it was not one: the
test's own stream order, [TODO 3 done](#todo-3-done-the-mmq-parity-failure-was-the-tests-stream-order-not-the-kernels-2026-10-05-night);
and TODO 2 found the path on by default). Everything else -
the HIP intrinsics, the router, the native QSA score, the expert cache staging, the hipBLAS prefill batch, the
sampler, the KV cache modes, the GDN and GR kernels - passes on the 8060S.

**Step 4, the Coder end to end** (by hand, the [3060M.md](3060M.md) route, with the `strata` conda env; not
`START-HERE.bat`, which would make a `.venv` here). The pack from the GGUF already on C:
(`tools/iq_pack.py`, 4 s: 1079 tensors, 302 served natively, arena 1.37 GiB) into
`C:\Users\Wei Lu\Documents\Strata-data\packs\coder-iq1_m`; the MTP draft layer (`tools/mtp_fetch.py`, 4.9 GB
from the original checkpoint, then `mtp_pack.py --experts q2_0` and `mtp_rt.py`) into `...\Strata-data\mtp\rt`;
the config `strata-coder-iq1_m.json` (32K context, 8-bit KV, `--expert-cache auto`, `--prefill auto`, MTP with
`--spec 4`, and `--mmap-experts` with no `experts.bin`: the GGUF read in place) and `docs/benchmarks/2026-10-05-run-coder-iq1_m.ps1`.
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
RTX 3060 PC.** The log says why. Per 8192-token chunk the prompt path streams the experts it does not find in the
GPU cache from the expert source (`prefill.cpp`: "every non-resident expert of every layer"; the cache lends it
1,945 of its slots for buffers, so those experts are among the streamed), and in this run the source was the GGUF
in place: `expert tiers: ... files 0 blobs 47584.5 MB read (the GGUF in place)` after the 16K prompt, 75.5 GB
after the session - a volume that says the file tier was read far beyond the lent slots' experts (the routing
prefetch of whole layers is a candidate; not isolated). On a discrete card that source is the pinned RAM arena
(the R9700 read 4K prompts at ~1,000-1,800 t/s with the same 12,288 resident experts). Here the source was the OS
file cache, and there was none: with the server up, Windows had
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

## After the BIOS change: 64 GiB / 64 GiB (2026-10-05, later)

The user set the BIOS split to 64/64 and rebooted. Windows now sees 63.6 GiB (50.7 free at idle), the registry
64 GiB dedicated for the GPU, HIP 99.7 GiB total. The default mode (`strata-coder-iq1_m-arena.json`: no
`--mmap-experts`, the experts pinned in RAM) started in **15 s**: `expert arena: cudaHostRegister PORTABLE ok;
large pages refused ... using 4 KB pages`, `loaded 23.42 GiB at 5.94 GiB/s`, `expert cache auto: 46.43 GiB free ->
12288 slots` (every expert on the GPU again), 22.7 GiB of VRAM left; the engine's working set 24.5 GiB, 22.5 GiB
of RAM free with it running. The same benchmark (`benchmarks/2026-10-05-halo-coder-iq1_m-arena.json`):

| | 96/32, mmap (above) | **64/64, arena (pinned RAM)** | this PC, llama.cpp | RTX 3060 Laptop, Strata |
|---|---|---|---|---|
| model load to listening | 45 s | **15 s** | 30 s | 11.5 s |
| cold first request, decode | 36.8 t/s | **35.0 t/s**, 88.7% acc | 26.1 | 43.3 |
| fresh code prompt, decode | 34.7 t/s | **33.0 t/s**, 81.4% acc | 21.9 | 42.1 |
| prefill @ ~4.75K | 146.1 t/s | **208.7 t/s** | 321.5 | 961.3 |
| decode tail after it | 30.2 t/s | **31.3 t/s** | 20.8 | 39.3 |
| pp4096 @ d0 | 146.4 | **208.8** | 318.1 | 926.9 |
| tg128 @ d0 (MTP here) | 27.3 | **26.5** | 20.92 | 36.2 |
| 16384-token prefix | 143.3 | **213.3** | | |
| pp4096 @ d16384 | ~137 (derived) | **~204** (derived: 4037 tokens in 96.6 - 76.9 s); 211.4 over all 20421 | 269.9 | ~1126 |
| tg128 @ d16384 (MTP here) | 26.9 | **27.5** | 17.98 | 37.4 |

So the file-cache starvation was worth 1.45x on prompts (146 -> 209-214 t/s) and nothing on decode, as
expected. Prompts are still 1.5x slower than llama.cpp on this PC. Three more measurements say why, and it is not
the transfer:

- **Host-to-device copies are fast here** (`docs/benchmarks/2026-10-05-halo-hip-h2d.hip`, host-clock timed,
  2 GiB): `hipMemcpy` from pinned memory 63-74 GB/s, from pageable 19.5 GB/s, `hipMemcpyAsync` + stream sync
  64-72 GB/s, a kernel reading mapped host memory into device memory 94-100 GB/s, 512 copies of 2.66 MB (one
  Coder expert blob each) 71.5 GB/s. The earlier "0 ms" event timing of H2D was the event, not the copy. A chunk's
  23.4 GB of experts take about 0.35 s to move; the prompt path spends 38 s on a chunk.
- **The prompt path's own phase timing** (`STRATA_PREFILL_TIMING=1`, a 4164-token prompt at 207 t/s, GPU timeline
  19,955 ms, host work under 300 ms): hc read 3,438 ms (17.2%), GDN 4,969 (24.9%: out proj 1,948, recurrence 799,
  conv+gates 128), QSA proj 2,077 (10.4%), QSA attn 1,986 (10.0%), router+shared 909 (4.6%), the expert GEMMs
  gate/up 1,772 (8.9%) + down 785 (3.9%) + dequant 489 (2.4%), gather 89, **wait copy 34 (0.2%)**, combine 225,
  PLE 187. The experts' streaming is 0.6% of the time. The time is in the dense parts: the hipBLAS GEMMs (QSA and
  GDN projections, the router, the expert GEMMs: about 7.5 s of 20) and the engine's own kernels for the
  hyper-connection read, the GDN recurrence and the FP32 prompt attention (about 8 s).
- **hipBLAS on gfx1151 runs at 2.0-2.6 TFLOP/s; routed to hipBLASLt it runs at 4.0** (the GEMM probe at the
  engine's shapes, T = 512 and 8192, BF16 and FP16: `ROCBLAS_USE_HIPBLASLT=1`, an environment variable rocBLAS
  honours, no code change; results correct, 3e-6 relative). The engine's `cublasGemmEx` calls go through hipBLAS ->
  rocBLAS, so this switch reaches every dense GEMM of the prompt path. The 8060S's matrix-core peak is far above
  either number: rocBLAS's gfx1151 kernels are plainly not tuned, hipBLASLt's less so.

**With `ROCBLAS_USE_HIPBLASLT=1` on the server** (the arena config, the same 4K prompt sent twice with
`benchmarks/2026-10-05-halo-pp4k.py`, 1 token out):

| | arena, plain hipBLAS | **arena, rocBLAS -> hipBLASLt** |
|---|---|---|
| pp4096 (4164-4171 tokens) | 199.4, 206.6 t/s | **264.2, 279.2 t/s** (+33%) |
| GPU timeline of the prompt | 19,955 ms | **14,725 ms** |
| hc read / GDN / QSA proj / router | 3,438 / 4,969 / 2,077 / 909 ms | 1,768 / 3,263 / 1,443 / 591 ms |
| QSA attn / expert GEMMs gate-up + down / GDN recurrence | 1,986 / 1,772 + 785 / 799 ms | 1,944 / 1,705 + 763 / 835 ms (unchanged: not rocBLAS calls) |
| the prime request, decode | 35.9 t/s, 162 of 212 drafts, the same `is_prime` | the same |

So on this card the switch is worth a third of the prompt speed and costs nothing; `strata-coder-iq1_m.json` now
carries it (`"env": {"ROCBLAS_USE_HIPBLASLT": "1"}`, the arena mode, no `--mmap-experts`) and is what
`docs/benchmarks/2026-10-05-run-coder-iq1_m.ps1` starts; `strata-coder-iq1_m-arena.json` is the measured variant without the switch. What
remains of the 14.7 s is the engine's own kernels on RDNA 3.5: the FP32 prompt attention (1.9 s; the RDNA4 WMMA
kernel is 7x faster on gfx12 and gfx11.5 has WMMA with gfx11's layout), the GDN recurrence (0.8 s), the
hyper-connection read (1.8 s), and the expert GEMMs (2.5 s, not through rocBLAS). Those are kernel work for
later, each with this phase timing to measure against.

**The whole sequence again with the final configuration** (pinned arena, hipBLASLt routing; the server started
once; `benchmarks/2026-10-05-halo-coder-iq1_m-arena-hipblaslt.json`): decode 38.6 / 35.1 / 34.7 t/s on the three
server rows (86.2 / 86.5 / 80.5% drafts accepted), 39.5 on the repeated prompt; prefill 274.0 t/s at 4.75K,
268.4 at pp4096, 277.1 on the 16,384-token prefix, 273.7 over 20,421 tokens (~260 derived for the 4,037 at depth:
4037 / (74.76 - 59.24) s), 276.7 before tg128 at depth; tg128 28.5 t/s at depth 0 and 28.8 at 16K (60-66%
accepted). Prefill is now flat from 4K to 20K: compute-bound, not memory- or transfer-bound.

## Against 3060M.md: the same prompts on the RTX 3060 Laptop PC

[3060M.md](3060M.md) ran this benchmark (and the prime request) with the same Coder IQ1_M, the same 32K / 8-bit
KV / `--spec 4` configuration, on 2026-10-02 to 04. The two machines, as measured in their pages:

| | this PC (EVO-X2) | the 3060M PC |
|---|---|---|
| GPU | Radeon 8060S, 40 CUs RDNA 3.5, integrated | RTX 3060 Laptop GPU, 30 SMs Ampere, 80 W, on a desktop card |
| GPU memory | 64 GiB of the shared LPDDR5X; **236 GB/s measured** (one pool for CPU and GPU) | 12 GB GDDR6, 336 GB/s spec; PCIe 4.0 x16, 26.8 GB/s measured |
| host | 16 Zen 5 cores, 63.6 GiB of the same memory | 16 Zen 4 cores (AVX-512), 92 GiB DDR5-5200, 55 GB/s measured |
| the Coder's experts | **all 12,288 on the GPU**; the CPU computes none | 2,666 on the GPU (5.1 GiB); the CPU computes the rest from RAM, `--pcie-frac` 0.55 (0.2 later) |
| engine | HIP (ROCm 10.0.0, hipBLAS -> hipBLASLt), Windows 11 | CUDA 13.0, Linux |

The numbers (MTP on in every decode row; prefill 1 token out; n = 1 per row on both):

| | **8060S, Strata (final config)** | RTX 3060 Laptop, Strata (3060M.md) | ratio |
|---|---|---|---|
| model load to listening | 15 s | 11.5 s | |
| cold first request, decode | 38.6 t/s, 86.2% acc | 43.3 t/s, 88.1% acc | 0.89 |
| fresh code prompt, decode | 35.1 t/s, 86.5% acc | 42.1 t/s, 80.0% acc | 0.83 |
| prefill @ ~4.75K | 274.0 t/s | 961.3 t/s | **0.29** |
| decode tail after that prefill | 34.7 t/s, 80.5% acc | 39.3 t/s, 77.5% acc | 0.88 |
| repeated prompt (reference) | 39.5 t/s | 45.1 t/s | 0.88 |
| the prime request, decode | 35.9 t/s (162 of 212 drafts) | 41.0 t/s (164 of 197) | 0.88 |
| pp4096 @ d0 | 268.4 t/s | 926.9 t/s | **0.29** |
| tg128 @ d0 | 28.5 t/s, 60.4% acc | 36.2 t/s, 61.4% acc | 0.79 |
| 16,384-token prefix | 277.1 t/s | ~1,065 t/s (the 16K prefill of the `--pcie-frac` table) | **0.26** |
| pp4096 @ d16384 (derived on both) | ~260 t/s | ~1,126 t/s | **0.23** |
| tg128 @ d16384 | 28.8 t/s, 66.3% acc | 37.4 t/s, 65.3% acc | 0.77 |
| decode at the 3060's best `--pcie-frac` (0.2; a 274-token prompt, median of 3) | 38.6 (the cold row above, a comparable prompt) | 62.3 t/s | 0.62 |

**Reading it.**

- **Decode is close: 0.8-0.9x** of the RTX 3060 PC as 3060M.md's tables measured it, 0.6x of that PC's best
  setting (`--pcie-frac 0.2`, found two days later; its token-weighted BFCL rate was 54 t/s). This although the
  two machines do the work very differently: the 3060 holds a fifth of the experts and its 16 Zen 4 cores compute
  the other four fifths from 55 GB/s of RAM; the 8060S holds every expert and runs the whole token on one GPU
  with 236 GB/s. The draft acceptance is the same (80-87% here, 77-88% there), so the per-round cost is what
  differs.
- **Prefill is 0.23-0.29x**, and flat with length here (268-277 t/s from 4K to 20K) where the 3060 rises from
  927 to ~1,100. The 3060 PC's prefill streams the experts over PCIe (26.8 GB/s) and runs the dense GEMMs on
  Ampere tensor cores through cuBLAS; here the streaming is free (0.2% of the time) and the GEMMs and the engine's
  own kernels are the whole cost: hipBLASLt at 4.0 TFLOP/s on gfx1151, FP32 prompt attention (the tensor-core
  attention the 3060 uses has no gfx11 twin yet), the GDN recurrence and the hyper-connection read. The phase
  timing above is the work list.
- **Load:** 15 s against 11.5 s; the 23.4 GiB of experts came off the NVMe at 5.9 GiB/s here (the page cache,
  after the earlier starts) and 2.86 GiB/s there.
- **Not compared:** UD-IQ4_XS (3060M.md: 33-39 t/s decode, 663-819 t/s prefill) was not run here yet; its 55.4
  GiB of experts fit the 64 GiB GPU side with the KV cache (55.4 + ~5), but its pinned arena does not fit the
  63.6 GiB host: with the Coder the host side used 12.9 GiB idle (OS, RDP, the conda tools) plus the engine's 1.1
  GiB beside its arena, so UD-IQ4_XS would need 55.4 + 14 = ~69 GiB of RAM for the default mode. No split of 128
  GB gives both ~61 GiB of GPU memory and ~69 GiB of RAM. The choices with the engine as it is: **GPU 48 / RAM 80**
  (the arena pinned and the prompt path at full speed, but the GPU holds ~43 of the 55.4 GiB of experts and the
  CPU computes the rest, as on the 3060 PC: slower decode, measurable), or 64/64 with the mmap mode (every expert
  on the GPU, decode as fast as the Coder's, but the prompt path re-reads 55.4 GB per chunk through a page cache
  that is ~5 GB too small: the 96/32 problem again). GPU 80 / RAM 48 is the worst of the three: the RAM holds
  neither the arena nor the page cache. The real fix is the prompt path using the GPU-resident experts; then
  80/48 or even 96/32 becomes the right split. The BFCL runs of 3060M.md (hours each) were not repeated.
- The 3060M.md comparison with llama.cpp on this Halo was 2x decode / 3x prefill in the 3060's favour; with Strata
  on the Halo itself the decode gap to the 3060 is 1.1-1.2x and the prefill gap 3.4-4.3x.

## The 3060's way on this PC: GPU 32 GiB / RAM 96 GiB, a partial expert cache, the CPU computes the rest

This BIOS offers 32, 64 and 96 GiB for the GPU, nothing between. The user set **32/96** (Windows sees 95.6 GiB,
HIP 89.4 GiB), and the Coder was run the way the 3060M PC runs it: the arena pinned in RAM, `--expert-cache 2666`
(the 3060's 2,666 resident experts; the engine made it 3,465 slots, 6.6 GiB, adding the slots the prompt path
borrows), `--pcie-frac 0.55` (the value of 3060M.md's tables) and then `--pcie-frac 0.2` (the value its
`--pcie-frac` section found best there). hipBLASLt routing on; configs `strata-coder-iq1_m-like3060.json` and
`...-pcie02.json`; the same benchmark; the engine's 15 expert-pool workers compute the misses.

| | **32/96, cache 3,465, pcie 0.55** | 32/96, cache 3,465, pcie 0.2 | 64/64, all 12,288 on the GPU | RTX 3060 Laptop (cache 2,666, pcie 0.55) |
|---|---|---|---|---|
| model load to listening | 15 s | 10 s | 15 s | 11.5 s |
| cold first request, decode | **29.0 t/s**, 91.8% acc | **28.0 t/s**, 86.5% acc | 38.6 | 43.3 |
| fresh code prompt, decode | **28.5 t/s**, 84.2% acc | **26.9 t/s**, 80.8% acc | 35.1 | 42.1 |
| prefill @ ~4.75K | 253.4 t/s | 258.5 t/s | 274.0 | 961.3 |
| decode tail after it | **24.2 t/s**, 74.3% acc | **25.3 t/s**, 75.5% acc | 34.7 | 39.3 |
| repeated prompt (reference) | 30.5 t/s | 29.9 t/s | 39.5 | 45.1 |
| pp4096 @ d0 | 256.7 t/s | 251.2 t/s | 268.4 | 926.9 |
| tg128 @ d0 | **23.3 t/s**, 68.1% acc | **23.3 t/s**, 66.7% acc | 28.5 | 36.2 |
| 16,384-token prefix | 265.3 t/s | 262.4 t/s | 277.1 | ~1,065 |
| pp4096 @ d16384 | 260.8 over 20,421 | 258.5 over 20,421 | 273.7 | ~1,126 (derived) |
| tg128 @ d16384 | **21.1 t/s**, 58.6% acc | **22.1 t/s**, 64.9% acc | 28.8 | 37.4 |
| decode expert-cache hit rate | 81.5-84.4% (400-token answers) | 81.2% | 100% | 64.6% |

At `--pcie-frac 0.55`: decode drops to 0.70-0.82x of the all-on-GPU run (29.0 / 28.5 / 24.2 vs 38.6 / 35.1 /
34.7) with 81-84% of the lookups still hitting the GPU (a bigger cache than the 3060's, so a higher hit rate than
its 64.6%); prefill is unchanged within noise (253-271 vs 268-277 t/s: streaming the 72% non-resident experts from pinned RAM
costs little, `wait copy` grew from 33 to 318 ms of a 15.7 s prompt). Against the 3060 itself: 0.62-0.69x on
decode with the same placement. **`--pcie-frac 0.2` changes nothing here** (28.0 / 26.9 / 25.3 and 23.3 / 22.1
t/s: every row within 1-2 t/s of 0.55, both ways), where on the 3060 PC it gained 20%. That gain came from the
PCIe link (26.8 GB/s) and the CPU's RAM (55 GB/s) being separate, finite resources that 0.2 balanced; here the
copies and the CPU's reads come out of the same 236 GB/s the GPU is using, and copying a missed expert costs
about what computing it on the CPU costs. The misses themselves cost more here than there, on a CPU of the same
core count, for the same reason: the 16 Zen 5 cores and the GPU share one memory. The placement that pays on
this APU is all experts on the GPU, which needs the 64 GiB (or larger) carve-out.

## UD-IQ4_XS on the 32/96 split

The pack as in 3060M.md (`tools/iq_pack.py --compat-bf16`, 25 s: 1079 tensors, 303 served natively, 195 tensors
to BF16, `native_experts.txt` v4 for layer 14's split roles), the Coder's config with the pack, shard 1 as
`--native` (no `--ple-gguf`: the engine found the PLE table in shard 2), the original model's
`data/expert-profile.bin`, the same draft layer.

**The first start failed, and the failure is a property of this APU worth knowing.** With `--expert-cache auto`
the engine pinned the 55.4 GiB arena (`cudaHostRegister PORTABLE ok`, loaded at 2.5 GiB/s) and then found
`expert cache auto: 0.00 GiB free ... -> 0 slots` and stopped; with `--expert-cache 1864` the same (the count is
clamped to the free figure). `docs/benchmarks/2026-10-05-halo-hip-pinfree.hip` measured what the engine saw:
**registering N GiB of mapped pinned host memory lowers `hipMemGetInfo`'s free figure by 2N GiB** (16 GiB pinned:
89.2 -> 57.2 free; 40 GiB: 89.2 -> 9.2), and the figure is a budget, not the memory: a 20 GiB `hipMalloc` with
9.2 GiB "free" succeeded and a kernel ran over it (free then read 0.0). The engine knows the mechanism from a
discrete-GPU Windows PC (#243 in `src/core/pinned.cu`: page-locked memory the GPU maps is charged to WDDM's
shared segment, about half the RAM) and has the remedy, `STRATA_ARENA_PIN_GIB=N`: only N GiB of the arena are
pinned, the rest stays pageable and is read by the CPU as before (its streamed copies go through the pinned
staging ring). The Coder's 23.4 GiB arena never hit this because 2 x 23.4 fit beside the cache in every split.
So for 32/96: `STRATA_ARENA_PIN_GIB=24` (costing 48 of the 89.4 GiB figure) and an explicit cache of 10,000
slots (about 24 GiB, inside the 32 GiB carve-out: `auto` would have sized itself from the figure, past the
dedicated memory into WDDM's shared pages), and the 3060-like 1,864 slots with `--pcie-frac 0.55`.

Both started (35 s and 25 s to listening; the arena "locked 33053 MiB via working-set minimum + VirtualLock,
cudaHostRegister limited to 24 GiB"; with the cap the PCIe probe read a real number, 68.8 GB/s, where the fully
mapped arena had given it 8-16 TB/s). The engine made the caches **14,358 slots (32.4 GiB, 1.3 GiB of VRAM left)**
and **2,683 slots (6.0 GiB)**: the asked-for count plus the prompt path's borrowable slots. The same benchmark
(`benchmarks/2026-10-05-strata-unsloth-ud-iq4_xs.json`, `...-cache1864.json`):

| UD-IQ4_XS, MTP on | **32/96, 14,358 slots** | **32/96, 2,683 slots, pcie 0.55** | RTX 3060 Laptop (1,864 slots) | 8060S llama.cpp (windows.md) |
|---|---|---|---|---|
| model load to listening | 35 s | 25 s | 42 s | 83 s |
| cold first request, decode | 31.4 t/s, 91.8% acc | 32.5 t/s, 93.3% acc | 37.6 t/s, 87.9% acc | 28.2 t/s |
| fresh code prompt, decode | 27.2 t/s, 84.8% acc | 25.1 t/s, 77.0% acc | 38.7 t/s, 87.1% acc | 25.1 t/s |
| prefill @ ~4.75K | 210.5 t/s | 213.7 t/s | 662.8 t/s | 277.2 t/s |
| decode tail after it | 22.1 t/s, 70.3% acc | 22.6 t/s, 72.9% acc | 33.1 t/s, 76.9% acc | 23.0 t/s |
| repeated prompt (reference) | 30.1 t/s | 31.1 t/s | 42.6 t/s | 58.6 t/s |
| pp4096 @ d0 | 210.4 t/s | 208.4 t/s | 819.4 t/s | 339.5 |
| tg128 @ d0 | 20.6 t/s, 59.6% acc | 21.3 t/s, 62.0% acc | 32.2 t/s, 71.4% acc | 21.83 (plain) |
| 16,384-token prefix | 230.2 t/s | 231.7 t/s | | |
| pp4096 @ d16384 | 228.7 over 20,421 | 228.0 over 20,421 | ~949 (derived) | 244.1 |
| tg128 @ d16384 | 20.2 t/s, 60.6% acc | 19.8 t/s, 55.4% acc | 31.7 t/s, 67.4% acc | 16.25 (plain) |
| decode expert-cache hit rate | **96.8-98.3%** | **59-72%** | 48.3% | |

**Reading it.**

- UD-IQ4_XS decodes at 22-32 t/s here against 33-39 on the 3060 PC (0.67-0.84x) and 23-28 for llama.cpp on
  this same PC (about even to 1.1x; llama.cpp's repeated-prompt row, 58.6, is its n-gram speculator). Prefill
  210-236 t/s is 0.24-0.32x the 3060 PC and 0.75-0.95x llama.cpp here.
- **The cache size made no difference to decode**: 14,358 resident experts with 97-98% hits and 2,683 with 59-72%
  hits give the same tokens/s on every row (31.4 vs 32.5, 27.2 vs 25.1, 22.1 vs 22.6; tg128 20.6 vs 21.3). With
  the Coder on this PC the hit rate did matter (100%: 35-39 t/s; 82%: 24-29). So UD-IQ4_XS's decode on the 8060S
  is bound by something the GPU-resident path and the CPU path share, not by where the experts are: a candidate
  is the GPU's own expert kernels for this file's formats (IQ3_S, IQ4_NL and Q8_0 blobs, 2.4 MB per pair against
  the Coder's 2.0 MB) on RDNA 3.5, another the per-token PLE rows (read from shard 2 on the NVMe by both paths).
  `STRATA_DECODE_TIMING=1` is the next measurement; not run yet.
- The prefill is 15-20% slower than the Coder's at the same split (210-236 vs 253-271 t/s): the prompt path
  streams 55.4 GB per chunk instead of 23.4, and `--compat-bf16`'s dense projections are BF16 GEMMs.
- **Later the same night**, with the calibrated hipBLASLt table and the gfx11 WMMA attention (and the same pin
  cap and 10,000-slot cache): prefill 324 / 312 / 350 / 342 / 353 t/s on the 4.75K / pp4096 / 16K / 20K rows,
  decode 29.7 / 26.7 / 22.4 and tg128 19.9 / 21.7 t/s, 98-99% hits (`benchmarks/2026-10-05-strata-unsloth-ud-iq4_xs-table-pa-wmma.json`).
  Prefill +50%, decode unchanged: the decode question above stands.

## One table: the best Halo setup per model against the 3060M PC

The best combination of memory split and engine measured on this Halo for each model (Strata from this page;
llama.cpp from EngramHalo.cpp's `docs/strix-halo/windows.md`, its 2026-09-02 and 2026-10-01 runs at 96/32 and
its 2026-10-04 chunked-GDN prefill kernel), beside 3060M.md's RTX 3060 Laptop PC (Strata, CUDA). MTP on in
every decode row (llama.cpp with the EasiiX MTP sidecar; its tg128 rows are plain decode). Where the other Halo
engine wins a row, its number is in brackets.

| Coder IQ1_M / UD-IQ4_XS | **Coder: Halo best = Strata, 64/64, every expert on the GPU, the gfx1151 hipBLASLt table** | Coder: 3060M PC | **UD-IQ4_XS: Halo best = llama.cpp, 96/32, chunked GDN kernel** | UD-IQ4_XS: 3060M PC |
|---|---|---|---|---|
| model load to listening | **15 s** | 11.5 s | 83-94 s (Strata 32/96: 25-35 s) | 42 s |
| cold first request, decode | **39.9 t/s** (llama.cpp: 26.1) | 43.3 t/s | 28.1-30.4 t/s (Strata 32/96: 31.4-32.5) | 37.6 t/s |
| fresh code prompt, decode | **36.1 t/s** (llama.cpp: 21.9) | 42.1 t/s | 25.1-30.7 t/s (Strata: 25.1-27.2) | 38.7 t/s |
| prefill @ ~4.75K | **451.0 t/s** (llama.cpp: 321.5) | 961.3 t/s | **322 t/s** (278-284 before the kernel; Strata: 210-214) | 662.8 t/s |
| decode tail after that prefill | **30.9 t/s** (llama.cpp: 20.8) | 39.3 t/s | 21.3-25.1 t/s (Strata: 22.1-22.6) | 33.1 t/s |
| repeated prompt (reference) | 40.1 t/s (llama.cpp: 49.9, n-gram drafts) | 45.1 t/s | 58.6 t/s (Strata: 30-31) | 42.6 t/s |
| pp4096 @ d0 | **440.9 t/s** (llama.cpp: 318.1) | 926.9 t/s | **372-391 t/s** (314-349 before; Strata: 208-210) | 819.4 t/s |
| tg128 @ d0 | **28.2 t/s** (llama.cpp: 20.9 plain) | 36.2 t/s | 21.8-22.8 plain (Strata: 20.6-21.3 MTP) | 32.2 t/s |
| pp4096 @ d16384 | **~430 t/s** (433.5 over 20,421; llama.cpp: 269.9) | ~1,126 t/s | **273-281 t/s** (244-260 before; Strata: 228-229 over 20K) | ~949 t/s |
| tg128 @ d16384 | **29.2 t/s** (llama.cpp: 18.0 plain) | 37.4 t/s | 16.3-19.4 plain (Strata: 19.8-20.2 MTP) | 31.7 t/s |
| ratio to the 3060M PC, decode | 0.8-0.9 | 1 | 0.7-0.8 | 1 |
| ratio to the 3060M PC, prefill | 0.38-0.48 | 1 | 0.3-0.5 | 1 |

(The Coder column is the evening's configuration with the calibrated hipBLASLt table, see
[A better GEMM](#a-better-gemm-what-was-searched-what-was-measured-what-it-gave-2026-10-05-evening); before it,
prefill was 268-277 t/s.) What the table says: for the Coder the Halo's best is Strata with the whole model on the
GPU, which puts its decode within 0.8-0.9x of the 3060 PC while llama.cpp's is 0.5-0.6x, and its prefill 1.4-1.6x
llama.cpp's on this PC and 0.4-0.5x the 3060's. For UD-IQ4_XS no Halo setup gets the experts on the GPU with the engine as it is (the 32/96 and 64/64
limits of [the memory section](#ud-iq4_xs-on-the-3296-split) and the BIOS's three choices), decode is a tie
between the two engines at 0.7-0.8x of the 3060, and llama.cpp's prefill is 1.3-1.8x Strata's. The prefill gap
to the 3060 is the same engine problem in both models: RDNA 3.5 GEMMs and kernels, not memory.

## Which GEMMs and kernels the prompt path's time goes to

The phase timing of the 4,165-token prompt with hipBLASLt routing (GPU timeline 14,725 ms, 207 -> 279 t/s after
the switch), mapped to the calls in `src/prefill/prefill.cpp` between its timing marks. The model's width N is
2,560, an expert's gate+up is 1,280 wide and its down 640; T is the chunk's tokens (up to 8,192), ne the tokens
routed to one expert. Every dense projection goes through `Gemm::bf16` / `Gemm::f16` (one `hipblasGemmEx` each,
`src/prefill/gemm.cu:388/407`), `Gemm::native` being a `dequant_f16` kernel of the quantized weight into FP16
scratch followed by the same FP16 GEMM.

| phase | ms | share | what runs | kind |
|---|---|---|---|---|
| gdn (projections in) | ~990 | 6.7% | `attn_qkv`, `attn_gate` (native: dequant + FP16 GEMM), `ssm_alpha`, `ssm_beta` (BF16 GEMM), 36 GDN layers | GEMM |
| gdn out proj | 1,320 | 9.0% | `ssm_out` (native) | GEMM |
| gdn recurrence | 835 | 5.7% | `gdn_recurrence`: the delta-rule state update, token after token, one wave per head (`gdn.cu` / `fused_gdn.cu`) | own kernel, serial in T |
| gdn conv+gates | 117 | 0.8% | `gdn_gates`, `gdn_conv` | own kernels |
| hc read | 1,768 | 12.0% | per layer and half (attn, ffn): `hc_*_down`, `hc_*_up`, `hc_*_inject` BF16 GEMMs, with `gr_norm_rs`, `gr_silu`, `gr_mix_r` around them (`fused_gr.cu`); 2 x 48 layers | GEMM + own kernels |
| qsa proj | 1,443 | 9.8% | `attn_k`, `attn_v`, `attn_q`, `attn_output` (native), `indexer.k_proj`, `indexer.q_proj` (BF16); `rms_rows`, `rope`, `gate_attn`, `kv_append`; 12 QSA layers | GEMM + small kernels |
| qsa attn | 1,944 | 13.2% | `qsa_prompt_attn_batch`: the portable FP32 prompt attention. The matrix-core version (`STRATA_HIP_WMMA`) is compiled for gfx12 only; gfx1151 has WMMA with gfx11's fragment layout and no kernel for it. On gfx1201 the WMMA kernel is 7.2-7.5x faster | own kernel |
| qsa select, indexer | 35 | 0.2% | `qsa_block_topk` | own kernel |
| router + shared | 591 | 4.0% | `ffn_gate_inp` (BF16 GEMM), `route`, shared expert `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` (native), `swiglu_pair`, the shared-gate BF16 GEMV | GEMM + kernels |
| dequant | 473 | 3.2% | `iq_dequant_gu_f16`, `iq_dequant_f16`: each streamed or resident expert's IQ blocks to FP16 (`iq_kernels.cu`) | own kernels |
| gemm gate/up | 1,705 | 11.6% | per expert `Gemm::f16`: [ne x 2560] x [2560 x 1280], then `swiglu_interleaved`; about 512 GEMMs per layer with ne of a few dozen to a few hundred rows | GEMM, small M |
| gemm down | 763 | 5.2% | per expert `Gemm::f16`: [ne x 640] x [640 x 2560] | GEMM, small M |
| combine | 225 | 1.5% | `moe_combine` | own kernel |
| gather, wait copy, host grouping | 152 | 1.0% | the expert stream (resident experts are computed from their cache slot; the non-resident ones copied from the arena) | copies |
| ple, embed, kv stage | 93 | 0.6% | `ple_block`, embedding | own kernels |
| unattributed | ~2,270 | 15% | gaps between marks (launch overhead of the many small kernels and GEMMs, stream waits) | |

In sums: the hipBLAS GEMMs of the dense layers and the expert GEMMs are 7.5-8 s of the 14.7 s; the engine's two
sequential-shaped kernels, the FP32 prompt attention and the GDN recurrence, 2.8 s; small kernels and gaps the
rest. What the hipBLASLt switch did and did not do is in the same table's history: hc read 3,438 -> 1,768, GDN
projections 4,969 -> 3,263, QSA proj 2,077 -> 1,443, router 909 -> 591 (the big-M GEMMs halved), while gemm
gate/up and gemm down stayed at 1,772 -> 1,705 and 785 -> 763: at ne rows per expert hipBLASLt's gfx1151
kernels are no better than rocBLAS's, and 512 separate GEMM launches per layer is the shape ggml's MMQ path
(grouped, quantized, no dequant pass) was built for - the path that fails `hip_prefill_mmq_parity` on this card.
The work list, in the order of the time at stake: the expert GEMMs (2.9 s with their dequant: a grouped GEMM or
a fixed MMQ), the prompt attention on gfx11 WMMA (1.9 s), the dense GEMMs' library efficiency (4 TFLOP/s against
the card's matrix-core peak), the GDN recurrence (0.8 s; llama.cpp's chunked form gained it 8% of a whole prompt
on this card).

## A better GEMM: what was searched, what was measured, what it gave (2026-10-05, evening)

**First, the clock.** A search turns up a Windows Strix Halo problem that would explain a 4 TFLOP/s GEMM
outright: the GPU drops to ~600 MHz for compute whenever the console display is off (lid closed, display
timeout), FP16 GEMM 31 -> 8 TFLOPS ([ROCm/legacy-rocm-build #6675](https://github.com/ROCm/legacy-rocm-build/issues/6675);
workarounds: `SetThreadExecutionState(ES_DISPLAY_REQUIRED)` in the process, or an injected 1-pixel mouse move).
This PC is driven over RDP with a 4K panel on the 8060S and a 300 s display timeout. Measured with
`docs/benchmarks/2026-10-05-halo-hip-clock.hip` (the 20-bit `SHADER_CYCLES` register over 100 us windows of the
100 MHz wall clock; gfx11 has no `s_memtime`, so `clock64()` reads 0): **2,780-2,900 MHz** under a spin load,
2,240-2,950 MHz while the hipBLASLt GEMM probe ran. Not throttled here, and the benchmark rows measured minutes
apart agree with each other. The risk stays for unattended runs longer than the display timeout; the engine does
not hold a display request.

**Second, the library, by variant** (`docs/benchmarks/2026-10-05-halo-gemm-variants.cpp`: `hipblasGemmEx` on
constant nonzero data, 10 timed calls, `ROCBLAS_USE_HIPBLASLT=1`; rocBLAS alone is 20-45% lower on every row):

| shape, layout, types | TFLOP/s |
|---|---|
| the engine's dense shape N=3072 T=8192 K=2048, T/N, **BF16 or FP16 in -> FP32 out** | **4.3** |
| the same, FP16 -> FP16 out, or BF16 -> BF16 out | **23.6-24.0** |
| N/N layout, FP16 -> FP32 / FP16 -> FP16 | 5.3 / 20.3 |
| 4096^3, FP16 -> FP32 / FP16 -> FP16 / BF16 -> BF16 | 4.3 / 25.4 / 26.1 |
| the expert shape N=1280 T=160 K=2560, FP16 -> FP32 / FP16 -> FP16 | 3.5 / 18.7 |
| the expert down shape N=2560 T=160 K=640, FP16 -> FP32 | 3.7 |
| decode-like N=3072 T=8 K=2048, FP16 -> FP32 | 1.0 |

So the slow path is one thing: **a 32-bit output**. With a 16-bit output the same library runs 5-6x faster on this
card, 24-26 TFLOP/s, which is the 32-37 TFLOP/s others measure on gfx1151 with hipBLASLt on big shapes
([llm-tracker](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance): 36.9 peak, "59.4
theoretical"), less the FP32 accumulate and this shape. Every `cublasGemmEx` the engine issues asks for FP32 out
(`gemm.cu:388/407`), because the consumers accumulate into FP32 (`beta`, the hi/lo split GEMMs). A 16-bit output
with a cast would cap those at 11 bits: a numerical change for the maintainers, not a setting. `hipBLASLt`'s
`compute 16F` variant is faster still on rocBLAS (17 TFLOP/s) but accumulates in FP16: 1.2e-2 relative error
against 3.3e-4, unusable here. Checked with `docs/benchmarks/2026-10-05-halo-gemm-f16out.cpp` on random data.

**Third, the engine's own answer to this: its hipBLASLt solution table.** `tools/hip/tune_hipblaslt` asks
hipBLASLt for every heuristic solution of each of the engine's 32 dense GEMM shapes (T = 4096 and 8192), times
them against `hipblasGemmEx`, checks them against it (finite, max abs, relative L2 tolerances) and writes the
best ids. On gfx1151 with hipBLASLt 1.4.1 (60 s): **every shape found a 5.2-5.4x faster solution with FP32
output** (f16 T=8192 N=12288 K=2560: 201.0 -> 38.4 ms, 2.6 -> 13.4 TFLOP/s; T=4096: 101.5 -> 18.9 ms; the
96x96x32 macro-tile `SAV` kernels), shipped as `tools/hip/gfx1151-hipblaslt-100401.txt` (setup uses it when the
installed hipBLASLt reports 1.4.1; the configs here carry `STRATA_HIPBLASLT_TUNING`). The engine with the table:

| Coder IQ1_M, arena mode (the first two columns at 64/64, the table column at 32/96: the split the BIOS was left on after the UD-IQ4_XS runs; the cache held all 12,288 experts and hit 100% in every run either way) | plain | `ROCBLAS_USE_HIPBLASLT=1` | **the table** (+ the switch for the shapes it lacks) |
|---|---|---|---|
| pp4096 (`pp4k.py`, 2 requests) | 199-207 t/s | 264-279 t/s | **404-452 t/s** |
| 4,165-token prompt, GPU timeline | 19,955 ms | 14,725 ms | **9,188 ms** |
| hc read / GDN projections / QSA proj / router | 3,438 / 4,969 / 2,077 / 909 ms | 1,768 / 3,263 / 1,443 / 591 | **673 / 938 / 468 / 274** |
| QSA attn / expert GEMMs gate-up + down / dequant | 1,986 / 1,772 + 785 / 489 | 1,944 / 1,705 + 763 / 473 | 2,034 / 1,803 + 798 / 482 (untouched: not in the table) |
| the benchmark, prefill @ 4.75K / pp4096 / 16K prefix / 20K | 209 / 209 / 213 / 211 | 274 / 268 / 277 / 274 | **451 / 441 / 440 / 434 t/s** |
| the benchmark, decode (cold / fresh / after 4.75K / tg128 @ 16K) | 35.0 / 33.0 / 31.3 / 27.5 | 38.6 / 35.1 / 34.7 / 28.8 | 39.9 / 36.1 / 30.9 / 29.2 |

(`benchmarks/2026-10-05-halo-coder-iq1_m-arena-lt-table.json`.) Prefill is now 1.6x what the day started with
at 64/64 and **ahead of llama.cpp on this PC at every length** (its 322-391 t/s at 4K with the chunked GDN
kernel, 273-281 at 16K); still 0.4-0.5x the RTX 3060 PC. The table only covers the dense shapes at the two chunk
sizes (`Lt fallback; no calibration for dtype=bf16 T=4165 N=256` for the partial last chunk and the 256-wide
alpha/beta projections: those take the plain path).

**What is left, and the candidates found for it.** Of the 9.2 s: the FP32 prompt attention 2.0 s (22%), the
expert GEMMs 2.6 s + their dequant 0.5 s (34%), the remaining dense work 2.4 s, gaps 1.5 s.

- **Strata's own PR #313** ([Niko1221/Strata#313](https://github.com/Niko1221/Strata/pull/313), branch
  `wmma-optin`, open, the maintainer asked for a rebase and per-arch guards): a 300-line `src/prefill/wmma_gemm.cu`
  (FP16/BF16 in, **FP32 out**, `__builtin_amdgcn_wmma_f32_16x16x16_{f16,bf16}_w32`, a 1-wave 16x16 and a 4-wave
  64x64 double-buffered LDS tile, beta 0/1 and ldy, dispatched from `Gemm::f16`/`bf16` ahead of hipBLAS; 440-shape
  parity test, <= 1 ULP) and a gfx11 WMMA prompt attention in `qsa_prompt_attn.cu`, opt-in by `STRATA_WMMA_GEMM=1`
  and `STRATA_PA_WMMA=1`. On the RX 7900 XTX: 1K prefill 376 -> 471 t/s with the GEMM, 619 with both; 32K 790 ->
  1,501. Its device guards are `__gfx1100__ || __gfx1101__ || __gfx1102__` and the runtime check is the `gfx11`
  prefix, which gfx1151 passes: the port is one `|| defined(__gfx1151__)` per guard. This is the direct route to the
  2.0 s of attention, and its GEMM is a second opinion on the dense shapes the table does not cover; no attention
  parity test comes with it.
- **The expert GEMMs** (512 FP16 GEMMs of ne rows per layer, FP32 out, 3.5-3.7 TFLOP/s): a shape no table can
  hold (ne varies). Three routes: (a) 16-bit output for these only (18.7 TFLOP/s measured at the shape, 5x; the
  consumers are `swiglu_interleaved` and `moe_combine`, the inputs are 1-4 bit experts: a precision change that
  needs the parity tests), (b) a WMMA kernel with FP32 accumulate at full speed - PR #313's 64x64 tile handles
  any ne, or the self-contained kernel of [ROCm/hip-ep#1002](https://github.com/ROCm/hip-ep/pull/1002) (gfx1151-tuned
  tiles 256x128 / 128x128, +19% over hipBLASLt on small/mid-M prefill, +22-344% at M=1), (c) ggml's MMQ path
  (`STRATA_PREFILL_MMQ`, grouped, INT8 dot, no dequant), which fails parity on this card; llama.cpp's own gfx1151
  work on it is in [ggml-org/llama.cpp#21284](https://github.com/ggml-org/llama.cpp/issues/21284) (MMQ tile
  `mmq_x=48, mmq_y=64, nwarps=4` against VGPR spills, `__builtin_amdgcn_sudot4`, `__expf`: pp128 +61-74%).
- **Other references found**: [glovepost/wmma_ops](https://github.com/glovepost/wmma_ops) (FP16 WMMA GEMM for
  gfx1151, 41.3 TFLOP/s at 4096^3, FP32 accumulate - but no license file); [ROCm/ROCm#4748](https://github.com/ROCm/ROCm/issues/4748)
  (gfx1151's rocBLAS kernels 2x slower than gfx1100's on the same card, `HSA_OVERRIDE_GFX_VERSION=11.0.0` on
  Linux - no Windows equivalent); [ROCm/ROCm#4566](https://github.com/ROCm/ROCm/issues/4566) and
  [#5643](https://github.com/ROCm/ROCm/issues/5643) (hipBLASLt on gfx1151: FP32 GEMMs slower than hipBLAS, the
  fast path refusing the arch on ROCm 7.1); [ggml-org/llama.cpp#16827](https://github.com/ggml-org/llama.cpp/pull/16827)
  (rocWMMA flash attention retuned for gfx1151: pp512 at 64K depth -58% -> +66% against the HIP baseline) and
  [#24437](https://github.com/ggml-org/llama.cpp/issues/24437) (its regression at long context);
  [Atlas-Inf/atlas#41](https://github.com/Atlas-Inf/atlas/pull/41) (another engine's Windows gfx1151 port of this
  model: BF16 GEMM fallback kernels, 17 t/s decode, grouped MoE GEMMs dominating its prefill);
  [pytorch/pytorch#171687](https://github.com/pytorch/pytorch/issues/171687) (gfx1151 decode 90% in
  `hipMemcpyWithStream`: a lead for the UD-IQ4_XS decode question above).

## TODO 1 done: PR #313's gfx11 matrix-core prompt attention on gfx1151 (2026-10-05, night)

[TODO.md](TODO.md) item 1. Upstream PR #313 (branch `wmma-optin`, on engine 0.1.31) was merged onto this 0.1.34 tree
on the branch `wmma-gfx1151` (two conflicts: `qsa_prompt_attn.cu`, where the gfx12 S6 kernel added since then and
the PR's gfx11 kernel both named their kernel `prompt_attn_wmma_kernel` and launcher `launch_wmma` - the PR's are
now `prompt_attn_wmma11_kernel` / `launch_wmma11`, dispatched on `STRATA_PA_WMMA=1` and a `gfx11` `gcnArchName`
before HEAD's HIP `return false`; and `docs/AMD_HIP.md`, both sides kept). The PR's device guards take
`__gfx1150__` / `__gfx1151__` as well. `src/kernels/qsa_prompt_attn_parity.cpp` (ctest `hip_prompt_attn_wmma`),
which skipped off gfx12, now drives the gfx11 kernel on gfx11 with both KV formats. Built with
`cmake --build build-hip-win`, the engine copied into `engine\` (the previous one kept as
`engine\strata-0.1.34-pre313.exe`).

**Parity on the 8060S.** `hip_prefill_wmma_gemm_parity`: 3,520 cases passed, 26 declined shapes skipped, max abs
2.4e-7. `hip_prompt_attn_wmma 32768 2048 3` (`STRATA_PA_WMMA=1`): 4 of 4 pass - int8 KV at 32K: the new kernel 3.9e-6
of FP64 against the FP32 kernel's 2.5e-6, 116.1 -> 34.3 ms per chunk (3.39x); FP16 KV 4.4e-6, 2.88x; 1,500-cell and
2,100-cell contexts 3.3x. `ctest` on the new build: 56 tests, 49 pass, the same 5 known failures as before
(`hip_handoff`, `ple_parity`, `expert_parity`, `pool_test` want fixtures or Linux; `hip_prefill_mmq_parity` is the
gfx1151 MMQ defect - later the test's stream order, [TODO 3 done](#todo-3-done-the-mmq-parity-failure-was-the-tests-stream-order-not-the-kernels-2026-10-05-night)),
`hip_prefill_hipblaslt_gemm` skips without `STRATA_HIPBLASLT_TUNING` in its environment.

**The engine, 4K prompts** (`pp4k.py`, 2 requests each, on top of the hipBLASLt table and the routing switch):

| switches | pp4096 | GPU timeline (4,165 tokens) | qsa attn | hc read / GDN / QSA proj (the dense GEMMs) |
|---|---|---|---|---|
| table only (the morning's best) | 404-452 t/s | 9,188 ms | 2,034 ms | 673 / 938 / 468 |
| **+ `STRATA_PA_WMMA=1`** | **494-526 t/s** | **7,733 ms** | **611 ms** | 669 / 922 / 503 |
| + `STRATA_PA_WMMA=1` + `STRATA_WMMA_GEMM=1` | 384-385 t/s | 10,638 ms | 605 | 1,755 / 1,900 / 929 |
| + `STRATA_WMMA_GEMM=1` alone | 339-341 t/s | 12,042 ms | 1,970 | 1,840 / 1,915 / 886 |

The attention kernel is the gain the parity test promised: 3.3x on its phase, 1.2x on the prompt. The PR's WMMA
GEMM is slower than hipBLASLt's calibrated solutions on this card (it runs ahead of BLAS, so it took the dense
shapes away from the table and roughly tripled them) and does not reach the expert GEMMs (unchanged at 1,756 +
784 ms: it declines their shapes). So `strata-coder-iq1_m.json` carries `STRATA_PA_WMMA=1` and not
`STRATA_WMMA_GEMM`. The full benchmark with it (`benchmarks/2026-10-05-halo-coder-iq1_m-arena-lt-table-pa-wmma.json`):

| Coder IQ1_M, arena mode at 32/96 (see the note below) | table | **table + gfx11 WMMA attention** | llama.cpp on this PC | RTX 3060 Laptop PC |
|---|---|---|---|---|
| cold / fresh / after-4.75K decode | 39.9 / 36.1 / 30.9 t/s | **40.8 / 37.1 / 32.6 t/s** | 26.1 / 21.9 / 20.8 | 43.3 / 42.1 / 39.3 |
| prefill @ 4.75K | 451.0 | **528.1 t/s** | 321.5 | 961.3 |
| pp4096 @ d0 | 440.9 | **516.8 t/s** | 318.1 | 926.9 |
| 16,384-token prefix / 20,421 over all | 440.2 / 433.5 | **542.7 / 534.5 t/s** | 269.9 (pp4096 @ d16384) | ~1,126 |
| tg128 @ d0 / @ d16384 | 28.2 / 29.2 | 27.9 / 28.2 | 20.9 / 18.0 (plain) | 36.2 / 37.4 |
| the prime request | 35.9 t/s | 36.5 t/s, 162 of 214 drafts, the same `is_prime` | | 41.0 |

**The split these ran on:** the BIOS was still at 32/96 from the UD-IQ4_XS runs, not 64/64 as the morning's arena
runs. For the Coder it makes no difference the logs can see: `expert cache auto: 36.08 GiB free -> 12288 slots`
(64/64: 46.43 GiB free, the same 12,288), 12.1 GiB of VRAM left after loading (64/64: 22.7), the pinned 23.4 GiB
arena in 95.6 GiB of RAM, and `decode expert cache hit rate: 100.0%` on every request of every run. The table
runs and the WMMA runs are therefore comparable with the 64/64 rows above them; what 32/96 would cost is only
room: a longer context or a second model would not have the 22.7 GiB. Prefill on the Halo is now 1.6-2.0x
llama.cpp's on the same PC and 0.48-0.56x the RTX 3060 PC's; decode 0.83-0.94x
the 3060's. The day's prompt speed: 146 -> 209 -> 274 -> 451 -> 528 t/s (page cache, BIOS split, hipBLASLt
routing, the calibrated table, the WMMA attention). Of the 7.7 s a 4K prompt now takes, the expert GEMMs and
their dequant are 3.0 s (39%): TODO item 2.

### The one table, redone with the night's engine

The same table as [One table](#one-table-the-best-halo-setup-per-model-against-the-3060m-pc) above, with the night's engine (the hipBLASLt table and the gfx11 WMMA attention) in both Strata columns; that table is left as it was. The best combination of memory split and engine measured on this Halo for each model (Strata from this page;
llama.cpp from EngramHalo.cpp's `docs/strix-halo/windows.md`, its 2026-09-02 and 2026-10-01 runs at 96/32 and
its 2026-10-04 chunked-GDN prefill kernel), beside 3060M.md's RTX 3060 Laptop PC (Strata, CUDA). MTP on in
every decode row (llama.cpp with the EasiiX MTP sidecar; its tg128 rows are plain decode). Where the other Halo
engine wins a row, its number is in brackets.

| Coder IQ1_M / UD-IQ4_XS | **Coder: Halo best = Strata, every expert on the GPU (64/64 or 32/96: the same 12,288-slot cache), the gfx1151 hipBLASLt table, the gfx11 WMMA attention** | Coder: 3060M PC | **UD-IQ4_XS: Halo, llama.cpp 96/32 with the chunked GDN kernel, against Strata 32/96 (the night's engine, 10,000-slot cache) in brackets; the better one in bold** | UD-IQ4_XS: 3060M PC |
|---|---|---|---|---|
| model load to listening | **15 s** | 11.5 s | 83-94 s (**Strata: 30 s**) | 42 s |
| cold first request, decode | **40.8 t/s** (llama.cpp: 26.1) | 43.3 t/s | 28.1-30.4 t/s (Strata: 29.7) - even | 37.6 t/s |
| fresh code prompt, decode | **37.1 t/s** (llama.cpp: 21.9) | 42.1 t/s | 25.1-30.7 t/s (Strata: 26.7) - even | 38.7 t/s |
| prefill @ ~4.75K | **528.1 t/s** (llama.cpp: 321.5) | 961.3 t/s | 322 t/s (278-284 before the kernel; **Strata: 324.1**) - even | 662.8 t/s |
| decode tail after that prefill | **32.6 t/s** (llama.cpp: 20.8) | 39.3 t/s | 21.3-25.1 t/s (Strata: 22.4) - even | 33.1 t/s |
| repeated prompt (reference) | 41.3 t/s (llama.cpp: 49.9, n-gram drafts) | 45.1 t/s | **58.6 t/s** (Strata: 30.7) | 42.6 t/s |
| pp4096 @ d0 | **516.8 t/s** (llama.cpp: 318.1) | 926.9 t/s | **372-391 t/s** (314-349 before; Strata: 311.9) | 819.4 t/s |
| tg128 @ d0 | **27.9 t/s** (llama.cpp: 20.9 plain) | 36.2 t/s | 21.8-22.8 plain (Strata: 19.9 MTP) - even | 32.2 t/s |
| pp4096 @ d16384 | **~530 t/s** (534.5 over 20,421; llama.cpp: 269.9) | ~1,126 t/s | 273-281 t/s (244-260 before; **Strata: 342-353** over 16-20K) | ~949 t/s |
| tg128 @ d16384 | **28.2 t/s** (llama.cpp: 18.0 plain) | 37.4 t/s | 16.3-19.4 plain (**Strata: 21.7** MTP) | 31.7 t/s |
| ratio to the 3060M PC, decode | 0.8-0.9 | 1 | 0.7-0.8 | 1 |
| ratio to the 3060M PC, prefill | 0.47-0.56 | 1 | 0.4-0.5 (Strata: 0.36-0.49) | 1 |

(The Coder column is the night's configuration: the calibrated hipBLASLt table and PR #313's gfx11 WMMA attention,
see [A better GEMM](#a-better-gemm-what-was-searched-what-was-measured-what-it-gave-2026-10-05-evening) and
[TODO 1 done](#todo-1-done-pr-313s-gfx11-matrix-core-prompt-attention-on-gfx1151-2026-10-05-night); the day began
at 146 t/s.) What the table says: for the Coder the Halo's best is Strata with the whole model on the GPU, which
puts its decode within 0.8-0.9x of the 3060 PC while llama.cpp's is 0.5-0.6x, and its prefill 1.6-2.0x llama.cpp's
on this PC and about half the 3060's. For UD-IQ4_XS the night's engine (`benchmarks/2026-10-05-strata-unsloth-ud-iq4_xs-table-pa-wmma.json`,
the table, the WMMA attention, 10,000 slots asked for, 98-99% hits) lifted Strata's prefill from 210-236 to
312-353 t/s: even with llama.cpp at 4.75K (324 vs 322), behind it at pp4096 (312 vs 372-391), ahead at 16-20K
(342-353 vs 273-281), with decode even (20-30 t/s both) and the load 30 s against 83-94. No longer one engine's
model: llama.cpp for short prompts, Strata for long ones and for the start-up. For UD-IQ4_XS no Halo setup gets the experts on the GPU with the engine as it is (the 32/96 and 64/64
limits of [the memory section](#ud-iq4_xs-on-the-3296-split) and the BIOS's three choices), decode is a tie
between the two engines at 0.7-0.8x of the 3060, and llama.cpp's prefill is 1.3-1.8x Strata's. The prefill gap
to the 3060 is the same engine problem in both models: RDNA 3.5 GEMMs and kernels, not memory.

## TODO 2 done: the expert products are MMQ, not FP16 GEMMs, and their J tile (2026-10-05, night)

[TODO.md](TODO.md) item 2 asked to decide between three faster routes for "512 FP16 GEMMs of ne rows per layer with
FP32 output at 3.5-3.7 TFLOP/s". The first measurement overturned the premise. Everything below is the Coder IQ1_M
in arena mode at 32/96 with the night's config (the table, `ROCBLAS_USE_HIPBLASLT=1`, `STRATA_PA_WMMA=1`), 4,16x-token
requests from `pp4k.py` (3 each), the phases from `STRATA_PREFILL_TIMING=1`; probe sources and logs in
`docs/benchmarks/2026-10-05-halo-expert-*` and `2026-10-05-halo-mmq-bench*`.

**What the expert phases actually run.** With `STRATA_HIPBLASLT_VERBOSE=1` the engine resolves a hipBLASLt solution
once per distinct shape; the 4K run logs 40 dense resolutions and none for an expert row count. With
`STRATA_PREFILL_MMQ=0` it logs 1,206 distinct `Lt fallback; no calibration for dtype=f16 T=<ne> N=1280 K=2560` and
the phases change: so by default the Coder's experts go through llama.cpp's MMQ (`src/prefill/moe_mmq.cu`: the
weights stay quantized, the activations rounded to q8_1, int8 dot products, 16 experts per launch), because its
pack's expert types are MMQ types (`native_experts.txt`: gate/up IQ3_XXS or IQ2_S, down IQ4_NL or Strata's Q2_0;
UD-IQ4_XS's IQ4_XS / IQ3_S / Q8_0 likewise). "dequant" in the timing line is the gathers of the experts into the
group buffer, "gemm gate/up" the MMQ product plus swiglu, "gemm down" the q8_1 quantization plus the product.

| 4,165-token prompt | dequant / gather | gemm gate/up | gemm down | GPU timeline | pp4096 |
|---|---|---|---|---|---|
| default (MMQ) | 478-489 ms | 1,777-1,839 | 787-816 | 7,712-8,042 ms | 503-527 t/s |
| `STRATA_PREFILL_MMQ=0` (FP16 dequant + `Gemm::f16`, the shapes uncalibrated) | 555 | 4,231 | 636 | 10,048 | 398-407 |
| `STRATA_PREFILL_MMQ=0` + the table rows below | 678-699 | 1,593-1,671 | 760-813 | 7,704-8,160 | 498-528 |
| default + the median J tile (shipped, below) | 479-484 | **1,581-1,604** | **764-769** | **7,516-7,741** | **524-541** |

The host is not the limit on either path: the expert loop's host time per 4K prompt (a new line in the timing
output) is 34-68 ms on MMQ and 447-516 ms on the FP16 path, against 2.4-2.6 s of GPU phases.

**The three FP16 routes, measured anyway** (`docs/benchmarks/2026-10-05-halo-expert-gemm.cpp`, linking
`wmma_gemm.cu`; 20 back-to-back launches per point, so launch latency is amortized; max relative error against a
double product in brackets):

| gate/up [T x 2560] x [2560 x 1280], us per GEMM | T=16 | 64 | 128 | 256 | 512 | one layer (512 experts, Zipf rows, 39,665 in all) |
|---|---|---|---|---|---|---|
| `hipblasGemmEx` FP16 -> FP32 (the engine's call, 7e-6) | 43 | 108 | 183 | 402 | 911 | 95.9 ms = 2.7 TFLOP/s |
| hipBLASLt solution from the tuner, FP32 out (7e-6) | 35 | 55 | 80 | 98 | 191 | 44.9 ms (lt2537/2538/2551 below T=96, lt2539 above) |
| `hipblasGemmEx` FP16 -> FP16, compute 32F (2.8e-4) | 51 | 67 | 66 | 66 | 174 | 21.0 ms |
| PR #313's WMMA, FP32 accumulate (8e-6) | 31 | 83 | 100 | 145 | 414 | 81.5 ms |
| down [T x 640] x [640 x 2560]: the same four | 33 / 22 / 26 / 9 | 62 / 45 / 27 / 22 | 95 / 52 / 64 / 49 | 185 / 72 / 50 / 56 | 411 / 81 / 62 / 102 | 47.0 / 19.5 / 17.6 / 20.1 ms |

hipBLASLt 1.4.1's grouped GEMM (`hipblaslt_ext::GroupedGemm`, 16 experts per launch) refuses the problem on gfx1151
(`setProblem`/heuristic fails at the first group). `tune_hipblaslt --case` at the two shapes for T = 16-512
(`2026-10-05-halo-expert-lt-tuning.log`) found FP32-output solutions 1.8-3.6x faster than `hipblasGemmEx`; those 16
rows are now in `tools/hip/gfx1151-hipblaslt-100401.txt` (the lookup takes the nearest T), which is the third table
row above: the FP16 path's gate/up 4,231 -> 1,593-1,671 ms and no uncalibrated shape left. The 16-bit output would
take the FP16 path further (half the gate/up time again in the probe) at a precision change (2.8e-4 against 7e-6),
and only matters when MMQ is off; not taken.

**The MMQ products at one layer's shape** (`docs/benchmarks/2026-10-05-halo-mmq-bench.cpp`, linking
`strata_mmq.lib`: 512 experts with Zipf-like row counts summing to 41,650 over the ids in random order, groups of 16
as `prefill.cpp` forms them, ms per layer averaged over 3 distributions x 5 reps; "padding" is the rows the J tile
adds, counted the way the kernel pads: each expert to a multiple of the group's tile):

| per layer, ms (TOPS) | id order, tile for the largest (the engine until tonight) | tile for the mean | tile for the median | sorted by rows, tile for the largest |
|---|---|---|---|---|
| J padding | 98% of the rows | 51% | 22% | 16% |
| gate/up IQ3_XXS / IQ2_S / IQ4_XS / IQ3_S | 47.8 (5.7) / 50.6 / 37.5 / 41.0 | 37.1 / 41.1 / 30.7 / 35.4 | 34.5 (7.9) / 38.1 / 29.7 / 34.9 | 29.3 (9.3) / 31.4 / 25.8 / 28.3 |
| down IQ4_NL / Q2_0 / IQ4_XS / Q8_0 | 21.7 (6.3) / 21.2 / 15.3 / 23.9 | 18.6 / 18.7 / 13.1 / 20.4 | 17.8 (7.7) / 17.8 / 12.8 / 18.9 | 15.4 (8.8) / 15.6 / 11.0 / 15.8 |
| groups of 32 / 64 (id order, largest), IQ3_XXS gate/up | 48.3 / 48.6 | | | 32.3 / 34.1 |

The mechanism: `mul_mat_q_switch_J` picks the J tile from `ncols_opt` and the launch grid from `ncols_max`;
`moe_mmq.cu` passed the group's largest expert for both, so a group of 16 in id order (10 rows next to 1,000)
computed its small experts on 128-row tiles. The engine's own count of this (the new timing line, the real routing):
a J tile for each group's largest expert pads **46% of a 4,16x-token chunk's rows and 161% of a 1,301-token
chunk's**.

**Shipped.** `mmq::Product::opt_rows`, the row count the tile is chosen for; `prefill.cpp` passes the group's
median (`STRATA_PREFILL_MMQ_OPT=max` restores the old choice, `=mean` the mean). Measured, same binary:

| Coder, 3 requests each | gemm gate/up | gemm down | GPU timeline (4,16x tokens) | pp4096 | 1,306-token prompt |
|---|---|---|---|---|---|
| tile for the largest (`STRATA_PREFILL_MMQ_OPT=max`) | 1,803-1,864 | 803-832 | 7,772-8,016 | 507-524 t/s | 395 t/s |
| tile for the mean (`=mean`) | 1,625-1,698 | 776-811 | 7,587-7,967 | 509-536 t/s | 428 t/s |
| **tile for the median (the default now)** | **1,581-1,604** | **764-769** | **7,516-7,741** | **524-541 t/s** | **424 t/s** |

The 96 tokens a temperature-0 request produced (reasoning text over a 1,306-token prompt) are identical between the
largest and the median tile, as they should be: the products are the same, only their tiling changes. (The FP16 path
diverges from MMQ after 150 characters: different arithmetic, not a defect.) The gain is smaller than the harness's
because the real routing pads less than its Zipf curve (46% against 98%).

**Sorting the experts by row count** (the harness's best column) is in `prefill.cpp` too (`STRATA_PREFILL_SORT_EXPERTS=0`
turns it off), but only takes effect when the chunk does not stream its experts: in arena mode every expert is
staged from the pinned arena through the ring in id order even when it is resident in the cache (the prompt path
borrows cache slots as its staging ring; TODO items 10 and 11), and the streamed walk consumes the ring in that
order, so for chunks of `STRATA_PREFILL_STREAM_MIN` = 1,024 tokens and more 0 of 48 layers sort. Where it applies it pays: 1,001-token requests (`pp4k.py ... 800`, 3 each, the chunk below the threshold, 48 of 48 layers sorted), the same binary with `STRATA_PREFILL_SORT_EXPERTS=0` against the default: gemm gate/up 610-643 -> 475-509 ms, gemm down 290-303 -> 226-240 ms, GPU timeline 2,437-2,620 -> 2,200-2,412 ms, 360-388 -> 388-428 t/s (the engine's padding count for a tile sized to the largest: 207% of the rows in id order, 31% sorted).
What it would be worth for long chunks is the harness's last column: group the gathers by size class instead (TODO
2a). The MMQ kernels' own efficiency, 6-12 TOPS at these shapes against the card's int8 peak, is TODO 2b: the
vendored ggml already carries upstream's RDNA 3.5 tile tables and uses `sudot4`; llama.cpp#21284's smaller tiles
are the untested lead.

### The one table, redone with TODO 2's engine

The same table as [the night's](#the-one-table-redone-with-the-nights-engine), with this section's engine (the
median J tile, the sort below 1,024 tokens, the extended hipBLASLt table, on top of the night's table and WMMA
attention) in both Strata columns; the earlier tables are left as they were. The same benchmark
(`2026-10-02-3060m-bench_halo.py`), the same configs (`strata-coder-iq1_m.json` at 32/96, all 12,288 experts
resident, 100% hits; `strata-unsloth-ud-iq4_xs.json` at 32/96, `STRATA_ARENA_PIN_GIB=24`, 10,000 slots asked and
14,358 made, 90-98% hits), run back to back late on 2026-10-05
(`benchmarks/2026-10-05-halo-coder-iq1_m-todo2.json`, `benchmarks/2026-10-05-halo-ud-iq4_xs-todo2.json`; the load
time is the server's start to its first `/v1/models` answer). llama.cpp and the 3060M PC columns are the night's.
MTP on in every decode row (llama.cpp with the EasiiX MTP sidecar; its tg128 rows are plain decode). Where the other
Halo engine wins a row, its number is in brackets; the night's Strata number follows in parentheses where it moved.

| Coder IQ1_M / UD-IQ4_XS | **Coder: Halo best = Strata, every expert on the GPU, TODO 2's engine** | Coder: 3060M PC | **UD-IQ4_XS: Halo, llama.cpp 96/32 with the chunked GDN kernel, against Strata 32/96 (TODO 2's engine) in brackets; the better one in bold** | UD-IQ4_XS: 3060M PC |
|---|---|---|---|---|
| model load to listening | **12.6 s** (was 15) | 11.5 s | 83-94 s (**Strata: 36 s**) | 42 s |
| cold first request, decode | **40.4 t/s** (llama.cpp: 26.1) | 43.3 t/s | 28.1-30.4 t/s (Strata: 30.9) - even | 37.6 t/s |
| fresh code prompt, decode | **36.6 t/s** (llama.cpp: 21.9) | 42.1 t/s | 25.1-30.7 t/s (Strata: 26.1) - even | 38.7 t/s |
| prefill @ ~4.75K | **548.1 t/s** (was 528.1; llama.cpp: 321.5) | 961.3 t/s | 322 t/s (**Strata: 317.5**, was 324.1) - even | 662.8 t/s |
| decode tail after that prefill | **32.3 t/s** (llama.cpp: 20.8) | 39.3 t/s | 21.3-25.1 t/s (Strata: 22.2) - even | 33.1 t/s |
| repeated prompt (reference) | 40.8 t/s (llama.cpp: 49.9, n-gram drafts) | 45.1 t/s | **58.6 t/s** (Strata: 30.7) | 42.6 t/s |
| pp4096 @ d0 | **536.6 t/s** (was 516.8; llama.cpp: 318.1) | 926.9 t/s | **372-391 t/s** (Strata: 322.0, was 311.9) | 819.4 t/s |
| tg128 @ d0 | **27.7 t/s** (llama.cpp: 20.9 plain) | 36.2 t/s | 21.8-22.8 plain (Strata: 20.5 MTP) - even | 32.2 t/s |
| pp4096 @ d16384 | **~560 t/s** (560.3 over 20,421, the 16,384 prefix at 568.3; was ~530; llama.cpp: 269.9) | ~1,126 t/s | 273-281 t/s (**Strata: 347-354** over 16-20K, was 342-353) | ~949 t/s |
| tg128 @ d16384 | **27.9 t/s** (llama.cpp: 18.0 plain) | 37.4 t/s | 16.3-19.4 plain (**Strata: 21.8** MTP) | 31.7 t/s |
| ratio to the 3060M PC, decode | 0.75-0.93 | 1 | 0.64-0.82 | 1 |
| ratio to the 3060M PC, prefill | 0.50-0.58 | 1 | 0.37-0.48 (llama.cpp 0.4-0.5) | 1 |

What moved: the Coder's prefill, +4-6% at every length (528 -> 548 at 4.75K, 517 -> 537 at pp4096, ~530 -> ~560 at
16-20K: the median J tile on a 4-8K chunk), its decode unchanged within the run-to-run spread (the expert products
of the decode path are not the prompt path's), the load 2 s shorter (the spread of a page-cached start, not the
engine). UD-IQ4_XS did not move (317-354 against 312-353): its prompt chunks stream most experts from the arena
(`host staging` 4.7-39.5 s per chunk in its timing lines), so the sort does not apply and the products wait on the
staging; TODO items 10 and 11 are its lever, not the tile. The day's prompt speed for the Coder, then: 146 -> 209 ->
274 -> 451 -> 528 -> 548 t/s at 4.75K.

## TODO 3 done: the MMQ parity failure was the test's stream order, not the kernels (2026-10-05, night)

[TODO.md](TODO.md) item 3 asked whether `hip_prefill_mmq_parity`'s "synthetic-Q2_0-GU-pass0: non-finite or unwritten
MMQ output" is ggml's gfx1151 handling or Strata's host glue. It is neither: the kernels compute the right numbers on
this card; the test filled its output with a sentinel on the null stream and ran the products on a non-blocking
stream, nothing ordered the two, and on Windows HIP the fill ran after the products. Three probes, their sources
and logs in `docs/benchmarks/2026-10-05-halo-mmq-*-probe*`:

**What was unwritten.** The test stops at its first bad value. `2026-10-05-halo-mmq-parity-probe.cpp` runs the same
products through the same glue (`strata_mmq.lib`) and classifies every output element: 17 cases over five types
(Q2_0, Q8_0, IQ4_NL, IQ3_XXS, IQ2_S), both shapes (gate/up [1280 x 2560], down [2560 x 640]), 1 to 135 rows per
launch, permuted and identity row maps, `opt_rows` smaller than the largest expert. In the test's order every case
failed the same way: every element still the 0xffffffff fill (8,960 of 8,960 for the test's {1,3,3} gate/up case),
no NaN, no inf, no error from `hipGetLastError`, `hipStreamSynchronize` or the copy back. Nothing type- or
shape-specific, and a kernel that returned at once could not have taken the 1.6 s per 4K chunk that TODO 2 timed.

**The tables agree.** `2026-10-05-halo-mmq-config-probe.hip` includes `mmq.cuh` and prints the tile table the host
picks from the compute-capability number (the glue maps `gfx1151` to 0x1151: RDNA 3.5) next to the one the device
code picks from its compiler macros (`RDNA3`, `RDNA3_5`, `AMD_WMMA_AVAILABLE`, `__gfx1151__` all set): identical rows
for Q2_0 and IQ3_XXS at every J (J 16 and 32: 128 threads, I 64; J 48 to 128: 256 threads, I 128; no stream-k).
`NO_DEVICE_CODE`, the kernel's exit when its table has no entry, does trap on this HIP - but the trap was not
reported by `hipDeviceSynchronize`, only by the next `hipMemcpy` ("unspecified launch failure"), so a sync that
returns success is not proof a kernel ran clean here.

**The kernel is right.** `2026-10-05-halo-mmq-kernel-probe.hip` instantiates `mul_mat_q<Q2_0, 16, false>` itself:
one Q2_0 expert [1280 x 2560] times 16 rows (J 16, 22,080 B of shared memory), against a CPU double product over
ggml's dequantizer. The library's kernel and this unit's, MoE mode and plain GEMM mode, built at -O3 and at -O1, on
the null stream, a blocking stream and a non-blocking stream, read after a stream or a device sync: 20,480 of
20,480 elements written, rel_l2 0.0010, max/rms 0.004 every time. The test's order - `hipMemset` on the null stream,
then the quantizer and the product on the non-blocking stream, `hipStreamSynchronize`, copy back - on the same
setup: 0 of 20,480 written. The parity probe with its fill moved to `hipMemsetAsync` on the compute stream: 17 of 17
pass, rel_l2 0.0005-0.0013, max/rms 0.001-0.005.

**The fix and the check.** `tests/hip/prefill_mmq_parity.cpp` now fills its sentinel with `hipMemsetAsync` on the
stream the products run on. Rebuilt, the test passes its six products (rel_l2 0.0005-0.0012, max_abs/ref_rms
0.002-0.0044, every all-zero activation row exactly zero), `ctest -R hip_prefill_mmq_parity` 0.68 s; the ctest
tally for this card is 4 known failures (fixtures or Linux), not 5. For the day's benchmark outputs: the prompt
path (`src/prefill`) has no synchronous memset, the glue launches on the stream it is given, and the kernel wrote
the right numbers in every ordered run, so the Coder's and UD-IQ4_XS's expert products were computed, not skipped.
And measured in the engine (`2026-10-05-halo-mmq-ab.py`: one server start per variant from `strata-coder-iq1_m.json`,
a 1,173-token prompt - the start of `serve/server.py` - temperature 0, thinking off, 64 tokens out): the default
and `STRATA_PREFILL_MMQ=0` continue the file with the same code (`self.n = 0`, the `generate` signature, the
`self.scripts[min(...)]` line), the one difference the run of spaces that opens the first line (12 against 35: the
int8 against the FP16 rounding of the expert products), 5.05 against 5.34 s for the request.

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

## The RTX 5090 over Thunderbolt (not pursued)

Windows lists an RTX 5090 (32 GB) as an external card that was attached before. With it attached, Strata's
ready-made CUDA engine runs without any port, in the low-RAM resident mode (a 32 GB card holds all of the Coder's
experts, most of Q2_0's, [DETAILS.md](DETAILS.md)). Its limits are the 31.6 GiB of system RAM for everything the
card does not hold and the Thunderbolt link (about 3 GB/s, a tenth of PCIe x16 Gen4: `--pcie-frac` near 0). It is
the easier route to a running Strata on this PC and the less interesting one; the question of this page is the
8060S.
