---
name: windows-shell-gotchas
description: "Pitfalls of the Bash tool and toolchain on this Windows PC (python stub, shared cwd across parallel calls, HIP probe build line)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 53cb97e9-f88c-45ee-a459-ee7480855903
  modified: 2026-10-05T18:55:28.125Z
---

- `python` / `python3` on PATH in Git Bash are the Windows Store stubs: they print nothing and do nothing. Use the
  conda env's python explicitly (see [[conda-env-policy]]) or do text edits with the Edit tool.
- Parallel Bash calls share one shell: a `cd` in one call changes the cwd of the others mid-flight. Start every
  command with `cd /e/work/AI/Strata &&` and use absolute paths for anything that runs after a possible `cd`.
- A long heredoc with apostrophes in the body failed to parse in the Bash tool once; for new source files use Write.
- HIP probes against the built libraries: the SDK clang++ at
  `C:/conda_envs/rocm100-py312/Lib/site-packages/_rocm_sdk_devel/lib/llvm/bin/clang++.exe` with
  `--offload-arch=gfx1151 --rocm-path=<sdk> --rocm-device-lib-path=<sdk>/lib/llvm/amdgcn/bitcode -std=c++20 -D_DLL -D_MT
  -Xclang --dependent-lib=msvcrt`, `-x hip <source> -x none <libs>` (without `-x none` the .lib files are parsed
  as HIP sources), link `build-hip-win/strata_mmq.lib build-hip-win/ggml/src/ggml-base.lib <sdk>/lib/amdhip64.lib
  -fuse-ld=lld -Xlinker /subsystem:console`. The full lines are in the headers of
  docs/benchmarks/2026-10-05-halo-mmq-*-probe*.
- On this Windows HIP, a kernel `__trap()` is not reported by `hipDeviceSynchronize`; it surfaces on the next
  `hipMemcpy` as "unspecified launch failure". And work on the null stream is not ordered against a non-blocking
  stream: a `hipMemset` issued before a kernel on such a stream can land after it.

**Why:** each of these cost a round trip on 2026-10-05 (a silent no-op patch, failed reads from a moved cwd, a
compile that parsed libraries as sources).

**How to apply:** follow the bullets before writing a probe or an in-place patch on this machine.
