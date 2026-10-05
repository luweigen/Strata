import sys, pathlib
root = pathlib.Path(r"E:\work\AI\Strata")


def patch(rel, pairs):
    p = root / rel
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{rel}: expected exactly one match for {old[:70]!r}, found {s.count(old)}")
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", rel)


patch("cmake/hip_backend.cmake", [
    ("# report ran it (#311), the maintainers have not.\n"
     "set(_strata_hip_validated gfx1100 gfx1201)\n"
     "set(_strata_hip_community gfx1101 gfx1200)\n"
     "set(_strata_hip_unvalidated gfx1102 gfx1030)\n",
     "# report ran it (#311), the maintainers have not.  RDNA 3.5 gfx1151 (Radeon 8060S / 8050S in the Ryzen AI MAX\n"
     "# 300 series, an integrated GPU with up to 96 GiB of carve-out) and gfx1150 (Ryzen AI 300's Radeon 880M / 890M)\n"
     "# are the gfx11 instruction set with RDNA3's wave32, 64 KiB LDS and sudot4: the port is docs/AIMAX+395-ROCm.md.\n"
     "set(_strata_hip_validated gfx1100 gfx1201)\n"
     "set(_strata_hip_community gfx1101 gfx1200)\n"
     "set(_strata_hip_unvalidated gfx1102 gfx1030 gfx1151 gfx1150)\n"),
    ('      "Strata HIP supports wave32 gfx1100, gfx1101, gfx1200 and gfx1201 (unvalidated: ${_strata_hip_unvalidated}); "',
     '      "Strata HIP supports wave32 gfx1100, gfx1101, gfx1200 and gfx1201 (unvalidated: ${_strata_hip_unvalidated}, see docs/AMD_HIP.md); "'),
])
patch("CMakeLists.txt", [
    ('option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1030 unvalidated)" OFF)',
     'option(STRATA_ENABLE_HIP "Build the HIP targets for wave32 AMD GPUs (gfx1100, gfx1201; gfx1030, gfx1151 unvalidated)" OFF)'),
])
patch("include/strata/hip_compat/intrinsics.hpp", [
    ("#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || \\\n"
     "     defined(__gfx1201__)) && __has_builtin(__builtin_amdgcn_sudot4)\n"
     "    // RDNA3 and RDNA4 expose the signed/unsigned dot4 form (v_dot4_i32_iu8). Mark\n",
     "#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || \\\n"
     "     defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)) && __has_builtin(__builtin_amdgcn_sudot4)\n"
     "    // RDNA3, RDNA 3.5 (gfx1150 / gfx1151: checked on a Radeon 8060S, docs/AIMAX+395-ROCm.md) and RDNA4 expose\n"
     "    // the signed/unsigned dot4 form (v_dot4_i32_iu8). Mark\n"),
])
patch("setup.py", [
    ("# their owners; the RX 6800 / 6900 series (gfx1030, #311) runs but is unvalidated.  There is no ready-made AMD engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo;\n",
     "# their owners; the RX 6800 / 6900 series (gfx1030, #311) runs but is unvalidated; the Radeon 8060S / 8050S integrated\n"
     "# GPU of the Ryzen AI MAX 300 series (gfx1151, docs/AIMAX+395-ROCm.md) is being ported.  There is no ready-made AMD engine: ROCm comes from AMD's TheRock Python wheels into .venv (no sudo;\n"),
    ('                "gfx1030": "https://rocm.nightlies.amd.com/v2/gfx103X-all/"}\n',
     '                "gfx1030": "https://rocm.nightlies.amd.com/v2/gfx103X-all/",\n'
     '                "gfx1151": "https://rocm.nightlies.amd.com/v2/gfx1151/"}\n'),
    ('AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1200", "gfx1201", "gfx1030")\n',
     'AMD_ARCHS = ("gfx1100", "gfx1101", "gfx1200", "gfx1201", "gfx1030", "gfx1151")\n'),
    ('             "gfx1030": "AMD Radeon RX 6800 / 6900 series (gfx1030)"}\n',
     '             "gfx1030": "AMD Radeon RX 6800 / 6900 series (gfx1030)",\n'
     '             "gfx1151": "AMD Radeon 8060S / 8050S, Ryzen AI MAX 300 series (gfx1151)"}\n'),
    ('             "RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), and the RX 6800 / 6900 series (gfx1030, unvalidated)")\n',
     '             "RX 9070 / 9070 XT / Radeon AI PRO R9700 (gfx1201), the RX 6800 / 6900 series (gfx1030, unvalidated) and "\n'
     '             "the Ryzen AI MAX 300 series\' Radeon 8060S / 8050S (gfx1151, unvalidated)")\n'),
    ('                0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030"}            # RX 6800 / 6800 XT / 6900 XT / 6950 XT\n',
     '                0x73BF: "gfx1030", 0x73AF: "gfx1030", 0x73A5: "gfx1030",            # RX 6800 / 6800 XT / 6900 XT / 6950 XT\n'
     '                0x1586: "gfx1151"}                                                  # Radeon 8060S / 8050S (Strix Halo)\n'),
    ('_WIN_AMD_NAME = ((re.compile(r"\\b9070\\b|R9700", re.I), "gfx1201"),\n',
     '_WIN_AMD_NAME = ((re.compile(r"\\b9070\\b|R9700", re.I), "gfx1201"),\n'
     '                 (re.compile(r"\\b80[56]0S\\b", re.I), "gfx1151"),\n'),
])
patch("tools/hip/build_windows.bat", [
    ('rem   STRATA_HIP_ARCHS     gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030 (the cards setup supports, + gfx1102)\n',
     'rem   STRATA_HIP_ARCHS     gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1151 (the cards setup supports, + gfx1102)\n'
     'rem   STRATA_ROCM_ROOT     an installed TheRock ROCm to build with (`rocm-sdk path --root` of any environment that\n'
     'rem                        has the libraries, devel and device-<arch> extras): step 1 is skipped, nothing is installed\n'),
    ('if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030"\n',
     'if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1151"\n'),
    ('rem ---- 1. ROCm (TheRock wheels: the compiler, the HIP runtime, hipBLAS/hipBLASLt/rocBLAS, a device package per arch)\n'
     'set "PY="\n'
     'py -3 -c "import sys" >nul 2>nul && set "PY=py -3"\n'
     'if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"\n'
     'if not defined PY (echo Python 3.10+ is needed & exit /b 1)\n'
     'if not exist "%ROCM_VENV%\\Scripts\\python.exe" %PY% -m venv "%ROCM_VENV%" || exit /b 1\n',
     'rem ---- 1. ROCm (TheRock wheels: the compiler, the HIP runtime, hipBLAS/hipBLASLt/rocBLAS, a device package per arch)\n'
     'set "PY="\n'
     'py -3 -c "import sys" >nul 2>nul && set "PY=py -3"\n'
     'if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"\n'
     'if not defined PY (echo Python 3.10+ is needed & exit /b 1)\n'
     'set "PKG_PY=%ROCM_VENV%\\Scripts\\python.exe"\n'
     'if defined STRATA_ROCM_ROOT (\n'
     '  set "ROCM=%STRATA_ROCM_ROOT%"\n'
     '  set "PKG_PY=%PY%"\n'
     '  echo Using the ROCm at %STRATA_ROCM_ROOT% ^(STRATA_ROCM_ROOT^)\n'
     '  goto :rocm_ready\n'
     ')\n'
     'if not exist "%ROCM_VENV%\\Scripts\\python.exe" %PY% -m venv "%ROCM_VENV%" || exit /b 1\n'),
    ('for /f "delims=" %%R in (\'"%ROCM_VENV%\\Scripts\\rocm-sdk.exe" path --root\') do set "ROCM=%%R"\n'
     'if not exist "%ROCM%\\lib\\llvm\\bin\\clang++.exe" (echo ROCm has no compiler in "%ROCM%" & exit /b 1)\n',
     'for /f "delims=" %%R in (\'"%ROCM_VENV%\\Scripts\\rocm-sdk.exe" path --root\') do set "ROCM=%%R"\n'
     ':rocm_ready\n'
     'if not exist "%ROCM%\\lib\\llvm\\bin\\clang++.exe" (echo ROCm has no compiler in "%ROCM%" & exit /b 1)\n'),
    ('"%ROCM_VENV%\\Scripts\\python.exe" "%SRC%\\tools\\hip\\package_windows.py" --build "%BUILD_DIR%" --rocm "%ROCM%" ^\n',
     '%PKG_PY% "%SRC%\\tools\\hip\\package_windows.py" --build "%BUILD_DIR%" --rocm "%ROCM%" ^\n'),
])
