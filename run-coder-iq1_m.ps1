# Starts Strata's server for the Coder IQ1_M on this PC's Radeon 8060S (docs/AIMAX+395-ROCm.md): the conda env
# `strata` (Python 3.14, C:\conda_envs\strata), the config written by hand (strata-coder-iq1_m.json: the experts
# pinned in RAM, which needs the BIOS 64/64 memory split, and rocBLAS routed to hipBLASLt), the HIP engine in
# engine\ with its ROCm DLLs in engine\rocm\bin (the config's lib_dirs).  Not START-HERE.bat: that would make a
# .venv in this folder.
Set-Location "E:\work\AI\Strata"
& "C:\conda_envs\strata\python.exe" "E:\work\AI\Strata\serve\server.py" --engine strata --config "E:\work\AI\Strata\strata-coder-iq1_m.json" --port 8080 @args
