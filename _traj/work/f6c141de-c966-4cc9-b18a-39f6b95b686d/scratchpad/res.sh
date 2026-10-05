#!/bin/bash
# usage: res.sh <type> [extra flags...]  -> J fallback VGPRs scratch occupancy for mul_mat_q kernels at J%16==0
cd /e/work/AI/Strata
SDK=C:/conda_envs/rocm100-py312/Lib/site-packages/_rocm_sdk_devel
SP=C:/Users/WEILU~1/AppData/Local/Temp/claude/E--work-AI-Strata/f6c141de-c966-4cc9-b18a-39f6b95b686d/scratchpad
t=$1; shift
$SDK/lib/llvm/bin/clang++.exe -x hip --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode -O3 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -D_DLL -D_MT -DGGML_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/ggml/src -Ithird_party/llama.cpp/ggml/src/ggml-cuda -Iinclude "$@" --cuda-device-only -c -Rpass-analysis=kernel-resource-usage third_party/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-$t.cu -o $SP/$t.o > $SP/$t-res.txt 2>&1
grep -E "Function Name|VGPRs:|ScratchSize|Occupancy" $SP/$t-res.txt | sed 's/.*remark: *//; s/ \[-Rpass.*//' | paste - - - - | grep "_ZL9mul_mat_q" | sed -E 's/.*Li([0-9]+)ELb([01]).*VGPRs: ([0-9]+).*: ([0-9]+).*: ([0-9]+)/\1 \2 \3 \4 \5/' | awk -v t=$t '$2==0 && $1%16==0 {printf "%s J=%d v=%d s=%d occ=%d | ", t,$1,$3,$4,$5} END{print ""}'
