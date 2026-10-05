#!/bin/bash
# res2.sh <tag> <type> flags... : like res.sh but output file per tag
cd /e/work/AI/Strata
SDK=C:/conda_envs/rocm100-py312/Lib/site-packages/_rocm_sdk_devel
SP=C:/Users/WEILU~1/AppData/Local/Temp/claude/E--work-AI-Strata/f6c141de-c966-4cc9-b18a-39f6b95b686d/scratchpad
tag=$1; t=$2; shift 2
$SDK/lib/llvm/bin/clang++.exe -x hip --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode -O3 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -D_DLL -D_MT -DGGML_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/ggml/src -Ithird_party/llama.cpp/ggml/src/ggml-cuda -Iinclude "$@" --cuda-device-only -c -Rpass-analysis=kernel-resource-usage third_party/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-$t.cu -o $SP/$tag-$t.o > $SP/$tag-$t-res.txt 2>&1 || { echo "$tag $t FAILED: $(grep -m2 error $SP/$tag-$t-res.txt)"; exit; }
grep -E "Function Name|VGPRs:|ScratchSize|Occupancy" $SP/$tag-$t-res.txt | sed 's/.*remark: *//; s/ \[-Rpass.*//' | paste - - - - | grep "_ZL9mul_mat_q" | sed -E 's/.*Li([0-9]+)ELb([01]).*VGPRs: ([0-9]+).*: ([0-9]+).*: ([0-9]+)/\1 \2 \3 \4 \5/' | awk -v t=$t -v g=$tag '$2==0 && $1%16==0 {s=s sprintf("J%d:%d/%d ",$1,$3,$4)} END{print g, t, s}'
