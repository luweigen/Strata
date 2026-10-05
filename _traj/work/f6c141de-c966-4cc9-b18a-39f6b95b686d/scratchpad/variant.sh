#!/bin/bash
# variant.sh <tag> flags... : the MMQ instances (all 9 types) with extra device flags, linked before strata_mmq.lib
cd /e/work/AI/Strata
SDK=C:/conda_envs/rocm100-py312/Lib/site-packages/_rocm_sdk_devel
SP=C:/Users/WEILU~1/AppData/Local/Temp/claude/E--work-AI-Strata/f6c141de-c966-4cc9-b18a-39f6b95b686d/scratchpad
tag=$1; shift
mkdir -p $SP/v-$tag
CXX="$SDK/lib/llvm/bin/clang++.exe"
COMMON="--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode -O3 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -DGGML_USE_HIP=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Ithird_party/llama.cpp/ggml/include -Ithird_party/llama.cpp/ggml/src -Ithird_party/llama.cpp/ggml/src/ggml-cuda -Iinclude"
for t in q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0; do
  $CXX -x hip $COMMON "$@" -c third_party/llama.cpp/ggml/src/ggml-cuda/template-instances/mmq-instance-$t.cu -o $SP/v-$tag/$t.obj 2> $SP/v-$tag/$t.err &
done; wait
$CXX -x hip --offload-arch=gfx1151 --rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode -O3 -std=c++20 -D_DLL -D_MT -Xclang --dependent-lib=msvcrt -D__HIP_PLATFORM_AMD__=1 -Iinclude -Ithird_party/llama.cpp/ggml/include docs/benchmarks/2026-10-06-halo-mmq-tiles.cpp -x none $SP/v-$tag/*.obj build-hip-win/strata_mmq.lib build-hip-win/ggml/src/ggml-base.lib $SDK/lib/amdhip64.lib -fuse-ld=lld -Xlinker /subsystem:console -o $SP/v-$tag/mmq_tiles.exe 2> $SP/v-$tag/link.err && echo "$tag built" || { echo "$tag link failed"; tail -5 $SP/v-$tag/link.err; }
