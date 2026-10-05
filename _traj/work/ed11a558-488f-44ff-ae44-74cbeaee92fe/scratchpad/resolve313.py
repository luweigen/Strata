import pathlib, subprocess, sys
root = pathlib.Path(r"E:\work\AI\Strata")

def show(rev, path):
    return subprocess.run(["git", "show", f"{rev}:{path}"], cwd=root, capture_output=True, check=True).stdout.decode("utf-8")

# ---- src/kernels/cuda/qsa_prompt_attn.cu
path = "src/kernels/cuda/qsa_prompt_attn.cu"
p = root / path
merged = p.read_text(encoding="utf-8")
head = show("HEAD", path)
pr = show("pr-313", path)
FN = "bool qsa_prompt_attn_batch("
assert head.count(FN) == 1 and pr.count(FN) == 1 and merged.count(FN) == 1

# HEAD's whole S6 block (the gfx12 kernel, hip_wmma_usable, its launcher) up to the dispatch function
h0 = head.index("#if defined(__HIPCC__)\n// ---- S6:")
head_block = head[h0:head.index(FN)]
# the PR's gfx11 launcher up to the dispatch function, renamed
p0 = pr.index("#if defined(STRATA_WMMA_GFX11) && STRATA_WMMA_GFX11\ntemplate <int KV_MODE>\nbool launch_wmma(")
pr_block = pr[p0:pr.index(FN)]
pr_block = pr_block.replace("bool launch_wmma(", "bool launch_wmma11(").replace("prompt_attn_wmma_kernel<KV_MODE>", "prompt_attn_wmma11_kernel<KV_MODE>")

# conflict 1 + the shared tail git hung after it: from the marker to the dispatch function
m0 = merged.index("<<<<<<< HEAD\n#if defined(__HIPCC__)\n// ---- S6:")
m1 = merged.index(FN)
s = merged[:m0] + head_block + pr_block + merged[m1:]

# the PR's gfx11 kernel definition (earlier in the file, auto-merged) gets its own name
old = "template <int KV_MODE>\n__global__ void __launch_bounds__(THREADS) prompt_attn_wmma_kernel("
assert s.count(old) == 1
s = s.replace(old, "template <int KV_MODE>\n__global__ void __launch_bounds__(THREADS) prompt_attn_wmma11_kernel(")

# conflict 2: HEAD's HIP dispatch with the PR's gfx11 opt-in inserted before its `return false`
gfx11 = """#if defined(STRATA_WMMA_GFX11) && STRATA_WMMA_GFX11
    // PR #313: the gfx11 (RDNA3 / RDNA 3.5) matrix-core prompt attention, opt-in (STRATA_PA_WMMA=1), gated at run time on
    // the device's gcnArchName; the pools it takes are the int8-KV and the float ones (no K8V4)
    static const bool pa_wmma_enabled = []() {
        const char* env = std::getenv("STRATA_PA_WMMA");
        if (env == nullptr || std::atoi(env) == 0) return false;
        int dev = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&p, dev) != hipSuccess) return false;
        return std::strncmp(p.gcnArchName, "gfx11", 5) == 0;
    }();
    if (pa_wmma_enabled && pools.k_q4 == nullptr && s.head_dim == HD && s.n_head == (int64_t) G * s.n_head_kv &&
        cap > 0 && ids && steps && pools.page_table) {
        if (pools.k_q != nullptr) {
            if (!pools.v_q || !pools.k_scale || !pools.v_scale) return false;
            return launch_wmma11<1>(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
        }
        if (!pools.k_pool || !pools.v_pool) return false;
        return launch_wmma11<0>(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);
    }
#endif
    return false;
#endif
"""
old = "<<<<<<< HEAD\n#if defined(__HIPCC__)\n    // the tensor-core kernels are compiled out on AMD"
assert s.count(old) == 1
s = s.replace(old, "#if defined(__HIPCC__)\n    // the tensor-core kernels are compiled out on AMD")
old = "        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);\n    return false;\n#endif\n=======\n>>>>>>> pr-313\n"
assert s.count(old) == 1
s = s.replace(old, "        return launch_wmma(q, pools, ids, steps, cap, s, attn, n_q, (cudaStream_t) stream);\n" + gfx11)
# the PR's later HIP-only dispatch (unreachable after HEAD's `return false`) goes
start = s.index("#if defined(STRATA_USE_HIP)\n#if defined(STRATA_WMMA_GFX11) && STRATA_WMMA_GFX11\n    static const bool pa_wmma_enabled")
end = s.index("    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4", start)
s = s[:start] + s[end:]
if "<<<<<<<" in s or ">>>>>>>" in s or "\n=======\n" in s:
    sys.exit("markers left in qsa_prompt_attn.cu")

# RDNA 3.5 in the PR's device guards
s = s.replace("defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)",
              "defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)")
p.write_text(s, encoding="utf-8", newline="\n")
print("qsa_prompt_attn.cu resolved; gfx1151 guards:", s.count("defined(__gfx1151__)"),
      "launch_wmma11:", s.count("launch_wmma11"), "wmma11 kernel:", s.count("prompt_attn_wmma11_kernel"))

# ---- src/prefill/wmma_gemm.cu: the same guards
p = root / "src/prefill/wmma_gemm.cu"
s = p.read_text(encoding="utf-8")
n = s.count("defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)")
s = s.replace("defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)",
              "defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || defined(__gfx1151__)")
s = s.replace("// src/prefill/wmma_gemm.cu - RDNA3 WMMA FP16 & BF16 GEMM for Strata prefill (gfx1100).",
              "// src/prefill/wmma_gemm.cu - RDNA3 / RDNA 3.5 WMMA FP16 & BF16 GEMM for Strata prefill (gfx1100, gfx1151).")
p.write_text(s, encoding="utf-8", newline="\n")
print("wmma_gemm.cu guards:", n)

# ---- docs/AMD_HIP.md: both sides
p = root / "docs/AMD_HIP.md"
s = p.read_text(encoding="utf-8")
s = s.replace("<<<<<<< HEAD\n", "", 1)
s = s.replace("=======\n## RDNA3 WMMA kernels", "\n## RDNA3 WMMA kernels", 1)
s = s.replace(">>>>>>> pr-313\n", "", 1)
if "<<<<<<<" in s or ">>>>>>>" in s:
    sys.exit("markers left in AMD_HIP.md")
p.write_text(s, encoding="utf-8", newline="\n")
print("AMD_HIP.md resolved")
