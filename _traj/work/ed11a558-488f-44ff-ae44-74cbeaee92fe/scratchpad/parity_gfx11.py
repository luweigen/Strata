import pathlib, sys
p = pathlib.Path(r"E:\work\AI\Strata\src\kernels\qsa_prompt_attn_parity.cpp")
s = p.read_text(encoding="utf-8")
edits = [
    ('int main(int argc, char** argv) {\n#if defined(__HIP_PLATFORM_AMD__)\n',
     'int main(int argc, char** argv) {\n    bool gfx11 = false;   // AMD: the gfx11 kernel also takes FP16 KV pools (fmt 0)\n#if defined(__HIP_PLATFORM_AMD__)\n'),
    ('        if (std::strncmp(prop.gcnArchName, "gfx12", 5) != 0) {\n'
     '            std::printf("SKIP: %s is not gfx12 (the matrix-core prompt attention is RDNA4 only)\\n", prop.gcnArchName);\n'
     '            return 77;\n'
     '        }\n'
     '#if defined(_WIN32)\n'
     '        _putenv_s("STRATA_HIP_WMMA", "1");\n'
     '#else\n'
     '        setenv("STRATA_HIP_WMMA", "1", 1);\n'
     '#endif\n',
     '        // gfx12: the S6 kernel (STRATA_HIP_WMMA, int8 KV only); gfx11 / gfx11.5: PR #313\'s kernel (STRATA_PA_WMMA, int8\n'
     '        // and FP16 KV); anything else skips\n'
     '        const char* sw = std::strncmp(prop.gcnArchName, "gfx12", 5) == 0 ? "STRATA_HIP_WMMA"\n'
     '                       : std::strncmp(prop.gcnArchName, "gfx11", 5) == 0 ? "STRATA_PA_WMMA" : nullptr;\n'
     '        if (sw == nullptr) {\n'
     '            std::printf("SKIP: %s is neither gfx12 nor gfx11 (no matrix-core prompt attention for it)\\n", prop.gcnArchName);\n'
     '            return 77;\n'
     '        }\n'
     '        gfx11 = sw[7] == \'P\';\n'
     '        std::printf("matrix-core prompt attention under test: %s=1 on %s\\n", sw, prop.gcnArchName);\n'
     '#if defined(_WIN32)\n'
     '        _putenv_s(sw, "1");\n'
     '#else\n'
     '        setenv(sw, "1", 1);\n'
     '#endif\n'),
    ('#if !defined(__HIP_PLATFORM_AMD__)\n    fails += run(0, ctx, nq, reps);   // FP16 KV: the RDNA4 kernel takes int8 KV only\n#endif\n',
     '#if !defined(__HIP_PLATFORM_AMD__)\n    fails += run(0, ctx, nq, reps);   // FP16 KV\n#else\n'
     '    if (gfx11) fails += run(0, ctx, nq, reps);   // FP16 KV: the RDNA4 kernel takes int8 KV only, the gfx11 one both\n#endif\n'),
]
for old, new in edits:
    n = s.count(old)
    if n != 1:
        sys.exit(f"found {n} of: {old[:80]!r}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("parity test extended to gfx11")
