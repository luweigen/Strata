// docs/benchmarks/2026-10-05-halo-gdn-chunk-fp32.cu - TODO 5's first try, kept for the record (not built): the chunked
// GDN recurrence in FP32 on the plain ALUs, as it stood in src/prefill/kernels.cu before the WMMA port replaced it.
// Parity against the serial kernels: y rel_l2 3.1e-7, state 2-5e-7 (2026-10-05-halo-gdn-chunk-parity-fp32.log).
// Time at T = 4,165 on the 8060S: 45 ms against the serial kernels' 26 ms (prep 32 ms, walk 25.6 ms with 220
// VGPRs spilled). It does the serial form's FMA count, so without matrix cores it cannot win.

// ---------------------------------------------------------------- GDN, chunked (TODO 5)
// The recurrence above, S_t = g_t S_{t-1} + k_t d_t^T with d_t = beta_t (v_t - g_t S_{t-1}^T k_t) and o_t = S_t^T q_t,
// in the chunkwise form of the Gated DeltaNet paper.  Inside a chunk of CL tokens, with G_t the sum of the log gates
// from the chunk's first token to t (so exp G_t is the decay since the chunk start):
//   A[t][i] = beta_t exp(G_t - G_i) k_t.k_i  (i < t)      B[t][i] = exp(G_t - G_i) q_t.k_i  (i <= t)
//   Tm = (I + A)^-1                                        Dw = Tm W, W_i = beta_i exp(G_i) k_i;  Du = Tm U, U_i = beta_i v_i
//   D  = Du - Dw S_0           (the chunk's d_t rows)      O  = exp(G) Q S_0 + B D
//   S_CL = exp(G_CL) S_0 + K'^T D, K'_i = exp(G_CL - G_i) k_i
// Everything is FP32 on the plain ALUs (no matrix cores): the serial kernels' arithmetic reassociated, not rounded
// to 16 bits.  A, B, Tm, Dw and Du depend on the chunk's tokens only, so one block per (chunk, head) computes them
// for every chunk at once (gdn_chunk_prep_kernel); the walk from chunk to chunk carries only S, and its value columns
// are independent, so it runs on HV * S/CBV blocks with the state slice in shared memory (gdn_chunk_scan_kernel).
// The serial dependency is CL tokens per step instead of one.  What the 8060S taught (the parity test's phase stamps,
// docs/benchmarks/2026-10-05-halo-hip-blockops.log): a load behind a per-element condition is issued and waited for
// one at a time, so every global batch is loaded into registers with clamped row indices and masked afterwards, and
// the batch for the next stage is loaded while this stage computes; the products are 4x4 register tiles fed by float4
// shared loads; a 64-token chunk's forward substitution is a 64-step dependent chain and its column blocks re-read the
// chunk's operands four times, so the chunk is 32 tokens and a block covers 64 value columns.
__device__ float g_gdn_clock[2];   // debug: the shader clock seen by the walk's block 0 (MHz sum over chunks, chunks)
#define GDN_SHADER_CYCLES_REG (29 | (0 << 6) | ((20 - 1) << 11))
constexpr int CL = 32;         // the chunk length
constexpr int CLP = CL + 4;    // the row stride of the transposed tiles: rows stay 16-byte aligned for float4 loads
constexpr int CBV = 64;        // value columns per scan block
// bm: [nchunks][HV][CL][CL] (B), cgo: [nchunks][HV][CL] (G), du: Du into y's rows, dw: Dw into the v slots of h (== h:
// the v rows are consumed before they are overwritten, so h loses its v after this kernel)
__global__ void __launch_bounds__(256) gdn_chunk_prep_kernel(const float* h, const float* __restrict__ gate,
                                                             const float* __restrict__ beta, float* dw,
                                                             float* __restrict__ bm, float* __restrict__ cgo,
                                                             float* __restrict__ du, int64_t T) {
    __shared__ __align__(16) float sKt[S * CLP];        // K^T [d][t]
    __shared__ __align__(16) float sQt[S * CLP];        // Q^T [d][t]
    __shared__ float sA[CL * (CL + 1)], sT[CL * (CL + 1)];   // A and Tm [t][i]
    __shared__ __align__(16) float sTT[CL * CLP];       // Tm^T scaled: [i][t] = Tm[t][i] beta_i
    __shared__ float sg[CL], sb[CL];
    const int c = blockIdx.x, head = blockIdx.y, qh = head % HK, tid = threadIdx.x;
    const int64_t t0 = (int64_t) c * CL;
    const int tv = (int) ((T - t0) < CL ? (T - t0) : CL);   // this chunk's tokens (the last chunk may be short)
    // the chunk's log gates and betas; G is the running sum from the chunk start (a padded token: 0, it decays nothing)
    if (tid < CL) {
        sg[tid] = tid < tv ? gate[(t0 + tid) * HV + head] : 0.0f;
        sb[tid] = tid < tv ? beta[(t0 + tid) * HV + head] : 0.0f;
    }
    // K and Q of the chunk, transposed; every load issued before the first store, a padded row reads a valid one
    {
        constexpr int J = CL * S / 256;
        float rk[J], rq[J];
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int e = tid + j * 256, t = e >> 7, d = e & 127, tc = t < tv ? t : tv - 1;
            const float* ht = h + (t0 + tc) * C;   // d runs across the threads: the reads of h coalesce
            rk[j] = ht[HK * S + qh * S + d];
            rq[j] = ht[qh * S + d];
        }
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int e = tid + j * 256, t = e >> 7, d = e & 127;
            sKt[d * CLP + t] = t < tv ? rk[j] : 0.0f;
            sQt[d * CLP + t] = t < tv ? rq[j] : 0.0f;
        }
    }
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f;
        for (int t = 0; t < CL; ++t) { a += sg[t]; sg[t] = a; }
    }
    // k_t.k_i and q_t.k_i: a thread's tile is 2 t x 2 i, the 16 x 16 tiles cover the 32 x 32 (the upper ones idle)
    const int tt = tid >> 4, ti = tid & 15;
    float aA[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}}, aB[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    if (tt >= ti) {
        for (int d = 0; d < S; ++d) {
            const float2 kt = *(const float2*) (sKt + d * CLP + tt * 2);
            const float2 ki = *(const float2*) (sKt + d * CLP + ti * 2);
            const float2 qt = *(const float2*) (sQt + d * CLP + tt * 2);
            aA[0][0] = fmaf(kt.x, ki.x, aA[0][0]); aA[0][1] = fmaf(kt.x, ki.y, aA[0][1]);
            aA[1][0] = fmaf(kt.y, ki.x, aA[1][0]); aA[1][1] = fmaf(kt.y, ki.y, aA[1][1]);
            aB[0][0] = fmaf(qt.x, ki.x, aB[0][0]); aB[0][1] = fmaf(qt.x, ki.y, aB[0][1]);
            aB[1][0] = fmaf(qt.y, ki.x, aB[1][0]); aB[1][1] = fmaf(qt.y, ki.y, aB[1][1]);
        }
    }
    __syncthreads();   // sg is summed
    if (tid < CL) cgo[((size_t) c * HV + head) * CL + tid] = sg[tid];
    float* bg = bm + ((size_t) c * HV + head) * CL * CL;
#pragma unroll
    for (int x = 0; x < 2; ++x)
#pragma unroll
        for (int yy = 0; yy < 2; ++yy) {
            const int t = tt * 2 + x, i = ti * 2 + yy;
            const float r = __expf(sg[t] - sg[i]);
            sA[t * (CL + 1) + i] = (i < t) ? sb[t] * r * aA[x][yy] : 0.0f;
            bg[t * CL + i] = (i <= t) ? r * aB[x][yy] : 0.0f;
        }
    __syncthreads();
    // Tm = (I + A)^-1 by forward substitution: a column per 8 lanes, the rows in order; Tm^T beta alongside
    {
        const int j = tid >> 3, q = tid & 7;
        for (int t = 0; t < CL; ++t) {
            float s = 0.0f;
            if (t > j)
                for (int i = j + q; i < t; i += 8) s = fmaf(sA[t * (CL + 1) + i], sT[i * (CL + 1) + j], s);
            s += __shfl_xor_sync(0xffffffffu, s, 1);
            s += __shfl_xor_sync(0xffffffffu, s, 2);
            s += __shfl_xor_sync(0xffffffffu, s, 4);
            if (q == 0) {
                const float v = (t == j ? 1.0f : 0.0f) - s;
                sT[t * (CL + 1) + j] = v;
                sTT[j * CLP + t] = v * sb[j];
            }
            __syncthreads();
        }
    }
    // Du = Tm U into y's rows and Dw = Tm W into the v slots of h: a 4 t x 4 column tile of each per thread, the K and
    // V rows straight from h (float4, the lanes contiguous in the column), Tm^T beta from shared memory (broadcast)
    {
        const int t4 = tid >> 5, k4 = tid & 31;   // t = t4 * 4.., column = k4 * 4..
        float aw[4][4], au[4][4];
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int yy = 0; yy < 4; ++yy) { aw[x][yy] = 0.0f; au[x][yy] = 0.0f; }
        for (int ib = 0; ib < CL; ib += 8) {
            float4 rk[8], rv[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = ib + j, ic = i < tv ? i : tv - 1;
                rk[j] = *(const float4*) (h + (t0 + ic) * C + HK * S + qh * S + k4 * 4);
                rv[j] = *(const float4*) (h + (t0 + ic) * C + 2 * HK * S + head * S + k4 * 4);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = ib + j;
                const float4 tt4 = *(const float4*) (sTT + i * CLP + t4 * 4);
                const float gi = __expf(sg[i]);
                const float tv4[4] = {tt4.x, tt4.y, tt4.z, tt4.w};
                const float kv[4] = {rk[j].x * gi, rk[j].y * gi, rk[j].z * gi, rk[j].w * gi};
                const float vv[4] = {rv[j].x, rv[j].y, rv[j].z, rv[j].w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int yy = 0; yy < 4; ++yy) {
                        aw[x][yy] = fmaf(tv4[x], kv[yy], aw[x][yy]);
                        au[x][yy] = fmaf(tv4[x], vv[yy], au[x][yy]);
                    }
            }
        }
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int t = t4 * 4 + x;
            if (t < tv) *(float4*) (du + (t0 + t) * (HV * S) + head * S + k4 * 4) = make_float4(au[x][0], au[x][1], au[x][2], au[x][3]);
        }
        __syncthreads();   // every V row is read before the v slots take Dw
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int t = t4 * 4 + x;
            if (t < tv) *(float4*) (dw + (t0 + t) * C + 2 * HK * S + head * S + k4 * 4) = make_float4(aw[x][0], aw[x][1], aw[x][2], aw[x][3]);
        }
    }
}
// The walk over the chunks: a block per (head, CBV value columns), the state slice [S][CBV] in shared memory.  Per
// chunk, in stages: Du; P = Dw S and O1 = exp(G) Q S over four 32-row slices of k; D = Du - P; O = O1 + B D; S = exp(G_CL) S
// + K'^T D over four 32-row slices of k.  A stage's global batch is loaded into registers while the stage before it
// computes; the products are 4x4 register tiles.
__global__ void __launch_bounds__(CL * CBV / 16) gdn_chunk_scan_kernel(float* __restrict__ state, const float* h,
                                                                        const float* dw, const float* __restrict__ bm,
                                                                        const float* __restrict__ cg,
                                                                        float* __restrict__ y, int64_t T) {
    constexpr int NT = CL * CBV / 16, NCT = CBV / 4;   // 128 threads; 16 column tiles of 4 by 8 token tiles of 4
    static_assert(NT == 8 * NCT, "the tile map");
    constexpr int JD = CL * CBV / NT;                  // Du elements per thread (16)
    constexpr int J32 = CL * 32 / NT;                  // elements of a [CL][32] slice per thread (8)
    __shared__ __align__(16) float sS[S * CBV];        // the state slice [k][c]
    __shared__ __align__(16) float sX[2 * 32 * CLP];   // Dw and Q' as [kk][t] | B as [i][t] | K' as [t][kk]
    __shared__ __align__(16) float sD[CL * CBV];       // Du, then D [t][c]
    __shared__ float sg[CL];
    const int head = blockIdx.x / (S / CBV), cb = blockIdx.x % (S / CBV), qh = head % HK, tid = threadIdx.x;
    const int tc = tid % NCT, tt = tid / NCT, c0 = cb * CBV;
#pragma unroll
    for (int j = 0; j < S * CBV / NT; ++j) {
        const int e = tid + j * NT;
        sS[e] = state[((size_t) (e / CBV) * HV + head) * S + c0 + e % CBV];
    }
    const int nch = (int) ((T + CL - 1) / CL);
    float R[JD];          // the batch in flight
    float rg = 0.0f;      // the chunk's G (threads < CL), with the Du batch
    // the loads of a stage: st 0 Du (+ G), 1-4 Dw and Q' slices, 5 B, 6-9 K' slices
    auto prefetch = [&](int c, int st) {
        if (c >= nch) return;
        const int64_t t0 = (int64_t) c * CL;
        const int tv = (int) ((T - t0) < CL ? (T - t0) : CL);
        if (st == 0) {
            if (tid < CL) rg = cg[((size_t) c * HV + head) * CL + tid];
#pragma unroll
            for (int j = 0; j < JD; ++j) {
                const int e = tid + j * NT, t = e / CBV, tcl = t < tv ? t : tv - 1;
                R[j] = y[(t0 + tcl) * (HV * S) + head * S + c0 + e % CBV];
            }
        } else if (st <= 4) {
            const int ks = st - 1;
#pragma unroll
            for (int j = 0; j < J32; ++j) {
                const int e = tid + j * NT, t = e >> 5, kk = e & 31, tcl = t < tv ? t : tv - 1;
                R[j] = dw[(t0 + tcl) * C + 2 * HK * S + head * S + ks * 32 + kk];
                R[J32 + j] = h[(t0 + tcl) * C + qh * S + ks * 32 + kk];
            }
        } else if (st == 5) {
            const float* bc = bm + ((size_t) c * HV + head) * CL * CL;
#pragma unroll
            for (int j = 0; j < J32; ++j) {
                const int e = tid + j * NT, t = e >> 5, ii = e & 31;
                R[j] = bc[t * CL + ii];
            }
        } else {
            const int kh = st - 6;
#pragma unroll
            for (int j = 0; j < J32; ++j) {
                const int e = tid + j * NT, t = e >> 5, kk = e & 31, tcl = t < tv ? t : tv - 1;
                R[j] = h[(t0 + tcl) * C + HK * S + qh * S + kh * 32 + kk];
            }
        }
    };
    // the batch into shared memory (after the barrier that ends the previous stage's compute)
    auto commit = [&](int c, int st) {
        const int64_t t0 = (int64_t) c * CL;
        const int tv = (int) ((T - t0) < CL ? (T - t0) : CL);
        if (st == 0) {
            if (tid < CL) sg[tid] = rg;
#pragma unroll
            for (int j = 0; j < JD; ++j) {
                const int e = tid + j * NT, t = e / CBV;
                sD[e] = t < tv ? R[j] : 0.0f;
            }
        } else if (st <= 4) {
#pragma unroll
            for (int j = 0; j < J32; ++j) {
                const int e = tid + j * NT, t = e >> 5, kk = e & 31;
                sX[kk * CLP + t] = t < tv ? R[j] : 0.0f;
                sX[(32 + kk) * CLP + t] = t < tv ? __expf(sg[t]) * R[J32 + j] : 0.0f;
            }
        } else if (st == 5) {
#pragma unroll
            for (int j = 0; j < J32; ++j) {
                const int e = tid + j * NT, t = e >> 5, ii = e & 31;
                sX[ii * CLP + t] = R[j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < J32; ++j) {
                const int e = tid + j * NT, t = e >> 5, kk = e & 31;
                sX[t * CLP + kk] = t < tv ? __expf(sg[CL - 1] - sg[t]) * R[j] : 0.0f;
            }
        }
    };
    prefetch(0, 0);
    float mhz_sum = 0.0f;
    for (int c = 0; c < nch; ++c) {
        const int64_t t0 = (int64_t) c * CL;
        const int tv = (int) ((T - t0) < CL ? (T - t0) : CL);
        long long w0 = 0; unsigned cy0 = 0;
        if (blockIdx.x == 0 && tid == 0) { w0 = wall_clock64(); cy0 = __builtin_amdgcn_s_getreg(GDN_SHADER_CYCLES_REG); }
        float aP[4][4], aO[4][4];
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int yy = 0; yy < 4; ++yy) { aP[x][yy] = 0.0f; aO[x][yy] = 0.0f; }
        // stage 0: Du
        commit(c, 0);
        prefetch(c, 1);
        __syncthreads();   // sg and sD are in (sg is read by the commits below)
        // stages 1-4: P and O1 over the k slices
        for (int ks = 0; ks < 4; ++ks) {
            commit(c, 1 + ks);
            prefetch(c, 2 + ks);
            __syncthreads();
            for (int kk = 0; kk < 32; ++kk) {
                const float4 s4 = *(const float4*) (sS + (ks * 32 + kk) * CBV + tc * 4);
                const float4 w4 = *(const float4*) (sX + kk * CLP + tt * 4);
                const float4 q4 = *(const float4*) (sX + (32 + kk) * CLP + tt * 4);
                const float sv[4] = {s4.x, s4.y, s4.z, s4.w}, wv[4] = {w4.x, w4.y, w4.z, w4.w},
                            qv[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int yy = 0; yy < 4; ++yy) {
                        aP[x][yy] = fmaf(wv[x], sv[yy], aP[x][yy]);
                        aO[x][yy] = fmaf(qv[x], sv[yy], aO[x][yy]);
                    }
            }
            __syncthreads();
        }
        // D = Du - P (the thread's own tile), then stage 5: O += B D
#pragma unroll
        for (int x = 0; x < 4; ++x)
#pragma unroll
            for (int yy = 0; yy < 4; ++yy) sD[(tt * 4 + x) * CBV + tc * 4 + yy] -= aP[x][yy];
        commit(c, 5);
        prefetch(c, 6);
        __syncthreads();
        for (int ii = 0; ii < CL; ++ii) {
            const float4 b4 = *(const float4*) (sX + ii * CLP + tt * 4);
            const float4 d4 = *(const float4*) (sD + ii * CBV + tc * 4);
            const float bv[4] = {b4.x, b4.y, b4.z, b4.w}, dv[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int yy = 0; yy < 4; ++yy) aO[x][yy] = fmaf(bv[x], dv[yy], aO[x][yy]);
        }
        const float sc = rsqrtf((float) S);
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            const int t = tt * 4 + x;
            if (t < tv)
                *(float4*) (y + (t0 + t) * (HV * S) + head * S + c0 + tc * 4) =
                    make_float4(aO[x][0] * sc, aO[x][1] * sc, aO[x][2] * sc, aO[x][3] * sc);
        }
        __syncthreads();
        // stages 6-9: S = exp(G_CL) S + K'^T D over the k slices; a thread's tile is 4 k x 4 c
        const float gL = __expf(sg[CL - 1]);
        for (int kh = 0; kh < 4; ++kh) {
            commit(c, 6 + kh);
            if (kh < 3) prefetch(c, 7 + kh); else prefetch(c + 1, 0);
            __syncthreads();
            const int tk = tid / NCT;   // 0..7: the slice's 32 rows of k in tiles of 4
            float aS[4][4];
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int yy = 0; yy < 4; ++yy) aS[x][yy] = 0.0f;
            for (int t = 0; t < CL; ++t) {
                const float4 k4 = *(const float4*) (sX + t * CLP + tk * 4);
                const float4 d4 = *(const float4*) (sD + t * CBV + tc * 4);
                const float kv[4] = {k4.x, k4.y, k4.z, k4.w}, dv[4] = {d4.x, d4.y, d4.z, d4.w};
#pragma unroll
                for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int yy = 0; yy < 4; ++yy) aS[x][yy] = fmaf(kv[x], dv[yy], aS[x][yy]);
            }
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int yy = 0; yy < 4; ++yy) {
                    const int idx = (kh * 32 + tk * 4 + x) * CBV + tc * 4 + yy;
                    sS[idx] = fmaf(gL, sS[idx], aS[x][yy]);
                }
            __syncthreads();
        }
        if (blockIdx.x == 0 && tid == 0) {
            const unsigned cy1 = __builtin_amdgcn_s_getreg(GDN_SHADER_CYCLES_REG);
            const long long w1 = wall_clock64();
            mhz_sum += ((cy1 - cy0) & 0xFFFFFu) / ((w1 - w0) / 100.0f);   // cycles per us
        }
    }
    if (blockIdx.x == 0 && tid == 0) { g_gdn_clock[0] = mhz_sum; g_gdn_clock[1] = (float) nch; }
#pragma unroll
    for (int j = 0; j < S * CBV / NT; ++j) {
        const int e = tid + j * NT;
        state[((size_t) (e / CBV) * HV + head) * S + c0 + e % CBV] = sS[e];
    }
}


bool gdn_recurrence_chunked(float* state, const float* h, const float* gate, const float* beta, const float* z,
                            const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, float* scratch,
                            size_t scratch_floats, void* stream) {
    if (T <= 0 || scratch == nullptr) return false;
    const size_t nch = (size_t) ((T + CL - 1) / CL), tp = nch * CL;
    const size_t need = tp * HV * CL + tp * HV;   // B, G (Dw takes the v slots of h, Du the rows of y)
    if (scratch_floats < need) return false;
    float* bm = scratch;
    float* cg = bm + tp * HV * CL;
    float* dw = const_cast<float*>(h);
    const cudaStream_t cs = (cudaStream_t) stream;
    // STRATA_GDN_CHUNK_PHASES (timing only; the output is wrong with either bit off): 1 the prep kernel, 2 the walk
    static const int phases = [] { const char* v = std::getenv("STRATA_GDN_CHUNK_PHASES"); return v ? std::atoi(v) : 3; }();
    if (phases & 1) gdn_chunk_prep_kernel<<<dim3((unsigned) nch, HV), 256, 0, cs>>>(h, gate, beta, dw, bm, cg, y, T);
    if (phases & 2) gdn_chunk_scan_kernel<<<HV * (S / CBV), CL * CBV / 16, 0, cs>>>(state, h, dw, bm, cg, y, T);
    if (phases & 4) {
        float clk[2] = {0.0f, 0.0f};
        cudaStreamSynchronize(cs);
        (void) hipMemcpyFromSymbol(clk, HIP_SYMBOL(g_gdn_clock), sizeof clk, 0, hipMemcpyDeviceToHost);
        std::fprintf(stderr, "gdn chunk walk, block 0: shader clock %.0f MHz over %.0f chunks\n", clk[0] / clk[1], clk[1]);
    }
    gdn_out_norm_kernel<<<dim3((unsigned) T, HV), S, 0, cs>>>(z, gamma, eps, y, y16);
    check("gdn_recurrence_chunked");
    return true;
}
