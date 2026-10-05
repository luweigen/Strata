// src/prefill/gdn_chunk_wmma.cu - the chunked GDN prompt recurrence on the matrix cores (docs/TODO.md item 5).
//
// The kernels are llama.cpp's chunked Gated DeltaNet prefill (ggml/src/ggml-cuda/chunk_gated_delta_net.cu, MIT,
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES, with the RDNA/CDNA fragment-layout fixes of the EngramHalo.cpp
// checkout that measured it on this card), adapted to Strata's buffers:
//   - q, k and v are read straight from the conv output h ([T][C]: q heads, k heads, v heads), with C as the token
//     stride of all three (llama.cpp: q/k contiguous, v strided);
//   - V_corr is written into the output rows y ([T][HV * S]) instead of a scratch buffer, and the state kernel reads it
//     from there before it writes the same rows: one block owns a (head, value tile) for the whole walk;
//   - the state is Strata's [k][head][v] (llama.cpp: [head][v][k]); one sequence; wave32 only (Strata's HIP target).
// What it computes is Strata's recurrence S_t = g_t S_{t-1} + k_t d_t^T, d_t = beta_t (v_t - g_t S_{t-1}^T k_t),
// o_t = S_t^T q_t / sqrt(128) in 16-token chunks: the intra-chunk solve exact in FP32, the products (Q K^T, the chunk
// against the state, the state update) as FP16 x FP16 -> FP32 WMMA, the state itself carried in FP32 registers.
// The FP16 operands are the numerical difference from the serial kernels (tests/hip/prefill_gdn_chunk_parity.cpp).
#include "common.cuh"
#include "mma.cuh"

#include "strata/prefill/kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if !defined(GGML_USE_HIP)
#error "gdn_chunk_wmma.cu is built for the HIP prompt path only"
#endif

#if defined(AMD_WMMA_AVAILABLE) || defined(AMD_MFMA_AVAILABLE)
#define SGDN_TC_AVAILABLE 1
#endif
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
#define SGDN_C_DL ggml_cuda_mma::DATA_LAYOUT_J_MAJOR
#else
#define SGDN_C_DL ggml_cuda_mma::DATA_LAYOUT_I_MAJOR
#endif
#define SGDN_AB_DL ggml_cuda_mma::get_input_data_layout()
#if defined(AMD_MFMA_AVAILABLE)
#define SGDN_LOAD ggml_cuda_mma::load_generic
#else
#define SGDN_LOAD ggml_cuda_mma::load_ldmatrix
#endif

namespace strata::prefill {
namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240, YV = HV * S;
constexpr int CS = 16, BK = 128;

// C[0:16][c_col:c_col+16] = A[16 x BK] @ B[16 x BK]^T, fp16 in, fp32 out, row-major into shared memory
template <int K_>
__device__ __forceinline__ void gemm_ABt_16(const __half* s_a, const __half* s_b, float* s_c, int ldc, int c_col) {
#ifdef SGDN_TC_AVAILABLE
    ggml_cuda_mma::tile<16, 16, float, SGDN_C_DL> acc;
#pragma unroll
    for (int kt = 0; kt < K_ / 16; kt++) {
        ggml_cuda_mma::tile<16, 8, half2, SGDN_AB_DL> ta, tb;
        SGDN_LOAD(ta, (const half2*) s_a + kt * 8, K_ / 2);
        SGDN_LOAD(tb, (const half2*) s_b + kt * 8, K_ / 2);
        ggml_cuda_mma::mma(acc, ta, tb);
    }
#pragma unroll
    for (int l = 0; l < acc.ne; l++) s_c[acc.get_i(l) * ldc + c_col + acc.get_j(l)] = acc.x[l];
#endif
}
// delta[v][0:BK] = sum_t Vnew[t][v] * Kch[t][k], stored V-major into s_hdelta[v * BK + k]
template <int BV>
__device__ __forceinline__ void gemm_ktv(const __half* s_vnew, const __half* s_kch, float* s_hdelta, int m_off) {
#ifdef SGDN_TC_AVAILABLE
    ggml_cuda_mma::tile<16, 8, half2, SGDN_AB_DL> x_vnew;
    ggml_cuda_mma::load_ldmatrix_trans(x_vnew, (const half2*) s_vnew + m_off / 2, BV / 2);
#pragma unroll
    for (int nk = 0; nk < BK; nk += 16) {
        ggml_cuda_mma::tile<16, 8, half2, SGDN_AB_DL> y_kch;
        ggml_cuda_mma::load_ldmatrix_trans(y_kch, (const half2*) (s_kch + nk), BK / 2);
        ggml_cuda_mma::tile<16, 16, float, SGDN_C_DL> acc;
        ggml_cuda_mma::mma(acc, x_vnew, y_kch);
#pragma unroll
        for (int l = 0; l < acc.ne; l++) s_hdelta[(m_off + acc.get_i(l)) * BK + nk + acc.get_j(l)] = acc.x[l];
    }
#endif
}
// O[0:16][v] = sum_t' qk[t][t'] * Vnew[t'][v], row-major into s_out[t * BV + v]
template <int BV>
__device__ __forceinline__ void gemm_qkv(const __half* s_qk, const __half* s_vnew, float* s_out, int n_off) {
#ifdef SGDN_TC_AVAILABLE
    ggml_cuda_mma::tile<16, 8, half2, SGDN_AB_DL> x_qk, y_vnew;
    SGDN_LOAD(x_qk, (const half2*) s_qk, 16 / 2);
    ggml_cuda_mma::load_ldmatrix_trans(y_vnew, (const half2*) s_vnew + n_off / 2, BV / 2);
    ggml_cuda_mma::tile<16, 16, float, SGDN_C_DL> acc;
    ggml_cuda_mma::mma(acc, x_qk, y_vnew);
#pragma unroll
    for (int l = 0; l < acc.ne; l++) s_out[acc.get_i(l) * BV + n_off + acc.get_j(l)] = acc.x[l];
#endif
}
__device__ __forceinline__ __half to_fp16(float v) { return __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f)); }

// Stage 1, per (head, chunk), exact FP32: G = cumsum of the log gates; L[t][s] = beta_t exp(G_t - G_s) k_t.k_s (s < t);
// (I + L) x = b solved for b = beta exp(G) k (K_cumdecay) and b = beta v (V_corr, into y's rows).
__launch_bounds__(128, 4) __global__ void fwdsub_kernel(const float* __restrict__ h, const float* __restrict__ beta,
                                                         const float* __restrict__ gate, float* __restrict__ vcorr,
                                                         float* __restrict__ kcd, float* __restrict__ gcum_out,
                                                         int T, int nch) {
    constexpr int SK = BK + 1;
    __shared__ float s_K[CS * SK], s_L[CS * CS], s_gcum[CS], s_beta[CS];
    const int tid = threadIdx.x, hv = blockIdx.x, ch = blockIdx.y, hk = hv % HK, t_off = ch * CS;
    const int valid = min(CS, T - t_off);
    const float* k_chunk = h + (int64_t) t_off * C + HK * S + hk * S;
    const float* v_chunk = h + (int64_t) t_off * C + 2 * HK * S + hv * S;
    for (int i = tid; i < CS * BK; i += 128) {
        const int t = i / BK, k = i % BK;
        s_K[t * SK + k] = t < valid ? k_chunk[(int64_t) t * C + k] : 0.0f;
    }
    if (tid < CS) {
        s_beta[tid] = tid < valid ? beta[(int64_t) (t_off + tid) * HV + hv] : 0.0f;
        s_gcum[tid] = tid < valid ? gate[(int64_t) (t_off + tid) * HV + hv] : 0.0f;
    }
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f;
        for (int t = 0; t < CS; t++) { a += s_gcum[t]; s_gcum[t] = a; }
    }
    __syncthreads();
    for (int idx = tid; idx < CS * CS; idx += 128) {
        const int t = idx / CS, s = idx % CS;
        float l = 0.0f;
        if (s < t) {
            float kk = 0.0f;
            for (int k = 0; k < BK; k++) kk += s_K[t * SK + k] * s_K[s * SK + k];
            l = s_beta[t] * __expf(s_gcum[t] - s_gcum[s]) * kk;
        }
        s_L[idx] = l;
    }
    __syncthreads();
    const int64_t bc = (int64_t) hv * nch + ch;
    if (tid < CS) gcum_out[bc * CS + tid] = s_gcum[tid];
    {
        float x[CS];
        for (int t = 0; t < CS; t++) {
            float xt = s_beta[t] * __expf(s_gcum[t]) * s_K[t * SK + tid];
            for (int s = 0; s < t; s++) xt -= s_L[t * CS + s] * x[s];
            x[t] = xt;
        }
        for (int t = 0; t < CS; t++) kcd[(bc * CS + t) * BK + tid] = x[t];
    }
    __syncthreads();
    for (int i = tid; i < CS * BK; i += 128) {
        const int t = i / BK, v = i % BK;
        s_K[t * SK + v] = t < valid ? v_chunk[(int64_t) t * C + v] * s_beta[t] : 0.0f;
    }
    __syncthreads();
    {
        float x[CS];
        for (int t = 0; t < CS; t++) {
            float xt = s_K[t * SK + tid];
            for (int s = 0; s < t; s++) xt -= s_L[t * CS + s] * x[s];
            x[t] = xt;
        }
        for (int t = 0; t < valid; t++) vcorr[(int64_t) (t_off + t) * YV + hv * S + tid] = x[t];
    }
}

// Stage 2, per (head, chunk), one wave: qk[i][j] = (q_i / sqrt(128)) . k_j exp(G_i - G_j) for j <= i, on the WMMA
__launch_bounds__(32, 8) __global__ void preqk_kernel(const float* __restrict__ h, const float* __restrict__ gcum,
                                                       float* __restrict__ qk, float scale, int T, int nch) {
#ifdef SGDN_TC_AVAILABLE
    __shared__ __align__(16) __half s_Q[CS * BK];
    __shared__ __align__(16) __half s_K[CS * BK];
    __shared__ float s_g[CS], s_acc[CS * CS];
    const int hv = blockIdx.x, ch = blockIdx.y, tid = threadIdx.x, hk = hv % HK, t_off = ch * CS;
    const int valid = min(CS, T - t_off);
    const int64_t bc = (int64_t) hv * nch + ch;
    if (tid < CS) s_g[tid] = gcum[bc * CS + tid];
    const float* q_chunk = h + (int64_t) t_off * C + hk * S;
    const float* k_chunk = h + (int64_t) t_off * C + HK * S + hk * S;
    for (int i = tid; i < CS * BK; i += 32) {
        const int row = i / BK, col = i % BK;
        s_Q[i] = __float2half(row < valid ? q_chunk[(int64_t) row * C + col] * scale : 0.0f);
        s_K[i] = __float2half(row < valid ? k_chunk[(int64_t) row * C + col] : 0.0f);
    }
    __syncthreads();
    gemm_ABt_16<BK>(s_Q, s_K, s_acc, CS, 0);
    __syncthreads();
    float* o = qk + bc * CS * CS;
    for (int f = tid; f < CS * CS; f += 32) {
        const int r = f / CS, c = f % CS;
        o[f] = c <= r ? s_acc[f] * __expf(s_g[r] - s_g[c]) : 0.0f;
    }
#else
    __trap();
#endif
}

// Stage 3, per (head, value tile of BV): the walk over the chunks with the state tile in FP32 registers and an FP16
// copy in shared memory for the WMMA.  y holds V_corr on entry and the output o / sqrt(128) on exit, row by row.
template <int BV, int NT, int OCC>
__launch_bounds__(NT, OCC) __global__ void state_kernel(float* y, const float* __restrict__ kcd,
                                                         const float* __restrict__ h, const float* __restrict__ gcum,
                                                         const float* __restrict__ qkb, float* __restrict__ state,
                                                         float scale, int T, int nch) {
#ifdef SGDN_TC_AVAILABLE
    static_assert(BV % 16 == 0 && NT % 32 == 0 && (CS * BV) % NT == 0 && (BK * BV) % NT == 0, "tile shapes");
    static_assert(NT / 32 >= BV / 16, "a wave per 16 value columns");
    constexpr int H_BYTES = BK * BV * 2, KBUF_BYTES = CS * BK * 2, RES_BYTES = CS * BV * 4, G_BYTES = CS * 4;
    extern __shared__ __align__(16) char smem[];
    __half* s_h16 = reinterpret_cast<__half*>(smem);
    __half* s_kbuf = reinterpret_cast<__half*>(smem + H_BYTES);
    float* s_res = reinterpret_cast<float*>(smem + H_BYTES + KBUF_BYTES);
    float* s_g = reinterpret_cast<float*>(smem + H_BYTES + KBUF_BYTES + RES_BYTES);
    float* s_hd = reinterpret_cast<float*>(smem + H_BYTES + KBUF_BYTES + RES_BYTES + G_BYTES);
    __half* s_vnew = reinterpret_cast<__half*>(s_res);
    constexpr int EPT = CS * BV / NT, EPT_H = BK * BV / NT, N_TILES = BV / 16;
    const int hv = blockIdx.x, v_off = blockIdx.y * BV, hk = hv % HK;
    const int warp = threadIdx.y, tid = threadIdx.y * 32 + threadIdx.x;
    const float* q_base = h + hk * S;
    const float* k_base = h + HK * S + hk * S;
    float* y_h = y + hv * S + v_off;
    float hr[EPT_H];   // the state tile, [v][k] V-major: element idx is v = idx / BK, k = idx % BK
    for (int j = 0; j < EPT_H; j++) {
        const int idx = tid + j * NT, v = idx / BK, k = idx % BK;
        hr[j] = state[((int64_t) k * HV + hv) * S + v_off + v];
        s_h16[idx] = to_fp16(hr[j]);
    }
    __syncthreads();
    for (int ci = 0; ci < nch; ci++) {
        const int64_t bc = (int64_t) hv * nch + ci;
        const int t_off = ci * CS, valid = min(CS, T - t_off);
        const float* kcd_c = kcd + bc * CS * BK;
        float vn[EPT], oi[EPT];
        for (int i = tid; i < CS; i += NT) s_g[i] = gcum[bc * CS + i];
        for (int i = tid; i < CS * BK; i += NT) s_kbuf[i] = to_fp16(kcd_c[i]);
        __syncthreads();
        if (warp < N_TILES) gemm_ABt_16<BK>(s_kbuf, s_h16 + warp * 16 * BK, s_res, BV, warp * 16);   // K_cumdecay H
        __syncthreads();
        for (int j = 0; j < EPT; j++) {   // v_new = V_corr - K_cumdecay H (padded rows: 0 - 0)
            const int idx = tid + j * NT, t = idx / BV, v = idx % BV;
            const float vc = t < valid ? y_h[(int64_t) (t_off + t) * YV + v] : 0.0f;
            vn[j] = vc - s_res[t * BV + v];
        }
        __syncthreads();
        for (int i = tid; i < CS * BK; i += NT) {
            const int t = i / BK, k = i % BK;
            s_kbuf[i] = to_fp16(t < valid ? q_base[(int64_t) (t_off + t) * C + k] * scale : 0.0f);
        }
        __syncthreads();
        if (warp < N_TILES) gemm_ABt_16<BK>(s_kbuf, s_h16 + warp * 16 * BK, s_res, BV, warp * 16);   // (q scale) H
        __syncthreads();
        for (int j = 0; j < EPT; j++) {
            const int idx = tid + j * NT, t = idx / BV, v = idx % BV;
            oi[j] = s_res[t * BV + v] * __expf(fminf(s_g[t], 88.72f));
        }
        __syncthreads();
        const float g_last = s_g[CS - 1];
        for (int i = tid; i < BK * CS; i += NT) {
            const int t = i / BK, k = i % BK;
            s_kbuf[t * BK + k] = to_fp16(t < valid ? k_base[(int64_t) (t_off + t) * C + k] * __expf(g_last - s_g[t]) : 0.0f);
        }
        for (int j = 0; j < EPT; j++) s_vnew[tid + j * NT] = to_fp16(vn[j]);
        __syncthreads();
        if (warp < N_TILES) gemm_ktv<BV>(s_vnew, s_kbuf, s_hd, warp * 16);   // delta[v][k] = sum_t vnew k'
        __syncthreads();
        const float eg = __expf(g_last);
        for (int j = 0; j < EPT_H; j++) hr[j] = eg * hr[j] + s_hd[tid + j * NT];
        __syncthreads();
        for (int j = 0; j < EPT_H; j++) s_h16[tid + j * NT] = to_fp16(hr[j]);
        {
            __half* s_qk = s_kbuf;
            const float* qk_c = qkb + bc * CS * CS;
            for (int i = tid; i < CS * CS; i += NT) s_qk[i] = to_fp16(qk_c[i]);
            __syncthreads();
            if (warp < N_TILES) gemm_qkv<BV>(s_qk, s_vnew, s_hd, warp * 16);   // the intra-chunk output
            __syncthreads();
            for (int j = 0; j < EPT; j++) {
                const int idx = tid + j * NT, t = idx / BV, v = idx % BV;
                if (t < valid) y_h[(int64_t) (t_off + t) * YV + v] = s_hd[t * BV + v] + oi[j];
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < EPT_H; j++) {
        const int idx = tid + j * NT, v = idx / BK, k = idx % BK;
        state[((int64_t) k * HV + hv) * S + v_off + v] = hr[j];
    }
#else
    __trap();
#endif
}

}  // namespace

bool gdn_chunk_wmma(float* state, const float* h, const float* gate, const float* beta, float* y, int64_t T,
                    float* scratch, size_t scratch_floats, void* stream) {
    if (T <= 0 || scratch == nullptr || T > (int64_t) 1 << 30) return false;
    const int nch = (int) ((T + CS - 1) / CS);
    const size_t per = (size_t) HV * nch * CS;
    if (scratch_floats < per * (BK + 1 + CS)) return false;   // K_cumdecay, G, QK
    float* kcd = scratch;
    float* gc = kcd + per * BK;
    float* qk = gc + per;
    const cudaStream_t cs = (cudaStream_t) stream;
    const float scale = 0.08838834764831845f;   // 1 / sqrt(128)
    // STRATA_GDN_CHUNK_TIMING=1: each kernel's time from events, printed per call (debug; it synchronizes)
    static const bool timing = std::getenv("STRATA_GDN_CHUNK_TIMING") != nullptr;
    hipEvent_t ev[4];
    if (timing) {
        for (auto& e : ev) (void) hipEventCreate(&e);
        (void) hipEventRecord(ev[0], cs);
    }
    fwdsub_kernel<<<dim3(HV, nch), 128, 0, cs>>>(h, beta, gate, y, kcd, gc, (int) T, nch);
    if (timing) (void) hipEventRecord(ev[1], cs);
    preqk_kernel<<<dim3(HV, nch), 32, 0, cs>>>(h, gc, qk, scale, (int) T, nch);
    if (timing) (void) hipEventRecord(ev[2], cs);
    // the walk: llama.cpp's tile, 32 value columns per block (192 blocks of 256 threads); 16 columns (384 blocks of
    // 128) measured slower on the 8060S, 18-21 against 12-16 ms at T = 4,165
    constexpr int BV = 32, NT = 256, OCC = 4;
    const size_t smem = (size_t) BK * BV * 2 + CS * BK * 2 + CS * BV * 4 + CS * 4 + (size_t) BK * BV * 4;
    state_kernel<BV, NT, OCC><<<dim3(HV, S / BV), dim3(32, NT / 32), smem, cs>>>(y, kcd, h, gc, qk, state, scale, (int) T, nch);
    if (timing) {
        (void) hipEventRecord(ev[3], cs);
        (void) hipEventSynchronize(ev[3]);
        float a = 0, b = 0, c = 0;
        (void) hipEventElapsedTime(&a, ev[0], ev[1]);
        (void) hipEventElapsedTime(&b, ev[1], ev[2]);
        (void) hipEventElapsedTime(&c, ev[2], ev[3]);
        std::fprintf(stderr, "gdn chunk T=%lld: fwdsub %.2f ms, preqk %.2f ms, walk %.2f ms\n", (long long) T, a, b, c);
        for (auto& e : ev) (void) hipEventDestroy(e);
    }
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::prefill
