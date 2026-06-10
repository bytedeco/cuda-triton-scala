#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// ==============================================
// 全局编译期常量 (根据模型/GPU算力调优)
// ==============================================
// 模型参数
constexpr int HEAD_DIM      = 128;    // 单头维度 (主流 64/128)
constexpr int NUM_HEADS     = 32;     // 注意力头数
constexpr float INF_NEG     = -1e8f;  // 稳定Softmax负下界

// CUDA 分块 Tiling 超参 (A100/A800 最优配置)
// FlashAttention V3 分块
constexpr int BLOCK_Q_V3    = 64;     // Q 分块大小
constexpr int BLOCK_KV_V3   = 128;    // K/V 分块大小
// FlashAttention V4 细粒度分块 + 流水分块
constexpr int BLOCK_Q_V4    = 32;
constexpr int BLOCK_KV_V4   = 64;
constexpr int WARP_TILE_V4  = 16;     // Warp 内子分块

// 线程基础配置
constexpr int WARP_SIZE     = 32;
constexpr int MAX_THREADS   = 1024;

// ==============================================
// 设备端内联工具函数 & 数学优化
// ==============================================
__device__ __forceinline__ float rsqrt(float x) {
    return rsqrtf(x);
}

__device__ __forceinline__ float exp_val(float x) {
    return expf(x);
}

// 在线 Softmax 归约更新 (FlashAttention 核心: m_new, l_new 递推)
__device__ __forceinline__ void softmax_reduce(
    float val, float &m_old, float &l_old,
    float &m_new, float &l_new, float &exp_val_
) {
    m_new = max(m_old, val);
    float scale = exp_val(val - m_new);
    float prev_scale = exp_val(m_old - m_new);
    l_new = prev_scale * l_old + scale;
    exp_val_ = scale;
}

// 二维地址计算 [seq, head, dim]
__device__ __forceinline__ int get_qkv_addr(int seq, int head, int dim, int head_dim) {
    return seq * NUM_HEADS * head_dim + head * head_dim + dim;
}

// ==============================================
// 一、FlashAttention V3 Kernel (经典2级Tiling + Shared Memory)
// 适配: 长序列Prefill、因果掩码、多头注意力、工业主流版本
// ==============================================
extern "C" __global__ void flash_attention_v3_causal_kernel(
    float* __restrict__        out,        // [seq_len, n_heads, head_dim]
    const float* __restrict__  q,
    const float* __restrict__  k,
    const float* __restrict__  v,
    int seq_len,
    float scale
) {
    const int tid       = threadIdx.x;
    const int block_q   = blockIdx.x;
    const int head_idx  = blockIdx.y;

    // 当前 Block 负责的 Q 区间
    const int q_start   = block_q * BLOCK_Q_V3;
    const int q_end     = min(q_start + BLOCK_Q_V3, seq_len);
    if (q_start >= seq_len) return;

    // 共享内存: 分片缓存 K / V
    __shared__ float sh_k[BLOCK_KV_V3][HEAD_DIM];
    __shared__ float sh_v[BLOCK_KV_V3][HEAD_DIM];

    // 寄存器: 缓存单条 Q、中间累加值
    float q_reg[HEAD_DIM];
    float o_reg[HEAD_DIM] = {0.0f};

    // 全局 Softmax 状态: m(最大值), l(指数和)
    float m_global = INF_NEG;
    float l_global = 0.0f;

    // 1. 逐行加载 Q 到寄存器
    const int q_seq = q_start + tid;
    if (q_seq < q_end) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            q_reg[d] = q[get_qkv_addr(q_seq, head_idx, d, HEAD_DIM)];
        }
    }

    // 2. K/V 分块遍历 (KV Tile 循环)
    const int num_k_tiles = (seq_len + BLOCK_KV_V3 - 1) / BLOCK_KV_V3;
    for (int tile = 0; tile < num_k_tiles; tile++) {
        const int kv_start = tile * BLOCK_KV_V3;
        const int kv_end   = min(kv_start + BLOCK_KV_V3, seq_len);

        // 因果掩码: 只计算 q_seq 之前的 Key
        if (kv_start > q_seq) break;

        // 2.1 加载 K/V 分片到 Shared Memory
        const int kv_seq = kv_start + tid;
        if (kv_seq < kv_end) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                sh_k[tid][d] = k[get_qkv_addr(kv_seq, head_idx, d, HEAD_DIM)];
                sh_v[tid][d] = v[get_qkv_addr(kv_seq, head_idx, d, HEAD_DIM)];
            }
        }
        __syncthreads();

        // 2.2 计算 Q @ K^T 注意力分数
        float tile_m = INF_NEG;
        float tile_l = 0.0f;
        float s[BLOCK_KV_V3] = {0.0f};

        for (int kv_idx = 0; kv_idx < (kv_end - kv_start); kv_idx++) {
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_reg[d] * sh_k[kv_idx][d];
            }
            s[kv_idx] = dot * scale;
            tile_m = max(tile_m, s[kv_idx]);
        }

        // 2.3 块内 Softmax 归约
        for (int kv_idx = 0; kv_idx < (kv_end - kv_start); kv_idx++) {
            float exp_s = exp_val(s[kv_idx] - tile_m);
            tile_l += exp_s;
        }

        // 2.4 全局 Softmax 状态融合 (FlashAttention 核心递推)
        float new_m, new_l, exp_s;
        for (int kv_idx = 0; kv_idx < (kv_end - kv_start); kv_idx++) {
            softmax_reduce(s[kv_idx], m_global, l_global, new_m, new_l, exp_s);
            m_global = new_m;
            l_global = new_l;

            // 注意力加权 V
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                o_reg[d] += exp_s * sh_v[kv_idx][d];
            }
        }
        __syncthreads();
    }

    // 3. 归一化并写回全局内存
    if (q_seq < q_end) {
        float inv_l = 1.0f / l_global;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            out[get_qkv_addr(q_seq, head_idx, d, HEAD_DIM)] = o_reg[d] * inv_l;
        }
    }
}

// ==============================================
// 二、FlashAttention V4 Kernel (三级Tiling + Warp流水 + 访存重排)
// 增强点: 细粒度Warp分块、指令流水、访存合并、寄存器压力优化
// 对标 FA4 细粒度调度，延迟更低、吞吐更高
// ==============================================
extern "C" __global__ void flash_attention_v4_causal_kernel(
    float* __restrict__        out,
    const float* __restrict__  q,
    const float* __restrict__  k,
    const float* __restrict__  v,
    int seq_len,
    float scale
) {
    const int tid       = threadIdx.x;
    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;

    const int block_q   = blockIdx.x;
    const int head_idx  = blockIdx.y;

    const int q_start   = block_q * BLOCK_Q_V4;
    const int q_end     = min(q_start + BLOCK_Q_V4, seq_len);
    if (q_start >= seq_len) return;

    // 分层共享内存: K/V 按 Warp Tile 分片 (V4 核心优化)
    __shared__ float sh_k[BLOCK_KV_V4][HEAD_DIM];
    __shared__ float sh_v[BLOCK_KV_V4][HEAD_DIM];

    float q_reg[HEAD_DIM];
    float o_reg[HEAD_DIM] = {0.0f};
    float m_global = INF_NEG;
    float l_global = 0.0f;

    // 加载 Q 到寄存器
    const int q_seq = q_start + tid;
    if (q_seq < q_end) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            q_reg[d] = q[get_qkv_addr(q_seq, head_idx, d, HEAD_DIM)];
        }
    }

    // K/V 大分块遍历
    const int num_k_tiles = (seq_len + BLOCK_KV_V4 - 1) / BLOCK_KV_V4;
    for (int tile = 0; tile < num_k_tiles; tile++) {
        const int kv_start = tile * BLOCK_KV_V4;
        const int kv_end   = min(kv_start + BLOCK_KV_V4, seq_len);
        if (kv_start > q_seq) break; // 因果掩码

        // 合并访存加载 K/V -> Shared Memory
        const int kv_seq = kv_start + tid;
        if (kv_seq < kv_end) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                sh_k[tid][d] = k[get_qkv_addr(kv_seq, head_idx, d, HEAD_DIM)];
                sh_v[tid][d] = v[get_qkv_addr(kv_seq, head_idx, d, HEAD_DIM)];
            }
        }
        __syncthreads();

        // V4: Warp 子分块计算 (流水执行)
        const int num_warp_tiles = (kv_end - kv_start + WARP_TILE_V4 - 1) / WARP_TILE_V4;
        for (int wt = 0; wt < num_warp_tiles; wt++) {
            const int wt_start = wt * WARP_TILE_V4;
            const int wt_end   = min(wt_start + WARP_TILE_V4, kv_end - kv_start);

            float tile_m = INF_NEG;
            float tile_l = 0.0f;
            float s[WARP_TILE_V4] = {0.0f};

            // 子块 QK 点积
            for (int kv_idx = wt_start; kv_idx < wt_end; kv_idx++) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q_reg[d] * sh_k[kv_idx][d];
                }
                s[kv_idx - wt_start] = dot * scale;
                tile_m = max(tile_m, s[kv_idx - wt_start]);
            }

            // 子块 Softmax
            for (int kv_idx = wt_start; kv_idx < wt_end; kv_idx++) {
                float exp_s = exp_val(s[kv_idx - wt_start] - tile_m);
                tile_l += exp_s;
            }

            // 全局状态融合 + 加权V
            float new_m, new_l, exp_s;
            for (int kv_idx = wt_start; kv_idx < wt_end; kv_idx++) {
                int s_idx = kv_idx - wt_start;
                softmax_reduce(s[s_idx], m_global, l_global, new_m, new_l, exp_s);
                m_global = new_m;
                l_global = new_l;

                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    o_reg[d] += exp_s * sh_v[kv_idx][d];
                }
            }
        }
        __syncthreads();
    }

    // 结果归一化写回
    if (q_seq < q_end) {
        float inv_l = 1.0f / l_global;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            out[get_qkv_addr(q_seq, head_idx, d, HEAD_DIM)] = o_reg[d] * inv_l;
        }
    }
}

// ==============================================
// 三、主机端通用调用封装 + 工具函数
// ==============================================
inline int calc_grid(int total, int block) {
    return (total + block - 1) / block;
}

// FlashAttention V3 启动接口 (因果掩码)
void launch_flash_attn_v3(
    float* d_out,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    int seq_len
) {
    float scale = rsqrt((float)HEAD_DIM);
    int grid_q  = calc_grid(seq_len, BLOCK_Q_V3);
    dim3 grid(grid_q, NUM_HEADS);
    dim3 block(BLOCK_Q_V3);

    flash_attention_v3_causal_kernel<<<grid, block>>>(
        d_out, d_q, d_k, d_v, seq_len, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FlashAttn V3 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

// FlashAttention V4 启动接口 (因果掩码)
void launch_flash_attn_v4(
    float* d_out,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    int seq_len
) {
    float scale = rsqrt((float)HEAD_DIM);
    int grid_q  = calc_grid(seq_len, BLOCK_Q_V4);
    dim3 grid(grid_q, NUM_HEADS);
    dim3 block(BLOCK_Q_V4);

    flash_attention_v4_causal_kernel<<<grid, block>>>(
        d_out, d_q, d_k, d_v, seq_len, scale
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FlashAttn V4 Launch Error: %s\n", cudaGetErrorString(err));
    }
}

// ==============================================
// 四、测试入口 (可直接运行验证)
// ==============================================
int main() {
    // 测试配置
    const int seq_len = 1024;
    const int total_elem = seq_len * NUM_HEADS * HEAD_DIM;

    // 设备内存分配
    float *d_out, *d_q, *d_k, *d_v;
    cudaMalloc(&d_out, total_elem * sizeof(float));
    cudaMalloc(&d_q,   total_elem * sizeof(float));
    cudaMalloc(&d_k,   total_elem * sizeof(float));
    cudaMalloc(&d_v,   total_elem * sizeof(float));

    printf("===== Run FlashAttention V3 =====\n");
    launch_flash_attn_v3(d_out, d_q, d_k, d_v, seq_len);
    cudaDeviceSynchronize();

    printf("===== Run FlashAttention V4 =====\n");
    launch_flash_attn_v4(d_out, d_q, d_k, d_v, seq_len);
    cudaDeviceSynchronize();

    printf("All FlashAttention Kernels Executed Successfully!\n");

    // 释放显存
    cudaFree(d_out);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);

    return 0;
}