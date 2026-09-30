#include <torch/types.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// FA-minimal-v5  = v4 + float4 向量化全局加载
//   Step A: shared memory 每行 padding 1 个 float -> 消除 32 路 bank conflict
//   Step B: 全局访存改成整个 block 线性协作搬运，并用 float4 把每条指令的
//           搬运量从 4B 提到 16B（加载指令数降为 1/4）；
//           smem 侧因为 padding(DS=D+1) 不是 16B 对齐，仍是 4 个标量写
//           —— 若把 padding 改成 D+4 来对齐，smem 会涨到 26112 B，
//              每 SM 的 block 数从 4 掉到 3，得不偿失
//   Step C: blockDim 32 -> 128（32 行 x 4 线程协作一行），
//           Oi / P 全部放寄存器，smem 只剩 Q/K/V -> 24960 B -> 4 blocks/SM
// ---------------------------------------------------------------------------

#define BR 32                 // 每个 block 负责的 Q 行数
#define BC 32                 // 每轮加载的 K/V 行数
#define NW 4                  // 每行由 4 个线程协作
#define NTHREADS (BR * NW)    // 128 线程 = 4 warp

// 4 个协作线程（lane 连续）之间做归约
__device__ __forceinline__ float reduce_max4(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

__device__ __forceinline__ float reduce_sum4(float v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}

template <int D>
__global__ void forward_kernel(const float* Q, const float* K, const float* V,
                               const int N, const int Tc, const float softmax_scale,
                               float* O) {
    constexpr int DS = D + 1;       // padding 后的 smem 行 stride
    constexpr int NY = BC / NW;     // 本线程负责的 S 列数
    constexpr int NX = D / NW;      // 本线程负责的输出列数
    constexpr int DV = D / 4;       // 每行的 float4 个数（向量化访存用）
    static_assert(D % 4 == 0, "D 必须是 4 的倍数才能走 float4 路径");

    const int tid = threadIdx.x;
    const int row = tid / NW;       // 本 tile 内的行号
    const int g   = tid % NW;       // 行内第几个协作线程

    const int i = blockIdx.x;       // Q tile
    const int b = blockIdx.y;       // batch
    const int h = blockIdx.z;       // head

    const int qkv_offset = (b * gridDim.z + h) * N * D;
    const int q_off = i * BR * D;

    extern __shared__ __align__(16) float sram[];
    float* Qi = sram;               // BR * DS
    float* Kj = Qi + BR * DS;       // BC * DS
    float* Vj = Kj + BC * DS;       // BC * DS

    // ---- Step B: 线性合并拷贝 Q tile ----
    // float4 向量化：每条指令搬 16B，连续线程访问连续 float4 -> 完美合并；
    // smem 侧因为 padding(DS=D+1) 不是 16B 对齐，所以拆成 4 个标量写。
    const float4* Qv = reinterpret_cast<const float4*>(Q + qkv_offset + q_off);
    for (int idx = tid; idx < BR * DV; idx += NTHREADS) {
        const int r = idx / DV;
        const int c = (idx % DV) * 4;
        const float4 v = Qv[idx];
        Qi[r * DS + c + 0] = v.x;
        Qi[r * DS + c + 1] = v.y;
        Qi[r * DS + c + 2] = v.z;
        Qi[r * DS + c + 3] = v.w;
    }
    __syncthreads();

    float m_prev = -INFINITY;
    float l_prev = 0.f;
    float acc[NX];
#pragma unroll
    for (int t = 0; t < NX; t++) acc[t] = 0.f;

    const float* qrow = Qi + row * DS;

    for (int j = 0; j < Tc; j++) {
        // ---- Step B: 线性合并拷贝 K/V tile（float4 向量化） ----
        const int kv_off = qkv_offset + j * BC * D;
        const float4* Kv = reinterpret_cast<const float4*>(K + kv_off);
        const float4* Vv = reinterpret_cast<const float4*>(V + kv_off);
        for (int idx = tid; idx < BC * DV; idx += NTHREADS) {
            const int r = idx / DV;
            const int c = (idx % DV) * 4;
            const float4 vk = Kv[idx];
            const float4 vv = Vv[idx];
            Kj[r * DS + c + 0] = vk.x;
            Kj[r * DS + c + 1] = vk.y;
            Kj[r * DS + c + 2] = vk.z;
            Kj[r * DS + c + 3] = vk.w;
            Vj[r * DS + c + 0] = vv.x;
            Vj[r * DS + c + 1] = vv.y;
            Vj[r * DS + c + 2] = vv.z;
            Vj[r * DS + c + 3] = vv.w;
        }
        __syncthreads();

        // S = Q K^T * scale：本线程算 NY 列，每列做完整 D 维点积（无冗余计算）
        // 列交错分配 y = yi*NW + g -> smem 无 bank conflict
        float s[NY];
#pragma unroll
        for (int yi = 0; yi < NY; yi++) {
            const float* krow = Kj + (yi * NW + g) * DS;
            float sum = 0.f;
#pragma unroll
            for (int x = 0; x < D; x++) sum += qrow[x] * krow[x];
            s[yi] = sum * softmax_scale;
        }

        // rowmax：本线程先归约，再跨 4 个协作线程 shuffle 归约
        float m = -INFINITY;
#pragma unroll
        for (int yi = 0; yi < NY; yi++) m = fmaxf(m, s[yi]);
        m = reduce_max4(m);

        // P = exp(S - m)，rowsum
        float p[NY];
        float l = 0.f;
#pragma unroll
        for (int yi = 0; yi < NY; yi++) {
            p[yi] = __expf(s[yi] - m);
            l += p[yi];
        }
        l = reduce_sum4(l);

        // online softmax 的 m, l（同行的 4 个线程结果一致）
        const float m_new = fmaxf(m_prev, m);
        const float alpha = __expf(m_prev - m_new);
        const float beta  = __expf(m - m_new);
        const float l_new = alpha * l_prev + beta * l;

        // Step C: 把 P 在 4 个协作线程间凑齐到寄存器（每线程拿全 BC 个 P）
        float Pall[BC];
        const int src0 = tid - g;
#pragma unroll
        for (int r = 0; r < NW; r++)
#pragma unroll
            for (int yi = 0; yi < NY; yi++)
                Pall[yi * NW + r] = __shfl_sync(0xffffffffu, p[yi], src0 + r);

        // Step C: 在线累加 O = P V（O 一直在寄存器里）
        const float inv_l = 1.f / l_new;
        const float decay = alpha * l_prev * inv_l;
#pragma unroll
        for (int y = 0; y < BC; y++) Pall[y] *= beta * inv_l;
#pragma unroll
        for (int t = 0; t < NX; t++) acc[t] *= decay;
#pragma unroll
        for (int y = 0; y < BC; y++) {
            const float py = Pall[y];
            const float* vrow = Vj + y * DS + g;
#pragma unroll
            for (int t = 0; t < NX; t++) acc[t] += py * vrow[t * NW];
        }

        m_prev = m_new;
        l_prev = l_new;
        __syncthreads();   // 下一轮会覆盖 Kj/Vj
    }

    // 本 Q tile 算完，一次性写回 HBM
#pragma unroll
    for (int t = 0; t < NX; t++)
        O[qkv_offset + q_off + row * D + t * NW + g] = acc[t];
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    TORCH_CHECK(N % BR == 0 && N % BC == 0,
                "v4 要求 N 能被 ", BR, " 整除，当前 N=", N);
    TORCH_CHECK(d == 32 || d == 64 || d == 128,
                "v4 目前只支持 d = 32/64/128，当前 d=", d);

    const int Tc = N / BC;
    const int Tr = N / BR;
    const float softmax_scale = 1.0f / sqrtf((float)d);

    auto O = torch::zeros_like(Q);

    // S 和 Oi 都不在 smem 里了：只有 Qi + Kj + Vj（含 padding）
    const int sram_size = (BR * (d + 1) + 2 * BC * (d + 1)) * sizeof(float);

    dim3 grid_dim(Tr, B, nh);
    dim3 block_dim(NTHREADS);

    const float* qp = Q.data_ptr<float>();
    const float* kp = K.data_ptr<float>();
    const float* vp = V.data_ptr<float>();
    float* op = O.data_ptr<float>();

    switch (d) {
        case 32:
            forward_kernel<32><<<grid_dim, block_dim, sram_size>>>(qp, kp, vp, N, Tc, softmax_scale, op);
            break;
        case 64:
            forward_kernel<64><<<grid_dim, block_dim, sram_size>>>(qp, kp, vp, N, Tc, softmax_scale, op);
            break;
        default:
            forward_kernel<128><<<grid_dim, block_dim, sram_size>>>(qp, kp, vp, N, Tc, softmax_scale, op);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return O;
}
