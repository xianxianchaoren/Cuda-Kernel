#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// 论文式 FlashAttention-2：grid 按 Q tile 展开，一个 block 负责一个 Q tile。
// blockIdx.x = Q tile 序号 (Tr)，blockIdx.y = batch，blockIdx.z = head。
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Bc, const int Br, const float softmax_scale,
                    float* O) {
    int tx = threadIdx.x;   // 行号：每个线程负责本 tile 的一行
    int i  = blockIdx.x;    // 本 block 负责的 Q tile
    int b  = blockIdx.y;    // batch
    int h  = blockIdx.z;    // head

    // 定位到 (batch, head) 段
    int qkv_offset = (b * gridDim.z + h) * N * d;

    extern __shared__ float sram[];
    int tile_q  = Br * d;   // Qi / Oi 大小
    int tile_kv = Bc * d;   // Kj / Vj 大小
    float* Qi = sram;
    float* Kj = sram + tile_q;
    float* Vj = sram + tile_q + tile_kv;
    float* Oi = sram + tile_q + 2 * tile_kv;
    float* S  = sram + 2 * tile_q + 2 * tile_kv;

    // 只加载本 block 的 Q tile 一次，Oi 清零
    for (int x = 0; x < d; x++) {
        Qi[tx * d + x] = Q[qkv_offset + i * tile_q + tx * d + x];
        Oi[tx * d + x] = 0.0f;
    }

    // online softmax 的累计量放在寄存器
    float row_m_prev = -INFINITY;
    float row_l_prev = 0.0f;

    for (int j = 0; j < Tc; j++) {
        // 加载 Kj, Vj（跨线程，所以下面要 __syncthreads）
        for (int k = tx; k < Bc; k += blockDim.x) {
            for (int x = 0; x < d; x++) {
                Kj[k * d + x] = K[qkv_offset + j * tile_kv + k * d + x];
                Vj[k * d + x] = V[qkv_offset + j * tile_kv + k * d + x];
            }
        }
        __syncthreads();

        // S = QK^T, row_m = rowmax(S)
        float row_m = -INFINITY;
        for (int y = 0; y < Bc; y++) {
            float sum = 0;
            for (int x = 0; x < d; x++)
                sum += Qi[tx * d + x] * Kj[y * d + x];
            sum *= softmax_scale;
            S[Bc * tx + y] = sum;
            if (sum > row_m)
                row_m = sum;
        }

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
            S[Bc * tx + y] = __expf(S[Bc * tx + y] - row_m);
            row_l += S[Bc * tx + y];
        }

        // 更新 online softmax 的 m, l
        float row_m_new = max(row_m_prev, row_m);
        float row_l_new = __expf(row_m_prev - row_m_new) * row_l_prev
                        + __expf(row_m - row_m_new) * row_l;

        // Oi = P V，并在 shared memory 里做在线累加（不再回 HBM）
        for (int x = 0; x < d; x++) {
            float pv = 0;
            for (int y = 0; y < Bc; y++)
                pv += S[Bc * tx + y] * Vj[y * d + x];
            Oi[tx * d + x] = (1.0f / row_l_new)
                * (row_l_prev * __expf(row_m_prev - row_m_new) * Oi[tx * d + x]
                   + __expf(row_m - row_m_new) * pv);
        }

        row_m_prev = row_m_new;
        row_l_prev = row_l_new;
        __syncthreads();   // 下一轮会覆盖 Kj/Vj
    }

    // 本 Q tile 算完，一次性写回 HBM
    for (int x = 0; x < d; x++)
        O[qkv_offset + i * tile_q + tx * d + x] = Oi[tx * d + x];
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32; const int Br = 32;

    // B: batch / nh: heads / N: seq len / d: head dim
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);

    // S 只占一份
    const int sram_size = (2 * Br * d + 2 * Bc * d + Bc * Br) * sizeof(float);

    // 关键区别：grid 按 Q tile 展开，一个 block 只算一个 Q tile
    dim3 grid_dim(Tr, B, nh);
    dim3 block_dim(Br);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Bc, Br, softmax_scale, O.data_ptr<float>());

    return O;
}
