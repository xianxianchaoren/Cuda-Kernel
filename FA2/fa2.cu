#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

_global__void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* O) {
    int tx = threadIdx.x; // 当前线程
    int batch_id = blockIdx.x; // 当前batch idx
    int head_id = blockIdx.y; // 当前head idx

    const int batch_element_offset = N * d * gridDim.y; // 一个batch中的数据量
    const int head_element_offset = N * d; // 一个head中的数据量
    int base_offset = (batch_id * batch_element_offset) + (head_id * head_element_offset); // 定位到某一head

    extern __shared__ float sram[]; // 定义QKVO的共享内存变量
    int tile_size_q = Br * d;
    int tile_size_kv = Bc * d;

    float *Qi = sram; // shared memory for Qi
    float *Ki = sram + tile_size_q; // shared memory for ki
    float *Vi = sram + tile_size_q + tile_size_kv; // shared memory for Vi
    float *Oi = sram + tile_size_q + 2 * tile_size_kv + tile_size_kv; // shared memory for Oi
    float *S = sram + 2 * tile_size_q + 2* tile_size_kv; // shared memory for S

    for(int i = 0; i < Tr; i++)
    {
        // load q into Qi
        // TODO 这里可以进行访存合并
        for(int x = 0;x < d;x++)
        {
            Qi[tx * d + x] = Q[base_offset + i * tile_size_q + tx * d + x];
        }

        float row_m_prev = -INFINITY;
        float row_l_prev = 0;

        for(int j = 0;j<Tc;j++)
        {
            // load k v into Ki Qi
            for (int k = tx; k < Bc; k += blockDim.x){ // 每个线程负责N行搬运，共Br个线程
                for(int x = 0; x < d; x++){
                    Ki[k * d + x] = K[base_offset + j * tile_size_kv + k * d + x];
                    Vi[k * d + x] = V[base_offset + j * tile_size_kv + k * d + x];
                }
            }
            __syncthreads(); // 等待所有线程加载完数据

            // S = QK^T row_m = rowmax(S)
            float row_m = -INFINITY;
            for(int y = 0;y < Bc;y++) // 进行矩阵乘法
            {
                float sum = 0;
                for(int x = 0; x < d; x++)
                {
                    sum += Qi[tx * d + x] * Ki[y * d + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                if(sum > row_m)
                    row_m = sum;
            }

            // P = exp(S-row_m), row_l = rowsum(P)
            float row_l = 0;
            for(int y = 0; y < Bc; y++)
            {
                S[(Bc * tx) + y] = expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = row_l_prev * expf(row_m_prev - row_m_new) + row_l * expf(row_m - row_m_new);

            // Compute new Qi
            for(int x =0 ; x < d; x++)
            {
                float pv = 0; // Pij * Vj
                for (int y = 0; y < Bc; y++)
                    pv += S[(Bc * tx) + y] * Vi[y * d + x];
                Oi[tx * d + x] = (row_l_prev * expf(row_m_prev - row_m_new) * Oi[y * d + x] + pv * expf(row_m - row_m_new)) / row_l_new;

            }
            row_m_prev = row_m_new;
            row_l_prev = row_l_new;
        }
        
        // write Oi to O-HBM
        for(int x =0 ;x<d;x++)
        {
            O[base_offset + i * tile_size_q + tx * d + x] = Oi[tx * d + x];
        }
        __syncthreads();

    }
}


torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // best condition is: Bc == Br
    const int Bc = 32; const int Br = 32;

    // B: batch size / nh: number of heads / N: sequence length / d: dimension of each head
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    const int sram_size = (2 * Bc * d * sizeof(float)) + (2 * Br * d * sizeof(float)) + (2 * Bc * Br * sizeof(float));
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        O.data_ptr<float>()
    );
    return 0;
}