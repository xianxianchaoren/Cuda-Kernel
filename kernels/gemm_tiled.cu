// gemm_tiled.cu —— 共享内存 tiling 版 GEMM
//
// v1: 每个 block 负责一个 TILE x TILE 的输出块，
//     外层循环以 TILE 为步长遍历 K，每步把 A 的 TILE x TILE 分块和
//     B 的 TILE x TILE 分块载入 shared memory，然后计算（两次 __syncthreads 同步）。
//     相比 naive，消除了对全局内存的重复访问（每个 A/B 元素只从 global 读一次）。
//
// 注意：M/N/K 不一定能被 TILE 整除，越界位置补 0。

#include <gemm/common.h>
#include <gemm/registry.h>

namespace {

constexpr int TILE = 32;

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 协作加载 A 的 [row, k0:k0+TILE) 分块
        const int aCol = k0 + threadIdx.x;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // 协作加载 B 的 [k0:k0+TILE, col) 分块
        const int bRow = k0 + threadIdx.y;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

}  // namespace

void launch_gemm_tiled(const float* d_A, const float* d_B, float* d_C, int M,
                       int N, int K, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("tiled", launch_gemm_tiled);
