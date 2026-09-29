#include <gemm/registry.h>

namespace {

constexpr int TILE = 16;

__global__ void gemm_tiled16_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        const int aCol = k0 + threadIdx.x;
        const int bRow = k0 + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
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

void launch_gemm_tiled16(const float* d_A, const float* d_B, float* d_C, int M,
                         int N, int K, cudaStream_t stream) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled16_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("tiled16", launch_gemm_tiled16);
