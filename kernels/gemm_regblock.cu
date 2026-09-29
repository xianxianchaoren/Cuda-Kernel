#include <gemm/registry.h>

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void gemm_regblock_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int index = tid; index < BM * BK; index += blockDim.x * blockDim.y) {
            const int row = index / BK;
            const int col = index % BK;
            const int globalRow = blockRow + row;
            const int globalCol = k0 + col;
            sA[row][col] = (globalRow < M && globalCol < K)
                                ? A[globalRow * K + globalCol]
                                : 0.0f;
        }

        for (int index = tid; index < BK * BN; index += blockDim.x * blockDim.y) {
            const int row = index / BN;
            const int col = index % BN;
            const int globalRow = k0 + row;
            const int globalCol = blockCol + col;
            sB[row][col] = (globalRow < K && globalCol < N)
                                ? B[globalRow * N + globalCol]
                                : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = sA[ty + i * blockDim.y][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = sB[k][tx + j * blockDim.x];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = blockRow + ty + i * blockDim.y;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = blockCol + tx + j * blockDim.x;
                if (col < N) {
                    C[row * N + col] = acc[i][j];
                }
            }
        }
    }
}

}  // namespace

void launch_gemm_regblock(const float* d_A, const float* d_B, float* d_C, int M,
                          int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_regblock_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("regblock", launch_gemm_regblock);
