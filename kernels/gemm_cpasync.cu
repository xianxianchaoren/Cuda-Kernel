#include <cuda_pipeline.h>

#include <gemm/registry.h>

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 4;
constexpr int TN = 4;

__device__ void load_tile(float sA[2][BM][BK], float sB[2][BK][BN], int stage,
                          const float* A, const float* B, int M, int N, int K,
                          int blockRow, int blockCol, int k0, int tid) {
    const bool vectorized = (K % 4 == 0) && (N % 4 == 0) &&
                            (blockRow + BM <= M) && (blockCol + BN <= N) &&
                            (k0 + BK <= K);

    if (vectorized && tid < BM * BK / 4) {
        const int index = tid * 4;
        const int row = index / BK;
        const int col = index % BK;
        __pipeline_memcpy_async(&sA[stage][row][col],
                                A + (blockRow + row) * K + k0 + col,
                                sizeof(float4));
    } else if (!vectorized) {
        for (int index = tid; index < BM * BK; index += blockDim.x * blockDim.y) {
            const int row = index / BK;
            const int col = index % BK;
            const int globalRow = blockRow + row;
            const int globalCol = k0 + col;
            sA[stage][row][col] = (globalRow < M && globalCol < K)
                                        ? A[globalRow * K + globalCol]
                                        : 0.0f;
        }
    }

    if (vectorized && tid < BK * BN / 4) {
        const int index = tid * 4;
        const int row = index / BN;
        const int col = index % BN;
        __pipeline_memcpy_async(&sB[stage][row][col],
                                B + (k0 + row) * N + blockCol + col,
                                sizeof(float4));
    } else if (!vectorized) {
        for (int index = tid; index < BK * BN; index += blockDim.x * blockDim.y) {
            const int row = index / BN;
            const int col = index % BN;
            const int globalRow = k0 + row;
            const int globalCol = blockCol + col;
            sB[stage][row][col] = (globalRow < K && globalCol < N)
                                        ? B[globalRow * N + globalCol]
                                        : 0.0f;
        }
    }

    if (vectorized) {
        __pipeline_commit();
    }
}

__global__ void gemm_cpasync_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[2][BM][BK];
    __shared__ float sB[2][BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const bool vectorized = (K % 4 == 0) && (N % 4 == 0) &&
                            (blockRow + BM <= M) && (blockCol + BN <= N);
    float acc[TM][TN] = {};

    load_tile(sA, sB, 0, A, B, M, N, K, blockRow, blockCol, 0, tid);
    if (vectorized) __pipeline_wait_prior(0);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        const int nextK = k0 + BK;
        const int nextStage = stage ^ 1;
        if (nextK < K) {
            load_tile(sA, sB, nextStage, A, B, M, N, K, blockRow, blockCol,
                      nextK, tid);
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = sA[stage][ty + i * blockDim.y][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = sB[stage][k][tx + j * blockDim.x];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
            }
        }

        if (nextK < K && vectorized) __pipeline_wait_prior(0);
        __syncthreads();
        stage = nextStage;
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = blockRow + ty + i * blockDim.y;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int col = blockCol + tx + j * blockDim.x;
                if (col < N) C[row * N + col] = acc[i][j];
            }
        }
    }
}

}  // namespace

void launch_gemm_cpasync(const float* d_A, const float* d_B, float* d_C, int M,
                         int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_cpasync_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("cpasync", launch_gemm_cpasync);
