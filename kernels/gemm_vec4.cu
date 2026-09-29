#include <gemm/registry.h>

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 4;
constexpr int TN = 4;

__global__ void gemm_vec4_kernel(const float* __restrict__ A,
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
        for (int index = tid * 4; index < BM * BK; index += blockDim.x * blockDim.y * 4) {
            const int localRow = index / BK;
            const int localCol = index % BK;
            const int globalRow = blockRow + localRow;
            const int globalCol = k0 + localCol;
            float4 value = {};
            if (globalRow < M && globalCol + 3 < K && (K & 3) == 0) {
                value = *reinterpret_cast<const float4*>(A + globalRow * K + globalCol);
            } else {
                const float* row = (globalRow < M) ? A + globalRow * K : nullptr;
                value.x = (row && globalCol < K) ? row[globalCol] : 0.0f;
                value.y = (row && globalCol + 1 < K) ? row[globalCol + 1] : 0.0f;
                value.z = (row && globalCol + 2 < K) ? row[globalCol + 2] : 0.0f;
                value.w = (row && globalCol + 3 < K) ? row[globalCol + 3] : 0.0f;
            }
            *reinterpret_cast<float4*>(&sA[localRow][localCol]) = value;
        }

        for (int index = tid * 4; index < BK * BN; index += blockDim.x * blockDim.y * 4) {
            const int localRow = index / BN;
            const int localCol = index % BN;
            const int globalRow = k0 + localRow;
            const int globalCol = blockCol + localCol;
            float4 value = {};
            if (globalRow < K && globalCol + 3 < N && (N & 3) == 0) {
                value = *reinterpret_cast<const float4*>(B + globalRow * N + globalCol);
            } else {
                const float* row = (globalRow < K) ? B + globalRow * N : nullptr;
                value.x = (row && globalCol < N) ? row[globalCol] : 0.0f;
                value.y = (row && globalCol + 1 < N) ? row[globalCol + 1] : 0.0f;
                value.z = (row && globalCol + 2 < N) ? row[globalCol + 2] : 0.0f;
                value.w = (row && globalCol + 3 < N) ? row[globalCol + 3] : 0.0f;
            }
            *reinterpret_cast<float4*>(&sB[localRow][localCol]) = value;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = sA[ty + i * blockDim.y][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = sB[k][tx + j * blockDim.x];
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
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
                if (col < N) C[row * N + col] = acc[i][j];
            }
        }
    }
}

}  // namespace

void launch_gemm_vec4(const float* d_A, const float* d_B, float* d_C, int M,
                      int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_vec4_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("vec4", launch_gemm_vec4);
