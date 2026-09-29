#include <mma.h>

#include <gemm/registry.h>

namespace {

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 8;

__global__ void gemm_tc_tf32_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int M, int N, int K) {
    __shared__ float sA[WMMA_M][WMMA_K];
    __shared__ float sB[WMMA_K][WMMA_N];
    __shared__ float sC[WMMA_M][WMMA_N];

    const int lane = threadIdx.x;
    const int blockRow = blockIdx.y * WMMA_M;
    const int blockCol = blockIdx.x * WMMA_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        for (int index = lane; index < WMMA_M * WMMA_K; index += warpSize) {
            const int row = index / WMMA_K;
            const int col = index % WMMA_K;
            const int globalRow = blockRow + row;
            const int globalCol = k0 + col;
            sA[row][col] = (globalRow < M && globalCol < K)
                                ? A[globalRow * K + globalCol]
                                : 0.0f;
        }
        for (int index = lane; index < WMMA_K * WMMA_N; index += warpSize) {
            const int row = index / WMMA_N;
            const int col = index % WMMA_N;
            const int globalRow = k0 + row;
            const int globalCol = blockCol + col;
            sB[row][col] = (globalRow < K && globalCol < N)
                                ? B[globalRow * N + globalCol]
                                : 0.0f;
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> b;
        wmma::load_matrix_sync(a, &sA[0][0], WMMA_K);
        wmma::load_matrix_sync(b, &sB[0][0], WMMA_N);
        wmma::mma_sync(acc, a, b, acc);
        __syncthreads();
    }

    wmma::store_matrix_sync(&sC[0][0], acc, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    for (int index = lane; index < WMMA_M * WMMA_N; index += warpSize) {
        const int row = index / WMMA_N;
        const int col = index % WMMA_N;
        const int globalRow = blockRow + row;
        const int globalCol = blockCol + col;
        if (globalRow < M && globalCol < N) {
            C[globalRow * N + globalCol] = sC[row][col];
        }
    }
}

}  // namespace

void launch_gemm_tc_tf32(const float* d_A, const float* d_B, float* d_C, int M,
                         int N, int K, cudaStream_t stream) {
    dim3 block(32);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    gemm_tc_tf32_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("tc_tf32", launch_gemm_tc_tf32);
