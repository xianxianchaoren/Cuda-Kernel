// gemm_naive.cu —— 最朴素的 GEMM kernel（baseline + 接口示例）
//
// 这个文件演示了完整的「三步接入」流程：
//   1. 实现 __global__ kernel
//   2. 实现 host launcher（签名必须与 gemm::KernelLauncher 一致）
//   3. 文件末尾调用 REGISTER_GEMM_KERNEL 注册
//
// 它可以作为你后续所有优化 kernel 的正确性 baseline，可以保留或删除。

#include <gemm/registry.h>

namespace {

// 每个线程计算 C 中的一个元素
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

}  // namespace

void launch_gemm_naive(const float* d_A, const float* d_B, float* d_C, int M,
                       int N, int K, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

REGISTER_GEMM_KERNEL("naive", launch_gemm_naive);
