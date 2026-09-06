// gemm_tiled.cu —— 你的第一个优化 kernel 的工作区
//
// 接入步骤（和 gemm_naive.cu 完全相同的三步）：
//   1. 在下方实现你的 __global__ kernel（建议：shared memory tiling）
//   2. 在 launch_gemm_tiled 中配置 block/grid 并 launch
//   3. 取消文件末尾 REGISTER_GEMM_KERNEL 的注释，框架自动发现它
//
// 契约：
//   - d_A: MxK row-major, d_B: KxN row-major, d_C: MxN row-major（设备端）
//   - 数据范围 [-1, 1]，float32
//   - 重要：M/N/K 不一定是 tile 尺寸的整数倍，必须处理越界访问
//   - kernel 必须 launch 在框架传入的 stream 上
//
// 优化思路提示（由易到难）：
//   v1: 每个 block 负责一个 TILE_M x TILE_N 的输出块，
//       循环 K/TILE_K 次，每次把 A 的 TILE_M x TILE_K 和 B 的 TILE_K x TILE_N
//       载入 shared memory 再计算（__syncthreads 同步）
//   v2: 线程按 float4 向量化访存，减少访存指令
//   v3: 双缓冲 / 预取，隐藏 global->shared 的延迟
//   v4: cp.async 异步拷贝
//   v5: 用 cutlass / mma 指令走 tensor core

#include <gemm/common.h>
#include <gemm/registry.h>

#define TILE_SIZE 32;

// TODO: 在这里实现你的 kernel
__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    __share__ float sA[TILE_SIZE][TILE_SIZE];
    __share__ float sB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    
    for (int i = 0; i < M; i += TILE_SIZE)
    {
        sA[threadIdx.]
    }

}

void launch_gemm_tiled(const float* d_A, const float* d_B, float* d_C, int M,
                       int N, int K, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}

// 实现完成后，取消下面这行注释即可注册进框架：
REGISTER_GEMM_KERNEL("tiled", launch_gemm_tiled);
