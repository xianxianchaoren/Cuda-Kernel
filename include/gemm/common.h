#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace gemm {

// ---------------------------------------------------------------------------
// CUDA 错误检查宏：任何 CUDA API 调用失败都会打印出错位置并终止
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,    \
                         __LINE__, cudaGetErrorString(err),                     \
                         cudaGetErrorName(err));                                \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 检查最近一次 kernel launch 是否成功（launch 是异步的，配置错误会在此暴露）
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

constexpr int kDefaultSeed = 42;

}  // namespace gemm
