#pragma once

#include <cuda_runtime.h>

namespace gemm {

// 基于 CUDA Event 的 GPU 计时器（计时 launch 在 GPU 上的真实执行时间）
class GpuTimer {
public:
    GpuTimer();
    ~GpuTimer();

    void start(cudaStream_t stream = 0);
    void stop(cudaStream_t stream = 0);  // 内部会同步该流，阻塞直到 kernel 完成
    double elapsedMs() const;

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    double ms_ = 0.0;
};

}  // namespace gemm
