#include <gemm/timer.h>

#include <gemm/common.h>

namespace gemm {

GpuTimer::GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void GpuTimer::start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
}

void GpuTimer::stop(cudaStream_t stream) {
    // 在同一个流上记录 stop 事件，并同步等待其完成
    CUDA_CHECK(cudaEventRecord(stop_, stream));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    ms_ = static_cast<double>(ms);
}

double GpuTimer::elapsedMs() const { return ms_; }

}  // namespace gemm
