#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace gemm {

// ---------------------------------------------------------------------------
// 内核契约：这是框架与 kernel 之间的唯一约定
//
// 你实现的每个 kernel 需要提供一个 host launcher，签名必须为：
//
//   void launch_xxx(const float* d_A, const float* d_B, float* d_C,
//                   int M, int N, int K, cudaStream_t stream);
//
//   - d_A: MxK row-major 设备端矩阵
//   - d_B: KxN row-major 设备端矩阵
//   - d_C: MxN row-major 设备端输出（框架已分配好）
//   - stream: 框架传入的流，kernel 必须 launch 在该流上
//
// 内存分配、数据拷贝、计时、校验都由框架负责，launcher 只需要 launch kernel。
// ---------------------------------------------------------------------------
using KernelLauncher = void (*)(const float* d_A, const float* d_B, float* d_C,
                                int M, int N, int K, cudaStream_t stream);

struct KernelEntry {
    const char* name;      // 显示名，如 "naive"、"tiled"、"tensorcore"
    KernelLauncher launcher;
};

namespace detail {

inline std::vector<KernelEntry>& kernelRegistry() {
    static std::vector<KernelEntry> regs;
    return regs;
}

}  // namespace detail

inline void registerKernel(const KernelEntry& entry) {
    detail::kernelRegistry().push_back(entry);
}

inline const std::vector<KernelEntry>& registeredKernels() {
    return detail::kernelRegistry();
}

}  // namespace gemm

// ---------------------------------------------------------------------------
// 注册宏：在 .cu 文件末尾调用一次即可
//
//   REGISTER_GEMM_KERNEL("naive", launch_gemm_naive);
//
// 注册后框架自动发现该 kernel，无需修改任何框架代码。
// 利用静态初始化在 main 之前完成注册（函数内静态变量，无初始化顺序问题）。
// ---------------------------------------------------------------------------
#define REGISTER_GEMM_KERNEL(kernel_name, launcher_fn)                         \
    static bool _gemm_reg_##launcher_fn = []() {                               \
        ::gemm::registerKernel(::gemm::KernelEntry{kernel_name, launcher_fn}); \
        return true;                                                           \
    }()
