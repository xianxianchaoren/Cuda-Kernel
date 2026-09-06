#pragma once

#include <cstddef>

namespace gemm {

struct CheckResult {
    bool pass = false;        // 是否通过校验
    double max_abs_err = 0.0; // 最大绝对误差
    double max_rel_err = 0.0; // 最大相对误差（分母带 abs_tol 下限，避免除零）
    double rms_err = 0.0;     // 均方根误差
};

// 逐元素校验 GPU 结果 vs CPU 参考。
// 通过条件：|got - ref| <= abs_tol + rel_tol * |ref|
CheckResult checkResult(const float* ref, const float* got, size_t n,
                        double abs_tol, double rel_tol);

}  // namespace gemm
