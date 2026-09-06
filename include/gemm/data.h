#pragma once

#include <cstdint>
#include <vector>

namespace gemm {

// 生成 rows x cols 的随机 float 矩阵（row-major），取值 [lo, hi)
// 由 seed 完全决定，可复现
std::vector<float> randomMatrix(int rows, int cols, float lo, float hi,
                                uint64_t seed);

}  // namespace gemm
