#pragma once

namespace gemm {

// ---------------------------------------------------------------------------
// CPU 参考实现：C = A * B
//   A: MxK row-major, B: KxN row-major, C: MxN row-major
//   float 输入/输出，内部用 double 累加，保证参考结果足够精确，
//   作为 GPU kernel 正确性校验的基准。
// ---------------------------------------------------------------------------
void referenceGemm(const float* A, const float* B, float* C, int M, int N, int K);

}  // namespace gemm
