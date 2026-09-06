#include <gemm/reference.h>

namespace gemm {

void referenceGemm(const float* A, const float* B, float* C, int M, int N,
                   int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // double 累加：避免浮点误差累积影响参考精度
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[i * K + k]) *
                       static_cast<double>(B[k * N + j]);
            }
            C[i * N + j] = static_cast<float>(sum);
        }
    }
}

}  // namespace gemm
