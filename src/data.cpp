#include <gemm/data.h>

#include <random>

namespace gemm {

std::vector<float> randomMatrix(int rows, int cols, float lo, float hi,
                                uint64_t seed) {
    std::vector<float> m(static_cast<size_t>(rows) * static_cast<size_t>(cols));
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : m) {
        v = dist(rng);
    }
    return m;
}

}  // namespace gemm
