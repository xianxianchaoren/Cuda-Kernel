#include <cutlass/cutlass.h>
#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

#include <gemm/registry.h>

#include <cstdio>
#include <cstdlib>

namespace {

using CutlassGemm = cutlass::gemm::device::Gemm<
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2,
    1,
    1,
    false,
    cutlass::arch::OpMultiplyAdd>;

}  // namespace

void launch_gemm_cutlass(const float* d_A, const float* d_B, float* d_C, int M,
                         int N, int K, cudaStream_t stream) {
    typename CutlassGemm::Arguments arguments(
        {M, N, K},
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {1.0f, 0.0f});

    CutlassGemm gemm;
    const cutlass::Status status = gemm(arguments, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS GEMM launch failed: %d\n", static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

REGISTER_GEMM_KERNEL("cutlass_simt", launch_gemm_cutlass);
