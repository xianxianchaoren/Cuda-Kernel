#include <gemm/checker.h>

#include <algorithm>
#include <cmath>

namespace gemm {

CheckResult checkResult(const float* ref, const float* got, size_t n,
                        double abs_tol, double rel_tol) {
    CheckResult r;
    r.pass = true;

    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double rv = static_cast<double>(ref[i]);
        const double gv = static_cast<double>(got[i]);
        const double diff = std::fabs(gv - rv);

        // 相对误差的分母带 abs_tol 下限：参考值接近 0 时退化为绝对误差
        const double denom = std::max(std::fabs(rv), abs_tol);
        const double rel = diff / denom;

        r.max_abs_err = std::max(r.max_abs_err, diff);
        r.max_rel_err = std::max(r.max_rel_err, rel);
        sum_sq += diff * diff;

        // 通过条件：|got - ref| <= abs_tol + rel_tol * |ref|
        if (diff > abs_tol + rel_tol * std::fabs(rv)) {
            r.pass = false;
        }
    }
    r.rms_err = std::sqrt(sum_sq / static_cast<double>(n));
    return r;
}

}  // namespace gemm
