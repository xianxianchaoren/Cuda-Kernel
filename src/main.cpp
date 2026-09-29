#include <gemm/checker.h>
#include <gemm/common.h>
#include <gemm/data.h>
#include <gemm/reference.h>
#include <gemm/registry.h>
#include <gemm/timer.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct Options {
    int M = 1024, N = 1024, K = 1024;
    std::vector<std::string> kernels;  // 空 = 全部已注册
    int iters = 10;                    // benchmark 迭代次数
    int warmup = 3;                    // 预热轮数
    uint64_t seed = gemm::kDefaultSeed;
    double rel_tol = 1e-2;  // 相对误差阈值
    double abs_tol = 1e-4;  // 绝对误差阈值
    int l2flush_mb = 0;     // 每次迭代前冲刷 L2 的大小，0 = 关闭
    bool list_only = false;
    bool skip_check = false;
};

void printUsage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "\n"
        "  GEMM 测试框架: 正确性校验 + 性能基准 (C = A * B, float32, row-major)\n"
        "\n"
        "Options:\n"
        "  -M <int>            矩阵 A 行数 (default: 1024)\n"
        "  -N <int>            矩阵 B 列数 (default: 1024)\n"
        "  -K <int>            内维 K (default: 1024)\n"
        "  --kernels <list>    要运行的 kernel，逗号分隔 (default: 全部已注册)\n"
        "  --list              列出所有已注册的 kernel 并退出\n"
        "  --skip-check        跳过 CPU 参考计算与正确性校验，仅运行性能基准\n"
        "  --iters <int>       benchmark 迭代次数 (default: 10)\n"
        "  --warmup <int>      预热轮数 (default: 3)\n"
        "  --seed <uint64>     随机种子，结果可复现 (default: 42)\n"
        "  --rel-tol <float>   相对误差阈值 (default: 1e-2)\n"
        "  --abs-tol <float>   绝对误差阈值 (default: 1e-4)\n"
        "  --l2flush-mb <int>  每次迭代前 memset 冲刷 L2 的大小 MB (default: 0=关闭)\n"
        "  -h, --help          显示帮助\n"
        "\n"
        "Examples:\n"
        "  %s -M 4096 -N 4096 -K 4096\n"
        "  %s --kernels naive,tiled -M 1000 -N 999 -K 1001 --iters 5\n",
        prog, prog, prog);
}

const char* requireValue(int argc, char** argv, int& i, const char* flag) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: missing value for %s\n", flag);
        std::exit(EXIT_FAILURE);
    }
    return argv[++i];
}

Options parseArgs(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (std::strcmp(a, "-M") == 0) {
            o.M = std::atoi(requireValue(argc, argv, i, a));
        } else if (std::strcmp(a, "-N") == 0) {
            o.N = std::atoi(requireValue(argc, argv, i, a));
        } else if (std::strcmp(a, "-K") == 0) {
            o.K = std::atoi(requireValue(argc, argv, i, a));
        } else if (std::strcmp(a, "--kernels") == 0) {
            std::string s(requireValue(argc, argv, i, a));
            size_t pos = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                o.kernels.push_back(s.substr(0, pos));
                s.erase(0, pos + 1);
            }
            if (!s.empty()) o.kernels.push_back(s);
        } else if (std::strcmp(a, "--list") == 0) {
            o.list_only = true;
        } else if (std::strcmp(a, "--skip-check") == 0) {
            o.skip_check = true;
        } else if (std::strcmp(a, "--iters") == 0) {
            o.iters = std::atoi(requireValue(argc, argv, i, a));
        } else if (std::strcmp(a, "--warmup") == 0) {
            o.warmup = std::atoi(requireValue(argc, argv, i, a));
        } else if (std::strcmp(a, "--seed") == 0) {
            o.seed = std::strtoull(requireValue(argc, argv, i, a), nullptr, 10);
        } else if (std::strcmp(a, "--rel-tol") == 0) {
            o.rel_tol = std::strtod(requireValue(argc, argv, i, a), nullptr);
        } else if (std::strcmp(a, "--abs-tol") == 0) {
            o.abs_tol = std::strtod(requireValue(argc, argv, i, a), nullptr);
        } else if (std::strcmp(a, "--l2flush-mb") == 0) {
            o.l2flush_mb = std::atoi(requireValue(argc, argv, i, a));
        } else {
            std::fprintf(stderr, "error: unknown option '%s'\n", a);
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (o.M <= 0 || o.N <= 0 || o.K <= 0) {
        std::fprintf(stderr, "error: M/N/K must be positive\n");
        std::exit(EXIT_FAILURE);
    }
    if (o.iters < 1) {
        std::fprintf(stderr, "error: --iters must be >= 1\n");
        std::exit(EXIT_FAILURE);
    }
    if (o.warmup < 0) {
        std::fprintf(stderr, "error: --warmup must be >= 0\n");
        std::exit(EXIT_FAILURE);
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);

    if (opt.list_only) {
        std::printf("Registered kernels (%zu):\n",
                    gemm::registeredKernels().size());
        for (const auto& k : gemm::registeredKernels()) {
            std::printf("  - %s\n", k.name);
        }
        return EXIT_SUCCESS;
    }

    // ---- 选择要运行的 kernel ----
    std::vector<const gemm::KernelEntry*> selected;
    for (const auto& k : gemm::registeredKernels()) {
        if (opt.kernels.empty() ||
            std::find(opt.kernels.begin(), opt.kernels.end(), k.name) !=
                opt.kernels.end()) {
            selected.push_back(&k);
        }
    }
    if (selected.empty()) {
        std::fprintf(stderr,
                     "error: no kernel selected (use --list to see available "
                     "kernels)\n");
        return EXIT_FAILURE;
    }

    // ---- 设备与问题信息 ----
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU      : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    const int M = opt.M, N = opt.N, K = opt.K;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    std::printf("Problem  : C(%dx%d) = A(%dx%d) x B(%dx%d), %.3f GFLOP\n", M, N,
                M, K, K, N, flops / 1e9);
    std::printf("Kernels  :");
    for (const auto* k : selected) std::printf(" %s", k->name);
    std::printf("\n");
    std::printf("Config   : iters=%d warmup=%d seed=%llu rel_tol=%.1e "
                "abs_tol=%.1e l2flush=%dMB skip_check=%s\n",
                opt.iters, opt.warmup,
                static_cast<unsigned long long>(opt.seed), opt.rel_tol,
                opt.abs_tol, opt.l2flush_mb, opt.skip_check ? "yes" : "no");

    // ---- 生成数据 + CPU 参考计算 ----
    std::vector<float> A = gemm::randomMatrix(M, K, -1.0f, 1.0f, opt.seed);
    std::vector<float> B = gemm::randomMatrix(K, N, -1.0f, 1.0f, opt.seed + 1);
    std::vector<float> C_ref;
    if (!opt.skip_check) {
        C_ref.resize(static_cast<size_t>(M) * N);
        gemm::referenceGemm(A.data(), B.data(), C_ref.data(), M, N, K);
    }

    // ---- 设备端准备 ----
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_flush = nullptr;
    const size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(M) * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(K) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), static_cast<size_t>(M) * K * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), static_cast<size_t>(K) * N * sizeof(float),
                          cudaMemcpyHostToDevice));

    size_t flush_bytes = 0;
    if (opt.l2flush_mb > 0) {
        flush_bytes = static_cast<size_t>(opt.l2flush_mb) * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));
        CUDA_CHECK(cudaMemset(d_flush, 0, flush_bytes));
    }

    std::vector<float> C_gpu(static_cast<size_t>(M) * N);

    // ---- 表头 ----
    std::printf("\n%-10s %-8s %-12s %-12s %-10s %-10s %s\n", "kernel", "result",
                "best(ms)", "avg(ms)", "max-rel", "max-abs", "GFLOPS");
    std::printf("%s\n", std::string(82, '-').c_str());

    gemm::GpuTimer timer;

    for (const auto* k : selected) {
        gemm::CheckResult cr;
        cr.pass = opt.skip_check;
        if (!opt.skip_check) {
            CUDA_CHECK(cudaMemset(d_C, 0, c_bytes));
            k->launcher(d_A, d_B, d_C, M, N, K, 0);
            CUDA_CHECK_LAST();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(C_gpu.data(), d_C, c_bytes,
                                  cudaMemcpyDeviceToHost));
            cr = gemm::checkResult(C_ref.data(), C_gpu.data(), C_ref.size(),
                                   opt.abs_tol, opt.rel_tol);
        }

        // ---- 性能基准（仅正确性通过时执行） ----
        double best_ms = 0.0, avg_ms = 0.0, gflops = 0.0;
        if (cr.pass) {
            // 预热：触发 lazy 初始化、填充缓存
            for (int i = 0; i < opt.warmup; ++i) {
                if (flush_bytes)
                    CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_bytes, 0));
                k->launcher(d_A, d_B, d_C, M, N, K, 0);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<double> times;
            times.reserve(opt.iters);
            for (int i = 0; i < opt.iters; ++i) {
                if (flush_bytes)
                    CUDA_CHECK(cudaMemsetAsync(d_flush, 0, flush_bytes, 0));
                timer.start(0);
                k->launcher(d_A, d_B, d_C, M, N, K, 0);
                timer.stop(0);
                times.push_back(timer.elapsedMs());
            }
            best_ms = *std::min_element(times.begin(), times.end());
            avg_ms = std::accumulate(times.begin(), times.end(), 0.0) /
                     static_cast<double>(times.size());
            gflops = flops / (best_ms * 1e-3) / 1e9;
        }

        std::printf("%-10s %-8s %-12.4f %-12.4f %-10.2e %-10.2e %.1f\n",
                    k->name, opt.skip_check ? "SKIPPED" : (cr.pass ? "PASS" : "FAIL"),
                    best_ms, avg_ms, cr.max_rel_err, cr.max_abs_err, gflops);
        if (!cr.pass) {
            std::printf("    ^ rms_err=%.2e  (rel_tol=%.1e, abs_tol=%.1e)\n",
                        cr.rms_err, opt.rel_tol, opt.abs_tol);
        }
    }

    std::printf("%s\n", std::string(82, '-').c_str());

    // ---- 清理 ----
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    if (d_flush) CUDA_CHECK(cudaFree(d_flush));

    return EXIT_SUCCESS;
}
